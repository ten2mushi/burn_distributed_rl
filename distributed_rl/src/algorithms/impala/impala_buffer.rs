//! IMPALA-specific trajectory buffer implementing ExperienceBuffer trait.
//!
//! This buffer is designed for off-policy algorithms where trajectories
//! are stored with behavior policy information for V-trace correction.

use crate::core::experience_buffer::{ExperienceBuffer, OffPolicyBuffer};
use crate::core::transition::{IMPALATransition, Trajectory};
use crossbeam_deque::{Injector, Steal};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for IMPALA trajectory buffer.
#[derive(Debug, Clone)]
pub struct IMPALABufferConfig {
    /// Number of actor threads.
    pub n_actors: usize,
    /// Environments per actor.
    pub n_envs_per_actor: usize,
    /// Trajectory length.
    pub trajectory_length: usize,
    /// Maximum trajectories to store.
    pub max_trajectories: usize,
    /// Batch size (number of trajectories).
    pub batch_size: usize,
}

impl Default for IMPALABufferConfig {
    fn default() -> Self {
        Self {
            n_actors: 4,
            n_envs_per_actor: 32,
            trajectory_length: 20,
            max_trajectories: 1000,
            batch_size: 32,
        }
    }
}

impl IMPALABufferConfig {
    /// Total environments.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    /// Approximate capacity in transitions.
    pub fn capacity(&self) -> usize {
        self.max_trajectories * self.trajectory_length
    }
}

/// IMPALA trajectory batch for training.
#[derive(Debug, Clone)]
pub struct IMPALABatch {
    /// Batch of trajectories.
    pub trajectories: Vec<Trajectory<IMPALATransition>>,
    /// Policy versions of the trajectories.
    pub policy_versions: Vec<u64>,
}

impl IMPALABatch {
    /// Total number of transitions.
    pub fn total_transitions(&self) -> usize {
        self.trajectories.iter().map(|t| t.len()).sum()
    }

    /// Number of trajectories.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// Get all states (flattened).
    pub fn states(&self) -> Vec<f32> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().flat_map(|tr| tr.base.state.iter().copied()))
            .collect()
    }

    /// Get all next states.
    pub fn next_states(&self) -> Vec<f32> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().flat_map(|tr| tr.base.next_state.iter().copied()))
            .collect()
    }

    /// Get all behavior log probs.
    pub fn behavior_log_probs(&self) -> Vec<f32> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| tr.behavior_log_prob))
            .collect()
    }

    /// Get all rewards.
    pub fn rewards(&self) -> Vec<f32> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| tr.base.reward))
            .collect()
    }

    /// Get all done flags.
    pub fn dones(&self) -> Vec<bool> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| tr.done()))
            .collect()
    }

    /// State dimension (from first transition).
    pub fn state_dim(&self) -> usize {
        self.trajectories
            .first()
            .and_then(|t| t.transitions.first())
            .map(|tr| tr.base.state.len())
            .unwrap_or(0)
    }

    /// Get maximum staleness (current_version - oldest_policy_version).
    pub fn max_staleness(&self, current_version: u64) -> u64 {
        self.policy_versions
            .iter()
            .map(|v| current_version.saturating_sub(*v))
            .max()
            .unwrap_or(0)
    }
}

/// Thread-safe trajectory buffer for distributed IMPALA.
///
/// Actors push complete trajectories, and the learner samples
/// batches in FIFO order for V-trace training.
pub struct IMPALABuffer {
    config: IMPALABufferConfig,
    /// Lock-free injection queue.
    injector: Injector<Trajectory<IMPALATransition>>,
    /// Consolidated storage.
    storage: RwLock<Vec<Trajectory<IMPALATransition>>>,
    /// Storage size (atomic).
    size: AtomicUsize,
    /// Pending injector size.
    pending_size: AtomicUsize,
}

impl IMPALABuffer {
    /// Create a new IMPALA buffer.
    pub fn new(config: IMPALABufferConfig) -> Self {
        Self {
            config,
            injector: Injector::new(),
            storage: RwLock::new(Vec::new()),
            size: AtomicUsize::new(0),
            pending_size: AtomicUsize::new(0),
        }
    }

    /// Push a complete trajectory (lock-free).
    /// Accepts all non-empty trajectories, including short ones from early episode termination.
    pub fn push_trajectory(&self, trajectory: Trajectory<IMPALATransition>) {
        if !trajectory.is_empty() {
            self.injector.push(trajectory);
            self.pending_size.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sample a batch of trajectories (FIFO).
    pub fn sample_batch(&self) -> Option<IMPALABatch> {
        // Consolidate first
        self.do_consolidate();

        let mut storage = self.storage.write();
        if storage.len() < self.config.batch_size {
            return None;
        }

        // Take oldest trajectories (FIFO)
        let trajectories: Vec<_> = storage.drain(0..self.config.batch_size).collect();
        self.size.store(storage.len(), Ordering::Relaxed);

        // Extract policy versions
        let policy_versions: Vec<u64> = trajectories
            .iter()
            .map(|t| {
                t.transitions
                    .first()
                    .map(|tr| tr.policy_version)
                    .unwrap_or(0)
            })
            .collect();

        Some(IMPALABatch {
            trajectories,
            policy_versions,
        })
    }

    /// Internal consolidation.
    fn do_consolidate(&self) {
        let mut storage = self.storage.write();
        let mut count = 0;

        loop {
            match self.injector.steal() {
                Steal::Success(traj) => {
                    storage.push(traj);
                    count += 1;
                }
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }

        // Enforce max capacity (FIFO eviction)
        while storage.len() > self.config.max_trajectories {
            storage.remove(0);
        }

        if count > 0 {
            self.pending_size.fetch_sub(count.min(self.pending_size.load(Ordering::Relaxed)), Ordering::Relaxed);
            self.size.store(storage.len(), Ordering::Relaxed);
        }
    }

    /// Check if ready for training.
    /// Consolidates pending data first to get an accurate count.
    pub fn is_training_ready(&self) -> bool {
        // Consolidate first to avoid race between pending_size and size
        self.do_consolidate();
        self.size.load(Ordering::Acquire) >= self.config.batch_size
    }

    /// Get configuration.
    pub fn config(&self) -> &IMPALABufferConfig {
        &self.config
    }

    /// Get pending count.
    pub fn pending_count(&self) -> usize {
        self.pending_size.load(Ordering::Relaxed)
    }
}

// Implement ExperienceBuffer trait
impl ExperienceBuffer for IMPALABuffer {
    type Item = Trajectory<IMPALATransition>;

    fn push(&self, item: Self::Item) {
        self.push_trajectory(item);
    }

    fn push_batch(&self, items: Vec<Self::Item>) {
        for item in items {
            self.push_trajectory(item);
        }
    }

    fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    fn capacity(&self) -> usize {
        self.config.capacity()
    }

    fn consolidate(&self) {
        self.do_consolidate();
    }

    fn clear(&self) {
        // Drain injector
        loop {
            match self.injector.steal() {
                Steal::Success(_) => continue,
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
        self.pending_size.store(0, Ordering::Relaxed);

        // Clear storage
        let mut storage = self.storage.write();
        storage.clear();
        self.size.store(0, Ordering::Relaxed);
    }
}

// Implement OffPolicyBuffer trait
impl OffPolicyBuffer for IMPALABuffer {
    fn sample(&self, batch_size: usize) -> Option<Vec<Self::Item>> {
        self.do_consolidate();

        let mut storage = self.storage.write();
        if storage.len() < batch_size {
            return None;
        }

        let items: Vec<_> = storage.drain(0..batch_size).collect();
        self.size.store(storage.len(), Ordering::Relaxed);
        Some(items)
    }

    fn pending_len(&self) -> usize {
        self.pending_size.load(Ordering::Relaxed)
    }
}

// Note: IMPALABuffer is automatically Send + Sync because:
// - Injector<T> is Send + Sync when T: Send
// - RwLock<T> is Send + Sync when T: Send
// - AtomicUsize is Send + Sync
// No manual unsafe impl needed.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_trajectory(len: usize, env_id: usize, version: u64) -> Trajectory<IMPALATransition> {
        let mut traj = Trajectory::new(env_id);
        for i in 0..len {
            traj.push(IMPALATransition {
                base: Transition::new_discrete(
                    vec![i as f32],
                    0,
                    1.0,
                    vec![(i + 1) as f32],
                    i == len - 1,
                    false,
                ),
                behavior_log_prob: -0.5,
                policy_version: version,
            });
        }
        traj
    }

    #[test]
    fn test_impala_buffer_new() {
        let config = IMPALABufferConfig::default();
        let buffer = IMPALABuffer::new(config);

        assert!(buffer.is_empty());
        assert!(!buffer.is_training_ready());
    }

    #[test]
    fn test_impala_buffer_push_and_sample() {
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push trajectories
        buffer.push_trajectory(make_trajectory(10, 0, 1));
        buffer.push_trajectory(make_trajectory(10, 1, 2));
        buffer.push_trajectory(make_trajectory(10, 2, 3));

        // Should be ready
        assert!(buffer.is_training_ready());

        // Sample batch
        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.policy_versions.len(), 2);

        // Check FIFO order
        assert_eq!(batch.trajectories[0].env_id, 0);
        assert_eq!(batch.trajectories[1].env_id, 1);
    }

    #[test]
    fn test_impala_batch_staleness() {
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 2,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        buffer.push_trajectory(make_trajectory(10, 0, 5));
        buffer.push_trajectory(make_trajectory(10, 1, 8));

        let batch = buffer.sample_batch().unwrap();
        let staleness = batch.max_staleness(10);
        assert_eq!(staleness, 5); // 10 - 5 = 5
    }

    // ========================================================================
    // Regression tests for bug fixes
    // ========================================================================

    #[test]
    fn regression_bug7_short_trajectories_accepted() {
        // Bug 7: Short trajectories were being dropped if shorter than trajectory_length.
        // This caused data loss on early episode termination.
        let config = IMPALABufferConfig {
            trajectory_length: 20,  // Config says 20
            batch_size: 1,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push a short trajectory (only 5 steps, less than trajectory_length of 20)
        let short_traj = make_trajectory(5, 0, 1);
        buffer.push_trajectory(short_traj);

        // Consolidate and verify it was accepted
        buffer.consolidate();
        assert_eq!(buffer.len(), 1, "Short trajectory should be accepted!");

        // Sample and verify we get the short trajectory
        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.trajectories.len(), 1);
        assert_eq!(batch.trajectories[0].len(), 5, "Should get 5-step trajectory");
    }

    #[test]
    fn regression_empty_trajectories_rejected() {
        // Empty trajectories should still be rejected
        let config = IMPALABufferConfig {
            trajectory_length: 20,
            batch_size: 1,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push an empty trajectory
        let empty_traj = Trajectory::new(0);
        buffer.push_trajectory(empty_traj);

        buffer.consolidate();
        assert_eq!(buffer.len(), 0, "Empty trajectory should be rejected");
    }

    #[test]
    fn regression_bug15_pending_size_race() {
        // Bug 15: is_training_ready() could return true based on pending_size
        // before consolidation, but sample_batch() would fail after consolidation.
        // Fix: Consolidate before checking readiness.
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push exactly batch_size trajectories
        buffer.push_trajectory(make_trajectory(10, 0, 1));
        buffer.push_trajectory(make_trajectory(10, 1, 1));

        // is_training_ready now consolidates first, so this should work
        assert!(buffer.is_training_ready());

        // sample_batch should succeed
        let batch = buffer.sample_batch();
        assert!(batch.is_some(), "sample_batch should succeed after is_training_ready");
        assert_eq!(batch.unwrap().len(), 2);
    }

    #[test]
    fn test_buffer_fifo_ordering() {
        // Verify FIFO: oldest trajectories are sampled first
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push trajectories with increasing env_ids
        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }

        // First batch should have env_ids 0, 1
        buffer.consolidate();
        let batch1 = buffer.sample_batch().unwrap();
        assert_eq!(batch1.trajectories[0].env_id, 0);
        assert_eq!(batch1.trajectories[1].env_id, 1);

        // Second batch should have env_ids 2, 3
        let batch2 = buffer.sample_batch().unwrap();
        assert_eq!(batch2.trajectories[0].env_id, 2);
        assert_eq!(batch2.trajectories[1].env_id, 3);
    }

    #[test]
    fn test_buffer_capacity_enforcement() {
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 1,
            max_trajectories: 3,  // Small capacity
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push more than capacity
        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }

        buffer.consolidate();

        // Buffer should be at max_trajectories, oldest removed
        assert!(buffer.len() <= 3, "Buffer should respect max capacity");
    }

    #[test]
    fn test_buffer_clear() {
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 1,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Add some data
        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();
        assert!(buffer.len() > 0);

        // Clear
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.pending_count(), 0);
        assert!(!buffer.is_training_ready());
    }

    #[test]
    fn test_buffer_sample_returns_none_when_insufficient() {
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 5,  // Need 5 trajectories
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Only push 3 trajectories
        for i in 0..3 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();

        // Should return None
        let batch = buffer.sample_batch();
        assert!(batch.is_none(), "Should return None when insufficient data");
    }

    #[test]
    fn test_buffer_policy_version_extraction() {
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 3,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push trajectories with different policy versions
        buffer.push_trajectory(make_trajectory(5, 0, 100));
        buffer.push_trajectory(make_trajectory(5, 1, 200));
        buffer.push_trajectory(make_trajectory(5, 2, 150));

        buffer.consolidate();
        let batch = buffer.sample_batch().unwrap();

        // Verify policy versions are correctly extracted
        assert_eq!(batch.policy_versions.len(), 3);
        assert_eq!(batch.policy_versions[0], 100);
        assert_eq!(batch.policy_versions[1], 200);
        assert_eq!(batch.policy_versions[2], 150);
    }
}
