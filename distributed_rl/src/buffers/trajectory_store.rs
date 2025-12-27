//! Trajectory store for IMPALA.
//!
//! Key characteristics:
//! - Stores complete or partial trajectories
//! - Supports multiple actors pushing concurrently (lock-free injection)
//! - FIFO consumption (oldest trajectories first)
//! - Tracks policy version for V-trace correction

use crate::core::transition::{IMPALATransition, Trajectory};
use crossbeam_deque::{Injector, Steal};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for trajectory store.
#[derive(Debug, Clone)]
pub struct TrajectoryStoreConfig {
    /// Maximum trajectories to store
    pub max_trajectories: usize,
    /// Minimum trajectory length for training
    pub min_trajectory_len: usize,
    /// Batch size (number of trajectories per batch)
    pub batch_size: usize,
}

impl Default for TrajectoryStoreConfig {
    fn default() -> Self {
        Self {
            max_trajectories: 1000,
            min_trajectory_len: 20,
            batch_size: 32,
        }
    }
}

/// Trajectory batch for IMPALA training.
#[derive(Debug)]
pub struct TrajectoryBatch {
    /// Batch of trajectories
    pub trajectories: Vec<Trajectory<IMPALATransition>>,
}

impl TrajectoryBatch {
    /// Get total number of transitions in batch.
    pub fn total_transitions(&self) -> usize {
        self.trajectories.iter().map(|t| t.len()).sum()
    }

    /// Get number of trajectories.
    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// Flatten all states from all trajectories.
    pub fn all_states(&self) -> Vec<&Vec<f32>> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| &tr.base.state))
            .collect()
    }

    /// Flatten all behavior log probs.
    pub fn all_behavior_log_probs(&self) -> Vec<f32> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| tr.behavior_log_prob))
            .collect()
    }

    /// Flatten all rewards.
    pub fn all_rewards(&self) -> Vec<f32> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| tr.base.reward))
            .collect()
    }

    /// Flatten all done flags.
    pub fn all_dones(&self) -> Vec<bool> {
        self.trajectories
            .iter()
            .flat_map(|t| t.iter().map(|tr| tr.done()))
            .collect()
    }
}

/// Lock-free trajectory store for IMPALA.
///
/// Uses crossbeam's Injector for lock-free trajectory insertion,
/// with periodic consolidation to a sorted storage for FIFO consumption.
pub struct TrajectoryStore {
    config: TrajectoryStoreConfig,
    /// Lock-free queue for trajectory injection
    injector: Injector<Trajectory<IMPALATransition>>,
    /// Consolidated storage for sampling
    storage: RwLock<Vec<Trajectory<IMPALATransition>>>,
    /// Size tracking (storage size)
    size: AtomicUsize,
    /// Injector pending count
    injector_size: AtomicUsize,
}

impl TrajectoryStore {
    /// Create a new trajectory store.
    pub fn new(config: TrajectoryStoreConfig) -> Self {
        Self {
            config,
            injector: Injector::new(),
            storage: RwLock::new(Vec::new()),
            size: AtomicUsize::new(0),
            injector_size: AtomicUsize::new(0),
        }
    }

    /// Push trajectory from actor (lock-free).
    ///
    /// Only stores trajectories that meet minimum length requirement.
    pub fn push(&self, trajectory: Trajectory<IMPALATransition>) {
        if trajectory.len() >= self.config.min_trajectory_len {
            self.injector.push(trajectory);
            self.injector_size.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Push multiple trajectories.
    pub fn push_batch(&self, trajectories: Vec<Trajectory<IMPALATransition>>) {
        for traj in trajectories {
            self.push(traj);
        }
    }

    /// Consolidate from injector to storage.
    ///
    /// Should be called periodically (e.g., by a consolidator thread or before sampling).
    pub fn consolidate(&self) {
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

        // Remove oldest if over capacity
        while storage.len() > self.config.max_trajectories {
            storage.remove(0);
        }

        if count > 0 {
            self.injector_size.fetch_sub(count, Ordering::Relaxed);
            self.size.store(storage.len(), Ordering::Relaxed);
        }
    }

    /// Sample batch of trajectories for training (FIFO).
    ///
    /// Consumes trajectories - they are removed from storage after sampling.
    pub fn sample(&self, n: usize) -> Option<TrajectoryBatch> {
        // First consolidate any pending trajectories
        self.consolidate();

        let mut storage = self.storage.write();
        if storage.len() < n {
            return None;
        }

        // FIFO: take oldest trajectories
        let trajectories: Vec<_> = storage.drain(0..n).collect();
        self.size.store(storage.len(), Ordering::Relaxed);

        Some(TrajectoryBatch { trajectories })
    }

    /// Get current size (consolidated storage only).
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Get pending injector size.
    pub fn pending_len(&self) -> usize {
        self.injector_size.load(Ordering::Relaxed)
    }

    /// Get total size (storage + pending).
    pub fn total_len(&self) -> usize {
        self.len() + self.pending_len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.total_len() == 0
    }

    /// Check if ready for training.
    pub fn is_ready(&self) -> bool {
        self.total_len() >= self.config.batch_size
    }

    /// Get configuration.
    pub fn config(&self) -> &TrajectoryStoreConfig {
        &self.config
    }
}

// Implement Send + Sync for thread safety
unsafe impl Send for TrajectoryStore {}
unsafe impl Sync for TrajectoryStore {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_trajectory(len: usize, env_id: usize) -> Trajectory<IMPALATransition> {
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
                policy_version: 1,
            });
        }
        traj
    }

    #[test]
    fn test_trajectory_store_new() {
        let store = TrajectoryStore::new(TrajectoryStoreConfig::default());
        assert!(store.is_empty());
        assert!(!store.is_ready());
    }

    #[test]
    fn test_trajectory_store_push_and_consolidate() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 2,
            max_trajectories: 100,
        };
        let store = TrajectoryStore::new(config);

        // Push trajectory that meets min length
        store.push(make_trajectory(10, 0));
        assert_eq!(store.pending_len(), 1);
        assert_eq!(store.len(), 0);

        // Consolidate
        store.consolidate();
        assert_eq!(store.pending_len(), 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_trajectory_store_rejects_short() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 10,
            batch_size: 1,
            max_trajectories: 100,
        };
        let store = TrajectoryStore::new(config);

        // Push trajectory that's too short
        store.push(make_trajectory(5, 0));
        assert_eq!(store.pending_len(), 0);

        // Push trajectory that meets min length
        store.push(make_trajectory(10, 0));
        assert_eq!(store.pending_len(), 1);
    }

    #[test]
    fn test_trajectory_store_sample_fifo() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 2,
            max_trajectories: 100,
        };
        let store = TrajectoryStore::new(config);

        // Push 3 trajectories with different env_ids
        store.push(make_trajectory(10, 0));
        store.push(make_trajectory(10, 1));
        store.push(make_trajectory(10, 2));
        store.consolidate();

        // Sample 2 - should get env_id 0 and 1 (FIFO)
        let batch = store.sample(2).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.trajectories[0].env_id, 0);
        assert_eq!(batch.trajectories[1].env_id, 1);

        // Only env_id 2 should remain
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_trajectory_store_max_capacity() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 1,
            max_trajectories: 3,
        };
        let store = TrajectoryStore::new(config);

        // Push 5 trajectories
        for i in 0..5 {
            store.push(make_trajectory(10, i));
        }
        store.consolidate();

        // Should only have 3 (max capacity)
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_trajectory_batch_helpers() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 2,
            batch_size: 2,
            max_trajectories: 100,
        };
        let store = TrajectoryStore::new(config);

        store.push(make_trajectory(3, 0));
        store.push(make_trajectory(3, 1));

        let batch = store.sample(2).unwrap();
        assert_eq!(batch.total_transitions(), 6);
        assert_eq!(batch.all_rewards().len(), 6);
        assert_eq!(batch.all_behavior_log_probs().len(), 6);
    }
}
