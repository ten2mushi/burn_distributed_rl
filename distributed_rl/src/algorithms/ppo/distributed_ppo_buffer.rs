//! Per-environment rollout buffer for distributed PPO training.
//!
//! This buffer stores transitions organized by environment ID, which is
//! essential for computing GAE correctly in a multi-actor setting. Each
//! environment's trajectory must be processed independently for proper
//! advantage estimation.

use crate::core::transition::PPOTransition;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-environment rollout buffer for distributed PPO.
///
/// Unlike a flat buffer, this stores transitions organized by environment,
/// enabling correct per-environment GAE computation. This is critical for
/// distributed training where actors push transitions asynchronously.
///
/// # Thread Safety
///
/// Uses `parking_lot::Mutex` for fast locking. Actors push transitions
/// with their global env offset, and the learner consumes all at once.
pub struct DistributedPPOBuffer {
    /// Per-environment rollout storage: `envs[global_env_id] = Vec<PPOTransition>`
    envs: Mutex<Vec<Vec<PPOTransition>>>,
    /// Rollout length per environment
    rollout_length: usize,
    /// Total number of environments across all actors
    total_envs: usize,
    /// Step counts per environment
    step_counts: Mutex<Vec<usize>>,
    /// Current consumed epoch (for synchronization)
    consumed_epoch: AtomicU64,
}

impl DistributedPPOBuffer {
    /// Create a new distributed PPO buffer.
    ///
    /// # Arguments
    ///
    /// * `rollout_length` - Steps to collect per environment
    /// * `total_envs` - Total environments across all actors
    pub fn new(rollout_length: usize, total_envs: usize) -> Self {
        Self {
            envs: Mutex::new(
                (0..total_envs)
                    .map(|_| Vec::with_capacity(rollout_length))
                    .collect(),
            ),
            rollout_length,
            total_envs,
            step_counts: Mutex::new(vec![0; total_envs]),
            consumed_epoch: AtomicU64::new(0),
        }
    }

    /// Push a batch of transitions from one actor's step.
    ///
    /// Each transition is stored in its corresponding environment's rollout.
    ///
    /// # Arguments
    ///
    /// * `transitions` - One transition per environment in this actor
    /// * `global_env_offset` - Starting global env ID for this actor
    ///                         (e.g., actor 2 with 32 envs/actor has offset 64)
    pub fn push_batch(&self, transitions: Vec<PPOTransition>, global_env_offset: usize) {
        let mut envs = self.envs.lock();
        let mut counts = self.step_counts.lock();

        for (local_idx, transition) in transitions.into_iter().enumerate() {
            let global_env_id = global_env_offset + local_idx;
            if global_env_id < self.total_envs && counts[global_env_id] < self.rollout_length {
                envs[global_env_id].push(transition);
                counts[global_env_id] += 1;
            }
        }
    }

    /// Check if the rollout is complete (all envs have rollout_length transitions).
    pub fn is_ready(&self) -> bool {
        let counts = self.step_counts.lock();
        counts.iter().all(|&c| c >= self.rollout_length)
    }

    /// Get current progress as (min_steps, total_envs).
    pub fn progress(&self) -> (usize, usize) {
        let counts = self.step_counts.lock();
        let min = counts.iter().copied().min().unwrap_or(0);
        (min, self.total_envs)
    }

    /// Total transitions currently stored.
    pub fn total_transitions(&self) -> usize {
        self.step_counts.lock().iter().sum()
    }

    /// Consume all rollouts, returning per-environment rollouts.
    ///
    /// This resets the buffer for the next rollout. Returns a vector where
    /// each element is one environment's complete rollout.
    ///
    /// # Returns
    ///
    /// `Vec<Vec<PPOTransition>>` - rollouts[env_id] = transitions for that env
    pub fn consume(&self) -> Vec<Vec<PPOTransition>> {
        let mut envs = self.envs.lock();
        let mut counts = self.step_counts.lock();

        // Take all rollouts and reset
        let rollouts = std::mem::replace(
            &mut *envs,
            (0..self.total_envs)
                .map(|_| Vec::with_capacity(self.rollout_length))
                .collect(),
        );
        *counts = vec![0; self.total_envs];

        // Increment consumed epoch for synchronization
        self.consumed_epoch.fetch_add(1, Ordering::Release);

        rollouts
    }

    /// Get current consumed epoch.
    ///
    /// Actors use this to synchronize - they wait until `consumed_epoch > their_epoch`
    /// before collecting more data.
    pub fn consumed_epoch(&self) -> u64 {
        self.consumed_epoch.load(Ordering::Acquire)
    }

    /// Clear the buffer without consuming.
    pub fn clear(&self) {
        let mut envs = self.envs.lock();
        let mut counts = self.step_counts.lock();

        for env_rollout in envs.iter_mut() {
            env_rollout.clear();
        }
        *counts = vec![0; self.total_envs];
    }

    /// Get configuration info.
    pub fn rollout_length(&self) -> usize {
        self.rollout_length
    }

    /// Get total environments.
    pub fn total_envs(&self) -> usize {
        self.total_envs
    }
}

/// Batch of per-environment rollouts ready for training.
///
/// Contains all rollouts organized by environment, along with metadata
/// needed for training (GAE computation, etc.).
#[derive(Debug, Clone)]
pub struct DistributedPPORollouts {
    /// Per-environment rollouts: `rollouts[env_id] = transitions`
    pub rollouts: Vec<Vec<PPOTransition>>,
    /// Total number of environments
    pub n_envs: usize,
    /// Rollout length per environment
    pub rollout_length: usize,
}

impl DistributedPPORollouts {
    /// Create from consumed buffer data.
    pub fn new(rollouts: Vec<Vec<PPOTransition>>, rollout_length: usize) -> Self {
        let n_envs = rollouts.len();
        Self {
            rollouts,
            n_envs,
            rollout_length,
        }
    }

    /// Total number of transitions.
    pub fn total_transitions(&self) -> usize {
        self.rollouts.iter().map(|r| r.len()).sum()
    }

    /// Filter out empty rollouts (for robustness).
    pub fn non_empty_rollouts(&self) -> impl Iterator<Item = (usize, &Vec<PPOTransition>)> {
        self.rollouts
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.is_empty())
    }

    /// Get observation dimension (from first transition).
    pub fn obs_dim(&self) -> Option<usize> {
        self.rollouts
            .iter()
            .find(|r| !r.is_empty())
            .and_then(|r| r.first())
            .map(|t| t.base.state.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_transition(env_id: usize, step: usize) -> PPOTransition {
        PPOTransition {
            base: Transition::new_discrete(
                vec![env_id as f32, step as f32],
                0,
                1.0,
                vec![env_id as f32, (step + 1) as f32],
                false,
                false,
            ),
            log_prob: -0.5,
            value: 1.0,
        }
    }

    #[test]
    fn test_buffer_creation() {
        let buffer = DistributedPPOBuffer::new(64, 128);
        assert_eq!(buffer.rollout_length(), 64);
        assert_eq!(buffer.total_envs(), 128);
        assert!(!buffer.is_ready());
        assert_eq!(buffer.total_transitions(), 0);
    }

    #[test]
    fn test_push_batch() {
        // 2 actors, 2 envs each, rollout length 3
        let buffer = DistributedPPOBuffer::new(3, 4);

        // Actor 0 (envs 0-1) pushes step 0
        buffer.push_batch(
            vec![make_transition(0, 0), make_transition(1, 0)],
            0, // global offset
        );
        assert_eq!(buffer.total_transitions(), 2);

        // Actor 1 (envs 2-3) pushes step 0
        buffer.push_batch(
            vec![make_transition(2, 0), make_transition(3, 0)],
            2, // global offset
        );
        assert_eq!(buffer.total_transitions(), 4);

        // Not ready yet (need 3 steps per env)
        assert!(!buffer.is_ready());
    }

    #[test]
    fn test_rollout_completion() {
        let buffer = DistributedPPOBuffer::new(2, 2);

        // Step 0
        buffer.push_batch(vec![make_transition(0, 0), make_transition(1, 0)], 0);
        assert!(!buffer.is_ready());

        // Step 1
        buffer.push_batch(vec![make_transition(0, 1), make_transition(1, 1)], 0);
        assert!(buffer.is_ready());
    }

    #[test]
    fn test_consume() {
        let buffer = DistributedPPOBuffer::new(2, 2);

        // Fill buffer
        buffer.push_batch(vec![make_transition(0, 0), make_transition(1, 0)], 0);
        buffer.push_batch(vec![make_transition(0, 1), make_transition(1, 1)], 0);

        assert!(buffer.is_ready());

        // Consume
        let rollouts = buffer.consume();
        assert_eq!(rollouts.len(), 2);
        assert_eq!(rollouts[0].len(), 2);
        assert_eq!(rollouts[1].len(), 2);

        // Buffer should be reset
        assert!(!buffer.is_ready());
        assert_eq!(buffer.total_transitions(), 0);
        assert_eq!(buffer.consumed_epoch(), 1);
    }

    #[test]
    fn test_consumed_epoch_synchronization() {
        let buffer = DistributedPPOBuffer::new(1, 1);

        assert_eq!(buffer.consumed_epoch(), 0);

        buffer.push_batch(vec![make_transition(0, 0)], 0);
        let _ = buffer.consume();
        assert_eq!(buffer.consumed_epoch(), 1);

        buffer.push_batch(vec![make_transition(0, 1)], 0);
        let _ = buffer.consume();
        assert_eq!(buffer.consumed_epoch(), 2);
    }

    #[test]
    fn test_distributed_ppo_rollouts() {
        let rollouts = vec![
            vec![make_transition(0, 0), make_transition(0, 1)],
            vec![make_transition(1, 0), make_transition(1, 1)],
            vec![], // Empty rollout (edge case)
        ];

        let batch = DistributedPPORollouts::new(rollouts, 2);
        assert_eq!(batch.n_envs, 3);
        assert_eq!(batch.total_transitions(), 4);
        assert_eq!(batch.obs_dim(), Some(2));

        let non_empty: Vec<_> = batch.non_empty_rollouts().collect();
        assert_eq!(non_empty.len(), 2);
    }
}
