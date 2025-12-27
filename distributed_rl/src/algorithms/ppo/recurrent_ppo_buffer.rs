//! Per-environment rollout buffer for distributed recurrent PPO training.
//!
//! This buffer stores `RecurrentPPOTransition`s which include hidden state
//! information needed for sequence-based training (TBPTT).

use crate::core::transition::RecurrentPPOTransition;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-environment rollout buffer for distributed recurrent PPO.
///
/// Stores transitions with hidden state information, enabling proper
/// sequence-based training with TBPTT (Truncated Backpropagation Through Time).
pub struct RecurrentPPOBuffer {
    /// Per-environment rollout storage
    envs: Mutex<Vec<Vec<RecurrentPPOTransition>>>,
    /// Rollout length per environment
    rollout_length: usize,
    /// Total number of environments across all actors
    total_envs: usize,
    /// Step counts per environment
    step_counts: Mutex<Vec<usize>>,
    /// Current consumed epoch (for synchronization)
    consumed_epoch: AtomicU64,
}

impl RecurrentPPOBuffer {
    /// Create a new recurrent PPO buffer.
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

    /// Push a batch of recurrent transitions from one actor's step.
    pub fn push_batch(&self, transitions: Vec<RecurrentPPOTransition>, global_env_offset: usize) {
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

    /// Check if the rollout is complete.
    pub fn is_ready(&self) -> bool {
        let counts = self.step_counts.lock();
        counts.iter().all(|&c| c >= self.rollout_length)
    }

    /// Total transitions currently stored.
    pub fn total_transitions(&self) -> usize {
        self.step_counts.lock().iter().sum()
    }

    /// Consume all rollouts, returning per-environment rollouts.
    pub fn consume(&self) -> Vec<Vec<RecurrentPPOTransition>> {
        let mut envs = self.envs.lock();
        let mut counts = self.step_counts.lock();

        let rollouts = std::mem::replace(
            &mut *envs,
            (0..self.total_envs)
                .map(|_| Vec::with_capacity(self.rollout_length))
                .collect(),
        );
        *counts = vec![0; self.total_envs];
        self.consumed_epoch.fetch_add(1, Ordering::Release);

        rollouts
    }

    /// Get current consumed epoch.
    pub fn consumed_epoch(&self) -> u64 {
        self.consumed_epoch.load(Ordering::Acquire)
    }

    /// Get rollout length.
    pub fn rollout_length(&self) -> usize {
        self.rollout_length
    }

    /// Get total environments.
    pub fn total_envs(&self) -> usize {
        self.total_envs
    }
}
