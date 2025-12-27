//! PPO-specific rollout buffer implementing ExperienceBuffer trait.
//!
//! This buffer is designed for on-policy algorithms where the entire
//! rollout is consumed after each training iteration.

use crate::core::experience_buffer::{ExperienceBuffer, OnPolicyBuffer};
use crate::core::transition::PPOTransition;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Configuration for PPO rollout buffer.
#[derive(Debug, Clone)]
pub struct PPORolloutBufferConfig {
    /// Number of actor threads.
    pub n_actors: usize,
    /// Environments per actor.
    pub n_envs_per_actor: usize,
    /// Steps per rollout per environment.
    pub rollout_length: usize,
}

impl Default for PPORolloutBufferConfig {
    fn default() -> Self {
        Self {
            n_actors: 4,
            n_envs_per_actor: 32,
            rollout_length: 128,
        }
    }
}

impl PPORolloutBufferConfig {
    /// Total environments across all actors.
    pub fn total_envs(&self) -> usize {
        self.n_actors * self.n_envs_per_actor
    }

    /// Expected capacity when full.
    pub fn capacity(&self) -> usize {
        self.total_envs() * self.rollout_length
    }
}

/// PPO rollout batch ready for training.
#[derive(Debug, Clone)]
pub struct PPORolloutBatch {
    /// All transitions from the rollout.
    pub transitions: Vec<PPOTransition>,
    /// Policy version when data was collected.
    pub policy_version: u64,
    /// Number of environments.
    pub n_envs: usize,
    /// Rollout length.
    pub rollout_length: usize,
}

impl PPORolloutBatch {
    /// Get all states as flat vector.
    pub fn states(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| t.base.state.iter().copied())
            .collect()
    }

    /// Get all next states.
    pub fn next_states(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| t.base.next_state.iter().copied())
            .collect()
    }

    /// Get all rewards.
    pub fn rewards(&self) -> Vec<f32> {
        self.transitions.iter().map(|t| t.base.reward).collect()
    }

    /// Get all values.
    pub fn values(&self) -> Vec<f32> {
        self.transitions.iter().map(|t| t.value).collect()
    }

    /// Get all log probs.
    pub fn log_probs(&self) -> Vec<f32> {
        self.transitions.iter().map(|t| t.log_prob).collect()
    }

    /// Get all done flags.
    pub fn dones(&self) -> Vec<bool> {
        self.transitions.iter().map(|t| t.done()).collect()
    }

    /// Number of transitions.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// State dimension (inferred from first transition).
    pub fn state_dim(&self) -> usize {
        self.transitions.first().map(|t| t.base.state.len()).unwrap_or(0)
    }
}

/// Thread-safe rollout buffer for distributed PPO.
///
/// Actors push transitions, and the learner drains the entire buffer
/// after each rollout is complete.
pub struct PPORolloutBuffer {
    config: PPORolloutBufferConfig,
    /// Storage for transitions.
    storage: RwLock<Vec<PPOTransition>>,
    /// Current step count (vectorized steps, not individual transitions).
    step_count: AtomicUsize,
    /// Ready flag.
    ready: AtomicBool,
    /// Policy version of collected data.
    policy_version: AtomicU64,
    /// Ready signal channel.
    ready_tx: Sender<()>,
    ready_rx: Receiver<()>,
}

impl PPORolloutBuffer {
    /// Create a new PPO rollout buffer.
    pub fn new(config: PPORolloutBufferConfig) -> Self {
        let capacity = config.capacity();
        let (ready_tx, ready_rx) = bounded(1);

        Self {
            config,
            storage: RwLock::new(Vec::with_capacity(capacity)),
            step_count: AtomicUsize::new(0),
            ready: AtomicBool::new(false),
            policy_version: AtomicU64::new(0),
            ready_tx,
            ready_rx,
        }
    }

    /// Push a batch of transitions from one vectorized step.
    ///
    /// Called by actors after stepping their environments.
    pub fn push_step(&self, transitions: Vec<PPOTransition>, version: u64) {
        let mut storage = self.storage.write();
        storage.extend(transitions);

        // Update policy version (take the most recent)
        self.policy_version.fetch_max(version, Ordering::Relaxed);

        let count = self.step_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Check if rollout is complete
        if count >= self.config.rollout_length {
            self.ready.store(true, Ordering::Release);
            let _ = self.ready_tx.try_send(());
        }
    }

    /// Wait for rollout to be ready (blocking).
    pub fn wait_ready(&self) -> bool {
        self.ready_rx.recv().is_ok()
    }

    /// Wait for rollout with timeout.
    pub fn wait_ready_timeout(&self, timeout: std::time::Duration) -> bool {
        self.ready_rx.recv_timeout(timeout).is_ok()
    }

    /// Consume the rollout and return a batch.
    pub fn consume(&self) -> PPORolloutBatch {
        let mut storage = self.storage.write();
        self.step_count.store(0, Ordering::SeqCst);
        self.ready.store(false, Ordering::Release);

        let version = self.policy_version.load(Ordering::Relaxed);

        PPORolloutBatch {
            transitions: std::mem::take(&mut *storage),
            policy_version: version,
            n_envs: self.config.total_envs(),
            rollout_length: self.config.rollout_length,
        }
    }

    /// Get current step count.
    pub fn step_count(&self) -> usize {
        self.step_count.load(Ordering::Relaxed)
    }

    /// Check if ready.
    pub fn is_rollout_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Get configuration.
    pub fn config(&self) -> &PPORolloutBufferConfig {
        &self.config
    }
}

// Implement ExperienceBuffer trait
impl ExperienceBuffer for PPORolloutBuffer {
    type Item = PPOTransition;

    fn push(&self, item: Self::Item) {
        let mut storage = self.storage.write();
        storage.push(item);

        // Check completion based on total items
        let total_envs = self.config.total_envs();
        if storage.len() >= self.config.rollout_length * total_envs {
            self.ready.store(true, Ordering::Release);
            let _ = self.ready_tx.try_send(());
        }
    }

    fn push_batch(&self, items: Vec<Self::Item>) {
        let mut storage = self.storage.write();
        storage.extend(items);

        let total_envs = self.config.total_envs();
        if storage.len() >= self.config.rollout_length * total_envs {
            self.ready.store(true, Ordering::Release);
            let _ = self.ready_tx.try_send(());
        }
    }

    fn len(&self) -> usize {
        self.storage.read().len()
    }

    fn capacity(&self) -> usize {
        self.config.capacity()
    }

    fn consolidate(&self) {
        // No-op for rollout buffer (no injector to consolidate)
    }

    fn clear(&self) {
        let mut storage = self.storage.write();
        storage.clear();
        self.step_count.store(0, Ordering::SeqCst);
        self.ready.store(false, Ordering::Release);
    }
}

// Implement OnPolicyBuffer trait
impl OnPolicyBuffer for PPORolloutBuffer {
    fn is_ready(&self, min_steps: usize) -> bool {
        self.len() >= min_steps || self.is_rollout_ready()
    }

    fn drain(&self) -> Vec<Self::Item> {
        let mut storage = self.storage.write();
        self.step_count.store(0, Ordering::SeqCst);
        self.ready.store(false, Ordering::Release);
        std::mem::take(&mut *storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_transition(i: usize) -> PPOTransition {
        PPOTransition {
            base: Transition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
                false,
            ),
            log_prob: -0.5,
            value: 1.0,
        }
    }

    #[test]
    fn test_ppo_buffer_new() {
        let config = PPORolloutBufferConfig {
            n_actors: 2,
            n_envs_per_actor: 4,
            rollout_length: 10,
        };
        let buffer = PPORolloutBuffer::new(config);

        assert!(buffer.is_empty());
        assert!(!buffer.is_rollout_ready());
        assert_eq!(buffer.capacity(), 80); // 2 * 4 * 10
    }

    #[test]
    fn test_ppo_buffer_push_and_consume() {
        let config = PPORolloutBufferConfig {
            n_actors: 1,
            n_envs_per_actor: 2,
            rollout_length: 3,
        };
        let buffer = PPORolloutBuffer::new(config);

        // Push 3 steps (6 transitions total)
        for step in 0..3 {
            let transitions = vec![
                make_transition(step * 2),
                make_transition(step * 2 + 1),
            ];
            buffer.push_step(transitions, 1);
        }

        assert!(buffer.is_rollout_ready());
        assert_eq!(buffer.len(), 6);

        let batch = buffer.consume();
        assert_eq!(batch.len(), 6);
        assert_eq!(batch.policy_version, 1);

        // Buffer should be reset
        assert!(buffer.is_empty());
        assert!(!buffer.is_rollout_ready());
    }

    #[test]
    fn test_experience_buffer_trait() {
        let config = PPORolloutBufferConfig {
            n_actors: 1,
            n_envs_per_actor: 2,
            rollout_length: 2,
        };
        let buffer = PPORolloutBuffer::new(config);

        // Test push_batch
        buffer.push_batch(vec![make_transition(0), make_transition(1)]);
        assert_eq!(buffer.len(), 2);

        // Test utilization
        assert!((buffer.utilization() - 0.5).abs() < 0.01);

        // Test clear
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_on_policy_buffer_trait() {
        let config = PPORolloutBufferConfig {
            n_actors: 1,
            n_envs_per_actor: 2,
            rollout_length: 2,
        };
        let buffer = PPORolloutBuffer::new(config);

        // Push transitions
        buffer.push_batch(vec![
            make_transition(0),
            make_transition(1),
            make_transition(2),
            make_transition(3),
        ]);

        assert!(buffer.is_ready(4));

        // Drain
        let items = buffer.drain();
        assert_eq!(items.len(), 4);
        assert!(buffer.is_empty());
    }
}
