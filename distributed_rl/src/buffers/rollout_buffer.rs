//! Rollout buffer for on-policy algorithms (PPO/A2C).
//!
//! Key characteristics:
//! - Preserves temporal ordering within each environment
//! - Supports multiple parallel environments (vectorized)
//! - Cleared after each training iteration
//! - Thread-safe for concurrent actor push

use crate::core::transition::{Action, PPOTransition};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Configuration for rollout buffer.
#[derive(Debug, Clone)]
pub struct RolloutBufferConfig {
    /// Number of parallel environments
    pub n_envs: usize,
    /// Steps per rollout per environment
    pub rollout_len: usize,
}

impl Default for RolloutBufferConfig {
    fn default() -> Self {
        Self {
            n_envs: 64,
            rollout_len: 128,
        }
    }
}

/// Rollout data for PPO training.
///
/// Contains all transitions from a complete rollout, organized by
/// environment and time step.
#[derive(Debug)]
pub struct RolloutBatch {
    /// All transitions (interleaved: [env0_t0, env1_t0, ..., env0_t1, env1_t1, ...])
    pub transitions: Vec<PPOTransition>,
    /// Number of parallel environments
    pub n_envs: usize,
    /// Rollout length (steps per environment)
    pub rollout_len: usize,
}

impl RolloutBatch {
    /// Get transitions for a specific environment.
    ///
    /// Returns an iterator over transitions for env_idx in temporal order.
    pub fn env_transitions(&self, env_idx: usize) -> impl Iterator<Item = &PPOTransition> {
        self.transitions
            .iter()
            .skip(env_idx)
            .step_by(self.n_envs)
    }

    /// Get all states as flat vector.
    pub fn states(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| t.base.state.iter().copied())
            .collect()
    }

    /// Get all next states as flat vector.
    pub fn next_states(&self) -> Vec<f32> {
        self.transitions
            .iter()
            .flat_map(|t| t.base.next_state.iter().copied())
            .collect()
    }

    /// Get all actions (discrete).
    pub fn actions(&self) -> Vec<u32> {
        self.transitions
            .iter()
            .map(|t| match &t.base.action {
                Action::Discrete(a) => *a,
                _ => panic!("Expected discrete actions"),
            })
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

    /// Get total number of transitions.
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }
}

/// Thread-safe rollout buffer for PPO.
///
/// Collects transitions from vectorized environments until a complete
/// rollout is ready, then signals the learner.
pub struct RolloutBuffer {
    config: RolloutBufferConfig,
    /// Current rollout storage
    storage: RwLock<Vec<PPOTransition>>,
    /// Step counter (number of vectorized steps, not transitions)
    step_count: AtomicUsize,
    /// Ready flag
    ready: AtomicBool,
    /// Channel to signal ready
    ready_tx: Sender<()>,
    ready_rx: Receiver<()>,
}

impl RolloutBuffer {
    /// Create a new rollout buffer.
    pub fn new(config: RolloutBufferConfig) -> Self {
        let capacity = config.n_envs * config.rollout_len;
        let (ready_tx, ready_rx) = bounded(1);
        Self {
            config,
            storage: RwLock::new(Vec::with_capacity(capacity)),
            step_count: AtomicUsize::new(0),
            ready: AtomicBool::new(false),
            ready_tx,
            ready_rx,
        }
    }

    /// Push transitions from one vectorized step.
    ///
    /// Should be called with n_envs transitions (one per environment).
    pub fn push_step(&self, transitions: Vec<PPOTransition>) {
        debug_assert_eq!(
            transitions.len(),
            self.config.n_envs,
            "Expected {} transitions, got {}",
            self.config.n_envs,
            transitions.len()
        );

        let mut storage = self.storage.write();
        storage.extend(transitions);

        let count = self.step_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Check if rollout complete
        if count >= self.config.rollout_len {
            self.ready.store(true, Ordering::Release);
            let _ = self.ready_tx.try_send(());
        }
    }

    /// Check if rollout is ready for training.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Wait for rollout to be ready (blocking).
    ///
    /// Returns true if ready, false if channel was closed.
    pub fn wait_ready(&self) -> bool {
        self.ready_rx.recv().is_ok()
    }

    /// Wait for rollout with timeout.
    pub fn wait_ready_timeout(&self, timeout: std::time::Duration) -> bool {
        self.ready_rx.recv_timeout(timeout).is_ok()
    }

    /// Consume rollout data and reset buffer.
    ///
    /// Returns all transitions and clears the buffer for the next rollout.
    pub fn consume(&self) -> RolloutBatch {
        let mut storage = self.storage.write();
        self.step_count.store(0, Ordering::SeqCst);
        self.ready.store(false, Ordering::Release);

        RolloutBatch {
            transitions: std::mem::take(&mut *storage),
            n_envs: self.config.n_envs,
            rollout_len: self.config.rollout_len,
        }
    }

    /// Get current number of transitions.
    pub fn len(&self) -> usize {
        self.storage.read().len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get current step count.
    pub fn step_count(&self) -> usize {
        self.step_count.load(Ordering::Relaxed)
    }

    /// Get configuration.
    pub fn config(&self) -> &RolloutBufferConfig {
        &self.config
    }

    /// Get expected total transitions when full.
    pub fn capacity(&self) -> usize {
        self.config.n_envs * self.config.rollout_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::transition::Transition;

    fn make_transitions(n: usize) -> Vec<PPOTransition> {
        (0..n)
            .map(|i| PPOTransition {
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
            })
            .collect()
    }

    #[test]
    fn test_rollout_buffer_new() {
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 10,
        };
        let buffer = RolloutBuffer::new(config);

        assert!(buffer.is_empty());
        assert!(!buffer.is_ready());
        assert_eq!(buffer.capacity(), 40);
    }

    #[test]
    fn test_rollout_buffer_push_step() {
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        // Push first step (4 transitions)
        buffer.push_step(make_transitions(4));
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.step_count(), 1);
        assert!(!buffer.is_ready());

        // Push second step
        buffer.push_step(make_transitions(4));
        assert_eq!(buffer.len(), 8);
        assert_eq!(buffer.step_count(), 2);
        assert!(!buffer.is_ready());

        // Push third step - should be ready now
        buffer.push_step(make_transitions(4));
        assert_eq!(buffer.len(), 12);
        assert_eq!(buffer.step_count(), 3);
        assert!(buffer.is_ready());
    }

    #[test]
    fn test_rollout_buffer_consume() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_transitions(2));
        buffer.push_step(make_transitions(2));
        assert!(buffer.is_ready());

        let batch = buffer.consume();
        assert_eq!(batch.len(), 4);
        assert_eq!(batch.n_envs, 2);
        assert_eq!(batch.rollout_len, 2);

        // Buffer should be reset
        assert!(buffer.is_empty());
        assert!(!buffer.is_ready());
        assert_eq!(buffer.step_count(), 0);
    }

    #[test]
    fn test_rollout_batch_env_transitions() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        // Push 3 steps
        for step in 0..3 {
            let transitions = vec![
                PPOTransition {
                    base: Transition::new_discrete(
                        vec![step as f32 * 10.0],  // env 0
                        0, 1.0, vec![0.0], false, false,
                    ),
                    log_prob: -0.5,
                    value: 1.0,
                },
                PPOTransition {
                    base: Transition::new_discrete(
                        vec![step as f32 * 10.0 + 1.0],  // env 1
                        0, 1.0, vec![0.0], false, false,
                    ),
                    log_prob: -0.5,
                    value: 1.0,
                },
            ];
            buffer.push_step(transitions);
        }

        let batch = buffer.consume();

        // Check env 0 transitions
        let env0: Vec<_> = batch.env_transitions(0).collect();
        assert_eq!(env0.len(), 3);
        assert_eq!(env0[0].base.state[0], 0.0);
        assert_eq!(env0[1].base.state[0], 10.0);
        assert_eq!(env0[2].base.state[0], 20.0);

        // Check env 1 transitions
        let env1: Vec<_> = batch.env_transitions(1).collect();
        assert_eq!(env1.len(), 3);
        assert_eq!(env1[0].base.state[0], 1.0);
        assert_eq!(env1[1].base.state[0], 11.0);
        assert_eq!(env1[2].base.state[0], 21.0);
    }

    #[test]
    fn test_rollout_batch_helpers() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_transitions(2));
        buffer.push_step(make_transitions(2));

        let batch = buffer.consume();
        assert_eq!(batch.rewards().len(), 4);
        assert_eq!(batch.values().len(), 4);
        assert_eq!(batch.log_probs().len(), 4);
        assert_eq!(batch.dones().len(), 4);
        assert_eq!(batch.actions().len(), 4);
    }
}
