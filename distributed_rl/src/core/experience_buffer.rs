//! Experience buffer trait for distributed RL algorithms.
//!
//! This module defines the `ExperienceBuffer` trait which abstracts over
//! different buffer types used by various RL algorithms:
//!
//! - **RolloutBuffer**: On-policy buffer for PPO (consumes entire rollout)
//! - **TrajectoryStore**: Off-policy buffer for IMPALA (FIFO sampling)

use std::sync::Arc;

/// Trait for experience buffers used in distributed RL.
///
/// Different algorithms use different buffer strategies:
/// - PPO: On-policy, collects rollouts and consumes them entirely
/// - IMPALA: Off-policy, stores trajectories and samples FIFO
///
/// # Type Parameters
/// - `Item`: The type of experience stored (Transition, PPOTransition, etc.)
pub trait ExperienceBuffer: Send + Sync {
    /// The type of individual experience items stored.
    type Item: Clone + Send;

    /// Push a single item to the buffer.
    fn push(&self, item: Self::Item);

    /// Push a batch of items to the buffer.
    fn push_batch(&self, items: Vec<Self::Item>);

    /// Get the current number of items in the buffer.
    fn len(&self) -> usize;

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the buffer capacity.
    fn capacity(&self) -> usize;

    /// Get buffer utilization as fraction (0.0 to 1.0).
    fn utilization(&self) -> f32 {
        self.len() as f32 / self.capacity() as f32
    }

    /// Consolidate any pending data.
    ///
    /// For lock-free buffers, this moves data from injector to storage.
    /// For simple buffers, this may be a no-op.
    fn consolidate(&self);

    /// Clear all data from the buffer.
    fn clear(&self);
}

/// Extension trait for on-policy buffers (like PPO's RolloutBuffer).
///
/// On-policy algorithms consume the entire buffer after each training step.
pub trait OnPolicyBuffer: ExperienceBuffer {
    /// Check if the buffer is ready for training.
    ///
    /// Returns true when enough rollout steps have been collected.
    fn is_ready(&self, min_steps: usize) -> bool;

    /// Drain all data from the buffer.
    ///
    /// Returns all stored items and clears the buffer.
    fn drain(&self) -> Vec<Self::Item>;
}

/// Extension trait for off-policy buffers (like IMPALA's TrajectoryStore).
///
/// Off-policy algorithms sample from the buffer without draining it.
pub trait OffPolicyBuffer: ExperienceBuffer {
    /// Sample a batch of items from the buffer.
    ///
    /// Returns `None` if the buffer has insufficient data.
    fn sample(&self, batch_size: usize) -> Option<Vec<Self::Item>>;

    /// Get the number of items in the pending queue (not yet consolidated).
    fn pending_len(&self) -> usize;
}

/// Shared experience buffer.
pub type SharedExperienceBuffer<B> = Arc<B>;

/// Configuration for experience buffers.
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Maximum capacity of the buffer.
    pub capacity: usize,

    /// Minimum samples before training can start.
    pub min_samples: usize,

    /// For on-policy: rollout length before draining.
    pub rollout_length: Option<usize>,

    /// For off-policy: trajectory length.
    pub trajectory_length: Option<usize>,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            capacity: 100_000,
            min_samples: 1_000,
            rollout_length: Some(128),
            trajectory_length: Some(20),
        }
    }
}

impl BufferConfig {
    /// Create config for PPO (on-policy).
    pub fn ppo(rollout_length: usize, n_envs: usize) -> Self {
        Self {
            capacity: rollout_length * n_envs * 2, // Double buffer
            min_samples: rollout_length * n_envs,
            rollout_length: Some(rollout_length),
            trajectory_length: None,
        }
    }

    /// Create config for IMPALA (off-policy).
    pub fn impala(trajectory_length: usize, capacity: usize, min_samples: usize) -> Self {
        Self {
            capacity,
            min_samples,
            rollout_length: None,
            trajectory_length: Some(trajectory_length),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_config_ppo() {
        let config = BufferConfig::ppo(128, 32);
        assert_eq!(config.rollout_length, Some(128));
        assert_eq!(config.min_samples, 128 * 32);
        assert!(config.trajectory_length.is_none());
    }

    #[test]
    fn test_buffer_config_impala() {
        let config = BufferConfig::impala(20, 100_000, 1_000);
        assert_eq!(config.trajectory_length, Some(20));
        assert!(config.rollout_length.is_none());
    }
}
