//! Reward normalization based on return statistics.
//!
//! Normalizes rewards by dividing by the running standard deviation of returns.
//! This helps maintain consistent gradient magnitudes across different environments
//! with varying reward scales.
//!
//! # Theory
//!
//! Different environments have different reward scales:
//! - CartPole: +1 per step, returns up to 500
//! - Pendulum: -16 to 0 per step, returns -1600 to 0
//!
//! Without normalization, the same hyperparameters won't work across environments.
//!
//! # Important
//!
//! We normalize by the return standard deviation, NOT individual reward variance.
//! This preserves the relative magnitude of rewards within an episode while
//! standardizing across episodes.
//!
//! # Usage
//!
//! ```ignore
//! let mut normalizer = RewardNormalizer::new();
//!
//! // Update with episode returns (call when episode ends)
//! normalizer.update_return(episode_return);
//!
//! // Normalize individual rewards
//! let normalized_reward = normalizer.normalize(reward);
//! ```

use crate::core::RunningScalarStats;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Reward normalizer based on return statistics.
///
/// Normalizes rewards by dividing by the running standard deviation of episodic returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardNormalizer {
    /// Running statistics for episodic returns
    return_stats: RunningScalarStats,
    /// Current accumulated return for each environment
    current_returns: Vec<f32>,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Whether to update statistics (false during evaluation)
    update_stats: bool,
    /// Clip normalized rewards to this range
    clip_range: Option<(f32, f32)>,
}

impl RewardNormalizer {
    /// Create a new reward normalizer for the given number of environments.
    pub fn new(n_envs: usize) -> Self {
        Self {
            return_stats: RunningScalarStats::new(),
            current_returns: vec![0.0; n_envs],
            epsilon: 1e-8,
            update_stats: true,
            clip_range: Some((-10.0, 10.0)),
        }
    }

    /// Create with custom epsilon.
    pub fn with_epsilon(n_envs: usize, epsilon: f64) -> Self {
        Self {
            return_stats: RunningScalarStats::new(),
            current_returns: vec![0.0; n_envs],
            epsilon,
            update_stats: true,
            clip_range: Some((-10.0, 10.0)),
        }
    }

    /// Set clip range for normalized rewards.
    pub fn with_clip_range(mut self, range: Option<(f32, f32)>) -> Self {
        self.clip_range = range;
        self
    }

    /// Update with a reward and done flag for a specific environment.
    ///
    /// Call this for each (reward, done) pair during training.
    /// When done=true, the return is recorded and the accumulator resets.
    pub fn update(&mut self, env_id: usize, reward: f32, done: bool) {
        if !self.update_stats {
            return;
        }

        self.current_returns[env_id] += reward;

        if done {
            self.return_stats.update(self.current_returns[env_id]);
            self.current_returns[env_id] = 0.0;
        }
    }

    /// Update with batched rewards and dones.
    ///
    /// # Arguments
    /// * `rewards` - Rewards for each environment
    /// * `dones` - Done flags for each environment
    pub fn update_batch(&mut self, rewards: &[f32], dones: &[bool]) {
        assert_eq!(rewards.len(), dones.len());
        assert_eq!(rewards.len(), self.current_returns.len());

        for (env_id, (&reward, &done)) in rewards.iter().zip(dones.iter()).enumerate() {
            self.update(env_id, reward, done);
        }
    }

    /// Normalize a single reward.
    pub fn normalize(&self, reward: f32) -> f32 {
        let std = self.return_stats.std().max(self.epsilon) as f32;
        let normalized = reward / std;

        if let Some((low, high)) = self.clip_range {
            normalized.clamp(low, high)
        } else {
            normalized
        }
    }

    /// Normalize a batch of rewards (DEPRECATED: use normalize_returns for GAE returns).
    ///
    /// WARNING: This divides per-step rewards by return_std, which creates a scale mismatch.
    /// For proper normalization, use `normalize_returns` on GAE-computed returns instead.
    pub fn normalize_batch(&self, rewards: &[f32]) -> Vec<f32> {
        let std = self.return_stats.std().max(self.epsilon) as f32;
        rewards
            .iter()
            .map(|&r| {
                let normalized = r / std;
                if let Some((low, high)) = self.clip_range {
                    normalized.clamp(low, high)
                } else {
                    normalized
                }
            })
            .collect()
    }

    /// Normalize GAE-computed returns by return standard deviation.
    ///
    /// This is the correct normalization for PPO training:
    /// - Returns and return_std are on the same scale
    /// - Dividing returns by return_std gives normalized targets for the value function
    /// - Clipping is applied to prevent extreme values
    ///
    /// # Arguments
    /// * `returns` - GAE-computed returns (advantages + values)
    ///
    /// # Returns
    /// Normalized returns suitable for value function training
    pub fn normalize_returns(&self, returns: &[f32]) -> Vec<f32> {
        let std = self.return_stats.std().max(self.epsilon) as f32;

        // Don't normalize if we haven't seen enough data
        if self.return_stats.count() < 2.0 {
            return returns.to_vec();
        }

        returns
            .iter()
            .map(|&r| {
                let normalized = r / std;
                if let Some((low, high)) = self.clip_range {
                    normalized.clamp(low, high)
                } else {
                    normalized
                }
            })
            .collect()
    }

    /// Get the current return standard deviation.
    pub fn std(&self) -> f64 {
        self.return_stats.std().max(self.epsilon)
    }

    /// Get the number of returns observed.
    pub fn count(&self) -> f64 {
        self.return_stats.count()
    }

    /// Get the mean return.
    pub fn mean_return(&self) -> f64 {
        self.return_stats.mean()
    }

    /// Set training mode.
    pub fn set_training(&mut self, training: bool) {
        self.update_stats = training;
    }

    /// Reset all statistics and accumulators.
    pub fn reset(&mut self) {
        self.return_stats.reset();
        self.current_returns.fill(0.0);
    }

    /// Reset accumulator for a specific environment (e.g., on manual reset).
    pub fn reset_env(&mut self, env_id: usize) {
        self.current_returns[env_id] = 0.0;
    }

    /// Merge statistics from another normalizer.
    pub fn merge(&mut self, other: &RewardNormalizer) {
        self.return_stats.merge(&other.return_stats);
    }

    /// Synchronize with aggregated statistics.
    pub fn sync_with(&mut self, other: &RewardNormalizer) {
        self.return_stats = other.return_stats.clone();
        // Keep local current_returns as they track in-progress episodes
    }

    /// Get underlying statistics for serialization.
    pub fn stats(&self) -> &RunningScalarStats {
        &self.return_stats
    }
}

/// Thread-safe wrapper for reward normalizer.
#[derive(Debug, Clone)]
pub struct SharedRewardNormalizer {
    inner: Arc<RwLock<RewardNormalizer>>,
}

impl SharedRewardNormalizer {
    /// Create a new thread-safe reward normalizer.
    pub fn new(n_envs: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RewardNormalizer::new(n_envs))),
        }
    }

    /// Update with reward and done flag.
    pub fn update(&self, env_id: usize, reward: f32, done: bool) {
        self.inner.write().update(env_id, reward, done);
    }

    /// Update with batched rewards and dones.
    pub fn update_batch(&self, rewards: &[f32], dones: &[bool]) {
        self.inner.write().update_batch(rewards, dones);
    }

    /// Normalize a reward.
    pub fn normalize(&self, reward: f32) -> f32 {
        self.inner.read().normalize(reward)
    }

    /// Normalize a batch of rewards (DEPRECATED: use normalize_returns instead).
    pub fn normalize_batch(&self, rewards: &[f32]) -> Vec<f32> {
        self.inner.read().normalize_batch(rewards)
    }

    /// Normalize GAE-computed returns by return standard deviation.
    ///
    /// This is the correct normalization for PPO training.
    pub fn normalize_returns(&self, returns: &[f32]) -> Vec<f32> {
        self.inner.read().normalize_returns(returns)
    }

    /// Get a snapshot of the normalizer.
    pub fn snapshot(&self) -> RewardNormalizer {
        self.inner.read().clone()
    }

    /// Merge with another normalizer's statistics.
    pub fn merge(&self, other: &RewardNormalizer) {
        self.inner.write().merge(other);
    }

    /// Sync with aggregated statistics.
    pub fn sync_with(&self, other: &RewardNormalizer) {
        self.inner.write().sync_with(other);
    }

    /// Get the current return std.
    pub fn std(&self) -> f64 {
        self.inner.read().std()
    }

    /// Get the return count.
    pub fn count(&self) -> f64 {
        self.inner.read().count()
    }

    /// Set training mode.
    pub fn set_training(&self, training: bool) {
        self.inner.write().set_training(training);
    }

    /// Reset statistics.
    pub fn reset(&self) {
        self.inner.write().reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_normalizer_basic() {
        let mut normalizer = RewardNormalizer::new(1);

        // Simulate some episodes with different returns
        // Episode 1: return = 100
        for _ in 0..99 {
            normalizer.update(0, 1.0, false);
        }
        normalizer.update(0, 1.0, true);

        // Episode 2: return = 200
        for _ in 0..199 {
            normalizer.update(0, 1.0, false);
        }
        normalizer.update(0, 1.0, true);

        // Should have seen 2 returns
        assert!((normalizer.count() - 2.0).abs() < 1e-10);

        // Mean return should be 150
        assert!((normalizer.mean_return() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_reward_normalizer_multi_env() {
        let mut normalizer = RewardNormalizer::new(2);

        // Env 0: short episode (return = 10)
        for _ in 0..9 {
            normalizer.update(0, 1.0, false);
        }
        normalizer.update(0, 1.0, true);

        // Env 1: still running
        for _ in 0..5 {
            normalizer.update(1, 1.0, false);
        }

        // Only 1 return recorded (from env 0)
        assert!((normalizer.count() - 1.0).abs() < 1e-10);

        // Env 1 finishes
        for _ in 0..14 {
            normalizer.update(1, 1.0, false);
        }
        normalizer.update(1, 1.0, true);

        // Now 2 returns (10 and 20)
        assert!((normalizer.count() - 2.0).abs() < 1e-10);
        assert!((normalizer.mean_return() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_reward_normalize() {
        let mut normalizer = RewardNormalizer::new(1);

        // Create returns with known std
        // Returns: [100, 200] -> mean=150, var=2500, std=50
        for _ in 0..99 {
            normalizer.update(0, 1.0, false);
        }
        normalizer.update(0, 1.0, true);
        for _ in 0..199 {
            normalizer.update(0, 1.0, false);
        }
        normalizer.update(0, 1.0, true);

        // Normalize a reward of 50 -> should be ~1.0 (50/50)
        let normalized = normalizer.normalize(50.0);
        assert!((normalized - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_training_mode() {
        let mut normalizer = RewardNormalizer::new(1);

        // Training mode
        normalizer.update(0, 10.0, true);
        assert!((normalizer.count() - 1.0).abs() < 1e-10);

        // Eval mode - stats should not update
        normalizer.set_training(false);
        normalizer.update(0, 20.0, true);
        assert!((normalizer.count() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clipping() {
        let mut normalizer = RewardNormalizer::new(1).with_clip_range(Some((-5.0, 5.0)));

        // Create normalizer with std ~= 1
        for _ in 0..100 {
            normalizer.update(0, 1.0, true);
        }

        // Large reward should be clipped
        let normalized = normalizer.normalize(100.0);
        assert!(normalized <= 5.0);
        assert!(normalized >= -5.0);
    }
}
