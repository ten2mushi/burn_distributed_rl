//! Shared training metrics for coordination between actors and learner.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Thread-safe training metrics.
#[derive(Debug)]
pub struct TrainingMetrics {
    /// Total environment steps
    env_steps: AtomicUsize,
    /// Total training steps
    train_steps: AtomicUsize,
    /// Total episodes completed
    episodes: AtomicUsize,
    /// Accumulated episode rewards (fixed-point: reward * 1000)
    reward_sum_fixed: AtomicUsize,
}

impl TrainingMetrics {
    /// Create new training metrics.
    pub fn new() -> Self {
        Self {
            env_steps: AtomicUsize::new(0),
            train_steps: AtomicUsize::new(0),
            episodes: AtomicUsize::new(0),
            reward_sum_fixed: AtomicUsize::new(0),
        }
    }

    /// Add environment steps.
    pub fn add_env_steps(&self, steps: usize) {
        self.env_steps.fetch_add(steps, Ordering::Relaxed);
    }

    /// Increment training steps.
    pub fn increment_train_steps(&self) {
        self.train_steps.fetch_add(1, Ordering::Relaxed);
    }

    /// Record episode completion.
    pub fn record_episode(&self, reward: f32) {
        self.episodes.fetch_add(1, Ordering::Relaxed);
        let reward_fixed = (reward * 1000.0) as usize;
        self.reward_sum_fixed.fetch_add(reward_fixed, Ordering::Relaxed);
    }

    /// Get total environment steps.
    pub fn env_steps(&self) -> usize {
        self.env_steps.load(Ordering::Relaxed)
    }

    /// Get total training steps.
    pub fn train_steps(&self) -> usize {
        self.train_steps.load(Ordering::Relaxed)
    }

    /// Get total episodes.
    pub fn episodes(&self) -> usize {
        self.episodes.load(Ordering::Relaxed)
    }

    /// Get average episode reward.
    pub fn avg_reward(&self) -> f32 {
        let episodes = self.episodes.load(Ordering::Relaxed);
        if episodes == 0 {
            return 0.0;
        }
        let sum = self.reward_sum_fixed.load(Ordering::Relaxed) as f32 / 1000.0;
        sum / episodes as f32
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        self.env_steps.store(0, Ordering::Relaxed);
        self.train_steps.store(0, Ordering::Relaxed);
        self.episodes.store(0, Ordering::Relaxed);
        self.reward_sum_fixed.store(0, Ordering::Relaxed);
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared training metrics.
pub type SharedTrainingMetrics = Arc<TrainingMetrics>;

/// Create new shared training metrics.
pub fn training_metrics() -> SharedTrainingMetrics {
    Arc::new(TrainingMetrics::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_new() {
        let metrics = TrainingMetrics::new();
        assert_eq!(metrics.env_steps(), 0);
        assert_eq!(metrics.train_steps(), 0);
        assert_eq!(metrics.episodes(), 0);
        assert_eq!(metrics.avg_reward(), 0.0);
    }

    #[test]
    fn test_training_metrics_add_steps() {
        let metrics = TrainingMetrics::new();
        metrics.add_env_steps(100);
        metrics.add_env_steps(50);
        assert_eq!(metrics.env_steps(), 150);
    }

    #[test]
    fn test_training_metrics_train_steps() {
        let metrics = TrainingMetrics::new();
        metrics.increment_train_steps();
        metrics.increment_train_steps();
        assert_eq!(metrics.train_steps(), 2);
    }

    #[test]
    fn test_training_metrics_episodes() {
        let metrics = TrainingMetrics::new();
        metrics.record_episode(100.0);
        metrics.record_episode(200.0);
        assert_eq!(metrics.episodes(), 2);
        assert!((metrics.avg_reward() - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_training_metrics_reset() {
        let metrics = TrainingMetrics::new();
        metrics.add_env_steps(100);
        metrics.increment_train_steps();
        metrics.record_episode(50.0);

        metrics.reset();

        assert_eq!(metrics.env_steps(), 0);
        assert_eq!(metrics.train_steps(), 0);
        assert_eq!(metrics.episodes(), 0);
    }

    #[test]
    fn test_shared_training_metrics() {
        let metrics = training_metrics();
        let metrics2 = Arc::clone(&metrics);

        metrics.add_env_steps(100);
        metrics2.increment_train_steps();

        assert_eq!(metrics.env_steps(), 100);
        assert_eq!(metrics2.train_steps(), 1);
    }
}
