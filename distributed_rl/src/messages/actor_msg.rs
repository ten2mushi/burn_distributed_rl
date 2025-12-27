//! Messages for Actor threads.
//!
//! # Data Integrity
//!
//! All statistics computations filter non-finite values (NaN, Inf) to prevent
//! corruption of running averages. This is critical for distributed training
//! where a single corrupted value could poison aggregated metrics.

/// Messages sent to actor threads from Coordinator.
///
/// Actors receive commands via crossbeam channel and respond
/// with stats via a separate channel.
#[derive(Debug, Clone)]
pub enum ActorMsg<M> {
    /// Stop the actor gracefully.
    Stop,

    /// Update the inference model with new parameters.
    /// Model is cloned from Learner via ModelSlot.
    UpdateModel(M),

    /// Set the exploration rate (epsilon).
    SetEpsilon(f32),

    /// Request statistics from the actor.
    RequestStats,
}

/// Statistics reported by an actor.
///
/// Uses numerically stable algorithms for running averages to prevent
/// overflow and precision loss over long training runs.
#[derive(Debug, Clone, Default)]
pub struct ActorStats {
    /// Actor identifier.
    pub actor_id: usize,

    /// Total environment steps taken.
    pub steps: usize,

    /// Total episodes completed (including those with non-finite rewards).
    pub episodes: usize,

    /// Number of episodes with valid (finite) rewards used in average.
    pub valid_episodes: usize,

    /// Number of episodes with non-finite rewards (NaN/Inf) that were filtered.
    pub filtered_episodes: usize,

    /// Average episode reward (lifetime, computed from valid episodes only).
    pub avg_episode_reward: f32,

    /// Most recent episode reward (may be non-finite for diagnostics).
    pub recent_episode_reward: f32,

    /// Current exploration rate.
    pub epsilon: f32,

    /// Current model version being used.
    pub model_version: u64,
}

impl ActorStats {
    /// Create new actor stats.
    pub fn new(actor_id: usize) -> Self {
        Self {
            actor_id,
            ..Default::default()
        }
    }

    /// Update stats after episode completion.
    ///
    /// Uses Welford's online algorithm for numerically stable mean calculation.
    /// Non-finite rewards (NaN, Inf) are filtered out to prevent average corruption,
    /// but are still tracked in `filtered_episodes` for diagnostics.
    ///
    /// # Arguments
    ///
    /// * `reward` - Episode reward. Non-finite values are filtered from average
    ///              but recorded in `recent_episode_reward` for debugging.
    pub fn record_episode(&mut self, reward: f32) {
        self.episodes += 1;
        self.recent_episode_reward = reward;

        // Filter non-finite values to prevent average corruption
        if !reward.is_finite() {
            self.filtered_episodes += 1;
            return;
        }

        // Welford's online algorithm: numerically stable incremental mean
        // avg_new = avg_old + (x - avg_old) / n
        // This avoids overflow from multiplying avg * (n-1)
        self.valid_episodes += 1;
        let delta = reward - self.avg_episode_reward;
        self.avg_episode_reward += delta / self.valid_episodes as f32;
    }

    /// Update step count.
    ///
    /// Uses saturating arithmetic to prevent overflow.
    pub fn add_steps(&mut self, n: usize) {
        self.steps = self.steps.saturating_add(n);
    }

    /// Check if any episodes had non-finite rewards.
    pub fn has_filtered_episodes(&self) -> bool {
        self.filtered_episodes > 0
    }

    /// Get the fraction of episodes that were filtered due to non-finite rewards.
    pub fn filtered_fraction(&self) -> f32 {
        if self.episodes == 0 {
            0.0
        } else {
            self.filtered_episodes as f32 / self.episodes as f32
        }
    }

    /// Reset the average (useful if NaN corruption is suspected in upstream code).
    pub fn reset_average(&mut self) {
        self.avg_episode_reward = 0.0;
        self.valid_episodes = 0;
        self.filtered_episodes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_stats_record_episode() {
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        assert_eq!(stats.episodes, 1);
        assert_eq!(stats.valid_episodes, 1);
        assert_eq!(stats.avg_episode_reward, 100.0);
        assert_eq!(stats.recent_episode_reward, 100.0);

        stats.record_episode(200.0);
        assert_eq!(stats.episodes, 2);
        assert_eq!(stats.valid_episodes, 2);
        assert_eq!(stats.avg_episode_reward, 150.0);
        assert_eq!(stats.recent_episode_reward, 200.0);
    }

    #[test]
    fn test_actor_stats_add_steps() {
        let mut stats = ActorStats::new(1);
        stats.add_steps(100);
        stats.add_steps(50);
        assert_eq!(stats.steps, 150);
    }

    #[test]
    fn test_actor_stats_nan_filtered() {
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        stats.record_episode(f32::NAN);
        stats.record_episode(200.0);

        // NaN is filtered, average only uses finite values
        assert_eq!(stats.episodes, 3);
        assert_eq!(stats.valid_episodes, 2);
        assert_eq!(stats.filtered_episodes, 1);
        assert_eq!(stats.avg_episode_reward, 150.0); // (100 + 200) / 2
        assert_eq!(stats.recent_episode_reward, 200.0); // Last reward was 200.0
        assert!(stats.has_filtered_episodes());
    }

    #[test]
    fn test_actor_stats_infinity_filtered() {
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        stats.record_episode(f32::INFINITY);
        stats.record_episode(f32::NEG_INFINITY);

        assert_eq!(stats.episodes, 3);
        assert_eq!(stats.valid_episodes, 1);
        assert_eq!(stats.filtered_episodes, 2);
        assert_eq!(stats.avg_episode_reward, 100.0); // Only finite value
    }

    #[test]
    fn test_actor_stats_steps_saturating() {
        let mut stats = ActorStats::new(0);
        stats.steps = usize::MAX - 10;
        stats.add_steps(100);
        assert_eq!(stats.steps, usize::MAX); // Saturates, doesn't overflow
    }

    #[test]
    fn test_actor_stats_welford_precision() {
        let mut stats = ActorStats::new(0);

        // Many episodes with the same value should maintain precision
        for _ in 0..10000 {
            stats.record_episode(1.0);
        }

        // Welford's algorithm maintains precision
        assert!((stats.avg_episode_reward - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_actor_stats_reset_average() {
        let mut stats = ActorStats::new(0);
        stats.record_episode(100.0);
        stats.record_episode(f32::NAN);

        stats.reset_average();

        assert_eq!(stats.avg_episode_reward, 0.0);
        assert_eq!(stats.valid_episodes, 0);
        assert_eq!(stats.filtered_episodes, 0);
        // episodes count is preserved (total history)
        assert_eq!(stats.episodes, 2);
    }

    #[test]
    fn test_filtered_fraction() {
        let mut stats = ActorStats::new(0);
        assert_eq!(stats.filtered_fraction(), 0.0); // No episodes

        stats.record_episode(100.0);
        stats.record_episode(f32::NAN);
        stats.record_episode(f32::INFINITY);
        stats.record_episode(200.0);

        assert_eq!(stats.filtered_fraction(), 0.5); // 2 out of 4 filtered
    }
}
