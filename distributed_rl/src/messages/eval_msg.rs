//! Messages for Evaluator pool.
//!
//! # Data Integrity
//!
//! `EvalResult::from_rewards()` validates input arrays and filters non-finite values
//! to ensure evaluation statistics are always meaningful and accurate.

/// Messages sent to evaluator threads.
#[derive(Debug)]
pub enum EvalMsg<M> {
    /// Stop the evaluator.
    Stop,

    /// Evaluate a model.
    Evaluate {
        /// Cloned model for evaluation.
        model: M,
        /// Training step when evaluation was requested.
        step: usize,
        /// Number of episodes to run.
        n_episodes: usize,
    },
}

impl<M: Clone> Clone for EvalMsg<M> {
    fn clone(&self) -> Self {
        match self {
            EvalMsg::Stop => EvalMsg::Stop,
            EvalMsg::Evaluate { model, step, n_episodes } => EvalMsg::Evaluate {
                model: model.clone(),
                step: *step,
                n_episodes: *n_episodes,
            },
        }
    }
}

/// Evaluation result.
///
/// Contains aggregated statistics from evaluation episodes.
/// All statistics are computed from finite values only; non-finite rewards
/// are filtered and tracked separately.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Training step when evaluation was requested.
    pub step: usize,

    /// Average reward over evaluation episodes (finite values only).
    pub avg_reward: f32,

    /// Standard deviation of rewards (finite values only).
    pub std_reward: f32,

    /// Minimum reward (finite values only).
    pub min_reward: f32,

    /// Maximum reward (finite values only).
    pub max_reward: f32,

    /// Number of episodes evaluated (total, including filtered).
    pub n_episodes: usize,

    /// Number of episodes with valid (finite) rewards.
    pub n_valid_episodes: usize,

    /// Number of episodes with non-finite rewards that were filtered.
    pub n_filtered_episodes: usize,

    /// Average episode length.
    pub avg_length: f32,
}

impl EvalResult {
    /// Create evaluation result from episode rewards and lengths.
    ///
    /// # Data Validation
    ///
    /// - **Array length mismatch**: If `rewards.len() != lengths.len()`, this function
    ///   panics in debug builds and uses `min(rewards.len(), lengths.len())` in release.
    /// - **Non-finite rewards**: NaN and Inf values are filtered from statistics but
    ///   tracked in `n_filtered_episodes` for diagnostics.
    /// - **Empty input**: Returns zeroed statistics with `n_episodes = 0`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `rewards.len() != lengths.len()`.
    pub fn from_rewards(step: usize, rewards: &[f32], lengths: &[usize]) -> Self {
        // Validate array lengths match
        debug_assert_eq!(
            rewards.len(),
            lengths.len(),
            "rewards and lengths arrays must have the same length: {} != {}",
            rewards.len(),
            lengths.len()
        );

        let n = rewards.len();
        if n == 0 {
            return Self {
                step,
                avg_reward: 0.0,
                std_reward: 0.0,
                min_reward: 0.0,
                max_reward: 0.0,
                n_episodes: 0,
                n_valid_episodes: 0,
                n_filtered_episodes: 0,
                avg_length: 0.0,
            };
        }

        // Use min length for safety in release builds
        let safe_len = n.min(lengths.len());

        // Filter finite rewards and track statistics
        let mut finite_rewards: Vec<f32> = Vec::with_capacity(n);
        let mut n_filtered = 0usize;

        for &r in rewards {
            if r.is_finite() {
                finite_rewards.push(r);
            } else {
                n_filtered += 1;
            }
        }

        let n_valid = finite_rewards.len();

        // Compute statistics from finite values only
        let (avg_reward, std_reward, min_reward, max_reward) = if n_valid == 0 {
            // All values were non-finite
            (0.0, 0.0, 0.0, 0.0)
        } else {
            // Use Welford's algorithm for numerically stable mean and variance
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;
            let mut min_r = f32::MAX;
            let mut max_r = f32::MIN;

            for (i, &r) in finite_rewards.iter().enumerate() {
                let delta = r - mean;
                mean += delta / (i + 1) as f32;
                let delta2 = r - mean;
                m2 += delta * delta2;

                min_r = min_r.min(r);
                max_r = max_r.max(r);
            }

            let variance = if n_valid > 1 { m2 / n_valid as f32 } else { 0.0 };
            (mean, variance.sqrt(), min_r, max_r)
        };

        // Compute average length using only the safe portion
        let total_length: usize = lengths.iter().take(safe_len).sum();
        let avg_length = if safe_len > 0 {
            total_length as f32 / safe_len as f32
        } else {
            0.0
        };

        Self {
            step,
            avg_reward,
            std_reward,
            min_reward,
            max_reward,
            n_episodes: n,
            n_valid_episodes: n_valid,
            n_filtered_episodes: n_filtered,
            avg_length,
        }
    }

    /// Check if this result meets a reward threshold.
    ///
    /// Uses the average of finite rewards only.
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.avg_reward >= threshold
    }

    /// Check if any episodes had non-finite rewards.
    pub fn has_filtered_episodes(&self) -> bool {
        self.n_filtered_episodes > 0
    }

    /// Get the fraction of episodes that were filtered due to non-finite rewards.
    pub fn filtered_fraction(&self) -> f32 {
        if self.n_episodes == 0 {
            0.0
        } else {
            self.n_filtered_episodes as f32 / self.n_episodes as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_result_from_rewards() {
        let rewards = vec![100.0, 200.0, 150.0, 250.0];
        let lengths = vec![100, 200, 150, 250];

        let result = EvalResult::from_rewards(1000, &rewards, &lengths);

        assert_eq!(result.step, 1000);
        assert_eq!(result.avg_reward, 175.0);
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 250.0);
        assert_eq!(result.n_episodes, 4);
        assert_eq!(result.n_valid_episodes, 4);
        assert_eq!(result.n_filtered_episodes, 0);
        assert_eq!(result.avg_length, 175.0);
    }

    #[test]
    fn test_eval_result_empty() {
        let result = EvalResult::from_rewards(0, &[], &[]);
        assert_eq!(result.n_episodes, 0);
        assert_eq!(result.n_valid_episodes, 0);
        assert_eq!(result.avg_reward, 0.0);
    }

    #[test]
    fn test_meets_threshold() {
        let result = EvalResult::from_rewards(0, &[450.0, 460.0, 440.0], &[100, 100, 100]);
        assert!(result.meets_threshold(450.0));
        assert!(!result.meets_threshold(500.0));
    }

    #[test]
    fn test_eval_result_nan_filtered() {
        let rewards = vec![100.0, f32::NAN, 200.0, f32::NAN];
        let lengths = vec![10, 20, 30, 40];

        let result = EvalResult::from_rewards(0, &rewards, &lengths);

        assert_eq!(result.n_episodes, 4);
        assert_eq!(result.n_valid_episodes, 2);
        assert_eq!(result.n_filtered_episodes, 2);
        assert_eq!(result.avg_reward, 150.0); // (100 + 200) / 2
        assert!(result.has_filtered_episodes());
        assert_eq!(result.filtered_fraction(), 0.5);
    }

    #[test]
    fn test_eval_result_infinity_filtered() {
        let rewards = vec![100.0, f32::INFINITY, f32::NEG_INFINITY];
        let lengths = vec![10, 20, 30];

        let result = EvalResult::from_rewards(0, &rewards, &lengths);

        assert_eq!(result.n_valid_episodes, 1);
        assert_eq!(result.n_filtered_episodes, 2);
        assert_eq!(result.avg_reward, 100.0);
    }

    #[test]
    fn test_eval_result_all_non_finite() {
        let rewards = vec![f32::NAN, f32::INFINITY];
        let lengths = vec![10, 20];

        let result = EvalResult::from_rewards(0, &rewards, &lengths);

        assert_eq!(result.n_episodes, 2);
        assert_eq!(result.n_valid_episodes, 0);
        assert_eq!(result.n_filtered_episodes, 2);
        // All filtered: return zeros
        assert_eq!(result.avg_reward, 0.0);
        assert_eq!(result.std_reward, 0.0);
        assert_eq!(result.min_reward, 0.0);
        assert_eq!(result.max_reward, 0.0);
    }

    #[test]
    fn test_eval_result_single_value() {
        let result = EvalResult::from_rewards(0, &[100.0], &[50]);

        assert_eq!(result.avg_reward, 100.0);
        assert_eq!(result.std_reward, 0.0); // Single value has 0 std
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 100.0);
    }

    #[test]
    fn test_eval_result_std_computation() {
        // Values: 10, 20, 30 -> mean = 20, variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 3 = 200/3
        let result = EvalResult::from_rewards(0, &[10.0, 20.0, 30.0], &[1, 1, 1]);

        assert!((result.avg_reward - 20.0).abs() < 1e-6);
        // std = sqrt(200/3) ≈ 8.165
        assert!((result.std_reward - 8.165).abs() < 0.01);
    }
}
