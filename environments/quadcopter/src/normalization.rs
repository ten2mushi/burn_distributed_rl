//! Reward normalization and clipping utilities.
//!
//! Provides running statistics tracking and reward normalization/clipping
//! for stabilizing RL training.

// ============================================================================
// Running Mean and Standard Deviation (Welford's algorithm)
// ============================================================================

/// Incremental computation of mean and variance using Welford's algorithm.
///
/// This provides numerically stable running statistics without storing all values.
#[derive(Clone, Debug)]
pub struct RunningMeanStd {
    mean: f32,
    var: f32,
    count: u64,
    epsilon: f32,
}

impl Default for RunningMeanStd {
    fn default() -> Self {
        Self::new(1e-4)
    }
}

impl RunningMeanStd {
    /// Create with specified epsilon for numerical stability.
    pub fn new(epsilon: f32) -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0,
            epsilon,
        }
    }

    /// Update statistics with a new value.
    #[inline]
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        // Use Welford's online algorithm for variance
        // M2 += delta * delta2
        // var = M2 / count
        self.var += (delta * delta2 - self.var) / self.count as f32;
    }

    /// Update statistics with multiple values.
    pub fn update_batch(&mut self, values: &[f32]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Get current mean.
    #[inline]
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Get current standard deviation.
    #[inline]
    pub fn std(&self) -> f32 {
        (self.var + self.epsilon).sqrt()
    }

    /// Get current variance.
    #[inline]
    pub fn var(&self) -> f32 {
        self.var
    }

    /// Get sample count.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Normalize a value using current statistics.
    #[inline]
    pub fn normalize(&self, value: f32) -> f32 {
        (value - self.mean) / self.std()
    }

    /// Normalize without centering (only scale).
    #[inline]
    pub fn scale(&self, value: f32) -> f32 {
        value / self.std()
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.var = 1.0;
        self.count = 0;
    }
}

// ============================================================================
// Normalization Configuration
// ============================================================================

/// Configuration for reward normalization and clipping.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizationConfig {
    /// Enable running normalization (zero-mean, unit-variance).
    pub enable_normalization: bool,

    /// Enable reward clipping after normalization.
    pub enable_clipping: bool,

    /// Clip range (min, max) applied after normalization.
    pub clip_range: (f32, f32),

    /// Epsilon for numerical stability in normalization.
    pub epsilon: f32,

    /// Minimum samples before normalization kicks in.
    pub warmup_samples: u64,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl NormalizationConfig {
    /// No normalization or clipping.
    pub const fn disabled() -> Self {
        Self {
            enable_normalization: false,
            enable_clipping: false,
            clip_range: (-10.0, 10.0),
            epsilon: 1e-4,
            warmup_samples: 100,
        }
    }

    /// Standard PPO-style normalization with clipping.
    pub fn ppo_default() -> Self {
        Self {
            enable_normalization: true,
            enable_clipping: true,
            clip_range: (-10.0, 10.0),
            epsilon: 1e-4,
            warmup_samples: 100,
        }
    }

    /// Clipping only (no normalization).
    pub fn clip_only(min: f32, max: f32) -> Self {
        Self {
            enable_normalization: false,
            enable_clipping: true,
            clip_range: (min, max),
            epsilon: 1e-4,
            warmup_samples: 0,
        }
    }

    /// Builder: enable normalization.
    pub fn with_normalization(mut self, enable: bool) -> Self {
        self.enable_normalization = enable;
        self
    }

    /// Builder: enable clipping with range.
    pub fn with_clipping(mut self, min: f32, max: f32) -> Self {
        self.enable_clipping = true;
        self.clip_range = (min, max);
        self
    }

    /// Builder: set warmup samples.
    pub fn with_warmup(mut self, samples: u64) -> Self {
        self.warmup_samples = samples;
        self
    }

    /// Check if any processing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enable_normalization || self.enable_clipping
    }
}

// ============================================================================
// Reward Processor
// ============================================================================

/// Processes rewards through normalization and clipping.
#[derive(Clone, Debug)]
pub struct RewardProcessor {
    config: NormalizationConfig,
    stats: RunningMeanStd,
}

impl RewardProcessor {
    /// Create a new processor with configuration.
    pub fn new(config: NormalizationConfig) -> Self {
        let stats = RunningMeanStd::new(config.epsilon);
        Self { config, stats }
    }

    /// Process a single reward value.
    #[inline]
    pub fn process(&mut self, reward: f32) -> f32 {
        // Update statistics
        if self.config.enable_normalization {
            self.stats.update(reward);
        }

        let mut processed = reward;

        // Apply normalization after warmup
        if self.config.enable_normalization && self.stats.count >= self.config.warmup_samples {
            processed = self.stats.normalize(processed);
        }

        // Apply clipping
        if self.config.enable_clipping {
            let (min, max) = self.config.clip_range;
            processed = processed.clamp(min, max);
        }

        processed
    }

    /// Process a batch of rewards in place.
    pub fn process_batch(&mut self, rewards: &mut [f32]) {
        for r in rewards.iter_mut() {
            *r = self.process(*r);
        }
    }

    /// Get current statistics (for logging/debugging).
    pub fn stats(&self) -> &RunningMeanStd {
        &self.stats
    }

    /// Reset the processor statistics.
    pub fn reset(&mut self) {
        self.stats.reset();
    }
}

impl Default for RewardProcessor {
    fn default() -> Self {
        Self::new(NormalizationConfig::disabled())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_mean_std_single_value() {
        let mut stats = RunningMeanStd::new(1e-8);
        stats.update(5.0);

        assert!((stats.mean() - 5.0).abs() < 1e-6);
        assert_eq!(stats.count(), 1);
    }

    #[test]
    fn test_running_mean_std_batch() {
        let mut stats = RunningMeanStd::new(1e-8);
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        stats.update_batch(&values);

        // Mean should be 3.0
        assert!((stats.mean() - 3.0).abs() < 1e-5);
        assert_eq!(stats.count(), 5);

        // Variance should be 2.0 (population variance)
        assert!((stats.var() - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_running_mean_std_normalize() {
        let mut stats = RunningMeanStd::new(1e-8);

        // Update with known distribution
        for i in 0..1000 {
            stats.update(i as f32 * 0.01);
        }

        // Normalized value at mean should be ~0
        let normalized = stats.normalize(stats.mean());
        assert!(normalized.abs() < 1e-5, "Normalized mean should be ~0");
    }

    #[test]
    fn test_normalization_config_disabled() {
        let config = NormalizationConfig::disabled();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_normalization_config_ppo() {
        let config = NormalizationConfig::ppo_default();
        assert!(config.enable_normalization);
        assert!(config.enable_clipping);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_reward_processor_passthrough() {
        let mut processor = RewardProcessor::new(NormalizationConfig::disabled());

        let reward = 5.0;
        let processed = processor.process(reward);

        assert_eq!(processed, reward, "Disabled processor should pass through");
    }

    #[test]
    fn test_reward_processor_clipping() {
        let mut processor = RewardProcessor::new(NormalizationConfig::clip_only(-1.0, 1.0));

        assert_eq!(processor.process(0.5), 0.5);
        assert_eq!(processor.process(5.0), 1.0);
        assert_eq!(processor.process(-5.0), -1.0);
    }

    #[test]
    fn test_reward_processor_normalization() {
        let config = NormalizationConfig::ppo_default().with_warmup(5);
        let mut processor = RewardProcessor::new(config);

        // Warmup phase
        for i in 0..10 {
            processor.process(i as f32);
        }

        // After warmup, values should be normalized
        let stats = processor.stats();
        assert!(stats.count() >= 10);
        assert!(stats.std() > 0.0);
    }

    #[test]
    fn test_reward_processor_batch() {
        let mut processor = RewardProcessor::new(NormalizationConfig::clip_only(-1.0, 1.0));

        let mut rewards = [0.0, 0.5, 2.0, -3.0, 0.8];
        processor.process_batch(&mut rewards);

        assert_eq!(rewards, [0.0, 0.5, 1.0, -1.0, 0.8]);
    }
}
