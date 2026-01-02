//! Observation normalization using running mean and standard deviation.
//!
//! Normalizes observations to approximately zero mean and unit variance,
//! which helps neural networks learn more efficiently.
//!
//! # Theory
//!
//! Different observation dimensions often have vastly different scales:
//! - CartPole position: [-4.8, 4.8]
//! - CartPole velocity: [-50, 50]
//! - CartPole angle: [-0.418, 0.418]
//!
//! Without normalization, gradients are dominated by large-scale features.
//! Normalization ensures equal gradient contribution from all features.
//!
//! # Usage
//!
//! ```ignore
//! let normalizer = ObservationNormalizer::new(4); // 4-dim observations
//!
//! // During training: update statistics and normalize
//! normalizer.update_and_normalize(&obs);
//!
//! // During evaluation: only normalize (don't update stats)
//! normalizer.normalize(&obs);
//! ```
//!
//! # Distributed Training
//!
//! In multi-actor setups, statistics must be synchronized:
//!
//! ```ignore
//! // Each actor accumulates local statistics
//! actor_normalizer.update(&obs);
//!
//! // Periodically sync with aggregated statistics
//! let aggregated = merge_all_actor_stats();
//! actor_normalizer.sync_with(&aggregated);
//! ```

use crate::core::{RunningMeanStd, SharedRunningMeanStd};
use burn::prelude::*;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for observation normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsNormalizationConfig {
    /// Clip normalized observations to this range (optional).
    /// Common values: (-10.0, 10.0) or (-5.0, 5.0)
    pub clip_range: Option<(f32, f32)>,

    /// Epsilon for numerical stability in division.
    pub epsilon: f64,

    /// Whether to update statistics during training.
    /// Set to false during evaluation to use fixed statistics.
    pub update_stats: bool,
}

impl Default for ObsNormalizationConfig {
    fn default() -> Self {
        Self {
            clip_range: Some((-10.0, 10.0)),
            epsilon: 1e-8,
            update_stats: true,
        }
    }
}

impl ObsNormalizationConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set clip range for normalized observations.
    pub fn with_clip_range(mut self, range: Option<(f32, f32)>) -> Self {
        self.clip_range = range;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set whether to update statistics during training.
    pub fn with_update_stats(mut self, update: bool) -> Self {
        self.update_stats = update;
        self
    }
}

/// Observation normalizer using running mean and standard deviation.
///
/// Wraps [`RunningMeanStd`] and provides both vector and tensor interfaces.
#[derive(Debug, Clone)]
pub struct ObservationNormalizer {
    stats: RunningMeanStd,
    config: ObsNormalizationConfig,
}

impl ObservationNormalizer {
    /// Create a new normalizer for observations with the given dimensionality.
    pub fn new(obs_dim: usize) -> Self {
        Self {
            stats: RunningMeanStd::new(obs_dim),
            config: ObsNormalizationConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(obs_dim: usize, config: ObsNormalizationConfig) -> Self {
        Self {
            stats: RunningMeanStd::with_epsilon(obs_dim, config.epsilon),
            config,
        }
    }

    /// Get the observation dimensionality.
    pub fn obs_dim(&self) -> usize {
        self.stats.dim()
    }

    /// Get the number of observations seen.
    pub fn count(&self) -> f64 {
        self.stats.count()
    }

    /// Update statistics with a single observation.
    pub fn update(&mut self, obs: &[f32]) {
        if self.config.update_stats {
            self.stats.update(obs);
        }
    }

    /// Update statistics with a batch of observations.
    ///
    /// # Arguments
    /// * `batch` - Flattened observations [obs1, obs2, ...] where each has `obs_dim` elements
    pub fn update_batch(&mut self, batch: &[f32]) {
        if self.config.update_stats {
            self.stats.update_batch(batch);
        }
    }

    /// Normalize an observation without updating statistics.
    pub fn normalize(&self, obs: &[f32]) -> Vec<f32> {
        let normalized = self.stats.normalize(obs);
        self.maybe_clip(normalized)
    }

    /// Update statistics and normalize in one step.
    pub fn update_and_normalize(&mut self, obs: &[f32]) -> Vec<f32> {
        self.update(obs);
        self.normalize(obs)
    }

    /// Normalize a batch of observations.
    ///
    /// # Arguments
    /// * `batch` - Flattened observations [obs1, obs2, ...]
    ///
    /// # Returns
    /// Flattened normalized observations
    pub fn normalize_batch(&self, batch: &[f32]) -> Vec<f32> {
        let obs_dim = self.stats.dim();
        batch
            .chunks_exact(obs_dim)
            .flat_map(|obs| self.normalize(obs))
            .collect()
    }

    /// Normalize a Burn tensor of observations.
    ///
    /// # Arguments
    /// * `obs` - Tensor of shape [batch_size, obs_dim]
    ///
    /// # Returns
    /// Normalized tensor of same shape
    pub fn normalize_tensor<B: Backend>(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let device = obs.device();
        let [batch_size, obs_dim] = obs.dims();

        // Get mean and std as tensors
        let mean: Vec<f32> = self.stats.mean().iter().map(|&x| x as f32).collect();
        let std: Vec<f32> = self.stats.std_vec().iter().map(|&x| x as f32).collect();

        let mean_tensor = Tensor::<B, 1>::from_floats(mean.as_slice(), &device)
            .reshape([1, obs_dim]);
        let std_tensor = Tensor::<B, 1>::from_floats(std.as_slice(), &device)
            .reshape([1, obs_dim]);

        // Normalize: (obs - mean) / std
        let normalized = (obs - mean_tensor) / std_tensor;

        // Apply clipping if configured
        if let Some((low, high)) = self.config.clip_range {
            normalized.clamp(low, high)
        } else {
            normalized
        }
    }

    /// Apply clipping if configured.
    fn maybe_clip(&self, mut values: Vec<f32>) -> Vec<f32> {
        if let Some((low, high)) = self.config.clip_range {
            for v in &mut values {
                *v = v.clamp(low, high);
            }
        }
        values
    }

    /// Merge statistics from another normalizer.
    pub fn merge(&mut self, other: &ObservationNormalizer) {
        self.stats.merge(&other.stats);
    }

    /// Synchronize with aggregated statistics.
    pub fn sync_with(&mut self, other: &ObservationNormalizer) {
        self.stats.sync_with(&other.stats);
    }

    /// Get the underlying statistics for serialization.
    pub fn stats(&self) -> &RunningMeanStd {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &ObsNormalizationConfig {
        &self.config
    }

    /// Serialize to bytes for cross-thread communication.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.stats.to_bytes()
    }

    /// Create from serialized bytes.
    pub fn from_bytes(bytes: &[u8], config: ObsNormalizationConfig) -> Result<Self, &'static str> {
        let stats = RunningMeanStd::from_bytes(bytes)?;
        Ok(Self { stats, config })
    }

    /// Set whether to update statistics.
    pub fn set_training(&mut self, training: bool) {
        self.config.update_stats = training;
    }

    /// Reset statistics to initial state.
    pub fn reset(&mut self) {
        self.stats.reset();
    }
}

/// Thread-safe wrapper for observation normalizer.
///
/// Suitable for sharing across actor threads in distributed training.
#[derive(Debug, Clone)]
pub struct SharedObservationNormalizer {
    inner: Arc<RwLock<ObservationNormalizer>>,
}

impl SharedObservationNormalizer {
    /// Create a new thread-safe normalizer.
    pub fn new(obs_dim: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ObservationNormalizer::new(obs_dim))),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(obs_dim: usize, config: ObsNormalizationConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ObservationNormalizer::with_config(
                obs_dim, config,
            ))),
        }
    }

    /// Update statistics with an observation.
    pub fn update(&self, obs: &[f32]) {
        self.inner.write().update(obs);
    }

    /// Update statistics with a batch.
    pub fn update_batch(&self, batch: &[f32]) {
        self.inner.write().update_batch(batch);
    }

    /// Normalize an observation.
    pub fn normalize(&self, obs: &[f32]) -> Vec<f32> {
        self.inner.read().normalize(obs)
    }

    /// Update and normalize in one step.
    pub fn update_and_normalize(&self, obs: &[f32]) -> Vec<f32> {
        self.inner.write().update_and_normalize(obs)
    }

    /// Normalize a batch of observations.
    pub fn normalize_batch(&self, batch: &[f32]) -> Vec<f32> {
        self.inner.read().normalize_batch(batch)
    }

    /// Normalize a tensor.
    pub fn normalize_tensor<B: Backend>(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.read().normalize_tensor(obs)
    }

    /// Get a snapshot of the current normalizer.
    pub fn snapshot(&self) -> ObservationNormalizer {
        self.inner.read().clone()
    }

    /// Merge statistics from another source.
    pub fn merge(&self, other: &ObservationNormalizer) {
        self.inner.write().merge(other);
    }

    /// Synchronize with aggregated statistics.
    pub fn sync_with(&self, other: &ObservationNormalizer) {
        self.inner.write().sync_with(other);
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.read().to_bytes()
    }

    /// Get observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.inner.read().obs_dim()
    }

    /// Get observation count.
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
    fn test_observation_normalizer_basic() {
        let mut normalizer = ObservationNormalizer::new(2);

        // Add some observations
        normalizer.update(&[0.0, 10.0]);
        normalizer.update(&[2.0, 10.0]);

        // Mean should be ~[1.0, 10.0]
        let normalized = normalizer.normalize(&[1.0, 10.0]);

        // Should be close to zero (at the mean)
        assert!(normalized[0].abs() < 0.1);
        assert!(normalized[1].abs() < 0.1);
    }

    #[test]
    fn test_observation_normalizer_clipping() {
        let config = ObsNormalizationConfig::default().with_clip_range(Some((-2.0, 2.0)));
        let mut normalizer = ObservationNormalizer::with_config(1, config);

        // Create a distribution with std ~1
        for i in 0..1000 {
            normalizer.update(&[(i % 10) as f32]);
        }

        // A value far from the mean should be clipped
        let normalized = normalizer.normalize(&[100.0]);
        assert!(normalized[0] <= 2.0);
        assert!(normalized[0] >= -2.0);
    }

    #[test]
    fn test_observation_normalizer_batch() {
        let mut normalizer = ObservationNormalizer::new(2);

        // Update with batch
        normalizer.update_batch(&[0.0, 0.0, 2.0, 2.0, 4.0, 4.0]);

        // Normalize batch
        let normalized = normalizer.normalize_batch(&[2.0, 2.0]);

        assert_eq!(normalized.len(), 2);
        // At mean, should be close to zero
        assert!(normalized[0].abs() < 0.1);
        assert!(normalized[1].abs() < 0.1);
    }

    #[test]
    fn test_shared_normalizer() {
        let normalizer = SharedObservationNormalizer::new(2);

        normalizer.update(&[1.0, 2.0]);
        normalizer.update(&[3.0, 4.0]);

        let normalized = normalizer.normalize(&[2.0, 3.0]);
        assert_eq!(normalized.len(), 2);
    }

    #[test]
    fn test_normalizer_merge() {
        let mut norm1 = ObservationNormalizer::new(1);
        let mut norm2 = ObservationNormalizer::new(1);

        for &x in &[1.0, 2.0, 3.0] {
            norm1.update(&[x]);
        }
        for &x in &[4.0, 5.0, 6.0] {
            norm2.update(&[x]);
        }

        norm1.merge(&norm2);

        // Combined count should be 6
        assert!((norm1.count() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_mode() {
        let mut normalizer = ObservationNormalizer::new(1);

        // In training mode, stats are updated
        normalizer.update(&[1.0]);
        assert!((normalizer.count() - 1.0).abs() < 1e-10);

        // Disable training mode
        normalizer.set_training(false);
        normalizer.update(&[2.0]);

        // Count should not have changed
        assert!((normalizer.count() - 1.0).abs() < 1e-10);
    }
}
