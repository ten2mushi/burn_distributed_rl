//! Running statistics using Welford's online algorithm.
//!
//! Provides numerically stable computation of running mean and variance
//! for observation and reward normalization in RL training.
//!
//! # Features
//! - Online updates with Welford's algorithm (numerically stable)
//! - Thread-safe with interior mutability via `parking_lot::RwLock`
//! - Support for distributed aggregation via `merge()` and `sync_with()`
//! - Per-dimension statistics for multi-dimensional observations
//!
//! # Example
//! ```ignore
//! use distributed_rl::core::RunningMeanStd;
//!
//! let mut stats = RunningMeanStd::new(4); // 4-dim observations
//! stats.update(&[1.0, 2.0, 3.0, 4.0]);
//! stats.update(&[2.0, 3.0, 4.0, 5.0]);
//!
//! let normalized = stats.normalize(&[1.5, 2.5, 3.5, 4.5]);
//! ```

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;

/// Running mean and standard deviation using Welford's online algorithm.
///
/// Maintains per-dimension statistics for normalizing multi-dimensional data.
/// Numerically stable even for large sample counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningMeanStd {
    /// Running mean per dimension
    mean: Vec<f64>,
    /// Running variance (sum of squared deviations) per dimension
    /// Note: actual variance = var_sum / count
    var_sum: Vec<f64>,
    /// Number of samples seen
    count: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
}

impl RunningMeanStd {
    /// Create a new RunningMeanStd for the given dimensionality.
    ///
    /// # Arguments
    /// * `dim` - Number of dimensions to track
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            var_sum: vec![0.0; dim],
            count: 0.0,
            epsilon: 1e-8,
        }
    }

    /// Create with a custom epsilon for numerical stability.
    pub fn with_epsilon(dim: usize, epsilon: f64) -> Self {
        Self {
            mean: vec![0.0; dim],
            var_sum: vec![0.0; dim],
            count: 0.0,
            epsilon,
        }
    }

    /// Update statistics with a single observation using Welford's algorithm.
    ///
    /// # Arguments
    /// * `obs` - Observation vector (must match dimensionality)
    ///
    /// # Panics
    /// Panics if observation dimensionality doesn't match.
    pub fn update(&mut self, obs: &[f32]) {
        assert_eq!(obs.len(), self.mean.len(), "Observation dimension mismatch");

        self.count += 1.0;
        for i in 0..obs.len() {
            let x = obs[i] as f64;
            let delta = x - self.mean[i];
            self.mean[i] += delta / self.count;
            let delta2 = x - self.mean[i];
            self.var_sum[i] += delta * delta2;
        }
    }

    /// Update statistics with a batch of observations.
    ///
    /// # Arguments
    /// * `batch` - Flattened batch of observations [obs1, obs2, ...] where each obs has `dim` elements
    pub fn update_batch(&mut self, batch: &[f32]) {
        let dim = self.mean.len();
        assert_eq!(batch.len() % dim, 0, "Batch size must be multiple of dimension");

        for obs in batch.chunks_exact(dim) {
            self.update(obs);
        }
    }

    /// Normalize an observation to zero mean and unit variance.
    ///
    /// # Arguments
    /// * `obs` - Observation vector to normalize
    ///
    /// # Returns
    /// Normalized observation vector
    pub fn normalize(&self, obs: &[f32]) -> Vec<f32> {
        assert_eq!(obs.len(), self.mean.len(), "Observation dimension mismatch");

        obs.iter()
            .enumerate()
            .map(|(i, &x)| {
                let std = self.std(i);
                ((x as f64 - self.mean[i]) / std) as f32
            })
            .collect()
    }

    /// Normalize an observation in-place.
    pub fn normalize_inplace(&self, obs: &mut [f32]) {
        assert_eq!(obs.len(), self.mean.len(), "Observation dimension mismatch");

        for (i, x) in obs.iter_mut().enumerate() {
            let std = self.std(i);
            *x = ((*x as f64 - self.mean[i]) / std) as f32;
        }
    }

    /// Normalize and clip to a range.
    ///
    /// # Arguments
    /// * `obs` - Observation vector to normalize
    /// * `clip_range` - Tuple of (min, max) to clip normalized values
    pub fn normalize_and_clip(&self, obs: &[f32], clip_range: (f32, f32)) -> Vec<f32> {
        self.normalize(obs)
            .into_iter()
            .map(|x| x.clamp(clip_range.0, clip_range.1))
            .collect()
    }

    /// Get the standard deviation for dimension i.
    #[inline]
    fn std(&self, i: usize) -> f64 {
        if self.count < 2.0 {
            1.0 // Avoid division issues with small sample counts
        } else {
            (self.var_sum[i] / self.count).sqrt().max(self.epsilon)
        }
    }

    /// Get the mean vector.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Get the variance vector (population variance).
    pub fn variance(&self) -> Vec<f64> {
        if self.count < 2.0 {
            vec![1.0; self.mean.len()]
        } else {
            self.var_sum.iter().map(|&v| v / self.count).collect()
        }
    }

    /// Get the standard deviation vector.
    pub fn std_vec(&self) -> Vec<f64> {
        self.variance().into_iter().map(|v| v.sqrt().max(self.epsilon)).collect()
    }

    /// Get the sample count.
    pub fn count(&self) -> f64 {
        self.count
    }

    /// Get the dimensionality.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }

    /// Merge statistics from another RunningMeanStd using parallel Welford's algorithm.
    ///
    /// This allows combining statistics from multiple distributed actors.
    ///
    /// # Arguments
    /// * `other` - Another RunningMeanStd to merge from
    ///
    /// # Panics
    /// Panics if dimensionalities don't match.
    pub fn merge(&mut self, other: &RunningMeanStd) {
        assert_eq!(self.mean.len(), other.mean.len(), "Dimension mismatch in merge");

        if other.count == 0.0 {
            return;
        }
        if self.count == 0.0 {
            self.mean.copy_from_slice(&other.mean);
            self.var_sum.copy_from_slice(&other.var_sum);
            self.count = other.count;
            return;
        }

        let total_count = self.count + other.count;

        for i in 0..self.mean.len() {
            let delta = other.mean[i] - self.mean[i];

            // Combined mean
            let new_mean = self.mean[i] + delta * other.count / total_count;

            // Combined variance (parallel Welford's algorithm)
            // M2_combined = M2_a + M2_b + delta^2 * n_a * n_b / (n_a + n_b)
            let new_var_sum = self.var_sum[i]
                + other.var_sum[i]
                + delta * delta * self.count * other.count / total_count;

            self.mean[i] = new_mean;
            self.var_sum[i] = new_var_sum;
        }

        self.count = total_count;
    }

    /// Create a merged copy from multiple RunningMeanStd instances.
    ///
    /// # Arguments
    /// * `stats` - Slice of RunningMeanStd instances to merge
    ///
    /// # Returns
    /// A new RunningMeanStd with combined statistics
    pub fn merge_all(stats: &[RunningMeanStd]) -> Option<Self> {
        if stats.is_empty() {
            return None;
        }

        let mut result = stats[0].clone();
        for s in &stats[1..] {
            result.merge(s);
        }
        Some(result)
    }

    /// Synchronize with another RunningMeanStd by replacing our statistics.
    ///
    /// Used when receiving aggregated statistics from a central coordinator.
    pub fn sync_with(&mut self, other: &RunningMeanStd) {
        assert_eq!(self.mean.len(), other.mean.len(), "Dimension mismatch in sync");
        self.mean.copy_from_slice(&other.mean);
        self.var_sum.copy_from_slice(&other.var_sum);
        self.count = other.count;
    }

    /// Reset statistics to initial state.
    pub fn reset(&mut self) {
        self.mean.fill(0.0);
        self.var_sum.fill(0.0);
        self.count = 0.0;
    }

    /// Serialize to bytes for cross-thread/process communication.
    ///
    /// Uses a simple binary format: [dim, count, mean..., var_sum..., epsilon]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + 8 + self.mean.len() * 16 + 8);
        bytes.extend_from_slice(&(self.mean.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.count.to_le_bytes());
        for &m in &self.mean {
            bytes.extend_from_slice(&m.to_le_bytes());
        }
        for &v in &self.var_sum {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&self.epsilon.to_le_bytes());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 16 {
            return Err("Buffer too small");
        }

        let dim = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let count = f64::from_le_bytes(bytes[8..16].try_into().unwrap());

        let expected_len = 16 + dim * 16 + 8;
        if bytes.len() < expected_len {
            return Err("Buffer too small for specified dimension");
        }

        let mut mean = Vec::with_capacity(dim);
        let mut var_sum = Vec::with_capacity(dim);
        let mut offset = 16;

        for _ in 0..dim {
            mean.push(f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()));
            offset += 8;
        }
        for _ in 0..dim {
            var_sum.push(f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()));
            offset += 8;
        }

        let epsilon = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());

        Ok(Self {
            mean,
            var_sum,
            count,
            epsilon,
        })
    }
}

/// Thread-safe wrapper for RunningMeanStd.
///
/// Uses `parking_lot::RwLock` for efficient concurrent access.
#[derive(Debug, Clone)]
pub struct SharedRunningMeanStd {
    inner: Arc<RwLock<RunningMeanStd>>,
}

impl SharedRunningMeanStd {
    /// Create a new thread-safe RunningMeanStd.
    pub fn new(dim: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(RunningMeanStd::new(dim))),
        }
    }

    /// Create from an existing RunningMeanStd.
    pub fn from_stats(stats: RunningMeanStd) -> Self {
        Self {
            inner: Arc::new(RwLock::new(stats)),
        }
    }

    /// Update statistics with an observation.
    pub fn update(&self, obs: &[f32]) {
        self.inner.write().update(obs);
    }

    /// Update statistics with a batch of observations.
    pub fn update_batch(&self, batch: &[f32]) {
        self.inner.write().update_batch(batch);
    }

    /// Normalize an observation.
    pub fn normalize(&self, obs: &[f32]) -> Vec<f32> {
        self.inner.read().normalize(obs)
    }

    /// Normalize and clip an observation.
    pub fn normalize_and_clip(&self, obs: &[f32], clip_range: (f32, f32)) -> Vec<f32> {
        self.inner.read().normalize_and_clip(obs, clip_range)
    }

    /// Get a snapshot of the current statistics.
    pub fn snapshot(&self) -> RunningMeanStd {
        self.inner.read().clone()
    }

    /// Merge statistics from another source.
    pub fn merge(&self, other: &RunningMeanStd) {
        self.inner.write().merge(other);
    }

    /// Synchronize with aggregated statistics.
    pub fn sync_with(&self, other: &RunningMeanStd) {
        self.inner.write().sync_with(other);
    }

    /// Reset statistics.
    pub fn reset(&self) {
        self.inner.write().reset();
    }

    /// Get the sample count.
    pub fn count(&self) -> f64 {
        self.inner.read().count()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.read().to_bytes()
    }

    /// Create from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        let stats = RunningMeanStd::from_bytes(bytes)?;
        Ok(Self::from_stats(stats))
    }
}

/// Running statistics for scalar values (e.g., returns for reward normalization).
///
/// Simpler version of RunningMeanStd for single-dimensional data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningScalarStats {
    mean: f64,
    var_sum: f64,
    count: f64,
    epsilon: f64,
}

impl RunningScalarStats {
    /// Create new scalar statistics tracker.
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var_sum: 0.0,
            count: 0.0,
            epsilon: 1e-8,
        }
    }

    /// Update with a single value.
    pub fn update(&mut self, x: f32) {
        self.count += 1.0;
        let x = x as f64;
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.var_sum += delta * delta2;
    }

    /// Update with multiple values.
    pub fn update_batch(&mut self, values: &[f32]) {
        for &x in values {
            self.update(x);
        }
    }

    /// Normalize a value.
    pub fn normalize(&self, x: f32) -> f32 {
        let std = self.std();
        ((x as f64 - self.mean) / std) as f32
    }

    /// Get the standard deviation.
    pub fn std(&self) -> f64 {
        if self.count < 2.0 {
            1.0
        } else {
            (self.var_sum / self.count).sqrt().max(self.epsilon)
        }
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get the sample count.
    pub fn count(&self) -> f64 {
        self.count
    }

    /// Merge with another instance.
    pub fn merge(&mut self, other: &RunningScalarStats) {
        if other.count == 0.0 {
            return;
        }
        if self.count == 0.0 {
            *self = other.clone();
            return;
        }

        let total_count = self.count + other.count;
        let delta = other.mean - self.mean;

        self.mean += delta * other.count / total_count;
        self.var_sum += other.var_sum + delta * delta * self.count * other.count / total_count;
        self.count = total_count;
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.var_sum = 0.0;
        self.count = 0.0;
    }
}

impl Default for RunningScalarStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_mean() {
        let mut stats = RunningMeanStd::new(2);
        stats.update(&[1.0, 2.0]);
        stats.update(&[3.0, 4.0]);
        stats.update(&[5.0, 6.0]);

        let mean = stats.mean();
        assert!((mean[0] - 3.0).abs() < 1e-10);
        assert!((mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_welford_variance() {
        let mut stats = RunningMeanStd::new(1);
        // Values: 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Variance = 4
        for &x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(&[x]);
        }

        let var = stats.variance();
        assert!((var[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let mut stats = RunningMeanStd::new(2);
        // Create stats with known mean and std
        for _ in 0..1000 {
            stats.update(&[0.0, 10.0]);
            stats.update(&[2.0, 10.0]);
        }

        // Mean should be ~1.0 for dim 0, ~10.0 for dim 1
        let normalized = stats.normalize(&[1.0, 10.0]);
        assert!(normalized[0].abs() < 0.1); // Should be close to 0
        assert!(normalized[1].abs() < 0.1); // Should be close to 0
    }

    #[test]
    fn test_merge() {
        let mut stats1 = RunningMeanStd::new(1);
        let mut stats2 = RunningMeanStd::new(1);

        // Split data between two stats
        for &x in &[1.0, 2.0, 3.0] {
            stats1.update(&[x]);
        }
        for &x in &[4.0, 5.0, 6.0] {
            stats2.update(&[x]);
        }

        stats1.merge(&stats2);

        // Combined mean should be 3.5
        assert!((stats1.mean()[0] - 3.5).abs() < 1e-10);
        assert!((stats1.count() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_serialization() {
        let mut stats = RunningMeanStd::new(3);
        stats.update(&[1.0, 2.0, 3.0]);
        stats.update(&[4.0, 5.0, 6.0]);

        let bytes = stats.to_bytes();
        let restored = RunningMeanStd::from_bytes(&bytes).unwrap();

        assert_eq!(stats.mean(), restored.mean());
        assert_eq!(stats.count(), restored.count());
    }

    #[test]
    fn test_scalar_stats() {
        let mut stats = RunningScalarStats::new();
        for &x in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.update(x);
        }

        assert!((stats.mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_shared_stats() {
        let stats = SharedRunningMeanStd::new(2);

        // Simulate concurrent updates
        stats.update(&[1.0, 2.0]);
        stats.update(&[3.0, 4.0]);

        let normalized = stats.normalize(&[2.0, 3.0]);
        assert_eq!(normalized.len(), 2);
    }
}
