//! Validated PSD (Power Spectral Density) Buffer
//!
//! Type-safe PSD buffer that enforces non-negative power at compile time.
//! This replaces the legacy `PsdBuffer` with Curry-Howard compliant types.

use crate::config::RFWorldConfig;
use crate::types::{
    frequency::ValidatedFrequencyGrid,
    primitives::PositivePower,
    power::PowerDbm,
};


// ============================================================================
// ValidatedPsd: Type-Safe PSD Buffer
// ============================================================================

/// Power Spectral Density buffer with compile-time safety guarantees.
///
/// # Invariants
/// - All power values are non-negative (enforced by `PositivePower` type)
/// - Frequency grid is valid (min < max, num_bins > 0)
/// - Total power is always non-negative
///
/// # Physical Meaning
/// Stores power spectral density in linear scale (Watts per bin).
/// Each bin represents the power within that frequency resolution.
#[derive(Clone)]
pub struct ValidatedPsd {
    /// PSD data - guaranteed non-negative by type
    data: Vec<PositivePower>,
    /// Number of environments
    num_envs: usize,
    /// Number of frequency bins per environment
    num_bins: usize,
    /// Validated frequency grid
    grid: ValidatedFrequencyGrid,
}

impl ValidatedPsd {
    /// Create a new validated PSD buffer.
    ///
    /// # Arguments
    /// * `config` - RF world configuration
    ///
    /// # Panics
    /// Panics if frequency range is invalid (min >= max) or num_bins is 0.
    pub fn new(config: &RFWorldConfig) -> Self {
        let num_envs = config.num_envs;
        let num_bins = config.num_freq_bins;

        // Create validated grid - will panic if invalid
        let grid = ValidatedFrequencyGrid::from_params(
            config.freq_min,
            config.freq_max,
            num_bins,
        )
        .expect("ValidatedPsd requires valid frequency grid");

        let data = vec![PositivePower::ZERO; num_envs * num_bins];

        Self {
            data,
            num_envs,
            num_bins,
            grid,
        }
    }

    /// Try to create a new validated PSD buffer.
    ///
    /// Returns `None` if configuration is invalid.
    pub fn try_new(config: &RFWorldConfig) -> Option<Self> {
        let num_envs = config.num_envs;
        let num_bins = config.num_freq_bins;

        let grid = ValidatedFrequencyGrid::from_params(
            config.freq_min,
            config.freq_max,
            num_bins,
        )?;

        let data = vec![PositivePower::ZERO; num_envs * num_bins];

        Some(Self {
            data,
            num_envs,
            num_bins,
            grid,
        })
    }

    /// Create from explicit parameters.
    pub fn from_params(
        num_envs: usize,
        freq_min: f32,
        freq_max: f32,
        num_bins: usize,
    ) -> Option<Self> {
        let grid = ValidatedFrequencyGrid::from_params(freq_min, freq_max, num_bins)?;
        let data = vec![PositivePower::ZERO; num_envs * num_bins];

        Some(Self {
            data,
            num_envs,
            num_bins,
            grid,
        })
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get reference to the validated frequency grid.
    #[inline]
    pub fn grid(&self) -> &ValidatedFrequencyGrid {
        &self.grid
    }

    /// Get number of environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get number of bins per environment.
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Get the offset into data for a given environment.
    #[inline]
    pub fn env_offset(&self, env: usize) -> usize {
        env * self.num_bins
    }

    // ========================================================================
    // Type-Safe Power Operations
    // ========================================================================

    /// Add power to a specific bin.
    ///
    /// This is type-safe: the power is guaranteed non-negative by `PositivePower`.
    #[inline]
    pub fn add_power(&mut self, env: usize, bin: usize, power: PositivePower) {
        let idx = self.env_offset(env) + bin;
        if idx < self.data.len() {
            self.data[idx] = self.data[idx].add(power);
        }
    }

    /// Get power at a specific bin.
    #[inline]
    pub fn get_power(&self, env: usize, bin: usize) -> PositivePower {
        let idx = self.env_offset(env) + bin;
        self.data.get(idx).copied().unwrap_or(PositivePower::ZERO)
    }

    /// Set power at a specific bin.
    #[inline]
    pub fn set_power(&mut self, env: usize, bin: usize, power: PositivePower) {
        let idx = self.env_offset(env) + bin;
        if idx < self.data.len() {
            self.data[idx] = power;
        }
    }

    /// Get total power for an environment (guaranteed non-negative).
    pub fn total_power(&self, env: usize) -> PositivePower {
        let offset = self.env_offset(env);
        self.data[offset..offset + self.num_bins]
            .iter()
            .copied()
            .sum()
    }

    /// Find peak power and its bin for an environment.
    pub fn peak_power(&self, env: usize) -> (PositivePower, usize) {
        let offset = self.env_offset(env);
        let slice = &self.data[offset..offset + self.num_bins];

        let (max_bin, &max_power) = slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.as_watts().partial_cmp(&b.as_watts()).unwrap())
            .unwrap_or((0, &PositivePower::ZERO));

        (max_power, max_bin)
    }

    // ========================================================================
    // Bulk Operations
    // ========================================================================

    /// Clear all PSD data to zero.
    pub fn clear(&mut self) {
        self.data.fill(PositivePower::ZERO);
    }

    /// Clear PSD for a single environment.
    pub fn clear_env(&mut self, env: usize) {
        let offset = self.env_offset(env);
        self.data[offset..offset + self.num_bins].fill(PositivePower::ZERO);
    }

    /// Set PSD to a constant noise floor value.
    pub fn set_noise_floor(&mut self, noise_floor: PositivePower) {
        self.data.fill(noise_floor);
    }

    // ========================================================================
    // Conversions
    // ========================================================================

    /// Convert to dBm representation.
    ///
    /// Returns a new vector containing power values in dBm.
    /// Very low power values are floored at a minimum dBm threshold.
    pub fn to_dbm(&self) -> Vec<PowerDbm> {
        self.data
            .iter()
            .map(|&p| PowerDbm::from_watts(p))
            .collect()
    }

    /// Get power values as raw f32 slice for an environment.
    ///
    /// This is a one-way conversion for interop with legacy code.
    pub fn env_slice_watts(&self, env: usize) -> Vec<f32> {
        let offset = self.env_offset(env);
        self.data[offset..offset + self.num_bins]
            .iter()
            .map(|p| p.watts())
            .collect()
    }

    // ========================================================================
    // Legacy Interop (Raw f32 Access)
    // ========================================================================

    /// Add power from raw f32 value.
    ///
    /// Returns false if the power value is invalid (negative or non-finite).
    #[inline]
    pub fn add_power_raw(&mut self, env: usize, bin: usize, power: f32) -> bool {
        if let Some(p) = PositivePower::try_new(power) {
            self.add_power(env, bin, p);
            true
        } else {
            false
        }
    }

    /// Get power at a bin as raw f32 (for legacy interop).
    #[inline]
    pub fn get_power_raw(&self, env: usize, bin: usize) -> f32 {
        self.get_power(env, bin).watts()
    }

    /// Get mutable access to raw data (for SIMD operations).
    ///
    /// # Safety
    /// The caller must ensure that only non-negative values are written.
    /// This breaks the type safety guarantees but is necessary for SIMD.
    #[inline]
    pub unsafe fn data_raw_mut(&mut self) -> &mut [f32] {
        // PositivePower is repr(transparent) over f32
        std::slice::from_raw_parts_mut(
            self.data.as_mut_ptr() as *mut f32,
            self.data.len(),
        )
    }

    /// Get reference to raw data as f32 slice.
    #[inline]
    pub fn data_raw(&self) -> &[f32] {
        // PositivePower is repr(transparent) over f32
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.data.len())
        }
    }

    /// Get mutable slice for a single environment's PSD as raw f32.
    ///
    /// # Safety
    /// The caller must ensure only non-negative values are written.
    #[inline]
    pub unsafe fn env_slice_mut_raw(&mut self, env: usize) -> &mut [f32] {
        let offset = self.env_offset(env);
        let num_bins = self.num_bins;
        let raw = self.data_raw_mut();
        &mut raw[offset..offset + num_bins]
    }

    /// Get slice for a single environment's PSD as raw f32.
    #[inline]
    pub fn env_slice_raw(&self, env: usize) -> &[f32] {
        let offset = self.env_offset(env);
        &self.data_raw()[offset..offset + self.num_bins]
    }

    // ========================================================================
    // SIMD Operations
    // ========================================================================

    /// SIMD clear for batch of 8 environments.
    #[cfg(feature = "simd")]
    pub fn clear_batch(&mut self, batch: usize) {
        let base_env = batch * 8;
        for env_offset in 0..8 {
            let env = base_env + env_offset;
            if env < self.num_envs {
                self.clear_env(env);
            }
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RFWorldConfig {
        RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(512)
            .with_freq_range(1e9, 2e9)
            .build()
    }

    #[test]
    fn test_validated_psd_creation() {
        let config = test_config();
        let buffer = ValidatedPsd::new(&config);

        assert_eq!(buffer.num_envs(), 8);
        assert_eq!(buffer.num_bins(), 512);
    }

    #[test]
    fn test_validated_psd_try_new() {
        let config = test_config();
        assert!(ValidatedPsd::try_new(&config).is_some());
    }

    #[test]
    fn test_validated_psd_clear() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        // Add some power
        buffer.add_power(0, 100, PositivePower::new(1.0));
        buffer.add_power(3, 200, PositivePower::new(2.0));

        // Clear
        buffer.clear();

        // Verify all zeros
        assert!(buffer.get_power(0, 100).watts() == 0.0);
        assert!(buffer.get_power(3, 200).watts() == 0.0);
    }

    #[test]
    fn test_validated_psd_add_get_power() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        buffer.add_power(2, 100, PositivePower::new(0.5));
        buffer.add_power(2, 100, PositivePower::new(0.3));

        let power = buffer.get_power(2, 100);
        assert!((power.watts() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_validated_psd_total_power() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        // Set uniform power
        let power_per_bin = PositivePower::new(1e-6);
        for bin in 0..buffer.num_bins() {
            buffer.add_power(0, bin, power_per_bin);
        }

        let total = buffer.total_power(0);
        let expected = 1e-6 * buffer.num_bins() as f32;
        assert!((total.watts() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_validated_psd_peak_power() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        buffer.add_power(0, 100, PositivePower::new(0.5));
        buffer.add_power(0, 200, PositivePower::new(2.0)); // Peak
        buffer.add_power(0, 300, PositivePower::new(1.0));

        let (peak, bin) = buffer.peak_power(0);
        assert_eq!(bin, 200);
        assert!((peak.watts() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_validated_psd_to_dbm() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        // 1 mW = 0 dBm
        buffer.add_power(0, 100, PositivePower::new(0.001));

        let dbm_values = buffer.to_dbm();
        let dbm = dbm_values[buffer.env_offset(0) + 100].as_dbm();

        assert!((dbm - 0.0).abs() < 0.1, "1 mW should be 0 dBm, got {}", dbm);
    }

    #[test]
    fn test_validated_psd_raw_interop() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        // Valid power
        assert!(buffer.add_power_raw(0, 100, 0.5));
        assert!((buffer.get_power_raw(0, 100) - 0.5).abs() < 1e-6);

        // Invalid (negative) power rejected
        assert!(!buffer.add_power_raw(0, 100, -0.5));
    }

    #[test]
    fn test_validated_psd_env_isolation() {
        let config = test_config();
        let mut buffer = ValidatedPsd::new(&config);

        // Modify env 1
        buffer.add_power(1, 50, PositivePower::new(1.0));

        // Verify only env 1 bin 50 is modified
        assert!(buffer.get_power(0, 50).watts() == 0.0);
        assert!((buffer.get_power(1, 50).watts() - 1.0).abs() < 1e-6);
        assert!(buffer.get_power(2, 50).watts() == 0.0);
    }

    #[test]
    fn test_validated_psd_grid_access() {
        let config = test_config();
        let buffer = ValidatedPsd::new(&config);

        let grid = buffer.grid();
        assert!((grid.freq_min().as_hz() - 1e9).abs() < 1.0);
        assert!((grid.freq_max().as_hz() - 2e9).abs() < 1.0);
    }
}
