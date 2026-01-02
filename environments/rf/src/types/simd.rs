//! SIMD Refinement Types
//!
//! SIMD-vectorized versions of the refinement types for high-performance
//! RF simulation. All types are `#[repr(transparent)]` for zero-cost abstraction.

use std::simd::{cmp::SimdPartialOrd, f32x8, num::SimdFloat};

use super::primitives::PositivePower;
use super::fading::UnitMeanFading;
use crate::constants::LN_10;
use crate::simd_rf::math::{simd_db_to_linear, simd_sqrt};

// ============================================================================
// PositiveF32x8: 8-wide Non-Negative Values
// ============================================================================

/// 8-wide SIMD vector of non-negative f32 values.
///
/// This is the SIMD equivalent of [`PositiveF32`](super::primitives::PositiveF32).
/// Validation is performed in debug builds only for performance.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PositiveF32x8(f32x8);

impl PositiveF32x8 {
    /// Zero vector.
    pub const ZERO: Self = Self(f32x8::from_array([0.0; 8]));

    /// One vector.
    pub const ONE: Self = Self(f32x8::from_array([1.0; 8]));

    /// Create from raw SIMD vector with validation.
    #[inline]
    pub fn new(values: f32x8) -> Self {
        #[cfg(debug_assertions)]
        {
            let arr: [f32; 8] = values.into();
            for (i, v) in arr.iter().enumerate() {
                assert!(
                    *v >= 0.0 && v.is_finite(),
                    "Lane {} contains invalid value: {}",
                    i,
                    v
                );
            }
        }
        Self(values)
    }

    /// Create from raw SIMD vector without validation.
    ///
    /// # Safety
    /// Caller must ensure all lanes are non-negative and finite.
    #[inline]
    pub unsafe fn new_unchecked(values: f32x8) -> Self {
        Self(values)
    }

    /// Create from a scalar (splat).
    #[inline]
    pub fn splat(value: f32) -> Self {
        Self::new(f32x8::splat(value))
    }

    /// Get the inner SIMD vector.
    #[inline]
    pub fn get(self) -> f32x8 {
        self.0
    }

    /// Add two vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Multiply two vectors element-wise.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Scale by a scalar vector.
    #[inline]
    pub fn scale(self, factor: f32x8) -> Self {
        // In release builds, we trust the factor is valid
        Self(self.0 * factor)
    }

    /// Element-wise square root.
    #[inline]
    pub fn sqrt(self) -> Self {
        Self(simd_sqrt(self.0))
    }

    /// Element-wise clamp.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.simd_clamp(min.0, max.0))
    }

    /// Reduce to sum.
    #[inline]
    pub fn reduce_sum(self) -> f32 {
        self.0.reduce_sum()
    }

    /// Reduce to max.
    #[inline]
    pub fn reduce_max(self) -> f32 {
        self.0.reduce_max()
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 8] {
        self.0.into()
    }
}

impl Default for PositiveF32x8 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::ops::Add for PositiveF32x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Mul for PositiveF32x8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

// ============================================================================
// PositivePower8: 8-wide Power Values
// ============================================================================

/// 8-wide SIMD vector of positive power values.
///
/// This is the SIMD equivalent of [`PositivePower`].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct PositivePower8(f32x8);

impl PositivePower8 {
    /// Zero power vector.
    pub const ZERO: Self = Self(f32x8::from_array([0.0; 8]));

    /// Create from raw SIMD vector with validation.
    #[inline]
    pub fn new(values: f32x8) -> Self {
        #[cfg(debug_assertions)]
        {
            let arr: [f32; 8] = values.into();
            for (i, v) in arr.iter().enumerate() {
                assert!(
                    *v >= 0.0 && v.is_finite(),
                    "Lane {} contains invalid power: {}",
                    i,
                    v
                );
            }
        }
        Self(values)
    }

    /// Create from scalar (splat).
    #[inline]
    pub fn splat(watts: f32) -> Self {
        Self::new(f32x8::splat(watts))
    }

    /// Create from array of PositivePower values.
    #[inline]
    pub fn from_array(powers: [PositivePower; 8]) -> Self {
        let arr: [f32; 8] = powers.map(|p| p.watts());
        Self(f32x8::from_array(arr))
    }

    /// Fallible constructor that returns Option.
    #[inline]
    pub fn try_new(watts: f32) -> Option<Self> {
        (watts >= 0.0 && watts.is_finite()).then(|| Self(f32x8::splat(watts)))
    }

    /// Get the inner SIMD vector (in Watts).
    #[inline]
    pub fn get(self) -> f32x8 {
        self.0
    }

    /// Get the raw f32x8 value (alias for get).
    #[inline]
    pub fn as_raw(self) -> f32x8 {
        self.0
    }

    /// Add two power vectors.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Scale by a factor vector.
    #[inline]
    pub fn scale(self, factor: PositiveF32x8) -> Self {
        Self(self.0 * factor.get())
    }

    /// Scale by raw SIMD factor.
    #[inline]
    pub fn scale_raw(self, factor: f32x8) -> Self {
        Self(self.0 * factor)
    }

    /// Apply fading coefficients.
    #[inline]
    pub fn apply_fading(self, fading: UnitMeanFading8) -> Self {
        Self(self.0 * fading.get())
    }

    /// Reduce to sum.
    #[inline]
    pub fn reduce_sum(self) -> PositivePower {
        PositivePower::new(self.0.reduce_sum())
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 8] {
        self.0.into()
    }

    /// Convert to array of PositivePower.
    #[inline]
    pub fn to_positive_power_array(self) -> [PositivePower; 8] {
        let arr: [f32; 8] = self.0.into();
        arr.map(PositivePower::new)
    }
}

impl Default for PositivePower8 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::ops::Add for PositivePower8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

// ============================================================================
// UnitMeanFading8: 8-wide Fading Coefficients
// ============================================================================

/// 8-wide SIMD vector of unit-mean fading coefficients.
///
/// This is the SIMD equivalent of [`UnitMeanFading`].
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct UnitMeanFading8(f32x8);

impl UnitMeanFading8 {
    /// Unity (no fading) vector.
    pub const ONE: Self = Self(f32x8::from_array([1.0; 8]));

    /// Create from raw SIMD vector.
    ///
    /// # Safety
    /// Caller must ensure values come from a unit-mean distribution.
    #[inline]
    pub(crate) fn from_raw_unchecked(values: f32x8) -> Self {
        #[cfg(debug_assertions)]
        {
            let arr: [f32; 8] = values.into();
            for (i, v) in arr.iter().enumerate() {
                assert!(
                    *v >= 0.0,
                    "Lane {} contains negative fading coefficient: {}",
                    i,
                    v
                );
            }
        }
        Self(values)
    }

    /// Get the inner SIMD vector.
    #[inline]
    pub fn get(self) -> f32x8 {
        self.0
    }

    /// Get the raw f32x8 value (alias for get).
    #[inline]
    pub fn as_raw(self) -> f32x8 {
        self.0
    }

    /// Apply fading to power.
    #[inline]
    pub fn apply(self, power: PositivePower8) -> PositivePower8 {
        PositivePower8(power.get() * self.0)
    }

    /// Apply to raw SIMD power vector.
    #[inline]
    pub fn apply_raw(self, power: f32x8) -> f32x8 {
        power * self.0
    }

    /// Combine two independent fading coefficients.
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Convert to array.
    #[inline]
    pub fn to_array(self) -> [f32; 8] {
        self.0.into()
    }

    /// Convert to array of UnitMeanFading.
    #[inline]
    pub fn to_fading_array(self) -> [UnitMeanFading; 8] {
        let arr: [f32; 8] = self.0.into();
        arr.map(UnitMeanFading::from_raw_unchecked)
    }
}

impl Default for UnitMeanFading8 {
    fn default() -> Self {
        Self::ONE
    }
}

// ============================================================================
// SIMD Fading Generators
// ============================================================================

/// SIMD Rayleigh fading with unit mean.
pub struct RayleighFading8;

impl RayleighFading8 {
    /// Sample 8 Rayleigh fading coefficients with unit mean.
    ///
    /// # Arguments
    /// * `i` - 8 in-phase components (standard normal)
    /// * `q` - 8 quadrature components (standard normal)
    #[inline]
    pub fn sample(i: f32x8, q: f32x8) -> UnitMeanFading8 {
        // Normalize by 0.5 to achieve unit mean
        let half = f32x8::splat(0.5);
        let coefficient = half * (i * i + q * q);
        UnitMeanFading8::from_raw_unchecked(coefficient)
    }
}

/// SIMD Rician fading with unit mean.
pub struct RicianFading8;

impl RicianFading8 {
    /// Sample 8 Rician fading coefficients with unit mean.
    ///
    /// # Arguments
    /// * `k_factor` - 8 K-factors (can vary per lane)
    /// * `i` - 8 in-phase scatter components (standard normal)
    /// * `q` - 8 quadrature scatter components (standard normal)
    #[inline]
    pub fn sample(k_factor: f32x8, i: f32x8, q: f32x8) -> UnitMeanFading8 {
        let half = f32x8::splat(0.5);
        let one = f32x8::splat(1.0);
        let k_plus_1 = k_factor + one;

        // LOS and scatter scales for unit mean
        let los_scale = simd_sqrt(k_factor / k_plus_1);
        let scatter_scale = simd_sqrt(half / k_plus_1);

        // Combine LOS (fixed phase) with scatter
        let i_total = los_scale + scatter_scale * i;
        let q_total = scatter_scale * q;

        // Power coefficient
        let coefficient = i_total * i_total + q_total * q_total;
        UnitMeanFading8::from_raw_unchecked(coefficient)
    }
}

/// SIMD corrected shadowing with unit mean.
#[derive(Clone, Copy, Debug)]
pub struct UnitMeanShadowing8 {
    sigma_db: f32x8,
    correction_db: f32x8,
}

impl UnitMeanShadowing8 {
    /// Create from per-lane sigma values.
    #[inline]
    pub fn new(sigma_db: f32x8) -> Self {
        let ln_10 = f32x8::splat(LN_10);
        let factor = f32x8::splat(20.0);

        // Correction: -(σ² × ln(10)) / 20
        let correction_db = -(sigma_db * sigma_db * ln_10) / factor;

        Self {
            sigma_db,
            correction_db,
        }
    }

    /// Create with same sigma for all lanes (returns Option for consistency).
    #[inline]
    pub fn splat(sigma_db: f32) -> Option<Self> {
        (sigma_db >= 0.0 && sigma_db.is_finite()).then(|| Self::new(f32x8::splat(sigma_db)))
    }

    /// Create from an array of per-lane sigma values.
    #[inline]
    pub fn from_sigma_db_array(sigma_db: [f32; 8]) -> Option<Self> {
        sigma_db
            .iter()
            .all(|&s| s >= 0.0 && s.is_finite())
            .then(|| Self::new(f32x8::from_array(sigma_db)))
    }

    /// Create from scalar sigma (same for all lanes).
    #[inline]
    pub fn from_scalar(sigma_db: f32) -> Self {
        Self::new(f32x8::splat(sigma_db))
    }

    /// Sample 8 shadowing coefficients with unit mean.
    ///
    /// # Arguments
    /// * `z` - 8 standard normal samples N(0,1)
    #[inline]
    pub fn sample(&self, z: f32x8) -> UnitMeanFading8 {
        // Corrected shadow in dB
        let shadow_db = self.sigma_db * z + self.correction_db;

        // Convert to linear
        let coefficient = simd_db_to_linear(shadow_db);

        UnitMeanFading8::from_raw_unchecked(coefficient)
    }

    /// Sample with SIMD (alias for sample).
    #[inline]
    pub fn sample_simd(&self, z: f32x8) -> UnitMeanFading8 {
        self.sample(z)
    }

    /// Get sigma values.
    #[inline]
    pub fn sigma_db(&self) -> f32x8 {
        self.sigma_db
    }

    /// Get correction values.
    #[inline]
    pub fn correction_db(&self) -> f32x8 {
        self.correction_db
    }
}

/// SIMD composite fading (shadowing + fast fading).
#[derive(Clone, Copy)]
pub struct CompositeFading8 {
    shadowing: UnitMeanShadowing8,
    k_factor: f32x8,
}

impl CompositeFading8 {
    /// Create a composite fading model.
    #[inline]
    pub fn new(shadow_sigma_db: f32x8, k_factor: f32x8) -> Self {
        Self {
            shadowing: UnitMeanShadowing8::new(shadow_sigma_db),
            k_factor,
        }
    }

    /// Create from scalar parameters (same for all lanes).
    #[inline]
    pub fn from_scalars(shadow_sigma_db: f32, k_factor: f32) -> Self {
        Self::new(f32x8::splat(shadow_sigma_db), f32x8::splat(k_factor))
    }

    /// Sample 8 composite fading coefficients.
    ///
    /// # Arguments
    /// * `z_shadow` - 8 standard normals for shadowing
    /// * `i_fast` - 8 in-phase components for fast fading
    /// * `q_fast` - 8 quadrature components for fast fading
    #[inline]
    pub fn sample(&self, z_shadow: f32x8, i_fast: f32x8, q_fast: f32x8) -> UnitMeanFading8 {
        // Sample shadowing
        let shadow = self.shadowing.sample(z_shadow);

        // Sample fast fading
        // Use Rayleigh where K < 0.01, Rician otherwise
        let threshold = f32x8::splat(0.01);
        let is_rayleigh = self.k_factor.simd_lt(threshold);

        let rayleigh = RayleighFading8::sample(i_fast, q_fast);
        let rician = RicianFading8::sample(self.k_factor, i_fast, q_fast);

        let fast_fading = is_rayleigh.select(rayleigh.get(), rician.get());

        // Combine
        UnitMeanFading8::from_raw_unchecked(shadow.get() * fast_fading)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_rf::random::SimdRng;

    const NUM_SAMPLES: usize = 10_000;
    const TOLERANCE: f32 = 0.1; // 10% tolerance for SIMD statistical tests

    #[test]
    fn test_positive_f32x8_construction() {
        let v = PositiveF32x8::new(f32x8::splat(1.0));
        assert_eq!(v.get(), f32x8::splat(1.0));

        let v = PositiveF32x8::ZERO;
        assert_eq!(v.get(), f32x8::splat(0.0));
    }

    #[test]
    fn test_positive_power8_operations() {
        let p1 = PositivePower8::splat(1.0);
        let p2 = PositivePower8::splat(2.0);

        let sum = p1.add(p2);
        let arr: [f32; 8] = sum.get().into();
        for v in arr {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rayleigh_fading8_unit_mean() {
        let mut rng = SimdRng::new(12345);
        let mut sum = 0.0;

        for _ in 0..NUM_SAMPLES {
            let (i, q) = rng.randn_pair();
            let fading = RayleighFading8::sample(i, q);
            let arr: [f32; 8] = fading.get().into();
            sum += arr.iter().sum::<f32>();
        }

        let mean = sum / (NUM_SAMPLES * 8) as f32;
        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "SIMD Rayleigh mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_rician_fading8_unit_mean() {
        let mut rng = SimdRng::new(54321);
        let k_factor = f32x8::splat(3.0);
        let mut sum = 0.0;

        for _ in 0..NUM_SAMPLES {
            let (i, q) = rng.randn_pair();
            let fading = RicianFading8::sample(k_factor, i, q);
            let arr: [f32; 8] = fading.get().into();
            sum += arr.iter().sum::<f32>();
        }

        let mean = sum / (NUM_SAMPLES * 8) as f32;
        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "SIMD Rician mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_shadowing8_corrected_unit_mean() {
        let mut rng = SimdRng::new(98765);
        let shadowing = UnitMeanShadowing8::from_scalar(8.0);
        let mut sum = 0.0;

        for _ in 0..NUM_SAMPLES {
            let z = rng.randn();
            let fading = shadowing.sample(z);
            let arr: [f32; 8] = fading.get().into();
            sum += arr.iter().sum::<f32>();
        }

        let mean = sum / (NUM_SAMPLES * 8) as f32;
        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "SIMD corrected shadowing mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_composite_fading8_unit_mean() {
        let mut rng = SimdRng::new(11111);
        let composite = CompositeFading8::from_scalars(6.0, 3.0);
        let mut sum = 0.0;

        for _ in 0..NUM_SAMPLES {
            let z = rng.randn();
            let (i, q) = rng.randn_pair();
            let fading = composite.sample(z, i, q);
            let arr: [f32; 8] = fading.get().into();
            sum += arr.iter().sum::<f32>();
        }

        let mean = sum / (NUM_SAMPLES * 8) as f32;
        assert!(
            (mean - 1.0).abs() < TOLERANCE * 1.5,
            "SIMD composite fading mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_fading8_apply() {
        let power = PositivePower8::splat(2.0);
        let fading = UnitMeanFading8::from_raw_unchecked(f32x8::splat(0.5));

        let result = fading.apply(power);
        let arr: [f32; 8] = result.get().into();
        for v in arr {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }
}
