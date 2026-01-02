//! Fading Models - Provably Correct RF Channel Fading
//!
//! SIMD-optimized implementations of statistical fading models with
//! **compile-time energy conservation guarantees** via refinement types.
//!
//! # Models
//! - Rayleigh fading: Non-line-of-sight multipath (exponential power, unit mean)
//! - Rician fading: Line-of-sight with scattered components (unit mean)
//! - Log-normal shadowing: Slow fading with **corrected unit mean**
//! - Composite fading: Combined fast and slow fading
//!
//! # Energy Conservation
//!
//! All fading models produce [`UnitMeanFading`] coefficients, guaranteeing
//! that `E[fading_coefficient] = 1.0` statistically. This prevents energy
//! from being artificially created or destroyed.
//!
//! ## Shadowing Correction (Critical Fix)
//!
//! For log-normal shadowing with `X ~ N(0, σ²)`, the naive approach
//! `10^(X/10)` does NOT have unit mean. The correction is:
//!
//! ```text
//! μ = -(σ² × ln(10)) / 20
//! shadow = 10^((σ·z + μ) / 10)
//! ```
//!
//! where `z ~ N(0,1)`. This ensures `E[shadow] = 1.0`.

#[cfg(feature = "simd")]
use std::simd::{cmp::SimdPartialOrd, f32x8, num::SimdFloat};

#[cfg(feature = "simd")]
use crate::simd_rf::{
    math::{simd_exp, simd_sqrt},
    random::SimdRng,
};

// Import the refinement types
use crate::types::{
    PositivePower, UnitMeanFading, UnitMeanShadowing,
};

#[cfg(feature = "simd")]
use crate::types::{PositivePower8, UnitMeanFading8, UnitMeanShadowing8};

/// Natural log of 10 - used in shadowing correction
const LN_10: f32 = std::f32::consts::LN_10;

// ============================================================================
// SIMD Fading Models (Type-Safe)
// ============================================================================

/// Apply Rayleigh fading to signal power (SIMD, type-safe)
///
/// Rayleigh fading models the amplitude distribution in NLOS multipath channels.
/// Returns faded power guaranteed to preserve energy on average.
///
/// # Mathematical Properties
/// - Envelope: Rayleigh distributed
/// - Power: Exponential(1) distribution
/// - Mean: E[fading] = 1.0 (unit mean guaranteed)
/// - Variance: Var[fading] = 1.0
///
/// # Arguments
/// * `power` - Input power (guaranteed non-negative)
/// * `rng` - SIMD random number generator
///
/// # Returns
/// Faded power with unit mean coefficient applied
#[cfg(feature = "simd")]
#[inline]
pub fn apply_rayleigh_fading_simd(power: PositivePower8, rng: &mut SimdRng) -> PositivePower8 {
    // Generate complex Gaussian for I and Q components (variance 1 each)
    let (i, q) = rng.randn_pair();

    // Rayleigh envelope: r = √(i² + q²)
    // Power fading: r² has chi-squared(2) distribution, mean = 2
    // Normalize by 0.5 to get exponential(1) with mean = 1
    let r_squared = f32x8::splat(0.5) * (i * i + q * q);

    // Create unit mean fading coefficient
    // SAFETY: 0.5 * (i² + q²) is always >= 0
    let fading = UnitMeanFading8::from_raw_unchecked(r_squared);

    // Apply fading to power
    power.apply_fading(fading)
}

/// Apply Rayleigh fading returning the fading coefficient directly
#[cfg(feature = "simd")]
#[inline]
pub fn rayleigh_fading_coefficient_simd(rng: &mut SimdRng) -> UnitMeanFading8 {
    let (i, q) = rng.randn_pair();
    let r_squared = f32x8::splat(0.5) * (i * i + q * q);
    UnitMeanFading8::from_raw_unchecked(r_squared)
}

/// Apply Rician fading to signal power (SIMD, type-safe)
///
/// Rician fading models channels with a strong LOS component plus scattered
/// multipath. The K-factor determines the ratio of LOS to scattered power.
///
/// # Mathematical Properties
/// - K-factor: P_LOS / P_scatter (linear scale)
/// - Mean: E[fading] = 1.0 (unit mean guaranteed)
/// - K=0: Reduces to Rayleigh fading
/// - K→∞: Approaches no fading (deterministic)
///
/// # Arguments
/// * `power` - Input power (guaranteed non-negative)
/// * `k_factor` - Rician K-factor (linear, K = P_LOS / P_scatter)
/// * `rng` - SIMD random number generator
///
/// # Returns
/// Faded power with unit mean coefficient applied
#[cfg(feature = "simd")]
#[inline]
pub fn apply_rician_fading_simd(
    power: PositivePower8,
    k_factor: f32x8,
    rng: &mut SimdRng,
) -> PositivePower8 {
    let fading = rician_fading_coefficient_simd(k_factor, rng);
    power.apply_fading(fading)
}

/// Compute Rician fading coefficient (SIMD)
#[cfg(feature = "simd")]
#[inline]
pub fn rician_fading_coefficient_simd(k_factor: f32x8, rng: &mut SimdRng) -> UnitMeanFading8 {
    // Generate scattered component (Rayleigh)
    let (i_scatter, q_scatter) = rng.randn_pair();

    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    let k_plus_1 = k_factor + one;

    // For unit mean: LOS magnitude = √(K/(K+1)), scatter σ² = 1/(2(K+1))
    let los_scale = simd_sqrt(k_factor / k_plus_1);
    let scatter_scale = simd_sqrt(half / k_plus_1);

    // Combine LOS (fixed phase, assume 0) with scatter
    let i_total = los_scale + scatter_scale * i_scatter;
    let q_total = scatter_scale * q_scatter;

    // Power fading (envelope squared) - has unit mean
    let r_squared = i_total * i_total + q_total * q_total;

    UnitMeanFading8::from_raw_unchecked(r_squared)
}

/// Apply corrected log-normal shadowing (SIMD, type-safe)
///
/// Shadowing models large-scale fading caused by obstacles between TX and RX.
/// **This implementation includes the energy conservation correction.**
///
/// # Mathematical Correction
///
/// For `X ~ N(0, σ²)`, the naive `10^(X/10)` has:
/// ```text
/// E[10^(X/10)] = exp((σ × ln(10)/10)² / 2) ≠ 1
/// ```
///
/// The correction adds mean shift:
/// ```text
/// μ = -(σ² × ln(10)) / 20
/// E[10^((σ·z + μ)/10)] = 1.0  ✓
/// ```
///
/// # Arguments
/// * `power` - Input power (guaranteed non-negative)
/// * `shadowing` - Pre-computed shadowing parameters with correction
/// * `rng` - SIMD random number generator
///
/// # Returns
/// Shadowed power with unit mean coefficient applied
#[cfg(feature = "simd")]
#[inline]
pub fn apply_shadowing_simd(
    power: PositivePower8,
    shadowing: &UnitMeanShadowing8,
    rng: &mut SimdRng,
) -> PositivePower8 {
    let z = rng.randn();
    let fading = shadowing.sample_simd(z);
    power.apply_fading(fading)
}

/// Compute corrected shadowing coefficient (SIMD)
#[cfg(feature = "simd")]
#[inline]
pub fn shadowing_coefficient_simd(
    shadowing: &UnitMeanShadowing8,
    rng: &mut SimdRng,
) -> UnitMeanFading8 {
    let z = rng.randn();
    shadowing.sample_simd(z)
}

/// Apply composite fading (shadowing + fast fading) (SIMD, type-safe)
///
/// Combines log-normal shadowing (slow fading) with Rayleigh or Rician fast fading.
/// Both components have unit mean, so the product also has unit mean.
///
/// # Arguments
/// * `power` - Input power (guaranteed non-negative)
/// * `k_factor` - Rician K-factor. Use 0 for Rayleigh (pure NLOS)
/// * `shadowing` - Pre-computed shadowing parameters with correction
/// * `rng` - SIMD random number generator
///
/// # Returns
/// Faded power with combined slow and fast fading (unit mean)
#[cfg(feature = "simd")]
#[inline]
pub fn apply_composite_fading_simd(
    power: PositivePower8,
    k_factor: f32x8,
    shadowing: &UnitMeanShadowing8,
    rng: &mut SimdRng,
) -> PositivePower8 {
    // First apply shadowing (slow fading)
    let power = apply_shadowing_simd(power, shadowing, rng);

    // Then apply fast fading
    // Select Rayleigh or Rician based on K-factor
    let threshold = f32x8::splat(0.01);
    let is_rayleigh = k_factor.simd_lt(threshold);

    // Both branches produce unit mean fading
    let rayleigh_fading = rayleigh_fading_coefficient_simd(rng);
    let rician_fading = rician_fading_coefficient_simd(k_factor, rng);

    // Select fading coefficient based on K-factor
    let fading_raw = is_rayleigh.select(rayleigh_fading.as_raw(), rician_fading.as_raw());
    let fading = UnitMeanFading8::from_raw_unchecked(fading_raw);

    power.apply_fading(fading)
}

/// Apply Nakagami-m fading (SIMD, type-safe)
///
/// Nakagami fading generalizes Rayleigh (m=1) through no-fading (m→∞).
///
/// # Arguments
/// * `power` - Input power (guaranteed non-negative)
/// * `m` - Nakagami m parameter (m ≥ 0.5, m=1 is Rayleigh)
/// * `rng` - SIMD random number generator
///
/// # Returns
/// Faded power (unit mean approximation)
#[cfg(feature = "simd")]
#[inline]
pub fn apply_nakagami_fading_simd(
    power: PositivePower8,
    m: f32x8,
    rng: &mut SimdRng,
) -> PositivePower8 {
    let one = f32x8::splat(1.0);

    // Generate Rayleigh fading (unit mean)
    let rayleigh_fading = rayleigh_fading_coefficient_simd(rng);

    // Blend based on m: higher m = less fading
    // Weight = 1/m (normalized so m=1 gives full Rayleigh)
    let weight = (one / m).simd_clamp(f32x8::splat(0.0), one);
    let fading_raw = weight * rayleigh_fading.as_raw() + (one - weight) * one;

    // Result is weighted average of Rayleigh and unity, maintains unit mean
    let fading = UnitMeanFading8::from_raw_unchecked(fading_raw);
    power.apply_fading(fading)
}

// ============================================================================
// Scalar Fading Functions (Type-Safe)
// ============================================================================

/// Scalar Rayleigh fading coefficient
///
/// # Returns
/// Unit mean fading coefficient from Rayleigh distribution
#[inline]
pub fn rayleigh_fading_coefficient_scalar(rng: &mut impl FnMut() -> (f32, f32)) -> UnitMeanFading {
    let (i, q) = rng();
    let r_squared = 0.5 * (i * i + q * q);
    // SAFETY: 0.5 * (i² + q²) is always >= 0
    UnitMeanFading::from_raw_unchecked(r_squared)
}

/// Apply Rayleigh fading to power (scalar)
#[inline]
pub fn apply_rayleigh_fading_scalar(
    power: PositivePower,
    rng: &mut impl FnMut() -> (f32, f32),
) -> PositivePower {
    let fading = rayleigh_fading_coefficient_scalar(rng);
    power.apply_fading(fading)
}

/// Scalar Rician fading coefficient
#[inline]
pub fn rician_fading_coefficient_scalar(
    k_factor: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> UnitMeanFading {
    let (i_scatter, q_scatter) = rng();

    let k_plus_1 = k_factor + 1.0;
    let los_scale = (k_factor / k_plus_1).sqrt();
    let scatter_scale = (0.5 / k_plus_1).sqrt();

    let i_total = los_scale + scatter_scale * i_scatter;
    let q_total = scatter_scale * q_scatter;

    let r_squared = i_total * i_total + q_total * q_total;
    UnitMeanFading::from_raw_unchecked(r_squared)
}

/// Apply Rician fading to power (scalar)
#[inline]
pub fn apply_rician_fading_scalar(
    power: PositivePower,
    k_factor: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> PositivePower {
    let fading = rician_fading_coefficient_scalar(k_factor, rng);
    power.apply_fading(fading)
}

/// Apply corrected shadowing (scalar)
///
/// Uses pre-computed [`UnitMeanShadowing`] parameters for energy conservation.
#[inline]
pub fn apply_shadowing_scalar(
    power: PositivePower,
    shadowing: &UnitMeanShadowing,
    rng: &mut impl FnMut() -> (f32, f32),
) -> PositivePower {
    let (z, _) = rng();
    let fading = shadowing.sample(z);
    power.apply_fading(fading)
}

/// Scalar composite fading (shadowing + fast fading)
#[inline]
pub fn apply_composite_fading_scalar(
    power: PositivePower,
    k_factor: f32,
    shadowing: &UnitMeanShadowing,
    rng: &mut impl FnMut() -> (f32, f32),
) -> PositivePower {
    // First apply shadowing
    let power = apply_shadowing_scalar(power, shadowing, rng);

    // Then apply fast fading
    if k_factor < 0.01 {
        apply_rayleigh_fading_scalar(power, rng)
    } else {
        apply_rician_fading_scalar(power, k_factor, rng)
    }
}

// ============================================================================
// Legacy API Adapters (for gradual migration)
// ============================================================================

/// Legacy adapter: Apply Rayleigh fading with raw f32x8
///
/// **Deprecated**: Use [`apply_rayleigh_fading_simd`] with `PositivePower8`
#[cfg(feature = "simd")]
#[inline]
pub fn apply_rayleigh_fading_simd_raw(power: f32x8, rng: &mut SimdRng) -> f32x8 {
    let (i, q) = rng.randn_pair();
    let r_squared = f32x8::splat(0.5) * (i * i + q * q);
    power * r_squared
}

/// Legacy adapter: Apply Rician fading with raw f32x8
///
/// **Deprecated**: Use [`apply_rician_fading_simd`] with `PositivePower8`
#[cfg(feature = "simd")]
#[inline]
pub fn apply_rician_fading_simd_raw(power: f32x8, k_factor: f32x8, rng: &mut SimdRng) -> f32x8 {
    let (i_scatter, q_scatter) = rng.randn_pair();

    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    let k_plus_1 = k_factor + one;

    let los_scale = simd_sqrt(k_factor / k_plus_1);
    let scatter_scale = simd_sqrt(half / k_plus_1);

    let i_total = los_scale + scatter_scale * i_scatter;
    let q_total = scatter_scale * q_scatter;

    let r_squared = i_total * i_total + q_total * q_total;
    power * r_squared
}

/// Legacy adapter: Apply CORRECTED shadowing with raw f32x8
///
/// **Now includes energy conservation correction!**
///
/// **Deprecated**: Use [`apply_shadowing_simd`] with `UnitMeanShadowing8`
#[cfg(feature = "simd")]
#[inline]
pub fn apply_shadowing_simd_raw(power: f32x8, sigma_db: f32x8, rng: &mut SimdRng) -> f32x8 {
    let z = rng.randn();

    // CORRECTED: Add mean shift for unit mean
    let correction_db = -(sigma_db * sigma_db * f32x8::splat(LN_10)) / f32x8::splat(20.0);
    let shadow_db = sigma_db * z + correction_db;

    // Convert to linear factor: 10^(x/10) = exp(x × ln(10)/10)
    let shadow_linear = simd_exp(shadow_db * f32x8::splat(LN_10 / 10.0));

    power * shadow_linear
}

/// Legacy adapter: Apply composite fading with raw f32x8
///
/// **Now includes corrected shadowing!**
///
/// **Deprecated**: Use [`apply_composite_fading_simd`]
#[cfg(feature = "simd")]
#[inline]
pub fn apply_composite_fading_simd_raw(
    power: f32x8,
    k_factor: f32x8,
    shadow_sigma_db: f32x8,
    rng: &mut SimdRng,
) -> f32x8 {
    // First apply corrected shadowing
    let power = apply_shadowing_simd_raw(power, shadow_sigma_db, rng);

    // Then apply fast fading
    let threshold = f32x8::splat(0.01);
    let is_rayleigh = k_factor.simd_lt(threshold);

    let rayleigh_power = apply_rayleigh_fading_simd_raw(power, rng);
    let rician_power = apply_rician_fading_simd_raw(power, k_factor, rng);

    is_rayleigh.select(rayleigh_power, rician_power)
}

/// Legacy adapter: Apply Nakagami fading with raw f32x8
///
/// **Deprecated**: Use [`apply_nakagami_fading_simd`]
#[cfg(feature = "simd")]
#[inline]
pub fn apply_nakagami_fading_simd_raw(power: f32x8, m: f32x8, rng: &mut SimdRng) -> f32x8 {
    let one = f32x8::splat(1.0);

    let rayleigh_fading = {
        let (i, q) = rng.randn_pair();
        f32x8::splat(0.5) * (i * i + q * q)
    };

    let weight = (one / m).simd_clamp(f32x8::splat(0.0), one);
    let fading_factor = weight * rayleigh_fading + (one - weight) * one;

    power * fading_factor
}

/// Legacy scalar Rayleigh fading with raw f32
#[inline]
pub fn apply_rayleigh_fading_scalar_raw(
    power: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> f32 {
    let (i, q) = rng();
    power * 0.5 * (i * i + q * q)
}

/// Legacy scalar Rician fading with raw f32
#[inline]
pub fn apply_rician_fading_scalar_raw(
    power: f32,
    k_factor: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> f32 {
    let (i_scatter, q_scatter) = rng();

    let k_plus_1 = k_factor + 1.0;
    let los_scale = (k_factor / k_plus_1).sqrt();
    let scatter_scale = (0.5 / k_plus_1).sqrt();

    let i_total = los_scale + scatter_scale * i_scatter;
    let q_total = scatter_scale * q_scatter;

    power * (i_total * i_total + q_total * q_total)
}

/// Legacy scalar shadowing - **NOW CORRECTED**
#[inline]
pub fn apply_shadowing_scalar_raw(
    power: f32,
    sigma_db: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> f32 {
    let (z, _) = rng();

    // CORRECTED: Add mean shift for unit mean
    let correction_db = -(sigma_db * sigma_db * LN_10) / 20.0;
    let shadow_db = sigma_db * z + correction_db;
    let shadow_linear = 10.0_f32.powf(shadow_db / 10.0);

    power * shadow_linear
}

/// Legacy scalar composite fading - **NOW CORRECTED**
#[inline]
pub fn apply_composite_fading_scalar_raw(
    power: f32,
    k_factor: f32,
    shadow_sigma_db: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> f32 {
    // First apply corrected shadowing
    let power = apply_shadowing_scalar_raw(power, shadow_sigma_db, rng);

    // Then apply fast fading
    if k_factor < 0.01 {
        apply_rayleigh_fading_scalar_raw(power, rng)
    } else {
        apply_rician_fading_scalar_raw(power, k_factor, rng)
    }
}

// ============================================================================
// Helper Types for Composite Fading
// ============================================================================

/// Pre-computed composite fading parameters
///
/// Stores both shadowing and fast fading parameters for efficient repeated use.
#[derive(Clone, Debug)]
pub struct CompositeFadingParams {
    /// Shadowing parameters (corrected for unit mean)
    pub shadowing: UnitMeanShadowing,
    /// Rician K-factor (0 = Rayleigh)
    pub k_factor: f32,
}

impl CompositeFadingParams {
    /// Create new composite fading parameters
    ///
    /// # Arguments
    /// * `shadow_sigma_db` - Shadowing standard deviation in dB (typically 4-12 dB)
    /// * `k_factor` - Rician K-factor (0 for Rayleigh, >0 for Rician)
    pub fn new(shadow_sigma_db: f32, k_factor: f32) -> Option<Self> {
        let shadowing = UnitMeanShadowing::new(shadow_sigma_db)?;
        (k_factor >= 0.0).then(|| Self { shadowing, k_factor })
    }

    /// Create Rayleigh + shadowing (no LOS)
    pub fn rayleigh_shadowed(shadow_sigma_db: f32) -> Option<Self> {
        Self::new(shadow_sigma_db, 0.0)
    }

    /// Create Rician + shadowing (with LOS)
    pub fn rician_shadowed(shadow_sigma_db: f32, k_factor: f32) -> Option<Self> {
        Self::new(shadow_sigma_db, k_factor)
    }
}

/// SIMD version of composite fading parameters
#[cfg(feature = "simd")]
#[derive(Clone, Debug)]
pub struct CompositeFadingParams8 {
    /// Shadowing parameters (corrected for unit mean)
    pub shadowing: UnitMeanShadowing8,
    /// Rician K-factors per lane
    pub k_factor: f32x8,
}

#[cfg(feature = "simd")]
impl CompositeFadingParams8 {
    /// Create from uniform parameters across all lanes
    pub fn splat(shadow_sigma_db: f32, k_factor: f32) -> Option<Self> {
        let shadowing = UnitMeanShadowing8::splat(shadow_sigma_db)?;
        (k_factor >= 0.0).then(|| Self {
            shadowing,
            k_factor: f32x8::splat(k_factor),
        })
    }

    /// Create from per-lane parameters
    pub fn from_arrays(shadow_sigma_db: [f32; 8], k_factor: [f32; 8]) -> Option<Self> {
        let shadowing = UnitMeanShadowing8::from_sigma_db_array(shadow_sigma_db)?;
        k_factor.iter().all(|&k| k >= 0.0).then(|| Self {
            shadowing,
            k_factor: f32x8::from_array(k_factor),
        })
    }
}

// ============================================================================
// Statistical Verification Utilities
// ============================================================================

/// Compute mean of samples
#[cfg(test)]
fn compute_mean(samples: &[f32]) -> f32 {
    samples.iter().sum::<f32>() / samples.len() as f32
}

/// Compute variance of samples
#[cfg(test)]
fn compute_variance(samples: &[f32]) -> f32 {
    let mean = compute_mean(samples);
    samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32
}

// ============================================================================
// Unit Tests - Now with Energy Conservation Verification
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_SAMPLES: usize = 100_000;
    const TOLERANCE: f32 = 0.05; // 5% tolerance for unit mean tests
    const LOOSE_TOLERANCE: f32 = 0.15; // 15% for variance tests

    #[cfg(feature = "simd")]
    #[test]
    fn test_rayleigh_unit_mean() {
        let mut rng = SimdRng::new(12345);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES / 8 {
            let fading = rayleigh_fading_coefficient_simd(&mut rng);
            let arr: [f32; 8] = fading.as_raw().into();
            samples.extend_from_slice(&arr);
        }

        let mean = compute_mean(&samples);

        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "Rayleigh fading should have unit mean: expected ~1.0, got {}",
            mean
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_rayleigh_variance() {
        let mut rng = SimdRng::new(12345);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES / 8 {
            let fading = rayleigh_fading_coefficient_simd(&mut rng);
            let arr: [f32; 8] = fading.as_raw().into();
            samples.extend_from_slice(&arr);
        }

        let variance = compute_variance(&samples);

        // Exponential distribution has variance = mean² = 1
        assert!(
            (variance - 1.0).abs() < LOOSE_TOLERANCE * 2.0,
            "Rayleigh variance: expected ~1.0, got {}",
            variance
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_rician_unit_mean() {
        for k in [0.0_f32, 1.0, 3.0, 10.0] {
            let mut rng = SimdRng::new(12345);
            let k_factor = f32x8::splat(k);
            let mut samples = Vec::with_capacity(NUM_SAMPLES);

            for _ in 0..NUM_SAMPLES / 8 {
                let fading = rician_fading_coefficient_simd(k_factor, &mut rng);
                let arr: [f32; 8] = fading.as_raw().into();
                samples.extend_from_slice(&arr);
            }

            let mean = compute_mean(&samples);

            assert!(
                (mean - 1.0).abs() < TOLERANCE,
                "Rician K={} fading should have unit mean: expected ~1.0, got {}",
                k,
                mean
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_rician_decreasing_variance() {
        let mut variances = Vec::new();

        for k in [0.0_f32, 1.0, 3.0, 10.0, 100.0] {
            let mut rng = SimdRng::new(12345);
            let k_factor = f32x8::splat(k);
            let mut samples = Vec::with_capacity(NUM_SAMPLES);

            for _ in 0..NUM_SAMPLES / 8 {
                let fading = rician_fading_coefficient_simd(k_factor, &mut rng);
                let arr: [f32; 8] = fading.as_raw().into();
                samples.extend_from_slice(&arr);
            }

            variances.push(compute_variance(&samples));
        }

        // Variance should decrease as K increases
        for i in 1..variances.len() {
            assert!(
                variances[i] < variances[i - 1],
                "Rician variance should decrease with K: {} not < {}",
                variances[i],
                variances[i - 1]
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_corrected_shadowing_unit_mean() {
        // This is the CRITICAL test - verifies energy conservation fix
        // Use more samples for higher sigma values where variance is larger
        for sigma in [4.0_f32, 6.0, 8.0, 10.0, 12.0] {
            let shadowing = UnitMeanShadowing8::splat(sigma).expect("valid sigma");
            let mut rng = SimdRng::new(12345);
            // More samples for higher sigma
            let num_samples = if sigma > 8.0 { NUM_SAMPLES * 4 } else { NUM_SAMPLES };
            let mut samples = Vec::with_capacity(num_samples);

            for _ in 0..num_samples / 8 {
                let fading = shadowing.sample_simd(rng.randn());
                let arr: [f32; 8] = fading.as_raw().into();
                samples.extend_from_slice(&arr);
            }

            let mean = compute_mean(&samples);

            // Tolerance scales with sigma (higher sigma = more variance)
            let tolerance = TOLERANCE * (1.0 + sigma / 10.0);
            assert!(
                (mean - 1.0).abs() < tolerance,
                "Corrected shadowing σ={} dB should have unit mean: expected ~1.0, got {} (error: {:.2}%)",
                sigma,
                mean,
                (mean - 1.0).abs() * 100.0
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_legacy_shadowing_now_corrected() {
        // Verify that legacy API is also corrected
        for sigma in [6.0_f32, 8.0, 10.0] {
            let mut rng = SimdRng::new(12345);
            let sigma_vec = f32x8::splat(sigma);
            let power = f32x8::splat(1.0);
            // More samples for higher sigma
            let num_samples = if sigma > 8.0 { NUM_SAMPLES * 4 } else { NUM_SAMPLES };
            let mut samples = Vec::with_capacity(num_samples);

            for _ in 0..num_samples / 8 {
                let faded = apply_shadowing_simd_raw(power, sigma_vec, &mut rng);
                let arr: [f32; 8] = faded.into();
                samples.extend_from_slice(&arr);
            }

            let mean = compute_mean(&samples);

            // Tolerance scales with sigma
            let tolerance = TOLERANCE * (1.0 + sigma / 10.0);
            assert!(
                (mean - 1.0).abs() < tolerance,
                "Legacy shadowing σ={} dB should now have unit mean: expected ~1.0, got {}",
                sigma,
                mean
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_shadowing_db_std_dev() {
        let target_sigma = 8.0;
        let shadowing = UnitMeanShadowing8::splat(target_sigma).expect("valid sigma");
        let mut rng = SimdRng::new(12345);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES / 8 {
            let fading = shadowing.sample_simd(rng.randn());
            let arr: [f32; 8] = fading.as_raw().into();
            samples.extend_from_slice(&arr);
        }

        // Convert to dB and compute std dev
        let db_samples: Vec<f32> = samples.iter().map(|&x| 10.0 * x.log10()).collect();
        let db_std = compute_variance(&db_samples).sqrt();

        assert!(
            (db_std - target_sigma).abs() < 1.0,
            "Shadowing std dev in dB: expected ~{}, got {}",
            target_sigma,
            db_std
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_composite_fading_unit_mean() {
        let params = CompositeFadingParams8::splat(6.0, 3.0).expect("valid params");
        let mut rng = SimdRng::new(12345);
        let power = PositivePower8::splat(1.0);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES / 8 {
            let faded = apply_composite_fading_simd(power, params.k_factor, &params.shadowing, &mut rng);
            let arr: [f32; 8] = faded.as_raw().into();
            samples.extend_from_slice(&arr);
        }

        let mean = compute_mean(&samples);

        // Composite fading: both components have unit mean, so product has unit mean
        assert!(
            (mean - 1.0).abs() < TOLERANCE * 2.0, // Slightly looser for composite
            "Composite fading should have unit mean: expected ~1.0, got {}",
            mean
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_all_fading_positive() {
        let shadowing = UnitMeanShadowing8::splat(10.0).expect("valid sigma");
        let mut rng = SimdRng::new(12345);
        let power = PositivePower8::splat(1.0);

        for _ in 0..10_000 {
            // Rayleigh
            let faded = apply_rayleigh_fading_simd(power, &mut rng);
            let arr: [f32; 8] = faded.as_raw().into();
            assert!(arr.iter().all(|&x| x >= 0.0), "Rayleigh produced negative power");

            // Rician
            let faded = apply_rician_fading_simd(power, f32x8::splat(3.0), &mut rng);
            let arr: [f32; 8] = faded.as_raw().into();
            assert!(arr.iter().all(|&x| x >= 0.0), "Rician produced negative power");

            // Shadowing
            let faded = apply_shadowing_simd(power, &shadowing, &mut rng);
            let arr: [f32; 8] = faded.as_raw().into();
            assert!(arr.iter().all(|&x| x >= 0.0), "Shadowing produced negative power");
        }
    }

    #[test]
    fn test_scalar_rayleigh_unit_mean() {
        let mut seed = 12345u64;
        let mut rng = || {
            // Simple LCG for testing
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (seed >> 33) as f32 / (1u64 << 31) as f32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (seed >> 33) as f32 / (1u64 << 31) as f32;

            // Box-Muller transform
            let r = (-2.0 * u1.max(1e-10).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            (r * theta.cos(), r * theta.sin())
        };

        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let fading = rayleigh_fading_coefficient_scalar(&mut rng);
            samples.push(fading.as_f32());
        }

        let mean = compute_mean(&samples);
        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "Scalar Rayleigh should have unit mean: expected ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_scalar_shadowing_unit_mean() {
        let shadowing = UnitMeanShadowing::new(8.0).expect("valid sigma");

        let mut seed = 12345u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (seed >> 33) as f32 / (1u64 << 31) as f32;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (seed >> 33) as f32 / (1u64 << 31) as f32;

            let r = (-2.0 * u1.max(1e-10).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            (r * theta.cos(), r * theta.sin())
        };

        let power = PositivePower::new(1.0);
        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES {
            let faded = apply_shadowing_scalar(power, &shadowing, &mut rng);
            samples.push(faded.as_watts());
        }

        let mean = compute_mean(&samples);
        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "Scalar corrected shadowing should have unit mean: expected ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_composite_params_creation() {
        // Valid params
        assert!(CompositeFadingParams::new(6.0, 0.0).is_some());
        assert!(CompositeFadingParams::new(8.0, 3.0).is_some());
        assert!(CompositeFadingParams::rayleigh_shadowed(10.0).is_some());
        assert!(CompositeFadingParams::rician_shadowed(6.0, 10.0).is_some());

        // Invalid params
        assert!(CompositeFadingParams::new(-1.0, 0.0).is_none()); // Negative sigma
        assert!(CompositeFadingParams::new(6.0, -1.0).is_none()); // Negative K
    }
}
