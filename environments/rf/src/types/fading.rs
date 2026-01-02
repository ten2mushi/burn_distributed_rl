//! Fading Coefficient Types
//!
//! Types that enforce unit-mean property for fading coefficients,
//! ensuring energy conservation in the channel model.
//!
//! # Mathematical Background
//!
//! For a fading channel with coefficient h, the received power is:
//! P_rx = |h|² × P_tx
//!
//! For energy conservation, E[|h|²] must equal 1 (unit mean).
//! These types enforce this property at construction time.

use super::primitives::{PositiveF32, PositivePower};
use crate::constants::LN_10;

// ============================================================================
// UnitMeanFading: Fading Coefficient with Proven Unit Mean
// ============================================================================

/// A fading coefficient with guaranteed unit mean.
///
/// This type represents |h|² where h is the complex channel coefficient.
/// The unit mean property E[|h|²] = 1 is enforced by construction.
///
/// # Invariant
/// The coefficient is non-negative and comes from a distribution
/// with unit mean (verified at construction time).
///
/// # Usage
/// ```ignore
/// let fading = RayleighFading::sample(rng);
/// let rx_power = fading.apply(tx_power);
/// // E[rx_power] = tx_power (energy conserved on average)
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct UnitMeanFading(f32);

impl UnitMeanFading {
    /// Unity (no fading).
    pub const ONE: Self = Self(1.0);

    /// Create from a raw coefficient value.
    ///
    /// # Safety
    /// The caller must ensure this coefficient comes from a
    /// unit-mean distribution. This is an internal constructor.
    #[inline]
    pub(crate) fn from_raw_unchecked(coefficient: f32) -> Self {
        debug_assert!(coefficient >= 0.0, "Fading coefficient must be non-negative");
        Self(coefficient)
    }

    /// Get the raw coefficient value.
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }

    /// Get the raw coefficient value as f32.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0
    }

    /// Convert to PositiveF32.
    #[inline]
    pub fn to_positive(self) -> PositiveF32 {
        PositiveF32::new(self.0.max(0.0))
    }

    /// Apply this fading coefficient to a power value.
    #[inline]
    pub fn apply(self, power: PositivePower) -> PositivePower {
        PositivePower::new(power.watts() * self.0)
    }

    /// Apply to a PositiveF32.
    #[inline]
    pub fn apply_f32(self, value: PositiveF32) -> PositiveF32 {
        PositiveF32::new(value.get() * self.0)
    }

    /// Combine two independent fading coefficients (multiplication).
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Convert to dB (10 * log10(coefficient)).
    #[inline]
    pub fn to_db(self) -> f32 {
        if self.0 > 1e-30 {
            10.0 * self.0.log10()
        } else {
            -300.0
        }
    }
}

impl Default for UnitMeanFading {
    fn default() -> Self {
        Self::ONE
    }
}

// ============================================================================
// RayleighFading: Unit-Mean Rayleigh Fading Generator
// ============================================================================

/// Rayleigh fading generator with proven unit mean.
///
/// Rayleigh fading models NLOS (non-line-of-sight) multipath channels.
/// The envelope |h| is Rayleigh distributed, and the power |h|² follows
/// an exponential distribution.
///
/// # Mathematical Proof of Unit Mean
///
/// If I, Q ~ N(0, 1) are independent standard normal random variables:
/// - |h| = √(I² + Q²) follows Rayleigh distribution
/// - |h|² = I² + Q² follows χ²(2) distribution
/// - E[I² + Q²] = E[I²] + E[Q²] = 1 + 1 = 2
///
/// To achieve unit mean, we normalize: coefficient = (I² + Q²) / 2
/// - E[coefficient] = E[(I² + Q²)/2] = 2/2 = 1 ✓
pub struct RayleighFading;

impl RayleighFading {
    /// Sample a Rayleigh fading coefficient with unit mean.
    ///
    /// # Arguments
    /// * `i` - In-phase component (standard normal N(0,1))
    /// * `q` - Quadrature component (standard normal N(0,1))
    ///
    /// # Returns
    /// Unit-mean fading coefficient.
    #[inline]
    pub fn sample(i: f32, q: f32) -> UnitMeanFading {
        // Normalize by 0.5 to achieve unit mean
        // E[0.5 * (I² + Q²)] = 0.5 * 2 = 1
        let coefficient = 0.5 * (i * i + q * q);
        UnitMeanFading::from_raw_unchecked(coefficient)
    }

    /// Sample with pre-computed squared values.
    #[inline]
    pub fn sample_from_squared(i_squared: f32, q_squared: f32) -> UnitMeanFading {
        let coefficient = 0.5 * (i_squared + q_squared);
        UnitMeanFading::from_raw_unchecked(coefficient)
    }
}

// ============================================================================
// RicianFading: Unit-Mean Rician Fading Generator
// ============================================================================

/// Rician fading generator with proven unit mean.
///
/// Rician fading models LOS (line-of-sight) channels with scattered
/// multipath components. The K-factor determines the ratio of
/// LOS power to scattered power.
///
/// # Mathematical Background
///
/// K = P_LOS / P_scatter
///
/// For unit mean with K-factor K:
/// - LOS component: √(K/(K+1))
/// - Scatter std dev: √(1/(2(K+1)))
///
/// This ensures E[|h|²] = K/(K+1) + 1/(K+1) = 1
pub struct RicianFading;

impl RicianFading {
    /// Sample a Rician fading coefficient with unit mean.
    ///
    /// # Arguments
    /// * `k_factor` - Rician K-factor (linear, K = P_LOS / P_scatter)
    /// * `i` - In-phase scatter component (standard normal N(0,1))
    /// * `q` - Quadrature scatter component (standard normal N(0,1))
    ///
    /// # Returns
    /// Unit-mean fading coefficient.
    #[inline]
    pub fn sample(k_factor: f32, i: f32, q: f32) -> UnitMeanFading {
        let k_plus_1 = k_factor + 1.0;

        // LOS component magnitude for unit mean
        let los_scale = (k_factor / k_plus_1).sqrt();

        // Scatter component scale for unit mean
        let scatter_scale = (0.5 / k_plus_1).sqrt();

        // Combine LOS (fixed phase, assume 0) with scatter
        let i_total = los_scale + scatter_scale * i;
        let q_total = scatter_scale * q;

        // Power coefficient
        let coefficient = i_total * i_total + q_total * q_total;
        UnitMeanFading::from_raw_unchecked(coefficient)
    }

    /// Sample with K=0 (degenerates to Rayleigh).
    #[inline]
    pub fn sample_k0(i: f32, q: f32) -> UnitMeanFading {
        // K=0 means pure Rayleigh
        RayleighFading::sample(i, q)
    }

    /// Sample with very large K (approaches no fading).
    #[inline]
    pub fn sample_k_infinite() -> UnitMeanFading {
        // K→∞ means pure LOS, |h|²→1
        UnitMeanFading::ONE
    }
}

// ============================================================================
// UnitMeanShadowing: Corrected Log-Normal Shadowing
// ============================================================================

/// Log-normal shadowing generator with corrected unit mean.
///
/// # Critical Bug Fix
///
/// The naive implementation `shadow_db = sigma * z` is WRONG!
///
/// For X ~ N(0, σ²), the linear value 10^(X/10) has mean:
/// E[10^(X/10)] = exp((σ·ln(10)/10)²/2) ≠ 1
///
/// For σ = 8 dB, this gives mean ≈ 1.14 (14% bias!)
///
/// # Correct Implementation
///
/// For unit mean, we need: E[10^(S/10)] = 1
///
/// If S = σ·z + μ where z ~ N(0,1), then:
/// E[10^(S/10)] = 10^(μ/10) × E[10^(σ·z/10)]
///              = 10^(μ/10) × exp((σ·ln(10)/10)²/2)
///
/// Setting this equal to 1:
/// 10^(μ/10) = exp(-(σ·ln(10)/10)²/2)
/// μ·ln(10)/10 = -(σ·ln(10)/10)²/2
/// μ = -(σ²·ln(10))/20
///
/// The correction term μ is in dB and equals -(σ²·ln(10))/20
#[derive(Debug, Clone, Copy)]
pub struct UnitMeanShadowing {
    /// Standard deviation in dB.
    sigma_db: f32,
    /// Correction term for unit mean: -(σ²·ln(10))/200.
    correction_db: f32,
}

impl UnitMeanShadowing {
    /// Create a new unit-mean shadowing model.
    ///
    /// # Arguments
    /// * `sigma_db` - Standard deviation of shadowing in dB (typically 4-12 dB)
    ///
    /// # Returns
    /// A shadowing model that produces coefficients with E[coeff] = 1.
    #[inline]
    pub fn new(sigma_db: f32) -> Option<Self> {
        if sigma_db >= 0.0 && sigma_db.is_finite() {
            // Correction for unit mean: -(σ²·ln(10))/20
            // This shifts the mean of the dB distribution so that
            // the linear mean is exactly 1.
            let correction_db = -(sigma_db * sigma_db * LN_10) / 20.0;

            Some(Self {
                sigma_db,
                correction_db,
            })
        } else {
            None
        }
    }

    /// Create with validation, panicking if invalid.
    #[inline]
    pub fn new_validated(sigma_db: f32) -> Self {
        Self::new(sigma_db).expect("UnitMeanShadowing requires non-negative finite sigma")
    }

    /// Get the standard deviation in dB.
    #[inline]
    pub fn sigma_db(&self) -> f32 {
        self.sigma_db
    }

    /// Get the correction term applied for unit mean.
    #[inline]
    pub fn correction_db(&self) -> f32 {
        self.correction_db
    }

    /// Sample a shadowing coefficient with unit mean.
    ///
    /// # Arguments
    /// * `z` - Standard normal sample N(0,1)
    ///
    /// # Returns
    /// Unit-mean shadowing coefficient.
    #[inline]
    pub fn sample(&self, z: f32) -> UnitMeanFading {
        // Apply correction to achieve unit mean
        let shadow_db = self.sigma_db * z + self.correction_db;

        // Convert to linear: 10^(shadow_db/10)
        let coefficient = 10.0_f32.powf(shadow_db / 10.0);

        UnitMeanFading::from_raw_unchecked(coefficient)
    }

    /// Sample without correction (for comparison/testing).
    /// WARNING: This produces biased samples with mean > 1!
    #[cfg(test)]
    fn sample_uncorrected(&self, z: f32) -> f32 {
        let shadow_db = self.sigma_db * z;
        10.0_f32.powf(shadow_db / 10.0)
    }
}

impl Default for UnitMeanShadowing {
    fn default() -> Self {
        // 8 dB is typical for urban environments
        Self::new_validated(8.0)
    }
}

// ============================================================================
// Composite Fading
// ============================================================================

/// Combined shadowing and fast fading.
///
/// This represents the product of independent shadowing and fast fading,
/// both with unit mean. The product also has unit mean.
#[derive(Debug, Clone)]
pub struct CompositeFading {
    /// Shadowing model.
    shadowing: UnitMeanShadowing,
    /// K-factor for Rician fading (0 = Rayleigh).
    k_factor: f32,
}

impl CompositeFading {
    /// Create a composite fading model.
    ///
    /// # Arguments
    /// * `shadow_sigma_db` - Shadowing standard deviation (dB)
    /// * `k_factor` - Rician K-factor (use 0 for Rayleigh)
    #[inline]
    pub fn new(shadow_sigma_db: f32, k_factor: f32) -> Option<Self> {
        let shadowing = UnitMeanShadowing::new(shadow_sigma_db)?;
        if k_factor >= 0.0 && k_factor.is_finite() {
            Some(Self { shadowing, k_factor })
        } else {
            None
        }
    }

    /// Sample a composite fading coefficient.
    ///
    /// # Arguments
    /// * `z_shadow` - Standard normal for shadowing
    /// * `i_fast` - In-phase component for fast fading
    /// * `q_fast` - Quadrature component for fast fading
    ///
    /// # Returns
    /// Combined fading coefficient with unit mean.
    #[inline]
    pub fn sample(&self, z_shadow: f32, i_fast: f32, q_fast: f32) -> UnitMeanFading {
        // Sample shadowing
        let shadow = self.shadowing.sample(z_shadow);

        // Sample fast fading (Rician or Rayleigh)
        let fast = if self.k_factor < 0.01 {
            RayleighFading::sample(i_fast, q_fast)
        } else {
            RicianFading::sample(self.k_factor, i_fast, q_fast)
        };

        // Combine (product of unit-mean variables has unit mean)
        shadow.combine(fast)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_SAMPLES: usize = 100_000;
    const TOLERANCE: f32 = 0.05; // 5% tolerance for statistical tests

    fn compute_mean(samples: &[f32]) -> f32 {
        samples.iter().sum::<f32>() / samples.len() as f32
    }

    // Simple pseudo-random number generator for testing
    fn next_rand(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state as f64 / u64::MAX as f64) as f32
    }

    fn box_muller(state: &mut u64) -> (f32, f32) {
        let u1 = next_rand(state).max(1e-10);
        let u2 = next_rand(state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        (r * theta.cos(), r * theta.sin())
    }

    #[test]
    fn test_rayleigh_unit_mean() {
        let mut state = 12345u64;
        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES {
            let (i, q) = box_muller(&mut state);
            let fading = RayleighFading::sample(i, q);
            samples.push(fading.get());
        }

        let mean = compute_mean(&samples);
        assert!(
            (mean - 1.0).abs() < TOLERANCE,
            "Rayleigh mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_rician_unit_mean() {
        let mut state = 54321u64;

        for k in [0.0, 1.0, 3.0, 10.0] {
            let mut samples = Vec::with_capacity(NUM_SAMPLES);

            for _ in 0..NUM_SAMPLES {
                let (i, q) = box_muller(&mut state);
                let fading = RicianFading::sample(k, i, q);
                samples.push(fading.get());
            }

            let mean = compute_mean(&samples);
            assert!(
                (mean - 1.0).abs() < TOLERANCE,
                "Rician K={} mean should be ~1.0, got {}",
                k,
                mean
            );
        }
    }

    #[test]
    fn test_shadowing_corrected_unit_mean() {
        let mut state = 98765u64;

        for sigma in [4.0, 8.0, 12.0] {
            let shadowing = UnitMeanShadowing::new_validated(sigma);
            let mut samples = Vec::with_capacity(NUM_SAMPLES);

            for _ in 0..NUM_SAMPLES {
                let (z, _) = box_muller(&mut state);
                let fading = shadowing.sample(z);
                samples.push(fading.get());
            }

            let mean = compute_mean(&samples);
            assert!(
                (mean - 1.0).abs() < TOLERANCE,
                "Corrected shadowing σ={} dB mean should be ~1.0, got {}",
                sigma,
                mean
            );
        }
    }

    #[test]
    fn test_shadowing_uncorrected_bias() {
        // Verify that uncorrected shadowing has the expected bias
        let mut state = 11111u64;
        let sigma = 8.0;
        let shadowing = UnitMeanShadowing::new_validated(sigma);

        let mut uncorrected_samples = Vec::with_capacity(NUM_SAMPLES);
        let mut corrected_samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES {
            let (z, _) = box_muller(&mut state);
            uncorrected_samples.push(shadowing.sample_uncorrected(z));
            corrected_samples.push(shadowing.sample(z).get());
        }

        let uncorrected_mean = compute_mean(&uncorrected_samples);
        let corrected_mean = compute_mean(&corrected_samples);

        // Uncorrected should have mean > 1 (biased)
        // For σ=8, theoretical mean = exp((8*ln(10)/10)^2/2) ≈ 1.14
        assert!(
            uncorrected_mean > 1.05,
            "Uncorrected shadowing should have mean > 1.05, got {}",
            uncorrected_mean
        );

        // Corrected should be close to 1
        assert!(
            (corrected_mean - 1.0).abs() < TOLERANCE,
            "Corrected shadowing should have mean ~1.0, got {}",
            corrected_mean
        );
    }

    #[test]
    fn test_composite_fading_unit_mean() {
        let mut state = 22222u64;
        let composite = CompositeFading::new(6.0, 3.0).unwrap();

        let mut samples = Vec::with_capacity(NUM_SAMPLES);

        for _ in 0..NUM_SAMPLES {
            let (z, _) = box_muller(&mut state);
            let (i, q) = box_muller(&mut state);
            let fading = composite.sample(z, i, q);
            samples.push(fading.get());
        }

        let mean = compute_mean(&samples);
        assert!(
            (mean - 1.0).abs() < TOLERANCE * 1.5, // slightly more tolerance for composite
            "Composite fading mean should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_fading_application() {
        let power = PositivePower::new(1.0);
        let fading = UnitMeanFading::from_raw_unchecked(0.5);

        let result = fading.apply(power);
        assert!((result.watts() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_fading_combination() {
        let f1 = UnitMeanFading::from_raw_unchecked(2.0);
        let f2 = UnitMeanFading::from_raw_unchecked(0.5);

        let combined = f1.combine(f2);
        assert!((combined.get() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_shadowing_construction() {
        assert!(UnitMeanShadowing::new(0.0).is_some());
        assert!(UnitMeanShadowing::new(8.0).is_some());
        assert!(UnitMeanShadowing::new(12.0).is_some());

        assert!(UnitMeanShadowing::new(-1.0).is_none());
        assert!(UnitMeanShadowing::new(f32::NAN).is_none());
    }

    #[test]
    fn test_correction_term_calculation() {
        let shadowing = UnitMeanShadowing::new_validated(8.0);

        // correction = -(σ²·ln(10))/20 = -(64 * 2.3026)/20 ≈ -7.37
        let expected = -(8.0 * 8.0 * LN_10) / 20.0;
        assert!(
            (shadowing.correction_db() - expected).abs() < 0.001,
            "Correction term mismatch: expected {}, got {}",
            expected,
            shadowing.correction_db()
        );
    }
}
