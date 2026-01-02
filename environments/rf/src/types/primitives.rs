//! Primitive Refinement Types
//!
//! Core types that enforce non-negativity and positivity constraints,
//! forming the foundation for all other type-safe RF constructs.

use std::ops::{Add, Mul, Div};

// ============================================================================
// PositiveF32: Non-negative f32
// ============================================================================

/// A non-negative f32 value (>= 0).
///
/// This is the foundation type for values that cannot be negative,
/// such as power, distance, and time duration.
///
/// # Invariant
/// `value >= 0.0 && value.is_finite()`
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PositiveF32(f32);

impl PositiveF32 {
    /// Zero value - always valid.
    pub const ZERO: Self = Self(0.0);

    /// One - useful for multiplicative identity.
    pub const ONE: Self = Self(1.0);

    /// Try to create a PositiveF32 from a raw f32.
    /// Returns `None` if the value is negative or non-finite.
    #[inline]
    pub fn try_new(value: f32) -> Option<Self> {
        (value >= 0.0 && value.is_finite()).then(|| Self(value))
    }

    /// Create a PositiveF32, panicking if invalid.
    /// Use this when you're certain the value is valid.
    #[inline]
    pub fn new(value: f32) -> Self {
        Self::try_new(value).expect("PositiveF32 requires non-negative finite value")
    }

    /// Get the inner f32 value.
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }

    /// Add two positive values (always produces positive result).
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Multiply by another positive value.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Scale by a positive scalar.
    #[inline]
    pub fn scale(self, factor: f32) -> Option<Self> {
        Self::try_new(self.0 * factor)
    }

    /// Divide by a non-zero positive value.
    #[inline]
    pub fn div(self, divisor: NonZeroPositiveF32) -> Self {
        Self(self.0 / divisor.0)
    }

    /// Compute square root (always valid for positive values).
    #[inline]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Clamp to a range (both bounds must be positive).
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    /// Convert to f32 for interop (one-way).
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

impl Default for PositiveF32 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Add for PositiveF32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul for PositiveF32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::iter::Sum for PositiveF32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc.add(x))
    }
}

// ============================================================================
// NonZeroPositiveF32: Strictly positive f32
// ============================================================================

/// A strictly positive f32 value (> 0).
///
/// Used for values that must be non-zero, such as frequency and
/// division denominators.
///
/// # Invariant
/// `value > 0.0 && value.is_finite()`
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct NonZeroPositiveF32(pub(crate) f32);

impl NonZeroPositiveF32 {
    /// One - useful default for many RF applications.
    pub const ONE: Self = Self(1.0);

    /// Try to create from raw f32.
    /// Returns `None` if value is not strictly positive or non-finite.
    #[inline]
    pub fn try_new(value: f32) -> Option<Self> {
        (value > 0.0 && value.is_finite()).then(|| Self(value))
    }

    /// Create, panicking if invalid.
    #[inline]
    pub fn new(value: f32) -> Self {
        Self::try_new(value).expect("NonZeroPositiveF32 requires positive finite value")
    }

    /// Get the inner value.
    #[inline]
    pub fn get(self) -> f32 {
        self.0
    }

    /// Multiply by another positive value.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }

    /// Divide by another non-zero positive (always valid).
    #[inline]
    pub fn div(self, other: Self) -> Self {
        Self(self.0 / other.0)
    }

    /// Square root (always produces positive result).
    #[inline]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Natural logarithm (always valid for positive values).
    #[inline]
    pub fn ln(self) -> f32 {
        self.0.ln()
    }

    /// Log base 10 (always valid for positive values).
    #[inline]
    pub fn log10(self) -> f32 {
        self.0.log10()
    }

    /// Convert to PositiveF32 (widening conversion).
    #[inline]
    pub fn to_positive(self) -> PositiveF32 {
        PositiveF32(self.0)
    }

    /// Convert to f32.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

impl From<NonZeroPositiveF32> for PositiveF32 {
    fn from(value: NonZeroPositiveF32) -> Self {
        PositiveF32(value.0)
    }
}

impl Mul for NonZeroPositiveF32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for NonZeroPositiveF32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

// ============================================================================
// PositivePower: Non-negative power in Watts
// ============================================================================

/// Power value in Watts, guaranteed non-negative.
///
/// This is the fundamental type for RF power calculations.
/// All operations preserve the non-negativity invariant.
///
/// # Invariant
/// `value >= 0.0 && value.is_finite()`
///
/// # Physical Meaning
/// Represents instantaneous power in linear scale (Watts).
/// For spectral density, units are Watts/Hz.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
#[repr(transparent)]
pub struct PositivePower(f32);

impl PositivePower {
    /// Zero power.
    pub const ZERO: Self = Self(0.0);

    /// One Watt.
    pub const ONE_WATT: Self = Self(1.0);

    /// One milliwatt (0.001 W).
    pub const ONE_MILLIWATT: Self = Self(0.001);

    /// One microwatt (1e-6 W).
    pub const ONE_MICROWATT: Self = Self(1e-6);

    /// Try to create from raw f32 value in Watts.
    #[inline]
    pub fn try_new(watts: f32) -> Option<Self> {
        (watts >= 0.0 && watts.is_finite()).then(|| Self(watts))
    }

    /// Create from raw f32, panicking if invalid.
    #[inline]
    pub fn new(watts: f32) -> Self {
        Self::try_new(watts).expect("PositivePower requires non-negative finite value")
    }

    /// Create from milliwatts.
    #[inline]
    pub fn from_milliwatts(mw: f32) -> Option<Self> {
        Self::try_new(mw * 0.001)
    }

    /// Create from microwatts.
    #[inline]
    pub fn from_microwatts(uw: f32) -> Option<Self> {
        Self::try_new(uw * 1e-6)
    }

    /// Get value in Watts.
    #[inline]
    pub fn watts(self) -> f32 {
        self.0
    }

    /// Get value in milliwatts.
    #[inline]
    pub fn milliwatts(self) -> f32 {
        self.0 * 1000.0
    }

    /// Get value in microwatts.
    #[inline]
    pub fn microwatts(self) -> f32 {
        self.0 * 1e6
    }

    /// Add two power values (always valid).
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Scale by a positive factor.
    #[inline]
    pub fn scale(self, factor: PositiveF32) -> Self {
        Self(self.0 * factor.get())
    }

    /// Scale by a raw factor, returning None if result is negative.
    #[inline]
    pub fn scale_by(self, factor: f32) -> Option<Self> {
        Self::try_new(self.0 * factor)
    }

    /// Apply attenuation (multiply by factor in [0, 1]).
    #[inline]
    pub fn attenuate(self, factor: PositiveF32) -> Self {
        Self(self.0 * factor.get())
    }

    /// Square root of power (useful for amplitude calculations).
    #[inline]
    pub fn sqrt(self) -> PositiveF32 {
        PositiveF32(self.0.sqrt())
    }

    /// Check if power is effectively zero (below threshold).
    #[inline]
    pub fn is_negligible(self, threshold: Self) -> bool {
        self.0 < threshold.0
    }

    /// Convert to f32 for interop.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0
    }

    /// Alias for watts() - get value in Watts.
    #[inline]
    pub fn as_watts(self) -> f32 {
        self.0
    }

    /// Apply a unit-mean fading coefficient.
    ///
    /// This is the type-safe way to apply fading to power.
    /// The fading coefficient is guaranteed to come from a unit-mean distribution.
    #[inline]
    pub fn apply_fading(self, fading: super::fading::UnitMeanFading) -> Self {
        Self(self.0 * fading.as_f32())
    }
}

impl Add for PositivePower {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::iter::Sum for PositivePower {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc.add(x))
    }
}

impl From<PositivePower> for PositiveF32 {
    fn from(power: PositivePower) -> Self {
        PositiveF32(power.0)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_f32_construction() {
        assert!(PositiveF32::try_new(0.0).is_some());
        assert!(PositiveF32::try_new(1.0).is_some());
        assert!(PositiveF32::try_new(f32::MAX).is_some());
        assert!(PositiveF32::try_new(-0.0).is_some()); // -0.0 == 0.0

        assert!(PositiveF32::try_new(-1.0).is_none());
        assert!(PositiveF32::try_new(f32::NEG_INFINITY).is_none());
        assert!(PositiveF32::try_new(f32::INFINITY).is_none());
        assert!(PositiveF32::try_new(f32::NAN).is_none());
    }

    #[test]
    fn test_non_zero_positive_f32_construction() {
        assert!(NonZeroPositiveF32::try_new(1.0).is_some());
        assert!(NonZeroPositiveF32::try_new(f32::MIN_POSITIVE).is_some());

        assert!(NonZeroPositiveF32::try_new(0.0).is_none());
        assert!(NonZeroPositiveF32::try_new(-1.0).is_none());
        assert!(NonZeroPositiveF32::try_new(f32::INFINITY).is_none());
    }

    #[test]
    fn test_positive_power_construction() {
        assert!(PositivePower::try_new(0.0).is_some());
        assert!(PositivePower::try_new(1.0).is_some());

        assert!(PositivePower::try_new(-1.0).is_none());
        assert!(PositivePower::try_new(f32::NAN).is_none());
    }

    #[test]
    fn test_positive_power_arithmetic() {
        let p1 = PositivePower::new(1.0);
        let p2 = PositivePower::new(2.0);

        let sum = p1.add(p2);
        assert!((sum.watts() - 3.0).abs() < 1e-6);

        let scaled = p1.scale(PositiveF32::new(0.5));
        assert!((scaled.watts() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_positive_power_conversions() {
        let p = PositivePower::new(0.001); // 1 mW
        assert!((p.milliwatts() - 1.0).abs() < 1e-6);
        assert!((p.microwatts() - 1000.0).abs() < 1e-3);

        let p2 = PositivePower::from_milliwatts(100.0).unwrap();
        assert!((p2.watts() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_positive_f32_operations() {
        let a = PositiveF32::new(4.0);
        let b = PositiveF32::new(2.0);

        assert!((a.sqrt().get() - 2.0).abs() < 1e-6);
        assert!((a.div(NonZeroPositiveF32::new(2.0)).get() - 2.0).abs() < 1e-6);

        let clamped = a.clamp(PositiveF32::new(1.0), PositiveF32::new(3.0));
        assert!((clamped.get() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_non_zero_positive_logarithms() {
        let x = NonZeroPositiveF32::new(10.0);
        assert!((x.log10() - 1.0).abs() < 1e-6);

        let e = NonZeroPositiveF32::new(std::f32::consts::E);
        assert!((e.ln() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_iterator() {
        let powers = vec![
            PositivePower::new(1.0),
            PositivePower::new(2.0),
            PositivePower::new(3.0),
        ];

        let total: PositivePower = powers.into_iter().sum();
        assert!((total.watts() - 6.0).abs() < 1e-6);
    }
}
