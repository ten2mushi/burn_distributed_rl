//! SIMD Complex Number Arithmetic
//!
//! Complex numbers are fundamental to RF signal processing for representing
//! baseband signals, channel coefficients, and spectral components.
//!
//! This module provides 8-wide SIMD complex arithmetic with real and imaginary
//! components stored as separate vectors for optimal memory layout.

use std::simd::f32x8;

use super::math::{simd_atan2, simd_cos, simd_sin, simd_sqrt};

/// SIMD Complex number with 8 lanes
///
/// Represents 8 complex numbers in parallel using Struct-of-Arrays layout.
/// Each component (re, im) is a separate f32x8 vector.
#[derive(Clone, Copy, Debug)]
pub struct SimdComplex {
    /// Real components [8 values]
    pub re: f32x8,
    /// Imaginary components [8 values]
    pub im: f32x8,
}

impl SimdComplex {
    /// Create a new SIMD complex number from real and imaginary parts
    #[inline]
    pub fn new(re: f32x8, im: f32x8) -> Self {
        Self { re, im }
    }

    /// Create a complex number with all lanes set to the same real value (imaginary = 0)
    #[inline]
    pub fn from_real(re: f32x8) -> Self {
        Self {
            re,
            im: f32x8::splat(0.0),
        }
    }

    /// Create a complex number from scalar values, broadcast to all lanes
    #[inline]
    pub fn splat(re: f32, im: f32) -> Self {
        Self {
            re: f32x8::splat(re),
            im: f32x8::splat(im),
        }
    }

    /// Create a zero complex number
    #[inline]
    pub fn zero() -> Self {
        Self {
            re: f32x8::splat(0.0),
            im: f32x8::splat(0.0),
        }
    }

    /// Create a complex number from polar coordinates
    ///
    /// # Arguments
    /// * `mag` - Magnitude (absolute value)
    /// * `phase` - Phase angle in radians
    ///
    /// # Returns
    /// Complex number z = mag * (cos(phase) + i*sin(phase))
    #[inline]
    pub fn from_polar(mag: f32x8, phase: f32x8) -> Self {
        Self {
            re: mag * simd_cos(phase),
            im: mag * simd_sin(phase),
        }
    }

    /// Complex addition: (a + bi) + (c + di) = (a+c) + (b+d)i
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Complex subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex conjugate: (a + bi)* = a - bi
    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Squared magnitude (absolute value squared): |z|² = re² + im²
    ///
    /// This is more efficient than `abs()` when you only need the squared
    /// magnitude, common for power calculations.
    #[inline]
    pub fn abs_sq(self) -> f32x8 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude (absolute value): |z| = sqrt(re² + im²)
    #[inline]
    pub fn abs(self) -> f32x8 {
        simd_sqrt(self.abs_sq())
    }

    /// Phase angle (argument): arg(z) = atan2(im, re)
    ///
    /// Returns angle in radians in the range [-π, π].
    #[inline]
    pub fn arg(self) -> f32x8 {
        simd_atan2(self.im, self.re)
    }

    /// Scale by a real factor: z * k = k*re + k*im*i
    #[inline]
    pub fn scale(self, factor: f32x8) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    /// Complex division: (a + bi) / (c + di) = (ac + bd)/(c² + d²) + (bc - ad)/(c² + d²)i
    #[inline]
    pub fn div(self, other: Self) -> Self {
        let denom = other.abs_sq();
        let eps = f32x8::splat(1e-10);
        let inv_denom = f32x8::splat(1.0) / (denom + eps);

        Self {
            re: (self.re * other.re + self.im * other.im) * inv_denom,
            im: (self.im * other.re - self.re * other.im) * inv_denom,
        }
    }

    /// Reciprocal: 1/z = z*/|z|²
    #[inline]
    pub fn recip(self) -> Self {
        let inv_abs_sq = f32x8::splat(1.0) / (self.abs_sq() + f32x8::splat(1e-10));
        Self {
            re: self.re * inv_abs_sq,
            im: -self.im * inv_abs_sq,
        }
    }

    /// Multiply by i: z * i = -im + re*i
    #[inline]
    pub fn mul_i(self) -> Self {
        Self {
            re: -self.im,
            im: self.re,
        }
    }

    /// Multiply by -i: z * (-i) = im - re*i
    #[inline]
    pub fn mul_neg_i(self) -> Self {
        Self {
            re: self.im,
            im: -self.re,
        }
    }

    /// Normalized version: z / |z|
    ///
    /// Returns unit complex number with same phase.
    #[inline]
    pub fn normalize(self) -> Self {
        let inv_mag = f32x8::splat(1.0) / (self.abs() + f32x8::splat(1e-10));
        Self {
            re: self.re * inv_mag,
            im: self.im * inv_mag,
        }
    }

    /// Complex exponential: e^(a + bi) = e^a * (cos(b) + i*sin(b))
    #[inline]
    pub fn exp(self) -> Self {
        let exp_re = super::math::simd_exp(self.re);
        Self::from_polar(exp_re, self.im)
    }

    /// Extract to arrays (for debugging or interop)
    #[inline]
    pub fn to_arrays(self) -> ([f32; 8], [f32; 8]) {
        (self.re.into(), self.im.into())
    }

    /// Create from arrays
    #[inline]
    pub fn from_arrays(re: [f32; 8], im: [f32; 8]) -> Self {
        Self {
            re: f32x8::from_array(re),
            im: f32x8::from_array(im),
        }
    }
}

// Implement standard traits for convenience

impl Default for SimdComplex {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Add for SimdComplex {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        SimdComplex::add(self, other)
    }
}

impl std::ops::Sub for SimdComplex {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        SimdComplex::sub(self, other)
    }
}

impl std::ops::Mul for SimdComplex {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        SimdComplex::mul(self, other)
    }
}

impl std::ops::Neg for SimdComplex {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_rf::math::PI;

    const TOLERANCE: f32 = 2e-4;

    fn assert_complex_close(actual: SimdComplex, expected: SimdComplex, tolerance: f32, msg: &str) {
        let (act_re, act_im) = actual.to_arrays();
        let (exp_re, exp_im) = expected.to_arrays();

        for i in 0..8 {
            let diff_re = (act_re[i] - exp_re[i]).abs();
            let diff_im = (act_im[i] - exp_im[i]).abs();
            assert!(
                diff_re < tolerance && diff_im < tolerance,
                "{}: lane {} - expected ({}, {}), got ({}, {})",
                msg,
                i,
                exp_re[i],
                exp_im[i],
                act_re[i],
                act_im[i]
            );
        }
    }

    #[test]
    fn test_complex_from_polar() {
        // Unit magnitude, various phases
        let mag = f32x8::splat(1.0);
        let phase = f32x8::from_array([0.0, PI / 4.0, PI / 2.0, PI, -PI / 2.0, -PI / 4.0, PI / 3.0, -PI / 3.0]);

        let z = SimdComplex::from_polar(mag, phase);

        // Check that |z| = mag
        let abs = z.abs();
        let abs_arr: [f32; 8] = abs.into();
        for (i, val) in abs_arr.iter().enumerate() {
            assert!(
                (*val - 1.0).abs() < TOLERANCE,
                "from_polar magnitude lane {}: expected 1.0, got {}",
                i,
                val
            );
        }

        // Check that arg(z) = phase
        let arg = z.arg();
        let arg_arr: [f32; 8] = arg.into();
        let phase_arr: [f32; 8] = phase.into();
        for i in 0..8 {
            let diff = (arg_arr[i] - phase_arr[i]).abs();
            assert!(
                diff < TOLERANCE || diff > 2.0 * PI - TOLERANCE, // Handle wrap-around
                "from_polar arg lane {}: expected {}, got {}",
                i,
                phase_arr[i],
                arg_arr[i]
            );
        }
    }

    #[test]
    fn test_complex_multiplication() {
        // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let a = SimdComplex::splat(1.0, 2.0);
        let b = SimdComplex::splat(3.0, 4.0);
        let result = a.mul(b);

        let expected = SimdComplex::splat(-5.0, 10.0);
        assert_complex_close(result, expected, TOLERANCE, "complex multiplication");
    }

    #[test]
    fn test_complex_conjugate() {
        let z = SimdComplex::splat(3.0, 4.0);
        let z_conj = z.conj();

        // z * conj(z) = |z|²
        let product = z.mul(z_conj);
        let abs_sq = z.abs_sq();

        let (prod_re, prod_im) = product.to_arrays();
        let abs_sq_arr: [f32; 8] = abs_sq.into();

        for i in 0..8 {
            assert!(
                (prod_re[i] - abs_sq_arr[i]).abs() < TOLERANCE,
                "z*conj(z) real lane {}: expected {}, got {}",
                i,
                abs_sq_arr[i],
                prod_re[i]
            );
            assert!(
                prod_im[i].abs() < TOLERANCE,
                "z*conj(z) imag lane {}: expected 0, got {}",
                i,
                prod_im[i]
            );
        }
    }

    #[test]
    fn test_complex_abs_sq() {
        // |3 + 4i|² = 9 + 16 = 25
        let z = SimdComplex::splat(3.0, 4.0);
        let abs_sq = z.abs_sq();
        let abs_sq_arr: [f32; 8] = abs_sq.into();

        for (i, val) in abs_sq_arr.iter().enumerate() {
            assert!(
                (*val - 25.0).abs() < TOLERANCE,
                "abs_sq lane {}: expected 25.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_complex_abs() {
        // |3 + 4i| = 5
        let z = SimdComplex::splat(3.0, 4.0);
        let abs = z.abs();
        let abs_arr: [f32; 8] = abs.into();

        for (i, val) in abs_arr.iter().enumerate() {
            assert!(
                (*val - 5.0).abs() < TOLERANCE,
                "abs lane {}: expected 5.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_complex_division() {
        // (1 + 2i) / (1 + 2i) = 1
        let z = SimdComplex::splat(1.0, 2.0);
        let result = z.div(z);

        let expected = SimdComplex::splat(1.0, 0.0);
        assert_complex_close(result, expected, TOLERANCE, "complex division self");
    }

    #[test]
    fn test_complex_mul_i() {
        // (3 + 4i) * i = -4 + 3i
        let z = SimdComplex::splat(3.0, 4.0);
        let result = z.mul_i();

        let expected = SimdComplex::splat(-4.0, 3.0);
        assert_complex_close(result, expected, TOLERANCE, "multiply by i");
    }

    #[test]
    fn test_complex_exp() {
        // e^(i*π) = -1
        let z = SimdComplex::splat(0.0, PI);
        let result = z.exp();

        let expected = SimdComplex::splat(-1.0, 0.0);
        assert_complex_close(result, expected, TOLERANCE, "e^(i*pi)");
    }
}
