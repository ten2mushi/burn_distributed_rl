//! SIMD helper functions for vectorized physics computation.
//!
//! Provides fast approximations for:
//! - Square root and reciprocal square root
//! - Trigonometric functions (sin, cos, atan2, asin)
//!
//! All functions operate on f32x8 vectors (8 values at once).

#![cfg(feature = "simd")]

use std::f32::consts::PI;
use std::simd::{cmp::SimdPartialOrd, f32x8, num::SimdFloat, StdFloat};

// ============================================================================
// Square Root Operations
// ============================================================================

/// Fast reciprocal square root using Newton-Raphson iteration.
/// rsqrt(x) ≈ 1 / sqrt(x)
#[inline(always)]
pub fn simd_rsqrt(x: f32x8) -> f32x8 {
    let half = f32x8::splat(0.5);
    let three = f32x8::splat(3.0);

    // Initial estimate using hardware sqrt
    let y = x.sqrt().recip();

    // One Newton-Raphson iteration: y = y * (3 - x * y * y) / 2
    y * (three - x * y * y) * half
}

/// Fast square root using reciprocal sqrt.
#[inline(always)]
pub fn simd_sqrt(x: f32x8) -> f32x8 {
    x.sqrt()
}

// ============================================================================
// Small Angle Trigonometric Approximations
// ============================================================================

/// Sin approximation for small angles (|x| < π/2).
/// Uses Taylor series: sin(x) ≈ x - x³/6 + x⁵/120
#[inline(always)]
pub fn simd_sin_small(x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;

    let c3 = f32x8::splat(1.0 / 6.0);
    let c5 = f32x8::splat(1.0 / 120.0);

    x - x3 * c3 + x5 * c5
}

/// Cos approximation for small angles (|x| < π/2).
/// Uses Taylor series: cos(x) ≈ 1 - x²/2 + x⁴/24
#[inline(always)]
pub fn simd_cos_small(x: f32x8) -> f32x8 {
    let x2 = x * x;
    let x4 = x2 * x2;

    let one = f32x8::splat(1.0);
    let c2 = f32x8::splat(0.5);
    let c4 = f32x8::splat(1.0 / 24.0);

    one - x2 * c2 + x4 * c4
}

// ============================================================================
// Full Range Trigonometric Functions
// ============================================================================

/// Sin approximation using Bhaskara's formula with range reduction.
/// Good accuracy over full range with reasonable performance.
#[inline(always)]
pub fn simd_sin(x: f32x8) -> f32x8 {
    // Range reduction to [-π, π]
    let two_pi = f32x8::splat(2.0 * PI);
    let pi = f32x8::splat(PI);
    let inv_two_pi = f32x8::splat(1.0 / (2.0 * PI));

    // x = x - floor(x / 2π + 0.5) * 2π
    let half = f32x8::splat(0.5);
    let n = (x * inv_two_pi + half).floor();
    let x_reduced = x - n * two_pi;

    // Use Taylor series for reduced range
    // For better accuracy, we use a 7th order approximation
    let x2 = x_reduced * x_reduced;
    let x3 = x2 * x_reduced;
    let x5 = x3 * x2;
    let x7 = x5 * x2;

    let c3 = f32x8::splat(-1.0 / 6.0);
    let c5 = f32x8::splat(1.0 / 120.0);
    let c7 = f32x8::splat(-1.0 / 5040.0);

    x_reduced + x3 * c3 + x5 * c5 + x7 * c7
}

/// Cos approximation using sin(x + π/2).
#[inline(always)]
pub fn simd_cos(x: f32x8) -> f32x8 {
    let half_pi = f32x8::splat(PI / 2.0);
    simd_sin(x + half_pi)
}

// ============================================================================
// Inverse Trigonometric Functions
// ============================================================================

/// Atan2 approximation using polynomial fit.
/// Returns angle in radians in range [-π, π].
#[inline(always)]
pub fn simd_atan2(y: f32x8, x: f32x8) -> f32x8 {
    let pi = f32x8::splat(PI);
    let half_pi = f32x8::splat(PI / 2.0);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);
    let epsilon = f32x8::splat(1e-10);

    // Compute abs values for quadrant handling
    let ax = x.abs();
    let ay = y.abs();

    // Compute atan(min/max) to keep argument in [0, 1]
    let swap_mask = ay.simd_gt(ax);
    let a = swap_mask.select(ax, ay);
    let b = swap_mask.select(ay, ax).simd_max(epsilon);
    let z = a / b;

    // Polynomial approximation for atan(z) where z in [0, 1]
    // atan(z) ≈ z - z³/3 + z⁵/5 - z⁷/7 (Taylor series, slow convergence)
    // Using faster rational approximation
    let z2 = z * z;
    let z4 = z2 * z2;

    // Coefficients for minimax polynomial
    let c1 = f32x8::splat(0.9998660);
    let c3 = f32x8::splat(-0.3302995);
    let c5 = f32x8::splat(0.1801410);
    let c7 = f32x8::splat(-0.0851330);
    let c9 = f32x8::splat(0.0208351);

    let atan_z = z * (c1 + z2 * (c3 + z2 * (c5 + z2 * (c7 + z2 * c9))));

    // Adjust for quadrant
    let atan_result = swap_mask.select(half_pi - atan_z, atan_z);

    // Handle sign and quadrants
    let x_neg = x.simd_lt(zero);
    let y_neg = y.simd_lt(zero);

    let result = x_neg.select(pi - atan_result, atan_result);
    y_neg.select(-result, result)
}

/// Asin approximation with clamping for numerical stability.
/// Returns angle in radians in range [-π/2, π/2].
#[inline(always)]
pub fn simd_asin_clamped(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let half_pi = f32x8::splat(PI / 2.0);

    // Clamp input to [-1, 1]
    let x_clamped = x.simd_clamp(-one, one);

    // Check for gimbal lock (|x| close to 1)
    let abs_x = x_clamped.abs();
    let threshold = f32x8::splat(0.9999);
    let gimbal_lock = abs_x.simd_ge(threshold);

    // Polynomial approximation for asin
    // asin(x) ≈ x + x³/6 + 3x⁵/40 + 15x⁷/336
    let x2 = x_clamped * x_clamped;
    let x3 = x2 * x_clamped;
    let x5 = x3 * x2;
    let x7 = x5 * x2;

    let c3 = f32x8::splat(1.0 / 6.0);
    let c5 = f32x8::splat(3.0 / 40.0);
    let c7 = f32x8::splat(15.0 / 336.0);

    let asin_approx = x_clamped + x3 * c3 + x5 * c5 + x7 * c7;

    // For gimbal lock, return ±π/2
    let gimbal_result = half_pi.copysign(x_clamped);
    gimbal_lock.select(gimbal_result, asin_approx)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Select between two values based on mask.
/// Returns a[i] if mask[i] else b[i].
#[inline(always)]
pub fn simd_select(mask: std::simd::Mask<i32, 8>, a: f32x8, b: f32x8) -> f32x8 {
    mask.select(a, b)
}

/// Clamp values to range [min, max].
#[inline(always)]
pub fn simd_clamp(x: f32x8, min: f32x8, max: f32x8) -> f32x8 {
    x.simd_clamp(min, max)
}

/// Fast floor function.
#[inline(always)]
pub fn simd_floor(x: f32x8) -> f32x8 {
    x.floor()
}

/// Compute modulo (for angle wrapping).
#[inline(always)]
pub fn simd_fmod(x: f32x8, y: f32x8) -> f32x8 {
    x - (x / y).floor() * y
}

/// Wrap angle to [-π, π].
#[inline(always)]
pub fn simd_wrap_angle(angle: f32x8) -> f32x8 {
    let two_pi = f32x8::splat(2.0 * PI);
    let pi = f32x8::splat(PI);
    let inv_two_pi = f32x8::splat(1.0 / (2.0 * PI));
    let half = f32x8::splat(0.5);

    let n = (angle * inv_two_pi + half).floor();
    angle - n * two_pi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_rsqrt() {
        let x = f32x8::splat(4.0);
        let result = simd_rsqrt(x);
        let expected = f32x8::splat(0.5);
        let diff = (result - expected).abs();
        assert!(diff.reduce_max() < 0.01);
    }

    #[test]
    fn test_simd_sin_small() {
        let x = f32x8::splat(0.1);
        let result = simd_sin_small(x);
        let expected = f32x8::splat(0.1_f32.sin());
        let diff = (result - expected).abs();
        assert!(diff.reduce_max() < 1e-4);
    }

    #[test]
    fn test_simd_cos_small() {
        let x = f32x8::splat(0.1);
        let result = simd_cos_small(x);
        let expected = f32x8::splat(0.1_f32.cos());
        let diff = (result - expected).abs();
        assert!(diff.reduce_max() < 1e-4);
    }

    #[test]
    fn test_simd_atan2() {
        // Test each quadrant
        let y = f32x8::from_array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0]);
        let x = f32x8::from_array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0]);
        let result = simd_atan2(y, x);

        // Check each result
        for i in 0..8 {
            let expected = y.as_array()[i].atan2(x.as_array()[i]);
            let actual = result.as_array()[i];
            assert!(
                (actual - expected).abs() < 0.05,
                "atan2 failed at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_simd_asin_clamped() {
        let x = f32x8::from_array([0.0, 0.5, -0.5, 0.9, -0.9, 1.0, -1.0, 0.1]);
        let result = simd_asin_clamped(x);

        for i in 0..8 {
            let expected = x.as_array()[i].asin();
            let actual = result.as_array()[i];
            assert!(
                (actual - expected).abs() < 0.05,
                "asin failed at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }
}
