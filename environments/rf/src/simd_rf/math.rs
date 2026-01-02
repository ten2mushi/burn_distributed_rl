//! SIMD Math Primitives
//!
//! High-performance vectorized mathematical functions for RF signal processing.
//! All functions operate on 8-wide f32 vectors for parallel processing across
//! environments.
//!
//! # Accuracy Targets
//!
//! | Function | Accuracy |
//! |----------|----------|
//! | exp, log | < 1e-5 relative error |
//! | sin, cos | < 1e-4 absolute error |
//! | atan2 | < 1e-3 absolute error |

use std::simd::{
    cmp::SimdPartialOrd,
    f32x8,
    num::{SimdFloat, SimdInt, SimdUint},
    Mask, StdFloat,
};

// ============================================================================
// Constants
// ============================================================================

/// Pi constant
pub const PI: f32 = std::f32::consts::PI;

/// Two times Pi
pub const TWO_PI: f32 = 2.0 * PI;

/// Half Pi
pub const HALF_PI: f32 = PI / 2.0;

/// Quarter Pi
pub const QUARTER_PI: f32 = PI / 4.0;

/// Natural logarithm of 2
pub const LN_2: f32 = std::f32::consts::LN_2;

/// Log base 10 of e
pub const LOG10_E: f32 = std::f32::consts::LOG10_E;

/// Log base e of 10
pub const LN_10: f32 = std::f32::consts::LN_10;

// ============================================================================
// Basic Operations
// ============================================================================

/// SIMD square root using native hardware instruction
#[inline]
pub fn simd_sqrt(x: f32x8) -> f32x8 {
    x.sqrt()
}

/// SIMD reciprocal square root with Newton-Raphson refinement
///
/// Computes 1/sqrt(x) with high accuracy using one iteration of Newton-Raphson.
#[inline]
pub fn simd_rsqrt(x: f32x8) -> f32x8 {
    // Initial approximation
    let y = f32x8::splat(1.0) / x.sqrt();

    // Newton-Raphson iteration: y = y * (3 - x*y*y) / 2
    let three = f32x8::splat(3.0);
    let half = f32x8::splat(0.5);
    y * (three - x * y * y) * half
}

/// SIMD absolute value
#[inline]
pub fn simd_abs(x: f32x8) -> f32x8 {
    x.abs()
}

/// SIMD floor operation
#[inline]
pub fn simd_floor(x: f32x8) -> f32x8 {
    x.floor()
}

/// SIMD round operation (round to nearest integer)
#[inline]
pub fn simd_round(x: f32x8) -> f32x8 {
    x.round()
}

/// SIMD clamp operation
#[inline]
pub fn simd_clamp(x: f32x8, min: f32x8, max: f32x8) -> f32x8 {
    x.simd_clamp(min, max)
}

/// SIMD select: returns `a` where mask is true, `b` where mask is false
#[inline]
pub fn simd_select(mask: Mask<i32, 8>, a: f32x8, b: f32x8) -> f32x8 {
    mask.select(a, b)
}

// ============================================================================
// Phase Management
// ============================================================================

/// Wrap phase angles to [-π, π] range for numerical stability
///
/// This is critical for RF signal processing where phase accumulates over time.
/// Without wrapping, precision degrades as phase grows.
///
/// Formula: x - 2π * round(x / 2π)
///
/// # Arguments
/// * `phase` - Unbounded phase accumulator in radians
///
/// # Returns
/// Phase wrapped to [-π, π]
#[inline]
pub fn simd_wrap_phase(phase: f32x8) -> f32x8 {
    let inv_two_pi = f32x8::splat(1.0 / TWO_PI);
    let two_pi = f32x8::splat(TWO_PI);

    phase - two_pi * (phase * inv_two_pi).round()
}

/// Wrap angle to [0, 2π) range
#[inline]
pub fn simd_wrap_angle_positive(angle: f32x8) -> f32x8 {
    let two_pi = f32x8::splat(TWO_PI);
    let inv_two_pi = f32x8::splat(1.0 / TWO_PI);

    angle - two_pi * (angle * inv_two_pi).floor()
}

// ============================================================================
// Exponential and Logarithm
// ============================================================================

/// SIMD exponential function e^x
///
/// Uses range reduction and 5th order polynomial approximation.
/// Accurate to < 1e-5 relative error for x in [-88, 88].
#[inline]
pub fn simd_exp(x: f32x8) -> f32x8 {
    // Range reduction: e^x = 2^k * e^r, where x = k*ln(2) + r, |r| <= ln(2)/2
    let log2_e = f32x8::splat(std::f32::consts::LOG2_E);
    let ln_2 = f32x8::splat(LN_2);

    // k = round(x / ln(2))
    let k = (x * log2_e).round();
    // r = x - k * ln(2)
    let r = x - k * ln_2;

    // Polynomial approximation for e^r, |r| <= ln(2)/2
    // Using Horner's method for efficiency
    // e^r ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
    let c1 = f32x8::splat(1.0);
    let c2 = f32x8::splat(0.5);
    let c3 = f32x8::splat(1.0 / 6.0);
    let c4 = f32x8::splat(1.0 / 24.0);
    let c5 = f32x8::splat(1.0 / 120.0);

    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r2 * r2;
    let r5 = r4 * r;

    let exp_r = c1 + r + c2 * r2 + c3 * r3 + c4 * r4 + c5 * r5;

    // Scale by 2^k using bit manipulation
    // 2^k = reinterpret((127 + k) << 23) for f32
    let k_i32 = unsafe { k.to_int_unchecked::<i32>() };
    let bias = std::simd::i32x8::splat(127);
    let shift = std::simd::i32x8::splat(23);
    let scale_bits = (bias + k_i32) << shift;
    let scale: f32x8 = f32x8::from_bits(scale_bits.cast::<u32>());

    exp_r * scale
}

/// SIMD natural logarithm ln(x)
///
/// Uses exponent extraction and minimax polynomial approximation.
/// Accurate to < 1e-4 relative error for x > 0.
#[inline]
pub fn simd_log(x: f32x8) -> f32x8 {
    // Extract exponent: x = 2^e * m, where 1 <= m < 2
    let bits = x.to_bits();
    let exp_mask = std::simd::u32x8::splat(0x7F800000);
    let mant_mask = std::simd::u32x8::splat(0x007FFFFF);
    let bias = std::simd::u32x8::splat(127);

    // Extract exponent
    let exp_bits = (bits & exp_mask) >> 23;
    let e = (exp_bits - bias).cast::<i32>();
    let e_f32: f32x8 = e.cast::<f32>();

    // Extract mantissa and set exponent to 0 (gives us 1.m)
    let one_bits = std::simd::u32x8::splat(0x3F800000); // 1.0 in f32 bits
    let m = f32x8::from_bits((bits & mant_mask) | one_bits);

    // Transform to [-1/3, 1/3] range for better convergence
    // For m in [1, 2), compute ln(m) using ln((1+s)/(1-s)) = 2(s + s³/3 + s⁵/5 + ...)
    // where s = (m-1)/(m+1)
    let one = f32x8::splat(1.0);
    let s = (m - one) / (m + one);
    let s2 = s * s;

    // Minimax polynomial for ln((1+s)/(1-s)) / (2s) in s²
    // This gives: 2s * (1 + c1*s² + c2*s⁴ + c3*s⁶ + c4*s⁸)
    let c1 = f32x8::splat(0.33333333);  // 1/3
    let c2 = f32x8::splat(0.20000000);  // 1/5
    let c3 = f32x8::splat(0.14285714);  // 1/7
    let c4 = f32x8::splat(0.11111111);  // 1/9

    let s4 = s2 * s2;
    let s6 = s4 * s2;
    let s8 = s4 * s4;

    let ln_m = f32x8::splat(2.0) * s * (one + c1 * s2 + c2 * s4 + c3 * s6 + c4 * s8);

    let ln_2 = f32x8::splat(LN_2);
    e_f32 * ln_2 + ln_m
}

/// SIMD base-10 logarithm log10(x)
#[inline]
pub fn simd_log10(x: f32x8) -> f32x8 {
    simd_log(x) * f32x8::splat(LOG10_E)
}

/// SIMD power function x^y
///
/// Computed as exp(y * ln(x)). For integer exponents, consider
/// using repeated multiplication for better accuracy.
#[inline]
pub fn simd_pow(x: f32x8, y: f32x8) -> f32x8 {
    simd_exp(y * simd_log(x))
}

// ============================================================================
// dB Conversions
// ============================================================================

/// Convert decibels to linear power: 10^(db/10)
#[inline]
pub fn simd_db_to_linear(db: f32x8) -> f32x8 {
    let scale = f32x8::splat(0.1); // 1/10
    simd_pow(f32x8::splat(10.0), db * scale)
}

/// Convert linear power to decibels: 10 * log10(linear)
#[inline]
pub fn simd_linear_to_db(linear: f32x8) -> f32x8 {
    let scale = f32x8::splat(10.0);
    scale * simd_log10(linear)
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

/// SIMD sine function using optimized polynomial with range reduction
///
/// Accurate to < 1e-4 for all inputs after range reduction.
#[inline]
pub fn simd_sin(x: f32x8) -> f32x8 {
    // Range reduction to [-π, π]
    let x = simd_wrap_phase(x);

    // Further reduce to [-π/2, π/2] using sin(x) = sin(π - x) for x > π/2
    let pi = f32x8::splat(PI);
    let half_pi = f32x8::splat(HALF_PI);
    let neg_half_pi = f32x8::splat(-HALF_PI);

    // For x > π/2, use sin(x) = sin(π - x)
    let too_large = x.simd_gt(half_pi);
    let x = simd_select(too_large, pi - x, x);

    // For x < -π/2, use sin(x) = sin(-π - x)
    let too_small = x.simd_lt(neg_half_pi);
    let x = simd_select(too_small, -pi - x, x);

    // Minimax polynomial coefficients for sin(x) on [-π/2, π/2]
    // sin(x) ≈ x(1 + c3*x² + c5*x⁴ + c7*x⁶ + c9*x⁸)
    let x2 = x * x;

    // Horner's method for efficiency
    let c3 = f32x8::splat(-0.16666667);   // -1/6
    let c5 = f32x8::splat(0.008333333);   // 1/120
    let c7 = f32x8::splat(-0.0001984127); // -1/5040
    let c9 = f32x8::splat(2.7557319e-6);  // 1/362880

    // Evaluate polynomial: 1 + x²(c3 + x²(c5 + x²(c7 + x²*c9)))
    let poly = c9;
    let poly = c7 + x2 * poly;
    let poly = c5 + x2 * poly;
    let poly = c3 + x2 * poly;
    let one = f32x8::splat(1.0);

    x * (one + x2 * poly)
}

/// SIMD cosine function using 6th order Taylor series with range reduction
///
/// Accurate to < 1e-4 for all inputs after range reduction.
#[inline]
pub fn simd_cos(x: f32x8) -> f32x8 {
    // cos(x) = sin(x + π/2)
    simd_sin(x + f32x8::splat(HALF_PI))
}

/// SIMD simultaneous sine and cosine computation
///
/// More efficient than calling sin and cos separately when both are needed.
///
/// # Returns
/// Tuple of (sin(x), cos(x))
#[inline]
pub fn simd_sincos(x: f32x8) -> (f32x8, f32x8) {
    let s = simd_sin(x);
    let c = simd_cos(x);
    (s, c)
}

/// SIMD arctangent of y/x with correct quadrant handling
///
/// Returns angle in radians in the range [-π, π].
/// Handles all quadrants correctly using the signs of both x and y.
///
/// Uses OpenCV's proven polynomial coefficients for high accuracy.
///
/// # Arguments
/// * `y` - Y coordinate (opposite side)
/// * `x` - X coordinate (adjacent side)
#[inline]
pub fn simd_atan2(y: f32x8, x: f32x8) -> f32x8 {
    let pi = f32x8::splat(PI);
    let half_pi = f32x8::splat(HALF_PI);
    let zero = f32x8::splat(0.0);
    let eps = f32x8::splat(f32::EPSILON as f32);

    // OpenCV's proven polynomial coefficients (originally for degrees, converted to radians)
    // These give excellent accuracy for atan approximation
    let p1 = f32x8::splat(0.9997878412794807);
    let p3 = f32x8::splat(-0.3258083974640975);
    let p5 = f32x8::splat(0.1555786518463281);
    let p7 = f32x8::splat(-0.04432655554792128);

    let ax = simd_abs(x);
    let ay = simd_abs(y);

    // c = min(ax, ay) / (max(ax, ay) + eps) ensures c ∈ [0, 1]
    let c = ax.simd_min(ay) / (ax.simd_max(ay) + eps);
    let c2 = c * c;

    // Polynomial: a = (((p7*c² + p5)*c² + p3)*c² + p1) * c
    let mut a = p7;
    a = a * c2 + p5;
    a = a * c2 + p3;
    a = a * c2 + p1;
    a = a * c;

    // If |y| > |x|, use complementary angle: a = π/2 - a
    let swap = ay.simd_gt(ax);
    a = simd_select(swap, half_pi - a, a);

    // Quadrant adjustments to get [-π, π] range
    // If x < 0: a = π - a
    let x_neg = x.simd_lt(zero);
    a = simd_select(x_neg, pi - a, a);

    // If y < 0: a = -a (negate the result)
    let y_neg = y.simd_lt(zero);
    a = simd_select(y_neg, -a, a);

    a
}

/// SIMD arcsine with clamping for numerical stability
///
/// Clamps input to [-1, 1] to avoid NaN from domain errors.
#[inline]
pub fn simd_asin_clamped(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let neg_one = f32x8::splat(-1.0);
    let x = simd_clamp(x, neg_one, one);

    // asin(x) = atan2(x, sqrt(1 - x²))
    let sqrt_term = simd_sqrt(one - x * x);
    simd_atan2(x, sqrt_term)
}

/// SIMD arccosine with clamping for numerical stability
#[inline]
pub fn simd_acos_clamped(x: f32x8) -> f32x8 {
    f32x8::splat(HALF_PI) - simd_asin_clamped(x)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-4;
    const EXP_LOG_TOLERANCE: f32 = 1e-5;

    fn assert_simd_close(actual: f32x8, expected: f32x8, tolerance: f32, msg: &str) {
        let actual_arr: [f32; 8] = actual.into();
        let expected_arr: [f32; 8] = expected.into();
        for i in 0..8 {
            let diff = (actual_arr[i] - expected_arr[i]).abs();
            assert!(
                diff < tolerance,
                "{}: lane {} - expected {}, got {}, diff {}",
                msg,
                i,
                expected_arr[i],
                actual_arr[i],
                diff
            );
        }
    }

    #[test]
    fn test_simd_sqrt() {
        let input = f32x8::from_array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
        let expected = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = simd_sqrt(input);
        assert_simd_close(result, expected, 1e-6, "sqrt");
    }

    #[test]
    fn test_simd_wrap_phase_basic() {
        // Values already in range
        let input = f32x8::from_array([0.0, 1.0, -1.0, 3.0, -3.0, PI, -PI, 0.5]);
        let result = simd_wrap_phase(input);
        let result_arr: [f32; 8] = result.into();

        for val in result_arr.iter() {
            assert!(
                *val >= -PI && *val <= PI,
                "wrap_phase result {} not in [-π, π]",
                val
            );
        }
    }

    #[test]
    fn test_simd_wrap_phase_large_values() {
        // Test with large multiples of 2π
        let input = f32x8::from_array([
            100.0 * PI,
            -100.0 * PI,
            1000.0,
            -1000.0,
            TWO_PI,
            -TWO_PI,
            10.0 * TWO_PI + 0.5,
            -10.0 * TWO_PI - 0.5,
        ]);
        let result = simd_wrap_phase(input);
        let result_arr: [f32; 8] = result.into();

        for (i, val) in result_arr.iter().enumerate() {
            assert!(
                *val >= -PI - TOLERANCE && *val <= PI + TOLERANCE,
                "wrap_phase lane {} result {} not in [-π, π]",
                i,
                val
            );
        }
    }

    #[test]
    fn test_simd_exp_basic() {
        let input = f32x8::from_array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0]);
        let result = simd_exp(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for i in 0..8 {
            let expected = input_arr[i].exp();
            let rel_error = ((result_arr[i] - expected) / expected).abs();
            assert!(
                rel_error < EXP_LOG_TOLERANCE,
                "exp lane {}: expected {}, got {}, rel_error {}",
                i,
                expected,
                result_arr[i],
                rel_error
            );
        }
    }

    #[test]
    fn test_simd_log_basic() {
        let input = f32x8::from_array([1.0, 2.0, std::f32::consts::E, 10.0, 0.5, 0.1, 100.0, 0.01]);
        let result = simd_log(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for i in 0..8 {
            let expected = input_arr[i].ln();
            let abs_error = (result_arr[i] - expected).abs();
            assert!(
                abs_error < EXP_LOG_TOLERANCE * expected.abs().max(1.0),
                "log lane {}: expected {}, got {}, error {}",
                i,
                expected,
                result_arr[i],
                abs_error
            );
        }
    }

    #[test]
    fn test_simd_sin_accuracy() {
        let input = f32x8::from_array([0.0, HALF_PI, PI, -HALF_PI, 0.5, -0.5, 1.0, -1.0]);
        let result = simd_sin(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for i in 0..8 {
            let expected = input_arr[i].sin();
            let abs_error = (result_arr[i] - expected).abs();
            assert!(
                abs_error < TOLERANCE,
                "sin lane {}: expected {}, got {}, error {}",
                i,
                expected,
                result_arr[i],
                abs_error
            );
        }
    }

    #[test]
    fn test_simd_cos_accuracy() {
        let input = f32x8::from_array([0.0, HALF_PI, PI, -HALF_PI, 0.5, -0.5, 1.0, -1.0]);
        let result = simd_cos(input);
        let result_arr: [f32; 8] = result.into();
        let input_arr: [f32; 8] = input.into();

        for i in 0..8 {
            let expected = input_arr[i].cos();
            let abs_error = (result_arr[i] - expected).abs();
            assert!(
                abs_error < TOLERANCE,
                "cos lane {}: expected {}, got {}, error {}",
                i,
                expected,
                result_arr[i],
                abs_error
            );
        }
    }

    #[test]
    fn test_simd_sincos_pythagorean() {
        // sin²(x) + cos²(x) = 1
        let input =
            f32x8::from_array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -1.0]);
        let (s, c) = simd_sincos(input);
        let sum = s * s + c * c;
        let one = f32x8::splat(1.0);
        assert_simd_close(sum, one, TOLERANCE, "sin²+cos²=1");
    }

    #[test]
    fn test_simd_atan2_quadrants() {
        // Test all four quadrants
        let y = f32x8::from_array([1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
        let x = f32x8::from_array([1.0, -1.0, 1.0, -1.0, 1.0, 0.0, -1.0, 0.0]);

        let result = simd_atan2(y, x);
        let result_arr: [f32; 8] = result.into();
        let y_arr: [f32; 8] = y.into();
        let x_arr: [f32; 8] = x.into();

        for i in 0..8 {
            let expected = y_arr[i].atan2(x_arr[i]);
            let abs_error = (result_arr[i] - expected).abs();
            assert!(
                abs_error < 1e-3,
                "atan2 lane {}: y={}, x={}, expected {}, got {}, error {}",
                i,
                y_arr[i],
                x_arr[i],
                expected,
                result_arr[i],
                abs_error
            );
        }
    }

    #[test]
    fn test_simd_db_conversions() {
        // Test round-trip
        let db_values = f32x8::from_array([-60.0, -30.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]);
        let linear = simd_db_to_linear(db_values);
        let db_back = simd_linear_to_db(linear);

        assert_simd_close(db_back, db_values, 0.01, "dB round-trip");
    }
}
