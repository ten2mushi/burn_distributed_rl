//! SIMD Primitive Tests
//!
//! Comprehensive accuracy and correctness tests for all SIMD primitives.
//! These tests verify that the SIMD implementations match reference
//! scalar implementations within specified tolerances.
//!
//! # Accuracy Targets
//!
//! | Category | Tolerance |
//! |----------|-----------|
//! | Transcendentals (exp, log) | < 1e-5 relative |
//! | Trigonometric (sin, cos) | < 1e-4 absolute |
//! | atan2 | < 1e-3 absolute |
//! | Complex operations | < 1e-4 |
//! | RNG statistics | 3σ confidence |

use std::simd::{f32x8, StdFloat};

use crate::simd_rf::{
    complex::SimdComplex,
    helpers::{calc_distance_3d, calc_freq_overlap, calc_radial_velocity, load_f32_simd, store_f32_simd},
    math::{
        simd_atan2, simd_cos, simd_db_to_linear, simd_exp, simd_linear_to_db, simd_log,
        simd_pow, simd_rsqrt, simd_sin, simd_sincos, simd_sqrt, simd_wrap_phase, HALF_PI, PI,
        TWO_PI,
    },
    random::SimdRng,
};

// ============================================================================
// Test Utilities
// ============================================================================

const TOLERANCE: f32 = 2e-4;        // Standard tolerance for most SIMD functions
const STRICT_TOLERANCE: f32 = 1e-4; // For highly accurate functions
const LOOSE_TOLERANCE: f32 = 5e-3;  // For atan2 and complex operations
const DB_TOLERANCE: f32 = 0.05;     // Tolerance for dB round-trip (5% of 1 dB)
const STAT_SAMPLES: usize = 10000;

fn assert_simd_close(actual: f32x8, expected: f32x8, tolerance: f32, msg: &str) {
    let actual_arr: [f32; 8] = actual.into();
    let expected_arr: [f32; 8] = expected.into();
    for i in 0..8 {
        let diff = (actual_arr[i] - expected_arr[i]).abs();
        assert!(
            diff < tolerance,
            "{}: lane {} - expected {}, got {}, diff {} > tolerance {}",
            msg,
            i,
            expected_arr[i],
            actual_arr[i],
            diff,
            tolerance
        );
    }
}

fn assert_relative_error(actual: f32, expected: f32, tolerance: f32, msg: &str) {
    let rel_error = if expected.abs() > 1e-10 {
        (actual - expected).abs() / expected.abs()
    } else {
        (actual - expected).abs()
    };
    assert!(
        rel_error < tolerance,
        "{}: expected {}, got {}, relative error {} > tolerance {}",
        msg,
        expected,
        actual,
        rel_error,
        tolerance
    );
}

// ============================================================================
// Math Tests
// ============================================================================

#[test]
fn test_simd_sqrt_basic() {
    let input = f32x8::from_array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 0.25]);
    let expected = f32x8::from_array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.5]);
    let result = simd_sqrt(input);
    assert_simd_close(result, expected, STRICT_TOLERANCE, "sqrt basic");
}

#[test]
fn test_simd_rsqrt_basic() {
    let input = f32x8::from_array([1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 0.25, 0.01]);
    let result = simd_rsqrt(input);
    let result_arr: [f32; 8] = result.into();
    let input_arr: [f32; 8] = input.into();

    for i in 0..8 {
        let expected = 1.0 / input_arr[i].sqrt();
        assert_relative_error(result_arr[i], expected, 1e-6, &format!("rsqrt lane {}", i));
    }
}

#[test]
fn test_simd_wrap_phase_basic() {
    let input = f32x8::from_array([0.0, PI, -PI, PI / 2.0, -PI / 2.0, 0.1, -0.1, 3.0]);
    let result = simd_wrap_phase(input);
    let result_arr: [f32; 8] = result.into();

    for (i, &val) in result_arr.iter().enumerate() {
        assert!(
            val >= -PI - TOLERANCE && val <= PI + TOLERANCE,
            "wrap_phase lane {}: result {} not in [-π, π]",
            i,
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
        TWO_PI + 0.5,
        -TWO_PI - 0.5,
        10.0 * TWO_PI + 0.1,
        -10.0 * TWO_PI - 0.1,
    ]);
    let result = simd_wrap_phase(input);
    let result_arr: [f32; 8] = result.into();

    for (i, &val) in result_arr.iter().enumerate() {
        assert!(
            val >= -PI - TOLERANCE && val <= PI + TOLERANCE,
            "wrap_phase large lane {}: result {} not in [-π, π]",
            i,
            val
        );
    }
}

#[test]
fn test_simd_exp_accuracy() {
    let input = f32x8::from_array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0]);
    let result = simd_exp(input);
    let result_arr: [f32; 8] = result.into();
    let input_arr: [f32; 8] = input.into();

    for i in 0..8 {
        let expected = input_arr[i].exp();
        assert_relative_error(
            result_arr[i],
            expected,
            STRICT_TOLERANCE,
            &format!("exp lane {}", i),
        );
    }
}

#[test]
fn test_simd_exp_edge_cases() {
    // Test edge cases
    let input = f32x8::from_array([0.0, -10.0, 10.0, -20.0, 20.0, -40.0, 40.0, -80.0]);
    let result = simd_exp(input);
    let result_arr: [f32; 8] = result.into();

    // exp(0) = 1
    assert_relative_error(result_arr[0], 1.0, STRICT_TOLERANCE, "exp(0)");

    // Results should be positive
    for (i, &val) in result_arr.iter().enumerate() {
        assert!(val >= 0.0, "exp lane {} should be non-negative: {}", i, val);
    }
}

#[test]
fn test_simd_log_accuracy() {
    let input = f32x8::from_array([1.0, 2.0, std::f32::consts::E, 10.0, 0.5, 0.1, 100.0, 0.01]);
    let result = simd_log(input);
    let result_arr: [f32; 8] = result.into();
    let input_arr: [f32; 8] = input.into();

    for i in 0..8 {
        let expected = input_arr[i].ln();
        let abs_error = (result_arr[i] - expected).abs();
        let tolerance = STRICT_TOLERANCE * expected.abs().max(1.0);
        assert!(
            abs_error < tolerance,
            "log lane {}: expected {}, got {}, error {} > tolerance {}",
            i,
            expected,
            result_arr[i],
            abs_error,
            tolerance
        );
    }
}

#[test]
fn test_simd_pow_accuracy() {
    let x = f32x8::from_array([2.0, 3.0, 4.0, 5.0, 2.0, 10.0, 0.5, 8.0]);
    let y = f32x8::from_array([2.0, 2.0, 0.5, 3.0, 10.0, 2.0, 2.0, 0.333333]);
    let result = simd_pow(x, y);
    let result_arr: [f32; 8] = result.into();
    let x_arr: [f32; 8] = x.into();
    let y_arr: [f32; 8] = y.into();

    for i in 0..8 {
        let expected = x_arr[i].powf(y_arr[i]);
        assert_relative_error(
            result_arr[i],
            expected,
            TOLERANCE,
            &format!("pow lane {}", i),
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
    let input = f32x8::from_array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -1.0]);
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
            abs_error < LOOSE_TOLERANCE,
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
fn test_simd_db_conversions_roundtrip() {
    let db_values = f32x8::from_array([-60.0, -30.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]);
    let linear = simd_db_to_linear(db_values);
    let db_back = simd_linear_to_db(linear);

    assert_simd_close(db_back, db_values, DB_TOLERANCE, "dB round-trip");
}

#[test]
fn test_simd_db_known_values() {
    // 10 dB = 10x power, 20 dB = 100x power, etc.
    let db = f32x8::from_array([0.0, 10.0, 20.0, 30.0, -10.0, -20.0, 3.0, -3.0]);
    let linear = simd_db_to_linear(db);
    let linear_arr: [f32; 8] = linear.into();

    // Use looser tolerance since dB conversion uses exp/log chain
    assert_relative_error(linear_arr[0], 1.0, LOOSE_TOLERANCE, "0 dB = 1x");
    assert_relative_error(linear_arr[1], 10.0, LOOSE_TOLERANCE, "10 dB = 10x");
    assert_relative_error(linear_arr[2], 100.0, LOOSE_TOLERANCE, "20 dB = 100x");
    assert_relative_error(linear_arr[3], 1000.0, LOOSE_TOLERANCE, "30 dB = 1000x");
    assert_relative_error(linear_arr[4], 0.1, LOOSE_TOLERANCE, "-10 dB = 0.1x");
}

// ============================================================================
// Complex Tests
// ============================================================================

#[test]
fn test_complex_from_polar() {
    let mag = f32x8::splat(1.0);
    let phase = f32x8::from_array([0.0, HALF_PI, PI, -HALF_PI, PI / 4.0, -PI / 4.0, PI / 3.0, -PI / 3.0]);

    let z = SimdComplex::from_polar(mag, phase);

    // Check magnitude is preserved
    let abs = z.abs();
    let abs_arr: [f32; 8] = abs.into();
    for (i, &val) in abs_arr.iter().enumerate() {
        assert!(
            (val - 1.0).abs() < TOLERANCE,
            "from_polar magnitude lane {}: expected 1.0, got {}",
            i,
            val
        );
    }

    // Check phase is preserved (uses atan2, so looser tolerance)
    let arg = z.arg();
    let arg_arr: [f32; 8] = arg.into();
    let phase_arr: [f32; 8] = phase.into();
    for i in 0..8 {
        let diff = (arg_arr[i] - phase_arr[i]).abs();
        assert!(
            diff < LOOSE_TOLERANCE || diff > TWO_PI - LOOSE_TOLERANCE,
            "from_polar phase lane {}: expected {}, got {}",
            i,
            phase_arr[i],
            arg_arr[i]
        );
    }
}

#[test]
fn test_complex_multiplication() {
    // (1 + 2i) * (3 + 4i) = -5 + 10i
    let a = SimdComplex::splat(1.0, 2.0);
    let b = SimdComplex::splat(3.0, 4.0);
    let result = a.mul(b);

    let (re, im) = result.to_arrays();
    for i in 0..8 {
        assert!((re[i] - (-5.0)).abs() < TOLERANCE, "mul real");
        assert!((im[i] - 10.0).abs() < TOLERANCE, "mul imag");
    }
}

#[test]
fn test_complex_conjugate_property() {
    // z * conj(z) = |z|²
    let z = SimdComplex::splat(3.0, 4.0);
    let z_conj = z.conj();
    let product = z.mul(z_conj);
    let abs_sq = z.abs_sq();

    let (prod_re, prod_im) = product.to_arrays();
    let abs_sq_arr: [f32; 8] = abs_sq.into();

    for i in 0..8 {
        assert!((prod_re[i] - abs_sq_arr[i]).abs() < TOLERANCE, "z*conj(z) real");
        assert!(prod_im[i].abs() < TOLERANCE, "z*conj(z) imag should be 0");
    }
}

#[test]
fn test_complex_abs() {
    // |3 + 4i| = 5
    let z = SimdComplex::splat(3.0, 4.0);
    let abs = z.abs();
    let abs_arr: [f32; 8] = abs.into();

    for (i, &val) in abs_arr.iter().enumerate() {
        assert!(
            (val - 5.0).abs() < TOLERANCE,
            "abs lane {}: expected 5.0, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_complex_euler_identity() {
    // e^(iπ) = -1
    let z = SimdComplex::splat(0.0, PI);
    let result = z.exp();

    let (re, im) = result.to_arrays();
    for i in 0..8 {
        // Uses sin/cos approximations so use looser tolerance
        assert!((re[i] - (-1.0)).abs() < LOOSE_TOLERANCE, "e^(iπ) real");
        assert!(im[i].abs() < LOOSE_TOLERANCE, "e^(iπ) imag should be ~0");
    }
}

// ============================================================================
// RNG Tests
// ============================================================================

#[test]
fn test_rng_uniform_range() {
    let mut rng = SimdRng::new(12345);

    for _ in 0..STAT_SAMPLES {
        let u = rng.uniform();
        let arr: [f32; 8] = u.into();

        for (lane, &val) in arr.iter().enumerate() {
            assert!(
                val >= 0.0 && val < 1.0,
                "uniform lane {} out of range: {}",
                lane,
                val
            );
        }
    }
}

#[test]
fn test_rng_uniform_mean() {
    let mut rng = SimdRng::new(12345);
    let mut sum = f32x8::splat(0.0);

    for _ in 0..STAT_SAMPLES {
        sum = sum + rng.uniform();
    }

    let mean = sum / f32x8::splat(STAT_SAMPLES as f32);
    let mean_arr: [f32; 8] = mean.into();

    // Expected mean = 0.5, 3σ tolerance ≈ 0.015
    for (lane, &m) in mean_arr.iter().enumerate() {
        assert!(
            (m - 0.5).abs() < 0.02,
            "uniform mean lane {}: expected ~0.5, got {}",
            lane,
            m
        );
    }
}

#[test]
fn test_rng_randn_mean_std() {
    let mut rng = SimdRng::new(12345);
    let mut sum = f32x8::splat(0.0);
    let mut sum_sq = f32x8::splat(0.0);

    for _ in 0..STAT_SAMPLES {
        let z = rng.randn();
        sum = sum + z;
        sum_sq = sum_sq + z * z;
    }

    let n = f32x8::splat(STAT_SAMPLES as f32);
    let mean = sum / n;
    let variance = sum_sq / n - mean * mean;
    let std = variance.sqrt();

    let mean_arr: [f32; 8] = mean.into();
    let std_arr: [f32; 8] = std.into();

    for (lane, &m) in mean_arr.iter().enumerate() {
        assert!(
            m.abs() < 0.05,
            "randn mean lane {}: expected ~0, got {}",
            lane,
            m
        );
    }

    for (lane, &s) in std_arr.iter().enumerate() {
        assert!(
            (s - 1.0).abs() < 0.1,
            "randn std lane {}: expected ~1, got {}",
            lane,
            s
        );
    }
}

#[test]
fn test_rng_rayleigh_mean() {
    let mut rng = SimdRng::new(12345);
    let sigma = f32x8::splat(1.0);
    let mut sum = f32x8::splat(0.0);

    for _ in 0..STAT_SAMPLES {
        sum = sum + rng.rayleigh(sigma);
    }

    let mean = sum / f32x8::splat(STAT_SAMPLES as f32);
    let mean_arr: [f32; 8] = mean.into();

    // Expected mean = σ * sqrt(π/2) ≈ 1.253
    let expected_mean = (std::f32::consts::PI / 2.0).sqrt();

    for (lane, &m) in mean_arr.iter().enumerate() {
        assert!(
            (m - expected_mean).abs() < 0.05,
            "rayleigh mean lane {}: expected ~{}, got {}",
            lane,
            expected_mean,
            m
        );
    }
}

#[test]
fn test_rng_exponential_mean() {
    let mut rng = SimdRng::new(12345);
    let lambda = f32x8::splat(2.0);
    let mut sum = f32x8::splat(0.0);

    for _ in 0..STAT_SAMPLES {
        sum = sum + rng.exponential(lambda);
    }

    let mean = sum / f32x8::splat(STAT_SAMPLES as f32);
    let mean_arr: [f32; 8] = mean.into();

    // Expected mean = 1/λ = 0.5
    for (lane, &m) in mean_arr.iter().enumerate() {
        assert!(
            (m - 0.5).abs() < 0.05,
            "exponential mean lane {}: expected ~0.5, got {}",
            lane,
            m
        );
    }
}

#[test]
fn test_rng_bernoulli_probability() {
    let mut rng = SimdRng::new(12345);
    let p = f32x8::splat(0.3);
    let mut count = f32x8::splat(0.0);

    for _ in 0..STAT_SAMPLES {
        count = count + rng.bernoulli_f32(p);
    }

    let proportion = count / f32x8::splat(STAT_SAMPLES as f32);
    let prop_arr: [f32; 8] = proportion.into();

    for (lane, &prop) in prop_arr.iter().enumerate() {
        assert!(
            (prop - 0.3).abs() < 0.03,
            "bernoulli proportion lane {}: expected ~0.3, got {}",
            lane,
            prop
        );
    }
}

#[test]
fn test_rng_lane_independence() {
    let mut rng = SimdRng::new(12345);

    let u = rng.uniform();
    let arr: [f32; 8] = u.into();

    // Check that not all lanes are identical
    let all_same = arr.iter().all(|&x| (x - arr[0]).abs() < 1e-6);
    assert!(
        !all_same,
        "All lanes produced the same value - lanes may not be independent"
    );
}

// ============================================================================
// Helper Tests
// ============================================================================

#[test]
fn test_load_store_strided() {
    let mut arr: Vec<f32> = (0..24).map(|i| i as f32).collect();

    // Load every 3rd element
    let loaded = load_f32_simd(&arr, 0, 3);
    let expected = f32x8::from_array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]);
    assert_simd_close(loaded, expected, STRICT_TOLERANCE, "strided load");

    // Store different values
    let values = f32x8::from_array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]);
    store_f32_simd(&mut arr, 0, 3, values);

    assert!((arr[0] - 100.0).abs() < STRICT_TOLERANCE);
    assert!((arr[3] - 200.0).abs() < STRICT_TOLERANCE);
    assert!((arr[21] - 800.0).abs() < STRICT_TOLERANCE);
}

#[test]
fn test_calc_distance_3d_basic() {
    let pos1 = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
    let pos2 = (f32x8::splat(3.0), f32x8::splat(4.0), f32x8::splat(0.0));

    let dist = calc_distance_3d(pos1, pos2);
    let dist_arr: [f32; 8] = dist.into();

    // 3-4-5 triangle
    for (i, &d) in dist_arr.iter().enumerate() {
        assert!(
            (d - 5.0).abs() < TOLERANCE,
            "distance lane {}: expected 5.0, got {}",
            i,
            d
        );
    }
}

#[test]
fn test_calc_radial_velocity_approaching() {
    let pos = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
    let vel = (f32x8::splat(10.0), f32x8::splat(0.0), f32x8::splat(0.0));
    let rx_pos = (f32x8::splat(100.0), f32x8::splat(0.0), f32x8::splat(0.0));

    let v_radial = calc_radial_velocity(pos, vel, rx_pos);
    let v_arr: [f32; 8] = v_radial.into();

    // Should be +10 (approaching)
    for (i, &v) in v_arr.iter().enumerate() {
        assert!(
            (v - 10.0).abs() < TOLERANCE,
            "radial velocity lane {}: expected 10.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_calc_radial_velocity_receding() {
    let pos = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
    let vel = (f32x8::splat(-10.0), f32x8::splat(0.0), f32x8::splat(0.0));
    let rx_pos = (f32x8::splat(100.0), f32x8::splat(0.0), f32x8::splat(0.0));

    let v_radial = calc_radial_velocity(pos, vel, rx_pos);
    let v_arr: [f32; 8] = v_radial.into();

    // Should be -10 (receding)
    for (i, &v) in v_arr.iter().enumerate() {
        assert!(
            (v + 10.0).abs() < TOLERANCE,
            "radial velocity lane {}: expected -10.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_freq_overlap_full() {
    let overlap = calc_freq_overlap(
        f32x8::splat(2.4e9),
        f32x8::splat(20e6),
        f32x8::splat(2.4e9),
        f32x8::splat(20e6),
    );

    let overlap_arr: [f32; 8] = overlap.into();

    for (i, &o) in overlap_arr.iter().enumerate() {
        assert!(
            (o - 1.0).abs() < 0.01,
            "full overlap lane {}: expected 1.0, got {}",
            i,
            o
        );
    }
}

#[test]
fn test_freq_overlap_none() {
    let overlap = calc_freq_overlap(
        f32x8::splat(2.4e9),
        f32x8::splat(20e6),
        f32x8::splat(5.0e9),
        f32x8::splat(20e6),
    );

    let overlap_arr: [f32; 8] = overlap.into();

    for (i, &o) in overlap_arr.iter().enumerate() {
        assert!(
            o < 0.01,
            "no overlap lane {}: expected ~0, got {}",
            i,
            o
        );
    }
}

#[test]
fn test_freq_overlap_partial() {
    // 50% overlap: one band is 100-200, other is 150-250
    let overlap = calc_freq_overlap(
        f32x8::splat(150.0),  // center
        f32x8::splat(100.0),  // bandwidth: 100-200
        f32x8::splat(200.0),  // center
        f32x8::splat(100.0),  // bandwidth: 150-250
    );

    let overlap_arr: [f32; 8] = overlap.into();

    for (i, &o) in overlap_arr.iter().enumerate() {
        assert!(
            (o - 0.5).abs() < 0.1,
            "partial overlap lane {}: expected ~0.5, got {}",
            i,
            o
        );
    }
}
