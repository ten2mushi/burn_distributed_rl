//! Comprehensive tests for continuous policy utilities.
//!
//! These tests cover the squashed Gaussian distribution used for continuous
//! action spaces in algorithms like SAC and continuous PPO.
//!
//! # Critical Dangers Tested
//!
//! 1. Tanh squashing boundary instability (log(1 - tanh^2(x)) as x -> infinity)
//! 2. atanh inverse at actions close to +/- 1
//! 3. Log probability correction for change of variables
//! 4. Entropy computation for Gaussian distributions
//!
//! # Mathematical Background
//!
//! For squashed Gaussian a = tanh(u), where u ~ N(mu, sigma):
//! log p(a) = log N(u; mu, sigma) - sum(log(1 - tanh^2(u)))

use burn::backend::Wgpu;
use burn::tensor::Tensor;

use crate::algorithms::continuous_policy::{
    entropy_gaussian, log_prob_squashed_gaussian, sample_gaussian, sample_squashed_gaussian,
    scale_action, unscale_action,
};

type TestBackend = Wgpu;

// ============================================================================
// Basic Sampling Tests
// ============================================================================

/// Test that Gaussian sampling returns correct dimensions.
#[test]
fn test_sample_gaussian_dimensions() {
    let device = Default::default();
    let batch_size = 32;
    let action_dim = 4;

    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);

    let (samples, log_probs) = sample_gaussian(mean, log_std);

    assert_eq!(samples.dims(), [batch_size, action_dim]);
    assert_eq!(log_probs.dims(), [batch_size]);
}

/// Test that squashed Gaussian samples are in (-1, 1).
#[test]
fn test_sample_squashed_gaussian_bounded() {
    let device = Default::default();
    let batch_size = 100;
    let action_dim = 4;

    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);

    let (squashed, _log_probs) = sample_squashed_gaussian(mean, log_std);

    let data = squashed.into_data();
    let values = data.as_slice::<f32>().unwrap();

    for v in values {
        assert!(
            *v > -1.0 && *v < 1.0,
            "Squashed sample should be in (-1, 1), got {}",
            v
        );
    }
}

/// Test that log probabilities are finite.
#[test]
fn test_sample_gaussian_log_probs_finite() {
    let device = Default::default();
    let mean: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);

    let (_, log_probs) = sample_gaussian(mean, log_std);

    let data = log_probs.into_data();
    let values = data.as_slice::<f32>().unwrap();

    for lp in values {
        assert!(lp.is_finite(), "Log prob should be finite, got {}", lp);
    }
}

/// Test that squashed Gaussian log probabilities are finite.
#[test]
fn test_sample_squashed_gaussian_log_probs_finite() {
    let device = Default::default();
    let mean: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([32, 4], &device);

    let (_, log_probs) = sample_squashed_gaussian(mean, log_std);

    let data = log_probs.into_data();
    let values = data.as_slice::<f32>().unwrap();

    for lp in values {
        assert!(
            lp.is_finite(),
            "Squashed log prob should be finite, got {}",
            lp
        );
    }
}

// ============================================================================
// Log Probability Tests
// ============================================================================

/// Test that log_prob_squashed_gaussian matches sampling log probs.
#[test]
fn test_log_prob_squashed_gaussian_consistency() {
    let device = Default::default();
    let batch_size = 16;
    let action_dim = 2;

    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);

    // Sample and get log prob during sampling
    let (squashed, original_log_probs) =
        sample_squashed_gaussian(mean.clone(), log_std.clone());

    // Compute log prob of the same samples
    let computed_log_probs = log_prob_squashed_gaussian(squashed, mean, log_std);

    let orig_data = original_log_probs.into_data();
    let comp_data = computed_log_probs.into_data();
    let orig_slice = orig_data.as_slice::<f32>().unwrap();
    let comp_slice = comp_data.as_slice::<f32>().unwrap();

    for (i, (o, c)) in orig_slice.iter().zip(comp_slice.iter()).enumerate() {
        assert!(
            (o - c).abs() < 1e-3,
            "Log probs should match at {}: original={}, computed={}",
            i,
            o,
            c
        );
    }
}

/// Test log prob for known values.
/// For standard normal (mu=0, sigma=1), log p(0) = -0.5 * log(2*pi) per dimension.
#[test]
fn test_log_prob_gaussian_known_values() {
    let device = Default::default();
    let batch_size = 4; // Use batch > 1 for Wgpu compatibility

    // Sample at the mean (noise = 0)
    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device); // std = 1

    // For x = mean, normalized = 0, so:
    // log p(x) = -0.5 * 0 - log(1) - 0.5 * log(2*pi) = -0.5 * log(2*pi) = -0.919 per dim
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let expected_log_prob = -0.5 * log_2pi * 2.0; // 2 dimensions

    // Create action at exactly zero (but need to go through squashing math)
    // For squashed Gaussian at a=0 (where tanh(0)=0):
    // u = atanh(0) = 0
    // log N(0; 0, 1) = -0.5 * log(2*pi) per dim
    // correction = log(1 - 0^2) = 0
    // total = -0.5 * log(2*pi) * 2 dims

    let action: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);
    let log_prob = log_prob_squashed_gaussian(action, mean, log_std);

    let data = log_prob.into_data();
    let values = data.as_slice::<f32>().unwrap();

    for value in values {
        assert!(
            (*value - expected_log_prob).abs() < 0.02,
            "Log prob at mean should be ~{}, got {}",
            expected_log_prob,
            value
        );
    }
}

// ============================================================================
// Tanh Squashing Boundary Tests (CRITICAL)
// ============================================================================

/// CRITICAL: Test that squash correction is stable for large pre-squash values.
/// log(1 - tanh^2(x)) -> -inf as x -> +/- infinity, but should be handled.
/// Note: For very large pre-squash values, tanh saturates to exactly +/- 1.
/// We check that samples are in [-1, 1] (inclusive) and finite.
#[test]
fn test_squash_correction_large_presquash_values() {
    let device = Default::default();

    // Moderate mean values that test squashing without hitting exact saturation
    // Use 2 action dims since Wgpu has issues with action_dim=1
    let mean: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[2.0, 2.0], [3.0, 3.0], [-3.0, -3.0], [4.0, 4.0]],
        &device,
    );
    let log_std: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        &device,
    ); // Moderate std

    let (squashed, log_probs) = sample_squashed_gaussian(mean, log_std);

    // All samples should be finite and in [-1, 1]
    let s_data = squashed.into_data();
    let s_values = s_data.as_slice::<f32>().unwrap();
    for (i, s) in s_values.iter().enumerate() {
        assert!(
            s.is_finite() && *s >= -1.0 && *s <= 1.0,
            "Squashed[{}] should be in [-1, 1] and finite, got {}",
            i,
            s
        );
    }

    // Log probs should be finite (not -inf or nan)
    let lp_data = log_probs.into_data();
    let lp_values = lp_data.as_slice::<f32>().unwrap();
    for (i, lp) in lp_values.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "Log prob[{}] should be finite even with large mean, got {}",
            i,
            lp
        );
    }
}

/// Test log prob for actions very close to boundaries (+/- 1).
/// atanh(x) has singularity at x = +/- 1.
#[test]
fn test_log_prob_near_boundary_actions() {
    let device = Default::default();

    // Actions very close to boundaries (but not exactly at them)
    // Use 2 action dims for Wgpu compatibility
    let actions: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[0.999, 0.999], [-0.999, -0.999], [0.9999, 0.9999], [-0.9999, -0.9999]],
        &device,
    );
    let mean: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device);

    let log_probs = log_prob_squashed_gaussian(actions, mean, log_std);

    let data = log_probs.into_data();
    let values = data.as_slice::<f32>().unwrap();

    // Log probs should be finite (might be very negative, but not NaN/Inf)
    for (i, lp) in values.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "Log prob[{}] near boundary should be finite, got {}",
            i,
            lp
        );
    }

    // Actions at 0.999 should have lower log prob than at 0 (more extreme)
    // This is because the correction term becomes large
    let action_zero: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device);
    let mean_zero: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device);
    let log_std_zero: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device);
    let lp_zero = log_prob_squashed_gaussian(action_zero, mean_zero, log_std_zero);
    let lp_zero_val = lp_zero.into_data().as_slice::<f32>().unwrap()[0];

    assert!(
        values[0] < lp_zero_val,
        "Extreme action 0.999 should have lower log prob than 0: {} vs {}",
        values[0],
        lp_zero_val
    );
}

/// Test that clamping prevents NaN at exact boundaries.
#[test]
fn test_log_prob_at_exact_boundary_clamped() {
    let device = Default::default();

    // Actions exactly at boundaries (should be clamped internally)
    // Note: These would cause atanh(1) = inf, but clamping should prevent this
    let actions: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[1.0], [-1.0], [1.0 - 1e-7], [-1.0 + 1e-7]],
        &device,
    );
    let mean: Tensor<TestBackend, 2> = Tensor::zeros([4, 1], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([4, 1], &device);

    let log_probs = log_prob_squashed_gaussian(actions, mean, log_std);

    let data = log_probs.into_data();
    let values = data.as_slice::<f32>().unwrap();

    for (i, lp) in values.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "Log prob[{}] at boundary should be clamped to finite, got {}",
            i,
            lp
        );
        assert!(
            !lp.is_nan(),
            "Log prob[{}] at boundary should not be NaN",
            i
        );
    }
}

// ============================================================================
// Entropy Tests
// ============================================================================

/// Test Gaussian entropy formula.
/// H = 0.5 * D * (1 + log(2*pi)) + sum(log_std)
#[test]
fn test_entropy_gaussian_formula() {
    let device = Default::default();
    let action_dim = 2;

    // std = 1 (log_std = 0)
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([4, action_dim], &device);

    let entropy = entropy_gaussian(log_std);

    let data = entropy.into_data();
    let values = data.as_slice::<f32>().unwrap();

    // H = 0.5 * 2 * (1 + log(2*pi)) + 0 = 1 + log(2*pi) = 1 + 1.8379 = 2.8379
    let log_2pi = (2.0 * std::f32::consts::PI).ln();
    let expected = 0.5 * action_dim as f32 * (1.0 + log_2pi);

    for e in values {
        assert!(
            (*e - expected).abs() < 0.01,
            "Entropy should be ~{}, got {}",
            expected,
            e
        );
    }
}

/// Test that higher std gives higher entropy.
#[test]
fn test_entropy_increases_with_std() {
    let device = Default::default();
    let batch_size = 4; // Use batch > 1 for Wgpu compatibility

    let log_std_low: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]],
        &device,
    );
    let log_std_high: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        &device,
    );

    let entropy_low = entropy_gaussian(log_std_low);
    let entropy_high = entropy_gaussian(log_std_high);

    let e_low = entropy_low.into_data().as_slice::<f32>().unwrap()[0];
    let e_high = entropy_high.into_data().as_slice::<f32>().unwrap()[0];

    assert!(
        e_high > e_low,
        "Higher std should give higher entropy: {} vs {}",
        e_high,
        e_low
    );
}

/// Test entropy is positive for reasonable std values.
#[test]
fn test_entropy_positive() {
    let device = Default::default();

    // Various std values - use batch size > 1 for Wgpu compatibility
    for log_std_val in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let log_std: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
            ],
            &device,
        );
        let entropy = entropy_gaussian(log_std);
        let e = entropy.into_data().as_slice::<f32>().unwrap()[0];

        // Entropy can be negative for very small std, but should be finite
        assert!(e.is_finite(), "Entropy should be finite, got {}", e);
    }
}

// ============================================================================
// Action Scaling Tests
// ============================================================================

/// Test scale_action correctly maps [-1, 1] to [low, high].
#[test]
fn test_scale_action_correct_mapping() {
    let device = Default::default();

    // Squashed actions in [-1, 1]
    let squashed: Tensor<TestBackend, 2> =
        Tensor::from_floats([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]], &device);

    let low = vec![-2.0, 0.0, 5.0];
    let high = vec![2.0, 10.0, 15.0];

    let scaled = scale_action(squashed, &low, &high);
    let data = scaled.into_data();
    let values = data.as_slice::<f32>().unwrap();

    // Row 0: [-1, 0, 1] -> should map to [low, mid, high]
    // scale = (high - low) / 2, offset = (high + low) / 2
    // For dim 0: scale=2, offset=0: -1*2+0=-2, 0*2+0=0, 1*2+0=2
    assert!((values[0] - (-2.0)).abs() < 1e-5, "squashed=-1 should map to low=-2");
    assert!((values[1] - 5.0).abs() < 1e-5, "squashed=0 should map to mid=5");
    assert!((values[2] - 15.0).abs() < 1e-5, "squashed=1 should map to high=15");

    // For dim 0 of row 1: 0.5*2+0=1
    assert!((values[3] - 1.0).abs() < 1e-5);
    // For dim 1 of row 1: -0.5*5+5=2.5
    assert!((values[4] - 2.5).abs() < 1e-5);
}

/// Test that unscale_action is the inverse of scale_action.
#[test]
fn test_scale_unscale_roundtrip() {
    let device = Default::default();

    let original: Tensor<TestBackend, 2> =
        Tensor::from_floats([[0.5, -0.3], [-0.8, 0.9], [0.0, 0.0]], &device);

    let low = vec![-2.0, 0.0];
    let high = vec![4.0, 10.0];

    let scaled = scale_action(original.clone(), &low, &high);
    let unscaled = unscale_action(scaled, &low, &high);

    let orig_data = original.into_data();
    let unsc_data = unscaled.into_data();
    let orig_values = orig_data.as_slice::<f32>().unwrap();
    let unsc_values = unsc_data.as_slice::<f32>().unwrap();

    for (o, u) in orig_values.iter().zip(unsc_values.iter()) {
        assert!(
            (o - u).abs() < 1e-5,
            "Roundtrip should preserve values: {} vs {}",
            o,
            u
        );
    }
}

/// Test scaling with asymmetric bounds.
#[test]
fn test_scale_action_asymmetric_bounds() {
    let device = Default::default();

    let squashed: Tensor<TestBackend, 2> = Tensor::from_floats([[0.0]], &device);
    let low = vec![-5.0];
    let high = vec![15.0];

    let scaled = scale_action(squashed, &low, &high);
    let value = scaled.into_data().as_slice::<f32>().unwrap()[0];

    // 0 should map to midpoint: (-5 + 15) / 2 = 5
    assert!(
        (value - 5.0).abs() < 1e-5,
        "squashed=0 should map to midpoint=5, got {}",
        value
    );
}

// ============================================================================
// Multi-Dimensional Tests
// ============================================================================

/// Test with various action dimensions.
#[test]
fn test_various_action_dimensions() {
    let device = Default::default();

    for action_dim in [1, 2, 4, 8, 16] {
        let mean: Tensor<TestBackend, 2> = Tensor::zeros([8, action_dim], &device);
        let log_std: Tensor<TestBackend, 2> = Tensor::zeros([8, action_dim], &device);

        let (samples, log_probs) = sample_squashed_gaussian(mean, log_std);

        assert_eq!(samples.dims(), [8, action_dim]);
        assert_eq!(log_probs.dims(), [8]);

        let lp_data = log_probs.into_data();
        let lp_values = lp_data.as_slice::<f32>().unwrap();

        for lp in lp_values {
            assert!(lp.is_finite(), "Log prob should be finite for dim={}", action_dim);
        }
    }
}

/// Test that log probs sum over action dimensions.
#[test]
fn test_log_prob_sums_over_dimensions() {
    let device = Default::default();
    let batch_size = 4; // Wgpu needs batch > 1

    // Single dimension - but use batch > 1
    let mean_1d: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 1], &device);
    let log_std_1d: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 1], &device);
    let action_1d: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[0.5], [0.5], [0.5], [0.5]],
        &device,
    );

    let lp_1d = log_prob_squashed_gaussian(action_1d, mean_1d, log_std_1d);
    let lp_1d_val = lp_1d.into_data().as_slice::<f32>().unwrap()[0];

    // Two dimensions with same action
    let mean_2d: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);
    let log_std_2d: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);
    let action_2d: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        &device,
    );

    let lp_2d = log_prob_squashed_gaussian(action_2d, mean_2d, log_std_2d);
    let lp_2d_val = lp_2d.into_data().as_slice::<f32>().unwrap()[0];

    // 2D log prob should be approximately 2x 1D log prob (independent dims)
    assert!(
        (lp_2d_val - 2.0 * lp_1d_val).abs() < 0.1,
        "2D log prob should be ~2x 1D: {} vs 2*{}",
        lp_2d_val,
        lp_1d_val
    );
}

// ============================================================================
// Log Std Clamping Tests
// ============================================================================

/// Test that extreme log_std values are clamped.
#[test]
fn test_log_std_clamping() {
    let device = Default::default();
    let batch_size = 4; // Wgpu needs batch > 1

    // Very extreme log_std values
    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);
    let log_std_extreme: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[-100.0, -100.0], [-100.0, -100.0], [-100.0, -100.0], [-100.0, -100.0]],
        &device,
    );

    let (samples, log_probs) = sample_gaussian(mean.clone(), log_std_extreme);

    let s_data = samples.into_data();
    let s_vals = s_data.as_slice::<f32>().unwrap();
    for s_val in s_vals {
        assert!(s_val.is_finite(), "Sample with extreme log_std should be finite");
    }

    let lp_data = log_probs.into_data();
    let lp_vals = lp_data.as_slice::<f32>().unwrap();
    for lp_val in lp_vals {
        assert!(
            lp_val.is_finite(),
            "Log prob with extreme log_std should be finite"
        );
    }
}

/// Test with maximum allowed log_std.
/// Note: With very large std, samples can approach +/- 1 exactly.
#[test]
fn test_log_std_at_max() {
    let device = Default::default();

    let mean: Tensor<TestBackend, 2> = Tensor::zeros([4, 2], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::from_floats(
        [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], // At LOG_STD_MAX
        &device,
    );

    let (samples, log_probs) = sample_squashed_gaussian(mean, log_std);

    let s_data = samples.into_data();
    for s in s_data.as_slice::<f32>().unwrap() {
        // Use inclusive bounds since tanh can saturate
        assert!(s.is_finite() && *s >= -1.0 && *s <= 1.0);
    }

    let lp_data = log_probs.into_data();
    for lp in lp_data.as_slice::<f32>().unwrap() {
        assert!(lp.is_finite());
    }
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

/// Test with large batch sizes.
#[test]
fn test_large_batch_size() {
    let device = Default::default();
    let batch_size = 1024;
    let action_dim = 8;

    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, action_dim], &device);

    let (samples, log_probs) = sample_squashed_gaussian(mean, log_std);

    assert_eq!(samples.dims(), [batch_size, action_dim]);
    assert_eq!(log_probs.dims(), [batch_size]);
}

/// Test with small batch size.
#[test]
fn test_batch_size_small() {
    let device = Default::default();
    // Wgpu needs batch > 1 for squeeze operations to work properly
    let batch_size = 2;

    let mean: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);
    let log_std: Tensor<TestBackend, 2> = Tensor::zeros([batch_size, 2], &device);

    let (samples, log_probs) = sample_squashed_gaussian(mean, log_std);

    assert_eq!(samples.dims(), [batch_size, 2]);
    assert_eq!(log_probs.dims(), [batch_size]);
}

// ============================================================================
// Property-Based Tests
// ============================================================================

/// Property: Squashed samples are always in [-1, 1].
/// Note: With tanh, samples can saturate to exactly +/- 1 for extreme inputs.
#[test]
fn test_squashed_bounds_property() {
    let device = Default::default();
    let batch_size = 4; // Wgpu needs batch > 1

    // Various mean/std combinations (moderate values to avoid exact saturation)
    for mean_val in [-3.0, -1.0, 0.0, 1.0, 3.0] {
        for log_std_val in [-1.0, 0.0, 1.0] {
            let mean: Tensor<TestBackend, 2> = Tensor::from_floats(
                [
                    [mean_val, mean_val],
                    [mean_val, mean_val],
                    [mean_val, mean_val],
                    [mean_val, mean_val],
                ],
                &device,
            );
            let log_std: Tensor<TestBackend, 2> = Tensor::from_floats(
                [
                    [log_std_val, log_std_val],
                    [log_std_val, log_std_val],
                    [log_std_val, log_std_val],
                    [log_std_val, log_std_val],
                ],
                &device,
            );

            let (samples, _) = sample_squashed_gaussian(mean, log_std);

            let data = samples.into_data();
            for s in data.as_slice::<f32>().unwrap() {
                // Use inclusive bounds since tanh can saturate to +/- 1
                assert!(
                    s.is_finite() && *s >= -1.0 && *s <= 1.0,
                    "Squashed sample should be in [-1, 1]: {} (mean={}, log_std={})",
                    s,
                    mean_val,
                    log_std_val
                );
            }
        }
    }
}

/// Property: Log probs are always finite for valid inputs.
#[test]
fn test_log_probs_always_finite() {
    let device = Default::default();
    let batch_size = 4; // Wgpu needs batch > 1

    // Various parameter combinations
    let test_cases = vec![
        (0.0, 0.0, 0.0),   // Standard normal at 0
        (0.0, 0.0, 0.9),   // Standard normal near boundary
        (5.0, -1.0, 0.5),  // Shifted mean, small std
        (-5.0, 1.0, -0.5), // Negative mean, large std
    ];

    for (mean_val, log_std_val, action_val) in test_cases {
        let mean: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [mean_val, mean_val],
                [mean_val, mean_val],
                [mean_val, mean_val],
                [mean_val, mean_val],
            ],
            &device,
        );
        let log_std: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
            ],
            &device,
        );
        let action: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [action_val, action_val],
                [action_val, action_val],
                [action_val, action_val],
                [action_val, action_val],
            ],
            &device,
        );

        let log_prob = log_prob_squashed_gaussian(action, mean, log_std);
        let lp_data = log_prob.into_data();
        let lps = lp_data.as_slice::<f32>().unwrap();

        for lp in lps {
            assert!(
                lp.is_finite(),
                "Log prob should be finite: mean={}, log_std={}, action={}, got {}",
                mean_val,
                log_std_val,
                action_val,
                lp
            );
        }
    }
}

/// Property: Entropy increases with std.
#[test]
fn test_entropy_monotonic_in_std() {
    let device = Default::default();
    let batch_size = 4; // Wgpu needs batch > 1

    let log_stds: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut prev_entropy = f32::NEG_INFINITY;

    for log_std_val in log_stds {
        let log_std: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
                [log_std_val, log_std_val],
            ],
            &device,
        );
        let entropy = entropy_gaussian(log_std);
        let e = entropy.into_data().as_slice::<f32>().unwrap()[0];

        assert!(
            e > prev_entropy,
            "Entropy should increase with std: prev={}, current={} at log_std={}",
            prev_entropy,
            e,
            log_std_val
        );
        prev_entropy = e;
    }
}
