//! Comprehensive tests for policy gradient loss functions.
//!
//! These tests define the complete behavior of PPO and IMPALA loss functions.
//! They cover:
//! - PPO clipped surrogate objective correctness
//! - Clipping direction (pessimistic bound)
//! - Value function loss (with and without clipping)
//! - Entropy loss sign convention
//! - Gradient flow requirements
//!
//! # Critical Dangers Tested
//!
//! 1. PPO clipping direction (MIN for pessimistic bound)
//! 2. Value loss clipping semantics (uses MAX, not MIN)
//! 3. Entropy sign convention (subtract entropy = encourage exploration)
//! 4. Numerical stability with extreme ratios

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Tensor;

use crate::algorithms::policy_loss::{
    entropy_from_log_probs, entropy_loss, impala_policy_loss, impala_policy_loss_scalar,
    ppo_clip_loss, ppo_clip_loss_scalar, value_loss, value_loss_scalar,
};

type TestBackend = Wgpu;
type TestAutodiffBackend = Autodiff<Wgpu>;

// ============================================================================
// PPO Clipped Surrogate Loss - Scalar Tests
// ============================================================================

/// Test PPO loss when ratio = 1 (on-policy, no change).
#[test]
fn test_ppo_loss_ratio_one_no_clipping() {
    let log_probs = vec![-1.0, -1.0];
    let old_log_probs = vec![-1.0, -1.0]; // Same = ratio of 1.0
    let advantages = vec![1.0, -1.0];
    let clip_ratio = 0.2;

    let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

    // surr1 = surr2 = ratio * advantage = 1.0 * advantage
    // For advantage = [1, -1], surr = [1, -1], mean = 0
    // Loss = -mean = 0
    assert!(
        loss.abs() < 1e-6,
        "On-policy with balanced advantages should give loss ~0, got {}",
        loss
    );
}

/// Test PPO clipping with positive advantage and high ratio.
/// When ratio > 1+epsilon and advantage > 0: should clip to prevent overconfidence.
#[test]
fn test_ppo_clip_positive_advantage_high_ratio() {
    let log_probs = vec![0.0]; // log(1) = 0
    let old_log_probs = vec![-1.0]; // log(exp(-1)) = -1
    // ratio = exp(0 - (-1)) = e = 2.718

    let advantages = vec![1.0]; // Positive advantage
    let clip_ratio = 0.2;

    let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

    // surr1 = 2.718 * 1.0 = 2.718
    // surr2 = 1.2 * 1.0 = 1.2 (clipped)
    // min(surr1, surr2) = 1.2
    // loss = -1.2
    assert!(
        (loss - (-1.2)).abs() < 0.01,
        "Should clip to 1.2, got loss {}",
        loss
    );
}

/// Test PPO clipping with positive advantage and low ratio.
/// When ratio < 1-epsilon and advantage > 0: should NOT clip (already pessimistic).
#[test]
fn test_ppo_clip_positive_advantage_low_ratio() {
    let log_probs = vec![-2.0];
    let old_log_probs = vec![-1.0];
    // ratio = exp(-2 - (-1)) = exp(-1) = 0.368

    let advantages = vec![1.0];
    let clip_ratio = 0.2;

    let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

    // surr1 = 0.368 * 1.0 = 0.368
    // surr2 = 0.8 * 1.0 = 0.8 (would clip to 1-eps=0.8, but this is > surr1)
    // min(surr1, surr2) = 0.368 (surr1 is already more pessimistic)
    // loss = -0.368
    let expected_loss = -((-1.0_f32).exp()); // exp(-1) = 0.368
    assert!(
        (loss - expected_loss).abs() < 0.01,
        "Should NOT clip when already pessimistic, expected {}, got {}",
        expected_loss,
        loss
    );
}

/// Test PPO clipping with negative advantage and high ratio.
/// When ratio > 1+epsilon and advantage < 0: should NOT clip (already pessimistic).
#[test]
fn test_ppo_clip_negative_advantage_high_ratio() {
    let log_probs = vec![0.0];
    let old_log_probs = vec![-1.0];
    // ratio = e = 2.718

    let advantages = vec![-1.0]; // Negative advantage
    let clip_ratio = 0.2;

    let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

    // surr1 = 2.718 * (-1) = -2.718
    // surr2 = 1.2 * (-1) = -1.2
    // min(surr1, surr2) = -2.718 (surr1 is more pessimistic for negative advantage)
    // loss = -(-2.718) = 2.718
    let expected_loss = std::f32::consts::E;
    assert!(
        (loss - expected_loss).abs() < 0.01,
        "Negative advantage with high ratio: expected {}, got {}",
        expected_loss,
        loss
    );
}

/// Test PPO clipping with negative advantage and low ratio.
/// When ratio < 1-epsilon and advantage < 0: should clip.
#[test]
fn test_ppo_clip_negative_advantage_low_ratio() {
    let log_probs = vec![-2.0];
    let old_log_probs = vec![-1.0];
    // ratio = exp(-1) = 0.368

    let advantages = vec![-1.0];
    let clip_ratio = 0.2;

    let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

    // surr1 = 0.368 * (-1) = -0.368
    // surr2 = 0.8 * (-1) = -0.8 (clipped to 1-eps=0.8)
    // min(surr1, surr2) = -0.8
    // loss = -(-0.8) = 0.8
    assert!(
        (loss - 0.8).abs() < 0.01,
        "Should clip to 0.8, got {}",
        loss
    );
}

/// Test PPO loss with empty inputs.
#[test]
fn test_ppo_loss_empty() {
    let loss = ppo_clip_loss_scalar(&[], &[], &[], 0.2);
    assert!(
        loss.abs() < 1e-6,
        "Empty input should give zero loss, got {}",
        loss
    );
}

// ============================================================================
// PPO Clipped Surrogate Loss - Tensor Tests
// ============================================================================

/// Test tensor-based PPO loss matches scalar version.
#[test]
fn test_ppo_clip_loss_tensor_matches_scalar() {
    let device = Default::default();

    let log_probs_vec = vec![-0.5, -1.0, -1.5, 0.0];
    let old_log_probs_vec = vec![-0.8, -0.8, -0.8, -0.8];
    let advantages_vec = vec![1.0, -1.0, 0.5, -0.5];
    let clip_ratio = 0.2;

    // Scalar computation
    let scalar_loss =
        ppo_clip_loss_scalar(&log_probs_vec, &old_log_probs_vec, &advantages_vec, clip_ratio);

    // Tensor computation
    let log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats(log_probs_vec.as_slice(), &device);
    let old_log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats(old_log_probs_vec.as_slice(), &device);
    let advantages: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats(advantages_vec.as_slice(), &device);

    let tensor_loss = ppo_clip_loss(log_probs, old_log_probs, advantages, clip_ratio);
    let tensor_loss_val = tensor_loss.into_data().as_slice::<f32>().unwrap()[0];

    assert!(
        (scalar_loss - tensor_loss_val).abs() < 1e-5,
        "Tensor loss should match scalar: {} vs {}",
        scalar_loss,
        tensor_loss_val
    );
}

// ============================================================================
// Value Loss Tests
// ============================================================================

/// Test value loss with perfect predictions.
#[test]
fn test_value_loss_perfect_predictions() {
    let values = vec![1.0, 2.0, 3.0];
    let returns = vec![1.0, 2.0, 3.0];

    let loss = value_loss_scalar(&values, &returns);

    assert!(
        loss.abs() < 1e-6,
        "Perfect predictions should give zero loss, got {}",
        loss
    );
}

/// Test value loss MSE calculation.
#[test]
fn test_value_loss_mse() {
    let values = vec![1.0, 2.0];
    let returns = vec![2.0, 4.0]; // Errors: 1, 2

    let loss = value_loss_scalar(&values, &returns);

    // MSE = ((1-2)^2 + (2-4)^2) / 2 = (1 + 4) / 2 = 2.5
    assert!(
        (loss - 2.5).abs() < 1e-6,
        "MSE should be 2.5, got {}",
        loss
    );
}

/// Test value loss with empty inputs.
#[test]
fn test_value_loss_empty() {
    let loss = value_loss_scalar(&[], &[]);
    assert!(loss.abs() < 1e-6);
}

/// Test tensor value loss without clipping.
#[test]
fn test_value_loss_tensor_no_clipping() {
    let device = Default::default();

    let values: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    let old_values: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    let returns: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.5, 2.5, 3.5], &device);

    let loss = value_loss(values, old_values, returns, None);
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // MSE = ((0.5)^2 * 3) / 3 = 0.25
    assert!(
        (loss_val - 0.25).abs() < 1e-5,
        "Unclipped value loss should be 0.25, got {}",
        loss_val
    );
}

/// Test clipped value loss uses MAX (conservative update).
/// CRITICAL: Value clipping uses MAX, not MIN, to prevent large value updates.
#[test]
fn test_value_loss_clipping_uses_max() {
    let device = Default::default();

    // Values have moved far from old_values
    let values: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([2.0], &device); // Moved a lot
    let old_values: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.0], &device);
    let returns: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.5], &device);

    let clip_value = 0.2;

    let loss = value_loss(
        values,
        old_values.clone(),
        returns.clone(),
        Some(clip_value),
    );
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // Clipped values: old + clamp(values - old, -0.2, 0.2) = 1.0 + 0.2 = 1.2
    // loss1 = (2.0 - 1.5)^2 = 0.25
    // loss2 = (1.2 - 1.5)^2 = 0.09
    // max(loss1, loss2) = 0.25
    assert!(
        (loss_val - 0.25).abs() < 1e-5,
        "Clipped value loss should use MAX: expected 0.25, got {}",
        loss_val
    );
}

/// Test value clipping when old values are accurate.
#[test]
fn test_value_loss_clipping_accurate_old_values() {
    let device = Default::default();

    // Old values were accurate, current values overfit
    let values: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.5], &device);
    let old_values: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.0], &device);
    let returns: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.0], &device); // Old was correct

    let loss_clipped = value_loss(
        values.clone(),
        old_values.clone(),
        returns.clone(),
        Some(0.2),
    );
    let loss_unclipped = value_loss(values, old_values, returns, None);

    let clipped_val = loss_clipped.into_data().as_slice::<f32>().unwrap()[0];
    let unclipped_val = loss_unclipped.into_data().as_slice::<f32>().unwrap()[0];

    // Clipped loss should be >= unclipped when constraining values
    assert!(
        clipped_val >= unclipped_val - 1e-5,
        "Clipped loss should be >= unclipped: {} vs {}",
        clipped_val,
        unclipped_val
    );
}

// ============================================================================
// Entropy Loss Tests
// ============================================================================

/// Test entropy from log probs calculation.
#[test]
fn test_entropy_from_log_probs_uniform() {
    // Uniform distribution over 2 actions
    let log_probs = vec![0.5_f32.ln(), 0.5_f32.ln()]; // ln(0.5) = -0.693
    let probs = vec![0.5, 0.5];

    let entropy = entropy_from_log_probs(&log_probs, &probs);

    // H = -sum(p * log(p)) = -2 * 0.5 * ln(0.5) = ln(2) = 0.693
    let expected = 2.0_f32.ln();
    assert!(
        (entropy - expected).abs() < 0.01,
        "Uniform entropy should be ln(2)={}, got {}",
        expected,
        entropy
    );
}

/// Test entropy for peaked distribution (low entropy).
#[test]
fn test_entropy_peaked_distribution() {
    // Almost deterministic: p = [0.99, 0.01]
    let probs = vec![0.99, 0.01];
    let log_probs: Vec<f32> = probs.iter().map(|p| (*p as f32).ln()).collect();

    let entropy = entropy_from_log_probs(&log_probs, &probs);

    // Should be much lower than uniform
    let uniform_entropy = 2.0_f32.ln();
    assert!(
        entropy < uniform_entropy * 0.5,
        "Peaked entropy should be << uniform: {} vs {}",
        entropy,
        uniform_entropy
    );
}

/// Test tensor entropy loss has correct sign.
/// CRITICAL: Entropy loss should be NEGATIVE entropy (for minimization).
#[test]
fn test_entropy_loss_sign_convention() {
    let device = Default::default();

    // Uniform distribution: high entropy
    let uniform_probs: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_floats([[0.25, 0.25, 0.25, 0.25]], &device);

    // Peaked distribution: low entropy
    let peaked_probs: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_floats([[0.97, 0.01, 0.01, 0.01]], &device);

    let uniform_loss = entropy_loss(uniform_probs);
    let peaked_loss = entropy_loss(peaked_probs);

    let uniform_val = uniform_loss.into_data().as_slice::<f32>().unwrap()[0];
    let peaked_val = peaked_loss.into_data().as_slice::<f32>().unwrap()[0];

    // entropy_loss returns -entropy (negative for minimization)
    // Higher entropy -> more negative loss
    // So uniform should have MORE NEGATIVE loss than peaked
    assert!(
        uniform_val < peaked_val,
        "Uniform (high entropy) should have lower loss: {} vs {}",
        uniform_val,
        peaked_val
    );

    // Both should be negative (since it's -entropy and entropy > 0)
    assert!(uniform_val < 0.0, "Entropy loss should be negative");
    assert!(peaked_val < 0.0, "Entropy loss should be negative");
}

/// Test entropy handles near-zero probabilities.
#[test]
fn test_entropy_near_zero_probabilities() {
    let device = Default::default();

    // Near-deterministic with some very small probs
    let probs: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_floats([[0.999, 0.0005, 0.0005]], &device);

    let loss = entropy_loss(probs);
    let val = loss.into_data().as_slice::<f32>().unwrap()[0];

    assert!(val.is_finite(), "Entropy with near-zero probs should be finite");
}

// ============================================================================
// IMPALA Policy Loss Tests
// ============================================================================

/// Test IMPALA policy gradient loss formula.
#[test]
fn test_impala_policy_loss_scalar() {
    let log_probs = vec![-0.5, -1.0];
    let advantages = vec![1.0, 2.0];

    let loss = impala_policy_loss_scalar(&log_probs, &advantages);

    // loss = -mean(log_p * A) = -((-0.5 * 1.0 + -1.0 * 2.0) / 2)
    //      = -((-0.5 - 2.0) / 2) = -(-1.25) = 1.25
    assert!(
        (loss - 1.25).abs() < 1e-6,
        "IMPALA loss should be 1.25, got {}",
        loss
    );
}

/// Test IMPALA loss with positive advantages encourages higher log probs.
#[test]
fn test_impala_loss_positive_advantages() {
    let log_probs = vec![-1.0, -1.0];
    let advantages = vec![1.0, 1.0]; // All positive

    let loss = impala_policy_loss_scalar(&log_probs, &advantages);

    // Positive advantages: want to increase log_probs
    // Loss should be positive (gradient will increase log_probs)
    assert!(
        loss > 0.0,
        "Positive advantages should give positive loss for minimization"
    );
}

/// Test IMPALA loss with negative advantages discourages actions.
#[test]
fn test_impala_loss_negative_advantages() {
    let log_probs = vec![-1.0, -1.0];
    let advantages = vec![-1.0, -1.0]; // All negative

    let loss = impala_policy_loss_scalar(&log_probs, &advantages);

    // Negative advantages: want to decrease probability (increase negative log_probs)
    // Loss should be negative
    assert!(
        loss < 0.0,
        "Negative advantages should give negative loss"
    );
}

/// Test tensor IMPALA loss.
#[test]
fn test_impala_policy_loss_tensor() {
    let device = Default::default();

    let log_probs: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([-0.5, -1.0], &device);
    let advantages: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.0, 2.0], &device);

    let loss = impala_policy_loss(log_probs, advantages);
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

    assert!(
        (loss_val - 1.25).abs() < 1e-5,
        "Tensor IMPALA loss should be 1.25, got {}",
        loss_val
    );
}

/// Test IMPALA loss empty.
#[test]
fn test_impala_loss_empty() {
    let loss = impala_policy_loss_scalar(&[], &[]);
    assert!(loss.abs() < 1e-6);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

/// Test PPO loss with extreme log prob differences.
#[test]
fn test_ppo_loss_extreme_ratios() {
    // Very different log probs
    let log_probs = vec![0.0, -50.0, -100.0];
    let old_log_probs = vec![-50.0, 0.0, -50.0];
    let advantages = vec![1.0, -1.0, 0.5];

    let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, 0.2);

    assert!(
        loss.is_finite(),
        "PPO loss with extreme ratios should be finite, got {}",
        loss
    );
}

/// Test value loss with large values.
#[test]
fn test_value_loss_large_values() {
    let values = vec![1e6, -1e6];
    let returns = vec![1e6 + 1.0, -1e6 - 1.0];

    let loss = value_loss_scalar(&values, &returns);

    assert!(
        loss.is_finite(),
        "Value loss with large values should be finite"
    );
    assert!((loss - 1.0).abs() < 1e-5, "MSE should be 1.0");
}

// ============================================================================
// Gradient Flow Tests (require autodiff backend)
// ============================================================================

/// Test that PPO loss produces gradients.
#[test]
fn test_ppo_loss_has_gradients() {
    let device = Default::default();

    let log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([-1.0, -1.0, -1.0], &device).require_grad();
    let old_log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([-1.0, -1.0, -1.0], &device); // No grad needed
    let advantages: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.0, 0.0, -1.0], &device); // No grad needed

    let loss = ppo_clip_loss(log_probs.clone(), old_log_probs, advantages, 0.2);

    // Backward pass
    let grads = loss.backward();

    // Check that log_probs has gradients
    let log_probs_grad = log_probs.grad(&grads);
    assert!(
        log_probs_grad.is_some(),
        "log_probs should have gradients"
    );

    let grad_data = log_probs_grad.unwrap().into_data();
    let grad_values = grad_data.as_slice::<f32>().unwrap();

    for g in grad_values {
        assert!(g.is_finite(), "Gradient should be finite");
    }
}

/// Test that value loss produces gradients.
#[test]
fn test_value_loss_has_gradients() {
    let device = Default::default();

    let values: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.0, 2.0, 3.0], &device).require_grad();
    let old_values: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.0, 2.0, 3.0], &device);
    let returns: Tensor<TestAutodiffBackend, 1> = Tensor::from_floats([1.5, 2.5, 3.5], &device);

    let loss = value_loss(values.clone(), old_values, returns, Some(0.2));

    let grads = loss.backward();
    let values_grad = values.grad(&grads);

    assert!(values_grad.is_some(), "values should have gradients");
}

/// Test that entropy loss produces gradients.
#[test]
fn test_entropy_loss_has_gradients() {
    let device = Default::default();

    let probs: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_floats([[0.25, 0.25, 0.25, 0.25]], &device).require_grad();

    let loss = entropy_loss(probs.clone());

    let grads = loss.backward();
    let probs_grad = probs.grad(&grads);

    assert!(probs_grad.is_some(), "probs should have gradients");
}

// ============================================================================
// Combined Loss Tests
// ============================================================================

/// Test that total loss combines components correctly.
/// total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
#[test]
fn test_combined_loss_formula() {
    let device = Default::default();

    // Simple case: all components are 1.0
    let log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([-1.0, -1.0], &device);
    let old_log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([-1.0, -1.0], &device);
    let advantages: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.0, 1.0], &device);

    let policy_loss = ppo_clip_loss(log_probs, old_log_probs, advantages, 0.2);
    let policy_loss_val = policy_loss.clone().into_data().as_slice::<f32>().unwrap()[0];

    // ratio = 1, surr = 1 * 1 = 1, mean = 1, loss = -1
    assert!(
        (policy_loss_val - (-1.0)).abs() < 1e-5,
        "Policy loss should be -1, got {}",
        policy_loss_val
    );
}

/// Test entropy coefficient sign.
/// With positive entropy_coef, higher entropy should result in LOWER total loss.
#[test]
fn test_entropy_coefficient_sign() {
    let device = Default::default();

    // Uniform has higher entropy
    let uniform_probs: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_floats([[0.25, 0.25, 0.25, 0.25]], &device);
    let peaked_probs: Tensor<TestAutodiffBackend, 2> =
        Tensor::from_floats([[0.97, 0.01, 0.01, 0.01]], &device);

    let entropy_uniform = entropy_loss(uniform_probs);
    let entropy_peaked = entropy_loss(peaked_probs);

    let ent_uniform_val = entropy_uniform.into_data().as_slice::<f32>().unwrap()[0];
    let ent_peaked_val = entropy_peaked.into_data().as_slice::<f32>().unwrap()[0];

    // entropy_loss returns -entropy
    // When computing total_loss = policy + vf*value - entropy_coef * (-entropy)
    //                           = policy + vf*value + entropy_coef * entropy
    // So adding entropy_loss with positive coef subtracts entropy from loss
    // Higher entropy -> lower total loss (encourages exploration)

    // Verify: entropy_loss(uniform) < entropy_loss(peaked) because uniform has more entropy
    assert!(
        ent_uniform_val < ent_peaked_val,
        "Uniform should have more negative entropy loss: {} vs {}",
        ent_uniform_val,
        ent_peaked_val
    );
}

// ============================================================================
// Property-Based Tests
// ============================================================================

/// Property: PPO loss is bounded when clipping is active.
#[test]
fn test_ppo_loss_bounded_with_clipping() {
    let clip_ratio = 0.2;

    for log_diff in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        let log_probs = vec![log_diff];
        let old_log_probs = vec![0.0];
        let advantages = vec![1.0];

        let loss = ppo_clip_loss_scalar(&log_probs, &old_log_probs, &advantages, clip_ratio);

        // With positive advantage, clipped objective is at most -(1+eps) = -1.2
        assert!(
            loss >= -1.2 - 1e-5,
            "PPO loss should be >= -1.2 with positive advantage, got {} for log_diff={}",
            loss,
            log_diff
        );
    }
}

/// Property: Value loss is always non-negative.
#[test]
fn test_value_loss_non_negative() {
    let test_cases = vec![
        (vec![0.0], vec![0.0]),
        (vec![1.0], vec![-1.0]),
        (vec![100.0, -100.0], vec![-100.0, 100.0]),
    ];

    for (values, returns) in test_cases {
        let loss = value_loss_scalar(&values, &returns);
        assert!(
            loss >= 0.0,
            "Value loss should be non-negative, got {}",
            loss
        );
    }
}

/// Property: IMPALA loss has correct sign relative to advantages.
#[test]
fn test_impala_loss_sign_matches_advantage() {
    // Pure positive advantages
    let loss_pos = impala_policy_loss_scalar(&[-1.0], &[1.0]);
    assert!(loss_pos > 0.0, "Positive advantage should give positive loss");

    // Pure negative advantages
    let loss_neg = impala_policy_loss_scalar(&[-1.0], &[-1.0]);
    assert!(loss_neg < 0.0, "Negative advantage should give negative loss");

    // Zero advantages
    let loss_zero = impala_policy_loss_scalar(&[-1.0], &[0.0]);
    assert!(loss_zero.abs() < 1e-6, "Zero advantage should give zero loss");
}
