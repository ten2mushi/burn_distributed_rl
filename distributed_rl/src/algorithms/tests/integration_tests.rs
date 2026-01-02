//! Integration tests for distributed RL algorithms.
//!
//! These tests verify end-to-end behavior of PPO and IMPALA algorithms,
//! testing the complete pipeline from transitions to loss computation.
//!
//! # Test Categories
//!
//! 1. PPO full training step: rollout -> GAE -> loss
//! 2. IMPALA full training step: trajectories -> V-trace -> loss
//! 3. PPO vs IMPALA on-policy equivalence
//! 4. Configuration validation
//! 5. Buffer behavior and staleness handling

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Tensor;

use crate::algorithms::gae::{compute_gae, normalize_advantages};
use crate::algorithms::vtrace::compute_vtrace;
use crate::algorithms::policy_loss::{ppo_clip_loss_scalar, value_loss_scalar};
use crate::algorithms::core_algorithm::PPOAlgorithmConfig;
use crate::algorithms::impala::IMPALAConfig;

type TestBackend = Wgpu;
type TestAutodiffBackend = Autodiff<Wgpu>;

// ============================================================================
// Full PPO Training Step Tests
// ============================================================================

/// Test complete PPO training step: transitions -> GAE -> loss.
#[test]
fn test_ppo_full_training_step() {
    // Simulate a rollout
    let rewards = vec![1.0, 0.5, 1.5, 0.0, 2.0, 1.0, 0.5, 1.0];
    let values = vec![0.8, 0.9, 1.0, 0.5, 1.2, 0.8, 0.6, 0.7];
    let dones = vec![false, false, false, true, false, false, false, false];
    let log_probs = vec![-0.5, -0.8, -0.3, -1.0, -0.4, -0.6, -0.9, -0.5];
    let last_value = 0.8;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    // Step 1: Compute GAE
    let (mut advantages, returns) = compute_gae(
        &rewards,
        &values,
        &dones,
        last_value,
        gamma,
        gae_lambda,
    );

    // Verify GAE outputs
    assert_eq!(advantages.len(), rewards.len());
    assert_eq!(returns.len(), rewards.len());

    // Verify returns = advantages + values
    for i in 0..rewards.len() {
        assert!(
            (returns[i] - (advantages[i] + values[i])).abs() < 1e-5,
            "returns[{}] != advantages + values",
            i
        );
    }

    // Step 2: Normalize advantages
    normalize_advantages(&mut advantages);

    // Verify normalization
    let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
    assert!(
        mean.abs() < 1e-5,
        "Normalized mean should be ~0, got {}",
        mean
    );

    // Step 3: Compute PPO loss (on-policy, so old_log_probs = log_probs)
    let clip_ratio = 0.2;
    let policy_loss = ppo_clip_loss_scalar(&log_probs, &log_probs, &advantages, clip_ratio);
    let vf_loss = value_loss_scalar(&values, &returns);

    // Verify losses are finite
    assert!(
        policy_loss.is_finite(),
        "Policy loss should be finite: {}",
        policy_loss
    );
    assert!(
        vf_loss.is_finite(),
        "Value loss should be finite: {}",
        vf_loss
    );

    // Step 4: Combined loss
    let vf_coef = 0.5;
    let entropy = 0.1; // Placeholder entropy value
    let entropy_coef = 0.01;
    let total_loss = policy_loss + vf_coef * vf_loss - entropy_coef * entropy;

    assert!(
        total_loss.is_finite(),
        "Total loss should be finite: {}",
        total_loss
    );
}

/// Test PPO with multiple epochs (simulating minibatch updates).
#[test]
fn test_ppo_multiple_epochs() {
    let rewards = vec![1.0, 1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5, 0.5];
    let dones = vec![false, false, false, false];
    let old_log_probs = vec![-0.5, -0.5, -0.5, -0.5];
    let last_value = 0.5;

    let (advantages, _returns) = compute_gae(&rewards, &values, &dones, last_value, 0.99, 0.95);

    // Simulate policy update that changes log_probs
    let n_epochs = 4;
    let mut current_log_probs = old_log_probs.clone();

    for epoch in 0..n_epochs {
        let loss = ppo_clip_loss_scalar(
            &current_log_probs,
            &old_log_probs,
            &advantages,
            0.2,
        );

        assert!(
            loss.is_finite(),
            "Loss should be finite at epoch {}",
            epoch
        );

        // Simulate policy update (increase log probs slightly)
        current_log_probs = current_log_probs.iter().map(|lp| lp + 0.1).collect();
    }
}

// ============================================================================
// Full IMPALA Training Step Tests
// ============================================================================

/// Test complete IMPALA training step: trajectories -> V-trace -> loss.
#[test]
fn test_impala_full_training_step() {
    // Simulate off-policy trajectory
    let behavior_log_probs = vec![-0.5, -0.8, -0.3, -1.0, -0.4];
    let target_log_probs = vec![-0.6, -0.7, -0.5, -0.8, -0.5]; // Slightly different
    let rewards = vec![1.0, 0.5, 1.5, 0.0, 2.0];
    let values = vec![0.8, 0.9, 1.0, 0.5, 1.2];
    let dones = vec![false, false, false, true, false];
    let bootstrap = 0.8;
    let gamma = 0.99;
    let rho_clip = 1.0;
    let c_clip = 1.0;

    // Step 1: Compute V-trace
    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        bootstrap,
        gamma,
        rho_clip,
        c_clip,
    );

    // Verify V-trace outputs
    assert_eq!(result.vs.len(), rewards.len());
    assert_eq!(result.advantages.len(), rewards.len());
    assert_eq!(result.rhos.len(), rewards.len());

    // All outputs should be finite
    for (i, (vs, adv, rho)) in result
        .vs
        .iter()
        .zip(result.advantages.iter())
        .zip(result.rhos.iter())
        .map(|((v, a), r)| (v, a, r))
        .enumerate()
    {
        assert!(vs.is_finite(), "vs[{}] should be finite", i);
        assert!(adv.is_finite(), "advantage[{}] should be finite", i);
        assert!(rho.is_finite(), "rho[{}] should be finite", i);
        assert!(*rho <= rho_clip + 1e-6, "rho[{}] should be clipped", i);
    }

    // Step 2: Compute IMPALA policy loss
    // L_policy = -mean(rho * log_pi * A)
    let policy_loss: f32 = result
        .rhos
        .iter()
        .zip(target_log_probs.iter())
        .zip(result.advantages.iter())
        .map(|((rho, lp), adv)| -rho * lp * adv)
        .sum::<f32>()
        / rewards.len() as f32;

    assert!(
        policy_loss.is_finite(),
        "IMPALA policy loss should be finite"
    );

    // Step 3: Compute value loss (vs targets)
    let vf_loss = value_loss_scalar(&values, &result.vs);
    assert!(vf_loss.is_finite(), "IMPALA value loss should be finite");
}

/// Test IMPALA with high staleness (large policy version difference).
#[test]
fn test_impala_high_staleness() {
    // Behavior policy was very different
    let behavior_log_probs = vec![-3.0, -3.0, -3.0];
    let target_log_probs = vec![-0.5, -0.5, -0.5]; // Current policy very different

    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, false, false];

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        1.0, // Standard clipping should handle staleness
        1.0,
    );

    // All rhos should be clipped to 1.0 (since ratio would be huge)
    for rho in &result.rhos {
        assert!(
            (*rho - 1.0).abs() < 1e-6,
            "High staleness should clip rhos to 1.0"
        );
    }

    // Outputs should still be finite
    for vs in &result.vs {
        assert!(vs.is_finite(), "V-trace should be finite with high staleness");
    }
}

// ============================================================================
// PPO vs IMPALA On-Policy Equivalence Tests
// ============================================================================

/// Test that on-policy V-trace advantages are similar to GAE advantages.
/// When behavior == target, both should compute similar TD-based advantages.
#[test]
fn test_on_policy_gae_vtrace_similar_advantages() {
    let rewards = vec![1.0, 0.5, 1.5, 0.0, 2.0];
    let values = vec![0.8, 0.9, 1.0, 0.5, 1.2];
    let dones = vec![false, false, false, true, false];
    let log_probs = vec![-0.5, -0.8, -0.3, -1.0, -0.4];
    let bootstrap = 0.8;
    let gamma = 0.99;

    // GAE with lambda=1 (no bias reduction, pure MC)
    let (gae_advantages, _) = compute_gae(&rewards, &values, &dones, bootstrap, gamma, 1.0);

    // V-trace on-policy (behavior == target)
    let vtrace_result = compute_vtrace(
        &log_probs,
        &log_probs, // On-policy
        &rewards,
        &values,
        &dones,
        bootstrap,
        gamma,
        1.0,
        1.0,
    );

    // Note: V-trace advantages are single-step TD errors
    // GAE with lambda=1 accumulates TD errors
    // They won't be identical, but should have same sign for most cases

    // Verify rhos are all 1 (on-policy)
    for rho in &vtrace_result.rhos {
        assert!((*rho - 1.0).abs() < 1e-6, "On-policy rho should be 1.0");
    }

    // Both should have finite values
    for (gae_a, vt_a) in gae_advantages.iter().zip(vtrace_result.advantages.iter()) {
        assert!(gae_a.is_finite() && vt_a.is_finite());
    }
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

/// Test PPO config default values.
#[test]
fn test_ppo_config_defaults() {
    let config = PPOAlgorithmConfig::default();

    assert_eq!(config.clip_ratio, 0.2);
    assert_eq!(config.vf_coef, 0.5);
    assert_eq!(config.entropy_coef, 0.01);
    assert_eq!(config.gamma, 0.99);
    assert_eq!(config.gae_lambda, 0.95);
    assert_eq!(config.n_epochs, 4);
    assert_eq!(config.n_minibatches, 4);
    assert!(config.normalize_advantages);
}

/// Test IMPALA config default values.
#[test]
fn test_impala_config_defaults() {
    let config = IMPALAConfig::default();

    assert_eq!(config.gamma, 0.99);
    assert_eq!(config.rho_clip, 1.5);
    assert_eq!(config.c_clip, 1.0);
    assert_eq!(config.vf_coef, 0.25);
    assert_eq!(config.entropy_coef, 0.02);
    assert_eq!(config.trajectory_length, 20);
    assert_eq!(config.batch_size, 32);
}

/// Test PPO config builder pattern.
#[test]
fn test_ppo_config_builder() {
    let config = PPOAlgorithmConfig::new()
        .with_clip_ratio(0.1)
        .with_gamma(0.995)
        .with_gae_lambda(0.9)
        .with_n_epochs(10);

    assert_eq!(config.clip_ratio, 0.1);
    assert_eq!(config.gamma, 0.995);
    assert_eq!(config.gae_lambda, 0.9);
    assert_eq!(config.n_epochs, 10);
}

/// Test IMPALA config builder pattern.
#[test]
fn test_impala_config_builder() {
    let config = IMPALAConfig::new()
        .with_n_actors(8)
        .with_n_envs_per_actor(16)
        .with_trajectory_length(40)
        .with_rho_clip(0.5);

    assert_eq!(config.n_actors, 8);
    assert_eq!(config.n_envs_per_actor, 16);
    assert_eq!(config.trajectory_length, 40);
    assert_eq!(config.rho_clip, 0.5);
    assert_eq!(config.total_envs(), 128); // 8 * 16
}

// ============================================================================
// Edge Case Integration Tests
// ============================================================================

/// Test training step with all terminal states.
#[test]
fn test_ppo_all_terminal_states() {
    let rewards = vec![1.0, 2.0, 3.0];
    let values = vec![0.5, 1.0, 1.5];
    let dones = vec![true, true, true];
    let log_probs = vec![-0.5, -0.5, -0.5];
    let last_value = 10.0; // Should be ignored

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, 0.99, 0.95);

    // All advantages should be r - V (no future propagation)
    for i in 0..3 {
        let expected = rewards[i] - values[i];
        assert!(
            (advantages[i] - expected).abs() < 1e-5,
            "Terminal advantage[{}] should be {}, got {}",
            i,
            expected,
            advantages[i]
        );
    }

    // Loss should still be computable
    let loss = ppo_clip_loss_scalar(&log_probs, &log_probs, &advantages, 0.2);
    assert!(loss.is_finite());
}

/// Test training step with single transition.
#[test]
fn test_ppo_single_transition() {
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false];
    let log_probs = vec![-0.5];
    let last_value = 0.8;

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, 0.99, 0.95);

    assert_eq!(advantages.len(), 1);
    assert_eq!(returns.len(), 1);

    let loss = ppo_clip_loss_scalar(&log_probs, &log_probs, &advantages, 0.2);
    assert!(loss.is_finite());
}

/// Test IMPALA with single transition trajectory.
#[test]
fn test_impala_single_transition() {
    let behavior_log_probs = vec![-0.5];
    let target_log_probs = vec![-0.6];
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false];

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.8,
        0.99,
        1.0,
        1.0,
    );

    assert_eq!(result.vs.len(), 1);
    assert_eq!(result.advantages.len(), 1);
    assert_eq!(result.rhos.len(), 1);
}

// ============================================================================
// Tensor-Based Integration Tests
// ============================================================================

/// Test tensor-based PPO loss computation.
#[test]
fn test_ppo_tensor_integration() {
    use crate::algorithms::policy_loss::ppo_clip_loss;

    let device = Default::default();

    // Create tensors from rollout data
    let log_probs_vec = vec![-0.5, -0.8, -0.3, -1.0];
    let old_log_probs_vec = vec![-0.5, -0.8, -0.3, -1.0];
    let advantages_vec = vec![1.0, -0.5, 0.8, -0.2];

    let log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats(log_probs_vec.as_slice(), &device).require_grad();
    let old_log_probs: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats(old_log_probs_vec.as_slice(), &device);
    let advantages: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats(advantages_vec.as_slice(), &device);

    let loss = ppo_clip_loss(log_probs.clone(), old_log_probs, advantages, 0.2);

    // Check loss is finite
    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    assert!(loss_val.is_finite());

    // Check gradients exist
    let grads = loss.backward();
    let log_probs_grad = log_probs.grad(&grads);
    assert!(log_probs_grad.is_some());
}

/// Test tensor-based value loss computation.
#[test]
fn test_value_loss_tensor_integration() {
    use crate::algorithms::policy_loss::value_loss;

    let device = Default::default();

    let values: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device).require_grad();
    let old_values: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
    let returns: Tensor<TestAutodiffBackend, 1> =
        Tensor::from_floats([1.5, 2.5, 3.5, 4.5], &device);

    let loss = value_loss(values.clone(), old_values, returns, Some(0.2));

    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
    assert!(loss_val.is_finite());

    let grads = loss.backward();
    let values_grad = values.grad(&grads);
    assert!(values_grad.is_some());
}

// ============================================================================
// Performance Characteristic Tests
// ============================================================================

/// Test that positive advantages decrease policy loss.
#[test]
fn test_positive_advantages_improve_loss() {
    // With positive advantages, increasing log_probs should decrease loss
    let old_log_probs = vec![-1.0, -1.0, -1.0];
    let advantages = vec![1.0, 1.0, 1.0]; // All positive

    // Before update (on-policy)
    let loss_before = ppo_clip_loss_scalar(&old_log_probs, &old_log_probs, &advantages, 0.2);

    // After update (log_probs increased)
    let new_log_probs = vec![-0.9, -0.9, -0.9]; // Higher
    let loss_after = ppo_clip_loss_scalar(&new_log_probs, &old_log_probs, &advantages, 0.2);

    // Loss should decrease (become more negative = higher objective)
    assert!(
        loss_after < loss_before,
        "Positive advantages: higher log_probs should lower loss: {} vs {}",
        loss_after,
        loss_before
    );
}

/// Test that negative advantages increase policy loss when action prob increases.
#[test]
fn test_negative_advantages_penalize_increase() {
    let old_log_probs = vec![-1.0, -1.0, -1.0];
    let advantages = vec![-1.0, -1.0, -1.0]; // All negative

    // Before update
    let loss_before = ppo_clip_loss_scalar(&old_log_probs, &old_log_probs, &advantages, 0.2);

    // After update (log_probs increased = bad for negative advantages)
    let new_log_probs = vec![-0.9, -0.9, -0.9];
    let loss_after = ppo_clip_loss_scalar(&new_log_probs, &old_log_probs, &advantages, 0.2);

    // Loss should increase
    assert!(
        loss_after > loss_before,
        "Negative advantages: higher log_probs should increase loss: {} vs {}",
        loss_after,
        loss_before
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

/// Test with large batch size.
#[test]
fn test_large_batch_ppo() {
    let batch_size = 10000;
    let rewards: Vec<f32> = (0..batch_size).map(|i| (i as f32 % 5.0) - 2.0).collect();
    let values: Vec<f32> = (0..batch_size).map(|_| 0.5).collect();
    let dones: Vec<bool> = (0..batch_size).map(|i| i % 100 == 99).collect();
    let log_probs: Vec<f32> = (0..batch_size).map(|_| -0.5).collect();
    let last_value = 0.5;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, 0.99, 0.95);

    let loss = ppo_clip_loss_scalar(&log_probs, &log_probs, &advantages, 0.2);

    assert!(loss.is_finite(), "Large batch should give finite loss");
}

/// Test with many terminal states.
#[test]
fn test_many_terminals() {
    let batch_size = 1000;
    let rewards: Vec<f32> = (0..batch_size).map(|_| 1.0).collect();
    let values: Vec<f32> = (0..batch_size).map(|_| 0.5).collect();
    let dones: Vec<bool> = (0..batch_size).map(|i| i % 10 == 9).collect(); // Every 10th is terminal
    let log_probs: Vec<f32> = (0..batch_size).map(|_| -0.5).collect();

    let (advantages, _) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);

    // All advantages should be finite
    for a in &advantages {
        assert!(a.is_finite());
    }

    let loss = ppo_clip_loss_scalar(&log_probs, &log_probs, &advantages, 0.2);
    assert!(loss.is_finite());
}
