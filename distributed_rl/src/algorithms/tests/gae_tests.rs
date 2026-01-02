//! Comprehensive tests for Generalized Advantage Estimation (GAE).
//!
//! These tests serve as the complete behavioral specification for the GAE implementation.
//! They cover:
//! - Core algorithm correctness (lambda extremes, bootstrapping)
//! - Terminal vs truncated state handling
//! - Advantage normalization edge cases
//! - Numerical stability boundaries
//! - Vectorized environment handling
//!
//! # References
//! - Schulman et al., "High-Dimensional Continuous Control Using GAE" (2016)

use crate::algorithms::gae::{
    compute_gae, compute_gae_vectorized, normalize_advantages, normalize_advantages_per_env,
};

// ============================================================================
// Core Algorithm Correctness Tests
// ============================================================================

/// Test that lambda=0 produces one-step TD advantages.
/// When lambda=0, GAE reduces to A_t = delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
/// No future deltas should be propagated.
#[test]
fn test_gae_lambda_zero_produces_one_step_td() {
    let rewards = vec![1.0, 2.0, 3.0];
    let values = vec![0.5, 0.8, 1.0];
    let dones = vec![false, false, false];
    let last_value = 1.2;
    let gamma = 0.99;
    let gae_lambda = 0.0;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // With lambda=0, A_t = r_t + gamma*V_{t+1} - V_t (one-step TD)
    // A_2 = 3.0 + 0.99*1.2 - 1.0 = 3.0 + 1.188 - 1.0 = 3.188
    let expected_a2 = 3.0 + gamma * last_value - values[2];
    assert!(
        (advantages[2] - expected_a2).abs() < 1e-5,
        "lambda=0 should give one-step TD. Expected {}, got {}",
        expected_a2,
        advantages[2]
    );

    // A_1 = 2.0 + 0.99*1.0 - 0.8 = 2.0 + 0.99 - 0.8 = 2.19
    let expected_a1 = 2.0 + gamma * values[2] - values[1];
    assert!(
        (advantages[1] - expected_a1).abs() < 1e-5,
        "lambda=0 should give one-step TD. Expected {}, got {}",
        expected_a1,
        advantages[1]
    );
}

/// Test that lambda=1 produces MC-like advantages (discounted sum of TD errors).
/// When lambda=1, GAE gives full propagation of future TD errors.
#[test]
fn test_gae_lambda_one_produces_mc_like_advantages() {
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.0, 0.0, 0.0]; // Zero values simplify calculation
    let dones = vec![false, false, false];
    let last_value = 0.0;
    let gamma = 0.99;
    let gae_lambda = 1.0;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // With lambda=1 and values=0, A_t = sum_{l=0}^{T-t-1} gamma^l * r_{t+l}
    // This is the MC return since values are 0
    // A_2 = 1.0 (just reward at t=2, no future with last_value=0)
    // A_1 = 1.0 + 0.99 * 1.0 = 1.99
    // A_0 = 1.0 + 0.99 * 1.0 + 0.99^2 * 1.0 = 1 + 0.99 + 0.9801 = 2.9701

    assert!(
        (advantages[2] - 1.0).abs() < 1e-5,
        "A_2 should be 1.0, got {}",
        advantages[2]
    );
    assert!(
        (advantages[1] - 1.99).abs() < 1e-4,
        "A_1 should be ~1.99, got {}",
        advantages[1]
    );
    assert!(
        (advantages[0] - 2.9701).abs() < 1e-4,
        "A_0 should be ~2.9701, got {}",
        advantages[0]
    );

    // Earlier advantages should be larger (accumulate future rewards)
    assert!(
        advantages[0] > advantages[1],
        "With lambda=1, earlier advantages should be larger"
    );
    assert!(
        advantages[1] > advantages[2],
        "With lambda=1, earlier advantages should be larger"
    );
}

/// Test that intermediate lambda values interpolate between TD and MC.
#[test]
fn test_gae_intermediate_lambda_interpolates() {
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.0, 0.0, 0.0];
    let dones = vec![false, false, false];
    let last_value = 0.0;
    let gamma = 0.99;

    let (adv_0, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, 0.0);
    let (adv_95, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, 0.95);
    let (adv_1, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, 1.0);

    // For earlier states, lambda=0.95 advantages should be between lambda=0 and lambda=1
    assert!(
        adv_95[0] > adv_0[0] && adv_95[0] < adv_1[0],
        "lambda=0.95 should interpolate: {} < {} < {}",
        adv_0[0],
        adv_95[0],
        adv_1[0]
    );
}

// ============================================================================
// Bootstrap Value Handling Tests
// ============================================================================

/// Test that bootstrap value is correctly used for non-terminal trajectories.
/// The last_value parameter represents V(s_T) and affects the last TD error.
#[test]
fn test_gae_bootstrap_value_affects_last_advantage() {
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false]; // NOT terminal
    let gamma = 0.99;
    let gae_lambda = 0.95;

    // With bootstrap = 2.0
    let (adv_high, _) = compute_gae(&rewards, &values, &dones, 2.0, gamma, gae_lambda);

    // With bootstrap = 0.0
    let (adv_zero, _) = compute_gae(&rewards, &values, &dones, 0.0, gamma, gae_lambda);

    // Higher bootstrap should give higher advantage
    // A = r + gamma*V_next - V = 1 + 0.99*bootstrap - 0.5
    assert!(
        adv_high[0] > adv_zero[0],
        "Higher bootstrap should give higher advantage: {} vs {}",
        adv_high[0],
        adv_zero[0]
    );

    // Check exact calculation
    let expected_high = 1.0 + gamma * 2.0 - 0.5; // 1 + 1.98 - 0.5 = 2.48
    let expected_zero = 1.0 + gamma * 0.0 - 0.5; // 1 + 0 - 0.5 = 0.5

    assert!(
        (adv_high[0] - expected_high).abs() < 1e-5,
        "Expected {}, got {}",
        expected_high,
        adv_high[0]
    );
    assert!(
        (adv_zero[0] - expected_zero).abs() < 1e-5,
        "Expected {}, got {}",
        expected_zero,
        adv_zero[0]
    );
}

/// Test that bootstrap value is zeroed out when episode terminates.
/// CRITICAL: Terminal states should NOT bootstrap future values.
#[test]
fn test_gae_terminal_state_zeroes_bootstrap() {
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![true]; // TERMINAL
    let gamma = 0.99;
    let gae_lambda = 0.95;

    // Bootstrap value should be IGNORED because done=true
    let (adv_with_bootstrap, _) = compute_gae(&rewards, &values, &dones, 10.0, gamma, gae_lambda);
    let (adv_no_bootstrap, _) = compute_gae(&rewards, &values, &dones, 0.0, gamma, gae_lambda);

    // Both should be identical because done=true zeroes the next_value contribution
    assert!(
        (adv_with_bootstrap[0] - adv_no_bootstrap[0]).abs() < 1e-6,
        "Terminal state should ignore bootstrap: {} vs {}",
        adv_with_bootstrap[0],
        adv_no_bootstrap[0]
    );

    // A = r + gamma*V_next*0 - V = 1 + 0 - 0.5 = 0.5
    assert!(
        (adv_with_bootstrap[0] - 0.5).abs() < 1e-5,
        "Terminal advantage should be r - V = 0.5, got {}",
        adv_with_bootstrap[0]
    );
}

/// Test terminal state in the middle of a trajectory.
/// After a terminal state, GAE should reset (not propagate advantages).
#[test]
fn test_gae_terminal_in_middle_resets_propagation() {
    // Trajectory with terminal at t=1
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, true, false]; // Terminal at t=1
    let last_value = 0.5;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // t=2: A_2 = r_2 + gamma*V_next - V_2 = 1 + 0.99*0.5 - 0.5 = 0.995
    let expected_a2 = 1.0 + gamma * last_value - values[2];
    assert!(
        (advantages[2] - expected_a2).abs() < 1e-5,
        "A_2 should be {}, got {}",
        expected_a2,
        advantages[2]
    );

    // t=1: Terminal, so not_done=0
    // A_1 = delta_1 + gamma*lambda*0*gae = delta_1 = r_1 + gamma*0 - V_1 = 1 - 0.5 = 0.5
    // Note: next_value at t=1 is V_2, but not_done=0 zeroes it
    assert!(
        (advantages[1] - 0.5).abs() < 1e-5,
        "Terminal t=1 should have A_1 = r - V = 0.5, got {}",
        advantages[1]
    );

    // t=0: Should see t=1's advantage but not_done zeroes propagation
    // delta_0 = r_0 + gamma*V_1 - V_0 = 1 + 0.99*0.5 - 0.5 = 0.995
    // A_0 = delta_0 + gamma*lambda*not_done[0]*A_1
    //     = 0.995 + 0.99*0.95*1*0.5 = 0.995 + 0.47025 = 1.46525
    let delta_0 = 1.0 + gamma * values[1] - values[0];
    let expected_a0 = delta_0 + gamma * gae_lambda * 1.0 * advantages[1];
    assert!(
        (advantages[0] - expected_a0).abs() < 1e-4,
        "A_0 should be {}, got {}",
        expected_a0,
        advantages[0]
    );
}

// ============================================================================
// Returns Calculation Tests
// ============================================================================

/// Test that returns = advantages + values (fundamental identity).
#[test]
fn test_gae_returns_equal_advantages_plus_values() {
    let rewards = vec![1.0, 2.0, 0.5, -1.0, 3.0];
    let values = vec![0.5, 1.0, 0.8, 0.3, 1.5];
    let dones = vec![false, false, true, false, false];
    let last_value = 0.7;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    for i in 0..rewards.len() {
        let expected_return = advantages[i] + values[i];
        assert!(
            (returns[i] - expected_return).abs() < 1e-5,
            "returns[{}] should equal advantages[{}] + values[{}]: {} vs {} + {}",
            i,
            i,
            i,
            returns[i],
            advantages[i],
            values[i]
        );
    }
}

// ============================================================================
// Advantage Normalization Tests
// ============================================================================

/// Test normalization produces zero mean.
#[test]
fn test_normalize_advantages_zero_mean() {
    let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    normalize_advantages(&mut advantages);

    let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
    assert!(
        mean.abs() < 1e-6,
        "Normalized mean should be ~0, got {}",
        mean
    );
}

/// Test normalization produces unit variance.
#[test]
fn test_normalize_advantages_unit_variance() {
    let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    normalize_advantages(&mut advantages);

    let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
    let variance: f32 =
        advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / advantages.len() as f32;
    let std = variance.sqrt();

    assert!(
        (std - 1.0).abs() < 1e-5,
        "Normalized std should be ~1, got {}",
        std
    );
}

/// Test normalization with constant values (zero variance).
/// CRITICAL: This is an edge case that could cause division by zero.
#[test]
fn test_normalize_advantages_constant_values() {
    let mut advantages = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    normalize_advantages(&mut advantages);

    // With constant values, variance=0, but we add epsilon to prevent NaN
    // Result should be all zeros (mean subtracted, divided by sqrt(eps))
    for a in &advantages {
        assert!(
            a.is_finite(),
            "Normalization of constant values should not produce NaN/Inf"
        );
    }
}

/// Test normalization of empty vector (edge case).
#[test]
fn test_normalize_advantages_empty() {
    let mut advantages: Vec<f32> = vec![];
    normalize_advantages(&mut advantages);
    assert!(advantages.is_empty());
}

/// Test normalization of single element.
#[test]
fn test_normalize_advantages_single_element() {
    let mut advantages = vec![42.0];
    normalize_advantages(&mut advantages);

    // Single element: (42 - 42) / sqrt(0 + eps) = 0 / small = ~0
    assert!(
        advantages[0].is_finite(),
        "Single element normalization should be finite"
    );
    assert!(
        advantages[0].abs() < 1e-2,
        "Single element should normalize to ~0, got {}",
        advantages[0]
    );
}

/// Test normalization with extreme values.
#[test]
fn test_normalize_advantages_extreme_values() {
    let mut advantages = vec![-1e6, 0.0, 1e6];
    normalize_advantages(&mut advantages);

    // Should still be finite and normalized
    for a in &advantages {
        assert!(a.is_finite(), "Extreme values should normalize to finite");
    }

    let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
    assert!(
        mean.abs() < 1e-4,
        "Mean of extreme normalized values should be ~0"
    );
}

/// Test normalization with mix of positive and negative.
#[test]
fn test_normalize_advantages_mixed_signs() {
    let mut advantages = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    normalize_advantages(&mut advantages);

    // Signs should be preserved (relative ordering)
    assert!(
        advantages[0] < advantages[2],
        "Negative should still be less than zero"
    );
    assert!(
        advantages[4] > advantages[2],
        "Positive should still be greater than zero"
    );

    // Mean should be zero
    let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
    assert!(mean.abs() < 1e-6, "Mean should be ~0");
}

// ============================================================================
// Vectorized Environment Tests
// ============================================================================

/// Test vectorized GAE correctly handles interleaved data.
/// Data layout: [env0_t0, env1_t0, ..., envN_t0, env0_t1, env1_t1, ...]
#[test]
fn test_gae_vectorized_interleaved_layout() {
    // 2 envs, 3 steps each
    // Interleaved: [e0t0, e1t0, e0t1, e1t1, e0t2, e1t2]
    let rewards = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
    let values = vec![0.5, 1.0, 0.5, 1.0, 0.5, 1.0];
    let dones = vec![false, false, false, false, false, false];
    let last_values = vec![0.5, 1.0]; // env0, env1
    let n_envs = 2;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, _) = compute_gae_vectorized(
        &rewards,
        &values,
        &dones,
        &last_values,
        n_envs,
        gamma,
        gae_lambda,
    );

    // Env 1 has higher rewards, so should have higher advantages
    // Check last step of each env
    assert!(
        advantages[5] > advantages[4],
        "Env 1 (higher rewards) should have higher advantage at last step"
    );

    // Verify the interleaved indexing is correct
    // env0, step2 should be at index 4 (2*2 + 0)
    // env1, step2 should be at index 5 (2*2 + 1)

    // Compute expected for env0 separately
    let env0_rewards = vec![1.0, 1.0, 1.0];
    let env0_values = vec![0.5, 0.5, 0.5];
    let env0_dones = vec![false, false, false];
    let (env0_adv, _) = compute_gae(
        &env0_rewards,
        &env0_values,
        &env0_dones,
        0.5,
        gamma,
        gae_lambda,
    );

    // Compare vectorized result with single-env result
    assert!(
        (advantages[4] - env0_adv[2]).abs() < 1e-5,
        "Vectorized env0 step2 should match single: {} vs {}",
        advantages[4],
        env0_adv[2]
    );
}

/// Test vectorized GAE with different done patterns per environment.
#[test]
fn test_gae_vectorized_different_done_patterns() {
    // 2 envs, 3 steps
    // Env 0: no terminals
    // Env 1: terminal at t=1
    let rewards = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let dones = vec![
        false, false, // t=0: both not done
        false, true,  // t=1: env0 continues, env1 terminal
        false, false, // t=2: both continue (env1 has reset)
    ];
    let last_values = vec![0.5, 0.5];
    let n_envs = 2;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, _) = compute_gae_vectorized(
        &rewards,
        &values,
        &dones,
        &last_values,
        n_envs,
        gamma,
        gae_lambda,
    );

    // At t=1, env1 is terminal, so its advantage should be different
    // Env0 t=1 advantage: delta + lambda*gamma*not_done*gae
    // Env1 t=1 advantage: delta + lambda*gamma*0*gae = delta only

    // Env1, t=1 (index 3): terminal, so A = r - V = 1 - 0.5 = 0.5
    assert!(
        (advantages[3] - 0.5).abs() < 1e-5,
        "Env1 terminal should have A=0.5, got {}",
        advantages[3]
    );

    // Env0, t=1 (index 2): not terminal, should propagate
    assert!(
        advantages[2] != advantages[3],
        "Different done patterns should give different advantages"
    );
}

/// Test per-environment normalization.
#[test]
fn test_normalize_advantages_per_env() {
    // 2 envs, 4 steps each (interleaved)
    let mut advantages = vec![
        1.0, 10.0, // t=0
        2.0, 20.0, // t=1
        3.0, 30.0, // t=2
        4.0, 40.0, // t=3
    ];
    let n_envs = 2;

    normalize_advantages_per_env(&mut advantages, n_envs);

    // Extract env0 and env1 advantages
    let env0: Vec<f32> = (0..4).map(|t| advantages[t * n_envs]).collect();
    let env1: Vec<f32> = (0..4).map(|t| advantages[t * n_envs + 1]).collect();

    // Each env should have zero mean
    let mean0: f32 = env0.iter().sum::<f32>() / env0.len() as f32;
    let mean1: f32 = env1.iter().sum::<f32>() / env1.len() as f32;

    assert!(mean0.abs() < 1e-5, "Env0 mean should be ~0, got {}", mean0);
    assert!(mean1.abs() < 1e-5, "Env1 mean should be ~0, got {}", mean1);

    // Each env should have unit std
    let var0: f32 = env0.iter().map(|a| (a - mean0).powi(2)).sum::<f32>() / env0.len() as f32;
    let var1: f32 = env1.iter().map(|a| (a - mean1).powi(2)).sum::<f32>() / env1.len() as f32;

    assert!(
        (var0.sqrt() - 1.0).abs() < 1e-4,
        "Env0 std should be ~1, got {}",
        var0.sqrt()
    );
    assert!(
        (var1.sqrt() - 1.0).abs() < 1e-4,
        "Env1 std should be ~1, got {}",
        var1.sqrt()
    );
}

// ============================================================================
// Edge Cases and Boundary Conditions
// ============================================================================

/// Test with zero rewards.
#[test]
fn test_gae_zero_rewards() {
    let rewards = vec![0.0, 0.0, 0.0];
    let values = vec![1.0, 1.0, 1.0];
    let dones = vec![false, false, false];
    let last_value = 1.0;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // With r=0 and V constant, delta = 0 + gamma*V - V = (gamma-1)*V
    // All deltas should be negative (gamma < 1)
    for a in &advantages {
        assert!(a.is_finite(), "Zero rewards should give finite advantages");
    }
}

/// Test with gamma=0 (myopic agent).
#[test]
fn test_gae_gamma_zero_myopic() {
    let rewards = vec![1.0, 2.0, 3.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, false, false];
    let last_value = 0.5;
    let gamma = 0.0; // Myopic
    let gae_lambda = 0.95;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // With gamma=0, A_t = r_t - V_t (immediate reward only)
    for i in 0..rewards.len() {
        let expected = rewards[i] - values[i];
        assert!(
            (advantages[i] - expected).abs() < 1e-5,
            "gamma=0 should give A = r - V. Expected {}, got {}",
            expected,
            advantages[i]
        );
    }
}

/// Test with gamma=1 (undiscounted).
#[test]
fn test_gae_gamma_one_undiscounted() {
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.0, 0.0, 0.0];
    let dones = vec![false, false, false];
    let last_value = 0.0;
    let gamma = 1.0; // Undiscounted
    let gae_lambda = 1.0;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // With gamma=1, lambda=1, V=0: A_t = sum of all future rewards
    // A_2 = 1
    // A_1 = 1 + 1 = 2
    // A_0 = 1 + 1 + 1 = 3
    assert!((advantages[2] - 1.0).abs() < 1e-5);
    assert!((advantages[1] - 2.0).abs() < 1e-5);
    assert!((advantages[0] - 3.0).abs() < 1e-5);
}

/// Test with negative rewards.
#[test]
fn test_gae_negative_rewards() {
    let rewards = vec![-1.0, -2.0, -3.0];
    let values = vec![0.0, 0.0, 0.0];
    let dones = vec![false, false, false];
    let last_value = 0.0;
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, _) = compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

    // All advantages should be negative
    for a in &advantages {
        assert!(
            *a < 0.0,
            "Negative rewards with zero values should give negative advantages"
        );
    }
}

/// Test that dimension mismatches panic (as per assertion).
#[test]
#[should_panic]
fn test_gae_dimension_mismatch_panics() {
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5]; // Wrong length!
    let dones = vec![false, false, false];

    compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
}

/// Test empty inputs return empty outputs.
#[test]
fn test_gae_empty_inputs() {
    let rewards: Vec<f32> = vec![];
    let values: Vec<f32> = vec![];
    let dones: Vec<bool> = vec![];

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);

    assert!(advantages.is_empty());
    assert!(returns.is_empty());
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

/// Test with very large rewards.
#[test]
fn test_gae_large_rewards_stability() {
    let rewards = vec![1e6, 1e6, 1e6];
    let values = vec![0.0, 0.0, 0.0];
    let dones = vec![false, false, false];
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, gamma, gae_lambda);

    for a in &advantages {
        assert!(a.is_finite(), "Large rewards should give finite advantages");
    }
    for r in &returns {
        assert!(r.is_finite(), "Large rewards should give finite returns");
    }
}

/// Test with very small rewards.
#[test]
fn test_gae_small_rewards_stability() {
    let rewards = vec![1e-10, 1e-10, 1e-10];
    let values = vec![1e-10, 1e-10, 1e-10];
    let dones = vec![false, false, false];
    let gamma = 0.99;
    let gae_lambda = 0.95;

    let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, gamma, gae_lambda);

    for a in &advantages {
        assert!(a.is_finite(), "Small rewards should give finite advantages");
    }
    for r in &returns {
        assert!(r.is_finite(), "Small rewards should give finite returns");
    }
}

// ============================================================================
// Property-Based Tests (Invariants)
// ============================================================================

/// Property: Advantage monotonicity with lambda.
/// Higher lambda should give larger magnitude advantages for early states
/// when there are consistent positive or negative TD errors.
#[test]
fn test_gae_lambda_monotonicity_property() {
    let rewards = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let values = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // All deltas positive
    let dones = vec![false, false, false, false, false];
    let gamma = 0.99;

    let lambdas = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0];
    let mut prev_a0 = f32::NEG_INFINITY;

    for lambda in lambdas {
        let (adv, _) = compute_gae(&rewards, &values, &dones, 0.0, gamma, lambda);
        assert!(
            adv[0] >= prev_a0,
            "A_0 should increase with lambda: prev={}, current={} at lambda={}",
            prev_a0,
            adv[0],
            lambda
        );
        prev_a0 = adv[0];
    }
}

/// Property: Terminal states should have bounded advantages.
/// When done=true, the advantage is simply r - V (no future contribution).
#[test]
fn test_gae_terminal_bounded_property() {
    let rewards = vec![5.0];
    let values = vec![2.0];
    let dones = vec![true];

    // With any gamma/lambda/bootstrap, terminal A should be r - V
    for gamma in [0.0, 0.5, 0.99, 1.0] {
        for lambda in [0.0, 0.5, 1.0] {
            for bootstrap in [0.0, 100.0, -100.0] {
                let (adv, _) = compute_gae(&rewards, &values, &dones, bootstrap, gamma, lambda);
                assert!(
                    (adv[0] - 3.0).abs() < 1e-5,
                    "Terminal A should always be r-V=3, got {} at gamma={}, lambda={}, boot={}",
                    adv[0],
                    gamma,
                    lambda,
                    bootstrap
                );
            }
        }
    }
}

// ============================================================================
// Property-Based Tests with Proptest
// ============================================================================

#[cfg(test)]
mod proptest_gae {
    use super::*;
    use proptest::prelude::*;

    /// Generate reasonable reward values (not too extreme to avoid overflow)
    fn reasonable_reward() -> impl Strategy<Value = f32> {
        -100.0f32..100.0
    }

    /// Generate reasonable value estimates
    fn reasonable_value() -> impl Strategy<Value = f32> {
        -100.0f32..100.0
    }

    proptest! {
        /// Property: GAE should always return finite values for reasonable inputs
        #[test]
        fn test_gae_always_finite(
            rewards in prop::collection::vec(reasonable_reward(), 1..50),
            gamma in 0.0f32..=1.0,
            gae_lambda in 0.0f32..=1.0,
            last_value in reasonable_value(),
        ) {
            let n = rewards.len();
            let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
            let dones: Vec<bool> = (0..n).map(|_| false).collect();

            let (advantages, returns) = compute_gae(
                &rewards, &values, &dones, last_value, gamma, gae_lambda
            );

            // All advantages should be finite
            for (i, a) in advantages.iter().enumerate() {
                prop_assert!(
                    a.is_finite(),
                    "Advantage at {} should be finite, got {} (r={}, v={}, gamma={}, lambda={})",
                    i, a, rewards[i], values[i], gamma, gae_lambda
                );
            }

            // All returns should be finite
            for (i, r) in returns.iter().enumerate() {
                prop_assert!(
                    r.is_finite(),
                    "Return at {} should be finite, got {}",
                    i, r
                );
            }
        }

        /// Property: Advantage normalization should produce bounded results
        #[test]
        fn test_normalization_bounded(
            raw_advantages in prop::collection::vec(-1e6f32..1e6, 2..100),
        ) {
            let mut advantages = raw_advantages.clone();
            normalize_advantages(&mut advantages);

            // All normalized values should be finite
            for (i, a) in advantages.iter().enumerate() {
                prop_assert!(
                    a.is_finite(),
                    "Normalized advantage at {} should be finite, got {} (original was {})",
                    i, a, raw_advantages[i]
                );
            }

            // Normalized advantages should be roughly bounded (within ~10 std devs)
            for a in &advantages {
                prop_assert!(
                    a.abs() < 50.0,
                    "Normalized advantage should be bounded, got {} (likely numerical issue)",
                    a
                );
            }

            // Mean should be approximately zero
            let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
            prop_assert!(
                mean.abs() < 1e-4,
                "Normalized mean should be ~0, got {}",
                mean
            );
        }

        /// Property: Returns should equal advantages plus values (fundamental identity)
        #[test]
        fn test_returns_identity(
            rewards in prop::collection::vec(reasonable_reward(), 1..30),
            gamma in 0.5f32..1.0,
            gae_lambda in 0.5f32..1.0,
        ) {
            let n = rewards.len();
            let values: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.5).collect();
            let dones: Vec<bool> = (0..n).map(|_| false).collect();

            let (advantages, returns) = compute_gae(
                &rewards, &values, &dones, 0.0, gamma, gae_lambda
            );

            for i in 0..n {
                let expected_return = advantages[i] + values[i];
                prop_assert!(
                    (returns[i] - expected_return).abs() < 1e-4,
                    "Return[{}] = {} should equal advantage + value = {} + {} = {}",
                    i, returns[i], advantages[i], values[i], expected_return
                );
            }
        }

        /// Property: Terminal states break advantage propagation
        #[test]
        fn test_terminal_breaks_propagation(
            reward_before in reasonable_reward(),
            reward_after in reasonable_reward(),
            value in reasonable_value(),
            gamma in 0.9f32..1.0,
            gae_lambda in 0.9f32..1.0,
        ) {
            // Two transitions: one ends terminal, one continues
            let rewards = vec![reward_before, reward_after];
            let values = vec![value, value];
            let dones_terminal = vec![true, false];
            let dones_continue = vec![false, false];

            let (adv_term, _) = compute_gae(
                &rewards, &values, &dones_terminal, value, gamma, gae_lambda
            );
            let (adv_cont, _) = compute_gae(
                &rewards, &values, &dones_continue, value, gamma, gae_lambda
            );

            // The second advantage should be the same (doesn't depend on first step's done)
            prop_assert!(
                (adv_term[1] - adv_cont[1]).abs() < 1e-4,
                "Second advantage should be same regardless of first step done: {} vs {}",
                adv_term[1], adv_cont[1]
            );

            // The first advantage should be different because terminal zeroes next_value
            // (unless by chance the values work out the same, which is rare)
            // Actually, when done[0]=true, the advantage at t=0 doesn't propagate to t=1,
            // but the calculation at t=0 itself differs because we zero next_value
            let delta_0_term = reward_before + gamma * 0.0 - value; // done zeroes next_value
            let delta_0_cont = reward_before + gamma * value - value;

            prop_assert!(
                (adv_term[0] - delta_0_term).abs() < 1e-4,
                "Terminal first advantage should be r - V = {}, got {}",
                delta_0_term, adv_term[0]
            );
        }

        /// Property: GAE vectorized should match single-env GAE
        #[test]
        fn test_vectorized_matches_single(
            rewards_env0 in prop::collection::vec(reasonable_reward(), 3..10),
            rewards_env1 in prop::collection::vec(reasonable_reward(), 3..10),
            gamma in 0.9f32..1.0,
            gae_lambda in 0.9f32..1.0,
        ) {
            // Use same length for both envs
            let n_steps = rewards_env0.len().min(rewards_env1.len());
            let rewards_env0: Vec<f32> = rewards_env0.into_iter().take(n_steps).collect();
            let rewards_env1: Vec<f32> = rewards_env1.into_iter().take(n_steps).collect();

            let values_env0: Vec<f32> = (0..n_steps).map(|i| i as f32 * 0.1).collect();
            let values_env1: Vec<f32> = (0..n_steps).map(|i| i as f32 * 0.2).collect();
            let dones_env0: Vec<bool> = vec![false; n_steps];
            let dones_env1: Vec<bool> = vec![false; n_steps];
            let last_value_0 = 0.5;
            let last_value_1 = 1.0;

            // Single env computation
            let (single_adv_0, _) = compute_gae(
                &rewards_env0, &values_env0, &dones_env0, last_value_0, gamma, gae_lambda
            );
            let (single_adv_1, _) = compute_gae(
                &rewards_env1, &values_env1, &dones_env1, last_value_1, gamma, gae_lambda
            );

            // Interleave for vectorized
            let mut interleaved_rewards = Vec::with_capacity(n_steps * 2);
            let mut interleaved_values = Vec::with_capacity(n_steps * 2);
            let mut interleaved_dones = Vec::with_capacity(n_steps * 2);
            for t in 0..n_steps {
                interleaved_rewards.push(rewards_env0[t]);
                interleaved_rewards.push(rewards_env1[t]);
                interleaved_values.push(values_env0[t]);
                interleaved_values.push(values_env1[t]);
                interleaved_dones.push(dones_env0[t]);
                interleaved_dones.push(dones_env1[t]);
            }

            let (vec_adv, _) = compute_gae_vectorized(
                &interleaved_rewards,
                &interleaved_values,
                &interleaved_dones,
                &[last_value_0, last_value_1],
                2,
                gamma,
                gae_lambda,
            );

            // Extract and compare
            for t in 0..n_steps {
                prop_assert!(
                    (vec_adv[t * 2] - single_adv_0[t]).abs() < 1e-4,
                    "Env0 step {} mismatch: vec={}, single={}",
                    t, vec_adv[t * 2], single_adv_0[t]
                );
                prop_assert!(
                    (vec_adv[t * 2 + 1] - single_adv_1[t]).abs() < 1e-4,
                    "Env1 step {} mismatch: vec={}, single={}",
                    t, vec_adv[t * 2 + 1], single_adv_1[t]
                );
            }
        }
    }
}
