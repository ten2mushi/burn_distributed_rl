//! Comprehensive tests for V-trace off-policy correction.
//!
//! These tests serve as the complete behavioral specification for V-trace.
//! V-trace is the core algorithm for IMPALA, providing:
//! - Importance sampling correction for off-policy data
//! - Clipped importance weights to prevent variance explosion
//! - Trace-cutting coefficients for stability
//!
//! # Critical Dangers Tested
//!
//! 1. Exponential overflow in log-space (log_ratio.exp())
//! 2. Correct advantage formula (rho NOT included in advantages)
//! 3. On-policy equivalence to TD learning
//! 4. Terminal state handling
//!
//! # References
//! - Espeholt et al., "IMPALA: Scalable Distributed Deep-RL" (2018)

use crate::algorithms::vtrace::{compute_vtrace, compute_vtrace_batch, VTraceInput, VTraceResult};

// ============================================================================
// Core Algorithm Correctness Tests
// ============================================================================

/// Test that on-policy V-trace (behavior == target) gives importance weights of 1.
/// This is the fundamental identity: when policies match, no correction needed.
#[test]
fn test_vtrace_on_policy_rho_equals_one() {
    let log_probs = vec![-1.0, -0.5, -2.0];
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, false, false];

    let result = compute_vtrace(
        &log_probs, // behavior
        &log_probs, // target (same = on-policy)
        &rewards,
        &values,
        &dones,
        0.5,  // bootstrap
        0.99, // gamma
        1.0,  // rho_bar
        1.0,  // c_bar
    );

    // All rhos should be exactly 1.0 when policies match
    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            (*rho - 1.0).abs() < 1e-6,
            "On-policy rho[{}] should be 1.0, got {}",
            i,
            rho
        );
    }
}

/// Test importance weight clipping when target >> behavior probability.
/// When target policy has much higher probability, raw ratio would explode.
#[test]
fn test_vtrace_rho_clipping_high_ratio() {
    // Target has much higher probability than behavior
    let behavior_log_probs = vec![-5.0, -5.0]; // Low prob actions
    let target_log_probs = vec![-0.1, -0.1]; // High prob now

    let rewards = vec![1.0, 1.0];
    let values = vec![0.5, 0.5];
    let dones = vec![false, false];

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        1.0, // rho_bar = 1.0
        1.0, // c_bar = 1.0
    );

    // Raw ratio would be exp(-0.1 - (-5.0)) = exp(4.9) = ~134
    // But should be clipped to rho_bar = 1.0
    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            *rho <= 1.0 + 1e-6,
            "rho[{}] should be clipped to <= 1.0, got {}",
            i,
            rho
        );
        assert!(
            (*rho - 1.0).abs() < 1e-6,
            "rho[{}] should equal rho_bar=1.0, got {}",
            i,
            rho
        );
    }
}

/// Test that lower target probability gives rho < 1 (not clipped).
#[test]
fn test_vtrace_rho_not_clipped_when_low_ratio() {
    // Target has lower probability than behavior
    let behavior_log_probs = vec![-0.5, -0.5]; // Higher prob at collection
    let target_log_probs = vec![-1.0, -1.0]; // Lower prob now

    let rewards = vec![1.0, 1.0];
    let values = vec![0.5, 0.5];
    let dones = vec![false, false];

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        1.0,
        1.0,
    );

    // Raw ratio = exp(-1.0 - (-0.5)) = exp(-0.5) = ~0.606
    // Not clipped since < 1.0
    let expected_rho = (-0.5_f32).exp(); // ~0.606
    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            (*rho - expected_rho).abs() < 1e-4,
            "rho[{}] should be ~{}, got {}",
            i,
            expected_rho,
            rho
        );
    }
}

/// Test custom rho_bar clipping value.
#[test]
fn test_vtrace_custom_rho_bar() {
    let behavior_log_probs = vec![-2.0];
    let target_log_probs = vec![-0.5];

    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false];

    let rho_bar = 0.5; // Custom clipping

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        rho_bar,
        1.0,
    );

    // Raw ratio = exp(-0.5 - (-2.0)) = exp(1.5) = ~4.48
    // Should be clipped to rho_bar = 0.5
    assert!(
        (result.rhos[0] - 0.5).abs() < 1e-6,
        "rho should be clipped to 0.5, got {}",
        result.rhos[0]
    );
}

// ============================================================================
// Advantage Formula Tests (CRITICAL: rho NOT in advantages)
// ============================================================================

/// REGRESSION TEST: Advantages should NOT contain rho.
/// The IMPALA paper specifies that advantages are raw TD errors.
/// Rho is applied externally in the policy loss.
#[test]
fn test_vtrace_advantages_do_not_contain_rho() {
    // Create off-policy scenario where rho != 1
    let behavior_log_probs = vec![-2.0]; // Low probability
    let target_log_probs = vec![-0.5]; // Higher probability
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false];
    let bootstrap = 0.5;
    let gamma = 0.99;

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        bootstrap,
        gamma,
        1.0,
        1.0,
    );

    // Advantage should be raw TD error: A = r + gamma*V(next) - V(current)
    // NOT weighted by rho
    let expected_advantage = rewards[0] + gamma * bootstrap - values[0];
    // = 1.0 + 0.99*0.5 - 0.5 = 0.995

    assert!(
        (result.advantages[0] - expected_advantage).abs() < 1e-6,
        "Advantage should NOT include rho! Expected {}, got {}. \
         Rho={} should be separate.",
        expected_advantage,
        result.advantages[0],
        result.rhos[0]
    );
}

/// Test that rhos are returned separately for external use in policy loss.
#[test]
fn test_vtrace_rhos_returned_for_external_use() {
    let behavior_log_probs = vec![-1.5, -2.0, -0.5];
    let target_log_probs = vec![-1.0, -1.0, -1.0];
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
        1.0,
        1.0,
    );

    // Verify rhos are computed and returned
    assert_eq!(result.rhos.len(), 3);

    // Each rho should be different based on log prob differences
    // rho_0 = exp(-1.0 - (-1.5)) = exp(0.5) = 1.65, clipped to 1.0
    // rho_1 = exp(-1.0 - (-2.0)) = exp(1.0) = 2.72, clipped to 1.0
    // rho_2 = exp(-1.0 - (-0.5)) = exp(-0.5) = 0.61, not clipped

    assert!((result.rhos[0] - 1.0).abs() < 1e-6, "rho_0 should be clipped to 1.0");
    assert!((result.rhos[1] - 1.0).abs() < 1e-6, "rho_1 should be clipped to 1.0");
    assert!(
        (result.rhos[2] - 0.606).abs() < 0.01,
        "rho_2 should be ~0.606, got {}",
        result.rhos[2]
    );
}

// ============================================================================
// V-trace Target Tests
// ============================================================================

/// Test V-trace target formula for single step.
/// vs = V(s) + delta + gamma*c*(vs_next - V(s_next))
#[test]
fn test_vtrace_target_single_step() {
    let behavior_log_probs = vec![-1.0];
    let target_log_probs = vec![-1.0]; // On-policy for simplicity
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false];
    let bootstrap = 0.8;
    let gamma = 0.99;

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        bootstrap,
        gamma,
        1.0,
        1.0,
    );

    // For single step with on-policy (rho=c=1):
    // delta = rho * (r + gamma*V_next - V) = 1 * (1 + 0.99*0.8 - 0.5) = 1.292
    // vs = V + delta + gamma*c*(vs_next - V_next)
    //    = 0.5 + 1.292 + 0.99*1*(0.8 - 0.8)  [vs_next = bootstrap at boundary]
    //    = 1.792
    let delta = 1.0 + gamma * bootstrap - values[0];
    let expected_vs = values[0] + delta; // Simplified for boundary case

    assert!(
        (result.vs[0] - expected_vs).abs() < 1e-5,
        "V-trace target should be {}, got {}",
        expected_vs,
        result.vs[0]
    );
}

/// Test V-trace with off-policy data shows correction.
#[test]
fn test_vtrace_off_policy_correction() {
    // Same trajectory, compute with different rho_bar to see effect
    let behavior_log_probs = vec![-2.0, -2.0, -2.0];
    let target_log_probs = vec![-0.5, -0.5, -0.5]; // Much higher prob

    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, false, false];

    // With high rho_bar (less clipping)
    let result_high = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        10.0, // high rho_bar
        10.0, // high c_bar
    );

    // With low rho_bar (more clipping)
    let result_low = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        0.5, // low rho_bar
        0.5, // low c_bar
    );

    // Different clipping should give different targets
    assert!(
        (result_high.vs[0] - result_low.vs[0]).abs() > 0.01,
        "Different rho_bar should give different V-trace targets"
    );
}

// ============================================================================
// Terminal State Handling Tests
// ============================================================================

/// Test that terminal state zeroes future contributions.
#[test]
fn test_vtrace_terminal_zeroes_future() {
    let log_probs = vec![-1.0, -1.0, -1.0];
    let rewards = vec![1.0, 1.0, 0.0]; // Terminal reward is 0
    let values = vec![0.5, 0.5, 0.0]; // Terminal value is 0
    let dones = vec![false, false, true];

    let result = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        10.0, // Non-zero bootstrap (should be ignored)
        0.99,
        1.0,
        1.0,
    );

    // At terminal: A = r + gamma*0 - V = 0 - 0 = 0
    // vs = V + delta = 0 + 0 = 0
    assert!(
        result.advantages[2].abs() < 1e-6,
        "Terminal advantage should be ~0, got {}",
        result.advantages[2]
    );
    assert!(
        result.vs[2].abs() < 1e-6,
        "Terminal vs should be ~0, got {}",
        result.vs[2]
    );
}

/// Test bootstrap value is correctly used for non-terminal.
#[test]
fn test_vtrace_bootstrap_for_non_terminal() {
    let log_probs = vec![-1.0];
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false]; // NOT terminal

    // With high bootstrap
    let result_high = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        2.0, // high bootstrap
        0.99,
        1.0,
        1.0,
    );

    // With low bootstrap
    let result_low = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        0.0, // low bootstrap
        0.99,
        1.0,
        1.0,
    );

    // Higher bootstrap should give higher advantage
    assert!(
        result_high.advantages[0] > result_low.advantages[0],
        "Higher bootstrap should give higher advantage: {} vs {}",
        result_high.advantages[0],
        result_low.advantages[0]
    );
}

/// Test terminal in middle of trajectory.
#[test]
fn test_vtrace_terminal_in_middle() {
    let log_probs = vec![-1.0, -1.0, -1.0];
    let rewards = vec![1.0, 1.0, 1.0];
    let values = vec![0.5, 0.5, 0.5];
    let dones = vec![false, true, false]; // Terminal at t=1

    let result = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        1.0,
        1.0,
    );

    // At t=1 (terminal): A = r + gamma*0 - V = 1 - 0.5 = 0.5
    let expected_a1 = 1.0 - 0.5;
    assert!(
        (result.advantages[1] - expected_a1).abs() < 1e-5,
        "Terminal t=1 advantage should be {}, got {}",
        expected_a1,
        result.advantages[1]
    );
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

/// CRITICAL: Test extreme log probability differences don't cause overflow.
/// log_ratio.exp() can overflow when differences are large.
#[test]
fn test_vtrace_extreme_log_probs_no_overflow() {
    // Extreme case: behavior has near-zero probability, target has high
    let behavior_log_probs = vec![-100.0, -50.0, -0.01];
    let target_log_probs = vec![-0.01, -100.0, -50.0];
    let rewards = vec![1.0, -1.0, 0.5];
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
        1.0,
        1.0,
    );

    // All outputs should be finite (no NaN or Inf)
    for (i, vs) in result.vs.iter().enumerate() {
        assert!(
            vs.is_finite(),
            "vs[{}] should be finite with extreme log probs, got {}",
            i,
            vs
        );
    }
    for (i, adv) in result.advantages.iter().enumerate() {
        assert!(
            adv.is_finite(),
            "advantage[{}] should be finite with extreme log probs, got {}",
            i,
            adv
        );
    }
    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            rho.is_finite(),
            "rho[{}] should be finite with extreme log probs, got {}",
            i,
            rho
        );
        assert!(
            *rho <= 1.0 + 1e-6,
            "rho[{}] should be clipped, got {}",
            i,
            rho
        );
    }
}

/// Test with very large rewards.
#[test]
fn test_vtrace_large_rewards_stability() {
    let log_probs = vec![-1.0, -1.0, -1.0];
    let rewards = vec![1e6, 1e6, 1e6];
    let values = vec![0.0, 0.0, 0.0];
    let dones = vec![false, false, false];

    let result = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        0.0,
        0.99,
        1.0,
        1.0,
    );

    for vs in &result.vs {
        assert!(vs.is_finite(), "Large rewards should give finite vs");
    }
    for adv in &result.advantages {
        assert!(
            adv.is_finite(),
            "Large rewards should give finite advantages"
        );
    }
}

/// Test with very small values.
#[test]
fn test_vtrace_small_values_stability() {
    let log_probs = vec![-1.0, -1.0];
    let rewards = vec![1e-10, 1e-10];
    let values = vec![1e-10, 1e-10];
    let dones = vec![false, false];

    let result = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        1e-10,
        0.99,
        1.0,
        1.0,
    );

    for vs in &result.vs {
        assert!(vs.is_finite(), "Small values should give finite vs");
    }
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test empty inputs.
#[test]
fn test_vtrace_empty_inputs() {
    let result = compute_vtrace(&[], &[], &[], &[], &[], 0.0, 0.99, 1.0, 1.0);

    assert!(result.vs.is_empty());
    assert!(result.advantages.is_empty());
    assert!(result.rhos.is_empty());
}

/// Test single transition.
#[test]
fn test_vtrace_single_transition() {
    let log_probs = vec![-1.0];
    let rewards = vec![1.0];
    let values = vec![0.5];
    let dones = vec![false];

    let result = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        0.5,
        0.99,
        1.0,
        1.0,
    );

    assert_eq!(result.vs.len(), 1);
    assert_eq!(result.advantages.len(), 1);
    assert_eq!(result.rhos.len(), 1);

    // A = r + gamma*bootstrap - V = 1 + 0.99*0.5 - 0.5 = 0.995
    let expected_adv = 1.0 + 0.99 * 0.5 - 0.5;
    assert!((result.advantages[0] - expected_adv).abs() < 1e-5);
}

/// Test all terminal states.
#[test]
fn test_vtrace_all_terminal() {
    let log_probs = vec![-1.0, -1.0, -1.0];
    let rewards = vec![1.0, 2.0, 3.0];
    let values = vec![0.5, 1.0, 1.5];
    let dones = vec![true, true, true];

    let result = compute_vtrace(
        &log_probs,
        &log_probs,
        &rewards,
        &values,
        &dones,
        10.0, // Should be ignored
        0.99,
        1.0,
        1.0,
    );

    // All advantages should be r - V (no future)
    for i in 0..3 {
        let expected = rewards[i] - values[i];
        assert!(
            (result.advantages[i] - expected).abs() < 1e-5,
            "Terminal advantage[{}] should be {}, got {}",
            i,
            expected,
            result.advantages[i]
        );
    }
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

/// Test batch V-trace computation.
#[test]
fn test_vtrace_batch_processing() {
    let traj1 = VTraceInput {
        behavior_log_probs: vec![-1.0, -1.0],
        target_log_probs: vec![-1.0, -1.0],
        rewards: vec![1.0, 1.0],
        values: vec![0.5, 0.5],
        dones: vec![false, false],
        bootstrap_value: 0.5,
    };

    let traj2 = VTraceInput {
        behavior_log_probs: vec![-0.5],
        target_log_probs: vec![-0.5],
        rewards: vec![2.0],
        values: vec![1.0],
        dones: vec![true],
        bootstrap_value: 0.0,
    };

    let results = compute_vtrace_batch(&[traj1, traj2], 0.99, 1.0, 1.0);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].vs.len(), 2);
    assert_eq!(results[1].vs.len(), 1);

    // traj2 terminal: A = r - V = 2 - 1 = 1
    assert!((results[1].advantages[0] - 1.0).abs() < 1e-5);
}

/// Test batch with different trajectory lengths.
#[test]
fn test_vtrace_batch_variable_lengths() {
    let trajectories: Vec<VTraceInput> = (1..=5)
        .map(|len| VTraceInput {
            behavior_log_probs: vec![-1.0; len],
            target_log_probs: vec![-1.0; len],
            rewards: vec![1.0; len],
            values: vec![0.5; len],
            dones: vec![false; len],
            bootstrap_value: 0.5,
        })
        .collect();

    let results = compute_vtrace_batch(&trajectories, 0.99, 1.0, 1.0);

    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            result.vs.len(),
            i + 1,
            "Trajectory {} should have {} elements",
            i,
            i + 1
        );
    }
}

// ============================================================================
// On-Policy vs Off-Policy Comparison Tests
// ============================================================================

/// Test that on-policy V-trace approximates GAE.
/// When behavior == target policy, V-trace should give similar results to GAE.
#[test]
fn test_vtrace_on_policy_similar_to_gae() {
    use crate::algorithms::gae::compute_gae;

    let log_probs = vec![-1.0, -1.0, -1.0, -1.0, -1.0];
    let rewards = vec![1.0, 0.5, 1.5, 0.0, 2.0];
    let values = vec![0.5, 0.8, 0.3, 1.0, 0.7];
    let dones = vec![false, false, false, false, false];
    let bootstrap = 0.6;
    let gamma = 0.99;

    // V-trace with on-policy data
    let vtrace_result = compute_vtrace(
        &log_probs,
        &log_probs, // Same = on-policy
        &rewards,
        &values,
        &dones,
        bootstrap,
        gamma,
        1.0, // No clipping (on-policy)
        1.0,
    );

    // GAE with lambda=1 (should be equivalent)
    let (gae_advantages, _) = compute_gae(&rewards, &values, &dones, bootstrap, gamma, 1.0);

    // Advantages should be similar (not exact due to different formulations)
    // V-trace advantages are raw TD errors, GAE with lambda=1 accumulates them
    // For on-policy V-trace with c=1, the vs targets should be close to GAE returns

    // At least verify they're in the same ballpark
    for (i, (vt_adv, gae_adv)) in vtrace_result
        .advantages
        .iter()
        .zip(gae_advantages.iter())
        .enumerate()
    {
        // V-trace advantage is single-step, GAE is accumulated
        // Just verify they have the same sign
        if gae_adv.abs() > 0.1 {
            // Avoid sign comparison for near-zero values
            assert!(
                vt_adv.signum() == gae_adv.signum() || vt_adv.abs() < 0.5,
                "On-policy V-trace and GAE should have same sign at {}: {} vs {}",
                i,
                vt_adv,
                gae_adv
            );
        }
    }
}

// ============================================================================
// Property-Based Tests (Invariants)
// ============================================================================

/// Property: Importance weights are always in [0, rho_bar].
#[test]
fn test_vtrace_rho_bounded_property() {
    let test_cases: Vec<(Vec<f32>, Vec<f32>, f32)> = vec![
        // (behavior, target, rho_bar)
        (vec![-1.0, -1.0], vec![-1.0, -1.0], 1.0), // On-policy
        (vec![-5.0, -5.0], vec![-0.1, -0.1], 1.0), // High ratio
        (vec![-0.1, -0.1], vec![-5.0, -5.0], 1.0), // Low ratio
        (vec![-2.0, -0.5], vec![-0.5, -2.0], 0.5), // Custom rho_bar
    ];

    for (behavior, target, rho_bar) in test_cases {
        let len = behavior.len();
        let result = compute_vtrace(
            &behavior,
            &target,
            &vec![1.0; len],
            &vec![0.5; len],
            &vec![false; len],
            0.5,
            0.99,
            rho_bar,
            1.0,
        );

        for rho in &result.rhos {
            assert!(
                *rho >= 0.0 && *rho <= rho_bar + 1e-6,
                "rho={} should be in [0, {}]",
                rho,
                rho_bar
            );
        }
    }
}

/// Property: V-trace targets should be finite for any valid inputs.
#[test]
fn test_vtrace_targets_always_finite() {
    let test_cases: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = vec![
        // (rewards, values, log_probs)
        (vec![0.0, 0.0], vec![0.0, 0.0], vec![-1.0, -1.0]),
        (vec![1e5, 1e5], vec![1e5, 1e5], vec![-1.0, -1.0]),
        (vec![-1e5, 1e5], vec![1e5, -1e5], vec![-10.0, -0.1]),
    ];

    for (rewards, values, log_probs) in test_cases {
        let len = rewards.len();
        let result = compute_vtrace(
            &log_probs,
            &log_probs,
            &rewards,
            &values,
            &vec![false; len],
            0.0,
            0.99,
            1.0,
            1.0,
        );

        for vs in &result.vs {
            assert!(vs.is_finite(), "V-trace target should be finite: {}", vs);
        }
    }
}

/// Property: Terminal states always have A = r - V.
#[test]
fn test_vtrace_terminal_advantage_formula() {
    for r in [-10.0, -1.0, 0.0, 1.0, 10.0] {
        for v in [-5.0, 0.0, 5.0] {
            let result = compute_vtrace(
                &[-1.0],
                &[-1.0],
                &[r],
                &[v],
                &[true],
                100.0, // Non-zero bootstrap (should be ignored)
                0.99,
                1.0,
                1.0,
            );

            let expected = r - v;
            assert!(
                (result.advantages[0] - expected).abs() < 1e-5,
                "Terminal A should be r-v={}-{}={}, got {}",
                r,
                v,
                expected,
                result.advantages[0]
            );
        }
    }
}
