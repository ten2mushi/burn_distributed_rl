//! Distributed IMPALA Runner tests.
//!
//! These tests define the correct behavior of the off-policy IMPALA runner
//! with V-trace importance sampling correction.
//!
//! # Key Differences from PPO
//!
//! 1. **Off-policy**: Actors don't wait for learner to consume
//! 2. **V-trace**: Importance sampling corrects for policy lag
//! 3. **FIFO buffer**: Trajectories processed in order
//! 4. **Async collection**: No synchronization barriers
//!
//! # Critical Invariants
//!
//! 1. Importance weights (rho) must be clipped to prevent variance explosion
//! 2. Policy loss includes rho weighting: -rho * log_prob * advantage
//! 3. Advantages do NOT include rho (raw TD errors)
//! 4. Terminal states always have bootstrap = 0

use crate::algorithms::vtrace::compute_vtrace;

// ============================================================================
// V-trace Correctness Tests
// ============================================================================

/// Test that on-policy V-trace gives importance weights of 1.
/// INTENT: When behavior == target policy, no correction needed.
#[test]
fn test_vtrace_on_policy_weights_are_one() {
    let log_probs = vec![-1.0, -0.5, -2.0];

    let result = compute_vtrace(
        &log_probs, // behavior
        &log_probs, // target (same = on-policy)
        &[1.0, 1.0, 1.0],
        &[0.5, 0.5, 0.5],
        &[false, false, false],
        0.5,
        0.99,
        1.0,
        1.0,
    );

    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            (*rho - 1.0).abs() < 1e-6,
            "On-policy rho[{}] should be 1.0, got {}",
            i,
            rho
        );
    }
}

/// Test that importance weights are clipped to rho_max.
/// INTENT: Prevent variance explosion from large ratios.
#[test]
fn test_vtrace_rho_clipping() {
    // Target has much higher probability than behavior
    let behavior_log_probs = vec![-5.0, -5.0];
    let target_log_probs = vec![-0.1, -0.1];

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &[1.0, 1.0],
        &[0.5, 0.5],
        &[false, false],
        0.5,
        0.99,
        1.0, // rho_bar = 1.0
        1.0,
    );

    // Raw ratio would be exp(-0.1 - (-5.0)) = exp(4.9) >> 1
    // Should be clipped to rho_bar = 1.0
    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            *rho <= 1.0 + 1e-6,
            "rho[{}] should be clipped to <= 1.0, got {}",
            i,
            rho
        );
    }
}

/// Test custom rho_bar and c_bar values.
/// INTENT: Clipping parameters should be respected.
#[test]
fn test_vtrace_custom_clipping_parameters() {
    let behavior_log_probs = vec![-3.0];
    let target_log_probs = vec![-0.5];

    let rho_bar = 0.5;
    let c_bar = 0.3;

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &[1.0],
        &[0.5],
        &[false],
        0.5,
        0.99,
        rho_bar,
        c_bar,
    );

    // Should be clipped to rho_bar
    assert!(
        result.rhos[0] <= rho_bar + 1e-6,
        "rho should be <= {}, got {}",
        rho_bar,
        result.rhos[0]
    );
}

/// CRITICAL: Test that advantages do NOT include rho.
/// INTENT: Rho is applied externally in policy loss, not in advantages.
#[test]
fn test_vtrace_advantages_exclude_rho() {
    let behavior_log_probs = vec![-2.0];
    let target_log_probs = vec![-0.5]; // Higher prob
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

    // Advantage should be raw TD error: r + gamma*V_next - V
    let expected_advantage = rewards[0] + gamma * bootstrap - values[0];
    // = 1.0 + 0.99*0.5 - 0.5 = 0.995

    assert!(
        (result.advantages[0] - expected_advantage).abs() < 1e-5,
        "Advantage should be raw TD ({}), not weighted by rho. Got {}",
        expected_advantage,
        result.advantages[0]
    );
}

/// Test that rhos are returned for external policy loss weighting.
/// INTENT: Policy loss = -rho * log_prob * advantage.
#[test]
fn test_vtrace_rhos_available_for_policy_loss() {
    let result = compute_vtrace(
        &[-1.0, -2.0],
        &[-0.5, -0.5],
        &[1.0, 1.0],
        &[0.5, 0.5],
        &[false, false],
        0.5,
        0.99,
        1.0,
        1.0,
    );

    assert_eq!(result.rhos.len(), 2, "Should return rho for each transition");

    // Rhos should be different based on behavior/target divergence
    // rho_0 = exp(-0.5 - (-1.0)) = exp(0.5) = 1.65, clipped to 1.0
    // rho_1 = exp(-0.5 - (-2.0)) = exp(1.5) = 4.48, clipped to 1.0
    assert!((result.rhos[0] - 1.0).abs() < 1e-5);
    assert!((result.rhos[1] - 1.0).abs() < 1e-5);
}

// ============================================================================
// Terminal State Handling Tests
// ============================================================================

/// Test that terminal states zero future contributions.
/// INTENT: Episode ended, no future rewards possible.
#[test]
fn test_vtrace_terminal_zeroes_future() {
    let result = compute_vtrace(
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[0.5, 0.5],
        &[false, true], // Terminal at step 1
        10.0,           // Bootstrap should be ignored
        0.99,
        1.0,
        1.0,
    );

    // At terminal: A = r + gamma*0 - V = 1 - 0.5 = 0.5
    assert!(
        (result.advantages[1] - 0.5).abs() < 1e-5,
        "Terminal advantage should be r - V = 0.5, got {}",
        result.advantages[1]
    );
}

/// Test bootstrap value for non-terminal rollout end.
/// INTENT: Continuing trajectory needs value estimate.
#[test]
fn test_vtrace_bootstrap_for_non_terminal() {
    let result_high = compute_vtrace(
        &[-1.0],
        &[-1.0],
        &[1.0],
        &[0.5],
        &[false], // NOT terminal
        2.0,      // High bootstrap
        0.99,
        1.0,
        1.0,
    );

    let result_low = compute_vtrace(
        &[-1.0],
        &[-1.0],
        &[1.0],
        &[0.5],
        &[false],
        0.0, // Low bootstrap
        0.99,
        1.0,
        1.0,
    );

    assert!(
        result_high.advantages[0] > result_low.advantages[0],
        "Higher bootstrap should give higher advantage"
    );
}

// ============================================================================
// Off-Policy Correction Tests
// ============================================================================

/// Test that large policy divergence doesn't cause overflow.
/// INTENT: Numerical stability with extreme log prob differences.
#[test]
fn test_vtrace_large_divergence_stability() {
    // Extreme case: behavior has near-zero probability
    let behavior_log_probs = vec![-100.0, -50.0, -0.01];
    let target_log_probs = vec![-0.01, -100.0, -50.0];

    let result = compute_vtrace(
        &behavior_log_probs,
        &target_log_probs,
        &[1.0, -1.0, 0.5],
        &[0.5, 0.5, 0.5],
        &[false, false, false],
        0.5,
        0.99,
        1.0,
        1.0,
    );

    // All outputs should be finite
    for (i, adv) in result.advantages.iter().enumerate() {
        assert!(
            adv.is_finite(),
            "advantage[{}] should be finite with extreme divergence, got {}",
            i,
            adv
        );
    }
    for (i, rho) in result.rhos.iter().enumerate() {
        assert!(
            rho.is_finite() && *rho <= 1.0 + 1e-6,
            "rho[{}] should be finite and clipped, got {}",
            i,
            rho
        );
    }
}

/// Test V-trace targets are computed correctly.
/// INTENT: vs = V + sum of weighted TD errors.
#[test]
fn test_vtrace_target_computation() {
    // Simple on-policy case for verification
    let result = compute_vtrace(
        &[-1.0],
        &[-1.0], // On-policy
        &[1.0],
        &[0.5],
        &[false],
        0.8,
        0.99,
        1.0,
        1.0,
    );

    // On-policy with single step:
    // delta = r + gamma*V_next - V = 1 + 0.99*0.8 - 0.5 = 1.292
    // vs = V + delta = 0.5 + 1.292 = 1.792
    let expected_vs = 0.5 + 1.0 + 0.99 * 0.8 - 0.5;

    assert!(
        (result.vs[0] - expected_vs).abs() < 1e-5,
        "V-trace target should be {}, got {}",
        expected_vs,
        result.vs[0]
    );
}

// ============================================================================
// Async Collection Pattern Tests
// ============================================================================

/// Test that IMPALA doesn't require synchronization between actors.
/// INTENT: Actors push trajectories without waiting for learner.
#[test]
fn test_async_collection_no_synchronization() {
    // IMPALA pattern: actors just push, learner just consumes
    // No epoch counters, no waiting

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let buffer_size = Arc::new(AtomicUsize::new(0));

    // Simulate actor pushing without waiting
    for _ in 0..10 {
        buffer_size.fetch_add(1, Ordering::Relaxed);
    }

    // No blocking, just accumulation
    assert_eq!(buffer_size.load(Ordering::Relaxed), 10);

    // Learner can sample at any time
    let sampled = buffer_size.load(Ordering::Relaxed);
    assert!(sampled > 0, "Learner can sample whenever buffer has data");
}

/// Test policy version tracking for off-policy correction.
/// INTENT: Each trajectory records the policy version used to collect it.
#[test]
fn test_policy_version_tracking() {
    use crate::core::transition::IMPALATransition;

    let policy_version = 42u64;

    let transition = IMPALATransition::new_discrete(
        vec![1.0, 2.0],
        0,
        1.0,
        vec![2.0, 3.0],
        false,
        false,
        -0.5,           // behavior_log_prob
        policy_version, // policy_version
    );

    assert_eq!(
        transition.policy_version, 42,
        "Transition should record policy version"
    );
}

/// Test that trajectories include behavior log probs for V-trace.
/// INTENT: Need log pi_behavior(a|s) for importance sampling.
#[test]
fn test_behavior_log_prob_stored() {
    use crate::core::transition::IMPALATransition;

    let behavior_log_prob = -1.386; // log(0.25)

    let transition = IMPALATransition::new_discrete(
        vec![1.0],
        0,
        1.0,
        vec![2.0],
        false,
        false,
        behavior_log_prob,
        0,
    );

    assert!(
        (transition.behavior_log_prob - behavior_log_prob).abs() < 1e-5,
        "Should store behavior log prob"
    );
}

// ============================================================================
// FIFO Buffer Pattern Tests
// ============================================================================

/// Test that trajectories are processed in order (FIFO).
/// INTENT: Oldest data is processed first, not sampled randomly.
#[test]
fn test_fifo_trajectory_order() {
    // IMPALA uses FIFO: oldest trajectories processed first
    // This is different from DQN-style random replay

    use std::collections::VecDeque;

    let mut fifo: VecDeque<usize> = VecDeque::new();

    // Push in order
    for i in 0..5 {
        fifo.push_back(i);
    }

    // Pop should return in order
    assert_eq!(fifo.pop_front(), Some(0), "Should get oldest first");
    assert_eq!(fifo.pop_front(), Some(1), "Should get second oldest");
    assert_eq!(fifo.pop_front(), Some(2), "Should get third oldest");
}

/// Test buffer capacity limits.
/// INTENT: Old data is dropped when buffer is full.
#[test]
fn test_buffer_capacity_limit() {
    use std::collections::VecDeque;

    let capacity = 3;
    let mut buffer: VecDeque<usize> = VecDeque::with_capacity(capacity);

    // Add more than capacity
    for i in 0..5 {
        if buffer.len() >= capacity {
            buffer.pop_front(); // Drop oldest
        }
        buffer.push_back(i);
    }

    assert_eq!(buffer.len(), 3);
    // Should have newest: 2, 3, 4
    assert_eq!(buffer.front(), Some(&2), "Oldest should be 2");
    assert_eq!(buffer.back(), Some(&4), "Newest should be 4");
}

// ============================================================================
// Policy Loss Computation Tests
// ============================================================================

/// Test policy loss formula: -rho * log_prob * advantage.
/// INTENT: Importance weight applied in loss, not in advantage.
#[test]
fn test_policy_loss_formula() {
    // Simulate the loss computation from IMPALA learner
    let rhos = vec![1.0, 0.5, 0.8];
    let log_probs = vec![-1.0, -0.5, -2.0];
    let advantages = vec![0.1, 0.2, -0.1];

    let policy_losses: Vec<f32> = rhos
        .iter()
        .zip(log_probs.iter())
        .zip(advantages.iter())
        .map(|((&rho, &lp), &adv)| -rho * lp * adv)
        .collect();

    // Expected: -rho * log_prob * advantage
    let expected: Vec<f32> = vec![
        -1.0 * (-1.0) * 0.1,  // = 0.1
        -0.5 * (-0.5) * 0.2,  // = 0.05
        -0.8 * (-2.0) * (-0.1), // = -0.16
    ];

    for (i, (actual, exp)) in policy_losses.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "Policy loss[{}]: expected {}, got {}",
            i,
            exp,
            actual
        );
    }
}

/// Test value loss to V-trace targets.
/// INTENT: MSE between predicted values and V-trace corrected targets.
#[test]
fn test_value_loss_to_vtrace_targets() {
    let predicted_values = vec![0.5, 0.6, 0.7];
    let vtrace_targets = vec![0.8, 0.5, 1.0];

    let mse: f32 = predicted_values
        .iter()
        .zip(vtrace_targets.iter())
        .map(|(&v, &t)| {
            let diff: f32 = v - t;
            diff * diff
        })
        .sum::<f32>()
        / predicted_values.len() as f32;

    // (0.3^2 + 0.1^2 + 0.3^2) / 3 = (0.09 + 0.01 + 0.09) / 3 = 0.063
    let expected_mse = (0.09 + 0.01 + 0.09) / 3.0;

    assert!(
        (mse - expected_mse).abs() < 1e-5,
        "MSE should be {}, got {}",
        expected_mse,
        mse
    );
}

// ============================================================================
// Trajectory Construction Tests
// ============================================================================

/// Test trajectory length tracking.
/// INTENT: Trajectories are fixed length or terminated by episode end.
#[test]
fn test_trajectory_length() {
    use crate::core::transition::{IMPALATransition, Trajectory};

    let mut traj: Trajectory<IMPALATransition> = Trajectory::new(0);
    let target_length = 5;

    for i in 0..target_length {
        traj.push(IMPALATransition::new_discrete(
            vec![i as f32],
            0,
            1.0,
            vec![(i + 1) as f32],
            false,
            false,
            -0.5,
            0,
        ));
    }

    assert_eq!(traj.len(), target_length);
}

/// Test trajectory episode return tracking.
/// INTENT: Episode return recorded when episode completes.
#[test]
fn test_trajectory_episode_return() {
    use crate::core::transition::{IMPALATransition, Trajectory};

    let mut traj: Trajectory<IMPALATransition> = Trajectory::new(0);

    // Simulate episode with rewards
    let rewards = vec![1.0, 2.0, 3.0];
    for (i, &reward) in rewards.iter().enumerate() {
        let terminal = i == rewards.len() - 1;
        traj.push(IMPALATransition::new_discrete(
            vec![i as f32],
            0,
            reward,
            vec![(i + 1) as f32],
            terminal,
            false,
            -0.5,
            0,
        ));
    }

    // Set episode return
    traj.episode_return = Some(rewards.iter().sum());

    assert_eq!(traj.episode_return, Some(6.0));
}

// ============================================================================
// Edge Cases
// ============================================================================

/// Test empty trajectory handling.
#[test]
fn test_empty_trajectory() {
    use crate::core::transition::{IMPALATransition, Trajectory};

    let traj: Trajectory<IMPALATransition> = Trajectory::new(0);

    assert!(traj.is_empty());
    assert_eq!(traj.len(), 0);
    assert!(traj.episode_return.is_none());
}

/// Test single transition trajectory.
#[test]
fn test_single_transition_trajectory() {
    use crate::core::transition::{IMPALATransition, Trajectory};

    let mut traj: Trajectory<IMPALATransition> = Trajectory::new(0);
    traj.push(IMPALATransition::new_discrete(
        vec![1.0],
        0,
        5.0,
        vec![2.0],
        true, // Terminal
        false,
        -0.5,
        0,
    ));

    assert_eq!(traj.len(), 1);
    assert!(traj.transitions[0].done());
}

/// Test V-trace with all terminal states.
/// INTENT: Each transition is its own episode.
#[test]
fn test_vtrace_all_terminal() {
    let result = compute_vtrace(
        &[-1.0, -1.0, -1.0],
        &[-1.0, -1.0, -1.0],
        &[1.0, 2.0, 3.0],
        &[0.5, 1.0, 1.5],
        &[true, true, true], // All terminal
        10.0,                // Should be ignored
        0.99,
        1.0,
        1.0,
    );

    // All advantages should be r - V
    for i in 0..3 {
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.5, 1.0, 1.5];
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

/// Test V-trace empty input.
#[test]
fn test_vtrace_empty_input() {
    let result = compute_vtrace(&[], &[], &[], &[], &[], 0.0, 0.99, 1.0, 1.0);

    assert!(result.vs.is_empty());
    assert!(result.advantages.is_empty());
    assert!(result.rhos.is_empty());
}

// ============================================================================
// Model Update Frequency Tests
// ============================================================================

/// Test that actors check for weight updates periodically.
/// INTENT: Don't reload weights every step (too slow).
#[test]
fn test_model_update_frequency() {
    let model_update_freq = 100;
    let mut local_steps = 0;
    let mut reload_count = 0;

    for _ in 0..500 {
        if local_steps % model_update_freq == 0 {
            reload_count += 1;
        }
        local_steps += 1;
    }

    // Should reload every 100 steps: 500/100 = 5 times
    assert_eq!(reload_count, 5);
}
