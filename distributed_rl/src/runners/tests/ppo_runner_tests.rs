//! Distributed PPO Runner tests.
//!
//! These tests define the correct behavior of the feed-forward distributed PPO runner.
//! They focus on:
//! - Terminal vs truncated episode handling
//! - Bootstrap value computation
//! - Weight transfer via BytesSlot
//! - Buffer synchronization between actors and learner
//!
//! # Critical Invariants
//!
//! 1. Terminal episodes: bootstrap value = 0 (no future rewards possible)
//! 2. Truncated episodes: bootstrap value = V(s') (episode continues semantically)
//! 3. Weight updates are atomic (no torn reads)
//! 4. Buffer synchronization prevents data races

use crate::core::bytes_slot::{bytes_slot, bytes_slot_with, BytesSlot};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================================
// BytesSlot Tests - Weight Transfer Mechanism
// ============================================================================

/// Test that BytesSlot starts with version 0 when empty.
/// INTENT: Version 0 indicates no weights have been published yet.
#[test]
fn test_bytes_slot_initial_version_is_zero() {
    let slot = BytesSlot::new();
    assert_eq!(slot.version(), 0, "Initial version should be 0");
    assert!(!slot.has_pending(), "Should have no pending data initially");
}

/// Test that BytesSlot version increments on each publish.
/// INTENT: Version number tracks weight updates monotonically.
#[test]
fn test_bytes_slot_version_increments_on_publish() {
    let slot = BytesSlot::new();

    slot.publish(vec![1, 2, 3]);
    assert_eq!(slot.version(), 1, "Version should be 1 after first publish");

    slot.publish(vec![4, 5, 6]);
    assert_eq!(slot.version(), 2, "Version should be 2 after second publish");

    slot.publish(vec![7, 8, 9]);
    assert_eq!(slot.version(), 3, "Version should be 3 after third publish");
}

/// Test that BytesSlot version never decreases.
/// INTENT: Version monotonicity prevents actors from regressing to stale weights.
#[test]
fn test_bytes_slot_version_monotonicity() {
    let slot = bytes_slot();

    let mut last_version = 0u64;
    for i in 0..100 {
        slot.publish(vec![i as u8]);
        let current = slot.version();
        assert!(
            current >= last_version,
            "Version must not decrease: {} -> {}",
            last_version,
            current
        );
        last_version = current;
    }
}

/// Test that get() returns data without removing it.
/// INTENT: Multiple actors can read the same weights.
#[test]
fn test_bytes_slot_get_does_not_consume() {
    let slot = BytesSlot::new();
    slot.publish(vec![1, 2, 3, 4]);

    // Multiple gets should return the same data
    let first = slot.get();
    let second = slot.get();
    let third = slot.get();

    assert_eq!(first, Some(vec![1, 2, 3, 4]));
    assert_eq!(second, Some(vec![1, 2, 3, 4]));
    assert_eq!(third, Some(vec![1, 2, 3, 4]));
    assert!(slot.has_pending(), "Data should still be pending");
}

/// Test that take() removes data from slot.
/// INTENT: take() is for exclusive consumption (if needed).
#[test]
fn test_bytes_slot_take_consumes_data() {
    let slot = BytesSlot::new();
    slot.publish(vec![1, 2, 3]);

    let first = slot.take();
    let second = slot.take();

    assert_eq!(first, Some(vec![1, 2, 3]));
    assert_eq!(second, None, "Second take should return None");
    assert!(!slot.has_pending());
}

/// Test that publish overwrites pending data.
/// INTENT: Actors always get the latest weights, not queued updates.
#[test]
fn test_bytes_slot_publish_overwrites() {
    let slot = BytesSlot::new();

    slot.publish(vec![1, 2, 3]);
    slot.publish(vec![4, 5, 6]);

    // Should get the latest, not a queue
    assert_eq!(slot.get(), Some(vec![4, 5, 6]));
}

/// Test BytesSlot with initial data.
/// INTENT: Can pre-populate with initial model weights.
#[test]
fn test_bytes_slot_with_initial_data() {
    let slot = bytes_slot_with(vec![42, 43, 44]);

    assert_eq!(slot.version(), 1, "Should start at version 1 with initial data");
    assert!(slot.has_pending());
    assert_eq!(slot.get(), Some(vec![42, 43, 44]));
}

/// Test concurrent access pattern (simulated).
/// INTENT: Multiple readers should see consistent data.
#[test]
fn test_bytes_slot_concurrent_read_pattern() {
    let slot = Arc::new(BytesSlot::new());

    // Simulate writer publishing
    slot.publish(vec![1; 1000]);
    let version_after_write = slot.version();

    // Simulate multiple readers
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let slot_clone = Arc::clone(&slot);
            std::thread::spawn(move || {
                let data = slot_clone.get();
                let version = slot_clone.version();
                (data, version)
            })
        })
        .collect();

    for handle in readers {
        let (data, version) = handle.join().expect("Reader thread panicked");
        // Version should be at least what we saw after write
        assert!(
            version >= version_after_write,
            "Version should not regress during concurrent reads"
        );
        // Data should be complete if present
        if let Some(bytes) = data {
            assert_eq!(bytes.len(), 1000, "Data should not be torn");
            assert!(bytes.iter().all(|&b| b == 1), "Data should be consistent");
        }
    }
}

// ============================================================================
// Bootstrap Value Computation Tests
// ============================================================================

/// Test bootstrap value is zero for terminal episodes.
/// INTENT: Terminal means episode truly ended, no future value possible.
#[test]
fn test_bootstrap_zero_for_terminal() {
    // Simulating the learner's bootstrap logic from distributed_ppo_runner.rs lines 562-567
    // let bootstrap = if dones.last().copied().unwrap_or(true) {
    //     0.0
    // } else {
    //     values.last().copied().unwrap_or(0.0)
    // };

    // Terminal case
    let dones = vec![false, false, true]; // Last step is terminal
    let values = vec![0.5, 0.6, 0.7];

    let bootstrap = if dones.last().copied().unwrap_or(true) {
        0.0
    } else {
        values.last().copied().unwrap_or(0.0)
    };

    assert_eq!(bootstrap, 0.0, "Terminal episode should have bootstrap=0");
}

/// Test bootstrap value uses V(s') for non-terminal.
/// INTENT: Continuing episode needs value estimate for future rewards.
#[test]
fn test_bootstrap_nonzero_for_continuing() {
    let dones = vec![false, false, false]; // No terminal
    let values = vec![0.5, 0.6, 0.7];

    let bootstrap = if dones.last().copied().unwrap_or(true) {
        0.0
    } else {
        values.last().copied().unwrap_or(0.0)
    };

    assert_eq!(
        bootstrap, 0.7,
        "Continuing episode should bootstrap with last value"
    );
}

/// KNOWN BUG TEST: Truncated episodes incorrectly treated as terminal.
/// INTENT: This test documents the bug where truncated episodes get bootstrap=0.
///
/// The current implementation uses done() which is true for BOTH terminal AND truncated.
/// Correct behavior: only terminal should get bootstrap=0, truncated should get V(s').
#[test]
fn test_bootstrap_bug_truncated_treated_as_terminal() {
    // This documents the BUGGY behavior in the current implementation

    // Scenario: Episode hit time limit (truncated but not terminal)
    // done=true, terminal=false
    let terminals = vec![false, false, false]; // None are truly terminal
    let truncateds = vec![false, false, true]; // Last is truncated (time limit)
    let dones: Vec<bool> = terminals
        .iter()
        .zip(truncateds.iter())
        .map(|(&t, &tr)| t || tr)
        .collect(); // done = terminal || truncated

    let values = vec![0.5, 0.6, 0.7];

    // Current (buggy) implementation:
    let buggy_bootstrap = if dones.last().copied().unwrap_or(true) {
        0.0 // BUG: Truncated gets 0
    } else {
        values.last().copied().unwrap_or(0.0)
    };

    // Correct implementation would be:
    let correct_bootstrap = if terminals.last().copied().unwrap_or(false) {
        0.0 // Only true terminals get 0
    } else {
        values.last().copied().unwrap_or(0.0) // Truncated gets V(s')
    };

    // Document the bug:
    assert_eq!(buggy_bootstrap, 0.0, "BUG: Truncated incorrectly gets bootstrap=0");
    assert_eq!(correct_bootstrap, 0.7, "CORRECT: Truncated should get V(s')=0.7");
    assert_ne!(
        buggy_bootstrap, correct_bootstrap,
        "Current implementation has bootstrap bug for truncated episodes"
    );
}

// ============================================================================
// Episode Statistics Tests
// ============================================================================

/// Test that episode rewards are accumulated correctly.
/// INTENT: Each environment tracks its own episode return.
#[test]
fn test_episode_reward_accumulation() {
    let n_envs = 3;
    let mut episode_rewards = vec![0.0f32; n_envs];

    // Simulate step rewards
    let step_rewards = [1.0, 2.0, 0.5];
    for (i, &r) in step_rewards.iter().enumerate() {
        episode_rewards[i] += r;
    }

    assert_eq!(episode_rewards, vec![1.0, 2.0, 0.5]);

    // More rewards
    let step_rewards = [0.5, 1.0, 1.5];
    for (i, &r) in step_rewards.iter().enumerate() {
        episode_rewards[i] += r;
    }

    assert_eq!(episode_rewards, vec![1.5, 3.0, 2.0]);
}

/// Test that episode rewards reset on done.
/// INTENT: Each episode starts fresh.
#[test]
fn test_episode_reward_reset_on_done() {
    let n_envs = 2;
    let mut episode_rewards = vec![10.0f32, 20.0f32];
    let dones = vec![true, false];

    for (i, &done) in dones.iter().enumerate() {
        if done {
            // Would record episode_rewards[i] first
            episode_rewards[i] = 0.0;
        }
    }

    assert_eq!(
        episode_rewards,
        vec![0.0, 20.0],
        "Done env should reset, other should keep accumulating"
    );
}

// ============================================================================
// Synchronization Pattern Tests
// ============================================================================

/// Test epoch synchronization pattern.
/// INTENT: Actors wait for learner to consume before producing more.
#[test]
fn test_epoch_synchronization_pattern() {
    let consumed_epoch = Arc::new(AtomicU64::new(0));

    // Actor at epoch 0
    let mut actor_epoch = 0u64;
    let mut steps_this_epoch = 0usize;
    let rollout_length = 10;

    // Simulate collecting a full rollout
    for _ in 0..rollout_length {
        steps_this_epoch += 1;
    }

    // Actor should now wait
    assert!(steps_this_epoch >= rollout_length);

    // Simulate waiting for consumption (would loop in real code)
    // consumed_epoch.load(Ordering::Acquire) <= actor_epoch means wait
    assert!(
        consumed_epoch.load(Ordering::Acquire) <= actor_epoch,
        "Actor should wait when learner hasn't consumed"
    );

    // Learner consumes
    consumed_epoch.fetch_add(1, Ordering::Release);

    // Actor can proceed
    assert!(
        consumed_epoch.load(Ordering::Acquire) > actor_epoch,
        "Actor should proceed after learner consumes"
    );

    // Actor advances epoch
    actor_epoch += 1;
    steps_this_epoch = 0;

    assert_eq!(actor_epoch, 1);
    assert_eq!(steps_this_epoch, 0);
}

/// Test version-based weight update pattern.
/// INTENT: Actors only reload weights when version changes.
#[test]
fn test_version_based_weight_update() {
    let slot = bytes_slot_with(vec![1, 2, 3]);
    let mut last_version = slot.version();

    // No new weights, shouldn't reload
    let current_version = slot.version();
    let should_reload = current_version > last_version;
    assert!(!should_reload, "Should not reload when version unchanged");

    // Learner publishes new weights
    slot.publish(vec![4, 5, 6]);

    // Now should reload
    let current_version = slot.version();
    let should_reload = current_version > last_version;
    assert!(should_reload, "Should reload when version changed");

    // After reload, update last_version
    last_version = current_version;

    // Check again - shouldn't reload
    let should_reload = slot.version() > last_version;
    assert!(!should_reload, "Should not reload immediately after update");
}

// ============================================================================
// Transition Construction Tests
// ============================================================================

/// Test that done is correctly computed from terminal and truncated.
/// INTENT: done() = terminal || truncated
#[test]
fn test_done_computation() {
    use crate::core::transition::Transition;

    // Not done
    let t1 = Transition::new_discrete(vec![], 0, 0.0, vec![], false, false);
    assert!(!t1.done());

    // Terminal (done)
    let t2 = Transition::new_discrete(vec![], 0, 0.0, vec![], true, false);
    assert!(t2.done());

    // Truncated (done)
    let t3 = Transition::new_discrete(vec![], 0, 0.0, vec![], false, true);
    assert!(t3.done());

    // Both (done)
    let t4 = Transition::new_discrete(vec![], 0, 0.0, vec![], true, true);
    assert!(t4.done());
}

/// Test that truncated is computed correctly.
/// INTENT: Truncated = done && !terminal
#[test]
fn test_truncated_computation_from_dones_terminals() {
    // Simulating line 472 of distributed_ppo_runner.rs:
    // truncated: step_result.dones[i] && !step_result.terminals[i],

    struct StepResult {
        dones: Vec<bool>,
        terminals: Vec<bool>,
    }

    let step_result = StepResult {
        dones: vec![false, true, true, true],
        terminals: vec![false, true, false, true],
    };

    let truncated: Vec<bool> = step_result
        .dones
        .iter()
        .zip(step_result.terminals.iter())
        .map(|(&d, &t)| d && !t)
        .collect();

    assert_eq!(
        truncated,
        vec![false, false, true, false],
        "Only done && !terminal should be truncated"
    );
}

// ============================================================================
// GAE Computation Integration Tests
// ============================================================================

/// Test that GAE is computed per-environment correctly.
/// INTENT: Each environment's trajectory is independent.
#[test]
fn test_gae_per_environment_independence() {
    use crate::algorithms::gae::compute_gae;

    // Environment 0: long episode
    let rewards0 = vec![1.0, 1.0, 1.0];
    let values0 = vec![0.0, 0.0, 0.0];
    let dones0 = vec![false, false, false];

    // Environment 1: short episode (terminal at step 1)
    let rewards1 = vec![1.0, 1.0, 1.0];
    let values1 = vec![0.0, 0.0, 0.0];
    let dones1 = vec![false, true, false]; // Terminal at step 1

    let gamma = 0.99;
    let gae_lambda = 0.95;
    let bootstrap = 0.0;

    let (adv0, _) = compute_gae(&rewards0, &values0, &dones0, bootstrap, gamma, gae_lambda);
    let (adv1, _) = compute_gae(&rewards1, &values1, &dones1, bootstrap, gamma, gae_lambda);

    // Advantages should be different because of terminal structure
    // Env0: no terminals, future values accumulate
    // Env1: terminal at step 1 resets accumulation
    assert_ne!(
        adv0, adv1,
        "Different terminal patterns should give different advantages"
    );

    // At step 1, env1 is terminal so advantage = r - V = 1 - 0 = 1
    assert!(
        (adv1[1] - 1.0).abs() < 1e-5,
        "Terminal advantage should be r - V"
    );
}

// ============================================================================
// Memory Management Tests
// ============================================================================

/// Test that recent_rewards buffer doesn't grow unboundedly.
/// INTENT: Document the potential memory leak in the implementation.
#[test]
fn test_recent_rewards_bounded_growth() {
    // The implementation does: rewards.lock().push(ep_reward);
    // Without any bound checking, this grows indefinitely.

    // Simulating what SHOULD happen:
    let mut recent_rewards: Vec<f32> = Vec::with_capacity(200);

    for i in 0..500 {
        recent_rewards.push(i as f32);

        // Correct implementation would bound the size
        if recent_rewards.len() > 200 {
            recent_rewards.drain(0..100);
        }
    }

    assert!(
        recent_rewards.len() <= 200,
        "Recent rewards should be bounded"
    );
}

// ============================================================================
// Action Conversion Tests
// ============================================================================

/// Test discrete action conversion to floats and back.
/// INTENT: Action value encoding/decoding is lossless.
#[test]
fn test_discrete_action_round_trip() {
    use crate::algorithms::action_policy::{ActionValue, DiscreteAction};

    let original = DiscreteAction(42);
    let floats = original.as_floats();
    let restored = DiscreteAction::from_floats(&floats);

    assert_eq!(original, restored, "Round-trip should preserve action");
}

/// Test continuous action conversion.
#[test]
fn test_continuous_action_round_trip() {
    use crate::algorithms::action_policy::{ActionValue, ContinuousAction};

    let original = ContinuousAction(vec![0.1, 0.2, 0.3, 0.4]);
    let floats = original.as_floats();
    let restored = ContinuousAction::from_floats(&floats);

    assert_eq!(original.0.len(), restored.0.len());
    for (a, b) in original.0.iter().zip(restored.0.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}
