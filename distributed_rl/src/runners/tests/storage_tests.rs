//! Rollout storage tests.
//!
//! These tests define the correct behavior of rollout storage and minibatch generation.
//!
//! # Intent
//!
//! RolloutStorage is the bridge between actor data collection and learner training.
//! It must correctly:
//! - Store transitions in step-major order (all envs at step 0, then step 1, etc.)
//! - Track step counts and detect when full
//! - Extract data as correctly shaped tensors
//! - Generate shuffled minibatches for training
//! - Prepare sequences for recurrent training with proper terminal handling

use crate::runners::rollout_storage::{
    extract_minibatch, generate_minibatches, ComputedValues, MinibatchIndices, RolloutStorage,
    Sequence,
};
use crate::algorithms::action_policy::DiscreteAction;

// ============================================================================
// Basic Storage Tests
// ============================================================================

/// Test that new storage is empty and not full.
/// INTENT: Fresh storage should have no data.
#[test]
fn test_storage_new_is_empty() {
    let storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(4, 10, 8);

    assert!(storage.is_empty(), "New storage should be empty");
    assert!(!storage.is_full(), "New storage should not be full");
    assert_eq!(storage.len(), 0, "New storage length should be 0");
    assert_eq!(storage.step_count, 0, "New storage step_count should be 0");
    assert_eq!(storage.n_envs, 4);
    assert_eq!(storage.rollout_len, 10);
    assert_eq!(storage.obs_size, 8);
}

/// Test that pushing steps updates storage state correctly.
/// INTENT: Each push_step adds exactly n_envs transitions.
#[test]
fn test_storage_push_step_updates_state() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 4);

    // Push first step
    storage.push_step(
        &[0.0; 8], // 2 envs * 4 obs_size
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[false, false],
        &[false, false],
        &[0.5, 0.6],
        &[-0.3, -0.4],
        (),
    );

    assert_eq!(storage.step_count, 1, "step_count should be 1 after first push");
    assert_eq!(storage.len(), 2, "len should be 2 (1 step * 2 envs)");
    assert!(!storage.is_empty(), "Storage should not be empty");
    assert!(!storage.is_full(), "Storage should not be full yet");
}

/// Test that storage becomes full after rollout_len steps.
/// INTENT: is_full() should trigger learner to consume.
#[test]
fn test_storage_becomes_full_at_rollout_length() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 4);

    for _ in 0..3 {
        storage.push_step(
            &[0.0; 8],
            vec![DiscreteAction(0), DiscreteAction(1)],
            &[1.0, 2.0],
            &[false, false],
            &[false, false],
            &[0.5, 0.6],
            &[-0.3, -0.4],
            (),
        );
    }

    assert!(storage.is_full(), "Storage should be full after rollout_len steps");
    assert_eq!(storage.step_count, 3);
    assert_eq!(storage.len(), 6); // 3 steps * 2 envs
}

/// Test that clear resets storage to empty state.
/// INTENT: After clear, storage should behave like new.
#[test]
fn test_storage_clear_resets_to_empty() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 4);

    storage.push_step(
        &[0.0; 8],
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[false, false],
        &[false, false],
        &[0.5, 0.6],
        &[-0.3, -0.4],
        (),
    );

    assert!(!storage.is_empty());

    storage.clear();

    assert!(storage.is_empty(), "Storage should be empty after clear");
    assert_eq!(storage.step_count, 0);
    assert_eq!(storage.len(), 0);
    assert!(storage.states.is_empty());
    assert!(storage.actions.is_empty());
    assert!(storage.rewards.is_empty());
}

// ============================================================================
// Data Layout Tests
// ============================================================================

/// Test that data is stored in step-major interleaved order.
/// INTENT: Layout should be [env0_t0, env1_t0, env0_t1, env1_t1, ...]
/// This is critical for correct extraction later.
#[test]
fn test_storage_interleaved_layout() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 2, 1);

    // Step 0: env0 gets reward 1.0, env1 gets reward 2.0
    storage.push_step(
        &[0.0, 0.1],              // obs: env0=0.0, env1=0.1
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],               // rewards
        &[false, false],
        &[false, false],
        &[0.5, 0.6],
        &[-0.3, -0.4],
        (),
    );

    // Step 1: env0 gets reward 3.0, env1 gets reward 4.0
    storage.push_step(
        &[1.0, 1.1],              // obs: env0=1.0, env1=1.1
        vec![DiscreteAction(2), DiscreteAction(3)],
        &[3.0, 4.0],               // rewards
        &[false, true],
        &[false, true],
        &[0.7, 0.8],
        &[-0.5, -0.6],
        (),
    );

    // Check interleaved order: [env0_t0, env1_t0, env0_t1, env1_t1]
    assert_eq!(storage.rewards, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(storage.states, vec![0.0, 0.1, 1.0, 1.1]);
    assert_eq!(storage.values, vec![0.5, 0.6, 0.7, 0.8]);
    assert_eq!(storage.dones, vec![false, false, false, true]);

    // Actions should also be interleaved
    assert_eq!(storage.actions[0].0, 0); // env0_t0
    assert_eq!(storage.actions[1].0, 1); // env1_t0
    assert_eq!(storage.actions[2].0, 2); // env0_t1
    assert_eq!(storage.actions[3].0, 3); // env1_t1
}

/// Test rewards_by_step extracts correctly from interleaved data.
/// INTENT: rewards_by_step[step][env] should give correct value.
#[test]
fn test_storage_rewards_by_step_extraction() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 3, 1);

    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[false, false],
        &[false, false],
        &[0.5, 0.6],
        &[-0.3, -0.4],
        (),
    );
    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(1), DiscreteAction(0)],
        &[3.0, 4.0],
        &[false, true],
        &[false, true],
        &[0.7, 0.8],
        &[-0.5, -0.6],
        (),
    );

    let rewards_by_step = storage.rewards_by_step();

    assert_eq!(rewards_by_step.len(), 2, "Should have 2 steps");
    assert_eq!(rewards_by_step[0], vec![1.0, 2.0], "Step 0 rewards");
    assert_eq!(rewards_by_step[1], vec![3.0, 4.0], "Step 1 rewards");
}

/// Test values_by_step extraction.
#[test]
fn test_storage_values_by_step_extraction() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 2, 1);

    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[false, false],
        &[false, false],
        &[0.1, 0.2],
        &[-0.3, -0.4],
        (),
    );
    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(1), DiscreteAction(0)],
        &[3.0, 4.0],
        &[false, true],
        &[false, true],
        &[0.3, 0.4],
        &[-0.5, -0.6],
        (),
    );

    let values_by_step = storage.values_by_step();

    assert_eq!(values_by_step[0], vec![0.1, 0.2]);
    assert_eq!(values_by_step[1], vec![0.3, 0.4]);
}

/// Test dones_by_step extraction.
#[test]
fn test_storage_dones_by_step_extraction() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 2, 1);

    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[false, true],
        &[false, true],
        &[0.1, 0.2],
        &[-0.3, -0.4],
        (),
    );
    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(1), DiscreteAction(0)],
        &[3.0, 4.0],
        &[true, false],
        &[true, false],
        &[0.3, 0.4],
        &[-0.5, -0.6],
        (),
    );

    let dones_by_step = storage.dones_by_step();

    assert_eq!(dones_by_step[0], vec![false, true]);
    assert_eq!(dones_by_step[1], vec![true, false]);
}

// ============================================================================
// Minibatch Generation Tests
// ============================================================================

/// Test that generate_minibatches produces correct batch count.
/// INTENT: Should produce ceil(total / minibatch_size) batches.
#[test]
fn test_generate_minibatches_count() {
    let batches = generate_minibatches(100, 32);

    // 100 / 32 = 3 full batches + 1 remainder = 4 total
    assert_eq!(batches.len(), 4);
}

/// Test that minibatch sizes are correct.
/// INTENT: First batches should be full, last may be smaller.
#[test]
fn test_generate_minibatches_sizes() {
    let batches = generate_minibatches(100, 32);

    assert_eq!(batches[0].len(), 32, "First batch should be full");
    assert_eq!(batches[1].len(), 32, "Second batch should be full");
    assert_eq!(batches[2].len(), 32, "Third batch should be full");
    assert_eq!(batches[3].len(), 4, "Last batch should have remainder");
}

/// Test that all indices are covered exactly once.
/// INTENT: Minibatches should partition the full set.
#[test]
fn test_generate_minibatches_covers_all_indices() {
    let batches = generate_minibatches(100, 32);

    let mut all_indices: Vec<usize> = batches.iter().flat_map(|b| b.indices.clone()).collect();
    all_indices.sort();

    let expected: Vec<usize> = (0..100).collect();
    assert_eq!(
        all_indices, expected,
        "All indices should appear exactly once"
    );
}

/// Test that minibatches are shuffled.
/// INTENT: Shuffling prevents learning order-dependent patterns.
#[test]
fn test_generate_minibatches_are_shuffled() {
    // Generate multiple times and check that order varies
    // (This is probabilistic but highly likely to show variance)
    let batches1 = generate_minibatches(100, 32);
    let batches2 = generate_minibatches(100, 32);

    let order1: Vec<usize> = batches1.iter().flat_map(|b| b.indices.clone()).collect();
    let order2: Vec<usize> = batches2.iter().flat_map(|b| b.indices.clone()).collect();

    // Very unlikely to be identical if shuffled
    assert_ne!(
        order1, order2,
        "Two shuffles should (very likely) produce different orders"
    );
}

/// Test minibatch generation with exact division.
#[test]
fn test_generate_minibatches_exact_division() {
    let batches = generate_minibatches(64, 32);

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].len(), 32);
    assert_eq!(batches[1].len(), 32);
}

/// Test minibatch generation with single batch.
#[test]
fn test_generate_minibatches_single_batch() {
    let batches = generate_minibatches(10, 32);

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 10);
}

/// Test minibatch generation with empty input.
#[test]
fn test_generate_minibatches_empty() {
    let batches = generate_minibatches(0, 32);

    assert!(batches.is_empty());
}

// ============================================================================
// Minibatch Extraction Tests
// ============================================================================

/// Test that extract_minibatch correctly extracts data.
/// INTENT: Extracted data should match indexed positions in storage.
#[test]
fn test_extract_minibatch_correct_data() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 2, 2);

    // Push with identifiable data
    storage.push_step(
        &[0.0, 0.1, 1.0, 1.1], // env0: [0.0, 0.1], env1: [1.0, 1.1]
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[false, false],
        &[false, false],
        &[0.1, 0.2],
        &[-0.1, -0.2],
        (),
    );
    storage.push_step(
        &[2.0, 2.1, 3.0, 3.1], // env0: [2.0, 2.1], env1: [3.0, 3.1]
        vec![DiscreteAction(2), DiscreteAction(3)],
        &[3.0, 4.0],
        &[false, false],
        &[false, false],
        &[0.3, 0.4],
        &[-0.3, -0.4],
        (),
    );

    let computed = ComputedValues::new(vec![0.5, 0.6, 0.7, 0.8], vec![1.5, 1.6, 1.7, 1.8]);

    // Extract indices 0 and 2 (env0 at step 0 and env0 at step 1)
    let indices = MinibatchIndices::new(vec![0, 2]);
    let batch = extract_minibatch(&storage, &computed, &indices);

    assert_eq!(batch.batch_size(), 2);
    assert_eq!(batch.obs_size, 2);

    // Check states - should be [env0_t0, env0_t1]
    assert_eq!(batch.states, vec![0.0, 0.1, 2.0, 2.1]);

    // Check actions
    assert_eq!(batch.actions[0].0, 0);
    assert_eq!(batch.actions[1].0, 2);

    // Check computed values
    assert_eq!(batch.advantages, vec![0.5, 0.7]);
    assert_eq!(batch.returns, vec![1.5, 1.7]);
}

// ============================================================================
// Sequence Preparation Tests
// ============================================================================

/// Test Sequence struct construction.
#[test]
fn test_sequence_construction() {
    let seq = Sequence::new(1, 5, 10, 3);

    assert_eq!(seq.env_idx, 1);
    assert_eq!(seq.start_step, 5);
    assert_eq!(seq.end_step, 10);
    assert_eq!(seq.hidden_state_idx, 3);
    assert_eq!(seq.length, 5); // 10 - 5
}

/// Test sequence length calculation.
#[test]
fn test_sequence_length_calculation() {
    let test_cases = [(0, 1, 1), (0, 10, 10), (5, 15, 10), (0, 0, 0)];

    for (start, end, expected_len) in test_cases {
        let seq = Sequence::new(0, start, end, 0);
        assert_eq!(
            seq.length, expected_len,
            "Sequence from {} to {} should have length {}",
            start, end, expected_len
        );
    }
}

// ============================================================================
// ComputedValues Tests
// ============================================================================

/// Test ComputedValues construction.
#[test]
fn test_computed_values_construction() {
    let advantages = vec![1.0, 2.0, 3.0];
    let returns = vec![4.0, 5.0, 6.0];

    let computed = ComputedValues::new(advantages.clone(), returns.clone());

    assert_eq!(computed.advantages, advantages);
    assert_eq!(computed.returns, returns);
}

// ============================================================================
// Edge Cases
// ============================================================================

/// Test storage with single environment.
#[test]
fn test_storage_single_environment() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(1, 5, 4);

    for i in 0..5 {
        storage.push_step(
            &[i as f32; 4],
            vec![DiscreteAction(i as u32)],
            &[i as f32],
            &[false],
            &[false],
            &[0.5],
            &[-0.5],
            (),
        );
    }

    assert!(storage.is_full());
    assert_eq!(storage.len(), 5);

    let rewards = storage.rewards_by_step();
    for (i, step_rewards) in rewards.iter().enumerate() {
        assert_eq!(step_rewards.len(), 1);
        assert_eq!(step_rewards[0], i as f32);
    }
}

/// Test storage with single step.
#[test]
fn test_storage_single_step() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(4, 1, 8);

    storage.push_step(
        &[0.0; 32], // 4 envs * 8 obs_size
        vec![
            DiscreteAction(0),
            DiscreteAction(1),
            DiscreteAction(2),
            DiscreteAction(3),
        ],
        &[1.0, 2.0, 3.0, 4.0],
        &[false, false, false, false],
        &[false, false, false, false],
        &[0.1, 0.2, 0.3, 0.4],
        &[-0.1, -0.2, -0.3, -0.4],
        (),
    );

    assert!(storage.is_full());
    assert_eq!(storage.len(), 4);
    assert_eq!(storage.step_count, 1);
}

/// Test storage with all done flags set.
#[test]
fn test_storage_all_done() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 2, 1);

    storage.push_step(
        &[0.0; 2],
        vec![DiscreteAction(0), DiscreteAction(1)],
        &[1.0, 2.0],
        &[true, true], // All done
        &[true, true], // All terminal
        &[0.5, 0.6],
        &[-0.3, -0.4],
        (),
    );

    let dones = storage.dones_by_step();
    assert!(dones[0].iter().all(|&d| d), "All should be done");
}

/// Test storage with mixed terminal and truncated.
/// INTENT: Storage should correctly distinguish terminal from truncated.
#[test]
fn test_storage_mixed_terminal_truncated() {
    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(4, 1, 1);

    storage.push_step(
        &[0.0; 4],
        vec![
            DiscreteAction(0),
            DiscreteAction(1),
            DiscreteAction(2),
            DiscreteAction(3),
        ],
        &[1.0, 2.0, 3.0, 4.0],
        &[false, true, true, true], // dones: env 1,2,3 ended
        &[false, true, false, false], // terminals: only env 1 truly terminal
        &[0.1, 0.2, 0.3, 0.4],
        &[-0.1, -0.2, -0.3, -0.4],
    (),
    );

    // env 0: not done
    // env 1: terminal (done && terminal)
    // env 2: truncated (done && !terminal)
    // env 3: truncated (done && !terminal)

    assert_eq!(storage.dones, vec![false, true, true, true]);
    assert_eq!(storage.terminals, vec![false, true, false, false]);

    // Can compute which are truncated
    let truncated: Vec<bool> = storage
        .dones
        .iter()
        .zip(storage.terminals.iter())
        .map(|(&done, &terminal)| done && !terminal)
        .collect();
    assert_eq!(truncated, vec![false, false, true, true]);
}
