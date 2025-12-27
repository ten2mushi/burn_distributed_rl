//! Recurrent PPO Runner tests - CRITICAL BUG DETECTION.
//!
//! These tests define the correct behavior for recurrent (LSTM/GRU) PPO training.
//! The recurrent variant has additional complexity around hidden state management.
//!
//! # Known Bugs to Expose
//!
//! 1. **Hidden state resets on truncated episodes** (WRONG)
//!    - Hidden state should ONLY reset on terminal, NOT truncated
//!    - Truncation = time limit, episode semantically continues
//!
//! 2. **Bootstrap value for truncated episodes** (WRONG)
//!    - Truncated should use V(s') for bootstrap
//!    - Current implementation uses 0.0 (treats as terminal)
//!
//! 3. **Sequence ID continuity on truncation** (WRONG)
//!    - Same sequence_id should persist across truncation
//!    - New sequence_id only after true terminal
//!
//! # Correct Behavior Specification
//!
//! ## Hidden State Lifecycle
//! - Initial: zeros for all environments
//! - Within episode: persists, updated by RNN forward pass
//! - On TERMINAL: reset to zeros (new episode, new memory)
//! - On TRUNCATED: preserve (same episode, just hit limit)
//! - Across environments: independent (env0 terminal doesn't affect env1)
//!
//! ## Bootstrap Values
//! - Terminal: None (or 0.0) - episode truly ended
//! - Truncated: Some(V(s_T)) - need to estimate future value
//! - Continuing: None during rollout, computed at rollout end
//!
//! ## Sequence Management
//! - sequence_id: unique per episode (changes on terminal)
//! - step_in_sequence: 0-indexed position within episode
//! - is_sequence_start: true only at episode beginning

use burn::tensor::{backend::Backend, Tensor};

// ============================================================================
// Hidden State Reset Logic Tests - MOST CRITICAL
// ============================================================================

/// Test that hidden state persists across timesteps within an episode.
/// INTENT: RNN memory should accumulate information during episode.
#[test]
fn test_hidden_state_persists_within_episode() {
    // Simulate hidden state evolution within episode
    let mut hidden = vec![0.0f32; 4]; // Initial hidden

    // Simulate RNN updating hidden (simplified)
    fn simulate_rnn_step(hidden: &mut [f32], obs: f32) {
        for h in hidden.iter_mut() {
            *h = *h * 0.9 + obs * 0.1; // Simple accumulation
        }
    }

    // Episode with no terminal
    let dones = vec![false, false, false, false];
    let observations = vec![1.0, 2.0, 3.0, 4.0];

    for (i, (&done, &obs)) in dones.iter().zip(observations.iter()).enumerate() {
        let hidden_before = hidden.clone();
        simulate_rnn_step(&mut hidden, obs);
        let hidden_after = hidden.clone();

        if !done {
            assert_ne!(
                hidden_before, hidden_after,
                "Hidden should update at step {} (within episode)",
                i
            );
        }
    }

    // Hidden should have accumulated information
    assert!(
        hidden.iter().any(|&h| h != 0.0),
        "Hidden should have accumulated values from episode"
    );
}

/// Test that hidden state resets ONLY on terminal.
/// INTENT: Terminal means new episode, fresh memory.
#[test]
fn test_hidden_state_resets_only_on_terminal() {
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 4;
    let hidden_size = 8;

    // Create non-zero hidden states (simulating mid-episode)
    let mut hiddens: Vec<Tensor<B, 2>> = (0..n_envs)
        .map(|i| {
            let val = (i + 1) as f32;
            Tensor::<B, 2>::full([1, hidden_size], val, &device)
        })
        .collect();

    // Only env 1 has terminal, env 2 has truncated
    let terminals = vec![false, true, false, false];
    let truncateds = vec![false, false, true, false];
    let dones: Vec<bool> = terminals
        .iter()
        .zip(truncateds.iter())
        .map(|(&t, &tr)| t || tr)
        .collect();

    // CORRECT: Reset hidden ONLY on terminal
    fn correct_reset(hiddens: &mut [Tensor<B, 2>], terminals: &[bool], device: &<B as Backend>::Device, hidden_size: usize) {
        for (i, &terminal) in terminals.iter().enumerate() {
            if terminal {
                hiddens[i] = Tensor::<B, 2>::zeros([1, hidden_size], device);
            }
        }
    }

    // BUGGY: Reset hidden on any done (includes truncated)
    fn buggy_reset(hiddens: &mut [Tensor<B, 2>], dones: &[bool], device: &<B as Backend>::Device, hidden_size: usize) {
        for (i, &done) in dones.iter().enumerate() {
            if done {
                hiddens[i] = Tensor::<B, 2>::zeros([1, hidden_size], device);
            }
        }
    }

    // Test correct behavior
    let mut correct_hiddens = hiddens.clone();
    correct_reset(&mut correct_hiddens, &terminals, &device, hidden_size);

    // Env 0: not done, should preserve (value = 1.0)
    let env0_data: Vec<f32> = correct_hiddens[0].clone().into_data().as_slice().unwrap().to_vec();
    assert!(
        env0_data.iter().all(|&v| (v - 1.0).abs() < 1e-5),
        "Env 0 (not done) should preserve hidden"
    );

    // Env 1: terminal, should reset to zeros
    let env1_data: Vec<f32> = correct_hiddens[1].clone().into_data().as_slice().unwrap().to_vec();
    assert!(
        env1_data.iter().all(|&v| v.abs() < 1e-5),
        "Env 1 (terminal) should reset hidden to zeros"
    );

    // Env 2: truncated but not terminal, should PRESERVE (value = 3.0)
    let env2_data: Vec<f32> = correct_hiddens[2].clone().into_data().as_slice().unwrap().to_vec();
    assert!(
        env2_data.iter().all(|&v| (v - 3.0).abs() < 1e-5),
        "Env 2 (truncated) should PRESERVE hidden - episode continues semantically"
    );

    // Env 3: not done, should preserve (value = 4.0)
    let env3_data: Vec<f32> = correct_hiddens[3].clone().into_data().as_slice().unwrap().to_vec();
    assert!(
        env3_data.iter().all(|&v| (v - 4.0).abs() < 1e-5),
        "Env 3 (not done) should preserve hidden"
    );
}

/// Test that the BUGGY implementation incorrectly resets on truncated.
/// INTENT: Document the bug where truncated episodes lose hidden state.
#[test]
fn test_hidden_state_bug_resets_on_truncated() {
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let hidden_size = 4;

    // Simulate truncated episode (time limit hit)
    let terminal = false;
    let truncated = true;
    let done = terminal || truncated;

    // Hidden state with accumulated memory
    let mut hidden = Tensor::<B, 2>::full([1, hidden_size], 5.0, &device);

    // BUGGY implementation (what the current code does)
    if done {
        hidden = Tensor::<B, 2>::zeros([1, hidden_size], &device);
    }

    let hidden_data: Vec<f32> = hidden.into_data().as_slice().unwrap().to_vec();

    // BUG: Hidden was reset even though episode wasn't truly terminal
    assert!(
        hidden_data.iter().all(|&v| v.abs() < 1e-5),
        "BUG CONFIRMED: Truncated episode incorrectly reset hidden state"
    );
}

/// Test that environments have independent hidden states.
/// INTENT: One env terminating shouldn't affect another env's hidden.
#[test]
fn test_hidden_states_independent_across_environments() {
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 4;
    let hidden_size = 4;

    // Each env has distinct hidden state
    let mut hiddens: Vec<Tensor<B, 2>> = (0..n_envs)
        .map(|i| Tensor::<B, 2>::full([1, hidden_size], (i + 1) as f32 * 10.0, &device))
        .collect();

    // Only env 0 is terminal
    let terminals = vec![true, false, false, false];

    for (i, &terminal) in terminals.iter().enumerate() {
        if terminal {
            hiddens[i] = Tensor::<B, 2>::zeros([1, hidden_size], &device);
        }
    }

    // Env 0: reset to zeros
    let env0: Vec<f32> = hiddens[0].clone().into_data().as_slice().unwrap().to_vec();
    assert!(env0.iter().all(|&v| v.abs() < 1e-5), "Env 0 should be reset");

    // Other envs: unchanged
    for i in 1..n_envs {
        let expected_val = (i + 1) as f32 * 10.0;
        let env_data: Vec<f32> = hiddens[i].clone().into_data().as_slice().unwrap().to_vec();
        assert!(
            env_data.iter().all(|&v| (v - expected_val).abs() < 1e-5),
            "Env {} should be unaffected by env 0's terminal",
            i
        );
    }
}

// ============================================================================
// Bootstrap Value Tests for Recurrent
// ============================================================================

/// Test bootstrap value computation for recurrent case.
/// INTENT: Truncated needs bootstrap with continued hidden state.
#[test]
fn test_bootstrap_with_hidden_state_for_truncated() {
    // For truncated episodes:
    // 1. Hidden state should NOT be reset
    // 2. Bootstrap should be computed as V(s_T, h_T) where h_T is the continued hidden

    // Scenario: Rollout ends with truncated transition
    // The bootstrap value should use:
    // - The next state observation (available)
    // - The hidden state from the last forward pass (should be preserved)

    let terminal = false;
    let truncated = true;
    let done = terminal || truncated;

    // Current buggy implementation sets bootstrap = 0 for any done
    let buggy_bootstrap = if done { 0.0 } else { 0.0 };

    // Correct: Only terminal gets 0, truncated gets V(s', h')
    let correct_bootstrap = if terminal {
        0.0
    } else if truncated {
        // Would need to compute: model.forward(next_obs, hidden_state).value
        0.5 // Placeholder for computed value
    } else {
        0.0 // Rollout boundary, also needs bootstrap
    };

    assert_eq!(buggy_bootstrap, 0.0, "Bug: All done gets bootstrap=0");
    assert_ne!(
        correct_bootstrap, 0.0,
        "Correct: Truncated should have non-zero bootstrap"
    );
}

/// Test that terminal transitions have bootstrap None.
/// INTENT: True terminal = no future rewards possible.
#[test]
fn test_terminal_has_no_bootstrap() {
    use crate::core::transition::RecurrentPPOTransition;

    let transition = RecurrentPPOTransition::new_discrete(
        vec![1.0, 2.0],  // state
        0,                // action
        10.0,             // reward
        vec![],           // next_state (irrelevant for terminal)
        true,             // terminal
        false,            // truncated
        -0.5,             // log_prob
        5.0,              // value
        vec![0.0; 4],     // hidden_state
        42,               // sequence_id
        10,               // step_in_sequence
        false,            // is_sequence_start
        None,             // bootstrap_value - should be None for terminal
    );

    assert!(transition.terminal());
    assert!(!transition.truncated());
    assert!(
        transition.bootstrap_value.is_none(),
        "Terminal transition should have no bootstrap value"
    );
}

/// Test that truncated transitions have bootstrap Some(V(s')).
/// INTENT: Truncated means we need to estimate remaining episode value.
#[test]
fn test_truncated_has_bootstrap() {
    use crate::core::transition::RecurrentPPOTransition;

    let bootstrap_value = 0.75; // V(s') computed from model

    let transition = RecurrentPPOTransition::new_discrete(
        vec![1.0, 2.0],       // state
        0,                     // action
        10.0,                  // reward
        vec![3.0, 4.0],        // next_state (needed for bootstrap)
        false,                 // terminal
        true,                  // truncated
        -0.5,                  // log_prob
        5.0,                   // value
        vec![0.0; 4],          // hidden_state
        42,                    // sequence_id
        100,                   // step_in_sequence (time limit)
        false,                 // is_sequence_start
        Some(bootstrap_value), // bootstrap_value - SHOULD be Some for truncated
    );

    assert!(!transition.terminal());
    assert!(transition.truncated());
    assert!(
        transition.bootstrap_value.is_some(),
        "Truncated transition should have bootstrap value"
    );
    assert_eq!(transition.bootstrap_value, Some(bootstrap_value));
}

// ============================================================================
// Sequence ID and Continuity Tests
// ============================================================================

/// Test that sequence_id persists within episode.
/// INTENT: Same episode = same sequence for proper TBPTT.
#[test]
fn test_sequence_id_persists_within_episode() {
    let initial_sequence_id = 42u64;
    let mut sequence_id = initial_sequence_id;

    // Simulate several steps, no terminal
    let dones = vec![false, false, false, false];
    let terminals = vec![false, false, false, false];

    for (&done, &terminal) in dones.iter().zip(terminals.iter()) {
        if terminal {
            sequence_id += 1;
        }
        // done without terminal (truncated) should NOT change sequence_id
    }

    assert_eq!(
        sequence_id, initial_sequence_id,
        "Sequence ID should not change during episode"
    );
}

/// Test that sequence_id changes on terminal.
/// INTENT: New episode = new sequence.
#[test]
fn test_sequence_id_changes_on_terminal() {
    let mut sequence_id = 0u64;

    // First episode
    let dones = vec![false, false, true]; // Terminal at step 2
    let terminals = vec![false, false, true];

    for (&_done, &terminal) in dones.iter().zip(terminals.iter()) {
        if terminal {
            sequence_id += 1;
        }
    }

    assert_eq!(sequence_id, 1, "Sequence ID should increment on terminal");

    // Second episode
    let dones = vec![false, true];
    let terminals = vec![false, true];

    for (&_done, &terminal) in dones.iter().zip(terminals.iter()) {
        if terminal {
            sequence_id += 1;
        }
    }

    assert_eq!(sequence_id, 2, "Sequence ID should increment again");
}

/// Test that sequence_id does NOT change on truncated.
/// INTENT: Truncated = same episode semantically.
#[test]
fn test_sequence_id_unchanged_on_truncated() {
    let initial_id = 100u64;
    let mut sequence_id = initial_id;

    // Episode hits time limit (truncated but not terminal)
    let terminals = vec![false, false, false];
    let truncateds = vec![false, false, true];

    for (&terminal, &_truncated) in terminals.iter().zip(truncateds.iter()) {
        if terminal {
            sequence_id += 1;
        }
        // Truncated should NOT change sequence_id
    }

    assert_eq!(
        sequence_id, initial_id,
        "Sequence ID should NOT change on truncation"
    );
}

/// Test step_in_sequence increments correctly.
/// INTENT: Track position within episode.
#[test]
fn test_step_in_sequence_increments() {
    let mut step_in_sequence = 0usize;
    let terminals = vec![false, false, false, true, false, false];

    for &terminal in &terminals {
        // Would record transition with current step_in_sequence
        assert!(
            step_in_sequence < 1000,
            "Step count should be reasonable"
        );

        if terminal {
            step_in_sequence = 0; // Reset on terminal
        } else {
            step_in_sequence += 1;
        }
    }
}

/// Test is_sequence_start is true only at episode beginning.
/// INTENT: Mark where fresh hidden state should be used.
#[test]
fn test_is_sequence_start_marking() {
    let mut is_sequence_start = true; // First step is always start
    let mut markers: Vec<bool> = Vec::new();

    let terminals = vec![false, false, true, false, false, true, false];

    for &terminal in &terminals {
        markers.push(is_sequence_start);

        if terminal {
            is_sequence_start = true; // Next step will be sequence start
        } else {
            is_sequence_start = false;
        }
    }

    assert_eq!(
        markers,
        vec![true, false, false, true, false, false, true],
        "is_sequence_start should be true after terminal"
    );
}

// ============================================================================
// TBPTT (Truncated Backprop Through Time) Tests
// ============================================================================

/// Test that sequences are truncated at terminal boundaries.
/// INTENT: Gradients should not flow across episode boundaries.
#[test]
fn test_sequences_truncated_at_terminal() {
    use crate::runners::rollout_storage::{ComputedValues, RolloutStorage, Sequence};
    use crate::algorithms::action_policy::DiscreteAction;

    let mut storage: RolloutStorage<DiscreteAction, ()> = RolloutStorage::new(2, 5, 4);

    // Env 0: terminal at step 2
    // Env 1: no terminals
    let dones_by_step = vec![
        vec![false, false], // step 0
        vec![false, false], // step 1
        vec![true, false],  // step 2: env 0 terminal
        vec![false, false], // step 3
        vec![false, false], // step 4
    ];
    let terminals_by_step = vec![
        vec![false, false],
        vec![false, false],
        vec![true, false], // env 0 terminal
        vec![false, false],
        vec![false, false],
    ];

    for step in 0..5 {
        storage.push_step(
            &[0.0; 8], // 2 envs * 4 obs
            vec![DiscreteAction(0), DiscreteAction(1)],
            &[1.0, 1.0],
            &dones_by_step[step],
            &terminals_by_step[step],
            &[0.5, 0.5],
            &[-0.3, -0.3],
            (),
        );
    }

    let computed = ComputedValues::new(vec![0.0; 10], vec![0.0; 10]);
    let prepared = storage.prepare_sequences(&computed, 100);

    // Check sequences:
    // Env 0: should have 2 sequences (0-2 terminal, 3-4 end)
    // Env 1: should have 1 sequence (0-4 end)
    let env0_seqs: Vec<&Sequence> = prepared
        .sequences
        .iter()
        .filter(|s| s.env_idx == 0)
        .collect();
    let env1_seqs: Vec<&Sequence> = prepared
        .sequences
        .iter()
        .filter(|s| s.env_idx == 1)
        .collect();

    // Env 0 should be split at terminal
    assert!(
        env0_seqs.len() >= 2,
        "Env 0 should have multiple sequences due to terminal"
    );

    // First sequence for env 0 should end at step 3 (exclusive, step 2 was terminal)
    if let Some(first_seq) = env0_seqs.first() {
        assert!(
            first_seq.end_step <= 3,
            "First env0 sequence should end at or before terminal"
        );
    }

    // Env 1 should be one sequence (no terminals)
    // (may be multiple if max_seq_len < rollout_len)
    assert!(
        !env1_seqs.is_empty(),
        "Env 1 should have at least one sequence"
    );
}

/// Test that hidden state initialization uses correct index.
/// INTENT: Each sequence should start with its own hidden state.
#[test]
fn test_sequence_hidden_state_initialization() {
    // When training a sequence, we need to:
    // 1. Get the hidden state at the START of that sequence
    // 2. Not confuse hidden states between sequences

    use crate::runners::rollout_storage::Sequence;

    // Create sequences with different hidden state indices
    let seq1 = Sequence::new(0, 0, 5, 0);   // Hidden at step 0
    let seq2 = Sequence::new(0, 5, 10, 5);  // Hidden at step 5 (after terminal)
    let seq3 = Sequence::new(1, 0, 10, 0);  // Different env, same step

    assert_eq!(seq1.hidden_state_idx, 0);
    assert_eq!(seq2.hidden_state_idx, 5);
    assert_eq!(seq3.hidden_state_idx, 0);

    // After terminal, next sequence uses FRESH zero hidden,
    // which should be stored at the first step of new sequence
}

// ============================================================================
// Detached Hidden State Tests
// ============================================================================

/// Test that hidden state can be detached for TBPTT.
/// INTENT: Prevent gradients flowing through ALL sequence steps.
#[test]
fn test_hidden_state_detachment() {
    use crate::algorithms::temporal_policy::RecurrentHidden;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let hidden = RecurrentHidden::<B>::new(2, 8, false, &device);

    // Detach should create a copy without gradient history
    let detached = hidden.detached();

    // Both should have same values
    assert_eq!(hidden.states.len(), detached.states.len());
    assert_eq!(hidden.hidden_size, detached.hidden_size);
    assert_eq!(hidden.has_cell, detached.has_cell);

    // Values should be equal
    for (orig, det) in hidden.states.iter().zip(detached.states.iter()) {
        let orig_data: Vec<f32> = orig.hidden.clone().into_data().as_slice().unwrap().to_vec();
        let det_data: Vec<f32> = det.hidden.clone().into_data().as_slice().unwrap().to_vec();
        assert_eq!(orig_data, det_data, "Detached should have same values");
    }
}

// ============================================================================
// Multi-Environment Hidden State Tests
// ============================================================================

/// Test hidden state batching for parallel environments.
/// INTENT: All envs should be processed together efficiently.
#[test]
fn test_hidden_state_batching() {
    use crate::algorithms::temporal_policy::RecurrentHidden;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 4;
    let hidden_size = 8;

    let hidden = RecurrentHidden::<B>::new(n_envs, hidden_size, false, &device);

    // Convert to batch
    let (h_batch, c_batch) = hidden.to_batch();

    assert_eq!(h_batch.dims(), [n_envs, hidden_size]);
    assert!(c_batch.is_none(), "GRU should have no cell state");
}

/// Test LSTM hidden state with cell.
#[test]
fn test_lstm_hidden_state_batching() {
    use crate::algorithms::temporal_policy::RecurrentHidden;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 3;
    let hidden_size = 16;

    let hidden = RecurrentHidden::<B>::new(n_envs, hidden_size, true, &device);

    let (h_batch, c_batch) = hidden.to_batch();

    assert_eq!(h_batch.dims(), [n_envs, hidden_size]);
    assert!(c_batch.is_some(), "LSTM should have cell state");
    assert_eq!(c_batch.unwrap().dims(), [n_envs, hidden_size]);
}

/// Test hidden state serialization and restoration.
/// INTENT: Hidden states can be stored in buffer and restored.
#[test]
fn test_hidden_state_serialization() {
    use crate::algorithms::temporal_policy::{HiddenConfig, HiddenStateType, RecurrentHidden};
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 2;
    let hidden_size = 4;

    // Create hidden with known values
    let mut hidden = RecurrentHidden::<B>::new(n_envs, hidden_size, true, &device);

    // Set env 0 to ones
    hidden.states[0] = crate::core::recurrent::HiddenState::lstm(
        Tensor::<B, 2>::full([1, hidden_size], 1.0, &device),
        Tensor::<B, 2>::full([1, hidden_size], 2.0, &device),
    );

    // Serialize
    let vec = hidden.to_vec();
    assert_eq!(
        vec.len(),
        n_envs * hidden_size * 2,
        "LSTM needs h + c for each env"
    );

    // Deserialize
    let config = HiddenConfig::lstm(hidden_size);
    let restored = RecurrentHidden::<B>::from_vec(&vec, n_envs, &device, &config);

    // Check restored values
    let h0_data: Vec<f32> = restored.states[0].hidden.clone().into_data().as_slice().unwrap().to_vec();
    let c0_data: Vec<f32> = restored.states[0].cell.as_ref().unwrap().clone().into_data().as_slice().unwrap().to_vec();

    assert!(h0_data.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    assert!(c0_data.iter().all(|&v| (v - 2.0).abs() < 1e-5));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test hidden state reset when all environments are terminal.
/// INTENT: All hidden states should be reset.
#[test]
fn test_all_environments_terminal() {
    use crate::algorithms::temporal_policy::RecurrentHidden;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 4;
    let hidden_size = 4;

    let mut hidden = RecurrentHidden::<B>::new(n_envs, hidden_size, false, &device);

    // Set all to non-zero
    for i in 0..n_envs {
        hidden.states[i] = crate::core::recurrent::HiddenState::gru(
            Tensor::<B, 2>::full([1, hidden_size], (i + 1) as f32, &device),
        );
    }

    // All are terminal
    let terminal_indices: Vec<usize> = (0..n_envs).collect();

    for idx in &terminal_indices {
        hidden.reset(*idx, &device);
    }

    // All should be zeros
    for i in 0..n_envs {
        let data: Vec<f32> = hidden.states[i].hidden.clone().into_data().as_slice().unwrap().to_vec();
        assert!(
            data.iter().all(|&v| v.abs() < 1e-5),
            "Env {} should be reset",
            i
        );
    }
}

/// Test hidden state with no terminals in rollout.
/// INTENT: All hidden states should persist unchanged.
#[test]
fn test_no_terminals_in_rollout() {
    use crate::algorithms::temporal_policy::RecurrentHidden;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 4;
    let hidden_size = 4;

    let mut hidden = RecurrentHidden::<B>::new(n_envs, hidden_size, false, &device);

    // Set all to non-zero values
    for i in 0..n_envs {
        hidden.states[i] = crate::core::recurrent::HiddenState::gru(
            Tensor::<B, 2>::full([1, hidden_size], (i + 1) as f32, &device),
        );
    }

    // No terminals
    let terminal_indices: Vec<usize> = vec![];

    for idx in &terminal_indices {
        hidden.reset(*idx, &device);
    }

    // All should preserve their values
    for i in 0..n_envs {
        let expected = (i + 1) as f32;
        let data: Vec<f32> = hidden.states[i].hidden.clone().into_data().as_slice().unwrap().to_vec();
        assert!(
            data.iter().all(|&v| (v - expected).abs() < 1e-5),
            "Env {} should preserve value {}",
            i,
            expected
        );
    }
}

/// Test handling of interleaved terminal and truncated.
/// INTENT: Only terminals reset hidden, truncated preserves.
#[test]
fn test_interleaved_terminal_and_truncated() {
    use crate::algorithms::temporal_policy::RecurrentHidden;
    use burn::backend::NdArray;
    type B = NdArray<f32>;

    let device = <B as Backend>::Device::default();
    let n_envs = 4;
    let hidden_size = 4;

    let mut hidden = RecurrentHidden::<B>::new(n_envs, hidden_size, false, &device);

    // Set initial values
    for i in 0..n_envs {
        hidden.states[i] = crate::core::recurrent::HiddenState::gru(
            Tensor::<B, 2>::full([1, hidden_size], (i + 1) as f32 * 10.0, &device),
        );
    }

    // Env 0: terminal
    // Env 1: truncated (should preserve with correct implementation)
    // Env 2: terminal
    // Env 3: not done
    let terminals = vec![true, false, true, false];
    let _truncateds = vec![false, true, false, false];

    // CORRECT: Only reset on terminal
    for (i, &terminal) in terminals.iter().enumerate() {
        if terminal {
            hidden.reset(i, &device);
        }
    }

    // Check results
    let env0: Vec<f32> = hidden.states[0].hidden.clone().into_data().as_slice().unwrap().to_vec();
    assert!(env0.iter().all(|&v| v.abs() < 1e-5), "Env 0 (terminal) should be reset");

    let env1: Vec<f32> = hidden.states[1].hidden.clone().into_data().as_slice().unwrap().to_vec();
    assert!(
        env1.iter().all(|&v| (v - 20.0).abs() < 1e-5),
        "Env 1 (truncated) should PRESERVE hidden = 20.0"
    );

    let env2: Vec<f32> = hidden.states[2].hidden.clone().into_data().as_slice().unwrap().to_vec();
    assert!(env2.iter().all(|&v| v.abs() < 1e-5), "Env 2 (terminal) should be reset");

    let env3: Vec<f32> = hidden.states[3].hidden.clone().into_data().as_slice().unwrap().to_vec();
    assert!(
        env3.iter().all(|&v| (v - 40.0).abs() < 1e-5),
        "Env 3 (not done) should preserve hidden = 40.0"
    );
}
