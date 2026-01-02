//! Comprehensive test suite for the buffers submodule.
//!
//! This test suite follows the "Tests as Definition: the Yoneda Way" philosophy,
//! providing complete behavioral specifications for:
//! - RolloutBuffer: On-policy PPO with vectorized environments
//! - SequenceBuffer: Recurrent PPO with TBPTT and hidden state management
//! - TrajectoryStore: IMPALA with lock-free injection and FIFO consumption
//!
//! Test categories:
//! 1. Configuration defaults and builders
//! 2. Basic operations (push, consume, sample)
//! 3. Interleaved data layout verification
//! 4. State management (ready flags, step counts)
//! 5. Edge cases (empty, single env, single step)
//! 6. Concurrency (multi-threaded push, Send+Sync)
//! 7. Capacity and eviction behavior

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::*;
use crate::core::transition::{
    Action, IMPALATransition, PPOTransition, RecurrentPPOTransition, Trajectory, Transition,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create n PPO transitions with distinguishable state values.
fn make_ppo_transitions(n: usize) -> Vec<PPOTransition> {
    (0..n)
        .map(|i| PPOTransition {
            base: Transition::new_discrete(
                vec![i as f32, (i * 10) as f32],
                (i % 4) as u32,
                (i as f32) * 0.1,
                vec![(i + 1) as f32, ((i + 1) * 10) as f32],
                false,
                false,
            ),
            log_prob: -(i as f32 + 1.0).ln(),
            value: i as f32 * 0.5,
            bootstrap_value: None,
        })
        .collect()
}

/// Create a PPO transition specifically tagged for a given environment and step.
/// The state encodes [env_id, step] for easy verification of interleaved layout.
fn make_ppo_transition_for_env(env_id: usize, step: usize) -> PPOTransition {
    PPOTransition {
        base: Transition::new_discrete(
            vec![env_id as f32, step as f32],
            (env_id % 4) as u32,
            (step as f32) * 0.1 + (env_id as f32) * 0.01,
            vec![(env_id + 1) as f32, (step + 1) as f32],
            false,
            false,
        ),
        log_prob: -0.5,
        value: 1.0 + (step as f32) * 0.1,
        bootstrap_value: None,
    }
}

/// Create a PPO transition marked as terminal.
fn make_terminal_ppo_transition(env_id: usize, step: usize) -> PPOTransition {
    PPOTransition {
        base: Transition::new_discrete(
            vec![env_id as f32, step as f32],
            0,
            10.0, // Terminal reward
            vec![0.0, 0.0],
            true,
            false,
        ),
        log_prob: -0.5,
        value: 0.0,
        bootstrap_value: None,
    }
}

/// Create a PPO transition marked as truncated.
fn make_truncated_ppo_transition(env_id: usize, step: usize) -> PPOTransition {
    PPOTransition {
        base: Transition::new_discrete(
            vec![env_id as f32, step as f32],
            0,
            1.0,
            vec![1.0, 1.0],
            false,
            true,
        ),
        log_prob: -0.5,
        value: 0.5, // Bootstrap value for truncation
        bootstrap_value: None,
    }
}

/// Create a recurrent PPO transition with specific env_id, step, and flags.
fn make_recurrent_transition(
    env_id: usize,
    step: usize,
    terminal: bool,
    seq_start: bool,
) -> RecurrentPPOTransition {
    let ppo = PPOTransition::new_discrete(
        vec![env_id as f32, step as f32],
        (step % 4) as u32,
        (step as f32) * 0.1,
        vec![(env_id + 1) as f32, (step + 1) as f32],
        terminal,
        false,
        -0.5,
        1.0,
    );
    RecurrentPPOTransition::from_ppo(
        ppo,
        vec![env_id as f32 * 0.1, step as f32 * 0.01], // Hidden state encodes position
        env_id as u64,
        step,
        seq_start,
    )
}

/// Create a recurrent transition marked as truncated.
fn make_truncated_recurrent_transition(env_id: usize, step: usize) -> RecurrentPPOTransition {
    let ppo = PPOTransition::new_discrete(
        vec![env_id as f32, step as f32],
        0,
        1.0,
        vec![1.0, 1.0],
        false,
        true,
        -0.5,
        0.5,
    );
    RecurrentPPOTransition::from_ppo(
        ppo,
        vec![0.0, 0.0],
        env_id as u64,
        step,
        false,
    )
}

/// Create an IMPALA trajectory with specified length and env_id.
fn make_impala_trajectory(len: usize, env_id: usize) -> Trajectory<IMPALATransition> {
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![env_id as f32, i as f32],
                (i % 4) as u32,
                (i as f32) * 0.1,
                vec![(env_id + 1) as f32, (i + 1) as f32],
                i == len - 1, // Last transition is terminal
                false,
            ),
            behavior_log_prob: -0.5 - (i as f32) * 0.01,
            policy_version: 1,
        });
    }
    traj
}

/// Create an IMPALA trajectory with specific policy version.
fn make_impala_trajectory_versioned(
    len: usize,
    env_id: usize,
    policy_version: u64,
) -> Trajectory<IMPALATransition> {
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![env_id as f32, i as f32],
                (i % 4) as u32,
                (i as f32) * 0.1,
                vec![(env_id + 1) as f32, (i + 1) as f32],
                i == len - 1,
                false,
            ),
            behavior_log_prob: -0.5,
            policy_version,
        });
    }
    traj
}

// =============================================================================
// ROLLOUT BUFFER CONFIG TESTS (~5 tests)
// =============================================================================

mod rollout_buffer_config_tests {
    use super::*;

    #[test]
    fn should_have_correct_default_values() {
        let config = RolloutBufferConfig::default();

        assert_eq!(config.n_envs, 64, "Default n_envs should be 64");
        assert_eq!(config.rollout_len, 128, "Default rollout_len should be 128");
    }

    #[test]
    fn should_calculate_correct_capacity() {
        let config = RolloutBufferConfig {
            n_envs: 8,
            rollout_len: 32,
        };
        let buffer = RolloutBuffer::new(config);

        // Capacity = n_envs * rollout_len
        assert_eq!(buffer.capacity(), 256);
    }

    #[test]
    fn should_allow_single_environment() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 10,
        };
        let buffer = RolloutBuffer::new(config);

        assert_eq!(buffer.capacity(), 10);
        assert_eq!(buffer.config().n_envs, 1);
    }

    #[test]
    fn should_allow_single_step_rollout() {
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        assert_eq!(buffer.capacity(), 4);
        assert_eq!(buffer.config().rollout_len, 1);
    }

    #[test]
    fn should_preserve_config_through_accessor() {
        let config = RolloutBufferConfig {
            n_envs: 16,
            rollout_len: 64,
        };
        let buffer = RolloutBuffer::new(config.clone());

        assert_eq!(buffer.config().n_envs, config.n_envs);
        assert_eq!(buffer.config().rollout_len, config.rollout_len);
    }
}

// =============================================================================
// ROLLOUT BUFFER TESTS (~20 tests)
// =============================================================================

mod rollout_buffer_tests {
    use super::*;

    #[test]
    fn should_start_empty_and_not_ready() {
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 10,
        };
        let buffer = RolloutBuffer::new(config);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.step_count(), 0);
        assert!(!buffer.is_ready());
    }

    #[test]
    fn should_track_step_count_not_transition_count() {
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 10,
        };
        let buffer = RolloutBuffer::new(config);

        // One step = 4 transitions (one per env)
        buffer.push_step(make_ppo_transitions(4));

        assert_eq!(buffer.step_count(), 1, "Step count should be 1");
        assert_eq!(buffer.len(), 4, "Transition count should be 4");
    }

    #[test]
    fn should_become_ready_after_rollout_len_steps() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_ppo_transitions(2));
        assert!(!buffer.is_ready(), "Should not be ready after step 1");

        buffer.push_step(make_ppo_transitions(2));
        assert!(!buffer.is_ready(), "Should not be ready after step 2");

        buffer.push_step(make_ppo_transitions(2));
        assert!(buffer.is_ready(), "Should be ready after step 3 (rollout_len)");
    }

    #[test]
    fn should_stay_ready_after_additional_steps() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));
        assert!(buffer.is_ready());

        // Push additional step
        buffer.push_step(make_ppo_transitions(2));
        assert!(buffer.is_ready(), "Should remain ready after extra steps");
        assert_eq!(buffer.step_count(), 3);
        assert_eq!(buffer.len(), 6);
    }

    #[test]
    fn should_reset_ready_flag_on_consume() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));
        assert!(buffer.is_ready());

        let _batch = buffer.consume();

        assert!(!buffer.is_ready(), "Ready flag should be false after consume");
    }

    #[test]
    fn should_clear_storage_on_consume() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));

        let _batch = buffer.consume();

        assert!(buffer.is_empty(), "Buffer should be empty after consume");
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn should_reset_step_count_on_consume() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));
        assert_eq!(buffer.step_count(), 2);

        let _batch = buffer.consume();

        assert_eq!(buffer.step_count(), 0, "Step count should be 0 after consume");
    }

    #[test]
    fn should_allow_refill_after_consume() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        // First rollout
        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));
        let batch1 = buffer.consume();
        assert_eq!(batch1.len(), 4);

        // Second rollout
        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));
        assert!(buffer.is_ready());

        let batch2 = buffer.consume();
        assert_eq!(batch2.len(), 4);
    }

    #[test]
    fn should_timeout_when_not_ready() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 10,
        };
        let buffer = RolloutBuffer::new(config);

        // Only push 1 step, need 10 to be ready
        buffer.push_step(make_ppo_transitions(2));

        let result = buffer.wait_ready_timeout(Duration::from_millis(10));
        assert!(!result, "Should timeout when buffer is not ready");
    }

    #[test]
    fn should_signal_ready_immediately_when_already_ready() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = Arc::new(RolloutBuffer::new(config));

        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));

        // Buffer is already ready, wait should return immediately
        let result = buffer.wait_ready_timeout(Duration::from_millis(100));
        assert!(result, "Should return true when buffer is ready");
    }

    #[test]
    fn should_store_interleaved_layout() {
        let config = RolloutBufferConfig {
            n_envs: 3,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        // Step 0: [env0_t0, env1_t0, env2_t0]
        buffer.push_step(vec![
            make_ppo_transition_for_env(0, 0),
            make_ppo_transition_for_env(1, 0),
            make_ppo_transition_for_env(2, 0),
        ]);

        // Step 1: [env0_t1, env1_t1, env2_t1]
        buffer.push_step(vec![
            make_ppo_transition_for_env(0, 1),
            make_ppo_transition_for_env(1, 1),
            make_ppo_transition_for_env(2, 1),
        ]);

        let batch = buffer.consume();

        // Verify interleaved layout: [env0_t0, env1_t0, env2_t0, env0_t1, env1_t1, env2_t1]
        assert_eq!(batch.transitions[0].base.state[0], 0.0); // env0
        assert_eq!(batch.transitions[0].base.state[1], 0.0); // step0

        assert_eq!(batch.transitions[1].base.state[0], 1.0); // env1
        assert_eq!(batch.transitions[1].base.state[1], 0.0); // step0

        assert_eq!(batch.transitions[2].base.state[0], 2.0); // env2
        assert_eq!(batch.transitions[2].base.state[1], 0.0); // step0

        assert_eq!(batch.transitions[3].base.state[0], 0.0); // env0
        assert_eq!(batch.transitions[3].base.state[1], 1.0); // step1

        assert_eq!(batch.transitions[4].base.state[0], 1.0); // env1
        assert_eq!(batch.transitions[4].base.state[1], 1.0); // step1

        assert_eq!(batch.transitions[5].base.state[0], 2.0); // env2
        assert_eq!(batch.transitions[5].base.state[1], 1.0); // step1
    }

    #[test]
    fn should_handle_terminal_transitions_within_rollout() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        // Step 0: both normal
        buffer.push_step(vec![
            make_ppo_transition_for_env(0, 0),
            make_ppo_transition_for_env(1, 0),
        ]);

        // Step 1: env0 terminates
        buffer.push_step(vec![
            make_terminal_ppo_transition(0, 1),
            make_ppo_transition_for_env(1, 1),
        ]);

        // Step 2: env0 resets and continues, env1 continues
        buffer.push_step(vec![
            make_ppo_transition_for_env(0, 2),
            make_ppo_transition_for_env(1, 2),
        ]);

        let batch = buffer.consume();

        assert_eq!(batch.len(), 6);

        let dones = batch.dones();
        assert!(!dones[0]); // env0 step0
        assert!(!dones[1]); // env1 step0
        assert!(dones[2]);  // env0 step1 (terminal)
        assert!(!dones[3]); // env1 step1
        assert!(!dones[4]); // env0 step2
        assert!(!dones[5]); // env1 step2
    }

    #[test]
    fn should_handle_truncated_transitions() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(vec![make_ppo_transition_for_env(0, 0)]);
        buffer.push_step(vec![make_truncated_ppo_transition(0, 1)]);

        let batch = buffer.consume();

        let dones = batch.dones();
        assert!(!dones[0]);
        assert!(dones[1], "Truncated transition should report done=true");
    }

    #[test]
    fn should_preserve_log_probs_and_values() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        let transitions = vec![
            PPOTransition {
                base: Transition::new_discrete(vec![0.0], 0, 1.0, vec![1.0], false, false),
                log_prob: -0.1,
                value: 0.5,
                bootstrap_value: None,
            },
            PPOTransition {
                base: Transition::new_discrete(vec![1.0], 1, 2.0, vec![2.0], false, false),
                log_prob: -0.2,
                value: 0.6,
                bootstrap_value: None,
            },
        ];
        buffer.push_step(transitions);
        buffer.push_step(make_ppo_transitions(2));

        let batch = buffer.consume();

        assert!((batch.log_probs()[0] - (-0.1)).abs() < 1e-6);
        assert!((batch.log_probs()[1] - (-0.2)).abs() < 1e-6);
        assert!((batch.values()[0] - 0.5).abs() < 1e-6);
        assert!((batch.values()[1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn should_handle_continuous_actions() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        let transition = PPOTransition {
            base: Transition::new_continuous(
                vec![0.0, 1.0],
                vec![0.5, -0.5], // Continuous action
                1.0,
                vec![1.0, 2.0],
                false,
                false,
            ),
            log_prob: -0.5,
            value: 1.0,
            bootstrap_value: None,
        };
        buffer.push_step(vec![transition]);

        let batch = buffer.consume();

        match &batch.transitions[0].base.action {
            Action::Continuous(a) => {
                assert_eq!(a.len(), 2);
                assert!((a[0] - 0.5).abs() < 1e-6);
                assert!((a[1] - (-0.5)).abs() < 1e-6);
            }
            Action::Discrete(_) => panic!("Expected continuous action"),
        }
    }

    #[test]
    fn should_not_panic_on_empty_consume() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        // Consume empty buffer
        let batch = buffer.consume();

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn should_handle_single_env_single_step() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(vec![make_ppo_transition_for_env(0, 0)]);

        assert!(buffer.is_ready());

        let batch = buffer.consume();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.n_envs, 1);
        assert_eq!(batch.rollout_len, 1);
    }

    #[test]
    fn should_handle_large_state_dimensions() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        let large_state: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let transition = PPOTransition {
            base: Transition::new_discrete(
                large_state.clone(),
                0,
                1.0,
                large_state.clone(),
                false,
                false,
            ),
            log_prob: -0.5,
            value: 1.0,
            bootstrap_value: None,
        };
        buffer.push_step(vec![transition]);

        let batch = buffer.consume();

        assert_eq!(batch.transitions[0].base.state.len(), 1000);
        assert_eq!(batch.states().len(), 1000);
    }

    #[test]
    fn should_verify_ready_signal_via_channel() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = Arc::new(RolloutBuffer::new(config));
        let buffer_clone = Arc::clone(&buffer);

        // Spawn thread that will wait for ready
        let handle = thread::spawn(move || {
            buffer_clone.wait_ready_timeout(Duration::from_secs(1))
        });

        // Push steps in main thread
        thread::sleep(Duration::from_millis(10));
        buffer.push_step(make_ppo_transitions(2));
        buffer.push_step(make_ppo_transitions(2));

        let result = handle.join().unwrap();
        assert!(result, "Waiting thread should receive ready signal");
    }
}

// =============================================================================
// ROLLOUT BATCH TESTS (~12 tests)
// =============================================================================

mod rollout_batch_tests {
    use super::*;

    fn create_test_batch() -> RolloutBatch {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        for step in 0..3 {
            buffer.push_step(vec![
                make_ppo_transition_for_env(0, step),
                make_ppo_transition_for_env(1, step),
            ]);
        }

        buffer.consume()
    }

    #[test]
    fn should_return_correct_len() {
        let batch = create_test_batch();
        assert_eq!(batch.len(), 6); // 2 envs * 3 steps
    }

    #[test]
    fn should_return_correct_n_envs() {
        let batch = create_test_batch();
        assert_eq!(batch.n_envs, 2);
    }

    #[test]
    fn should_return_correct_rollout_len() {
        let batch = create_test_batch();
        assert_eq!(batch.rollout_len, 3);
    }

    #[test]
    fn should_iterate_env_transitions_correctly() {
        let batch = create_test_batch();

        // Check env 0
        let env0_transitions: Vec<_> = batch.env_transitions(0).collect();
        assert_eq!(env0_transitions.len(), 3);

        for (step, trans) in env0_transitions.iter().enumerate() {
            assert_eq!(trans.base.state[0], 0.0, "Should be env 0");
            assert_eq!(trans.base.state[1], step as f32, "Should be step {}", step);
        }

        // Check env 1
        let env1_transitions: Vec<_> = batch.env_transitions(1).collect();
        assert_eq!(env1_transitions.len(), 3);

        for (step, trans) in env1_transitions.iter().enumerate() {
            assert_eq!(trans.base.state[0], 1.0, "Should be env 1");
            assert_eq!(trans.base.state[1], step as f32, "Should be step {}", step);
        }
    }

    #[test]
    fn should_flatten_states_correctly() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(vec![
            make_ppo_transition_for_env(0, 0),
            make_ppo_transition_for_env(1, 0),
        ]);
        buffer.push_step(vec![
            make_ppo_transition_for_env(0, 1),
            make_ppo_transition_for_env(1, 1),
        ]);

        let batch = buffer.consume();
        let states = batch.states();

        // 4 transitions * 2 state dimensions = 8 floats
        assert_eq!(states.len(), 8);

        // First transition (env0, step0): state = [0.0, 0.0]
        assert_eq!(states[0], 0.0);
        assert_eq!(states[1], 0.0);

        // Second transition (env1, step0): state = [1.0, 0.0]
        assert_eq!(states[2], 1.0);
        assert_eq!(states[3], 0.0);
    }

    #[test]
    fn should_flatten_next_states_correctly() {
        let batch = create_test_batch();
        let next_states = batch.next_states();

        // 6 transitions * 2 dimensions = 12 floats
        assert_eq!(next_states.len(), 12);
    }

    #[test]
    fn should_extract_all_actions() {
        let batch = create_test_batch();
        let actions = batch.actions();

        assert_eq!(actions.len(), 6);
    }

    #[test]
    fn should_extract_all_rewards() {
        let batch = create_test_batch();
        let rewards = batch.rewards();

        assert_eq!(rewards.len(), 6);
    }

    #[test]
    fn should_extract_all_values() {
        let batch = create_test_batch();
        let values = batch.values();

        assert_eq!(values.len(), 6);
    }

    #[test]
    fn should_extract_all_log_probs() {
        let batch = create_test_batch();
        let log_probs = batch.log_probs();

        assert_eq!(log_probs.len(), 6);
    }

    #[test]
    fn should_report_empty_correctly() {
        let batch = RolloutBatch {
            transitions: vec![],
            n_envs: 0,
            rollout_len: 0,
        };

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn should_handle_empty_batch_accessors() {
        let batch = RolloutBatch {
            transitions: vec![],
            n_envs: 0,
            rollout_len: 0,
        };

        assert!(batch.states().is_empty());
        assert!(batch.next_states().is_empty());
        assert!(batch.rewards().is_empty());
        assert!(batch.values().is_empty());
        assert!(batch.log_probs().is_empty());
        assert!(batch.dones().is_empty());
    }
}

// =============================================================================
// SEQUENCE BUFFER CONFIG TESTS (~6 tests)
// =============================================================================

mod sequence_buffer_config_tests {
    use super::*;

    #[test]
    fn should_have_correct_default_values() {
        let config = SequenceBufferConfig::default();

        assert_eq!(config.n_envs, 64);
        assert_eq!(config.sequence_length, 16);
        assert_eq!(config.burn_in_length, 0);
        assert_eq!(config.max_sequences, 256);
    }

    #[test]
    fn should_create_with_new() {
        let config = SequenceBufferConfig::new(8, 32);

        assert_eq!(config.n_envs, 8);
        assert_eq!(config.sequence_length, 32);
        assert_eq!(config.burn_in_length, 0); // Default
        assert_eq!(config.max_sequences, 256); // Default
    }

    #[test]
    fn should_set_burn_in_with_builder() {
        let config = SequenceBufferConfig::new(8, 16).with_burn_in(4);

        assert_eq!(config.burn_in_length, 4);
        assert_eq!(config.n_envs, 8);
        assert_eq!(config.sequence_length, 16);
    }

    #[test]
    fn should_set_max_sequences_with_builder() {
        let config = SequenceBufferConfig::new(8, 16).with_max_sequences(100);

        assert_eq!(config.max_sequences, 100);
    }

    #[test]
    fn should_chain_builders() {
        let config = SequenceBufferConfig::new(4, 8)
            .with_burn_in(2)
            .with_max_sequences(50);

        assert_eq!(config.n_envs, 4);
        assert_eq!(config.sequence_length, 8);
        assert_eq!(config.burn_in_length, 2);
        assert_eq!(config.max_sequences, 50);
    }

    #[test]
    fn should_be_cloneable() {
        let config1 = SequenceBufferConfig::new(4, 8).with_burn_in(2);
        let config2 = config1.clone();

        assert_eq!(config1.n_envs, config2.n_envs);
        assert_eq!(config1.sequence_length, config2.sequence_length);
        assert_eq!(config1.burn_in_length, config2.burn_in_length);
        assert_eq!(config1.max_sequences, config2.max_sequences);
    }
}

// =============================================================================
// SEQUENCE BUFFER TESTS (~25 tests)
// =============================================================================

mod sequence_buffer_tests {
    use super::*;

    #[test]
    fn should_start_empty() {
        let config = SequenceBufferConfig::new(4, 8);
        let buffer = SequenceBuffer::new(config);

        assert_eq!(buffer.n_sequences(), 0);
        assert_eq!(buffer.n_in_progress(), 0);
        assert_eq!(buffer.n_completed(), 0);
    }

    #[test]
    fn should_create_sequence_at_chunk_length() {
        let config = SequenceBufferConfig::new(2, 4);
        let mut buffer = SequenceBuffer::new(config);

        // Push 4 transitions to env 0
        for step in 0..4 {
            buffer.push(0, make_recurrent_transition(0, step, false, step == 0));
        }

        assert_eq!(buffer.n_sequences(), 1, "Should have created 1 sequence");
        assert_eq!(buffer.n_in_progress(), 0, "No in-progress after chunk completion");
    }

    #[test]
    fn should_create_padded_sequence_on_episode_done() {
        let config = SequenceBufferConfig::new(2, 8); // sequence_length = 8
        let mut buffer = SequenceBuffer::new(config);

        // Push only 3 transitions, last one terminal
        for step in 0..3 {
            buffer.push(0, make_recurrent_transition(0, step, step == 2, step == 0));
        }

        assert_eq!(buffer.n_sequences(), 1);

        let seq = buffer.pop().unwrap();
        assert_eq!(seq.len(), 8, "Should be padded to sequence_length");
        assert_eq!(seq.n_valid(), 3, "Only 3 valid transitions");
    }

    #[test]
    fn should_track_valid_mask_correctly() {
        let config = SequenceBufferConfig::new(1, 5);
        let mut buffer = SequenceBuffer::new(config);

        // Push 2 transitions, terminal
        buffer.push(0, make_recurrent_transition(0, 0, false, true));
        buffer.push(0, make_recurrent_transition(0, 1, true, false));

        let seq = buffer.pop().unwrap();

        assert_eq!(seq.valid_mask.len(), 5);
        assert!(seq.valid_mask[0]);
        assert!(seq.valid_mask[1]);
        assert!(!seq.valid_mask[2]); // Padded
        assert!(!seq.valid_mask[3]); // Padded
        assert!(!seq.valid_mask[4]); // Padded
    }

    #[test]
    fn should_handle_multiple_environments_independently() {
        let config = SequenceBufferConfig::new(2, 4);
        let mut buffer = SequenceBuffer::new(config);

        // Push 4 to env 0, 2 to env 1
        for step in 0..4 {
            buffer.push(0, make_recurrent_transition(0, step, false, step == 0));
        }
        for step in 0..2 {
            buffer.push(1, make_recurrent_transition(1, step, false, step == 0));
        }

        assert_eq!(buffer.n_sequences(), 1, "Only env 0 should have completed a sequence");
        assert_eq!(buffer.n_in_progress(), 2, "Env 1 has 2 in-progress transitions");
    }

    #[test]
    fn should_finalize_all_in_progress() {
        let config = SequenceBufferConfig::new(2, 8);
        let mut buffer = SequenceBuffer::new(config);

        // Push partial episodes to both envs
        buffer.push(0, make_recurrent_transition(0, 0, false, true));
        buffer.push(0, make_recurrent_transition(0, 1, false, false));
        buffer.push(1, make_recurrent_transition(1, 0, false, true));

        assert_eq!(buffer.n_sequences(), 0);
        assert_eq!(buffer.n_in_progress(), 3);

        buffer.finalize_all();

        assert_eq!(buffer.n_sequences(), 2, "Both envs should have padded sequences");
        assert_eq!(buffer.n_in_progress(), 0);
    }

    #[test]
    fn should_store_initial_hidden_state() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        let t1 = make_recurrent_transition(0, 0, false, true);
        let expected_hidden = t1.hidden_state.clone();

        buffer.push(0, t1);
        buffer.push(0, make_recurrent_transition(0, 1, false, false));

        let seq = buffer.pop().unwrap();

        assert_eq!(seq.initial_hidden, expected_hidden);
    }

    #[test]
    fn should_update_hidden_for_continuation_chunks() {
        // When a chunk is completed and there are remaining transitions in the episode,
        // the hidden state from the last transition of that chunk becomes the initial
        // hidden for the next chunk. When the episode becomes empty after draining,
        // no hidden state is set (it will be captured from next sequence start).
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        // First chunk - two transitions
        let t1 = make_recurrent_transition(0, 0, false, true);
        let t2 = make_recurrent_transition(0, 1, false, false);

        buffer.push(0, t1.clone());
        buffer.push(0, t2.clone());
        // Creates first sequence, episode becomes empty

        // Second chunk - mark as new sequence start
        let t3 = make_recurrent_transition(0, 2, false, true);
        let t4 = make_recurrent_transition(0, 3, false, false);

        buffer.push(0, t3.clone());
        buffer.push(0, t4);

        let seq1 = buffer.pop().unwrap();
        let seq2 = buffer.pop().unwrap();

        // First sequence should have the hidden from t1 (the seq_start transition)
        assert_eq!(seq1.initial_hidden, t1.hidden_state);

        // Second sequence starts fresh with is_sequence_start=true
        // so it captures t3's hidden state
        assert_eq!(seq2.initial_hidden, t3.hidden_state);

        // The hiddens should reflect different starting points
        assert_eq!(seq1.sequence_id, 0);
        assert_eq!(seq2.sequence_id, 1);
    }

    #[test]
    fn should_carry_hidden_when_episode_continues_past_chunk() {
        // This test verifies hidden state carryover when a chunk is created
        // but the episode continues (remaining transitions in the episode)
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        // Push 5 transitions without terminal - creates 2 chunks with carryover
        let t1 = make_recurrent_transition(0, 0, false, true);
        let t2 = make_recurrent_transition(0, 1, false, false);
        let t3 = make_recurrent_transition(0, 2, false, false);
        let t4 = make_recurrent_transition(0, 3, false, false);
        let t5 = make_recurrent_transition(0, 4, false, false);

        // Push all 5 - this will create 2 full sequences immediately
        // First sequence created when t2 is pushed (2 transitions)
        // At that point, episode has 0 transitions left, so no hidden carryover
        // Second sequence created when t4 is pushed (2 more transitions)
        buffer.push(0, t1.clone());
        buffer.push(0, t2.clone()); // Creates seq 0, episode empty
        buffer.push(0, t3.clone()); // Starts new in-progress
        buffer.push(0, t4.clone()); // Creates seq 1, episode empty
        buffer.push(0, t5.clone()); // In-progress

        // Only 2 sequences should be complete
        assert_eq!(buffer.n_sequences(), 2);
        assert_eq!(buffer.n_in_progress(), 1);

        let seq1 = buffer.pop().unwrap();
        let seq2 = buffer.pop().unwrap();

        // First sequence initial_hidden from t1 (seq_start)
        assert_eq!(seq1.initial_hidden, t1.hidden_state);

        // Second sequence: the hidden state from t2 (last transition of chunk 1)
        // should be carried over as the initial_hidden for chunk 2.
        // This is critical for recurrent training - the RNN hidden state must
        // propagate across chunk boundaries for episodes that span multiple chunks.
        assert_eq!(seq2.initial_hidden, t2.hidden_state);
    }

    #[test]
    fn should_respect_max_sequences_limit() {
        let config = SequenceBufferConfig::new(1, 2).with_max_sequences(3);
        let mut buffer = SequenceBuffer::new(config);

        // Create 5 sequences
        for i in 0..10 {
            buffer.push(0, make_recurrent_transition(0, i, false, i % 2 == 0));
        }

        assert_eq!(buffer.n_sequences(), 3, "Should only keep max_sequences");
    }

    #[test]
    fn should_evict_oldest_when_over_capacity() {
        let config = SequenceBufferConfig::new(1, 2).with_max_sequences(2);
        let mut buffer = SequenceBuffer::new(config);

        // Create sequence 0
        buffer.push(0, make_recurrent_transition(0, 0, false, true));
        buffer.push(0, make_recurrent_transition(0, 1, false, false));

        // Create sequence 1
        buffer.push(0, make_recurrent_transition(0, 2, false, true));
        buffer.push(0, make_recurrent_transition(0, 3, false, false));

        // Create sequence 2 (should evict sequence 0)
        buffer.push(0, make_recurrent_transition(0, 4, false, true));
        buffer.push(0, make_recurrent_transition(0, 5, false, false));

        assert_eq!(buffer.n_sequences(), 2);

        // First sequence should be sequence_id 1 (0 was evicted)
        let seq = buffer.pop().unwrap();
        assert_eq!(seq.sequence_id, 1);
    }

    #[test]
    fn should_set_initial_hidden_externally() {
        let config = SequenceBufferConfig::new(2, 2);
        let mut buffer = SequenceBuffer::new(config);

        buffer.set_initial_hidden(0, vec![1.0, 2.0, 3.0]);

        // Push non-start transition (won't set initial hidden itself)
        let mut t = make_recurrent_transition(0, 0, false, false);
        t.is_sequence_start = false;
        buffer.push(0, t);
        buffer.push(0, make_recurrent_transition(0, 1, false, false));

        let seq = buffer.pop().unwrap();

        assert_eq!(seq.initial_hidden, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn should_pop_sequences_fifo() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        // Create 3 sequences
        for chunk in 0..3 {
            buffer.push(0, make_recurrent_transition(0, chunk * 2, false, true));
            buffer.push(0, make_recurrent_transition(0, chunk * 2 + 1, false, false));
        }

        let seq0 = buffer.pop().unwrap();
        let seq1 = buffer.pop().unwrap();
        let seq2 = buffer.pop().unwrap();

        assert_eq!(seq0.sequence_id, 0);
        assert_eq!(seq1.sequence_id, 1);
        assert_eq!(seq2.sequence_id, 2);
    }

    #[test]
    fn should_sample_minibatch_without_removing() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        // Create 3 sequences
        for chunk in 0..3 {
            buffer.push(0, make_recurrent_transition(0, chunk * 2, false, true));
            buffer.push(0, make_recurrent_transition(0, chunk * 2 + 1, false, false));
        }

        let sample = buffer.sample_minibatch(2);
        assert_eq!(sample.len(), 2);
        assert_eq!(buffer.n_sequences(), 3, "Should not remove sequences");
    }

    #[test]
    fn should_take_minibatch_and_remove() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        // Create 3 sequences
        for chunk in 0..3 {
            buffer.push(0, make_recurrent_transition(0, chunk * 2, false, true));
            buffer.push(0, make_recurrent_transition(0, chunk * 2 + 1, false, false));
        }

        let taken = buffer.take_minibatch(2);
        assert_eq!(taken.len(), 2);
        assert_eq!(buffer.n_sequences(), 1, "Should have removed 2 sequences");
    }

    #[test]
    fn should_report_has_minibatch_correctly() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        assert!(!buffer.has_minibatch(1));

        buffer.push(0, make_recurrent_transition(0, 0, false, true));
        buffer.push(0, make_recurrent_transition(0, 1, false, false));

        assert!(buffer.has_minibatch(1));
        assert!(!buffer.has_minibatch(2));
    }

    #[test]
    fn should_clear_all_data() {
        let config = SequenceBufferConfig::new(2, 4);
        let mut buffer = SequenceBuffer::new(config);

        // Add completed and in-progress
        for step in 0..4 {
            buffer.push(0, make_recurrent_transition(0, step, false, step == 0));
        }
        buffer.push(1, make_recurrent_transition(1, 0, false, true));

        assert_eq!(buffer.n_sequences(), 1);
        assert_eq!(buffer.n_in_progress(), 1);

        buffer.clear();

        assert_eq!(buffer.n_sequences(), 0);
        assert_eq!(buffer.n_in_progress(), 0);
    }

    #[test]
    fn should_handle_truncated_episodes() {
        let config = SequenceBufferConfig::new(1, 8);
        let mut buffer = SequenceBuffer::new(config);

        buffer.push(0, make_recurrent_transition(0, 0, false, true));
        buffer.push(0, make_truncated_recurrent_transition(0, 1));

        // Truncation should also trigger sequence creation
        assert_eq!(buffer.n_sequences(), 1);

        let seq = buffer.pop().unwrap();
        assert_eq!(seq.n_valid(), 2);
    }

    #[test]
    fn should_count_n_completed_correctly() {
        let config = SequenceBufferConfig::new(1, 4);
        let mut buffer = SequenceBuffer::new(config);

        // Create 2 full sequences (4 transitions each)
        for i in 0..8 {
            buffer.push(0, make_recurrent_transition(0, i, false, i % 4 == 0));
        }

        assert_eq!(buffer.n_sequences(), 2);
        assert_eq!(buffer.n_completed(), 8, "2 sequences * 4 transitions each");
    }

    #[test]
    fn should_ignore_invalid_env_id() {
        let config = SequenceBufferConfig::new(2, 4); // Only 2 envs: 0 and 1
        let mut buffer = SequenceBuffer::new(config);

        // Push to invalid env_id 5
        buffer.push(5, make_recurrent_transition(5, 0, false, true));

        assert_eq!(buffer.n_in_progress(), 0, "Should ignore invalid env_id");
    }

    #[test]
    fn should_iterate_over_sequences() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        for chunk in 0..3 {
            buffer.push(0, make_recurrent_transition(0, chunk * 2, false, true));
            buffer.push(0, make_recurrent_transition(0, chunk * 2 + 1, false, false));
        }

        let count = buffer.iter().count();
        assert_eq!(count, 3);

        let ids: Vec<_> = buffer.iter().map(|s| s.sequence_id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn should_handle_empty_episode_gracefully() {
        let config = SequenceBufferConfig::new(1, 4);
        let mut buffer = SequenceBuffer::new(config);

        // Don't push anything, just finalize
        buffer.finalize_all();

        assert_eq!(buffer.n_sequences(), 0);
    }

    #[test]
    fn should_track_env_id_in_sequences() {
        let config = SequenceBufferConfig::new(3, 2);
        let mut buffer = SequenceBuffer::new(config);

        // Create sequences from different envs
        for env_id in 0..3 {
            buffer.push(env_id, make_recurrent_transition(env_id, 0, false, true));
            buffer.push(env_id, make_recurrent_transition(env_id, 1, false, false));
        }

        let seq0 = buffer.pop().unwrap();
        let seq1 = buffer.pop().unwrap();
        let seq2 = buffer.pop().unwrap();

        assert_eq!(seq0.env_id, 0);
        assert_eq!(seq1.env_id, 1);
        assert_eq!(seq2.env_id, 2);
    }

    #[test]
    fn should_return_config_through_accessor() {
        let config = SequenceBufferConfig::new(4, 8).with_burn_in(2);
        let buffer = SequenceBuffer::new(config);

        assert_eq!(buffer.config().n_envs, 4);
        assert_eq!(buffer.config().sequence_length, 8);
        assert_eq!(buffer.config().burn_in_length, 2);
    }
}

// =============================================================================
// SEQUENCE TESTS (~8 tests)
// =============================================================================

mod sequence_tests {
    use super::*;

    #[test]
    fn should_create_full_sequence() {
        let transitions = vec![
            make_recurrent_transition(0, 0, false, true),
            make_recurrent_transition(0, 1, false, false),
            make_recurrent_transition(0, 2, false, false),
        ];
        let seq = Sequence::new(transitions, vec![0.0, 0.0], 0, 0);

        assert_eq!(seq.len(), 3);
        assert_eq!(seq.n_valid(), 3);
        assert!(seq.valid_mask.iter().all(|&v| v));
    }

    #[test]
    fn should_create_padded_sequence() {
        let transitions = vec![
            make_recurrent_transition(0, 0, false, true),
            make_recurrent_transition(0, 1, true, false),
        ];
        let seq = Sequence::new_padded(transitions, vec![0.0, 0.0], 0, 0, 5);

        assert_eq!(seq.len(), 5, "Should be padded to target length");
        assert_eq!(seq.n_valid(), 2, "Only 2 valid transitions");
    }

    #[test]
    fn should_have_correct_valid_mask_for_padded() {
        let transitions = vec![
            make_recurrent_transition(0, 0, false, true),
        ];
        let seq = Sequence::new_padded(transitions, vec![0.0], 0, 0, 4);

        assert_eq!(seq.valid_mask, vec![true, false, false, false]);
    }

    #[test]
    fn should_pad_with_clones_of_last_transition() {
        let transitions = vec![
            make_recurrent_transition(0, 0, false, true),
            make_recurrent_transition(0, 1, true, false),
        ];
        let seq = Sequence::new_padded(transitions, vec![0.0], 0, 0, 4);

        // Padded transitions should be clones of the last one
        assert_eq!(seq.transitions[2].step_in_sequence, 1);
        assert_eq!(seq.transitions[3].step_in_sequence, 1);
    }

    #[test]
    fn should_report_empty_correctly() {
        let seq = Sequence::new(vec![], vec![], 0, 0);
        assert!(seq.is_empty());
    }

    #[test]
    fn should_store_env_id() {
        let seq = Sequence::new(
            vec![make_recurrent_transition(5, 0, false, true)],
            vec![],
            5, // env_id
            0,
        );
        assert_eq!(seq.env_id, 5);
    }

    #[test]
    fn should_store_sequence_id() {
        let seq = Sequence::new(
            vec![make_recurrent_transition(0, 0, false, true)],
            vec![],
            0,
            42, // sequence_id
        );
        assert_eq!(seq.sequence_id, 42);
    }

    #[test]
    fn should_handle_empty_padded_sequence() {
        // Edge case: empty transitions with target length
        let seq = Sequence::new_padded(vec![], vec![], 0, 0, 3);

        // When transitions is empty, no padding happens (no last element to clone)
        assert!(seq.is_empty());
        assert_eq!(seq.n_valid(), 0);
    }
}

// =============================================================================
// SEQUENCE BATCH TESTS (~10 tests)
// =============================================================================

mod sequence_batch_tests {
    use super::*;

    fn create_test_batch() -> SequenceBatch {
        let seq1 = Sequence::new(
            vec![
                make_recurrent_transition(0, 0, false, true),
                make_recurrent_transition(0, 1, false, false),
            ],
            vec![0.1, 0.2],
            0,
            0,
        );
        let seq2 = Sequence::new(
            vec![
                make_recurrent_transition(1, 0, false, true),
                make_recurrent_transition(1, 1, false, false),
            ],
            vec![0.3, 0.4],
            1,
            1,
        );
        SequenceBatch::new(vec![seq1, seq2])
    }

    #[test]
    fn should_report_correct_dims() {
        let batch = create_test_batch();
        let (batch_size, seq_len) = batch.dims();

        assert_eq!(batch_size, 2);
        assert_eq!(seq_len, 2);
    }

    #[test]
    fn should_report_correct_len() {
        let batch = create_test_batch();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn should_flatten_states_with_shape_info() {
        let batch = create_test_batch();
        let (batch_size, seq_len, obs_size, data) = batch.states_flat(2);

        assert_eq!(batch_size, 2);
        assert_eq!(seq_len, 2);
        assert_eq!(obs_size, 2);
        assert_eq!(data.len(), 8); // 2 * 2 * 2
    }

    #[test]
    fn should_extract_discrete_actions() {
        let batch = create_test_batch();
        let actions = batch.actions_discrete();

        assert_eq!(actions.len(), 4); // 2 sequences * 2 steps
    }

    #[test]
    fn should_extract_rewards() {
        let batch = create_test_batch();
        let rewards = batch.rewards();

        assert_eq!(rewards.len(), 4);
    }

    #[test]
    fn should_extract_log_probs() {
        let batch = create_test_batch();
        let log_probs = batch.log_probs();

        assert_eq!(log_probs.len(), 4);
    }

    #[test]
    fn should_extract_values() {
        let batch = create_test_batch();
        let values = batch.values();

        assert_eq!(values.len(), 4);
    }

    #[test]
    fn should_extract_valid_masks() {
        let batch = create_test_batch();
        let masks = batch.valid_masks();

        assert_eq!(masks.len(), 4);
        assert!(masks.iter().all(|&v| v), "All should be valid in full sequences");
    }

    #[test]
    fn should_extract_initial_hiddens() {
        let batch = create_test_batch();
        let hiddens = batch.initial_hiddens();

        assert_eq!(hiddens.len(), 2);
        assert_eq!(hiddens[0], &[0.1, 0.2]);
        assert_eq!(hiddens[1], &[0.3, 0.4]);
    }

    #[test]
    fn should_handle_empty_batch() {
        let batch = SequenceBatch::new(vec![]);

        assert!(batch.is_empty());
        assert_eq!(batch.dims(), (0, 0));
        assert!(batch.rewards().is_empty());
        assert!(batch.valid_masks().is_empty());
    }
}

// =============================================================================
// TRAJECTORY STORE CONFIG TESTS (~4 tests)
// =============================================================================

mod trajectory_store_config_tests {
    use super::*;

    #[test]
    fn should_have_correct_default_values() {
        let config = TrajectoryStoreConfig::default();

        assert_eq!(config.max_trajectories, 1000);
        assert_eq!(config.min_trajectory_len, 20);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn should_allow_custom_max_trajectories() {
        let config = TrajectoryStoreConfig {
            max_trajectories: 500,
            ..Default::default()
        };
        assert_eq!(config.max_trajectories, 500);
    }

    #[test]
    fn should_allow_custom_min_trajectory_len() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        assert_eq!(config.min_trajectory_len, 5);
    }

    #[test]
    fn should_be_cloneable() {
        let config1 = TrajectoryStoreConfig {
            max_trajectories: 100,
            min_trajectory_len: 10,
            batch_size: 16,
        };
        let config2 = config1.clone();

        assert_eq!(config1.max_trajectories, config2.max_trajectories);
        assert_eq!(config1.min_trajectory_len, config2.min_trajectory_len);
        assert_eq!(config1.batch_size, config2.batch_size);
    }
}

// =============================================================================
// TRAJECTORY STORE TESTS (~20 tests)
// =============================================================================

mod trajectory_store_tests {
    use super::*;

    #[test]
    fn should_start_empty() {
        let store = TrajectoryStore::new(TrajectoryStoreConfig::default());

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.pending_len(), 0);
        assert_eq!(store.total_len(), 0);
    }

    #[test]
    fn should_track_pending_before_consolidate() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));

        assert_eq!(store.pending_len(), 1);
        assert_eq!(store.len(), 0);
        assert_eq!(store.total_len(), 1);
    }

    #[test]
    fn should_move_to_storage_on_consolidate() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));
        store.consolidate();

        assert_eq!(store.pending_len(), 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn should_reject_short_trajectories() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 10,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        // Push trajectory shorter than min_trajectory_len
        store.push(make_impala_trajectory(5, 0));

        assert_eq!(store.pending_len(), 0, "Short trajectory should be rejected");
    }

    #[test]
    fn should_accept_exact_min_length() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 10,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));

        assert_eq!(store.pending_len(), 1, "Exact min length should be accepted");
    }

    #[test]
    fn should_sample_fifo_order() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 2,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0)); // First
        store.push(make_impala_trajectory(10, 1)); // Second
        store.push(make_impala_trajectory(10, 2)); // Third
        store.consolidate();

        let batch = store.sample(2).unwrap();

        assert_eq!(batch.trajectories[0].env_id, 0, "First pushed should be first sampled");
        assert_eq!(batch.trajectories[1].env_id, 1, "Second pushed should be second sampled");
        assert_eq!(store.len(), 1, "Third should remain");
    }

    #[test]
    fn should_return_none_when_not_enough_trajectories() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));
        store.consolidate();

        let result = store.sample(2);
        assert!(result.is_none(), "Should return None when requesting more than available");
    }

    #[test]
    fn should_respect_max_capacity() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            max_trajectories: 3,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        for i in 0..5 {
            store.push(make_impala_trajectory(10, i));
        }
        store.consolidate();

        assert_eq!(store.len(), 3, "Should not exceed max_trajectories");
    }

    #[test]
    fn should_evict_oldest_when_over_capacity() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            max_trajectories: 2,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0)); // Will be evicted
        store.push(make_impala_trajectory(10, 1));
        store.push(make_impala_trajectory(10, 2));
        store.consolidate();

        let batch = store.sample(2).unwrap();

        // Oldest (env_id 0) should have been evicted
        assert_eq!(batch.trajectories[0].env_id, 1);
        assert_eq!(batch.trajectories[1].env_id, 2);
    }

    #[test]
    fn should_report_is_ready_correctly() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 2,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        assert!(!store.is_ready());

        store.push(make_impala_trajectory(10, 0));
        assert!(!store.is_ready(), "1 trajectory, need 2");

        store.push(make_impala_trajectory(10, 1));
        assert!(store.is_ready(), "2 trajectories, batch_size is 2");
    }

    #[test]
    fn should_consolidate_on_sample() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 1,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));
        // Don't explicitly consolidate

        let batch = store.sample(1);
        assert!(batch.is_some(), "Sample should consolidate pending first");
    }

    #[test]
    fn should_push_batch_of_trajectories() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        let trajectories = vec![
            make_impala_trajectory(10, 0),
            make_impala_trajectory(10, 1),
            make_impala_trajectory(10, 2),
        ];
        store.push_batch(trajectories);

        assert_eq!(store.pending_len(), 3);
    }

    #[test]
    fn should_filter_short_in_batch_push() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 10,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        let trajectories = vec![
            make_impala_trajectory(5, 0),  // Too short
            make_impala_trajectory(15, 1), // OK
            make_impala_trajectory(8, 2),  // Too short
        ];
        store.push_batch(trajectories);

        assert_eq!(store.pending_len(), 1, "Only one trajectory should pass filter");
    }

    #[test]
    fn should_return_config_through_accessor() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 15,
            batch_size: 8,
            max_trajectories: 500,
        };
        let store = TrajectoryStore::new(config);

        assert_eq!(store.config().min_trajectory_len, 15);
        assert_eq!(store.config().batch_size, 8);
        assert_eq!(store.config().max_trajectories, 500);
    }

    #[test]
    fn should_handle_empty_sample() {
        let config = TrajectoryStoreConfig::default();
        let store = TrajectoryStore::new(config);

        let result = store.sample(1);
        assert!(result.is_none());
    }

    #[test]
    fn should_handle_repeated_consolidate() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));
        store.consolidate();
        store.consolidate(); // Should be idempotent
        store.consolidate();

        assert_eq!(store.len(), 1);
        assert_eq!(store.pending_len(), 0);
    }

    #[test]
    fn should_track_policy_version() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 1,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory_versioned(10, 0, 42));

        let batch = store.sample(1).unwrap();
        assert_eq!(batch.trajectories[0].transitions[0].policy_version, 42);
    }

    #[test]
    fn should_preserve_trajectory_env_id() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 1,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 42));

        let batch = store.sample(1).unwrap();
        assert_eq!(batch.trajectories[0].env_id, 42);
    }

    #[test]
    fn should_update_size_after_sample_removes() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 1,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(10, 0));
        store.push(make_impala_trajectory(10, 1));
        store.consolidate();

        assert_eq!(store.len(), 2);

        let _ = store.sample(1);
        assert_eq!(store.len(), 1, "Size should decrease after sample");
    }
}

// =============================================================================
// TRAJECTORY BATCH TESTS (~8 tests)
// =============================================================================

mod trajectory_batch_tests {
    use super::*;

    fn create_test_batch() -> TrajectoryBatch {
        TrajectoryBatch {
            trajectories: vec![
                make_impala_trajectory(3, 0),
                make_impala_trajectory(4, 1),
            ],
        }
    }

    #[test]
    fn should_report_correct_len() {
        let batch = create_test_batch();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn should_report_total_transitions() {
        let batch = create_test_batch();
        assert_eq!(batch.total_transitions(), 7); // 3 + 4
    }

    #[test]
    fn should_flatten_all_states() {
        let batch = create_test_batch();
        let states = batch.all_states();

        assert_eq!(states.len(), 7);
    }

    #[test]
    fn should_flatten_all_behavior_log_probs() {
        let batch = create_test_batch();
        let log_probs = batch.all_behavior_log_probs();

        assert_eq!(log_probs.len(), 7);
    }

    #[test]
    fn should_flatten_all_rewards() {
        let batch = create_test_batch();
        let rewards = batch.all_rewards();

        assert_eq!(rewards.len(), 7);
    }

    #[test]
    fn should_flatten_all_dones() {
        let batch = create_test_batch();
        let dones = batch.all_dones();

        assert_eq!(dones.len(), 7);

        // Last transition of each trajectory is terminal
        assert!(dones[2], "Last of trajectory 0");
        assert!(dones[6], "Last of trajectory 1");
    }

    #[test]
    fn should_report_empty_correctly() {
        let batch = TrajectoryBatch {
            trajectories: vec![],
        };

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.total_transitions(), 0);
    }

    #[test]
    fn should_handle_variable_length_trajectories() {
        let batch = TrajectoryBatch {
            trajectories: vec![
                make_impala_trajectory(2, 0),
                make_impala_trajectory(5, 1),
                make_impala_trajectory(1, 2),
            ],
        };

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.total_transitions(), 8); // 2 + 5 + 1
        assert_eq!(batch.all_rewards().len(), 8);
    }
}

// =============================================================================
// CONCURRENCY TESTS (~12 tests)
// =============================================================================

mod concurrency_tests {
    use super::*;

    #[test]
    fn rollout_buffer_should_be_send() {
        fn assert_send<T: Send>() {}
        assert_send::<RolloutBuffer>();
    }

    #[test]
    fn rollout_buffer_should_be_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<RolloutBuffer>();
    }

    #[test]
    fn trajectory_store_should_be_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TrajectoryStore>();
    }

    #[test]
    fn trajectory_store_should_be_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TrajectoryStore>();
    }

    #[test]
    fn rollout_buffer_should_handle_concurrent_push() {
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 100,
        };
        let buffer = Arc::new(RolloutBuffer::new(config));

        let mut handles = vec![];

        for _ in 0..10 {
            let buffer_clone = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    buffer_clone.push_step(make_ppo_transitions(4));
                    thread::yield_now();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 10 threads * 10 steps = 100 steps
        assert_eq!(buffer.step_count(), 100);
        assert!(buffer.is_ready());
    }

    #[test]
    fn trajectory_store_should_handle_concurrent_push() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            max_trajectories: 1000,
            ..Default::default()
        };
        let store = Arc::new(TrajectoryStore::new(config));

        let mut handles = vec![];

        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    store_clone.push(make_impala_trajectory(10, i * 100 + j));
                    thread::yield_now();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        store.consolidate();

        // 10 threads * 10 trajectories = 100
        assert_eq!(store.len(), 100);
    }

    #[test]
    fn rollout_buffer_should_handle_producer_consumer() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 10,
        };
        let buffer = Arc::new(RolloutBuffer::new(config));

        let producer_buffer = Arc::clone(&buffer);
        let producer = thread::spawn(move || {
            for _ in 0..10 {
                producer_buffer.push_step(make_ppo_transitions(2));
                thread::sleep(Duration::from_micros(100));
            }
        });

        let consumer_buffer = Arc::clone(&buffer);
        let consumer = thread::spawn(move || {
            let ready = consumer_buffer.wait_ready_timeout(Duration::from_secs(1));
            assert!(ready);
            let batch = consumer_buffer.consume();
            batch.len()
        });

        producer.join().unwrap();
        let batch_len = consumer.join().unwrap();

        assert_eq!(batch_len, 20); // 10 steps * 2 envs
    }

    #[test]
    fn trajectory_store_should_handle_concurrent_push_and_sample() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            batch_size: 5,
            max_trajectories: 1000,
        };
        let store = Arc::new(TrajectoryStore::new(config));

        // Producer thread
        let producer_store = Arc::clone(&store);
        let producer = thread::spawn(move || {
            for i in 0..50 {
                producer_store.push(make_impala_trajectory(10, i));
                if i % 10 == 0 {
                    thread::yield_now();
                }
            }
        });

        // Consumer thread
        let consumer_store = Arc::clone(&store);
        let consumer = thread::spawn(move || {
            let mut sampled = 0;
            for _ in 0..10 {
                thread::sleep(Duration::from_millis(5));
                if let Some(batch) = consumer_store.sample(5) {
                    sampled += batch.len();
                }
            }
            sampled
        });

        producer.join().unwrap();
        let sampled = consumer.join().unwrap();

        // Some trajectories should have been sampled
        assert!(sampled > 0);
    }

    #[test]
    fn rollout_buffer_should_handle_multiple_rollouts() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 5,
        };
        let buffer = Arc::new(RolloutBuffer::new(config));

        let mut total_batches = 0;

        for _ in 0..3 {
            // Push a complete rollout
            for _ in 0..5 {
                buffer.push_step(make_ppo_transitions(2));
            }

            assert!(buffer.is_ready());
            let batch = buffer.consume();
            assert_eq!(batch.len(), 10);
            total_batches += 1;
        }

        assert_eq!(total_batches, 3);
    }

    #[test]
    fn trajectory_store_lock_free_push_should_not_block() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 5,
            ..Default::default()
        };
        let store = Arc::new(TrajectoryStore::new(config));

        let start = std::time::Instant::now();

        // Push many trajectories quickly
        for i in 0..1000 {
            store.push(make_impala_trajectory(10, i));
        }

        let elapsed = start.elapsed();

        // Should complete in reasonable time (lock-free = fast)
        assert!(elapsed < Duration::from_secs(1), "Push should be fast (lock-free)");
        assert_eq!(store.pending_len(), 1000);
    }

    #[test]
    fn rollout_buffer_atomic_ready_flag_should_be_consistent() {
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 10,
        };
        let buffer = Arc::new(RolloutBuffer::new(config));

        let mut handles = vec![];

        // Multiple threads checking ready state
        for _ in 0..5 {
            let buffer_clone = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                let mut ready_count = 0;
                for _ in 0..100 {
                    if buffer_clone.is_ready() {
                        ready_count += 1;
                    }
                    thread::yield_now();
                }
                ready_count
            });
            handles.push(handle);
        }

        // Producer
        for _ in 0..10 {
            buffer.push_step(make_ppo_transitions(2));
            thread::sleep(Duration::from_micros(100));
        }

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        // Final state should be ready
        assert!(buffer.is_ready());
    }
}

// =============================================================================
// EDGE CASE TESTS (~10 tests)
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn rollout_buffer_single_env_should_work() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        for step in 0..3 {
            buffer.push_step(vec![make_ppo_transition_for_env(0, step)]);
        }

        assert!(buffer.is_ready());
        let batch = buffer.consume();

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.n_envs, 1);

        // env_transitions should return all for the single env
        let env0: Vec<_> = batch.env_transitions(0).collect();
        assert_eq!(env0.len(), 3);
    }

    #[test]
    fn sequence_buffer_single_env_should_work() {
        let config = SequenceBufferConfig::new(1, 4);
        let mut buffer = SequenceBuffer::new(config);

        for step in 0..4 {
            buffer.push(0, make_recurrent_transition(0, step, false, step == 0));
        }

        assert_eq!(buffer.n_sequences(), 1);
    }

    #[test]
    fn trajectory_store_single_trajectory_should_work() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 1,
            batch_size: 1,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(5, 0));

        let batch = store.sample(1).unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn rollout_buffer_very_large_n_envs_should_work() {
        let config = RolloutBufferConfig {
            n_envs: 256,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        buffer.push_step(make_ppo_transitions(256));

        assert!(buffer.is_ready());
        assert_eq!(buffer.len(), 256);
    }

    #[test]
    fn sequence_buffer_sequence_length_one_should_work() {
        let config = SequenceBufferConfig::new(2, 1);
        let mut buffer = SequenceBuffer::new(config);

        buffer.push(0, make_recurrent_transition(0, 0, false, true));

        assert_eq!(buffer.n_sequences(), 1);

        let seq = buffer.pop().unwrap();
        assert_eq!(seq.len(), 1);
    }

    #[test]
    fn trajectory_store_min_len_one_should_accept_all() {
        let config = TrajectoryStoreConfig {
            min_trajectory_len: 1,
            ..Default::default()
        };
        let store = TrajectoryStore::new(config);

        store.push(make_impala_trajectory(1, 0));

        assert_eq!(store.pending_len(), 1);
    }

    #[test]
    fn rollout_buffer_zero_reward_should_preserve() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        let transition = PPOTransition {
            base: Transition::new_discrete(vec![0.0], 0, 0.0, vec![1.0], false, false),
            log_prob: 0.0,
            value: 0.0,
            bootstrap_value: None,
        };
        buffer.push_step(vec![transition]);

        let batch = buffer.consume();

        assert!((batch.rewards()[0] - 0.0).abs() < 1e-10);
        assert!((batch.values()[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sequence_buffer_empty_hidden_state_should_work() {
        let config = SequenceBufferConfig::new(1, 2);
        let mut buffer = SequenceBuffer::new(config);

        let t = RecurrentPPOTransition::from_ppo(
            PPOTransition::new_discrete(vec![0.0], 0, 1.0, vec![1.0], false, false, -0.5, 1.0),
            vec![], // Empty hidden state
            0,
            0,
            true,
        );
        buffer.push(0, t);
        buffer.push(0, make_recurrent_transition(0, 1, false, false));

        let seq = buffer.pop().unwrap();
        assert!(seq.initial_hidden.is_empty());
    }

    #[test]
    fn trajectory_batch_empty_trajectory_in_batch() {
        // A trajectory with zero transitions
        let empty_traj: Trajectory<IMPALATransition> = Trajectory::new(0);
        let normal_traj = make_impala_trajectory(3, 1);

        let batch = TrajectoryBatch {
            trajectories: vec![empty_traj, normal_traj],
        };

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.total_transitions(), 3); // Only from the non-empty one
    }

    #[test]
    fn rollout_buffer_negative_rewards_should_preserve() {
        let config = RolloutBufferConfig {
            n_envs: 1,
            rollout_len: 1,
        };
        let buffer = RolloutBuffer::new(config);

        let transition = PPOTransition {
            base: Transition::new_discrete(vec![0.0], 0, -10.5, vec![1.0], false, false),
            log_prob: -2.3,
            value: -5.0,
            bootstrap_value: None,
        };
        buffer.push_step(vec![transition]);

        let batch = buffer.consume();

        assert!((batch.rewards()[0] - (-10.5)).abs() < 1e-6);
        assert!((batch.log_probs()[0] - (-2.3)).abs() < 1e-6);
        assert!((batch.values()[0] - (-5.0)).abs() < 1e-6);
    }
}

// =============================================================================
// BUFFER TRAITS TESTS
// =============================================================================

mod buffer_traits_tests {
    use super::*;

    #[test]
    fn experience_buffer_is_empty_default_implementation() {
        // Test that is_empty uses len() correctly
        let config = RolloutBufferConfig {
            n_envs: 2,
            rollout_len: 2,
        };
        let buffer = RolloutBuffer::new(config);

        assert!(buffer.is_empty());

        buffer.push_step(make_ppo_transitions(2));

        assert!(!buffer.is_empty());
    }
}

// =============================================================================
// ADDITIONAL INTERLEAVED LAYOUT VERIFICATION TESTS
// =============================================================================

mod interleaved_layout_tests {
    use super::*;

    #[test]
    fn should_verify_interleaved_access_formula() {
        // Formula: transitions[step * n_envs + env_id] is env_id at step
        let config = RolloutBufferConfig {
            n_envs: 4,
            rollout_len: 3,
        };
        let buffer = RolloutBuffer::new(config);

        for step in 0..3 {
            let transitions: Vec<_> = (0..4)
                .map(|env_id| make_ppo_transition_for_env(env_id, step))
                .collect();
            buffer.push_step(transitions);
        }

        let batch = buffer.consume();
        let n_envs = batch.n_envs;

        for step in 0..3 {
            for env_id in 0..4 {
                let idx = step * n_envs + env_id;
                let trans = &batch.transitions[idx];

                assert_eq!(
                    trans.base.state[0] as usize,
                    env_id,
                    "At index {}, expected env_id {}", idx, env_id
                );
                assert_eq!(
                    trans.base.state[1] as usize,
                    step,
                    "At index {}, expected step {}", idx, step
                );
            }
        }
    }

    #[test]
    fn should_access_environment_transitions_via_iterator() {
        let config = RolloutBufferConfig {
            n_envs: 3,
            rollout_len: 4,
        };
        let buffer = RolloutBuffer::new(config);

        for step in 0..4 {
            let transitions: Vec<_> = (0..3)
                .map(|env_id| make_ppo_transition_for_env(env_id, step))
                .collect();
            buffer.push_step(transitions);
        }

        let batch = buffer.consume();

        // Verify each environment's transitions are in temporal order
        for env_id in 0..3 {
            let env_trans: Vec<_> = batch.env_transitions(env_id).collect();

            for (expected_step, trans) in env_trans.iter().enumerate() {
                assert_eq!(trans.base.state[0] as usize, env_id);
                assert_eq!(trans.base.state[1] as usize, expected_step);
            }
        }
    }
}
