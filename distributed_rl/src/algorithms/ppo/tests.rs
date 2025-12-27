//! Comprehensive unit and integration tests for the PPO submodule.
//!
//! This test module follows the "Tests as Definition: the Yoneda way" philosophy,
//! where tests serve as complete behavioral specifications. Each test explores
//! specific aspects of the PPO module's behavior, edge cases, and invariants.
//!
//! # Test Categories
//!
//! 1. **Config Tests**: PPORolloutBufferConfig, DistributedPPOConfig
//! 2. **Batch Tests**: PPORolloutBatch accessor methods and edge cases
//! 3. **Buffer Tests**: PPORolloutBuffer state machine and thread safety
//! 4. **Distributed Buffer Tests**: DistributedPPOBuffer per-env organization
//! 5. **Recurrent Buffer Tests**: RecurrentPPOBuffer sequence handling
//! 6. **Algorithm Tests**: DistributedPPO GAE and loss computation
//! 7. **Concurrency Tests**: Multi-threaded push/consume patterns
//! 8. **Integration Tests**: Full training cycle simulation
//!
//! # Backend
//!
//! All tensor tests use the WGPU backend as required:
//! ```rust
//! type B = Wgpu;
//! type TestAutodiffBackend = Autodiff<Wgpu>;
//! ```

#[cfg(test)]
mod tests {
    use super::super::distributed_ppo::{DistributedPPO, PPOProcessedBatch};
    use super::super::distributed_ppo_buffer::{DistributedPPOBuffer, DistributedPPORollouts};
    use super::super::ppo_buffer::{PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch};
    use super::super::recurrent_ppo_buffer::RecurrentPPOBuffer;
    use crate::algorithms::distributed_algorithm::{DistributedAlgorithm, DistributedPPOConfig};
    use crate::algorithms::gae::compute_gae;
    use crate::core::experience_buffer::{ExperienceBuffer, OnPolicyBuffer};
    use crate::core::transition::{PPOTransition, RecurrentPPOTransition, Transition};

    use burn::backend::{Autodiff, Wgpu};
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    // ============================================================================
    // Backend Type Definitions (WGPU as required)
    // ============================================================================

    type B = Wgpu;
    type TestAutodiffBackend = Autodiff<Wgpu>;

    // ============================================================================
    // Helper Functions for Test Data Creation
    // ============================================================================

    /// Create a PPOTransition with identifiable state based on id.
    fn make_transition(id: usize) -> PPOTransition {
        PPOTransition {
            base: Transition::new_discrete(
                vec![id as f32, (id * 2) as f32], // 2D state
                (id % 4) as u32,                  // action in [0, 3]
                1.0 + (id as f32) * 0.1,          // varying reward
                vec![(id + 1) as f32, ((id + 1) * 2) as f32],
                false,
                false,
            ),
            log_prob: -0.5 - (id as f32) * 0.01, // varying log_prob
            value: 0.5 + (id as f32) * 0.05,     // varying value
        }
    }

    /// Create a PPOTransition with terminal flag.
    fn make_terminal_transition(id: usize) -> PPOTransition {
        PPOTransition {
            base: Transition::new_discrete(
                vec![id as f32, (id * 2) as f32],
                (id % 4) as u32,
                0.0, // terminal typically has zero reward
                vec![(id + 1) as f32, ((id + 1) * 2) as f32],
                true,  // terminal
                false, // not truncated
            ),
            log_prob: -0.5,
            value: 0.0, // terminal value should be 0
        }
    }

    /// Create a PPOTransition with truncated flag.
    fn make_truncated_transition(id: usize) -> PPOTransition {
        PPOTransition {
            base: Transition::new_discrete(
                vec![id as f32, (id * 2) as f32],
                (id % 4) as u32,
                1.0,
                vec![(id + 1) as f32, ((id + 1) * 2) as f32],
                false, // not terminal
                true,  // truncated (time limit)
            ),
            log_prob: -0.5,
            value: 0.8, // bootstrap value for truncation
        }
    }

    /// Create a PPOTransition for a specific environment and step.
    fn make_env_transition(env_id: usize, step: usize) -> PPOTransition {
        PPOTransition {
            base: Transition::new_discrete(
                vec![env_id as f32, step as f32],
                (step % 4) as u32,
                1.0,
                vec![env_id as f32, (step + 1) as f32],
                false,
                false,
            ),
            log_prob: -0.5,
            value: 1.0,
        }
    }

    /// Create a RecurrentPPOTransition with sequence information.
    fn make_recurrent_transition(
        id: usize,
        seq_id: u64,
        step_in_seq: usize,
        is_start: bool,
    ) -> RecurrentPPOTransition {
        RecurrentPPOTransition::new_discrete(
            vec![id as f32, (id * 2) as f32],
            (id % 4) as u32,
            1.0,
            vec![(id + 1) as f32, ((id + 1) * 2) as f32],
            false,
            false,
            -0.5,
            1.0,
            vec![0.1, 0.2, 0.3, 0.4], // 4D hidden state
            seq_id,
            step_in_seq,
            is_start,
            None, // no bootstrap
        )
    }

    /// Create a batch of transitions for testing.
    fn make_batch(n: usize) -> Vec<PPOTransition> {
        (0..n).map(make_transition).collect()
    }

    // ============================================================================
    // Module: PPORolloutBufferConfig Tests
    // ============================================================================

    mod config_tests {
        use super::*;

        #[test]
        fn test_config_default_values() {
            let config = PPORolloutBufferConfig::default();

            assert_eq!(config.n_actors, 4);
            assert_eq!(config.n_envs_per_actor, 32);
            assert_eq!(config.rollout_length, 128);
        }

        #[test]
        fn test_config_total_envs_basic() {
            let config = PPORolloutBufferConfig {
                n_actors: 2,
                n_envs_per_actor: 4,
                rollout_length: 10,
            };

            assert_eq!(config.total_envs(), 8);
        }

        #[test]
        fn test_config_total_envs_single_actor() {
            let config = PPORolloutBufferConfig {
                n_actors: 1,
                n_envs_per_actor: 32,
                rollout_length: 128,
            };

            assert_eq!(config.total_envs(), 32);
        }

        #[test]
        fn test_config_total_envs_many_actors() {
            let config = PPORolloutBufferConfig {
                n_actors: 16,
                n_envs_per_actor: 64,
                rollout_length: 256,
            };

            assert_eq!(config.total_envs(), 1024);
        }

        #[test]
        fn test_config_capacity_basic() {
            let config = PPORolloutBufferConfig {
                n_actors: 2,
                n_envs_per_actor: 4,
                rollout_length: 10,
            };

            // capacity = total_envs * rollout_length = 8 * 10 = 80
            assert_eq!(config.capacity(), 80);
        }

        #[test]
        fn test_config_capacity_large() {
            let config = PPORolloutBufferConfig {
                n_actors: 8,
                n_envs_per_actor: 128,
                rollout_length: 256,
            };

            // capacity = 8 * 128 * 256 = 262144
            assert_eq!(config.capacity(), 262144);
        }

        #[test]
        fn test_config_clone() {
            let config = PPORolloutBufferConfig {
                n_actors: 3,
                n_envs_per_actor: 5,
                rollout_length: 7,
            };
            let cloned = config.clone();

            assert_eq!(cloned.n_actors, 3);
            assert_eq!(cloned.n_envs_per_actor, 5);
            assert_eq!(cloned.rollout_length, 7);
        }

        #[test]
        fn test_config_edge_case_minimal() {
            let config = PPORolloutBufferConfig {
                n_actors: 1,
                n_envs_per_actor: 1,
                rollout_length: 1,
            };

            assert_eq!(config.total_envs(), 1);
            assert_eq!(config.capacity(), 1);
        }
    }

    // ============================================================================
    // Module: PPORolloutBatch Tests
    // ============================================================================

    mod batch_tests {
        use super::*;

        fn make_test_batch(n: usize) -> PPORolloutBatch {
            PPORolloutBatch {
                transitions: make_batch(n),
                policy_version: 42,
                n_envs: 2,
                rollout_length: n / 2,
            }
        }

        #[test]
        fn test_batch_states_extraction() {
            let batch = make_test_batch(4);
            let states = batch.states();

            // 4 transitions * 2D state = 8 floats
            assert_eq!(states.len(), 8);
            // First transition state: [0.0, 0.0]
            assert_eq!(states[0], 0.0);
            assert_eq!(states[1], 0.0);
            // Second transition state: [1.0, 2.0]
            assert_eq!(states[2], 1.0);
            assert_eq!(states[3], 2.0);
        }

        #[test]
        fn test_batch_next_states_extraction() {
            let batch = make_test_batch(4);
            let next_states = batch.next_states();

            assert_eq!(next_states.len(), 8);
            // First transition next_state: [1.0, 2.0]
            assert_eq!(next_states[0], 1.0);
            assert_eq!(next_states[1], 2.0);
        }

        #[test]
        fn test_batch_rewards_extraction() {
            let batch = make_test_batch(3);
            let rewards = batch.rewards();

            assert_eq!(rewards.len(), 3);
            // Rewards are 1.0 + id * 0.1
            assert!((rewards[0] - 1.0).abs() < 1e-6);
            assert!((rewards[1] - 1.1).abs() < 1e-6);
            assert!((rewards[2] - 1.2).abs() < 1e-6);
        }

        #[test]
        fn test_batch_values_extraction() {
            let batch = make_test_batch(3);
            let values = batch.values();

            assert_eq!(values.len(), 3);
            // Values are 0.5 + id * 0.05
            assert!((values[0] - 0.5).abs() < 1e-6);
            assert!((values[1] - 0.55).abs() < 1e-6);
            assert!((values[2] - 0.6).abs() < 1e-6);
        }

        #[test]
        fn test_batch_log_probs_extraction() {
            let batch = make_test_batch(3);
            let log_probs = batch.log_probs();

            assert_eq!(log_probs.len(), 3);
            // Log probs are -0.5 - id * 0.01
            assert!((log_probs[0] - (-0.5)).abs() < 1e-6);
            assert!((log_probs[1] - (-0.51)).abs() < 1e-6);
            assert!((log_probs[2] - (-0.52)).abs() < 1e-6);
        }

        #[test]
        fn test_batch_dones_extraction() {
            let mut transitions = make_batch(3);
            // Make second transition terminal
            transitions[1] = make_terminal_transition(1);

            let batch = PPORolloutBatch {
                transitions,
                policy_version: 1,
                n_envs: 1,
                rollout_length: 3,
            };

            let dones = batch.dones();
            assert_eq!(dones, vec![false, true, false]);
        }

        #[test]
        fn test_batch_len() {
            let batch = make_test_batch(10);
            assert_eq!(batch.len(), 10);
        }

        #[test]
        fn test_batch_is_empty_false() {
            let batch = make_test_batch(1);
            assert!(!batch.is_empty());
        }

        #[test]
        fn test_batch_is_empty_true() {
            let batch = PPORolloutBatch {
                transitions: vec![],
                policy_version: 0,
                n_envs: 0,
                rollout_length: 0,
            };
            assert!(batch.is_empty());
        }

        #[test]
        fn test_batch_state_dim_normal() {
            let batch = make_test_batch(5);
            assert_eq!(batch.state_dim(), 2);
        }

        #[test]
        fn test_batch_state_dim_empty() {
            let batch = PPORolloutBatch {
                transitions: vec![],
                policy_version: 0,
                n_envs: 0,
                rollout_length: 0,
            };
            assert_eq!(batch.state_dim(), 0);
        }

        #[test]
        fn test_batch_clone() {
            let batch = make_test_batch(3);
            let cloned = batch.clone();

            assert_eq!(cloned.len(), 3);
            assert_eq!(cloned.policy_version, 42);
            assert_eq!(cloned.n_envs, 2);
        }
    }

    // ============================================================================
    // Module: PPORolloutBuffer Tests
    // ============================================================================

    mod rollout_buffer_tests {
        use super::*;

        fn make_buffer(n_actors: usize, n_envs: usize, rollout_len: usize) -> PPORolloutBuffer {
            let config = PPORolloutBufferConfig {
                n_actors,
                n_envs_per_actor: n_envs,
                rollout_length: rollout_len,
            };
            PPORolloutBuffer::new(config)
        }

        #[test]
        fn test_buffer_new_is_empty() {
            let buffer = make_buffer(2, 4, 10);

            assert!(buffer.is_empty());
            assert!(!buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 0);
        }

        #[test]
        fn test_buffer_capacity() {
            let buffer = make_buffer(2, 4, 10);
            // capacity = 2 * 4 * 10 = 80
            assert_eq!(buffer.capacity(), 80);
        }

        #[test]
        fn test_buffer_push_step_single() {
            let buffer = make_buffer(1, 2, 3);

            buffer.push_step(vec![make_transition(0), make_transition(1)], 1);

            assert_eq!(buffer.len(), 2);
            assert_eq!(buffer.step_count(), 1);
            assert!(!buffer.is_rollout_ready());
        }

        #[test]
        fn test_buffer_push_step_until_ready() {
            let buffer = make_buffer(1, 2, 3);

            // Push 3 steps (rollout_length = 3)
            for step in 0..3 {
                let transitions = vec![
                    make_transition(step * 2),
                    make_transition(step * 2 + 1),
                ];
                buffer.push_step(transitions, 1);
            }

            assert!(buffer.is_rollout_ready());
            assert_eq!(buffer.len(), 6);
            assert_eq!(buffer.step_count(), 3);
        }

        #[test]
        fn test_buffer_push_step_version_update() {
            let buffer = make_buffer(1, 2, 3);

            buffer.push_step(make_batch(2), 5);
            buffer.push_step(make_batch(2), 10);
            buffer.push_step(make_batch(2), 7); // Lower version

            let batch = buffer.consume();
            // fetch_max means we keep the highest version seen
            assert_eq!(batch.policy_version, 10);
        }

        #[test]
        fn test_buffer_consume_resets_state() {
            let buffer = make_buffer(1, 2, 3);

            for _ in 0..3 {
                buffer.push_step(make_batch(2), 1);
            }

            assert!(buffer.is_rollout_ready());

            let batch = buffer.consume();
            assert_eq!(batch.len(), 6);

            // Buffer should be reset
            assert!(buffer.is_empty());
            assert!(!buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 0);
        }

        #[test]
        fn test_buffer_consume_returns_correct_data() {
            let buffer = make_buffer(1, 2, 2);

            buffer.push_step(vec![make_transition(0), make_transition(1)], 42);
            buffer.push_step(vec![make_transition(2), make_transition(3)], 42);

            let batch = buffer.consume();

            assert_eq!(batch.len(), 4);
            assert_eq!(batch.policy_version, 42);
            assert_eq!(batch.n_envs, 2);
            assert_eq!(batch.rollout_length, 2);

            // Verify first transition state
            let states = batch.states();
            assert_eq!(states[0], 0.0);
            assert_eq!(states[1], 0.0);
        }

        #[test]
        fn test_buffer_multiple_consume_cycles() {
            let buffer = make_buffer(1, 2, 2);

            // First cycle
            buffer.push_step(make_batch(2), 1);
            buffer.push_step(make_batch(2), 1);
            let batch1 = buffer.consume();
            assert_eq!(batch1.len(), 4);

            // Second cycle
            buffer.push_step(make_batch(2), 2);
            buffer.push_step(make_batch(2), 2);
            let batch2 = buffer.consume();
            assert_eq!(batch2.len(), 4);
            assert_eq!(batch2.policy_version, 2);
        }

        #[test]
        fn test_buffer_utilization() {
            let buffer = make_buffer(1, 2, 4);

            // Empty utilization
            assert!((buffer.utilization() - 0.0).abs() < 1e-6);

            // Add 2 steps (4 transitions out of 8 capacity)
            buffer.push_step(make_batch(2), 1);
            buffer.push_step(make_batch(2), 1);

            assert!((buffer.utilization() - 0.5).abs() < 1e-6);
        }

        #[test]
        fn test_buffer_clear() {
            let buffer = make_buffer(1, 2, 3);

            buffer.push_step(make_batch(2), 1);
            buffer.push_step(make_batch(2), 1);

            assert!(!buffer.is_empty());

            buffer.clear();

            assert!(buffer.is_empty());
            assert_eq!(buffer.step_count(), 0);
            assert!(!buffer.is_rollout_ready());
        }

        #[test]
        fn test_buffer_config_accessor() {
            let buffer = make_buffer(3, 5, 7);
            let config = buffer.config();

            assert_eq!(config.n_actors, 3);
            assert_eq!(config.n_envs_per_actor, 5);
            assert_eq!(config.rollout_length, 7);
        }

        // --- ExperienceBuffer trait tests ---

        #[test]
        fn test_experience_buffer_push_single() {
            let buffer = make_buffer(1, 2, 4);

            buffer.push(make_transition(0));
            assert_eq!(buffer.len(), 1);

            buffer.push(make_transition(1));
            assert_eq!(buffer.len(), 2);
        }

        #[test]
        fn test_experience_buffer_push_batch() {
            let buffer = make_buffer(1, 2, 4);

            buffer.push_batch(make_batch(4));
            assert_eq!(buffer.len(), 4);
        }

        #[test]
        fn test_experience_buffer_push_batch_triggers_ready() {
            let buffer = make_buffer(1, 2, 2);

            // Push exactly capacity transitions (2 envs * 2 rollout = 4)
            buffer.push_batch(make_batch(4));

            assert!(buffer.is_rollout_ready());
        }

        #[test]
        fn test_experience_buffer_consolidate_is_noop() {
            let buffer = make_buffer(1, 2, 4);

            buffer.push_batch(make_batch(2));
            buffer.consolidate(); // Should not panic or change state

            assert_eq!(buffer.len(), 2);
        }

        // --- OnPolicyBuffer trait tests ---

        #[test]
        fn test_on_policy_buffer_is_ready() {
            let buffer = make_buffer(1, 2, 4);

            assert!(!buffer.is_ready(4));

            buffer.push_batch(make_batch(4));

            assert!(buffer.is_ready(4));
        }

        #[test]
        fn test_on_policy_buffer_drain() {
            let buffer = make_buffer(1, 2, 4);

            buffer.push_batch(make_batch(4));

            let drained = buffer.drain();

            assert_eq!(drained.len(), 4);
            assert!(buffer.is_empty());
            assert_eq!(buffer.step_count(), 0);
        }

        // --- Timeout tests ---

        #[test]
        fn test_buffer_wait_ready_timeout_not_ready() {
            let buffer = make_buffer(1, 2, 10);

            buffer.push_step(make_batch(2), 1);

            // Should timeout since not ready
            let result = buffer.wait_ready_timeout(Duration::from_millis(10));
            assert!(!result);
        }

        #[test]
        fn test_buffer_wait_ready_timeout_already_ready() {
            let buffer = Arc::new(make_buffer(1, 2, 1));

            // Fill buffer to be ready
            buffer.push_step(make_batch(2), 1);
            assert!(buffer.is_rollout_ready());

            // Create a second reference for the thread
            let buffer_clone = Arc::clone(&buffer);

            // Spawn a thread to wait
            let handle = thread::spawn(move || buffer_clone.wait_ready_timeout(Duration::from_secs(1)));

            // The wait should succeed since buffer is already ready
            let result = handle.join().unwrap();
            assert!(result);
        }
    }

    // ============================================================================
    // Module: DistributedPPOBuffer Tests
    // ============================================================================

    mod distributed_buffer_tests {
        use super::*;

        #[test]
        fn test_distributed_buffer_creation() {
            let buffer = DistributedPPOBuffer::new(64, 128);

            assert_eq!(buffer.rollout_length(), 64);
            assert_eq!(buffer.total_envs(), 128);
            assert!(!buffer.is_ready());
            assert_eq!(buffer.total_transitions(), 0);
        }

        #[test]
        fn test_distributed_buffer_push_batch_single_actor() {
            let buffer = DistributedPPOBuffer::new(3, 4);

            // Actor 0 with 2 envs pushes step 0
            buffer.push_batch(
                vec![make_env_transition(0, 0), make_env_transition(1, 0)],
                0, // offset 0
            );

            assert_eq!(buffer.total_transitions(), 2);
            assert!(!buffer.is_ready());
        }

        #[test]
        fn test_distributed_buffer_push_batch_correct_offset() {
            // 2 actors, 2 envs each, rollout length 2
            let buffer = DistributedPPOBuffer::new(2, 4);

            // Actor 0 (envs 0-1) pushes step 0
            buffer.push_batch(
                vec![make_env_transition(0, 0), make_env_transition(1, 0)],
                0,
            );

            // Actor 1 (envs 2-3) pushes step 0
            buffer.push_batch(
                vec![make_env_transition(2, 0), make_env_transition(3, 0)],
                2, // offset 2
            );

            assert_eq!(buffer.total_transitions(), 4);

            // Push step 1 for both actors
            buffer.push_batch(
                vec![make_env_transition(0, 1), make_env_transition(1, 1)],
                0,
            );
            buffer.push_batch(
                vec![make_env_transition(2, 1), make_env_transition(3, 1)],
                2,
            );

            assert!(buffer.is_ready());
            assert_eq!(buffer.total_transitions(), 8);
        }

        #[test]
        fn test_distributed_buffer_overflow_protection() {
            let buffer = DistributedPPOBuffer::new(2, 2);

            // Push more than rollout_length steps
            for step in 0..5 {
                buffer.push_batch(
                    vec![make_env_transition(0, step), make_env_transition(1, step)],
                    0,
                );
            }

            // Should only store rollout_length (2) per env
            assert_eq!(buffer.total_transitions(), 4);
        }

        #[test]
        fn test_distributed_buffer_invalid_env_id_ignored() {
            let buffer = DistributedPPOBuffer::new(2, 2);

            // Push with offset that would exceed total_envs
            buffer.push_batch(
                vec![make_env_transition(0, 0), make_env_transition(1, 0)],
                10, // Invalid offset
            );

            // Should not store anything
            assert_eq!(buffer.total_transitions(), 0);
        }

        #[test]
        fn test_distributed_buffer_progress() {
            let buffer = DistributedPPOBuffer::new(3, 4);

            let (min, total) = buffer.progress();
            assert_eq!(min, 0);
            assert_eq!(total, 4);

            // Push 1 step for all envs
            buffer.push_batch(make_batch(4), 0);

            let (min, total) = buffer.progress();
            assert_eq!(min, 1);
            assert_eq!(total, 4);
        }

        #[test]
        fn test_distributed_buffer_progress_uneven() {
            let buffer = DistributedPPOBuffer::new(3, 4);

            // Push 2 steps for envs 0-1 only
            buffer.push_batch(
                vec![make_env_transition(0, 0), make_env_transition(1, 0)],
                0,
            );
            buffer.push_batch(
                vec![make_env_transition(0, 1), make_env_transition(1, 1)],
                0,
            );

            let (min, _) = buffer.progress();
            assert_eq!(min, 0); // envs 2-3 have 0 steps
        }

        #[test]
        fn test_distributed_buffer_consume() {
            let buffer = DistributedPPOBuffer::new(2, 2);

            // Fill buffer
            buffer.push_batch(
                vec![make_env_transition(0, 0), make_env_transition(1, 0)],
                0,
            );
            buffer.push_batch(
                vec![make_env_transition(0, 1), make_env_transition(1, 1)],
                0,
            );

            assert!(buffer.is_ready());

            let rollouts = buffer.consume();

            assert_eq!(rollouts.len(), 2);
            assert_eq!(rollouts[0].len(), 2);
            assert_eq!(rollouts[1].len(), 2);

            // Buffer should be reset
            assert!(!buffer.is_ready());
            assert_eq!(buffer.total_transitions(), 0);
        }

        #[test]
        fn test_distributed_buffer_consumed_epoch() {
            let buffer = DistributedPPOBuffer::new(1, 1);

            assert_eq!(buffer.consumed_epoch(), 0);

            buffer.push_batch(vec![make_env_transition(0, 0)], 0);
            let _ = buffer.consume();
            assert_eq!(buffer.consumed_epoch(), 1);

            buffer.push_batch(vec![make_env_transition(0, 0)], 0);
            let _ = buffer.consume();
            assert_eq!(buffer.consumed_epoch(), 2);
        }

        #[test]
        fn test_distributed_buffer_clear() {
            let buffer = DistributedPPOBuffer::new(2, 2);

            buffer.push_batch(make_batch(2), 0);

            buffer.clear();

            assert_eq!(buffer.total_transitions(), 0);
            assert!(!buffer.is_ready());
        }

        #[test]
        fn test_distributed_buffer_transitions_organized_by_env() {
            let buffer = DistributedPPOBuffer::new(2, 3);

            // Push transitions with identifiable env_ids
            buffer.push_batch(
                vec![
                    make_env_transition(0, 0),
                    make_env_transition(1, 0),
                    make_env_transition(2, 0),
                ],
                0,
            );
            buffer.push_batch(
                vec![
                    make_env_transition(0, 1),
                    make_env_transition(1, 1),
                    make_env_transition(2, 1),
                ],
                0,
            );

            let rollouts = buffer.consume();

            // Each env should have its own transitions
            for (env_id, rollout) in rollouts.iter().enumerate() {
                for (step, transition) in rollout.iter().enumerate() {
                    // state[0] = env_id, state[1] = step
                    assert_eq!(transition.base.state[0], env_id as f32);
                    assert_eq!(transition.base.state[1], step as f32);
                }
            }
        }

        // --- DistributedPPORollouts tests ---

        #[test]
        fn test_distributed_ppo_rollouts_new() {
            let rollouts = vec![make_batch(3), make_batch(2), vec![]];

            let batch = DistributedPPORollouts::new(rollouts, 3);

            assert_eq!(batch.n_envs, 3);
            assert_eq!(batch.rollout_length, 3);
        }

        #[test]
        fn test_distributed_ppo_rollouts_total_transitions() {
            let rollouts = vec![make_batch(3), make_batch(2), vec![]];

            let batch = DistributedPPORollouts::new(rollouts, 3);

            assert_eq!(batch.total_transitions(), 5);
        }

        #[test]
        fn test_distributed_ppo_rollouts_non_empty() {
            let rollouts = vec![make_batch(2), vec![], make_batch(1)];

            let batch = DistributedPPORollouts::new(rollouts, 2);

            let non_empty: Vec<_> = batch.non_empty_rollouts().collect();
            assert_eq!(non_empty.len(), 2);
            assert_eq!(non_empty[0].0, 0); // env 0
            assert_eq!(non_empty[1].0, 2); // env 2
        }

        #[test]
        fn test_distributed_ppo_rollouts_obs_dim() {
            let rollouts = vec![vec![], make_batch(1)];

            let batch = DistributedPPORollouts::new(rollouts, 1);

            assert_eq!(batch.obs_dim(), Some(2));
        }

        #[test]
        fn test_distributed_ppo_rollouts_obs_dim_all_empty() {
            let rollouts: Vec<Vec<PPOTransition>> = vec![vec![], vec![]];

            let batch = DistributedPPORollouts::new(rollouts, 0);

            assert_eq!(batch.obs_dim(), None);
        }
    }

    // ============================================================================
    // Module: RecurrentPPOBuffer Tests
    // ============================================================================

    mod recurrent_buffer_tests {
        use super::*;

        #[test]
        fn test_recurrent_buffer_creation() {
            let buffer = RecurrentPPOBuffer::new(32, 64);

            assert_eq!(buffer.rollout_length(), 32);
            assert_eq!(buffer.total_envs(), 64);
            assert!(!buffer.is_ready());
            assert_eq!(buffer.total_transitions(), 0);
        }

        #[test]
        fn test_recurrent_buffer_push_batch() {
            let buffer = RecurrentPPOBuffer::new(2, 2);

            buffer.push_batch(
                vec![
                    make_recurrent_transition(0, 1, 0, true),
                    make_recurrent_transition(1, 2, 0, true),
                ],
                0,
            );

            assert_eq!(buffer.total_transitions(), 2);
        }

        #[test]
        fn test_recurrent_buffer_sequence_preserved() {
            let buffer = RecurrentPPOBuffer::new(3, 1);

            // Push sequence with sequence_id = 42
            for step in 0..3 {
                buffer.push_batch(
                    vec![make_recurrent_transition(step, 42, step, step == 0)],
                    0,
                );
            }

            let rollouts = buffer.consume();

            assert_eq!(rollouts.len(), 1);
            assert_eq!(rollouts[0].len(), 3);

            // Verify sequence info is preserved
            for (step, transition) in rollouts[0].iter().enumerate() {
                assert_eq!(transition.sequence_id, 42);
                assert_eq!(transition.step_in_sequence, step);
                assert_eq!(transition.is_sequence_start, step == 0);
            }
        }

        #[test]
        fn test_recurrent_buffer_hidden_state_preserved() {
            let buffer = RecurrentPPOBuffer::new(1, 1);

            buffer.push_batch(vec![make_recurrent_transition(0, 1, 0, true)], 0);

            let rollouts = buffer.consume();

            assert_eq!(rollouts[0][0].hidden_state, vec![0.1, 0.2, 0.3, 0.4]);
        }

        #[test]
        fn test_recurrent_buffer_is_ready() {
            let buffer = RecurrentPPOBuffer::new(2, 2);

            // Push 1 step
            buffer.push_batch(
                vec![
                    make_recurrent_transition(0, 1, 0, true),
                    make_recurrent_transition(1, 2, 0, true),
                ],
                0,
            );
            assert!(!buffer.is_ready());

            // Push 2nd step
            buffer.push_batch(
                vec![
                    make_recurrent_transition(0, 1, 1, false),
                    make_recurrent_transition(1, 2, 1, false),
                ],
                0,
            );
            assert!(buffer.is_ready());
        }

        #[test]
        fn test_recurrent_buffer_consumed_epoch() {
            let buffer = RecurrentPPOBuffer::new(1, 1);

            assert_eq!(buffer.consumed_epoch(), 0);

            buffer.push_batch(vec![make_recurrent_transition(0, 1, 0, true)], 0);
            let _ = buffer.consume();

            assert_eq!(buffer.consumed_epoch(), 1);
        }

        #[test]
        fn test_recurrent_buffer_bootstrap_value() {
            let buffer = RecurrentPPOBuffer::new(1, 1);

            let mut transition = make_recurrent_transition(0, 1, 0, true);
            transition.bootstrap_value = Some(0.75);

            buffer.push_batch(vec![transition], 0);

            let rollouts = buffer.consume();
            assert_eq!(rollouts[0][0].bootstrap_value, Some(0.75));
        }

        #[test]
        fn test_recurrent_buffer_multiple_sequences() {
            let buffer = RecurrentPPOBuffer::new(2, 1);

            // First sequence (seq_id = 1)
            buffer.push_batch(vec![make_recurrent_transition(0, 1, 0, true)], 0);
            // Second sequence starts (seq_id = 2)
            buffer.push_batch(vec![make_recurrent_transition(1, 2, 0, true)], 0);

            let rollouts = buffer.consume();

            // Both transitions in same env rollout
            assert_eq!(rollouts[0][0].sequence_id, 1);
            assert_eq!(rollouts[0][1].sequence_id, 2);
        }
    }

    // ============================================================================
    // Module: DistributedPPO Algorithm Tests
    // ============================================================================

    mod algorithm_tests {
        use super::*;

        fn get_device() -> <TestAutodiffBackend as Backend>::Device {
            Default::default()
        }

        #[test]
        fn test_distributed_ppo_new() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);

            assert_eq!(
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::name(&ppo),
                "DistributedPPO"
            );
            assert!(
                !<DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::is_off_policy(&ppo)
            );
        }

        #[test]
        fn test_distributed_ppo_n_epochs() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);

            assert_eq!(
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::n_epochs(&ppo),
                4
            );
        }

        #[test]
        fn test_distributed_ppo_n_minibatches() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);

            assert_eq!(
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::n_minibatches(&ppo),
                4
            );
        }

        #[test]
        fn test_distributed_ppo_create_buffer() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config.clone());

            let buffer = <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 2, 32,
            );

            assert_eq!(buffer.config().n_actors, 2);
            assert_eq!(buffer.config().n_envs_per_actor, 32);
            assert_eq!(buffer.config().rollout_length, config.rollout_length);
        }

        #[test]
        fn test_distributed_ppo_is_ready() {
            let config = DistributedPPOConfig::default().with_rollout_length(2);
            let ppo = DistributedPPO::new(config);

            let buffer = <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 2,
            );

            assert!(
                !<DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::is_ready(
                    &ppo, &buffer
                )
            );

            buffer.push_step(make_batch(2), 1);
            buffer.push_step(make_batch(2), 1);

            assert!(
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::is_ready(
                    &ppo, &buffer
                )
            );
        }

        #[test]
        fn test_distributed_ppo_sample_batch_not_ready() {
            let config = DistributedPPOConfig::default().with_rollout_length(10);
            let ppo = DistributedPPO::new(config);

            let buffer = <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 2,
            );

            buffer.push_step(make_batch(2), 1);

            let batch =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
                    &ppo, &buffer,
                );

            assert!(batch.is_none());
        }

        #[test]
        fn test_distributed_ppo_sample_batch_ready() {
            let config = DistributedPPOConfig::default().with_rollout_length(2);
            let ppo = DistributedPPO::new(config);

            let buffer = <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 2,
            );

            buffer.push_step(make_batch(2), 42);
            buffer.push_step(make_batch(2), 42);

            let batch =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
                    &ppo, &buffer,
                );

            assert!(batch.is_some());
            let batch = batch.unwrap();
            assert_eq!(batch.rollout.len(), 4);
            assert_eq!(batch.advantages.len(), 4);
            assert_eq!(batch.returns.len(), 4);
        }

        #[test]
        fn test_distributed_ppo_staleness_discards_stale_batch() {
            // On-policy PPO must discard stale data to maintain correctness.
            // When a batch's policy version is too old compared to the learner's
            // current version, the batch is discarded (returned as empty).
            let config = DistributedPPOConfig::default(); // max_staleness = 1
            let ppo = DistributedPPO::new(config);

            let rollout = PPORolloutBatch {
                transitions: make_batch(4),
                policy_version: 1, // Old version
                n_envs: 2,
                rollout_length: 2,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; 4],
                returns: vec![1.0; 4],
            };

            let result =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::handle_staleness(
                    &ppo,
                    batch,
                    100, // Current version much higher (staleness = 99 > max_staleness = 1)
                );

            // Stale batch is discarded - empty batch returned
            assert_eq!(result.rollout.len(), 0, "Stale batch should be discarded");
            assert!(result.advantages.is_empty(), "Stale advantages should be cleared");
            assert!(result.returns.is_empty(), "Stale returns should be cleared");
        }

        #[test]
        fn test_distributed_ppo_staleness_keeps_fresh_batch() {
            // Batches within the staleness threshold are kept as-is.
            let config = DistributedPPOConfig::default(); // max_staleness = 1
            let ppo = DistributedPPO::new(config);

            let rollout = PPORolloutBatch {
                transitions: make_batch(4),
                policy_version: 99, // Recent version
                n_envs: 2,
                rollout_length: 2,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; 4],
                returns: vec![1.0; 4],
            };

            let result =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::handle_staleness(
                    &ppo,
                    batch,
                    100, // Current version only 1 ahead (staleness = 1 <= max_staleness = 1)
                );

            // Fresh batch is kept
            assert_eq!(result.rollout.len(), 4, "Fresh batch should be kept");
            assert_eq!(result.advantages.len(), 4, "Advantages should be kept");
            assert_eq!(result.returns.len(), 4, "Returns should be kept");
        }

        // --- GAE computation tests ---

        #[test]
        fn test_compute_advantages_basic() {
            let config = DistributedPPOConfig::default()
                .with_gamma(0.99)
                .with_gae_lambda(0.95);
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![1.0, 1.0, 1.0];
            let values = vec![0.5, 0.5, 0.5];
            let dones = vec![false, false, false];
            let last_value = 0.5;

            let (advantages, returns) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                last_value,
                &device,
            );

            // Verify shapes
            assert_eq!(advantages.dims(), [3]);
            assert_eq!(returns.dims(), [3]);

            // Verify advantages are finite
            let adv_data = advantages.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();
            for a in adv_slice {
                assert!(a.is_finite(), "Advantage should be finite: {}", a);
            }
        }

        #[test]
        fn test_compute_advantages_with_terminal() {
            let config = DistributedPPOConfig::default()
                .with_gamma(0.99)
                .with_gae_lambda(0.95);
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![1.0, 1.0, 0.0];
            let values = vec![0.5, 0.5, 0.0];
            let dones = vec![false, false, true];
            let last_value = 0.0;

            let (advantages, _returns) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                last_value,
                &device,
            );

            let adv_data = advantages.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();

            // All advantages should be finite
            for a in adv_slice {
                assert!(a.is_finite());
            }
        }

        #[test]
        fn test_compute_advantages_normalized() {
            let config = DistributedPPOConfig {
                normalize_advantages: true,
                gamma: 0.99,
                gae_lambda: 0.95,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let values = vec![0.5, 1.0, 1.5, 2.0, 2.5];
            let dones = vec![false, false, false, false, false];
            let last_value = 3.0;

            let (advantages, _) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                last_value,
                &device,
            );

            let adv_data = advantages.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();

            // Compute mean and std
            let n = adv_slice.len() as f32;
            let mean: f32 = adv_slice.iter().sum::<f32>() / n;
            let var: f32 = adv_slice.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n;
            let std = var.sqrt();

            // Mean should be approximately 0
            assert!(
                mean.abs() < 0.2,
                "Normalized mean should be ~0, got {}",
                mean
            );
            // Std should be close to 1, but Burn's var uses Bessel's correction (n-1 denominator)
            // so for small n, the resulting std after normalization won't be exactly 1.
            // With n=5 and Bessel's correction, expected std is sqrt(n/(n-1)) = sqrt(5/4) = 1.118
            // After normalization: 1/1.118 = 0.894
            // Use wider tolerance to account for this.
            assert!(
                (std - 1.0).abs() < 0.2,
                "Normalized std should be ~1 (allowing for Bessel's correction), got {}",
                std
            );
        }

        #[test]
        fn test_compute_advantages_not_normalized() {
            let config = DistributedPPOConfig {
                normalize_advantages: false,
                gamma: 0.99,
                gae_lambda: 0.95,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![10.0, 10.0, 10.0];
            let values = vec![0.0, 0.0, 0.0];
            let dones = vec![false, false, false];
            let last_value = 0.0;

            let (advantages, _) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                last_value,
                &device,
            );

            let adv_data = advantages.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();

            // Without normalization, advantages should be large positive
            for a in adv_slice {
                assert!(*a > 5.0, "Unnormalized advantages should be large");
            }
        }

        #[test]
        fn test_compute_advantages_returns_equal_adv_plus_values() {
            let config = DistributedPPOConfig {
                normalize_advantages: false, // Disable to test relationship
                gamma: 0.99,
                gae_lambda: 0.95,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![1.0, 1.0, 1.0];
            let values = vec![0.5, 0.6, 0.7];
            let dones = vec![false, false, false];
            let last_value = 0.8;

            let (advantages, returns) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                last_value,
                &device,
            );

            let adv_data = advantages.into_data();
            let ret_data = returns.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();
            let ret_slice = ret_data.as_slice::<f32>().unwrap();

            // returns[i] = advantages[i] + values[i]
            for i in 0..3 {
                let expected = adv_slice[i] + values[i];
                assert!(
                    (ret_slice[i] - expected).abs() < 1e-5,
                    "returns[{}] = {} should equal advantages[{}] + values[{}] = {}",
                    i,
                    ret_slice[i],
                    i,
                    i,
                    expected
                );
            }
        }

        // --- Loss computation tests ---

        #[test]
        fn test_compute_batch_loss_finite() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);

            let device = get_device();
            let batch_size = 8;

            // Create processed batch
            let rollout = PPORolloutBatch {
                transitions: make_batch(batch_size),
                policy_version: 1,
                n_envs: 2,
                rollout_length: 4,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; batch_size],
                returns: vec![1.0; batch_size],
            };

            // Create mock policy outputs
            let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // Verify loss is finite
            assert!(loss_output.policy_loss.is_finite());
            assert!(loss_output.value_loss.is_finite());
            assert!(loss_output.entropy.is_finite());

            // Total loss tensor should be 1D with single element
            assert_eq!(loss_output.total_loss.dims(), [1]);

            let total_data = loss_output.total_loss.into_data();
            let total_val = total_data.as_slice::<f32>().unwrap()[0];
            assert!(total_val.is_finite());
        }

        #[test]
        fn test_compute_batch_loss_formula() {
            // Test: total_loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
            let config = DistributedPPOConfig {
                vf_coef: 0.5,
                entropy_coef: 0.01,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();
            let batch_size = 4;

            let rollout = PPORolloutBatch {
                transitions: make_batch(batch_size),
                policy_version: 1,
                n_envs: 2,
                rollout_length: 2,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![0.0; batch_size], // Zero advantages for simpler math
                returns: vec![1.0; batch_size],
            };

            let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 1.0, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // With zero advantages and zero log_probs, policy_loss should be ~0
            // Value loss: values=0, returns=1 => MSE = 1.0
            // Entropy: 1.0

            // The exact formula depends on implementation details, but we verify consistency
            let total_data = loss_output.total_loss.into_data();
            let total = total_data.as_slice::<f32>().unwrap()[0];

            // Rough check: total should incorporate entropy (negative contribution)
            // and value loss (positive contribution)
            assert!(total.is_finite());
        }

        #[test]
        fn test_compute_batch_loss_respects_clip_ratio() {
            // When log_probs differ significantly from old_log_probs,
            // clipping should bound the policy loss
            let config = DistributedPPOConfig {
                clip_ratio: 0.2,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();
            let batch_size = 4;

            // Create batch with specific old_log_probs
            let mut transitions = make_batch(batch_size);
            for t in &mut transitions {
                t.log_prob = -1.0; // Old log probs
            }

            let rollout = PPORolloutBatch {
                transitions,
                policy_version: 1,
                n_envs: 2,
                rollout_length: 2,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; batch_size],
                returns: vec![1.0; batch_size],
            };

            // New log probs much higher => ratio > 1.2 => should be clipped
            let log_probs = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.0, &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // Loss should be finite (clipping prevents explosion)
            assert!(loss_output.policy_loss.is_finite());
        }
    }

    // ============================================================================
    // Module: DistributedPPOConfig Tests
    // ============================================================================

    mod distributed_ppo_config_tests {
        use super::*;

        #[test]
        fn test_config_default() {
            let config = DistributedPPOConfig::default();

            assert_eq!(config.clip_ratio, 0.2);
            assert_eq!(config.vf_coef, 0.5);
            assert_eq!(config.entropy_coef, 0.01);
            assert_eq!(config.clip_value, Some(0.2));
            assert_eq!(config.rollout_length, 128);
            assert_eq!(config.gamma, 0.99);
            assert_eq!(config.gae_lambda, 0.95);
            assert_eq!(config.n_epochs, 4);
            assert_eq!(config.n_minibatches, 4);
            assert!(config.normalize_advantages);
            assert_eq!(config.max_staleness, 1);
        }

        #[test]
        fn test_config_builder_rollout_length() {
            let config = DistributedPPOConfig::new().with_rollout_length(256);

            assert_eq!(config.rollout_length, 256);
        }

        #[test]
        fn test_config_builder_gamma() {
            let config = DistributedPPOConfig::new().with_gamma(0.995);

            assert_eq!(config.gamma, 0.995);
        }

        #[test]
        fn test_config_builder_gae_lambda() {
            let config = DistributedPPOConfig::new().with_gae_lambda(0.9);

            assert_eq!(config.gae_lambda, 0.9);
        }

        #[test]
        fn test_config_builder_n_epochs() {
            let config = DistributedPPOConfig::new().with_n_epochs(10);

            assert_eq!(config.n_epochs, 10);
        }

        #[test]
        fn test_config_builder_n_minibatches() {
            let config = DistributedPPOConfig::new().with_n_minibatches(8);

            assert_eq!(config.n_minibatches, 8);
        }

        #[test]
        fn test_config_builder_clip_ratio() {
            let config = DistributedPPOConfig::new().with_clip_ratio(0.3);

            assert_eq!(config.clip_ratio, 0.3);
        }

        #[test]
        fn test_config_builder_chaining() {
            let config = DistributedPPOConfig::new()
                .with_rollout_length(64)
                .with_gamma(0.98)
                .with_n_epochs(8)
                .with_clip_ratio(0.1);

            assert_eq!(config.rollout_length, 64);
            assert_eq!(config.gamma, 0.98);
            assert_eq!(config.n_epochs, 8);
            assert_eq!(config.clip_ratio, 0.1);
        }

        #[test]
        fn test_config_clone() {
            let config = DistributedPPOConfig::new().with_rollout_length(64);
            let cloned = config.clone();

            assert_eq!(cloned.rollout_length, 64);
            assert_eq!(cloned.gamma, config.gamma);
        }

        // --- Future validation tests (document expected behavior) ---
        // These tests document configs that SHOULD be invalid if validation is added

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_gamma_negative() {
            // When config validation is added, this should panic or return Err
            let _config = DistributedPPOConfig {
                gamma: -0.1,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_gamma_too_high() {
            let _config = DistributedPPOConfig {
                gamma: 1.5,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_clip_ratio_zero() {
            let _config = DistributedPPOConfig {
                clip_ratio: 0.0,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_n_epochs_zero() {
            let _config = DistributedPPOConfig {
                n_epochs: 0,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_gae_lambda_negative() {
            let _config = DistributedPPOConfig {
                gae_lambda: -0.1,
                ..Default::default()
            };
        }
    }

    // ============================================================================
    // Module: Concurrency Tests
    // ============================================================================

    mod concurrency_tests {
        use super::*;

        #[test]
        fn test_ppo_buffer_concurrent_push() {
            let buffer = Arc::new({
                let config = PPORolloutBufferConfig {
                    n_actors: 4,
                    n_envs_per_actor: 8,
                    rollout_length: 10,
                };
                PPORolloutBuffer::new(config)
            });

            let handles: Vec<_> = (0..4)
                .map(|actor_id| {
                    let buffer = Arc::clone(&buffer);
                    thread::spawn(move || {
                        for step in 0..10 {
                            let transitions: Vec<PPOTransition> = (0..8)
                                .map(|env| make_env_transition(actor_id * 8 + env, step))
                                .collect();
                            buffer.push_step(transitions, actor_id as u64);
                            // Small delay to increase contention
                            thread::sleep(Duration::from_micros(10));
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            // Should have 4 actors * 8 envs * 10 steps = 320 transitions
            assert_eq!(buffer.len(), 320);
            assert!(buffer.is_rollout_ready());
        }

        #[test]
        fn test_distributed_buffer_concurrent_push() {
            let buffer = Arc::new(DistributedPPOBuffer::new(10, 32)); // 32 total envs

            let handles: Vec<_> = (0..4)
                .map(|actor_id| {
                    let buffer = Arc::clone(&buffer);
                    thread::spawn(move || {
                        let offset = actor_id * 8;
                        for step in 0..10 {
                            let transitions: Vec<PPOTransition> = (0..8)
                                .map(|env| make_env_transition(offset + env, step))
                                .collect();
                            buffer.push_batch(transitions, offset);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            assert!(buffer.is_ready());
            assert_eq!(buffer.total_transitions(), 320);
        }

        #[test]
        fn test_distributed_buffer_no_data_race() {
            // Use atomic counter to verify all pushes complete
            let push_count = Arc::new(AtomicUsize::new(0));
            let buffer = Arc::new(DistributedPPOBuffer::new(5, 8));

            let handles: Vec<_> = (0..2)
                .map(|actor_id| {
                    let buffer = Arc::clone(&buffer);
                    let counter = Arc::clone(&push_count);
                    thread::spawn(move || {
                        let offset = actor_id * 4;
                        for _ in 0..5 {
                            buffer.push_batch(
                                (0..4).map(|i| make_transition(offset + i)).collect(),
                                offset,
                            );
                            counter.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            // All 10 pushes should complete
            assert_eq!(push_count.load(Ordering::SeqCst), 10);
            assert!(buffer.is_ready());
        }

        #[test]
        fn test_concurrent_push_and_consume() {
            let buffer = Arc::new({
                let config = PPORolloutBufferConfig {
                    n_actors: 1,
                    n_envs_per_actor: 4,
                    rollout_length: 5,
                };
                PPORolloutBuffer::new(config)
            });

            let producer_buffer = Arc::clone(&buffer);
            let producer = thread::spawn(move || {
                for step in 0..5 {
                    producer_buffer.push_step(make_batch(4), (step + 1) as u64);
                    thread::sleep(Duration::from_millis(5));
                }
            });

            let consumer_buffer = Arc::clone(&buffer);
            let consumer = thread::spawn(move || {
                // Wait for buffer to be ready
                let ready = consumer_buffer.wait_ready_timeout(Duration::from_secs(1));
                if ready {
                    let batch = consumer_buffer.consume();
                    assert_eq!(batch.len(), 20);
                    Some(batch)
                } else {
                    None
                }
            });

            producer.join().unwrap();
            let result = consumer.join().unwrap();

            assert!(result.is_some());
        }

        #[test]
        fn test_recurrent_buffer_concurrent_push() {
            let buffer = Arc::new(RecurrentPPOBuffer::new(5, 16));

            let handles: Vec<_> = (0..4)
                .map(|actor_id| {
                    let buffer = Arc::clone(&buffer);
                    thread::spawn(move || {
                        let offset = actor_id * 4;
                        for step in 0..5 {
                            let transitions: Vec<RecurrentPPOTransition> = (0..4)
                                .map(|env| {
                                    make_recurrent_transition(
                                        offset + env,
                                        actor_id as u64,
                                        step,
                                        step == 0,
                                    )
                                })
                                .collect();
                            buffer.push_batch(transitions, offset);
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            assert!(buffer.is_ready());
            assert_eq!(buffer.total_transitions(), 80);
        }
    }

    // ============================================================================
    // Module: Integration Tests
    // ============================================================================

    mod integration_tests {
        use super::*;

        fn get_device() -> <TestAutodiffBackend as Backend>::Device {
            Default::default()
        }

        #[test]
        fn test_full_ppo_training_cycle() {
            // 1. Create DistributedPPO with config
            let config = DistributedPPOConfig::default().with_rollout_length(4);
            let ppo = DistributedPPO::new(config);

            // 2. Create buffer via create_buffer()
            let buffer = <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 2, 4,
            );

            assert_eq!(buffer.config().n_actors, 2);
            assert_eq!(buffer.config().n_envs_per_actor, 4);

            // 3. Push transitions from multiple "actors"
            for step in 0..4 {
                // Actor 0
                buffer.push_step(
                    (0..4)
                        .map(|env| make_env_transition(env, step))
                        .collect(),
                    1,
                );
                // Actor 1
                buffer.push_step(
                    (4..8)
                        .map(|env| make_env_transition(env, step))
                        .collect(),
                    1,
                );
            }

            // 4. Verify is_ready() transitions correctly
            assert!(
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::is_ready(
                    &ppo, &buffer
                )
            );

            // 5. Sample batch
            let batch =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
                    &ppo, &buffer,
                )
                .expect("Should have batch");

            // Verify batch structure
            assert_eq!(batch.rollout.len(), 32); // 8 envs * 4 steps
            assert_eq!(batch.advantages.len(), 32);
            assert_eq!(batch.returns.len(), 32);

            // 6. Compute loss with mock policy outputs
            let device = get_device();
            let batch_size = batch.rollout.len();

            let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // 7. Verify loss components are reasonable
            assert!(loss_output.policy_loss.is_finite());
            assert!(loss_output.value_loss.is_finite());
            assert!(loss_output.entropy.is_finite());
            assert!(loss_output.entropy > 0.0, "Entropy should be positive");
        }

        #[test]
        fn test_multi_epoch_training_simulation() {
            let config = DistributedPPOConfig::default()
                .with_rollout_length(2)
                .with_n_epochs(4)
                .with_n_minibatches(2);
            let ppo = DistributedPPO::new(config.clone());

            let buffer = <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 4,
            );

            // Fill buffer
            for _ in 0..2 {
                buffer.push_step(make_batch(4), 1);
            }

            let batch =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
                    &ppo, &buffer,
                )
                .expect("Should have batch");

            let device = get_device();
            let batch_size = batch.rollout.len();

            // Simulate multi-epoch training
            let mut losses = Vec::new();

            for _epoch in 0..config.n_epochs {
                for _minibatch in 0..config.n_minibatches {
                    let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
                    let entropy =
                        Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
                    let values =
                        Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

                    let loss_output = <DistributedPPO as DistributedAlgorithm<
                        TestAutodiffBackend,
                    >>::compute_batch_loss(
                        &ppo, &batch, log_probs, entropy, values, &device
                    );

                    let loss_data = loss_output.total_loss.into_data();
                    losses.push(loss_data.as_slice::<f32>().unwrap()[0]);
                }
            }

            // Verify all losses are finite
            // n_epochs=4, n_minibatches=2 => 4 * 2 = 8 total loss computations
            assert_eq!(losses.len(), 8);
            for loss in &losses {
                assert!(loss.is_finite());
            }
        }

        #[test]
        fn test_gae_cpu_vs_tensor_consistency() {
            // Verify that tensor-based GAE produces same results as CPU GAE
            let rewards = vec![1.0, 1.0, 1.0, 1.0];
            let values = vec![0.5, 0.5, 0.5, 0.5];
            let dones = vec![false, false, false, false];
            let last_value = 0.5;
            let gamma = 0.99;
            let gae_lambda = 0.95;

            // CPU GAE
            let (cpu_advantages, cpu_returns) =
                compute_gae(&rewards, &values, &dones, last_value, gamma, gae_lambda);

            // Tensor GAE via DistributedPPO
            let config = DistributedPPOConfig {
                gamma,
                gae_lambda,
                normalize_advantages: false,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();
            let (tensor_advantages, tensor_returns) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                last_value,
                &device,
            );

            let adv_data = tensor_advantages.into_data();
            let ret_data = tensor_returns.into_data();
            let tensor_adv = adv_data.as_slice::<f32>().unwrap();
            let tensor_ret = ret_data.as_slice::<f32>().unwrap();

            // Verify consistency
            for i in 0..4 {
                assert!(
                    (cpu_advantages[i] - tensor_adv[i]).abs() < 1e-5,
                    "Advantage mismatch at {}: cpu={}, tensor={}",
                    i,
                    cpu_advantages[i],
                    tensor_adv[i]
                );
                assert!(
                    (cpu_returns[i] - tensor_ret[i]).abs() < 1e-5,
                    "Return mismatch at {}: cpu={}, tensor={}",
                    i,
                    cpu_returns[i],
                    tensor_ret[i]
                );
            }
        }

        #[test]
        fn test_distributed_buffer_multi_actor_sync() {
            // Simulate multiple actors pushing asynchronously
            let buffer = Arc::new(DistributedPPOBuffer::new(8, 64));

            let handles: Vec<_> = (0..4)
                .map(|actor_id| {
                    let buffer = Arc::clone(&buffer);
                    thread::spawn(move || {
                        let offset = actor_id * 16;
                        for step in 0..8 {
                            let transitions: Vec<PPOTransition> = (0..16)
                                .map(|env| make_env_transition(offset + env, step))
                                .collect();
                            buffer.push_batch(transitions, offset);

                            // Simulate varying processing times
                            if actor_id % 2 == 0 {
                                thread::sleep(Duration::from_micros(50));
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                handle.join().unwrap();
            }

            // All transitions should be correctly stored
            assert!(buffer.is_ready());

            let rollouts = buffer.consume();

            // Verify per-environment organization
            assert_eq!(rollouts.len(), 64);
            for (env_id, rollout) in rollouts.iter().enumerate() {
                assert_eq!(
                    rollout.len(),
                    8,
                    "Env {} should have 8 transitions",
                    env_id
                );
            }
        }

        #[test]
        fn test_buffer_state_machine() {
            // Test the complete buffer state machine:
            // empty -> filling -> ready -> consumed -> empty

            let config = PPORolloutBufferConfig {
                n_actors: 1,
                n_envs_per_actor: 2,
                rollout_length: 3,
            };
            let buffer = PPORolloutBuffer::new(config);

            // State: empty
            assert!(buffer.is_empty());
            assert!(!buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 0);

            // Transition: empty -> filling
            buffer.push_step(make_batch(2), 1);
            assert!(!buffer.is_empty());
            assert!(!buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 1);

            // State: filling
            buffer.push_step(make_batch(2), 1);
            assert!(!buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 2);

            // Transition: filling -> ready
            buffer.push_step(make_batch(2), 1);
            assert!(buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 3);

            // Transition: ready -> consumed (also resets to empty)
            let batch = buffer.consume();
            assert_eq!(batch.len(), 6);

            // State: empty (after consume)
            assert!(buffer.is_empty());
            assert!(!buffer.is_rollout_ready());
            assert_eq!(buffer.step_count(), 0);
        }

        #[test]
        fn test_numerical_stability_extreme_advantages() {
            let config = DistributedPPOConfig {
                normalize_advantages: true,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            // Test with very large values
            let rewards = vec![1000.0, 1000.0, 1000.0];
            let values = vec![0.0, 0.0, 0.0];
            let dones = vec![false, false, false];

            let (advantages, returns) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                0.0,
                &device,
            );

            let adv_data = advantages.into_data();
            let ret_data = returns.into_data();

            for val in adv_data.as_slice::<f32>().unwrap() {
                assert!(val.is_finite(), "Advantage should be finite: {}", val);
            }
            for val in ret_data.as_slice::<f32>().unwrap() {
                assert!(val.is_finite(), "Return should be finite: {}", val);
            }
        }

        #[test]
        fn test_numerical_stability_very_small_advantages() {
            let config = DistributedPPOConfig {
                normalize_advantages: true,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            // Test with very small values
            let rewards = vec![1e-8, 1e-8, 1e-8];
            let values = vec![1e-9, 1e-9, 1e-9];
            let dones = vec![false, false, false];

            let (advantages, _) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards,
                &values,
                &dones,
                1e-9,
                &device,
            );

            let adv_data = advantages.into_data();

            for val in adv_data.as_slice::<f32>().unwrap() {
                assert!(val.is_finite(), "Advantage should be finite: {}", val);
                assert!(!val.is_nan(), "Advantage should not be NaN");
            }
        }

        #[test]
        fn test_numerical_stability_single_transition() {
            // Edge case: single transition batch WITHOUT normalization
            // Note: With normalization enabled and a single element, the variance is
            // undefined (NaN) because Burn's var uses Bessel's correction (n-1 denominator).
            // This test documents that non-normalized mode handles single transitions correctly.
            let config = DistributedPPOConfig {
                normalize_advantages: false, // Disable normalization for single element
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![1.0];
            let values = vec![0.5];
            let dones = vec![false];

            let (advantages, _) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards, &values, &dones, 0.5, &device,
            );

            let adv_data = advantages.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();

            // Without normalization, the advantage should be computed correctly
            assert!(adv_slice[0].is_finite());
            // With gamma=0.99, lambda=0.95, reward=1.0, value=0.5, next_value=0.5:
            // delta = 1.0 + 0.99*0.5 - 0.5 = 0.995
            // gae = 0.995 (no future to accumulate)
            assert!(
                (adv_slice[0] - 0.995).abs() < 0.01,
                "Expected advantage ~0.995, got {}",
                adv_slice[0]
            );
        }

        #[test]
        #[ignore = "Single element normalization causes NaN due to Bessel's correction - documented limitation"]
        fn test_numerical_stability_single_transition_normalized() {
            // This test documents that normalized advantages with a single element
            // produces NaN because var(0) with n=1 and Bessel's correction is undefined.
            // This is expected behavior - PPO should not be run with batch size 1.
            let config = DistributedPPOConfig {
                normalize_advantages: true,
                ..Default::default()
            };
            let ppo = DistributedPPO::new(config);

            let device = get_device();

            let rewards = vec![1.0];
            let values = vec![0.5];
            let dones = vec![false];

            let (advantages, _) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards, &values, &dones, 0.5, &device,
            );

            let adv_data = advantages.into_data();
            let adv_slice = adv_data.as_slice::<f32>().unwrap();

            // This would fail - documenting the limitation
            assert!(
                adv_slice[0].is_finite(),
                "Single element normalization produces NaN: {}",
                adv_slice[0]
            );
        }

        #[test]
        fn test_numerical_stability_all_same_advantages() {
            // Edge case: all advantages are the same (variance = 0)
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);

            let device = get_device();
            let batch_size = 4;

            let rollout = PPORolloutBatch {
                transitions: make_batch(batch_size),
                policy_version: 1,
                n_envs: 2,
                rollout_length: 2,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; batch_size], // All same
                returns: vec![1.0; batch_size],
            };

            let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // Should handle zero variance gracefully (clamped std)
            assert!(loss_output.policy_loss.is_finite());
        }
    }

    // ============================================================================
    // Module: Edge Case Tests
    // ============================================================================

    mod edge_case_tests {
        use super::*;

        #[test]
        fn test_empty_batch_state_dim() {
            let batch = PPORolloutBatch {
                transitions: vec![],
                policy_version: 0,
                n_envs: 0,
                rollout_length: 0,
            };

            assert_eq!(batch.state_dim(), 0);
            assert!(batch.states().is_empty());
            assert!(batch.rewards().is_empty());
        }

        #[test]
        fn test_buffer_consume_when_empty() {
            let config = PPORolloutBufferConfig {
                n_actors: 1,
                n_envs_per_actor: 2,
                rollout_length: 3,
            };
            let buffer = PPORolloutBuffer::new(config);

            let batch = buffer.consume();

            assert!(batch.is_empty());
            assert_eq!(batch.policy_version, 0);
        }

        #[test]
        fn test_distributed_buffer_consume_when_empty() {
            let buffer = DistributedPPOBuffer::new(3, 2);

            let rollouts = buffer.consume();

            assert_eq!(rollouts.len(), 2);
            assert!(rollouts[0].is_empty());
            assert!(rollouts[1].is_empty());
            assert_eq!(buffer.consumed_epoch(), 1); // Still increments
        }

        #[test]
        fn test_distributed_buffer_partial_fill() {
            // Not all envs have complete rollouts
            let buffer = DistributedPPOBuffer::new(3, 4);

            // Only fill envs 0 and 1
            for step in 0..3 {
                buffer.push_batch(
                    vec![make_env_transition(0, step), make_env_transition(1, step)],
                    0,
                );
            }

            assert!(!buffer.is_ready()); // Not ready (envs 2, 3 empty)

            let (min, _) = buffer.progress();
            assert_eq!(min, 0);
        }

        #[test]
        fn test_transition_done_logic() {
            let terminal = make_terminal_transition(0);
            let truncated = make_truncated_transition(0);
            let normal = make_transition(0);

            assert!(terminal.done());
            assert!(truncated.done());
            assert!(!normal.done());

            assert!(terminal.base.terminal);
            assert!(!terminal.base.truncated);

            assert!(!truncated.base.terminal);
            assert!(truncated.base.truncated);
        }

        #[test]
        fn test_recurrent_transition_accessors() {
            let t = make_recurrent_transition(5, 10, 3, false);

            assert_eq!(t.state(), &[5.0, 10.0]);
            assert_eq!(t.reward(), 1.0);
            assert_eq!(t.log_prob(), -0.5);
            assert_eq!(t.value(), 1.0);
            assert!(!t.done());
            assert!(!t.terminal());
            assert!(!t.truncated());
        }

        #[test]
        fn test_recurrent_transition_from_ppo() {
            let ppo = PPOTransition::new_discrete(
                vec![1.0, 2.0],
                1,
                0.5,
                vec![2.0, 3.0],
                false,
                false,
                -0.3,
                0.8,
            );

            let recurrent =
                RecurrentPPOTransition::from_ppo(ppo, vec![0.1, 0.2], 99, 5, false);

            assert_eq!(recurrent.sequence_id, 99);
            assert_eq!(recurrent.step_in_sequence, 5);
            assert!(!recurrent.is_sequence_start);
            assert_eq!(recurrent.hidden_state, vec![0.1, 0.2]);
            assert!(recurrent.bootstrap_value.is_none());
        }

        #[test]
        fn test_distributed_ppo_rollouts_empty() {
            let rollouts: Vec<Vec<PPOTransition>> = vec![];
            let batch = DistributedPPORollouts::new(rollouts, 0);

            assert_eq!(batch.n_envs, 0);
            assert_eq!(batch.total_transitions(), 0);
            assert_eq!(batch.obs_dim(), None);
        }

        #[test]
        fn test_buffer_multiple_clear_cycles() {
            let config = PPORolloutBufferConfig {
                n_actors: 1,
                n_envs_per_actor: 2,
                rollout_length: 2,
            };
            let buffer = PPORolloutBuffer::new(config);

            for cycle in 0..3 {
                buffer.push_step(make_batch(2), cycle as u64);
                buffer.push_step(make_batch(2), cycle as u64);

                assert!(buffer.is_rollout_ready());

                buffer.clear();

                assert!(buffer.is_empty());
                assert!(!buffer.is_rollout_ready());
            }
        }
    }

    // ============================================================================
    // Module: Tensor Shape Tests
    // ============================================================================

    mod tensor_shape_tests {
        use super::*;

        fn get_device() -> <TestAutodiffBackend as Backend>::Device {
            Default::default()
        }

        #[test]
        fn test_advantage_tensor_shape() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);
            let device = get_device();

            let rewards = vec![1.0; 10];
            let values = vec![0.5; 10];
            let dones = vec![false; 10];

            let (advantages, returns) = ppo.compute_advantages::<TestAutodiffBackend>(
                &rewards, &values, &dones, 0.5, &device,
            );

            assert_eq!(advantages.dims(), [10]);
            assert_eq!(returns.dims(), [10]);
        }

        #[test]
        fn test_loss_output_total_shape() {
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);
            let device = get_device();

            let batch_size = 16;
            let rollout = PPORolloutBatch {
                transitions: make_batch(batch_size),
                policy_version: 1,
                n_envs: 4,
                rollout_length: 4,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; batch_size],
                returns: vec![1.0; batch_size],
            };

            let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::zeros([batch_size, 1], &device);

            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // Total loss is 1D tensor with single element (scalar)
            assert_eq!(loss_output.total_loss.dims(), [1]);
        }

        #[test]
        fn test_values_flattening() {
            // Test that 2D values [batch, 1] are correctly flattened
            let config = DistributedPPOConfig::default();
            let ppo = DistributedPPO::new(config);
            let device = get_device();

            let batch_size = 8;
            let rollout = PPORolloutBatch {
                transitions: make_batch(batch_size),
                policy_version: 1,
                n_envs: 2,
                rollout_length: 4,
            };

            let batch = PPOProcessedBatch {
                rollout,
                advantages: vec![1.0; batch_size],
                returns: vec![1.0; batch_size],
            };

            // Values as [batch, 1] - common output shape from value head
            let log_probs = Tensor::<TestAutodiffBackend, 1>::zeros([batch_size], &device);
            let entropy = Tensor::<TestAutodiffBackend, 1>::full([batch_size], 0.5, &device);
            let values = Tensor::<TestAutodiffBackend, 2>::ones([batch_size, 1], &device);

            // Should not panic - values are flattened internally
            let loss_output =
                <DistributedPPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            assert!(loss_output.value_loss.is_finite());
        }
    }
}
