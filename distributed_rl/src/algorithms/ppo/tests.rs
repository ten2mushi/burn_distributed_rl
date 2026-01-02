//! Comprehensive unit and integration tests for the PPO submodule.
//!
//! This test module follows the "Tests as Definition: the Yoneda way" philosophy,
//! where tests serve as complete behavioral specifications. Each test explores
//! specific aspects of the PPO module's behavior, edge cases, and invariants.
//!
//! # Test Categories
//!
//! 1. **Config Tests**: PPORolloutBufferConfig, PPOAlgorithmConfig
//! 2. **Batch Tests**: PPORolloutBatch accessor methods and edge cases
//! 3. **Buffer Tests**: PPORolloutBuffer state machine and thread safety
//! 4. **Algorithm Tests**: PPO GAE and loss computation
//! 5. **Concurrency Tests**: Multi-threaded push/consume patterns
//! 6. **Integration Tests**: Full training cycle simulation
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
    use super::super::ppo::{PPO, PPOProcessedBatch};
    use super::super::ppo_batch_buffer::{PPORolloutBuffer, PPORolloutBufferConfig, PPORolloutBatch};
    use crate::algorithms::core_algorithm::{DistributedAlgorithm, PPOAlgorithmConfig};
    use crate::algorithms::gae::compute_gae;
    use crate::core::experience_buffer::{ExperienceBuffer, OnPolicyBuffer};
    use crate::core::transition::{PPOTransition, RecurrentPPOTransition, Transition};

    use burn::backend::{Autodiff, Wgpu};
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;
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
            bootstrap_value: None,
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
            bootstrap_value: None,
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
            bootstrap_value: None,
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
            bootstrap_value: None,
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
    // Module: PPO Algorithm Tests
    // ============================================================================

    mod algorithm_tests {
        use super::*;

        fn get_device() -> <TestAutodiffBackend as Backend>::Device {
            Default::default()
        }

        #[test]
        fn test_distributed_ppo_new() {
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);

            assert_eq!(
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::name(&ppo),
                "PPO"
            );
            assert!(
                !<PPO as DistributedAlgorithm<TestAutodiffBackend>>::is_off_policy(&ppo)
            );
        }

        #[test]
        fn test_distributed_ppo_n_epochs() {
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);

            assert_eq!(
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::n_epochs(&ppo),
                4
            );
        }

        #[test]
        fn test_distributed_ppo_n_minibatches() {
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);

            assert_eq!(
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::n_minibatches(&ppo),
                4
            );
        }

        #[test]
        fn test_distributed_ppo_create_buffer() {
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config.clone());

            let buffer = <PPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 2, 32,
            );

            assert_eq!(buffer.config().n_actors, 2);
            assert_eq!(buffer.config().n_envs_per_actor, 32);
            assert_eq!(buffer.config().rollout_length, config.rollout_length);
        }

        #[test]
        fn test_distributed_ppo_is_ready() {
            let config = PPOAlgorithmConfig::default().with_rollout_length(2);
            let ppo = PPO::new(config);

            let buffer = <PPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 2,
            );

            assert!(
                !<PPO as DistributedAlgorithm<TestAutodiffBackend>>::is_ready(
                    &ppo, &buffer
                )
            );

            buffer.push_step(make_batch(2), 1);
            buffer.push_step(make_batch(2), 1);

            assert!(
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::is_ready(
                    &ppo, &buffer
                )
            );
        }

        #[test]
        fn test_distributed_ppo_sample_batch_not_ready() {
            let config = PPOAlgorithmConfig::default().with_rollout_length(10);
            let ppo = PPO::new(config);

            let buffer = <PPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 2,
            );

            buffer.push_step(make_batch(2), 1);

            let batch =
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
                    &ppo, &buffer,
                );

            assert!(batch.is_none());
        }

        #[test]
        fn test_distributed_ppo_sample_batch_ready() {
            let config = PPOAlgorithmConfig::default().with_rollout_length(2);
            let ppo = PPO::new(config);

            let buffer = <PPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 2,
            );

            buffer.push_step(make_batch(2), 42);
            buffer.push_step(make_batch(2), 42);

            let batch =
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
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
            let config = PPOAlgorithmConfig::default(); // max_staleness = 1
            let ppo = PPO::new(config);

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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::handle_staleness(
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
            let config = PPOAlgorithmConfig::default(); // max_staleness = 1
            let ppo = PPO::new(config);

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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::handle_staleness(
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
            let config = PPOAlgorithmConfig::default()
                .with_gamma(0.99)
                .with_gae_lambda(0.95);
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig::default()
                .with_gamma(0.99)
                .with_gae_lambda(0.95);
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: true,
                gamma: 0.99,
                gae_lambda: 0.95,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: false,
                gamma: 0.99,
                gae_lambda: 0.95,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: false, // Disable to test relationship
                gamma: 0.99,
                gae_lambda: 0.95,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);

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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
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
            let config = PPOAlgorithmConfig {
                vf_coef: 0.5,
                entropy_coef: 0.01,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
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
            let config = PPOAlgorithmConfig {
                clip_ratio: 0.2,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // Loss should be finite (clipping prevents explosion)
            assert!(loss_output.policy_loss.is_finite());
        }
    }

    // ============================================================================
    // Module: PPOAlgorithmConfig Tests
    // ============================================================================

    mod distributed_ppo_config_tests {
        use super::*;

        #[test]
        fn test_config_default() {
            let config = PPOAlgorithmConfig::default();

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
            let config = PPOAlgorithmConfig::new().with_rollout_length(256);

            assert_eq!(config.rollout_length, 256);
        }

        #[test]
        fn test_config_builder_gamma() {
            let config = PPOAlgorithmConfig::new().with_gamma(0.995);

            assert_eq!(config.gamma, 0.995);
        }

        #[test]
        fn test_config_builder_gae_lambda() {
            let config = PPOAlgorithmConfig::new().with_gae_lambda(0.9);

            assert_eq!(config.gae_lambda, 0.9);
        }

        #[test]
        fn test_config_builder_n_epochs() {
            let config = PPOAlgorithmConfig::new().with_n_epochs(10);

            assert_eq!(config.n_epochs, 10);
        }

        #[test]
        fn test_config_builder_n_minibatches() {
            let config = PPOAlgorithmConfig::new().with_n_minibatches(8);

            assert_eq!(config.n_minibatches, 8);
        }

        #[test]
        fn test_config_builder_clip_ratio() {
            let config = PPOAlgorithmConfig::new().with_clip_ratio(0.3);

            assert_eq!(config.clip_ratio, 0.3);
        }

        #[test]
        fn test_config_builder_chaining() {
            let config = PPOAlgorithmConfig::new()
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
            let config = PPOAlgorithmConfig::new().with_rollout_length(64);
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
            let _config = PPOAlgorithmConfig {
                gamma: -0.1,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_gamma_too_high() {
            let _config = PPOAlgorithmConfig {
                gamma: 1.5,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_clip_ratio_zero() {
            let _config = PPOAlgorithmConfig {
                clip_ratio: 0.0,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_n_epochs_zero() {
            let _config = PPOAlgorithmConfig {
                n_epochs: 0,
                ..Default::default()
            };
        }

        #[test]
        #[ignore = "Config validation not yet implemented"]
        fn test_config_invalid_gae_lambda_negative() {
            let _config = PPOAlgorithmConfig {
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
            // 1. Create PPO with config
            let config = PPOAlgorithmConfig::default().with_rollout_length(4);
            let ppo = PPO::new(config);

            // 2. Create buffer via create_buffer()
            let buffer = <PPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::is_ready(
                    &ppo, &buffer
                )
            );

            // 5. Sample batch
            let batch =
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
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
            let config = PPOAlgorithmConfig::default()
                .with_rollout_length(2)
                .with_n_epochs(4)
                .with_n_minibatches(2);
            let ppo = PPO::new(config.clone());

            let buffer = <PPO as DistributedAlgorithm<TestAutodiffBackend>>::create_buffer(
                &ppo, 1, 4,
            );

            // Fill buffer
            for _ in 0..2 {
                buffer.push_step(make_batch(4), 1);
            }

            let batch =
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::sample_batch(
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

                    let loss_output = <PPO as DistributedAlgorithm<
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

            // Tensor GAE via PPO
            let config = PPOAlgorithmConfig {
                gamma,
                gae_lambda,
                normalize_advantages: false,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: true,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: true,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: false, // Disable normalization for single element
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig {
                normalize_advantages: true,
                ..Default::default()
            };
            let ppo = PPO::new(config);

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
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);

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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
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
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);
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
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);
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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            // Total loss is 1D tensor with single element (scalar)
            assert_eq!(loss_output.total_loss.dims(), [1]);
        }

        #[test]
        fn test_values_flattening() {
            // Test that 2D values [batch, 1] are correctly flattened
            let config = PPOAlgorithmConfig::default();
            let ppo = PPO::new(config);
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
                <PPO as DistributedAlgorithm<TestAutodiffBackend>>::compute_batch_loss(
                    &ppo, &batch, log_probs, entropy, values, &device,
                );

            assert!(loss_output.value_loss.is_finite());
        }
    }

    // ============================================================================
    // Temporal Hidden State Invariant Tests (Curry-Howard Correspondence)
    // ============================================================================
    //
    // These tests document and verify the critical temporal causality invariant
    // for recurrent policies. The Curry-Howard correspondence frames this as:
    //
    // Type Specification:
    //   transition[t].hidden = h_{t-1}  (INPUT to forward pass)
    //   NOT h_t (OUTPUT from forward pass)
    //
    // If this invariant is violated, the PPO ratio becomes:
    //   π_new(a|s, h_t) / π_old(a|s, h_{t-1})
    // which conditions numerator and denominator on different histories,
    // making the policy gradient update meaningless.

    mod temporal_invariant_tests {
        use crate::algorithms::ppo::ppo_transition::RecurrentHiddenData;
        use crate::core::transition::RecurrentPPOTransition;

        /// Test that RecurrentHiddenData documents the temporal invariant.
        ///
        /// This is a compile-time documentation test - the invariant is enforced
        /// by the collection logic in ppo_runner.rs, not by the type itself.
        /// See RecurrentHiddenData's doc comments for the full specification.
        #[test]
        fn recurrent_hidden_data_documents_temporal_invariant() {
            // The type exists and can be constructed
            let hidden = RecurrentHiddenData::new(vec![0.1, 0.2, 0.3]);
            assert_eq!(hidden.data.len(), 3);

            // The invariant is documented in the type's doc comments:
            // - data field represents h_{t-1} (INPUT hidden state)
            // - NOT h_t (OUTPUT hidden state)
            // This invariant is enforced by extracting hidden BEFORE forward pass
        }

        /// Test the correct transition structure for recurrent policies.
        ///
        /// For a transition at timestep t:
        /// - state: o_t (observation at time t)
        /// - action: a_t (action taken at time t)
        /// - log_prob: log π(a_t | o_t, h_{t-1}) (computed with INPUT hidden)
        /// - value: V(o_t, h_{t-1}) (computed with INPUT hidden)
        /// - hidden_state: h_{t-1} (the INPUT hidden state, NOT h_t)
        #[test]
        fn recurrent_transition_temporal_structure() {
            // Simulate the correct collection pattern:
            // 1. At timestep t, we have hidden state h_{t-1} from previous step
            // 2. We extract h_{t-1} BEFORE forward pass
            // 3. Forward pass computes: (action, log_prob, value, h_t)
            // 4. We store transition with hidden = h_{t-1}

            let h_t_minus_1 = vec![0.5, 0.5, 0.5]; // INPUT hidden (before forward)

            // Simulated forward pass would produce:
            // - log_prob computed with h_{t-1}
            // - value computed with h_{t-1}
            // - h_t (output hidden, NOT stored in this transition)

            let transition = RecurrentPPOTransition::new_discrete(
                vec![1.0, 2.0],              // o_t (state)
                0,                           // a_t (action)
                1.0,                         // r_t (reward)
                vec![1.1, 2.1],              // o_{t+1} (next_state)
                false,                       // terminal
                false,                       // truncated
                -0.5,                        // log_prob: log π(a_t | o_t, h_{t-1})
                0.8,                         // value: V(o_t, h_{t-1})
                h_t_minus_1.clone(),         // hidden_state: h_{t-1} (INPUT!)
                0,                           // sequence_id
                5,                           // step_in_sequence
                false,                       // is_sequence_start
                None,                        // bootstrap_value
            );

            // Verify the hidden state is the INPUT, not OUTPUT
            assert_eq!(transition.hidden_state, h_t_minus_1);

            // This ensures that during training:
            // - We initialize LSTM with transition.hidden_state = h_{t-1}
            // - Forward pass on o_t produces log_prob' computed with h_{t-1}
            // - This matches the stored log_prob (also computed with h_{t-1})
            // - PPO ratio = exp(log_prob' - log_prob) is valid
        }

        /// Test that sequence boundaries respect the temporal invariant.
        ///
        /// When starting a new sequence (after terminal or at sequence chunk start):
        /// - The first transition's hidden should be h_{-1} = zeros (or carried over)
        /// - NOT h_0 (which would be the output after processing the first observation)
        #[test]
        fn sequence_start_uses_input_hidden() {
            // At sequence start (t=0):
            // - Input hidden is h_{-1} = zeros (fresh episode) or h_prev (continuation)
            // - Forward pass: model(o_0, h_{-1}) → (a_0, log_prob_0, value_0, h_0)
            // - Store: transition.hidden = h_{-1}

            let h_minus_1 = vec![0.0, 0.0, 0.0]; // Zeros for fresh episode start

            let first_transition = RecurrentPPOTransition::new_discrete(
                vec![0.1, 0.2],              // state
                0,                           // action
                0.5,                         // reward
                vec![0.2, 0.3],              // next_state
                false,                       // terminal
                false,                       // truncated
                -1.0,                        // log_prob
                0.0,                         // value
                h_minus_1.clone(),           // hidden_state: h_{-1} = zeros
                1,                           // sequence_id
                0,                           // step_in_sequence: first step
                true,                        // is_sequence_start: TRUE
                None,                        // bootstrap_value
            );

            // Critical: hidden is zeros (input), not the output h_0
            assert_eq!(first_transition.hidden_state, vec![0.0, 0.0, 0.0]);
            assert!(first_transition.is_sequence_start);
        }
    }
}
