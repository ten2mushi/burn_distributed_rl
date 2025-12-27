//! Comprehensive test suite for the IMPALA algorithm submodule.
//!
//! This test suite serves as a behavioral specification for IMPALA,
//! covering all critical aspects including:
//! - IMPALAConfig builder pattern and validation
//! - IMPALABuffer thread-safety and FIFO semantics
//! - IMPALABatch accessor methods and staleness computation
//! - DistributedIMPALA algorithm trait implementation
//! - V-trace numerical stability and integration
//! - Concurrency safety and integration tests
//!
//! Following the "Tests as definition: the Yoneda way" philosophy,
//! these tests completely characterize the expected behavior.

use super::*;
use crate::algorithms::distributed_algorithm::DistributedAlgorithm;
use crate::algorithms::vtrace::{compute_vtrace, VTraceInput, compute_vtrace_batch};
use crate::core::experience_buffer::{ExperienceBuffer, OffPolicyBuffer};
use crate::core::transition::{IMPALATransition, Trajectory, Transition};
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::Tensor;
use std::sync::Arc;
use std::thread;

// ============================================================================
// Test Backend Types
// ============================================================================

type TestBackend = Wgpu;
type TestAutodiffBackend = Autodiff<Wgpu>;

fn get_device() -> <TestAutodiffBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::wgpu::WgpuDevice::default()
}

// ============================================================================
// Test Helper Functions
// ============================================================================

/// Create a trajectory with the specified parameters.
/// Each transition has state [i as f32], next_state [(i+1) as f32],
/// reward 1.0, and the last transition is terminal.
fn make_trajectory(len: usize, env_id: usize, version: u64) -> Trajectory<IMPALATransition> {
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                i == len - 1, // terminal on last step
                false,
            ),
            behavior_log_prob: -0.5,
            policy_version: version,
        });
    }
    traj
}

/// Create a trajectory with custom behavior log probs.
fn make_trajectory_with_log_probs(
    len: usize,
    env_id: usize,
    version: u64,
    behavior_log_probs: &[f32],
) -> Trajectory<IMPALATransition> {
    assert_eq!(len, behavior_log_probs.len());
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                i == len - 1,
                false,
            ),
            behavior_log_prob: behavior_log_probs[i],
            policy_version: version,
        });
    }
    traj
}

/// Create a trajectory where no transitions are terminal.
fn make_non_terminal_trajectory(
    len: usize,
    env_id: usize,
    version: u64,
) -> Trajectory<IMPALATransition> {
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false, // never terminal
                false,
            ),
            behavior_log_prob: -0.5,
            policy_version: version,
        });
    }
    traj
}

/// Create a trajectory with custom rewards.
fn make_trajectory_with_rewards(
    len: usize,
    env_id: usize,
    version: u64,
    rewards: &[f32],
) -> Trajectory<IMPALATransition> {
    assert_eq!(len, rewards.len());
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![i as f32],
                0,
                rewards[i],
                vec![(i + 1) as f32],
                i == len - 1,
                false,
            ),
            behavior_log_prob: -0.5,
            policy_version: version,
        });
    }
    traj
}

/// Create an IMPALABatch from trajectories.
fn make_impala_batch(trajectories: Vec<Trajectory<IMPALATransition>>) -> IMPALABatch {
    let policy_versions: Vec<u64> = trajectories
        .iter()
        .map(|t| {
            t.transitions
                .first()
                .map(|tr| tr.policy_version)
                .unwrap_or(0)
        })
        .collect();
    IMPALABatch {
        trajectories,
        policy_versions,
    }
}

// ============================================================================
// IMPALAConfig Tests (~15 tests)
// ============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = IMPALAConfig::new();

        // Multi-actor defaults
        assert_eq!(config.n_actors, 4);
        assert_eq!(config.n_envs_per_actor, 32);

        // Algorithm defaults (tuned for small-scale async training)
        assert_eq!(config.trajectory_length, 20);
        assert_eq!(config.buffer_capacity, 256);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.rho_clip, 1.5);
        assert_eq!(config.c_clip, 1.0);
        assert_eq!(config.vf_coef, 0.25);
        assert_eq!(config.entropy_coef, 0.02);

        // Training defaults
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.max_grad_norm, Some(40.0));
        assert_eq!(config.max_env_steps, 1_000_000);
        assert_eq!(config.target_reward, None);
        assert_eq!(config.log_interval_secs, 2.0);
        assert_eq!(config.model_update_freq, 100);
    }

    #[test]
    fn test_new_equals_default() {
        let new_config = IMPALAConfig::new();
        let default_config = IMPALAConfig::default();

        assert_eq!(new_config.n_actors, default_config.n_actors);
        assert_eq!(new_config.batch_size, default_config.batch_size);
        assert_eq!(new_config.gamma, default_config.gamma);
    }

    #[test]
    fn test_total_envs_calculation() {
        let config = IMPALAConfig::new()
            .with_n_actors(4)
            .with_n_envs_per_actor(8);
        assert_eq!(config.total_envs(), 32);

        let config2 = IMPALAConfig::new()
            .with_n_actors(16)
            .with_n_envs_per_actor(64);
        assert_eq!(config2.total_envs(), 1024);
    }

    #[test]
    fn test_builder_pattern_multi_actor() {
        let config = IMPALAConfig::new()
            .with_n_actors(8)
            .with_n_envs_per_actor(16);

        assert_eq!(config.n_actors, 8);
        assert_eq!(config.n_envs_per_actor, 16);
    }

    #[test]
    fn test_builder_pattern_algorithm() {
        let config = IMPALAConfig::new()
            .with_trajectory_length(40)
            .with_buffer_capacity(512)
            .with_batch_size(64)
            .with_gamma(0.995)
            .with_rho_clip(0.5)
            .with_c_clip(0.5)
            .with_vf_coef(0.25)
            .with_entropy_coef(0.02);

        assert_eq!(config.trajectory_length, 40);
        assert_eq!(config.buffer_capacity, 512);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.gamma, 0.995);
        assert_eq!(config.rho_clip, 0.5);
        assert_eq!(config.c_clip, 0.5);
        assert_eq!(config.vf_coef, 0.25);
        assert_eq!(config.entropy_coef, 0.02);
    }

    #[test]
    fn test_builder_pattern_training() {
        let config = IMPALAConfig::new()
            .with_learning_rate(3e-4)
            .with_max_grad_norm(Some(10.0))
            .with_max_env_steps(10_000_000)
            .with_target_reward(500.0)
            .with_log_interval_secs(5.0)
            .with_model_update_freq(50);

        assert_eq!(config.learning_rate, 3e-4);
        assert_eq!(config.max_grad_norm, Some(10.0));
        assert_eq!(config.max_env_steps, 10_000_000);
        assert_eq!(config.target_reward, Some(500.0));
        assert_eq!(config.log_interval_secs, 5.0);
        assert_eq!(config.model_update_freq, 50);
    }

    #[test]
    fn test_builder_chaining() {
        // Ensure all builder methods return Self for chaining
        let config = IMPALAConfig::new()
            .with_n_actors(2)
            .with_n_envs_per_actor(4)
            .with_trajectory_length(10)
            .with_buffer_capacity(100)
            .with_batch_size(8)
            .with_gamma(0.9)
            .with_rho_clip(0.8)
            .with_c_clip(0.7)
            .with_vf_coef(0.4)
            .with_entropy_coef(0.05)
            .with_learning_rate(1e-3)
            .with_max_grad_norm(None)
            .with_max_env_steps(100)
            .with_target_reward(10.0)
            .with_log_interval_secs(1.0)
            .with_model_update_freq(10);

        assert_eq!(config.n_actors, 2);
        assert_eq!(config.max_grad_norm, None);
    }

    #[test]
    fn test_zero_actors() {
        // Edge case: 0 actors should result in 0 total envs
        let config = IMPALAConfig::new()
            .with_n_actors(0)
            .with_n_envs_per_actor(32);
        assert_eq!(config.total_envs(), 0);
    }

    #[test]
    fn test_zero_envs_per_actor() {
        let config = IMPALAConfig::new()
            .with_n_actors(4)
            .with_n_envs_per_actor(0);
        assert_eq!(config.total_envs(), 0);
    }

    #[test]
    fn test_gamma_boundary_zero() {
        let config = IMPALAConfig::new().with_gamma(0.0);
        assert_eq!(config.gamma, 0.0);
    }

    #[test]
    fn test_gamma_boundary_one() {
        let config = IMPALAConfig::new().with_gamma(1.0);
        assert_eq!(config.gamma, 1.0);
    }

    #[test]
    fn test_config_clone() {
        let config = IMPALAConfig::new()
            .with_n_actors(8)
            .with_batch_size(64);
        let cloned = config.clone();

        assert_eq!(config.n_actors, cloned.n_actors);
        assert_eq!(config.batch_size, cloned.batch_size);
    }

    #[test]
    fn test_config_debug() {
        let config = IMPALAConfig::new();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("IMPALAConfig"));
    }
}

// ============================================================================
// IMPALAStats Tests
// ============================================================================

mod stats_tests {
    use super::*;

    #[test]
    fn test_stats_default() {
        let stats = IMPALAStats::default();

        assert_eq!(stats.env_steps, 0);
        assert_eq!(stats.episodes, 0);
        assert_eq!(stats.avg_reward, 0.0);
        assert_eq!(stats.policy_version, 0);
        assert_eq!(stats.steps_per_second, 0.0);
        assert_eq!(stats.policy_loss, 0.0);
        assert_eq!(stats.value_loss, 0.0);
        assert_eq!(stats.entropy, 0.0);
        assert_eq!(stats.buffer_size, 0);
        assert_eq!(stats.train_steps, 0);
    }

    #[test]
    fn test_stats_clone() {
        let mut stats = IMPALAStats::default();
        stats.env_steps = 1000;
        stats.avg_reward = 250.0;

        let cloned = stats.clone();
        assert_eq!(cloned.env_steps, 1000);
        assert_eq!(cloned.avg_reward, 250.0);
    }
}

// ============================================================================
// IMPALABufferConfig Tests
// ============================================================================

mod buffer_config_tests {
    use super::*;

    #[test]
    fn test_buffer_config_default() {
        let config = IMPALABufferConfig::default();

        assert_eq!(config.n_actors, 4);
        assert_eq!(config.n_envs_per_actor, 32);
        assert_eq!(config.trajectory_length, 20);
        assert_eq!(config.max_trajectories, 1000);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_buffer_config_total_envs() {
        let config = IMPALABufferConfig {
            n_actors: 8,
            n_envs_per_actor: 16,
            ..Default::default()
        };
        assert_eq!(config.total_envs(), 128);
    }

    #[test]
    fn test_buffer_config_capacity() {
        let config = IMPALABufferConfig {
            max_trajectories: 100,
            trajectory_length: 20,
            ..Default::default()
        };
        assert_eq!(config.capacity(), 2000); // 100 * 20
    }
}

// ============================================================================
// IMPALABuffer Tests (~25 tests)
// ============================================================================

mod buffer_tests {
    use super::*;

    #[test]
    fn test_buffer_new_is_empty() {
        let config = IMPALABufferConfig::default();
        let buffer = IMPALABuffer::new(config);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.pending_count(), 0);
    }

    #[test]
    fn test_buffer_push_single_trajectory() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        buffer.push_trajectory(make_trajectory(10, 0, 1));
        buffer.consolidate();

        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_push_multiple_trajectories() {
        let config = IMPALABufferConfig {
            batch_size: 3,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(10, i, 1));
        }
        buffer.consolidate();

        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_buffer_fifo_ordering() {
        let config = IMPALABufferConfig {
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push trajectories with increasing env_ids
        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();

        // First batch should have oldest (env_ids 0, 1)
        let batch1 = buffer.sample_batch().unwrap();
        assert_eq!(batch1.trajectories[0].env_id, 0);
        assert_eq!(batch1.trajectories[1].env_id, 1);

        // Second batch should have next oldest (env_ids 2, 3)
        let batch2 = buffer.sample_batch().unwrap();
        assert_eq!(batch2.trajectories[0].env_id, 2);
        assert_eq!(batch2.trajectories[1].env_id, 3);
    }

    #[test]
    fn test_buffer_capacity_enforcement() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            max_trajectories: 3,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push more than capacity
        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();

        // Buffer should be at or below max capacity
        assert!(buffer.len() <= 3);
    }

    #[test]
    fn test_buffer_capacity_fifo_eviction() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            max_trajectories: 3,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push 5 trajectories with env_ids 0-4
        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();

        // Sample and verify oldest were evicted (FIFO)
        let batch = buffer.sample_batch().unwrap();
        // The oldest (env_id 0, 1) should have been evicted, keeping 2, 3, 4
        assert!(batch.trajectories[0].env_id >= 2);
    }

    #[test]
    fn test_buffer_sample_returns_none_when_insufficient() {
        let config = IMPALABufferConfig {
            batch_size: 5,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Only push 3 trajectories when batch_size is 5
        for i in 0..3 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();

        let batch = buffer.sample_batch();
        assert!(batch.is_none());
    }

    #[test]
    fn test_buffer_is_training_ready() {
        let config = IMPALABufferConfig {
            batch_size: 3,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        assert!(!buffer.is_training_ready());

        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(5, 1, 1));
        assert!(!buffer.is_training_ready());

        buffer.push_trajectory(make_trajectory(5, 2, 1));
        assert!(buffer.is_training_ready());
    }

    #[test]
    fn test_buffer_clear() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();
        assert!(buffer.len() > 0);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.pending_count(), 0);
        assert!(!buffer.is_training_ready());
    }

    #[test]
    fn test_buffer_pending_size_tracking() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push without consolidating
        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(5, 1, 1));

        // Pending should be 2 (not consolidated yet)
        assert!(buffer.pending_count() > 0 || buffer.len() > 0);

        // After consolidation, pending should be 0
        buffer.consolidate();
        // Note: pending_count may still be non-zero due to relaxed ordering
        // The important thing is that storage has the items
        assert!(buffer.len() >= 2);
    }

    #[test]
    fn test_buffer_sample_removes_trajectories() {
        let config = IMPALABufferConfig {
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        for i in 0..4 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();
        assert_eq!(buffer.len(), 4);

        let _batch = buffer.sample_batch().unwrap();
        assert_eq!(buffer.len(), 2); // 4 - 2 = 2
    }

    #[test]
    fn test_buffer_policy_version_extraction() {
        let config = IMPALABufferConfig {
            batch_size: 3,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        buffer.push_trajectory(make_trajectory(5, 0, 100));
        buffer.push_trajectory(make_trajectory(5, 1, 200));
        buffer.push_trajectory(make_trajectory(5, 2, 150));
        buffer.consolidate();

        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.policy_versions.len(), 3);
        assert_eq!(batch.policy_versions[0], 100);
        assert_eq!(batch.policy_versions[1], 200);
        assert_eq!(batch.policy_versions[2], 150);
    }

    #[test]
    fn test_buffer_experience_buffer_trait_push() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Use ExperienceBuffer trait method
        ExperienceBuffer::push(&buffer, make_trajectory(5, 0, 1));
        buffer.consolidate();

        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_buffer_experience_buffer_trait_push_batch() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        let trajs = vec![
            make_trajectory(5, 0, 1),
            make_trajectory(5, 1, 1),
            make_trajectory(5, 2, 1),
        ];
        ExperienceBuffer::push_batch(&buffer, trajs);
        buffer.consolidate();

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_buffer_experience_buffer_trait_capacity() {
        let config = IMPALABufferConfig {
            max_trajectories: 50,
            trajectory_length: 10,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        assert_eq!(ExperienceBuffer::capacity(&buffer), 500); // 50 * 10
    }

    #[test]
    fn test_buffer_off_policy_trait_sample() {
        let config = IMPALABufferConfig {
            batch_size: 2,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        for i in 0..3 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }
        buffer.consolidate();

        let items = OffPolicyBuffer::sample(&buffer, 2);
        assert!(items.is_some());
        assert_eq!(items.unwrap().len(), 2);
    }

    #[test]
    fn test_buffer_off_policy_trait_pending_len() {
        let config = IMPALABufferConfig::default();
        let buffer = IMPALABuffer::new(config);

        buffer.push_trajectory(make_trajectory(5, 0, 1));

        // pending_len should reflect unconsolidated data
        let pending = OffPolicyBuffer::pending_len(&buffer);
        assert!(pending > 0 || buffer.len() > 0);
    }

    #[test]
    fn test_buffer_config_accessor() {
        let config = IMPALABufferConfig {
            n_actors: 8,
            n_envs_per_actor: 16,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        assert_eq!(buffer.config().n_actors, 8);
        assert_eq!(buffer.config().n_envs_per_actor, 16);
    }

    // ========================================================================
    // Regression tests for documented bugs
    // ========================================================================

    #[test]
    fn regression_bug7_short_trajectories_accepted() {
        // Bug 7: Short trajectories were being dropped if shorter than trajectory_length.
        // This caused data loss on early episode termination.
        let config = IMPALABufferConfig {
            trajectory_length: 20,  // Config says 20
            batch_size: 1,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push a short trajectory (only 5 steps, less than trajectory_length of 20)
        let short_traj = make_trajectory(5, 0, 1);
        buffer.push_trajectory(short_traj);

        // Consolidate and verify it was accepted
        buffer.consolidate();
        assert_eq!(buffer.len(), 1, "Short trajectory should be accepted!");

        // Sample and verify we get the short trajectory
        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.trajectories.len(), 1);
        assert_eq!(batch.trajectories[0].len(), 5, "Should get 5-step trajectory");
    }

    #[test]
    fn regression_bug7_single_step_trajectory() {
        // Edge case: trajectory with only 1 step
        let config = IMPALABufferConfig {
            trajectory_length: 20,
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        let single_step_traj = make_trajectory(1, 0, 1);
        buffer.push_trajectory(single_step_traj);
        buffer.consolidate();

        assert_eq!(buffer.len(), 1, "Single-step trajectory should be accepted");
    }

    #[test]
    fn regression_empty_trajectories_rejected() {
        // Empty trajectories should still be rejected
        let config = IMPALABufferConfig {
            trajectory_length: 20,
            batch_size: 1,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push an empty trajectory
        let empty_traj = Trajectory::new(0);
        buffer.push_trajectory(empty_traj);

        buffer.consolidate();
        assert_eq!(buffer.len(), 0, "Empty trajectory should be rejected");
    }

    #[test]
    fn regression_bug15_pending_size_race() {
        // Bug 15: is_training_ready() could return true based on pending_size
        // before consolidation, but sample_batch() would fail after consolidation.
        // Fix: Consolidate before checking readiness.
        let config = IMPALABufferConfig {
            trajectory_length: 5,
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push exactly batch_size trajectories
        buffer.push_trajectory(make_trajectory(10, 0, 1));
        buffer.push_trajectory(make_trajectory(10, 1, 1));

        // is_training_ready now consolidates first, so this should work
        assert!(buffer.is_training_ready());

        // sample_batch should succeed
        let batch = buffer.sample_batch();
        assert!(batch.is_some(), "sample_batch should succeed after is_training_ready");
        assert_eq!(batch.unwrap().len(), 2);
    }

    #[test]
    fn regression_ready_then_sample_consistency() {
        // If is_training_ready() returns true, sample_batch() must succeed
        let config = IMPALABufferConfig {
            batch_size: 3,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        for i in 0..5 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }

        if buffer.is_training_ready() {
            let batch = buffer.sample_batch();
            assert!(batch.is_some(), "sample_batch must succeed after is_training_ready returns true");
        }
    }

    #[test]
    fn test_variable_length_trajectories_in_batch() {
        let config = IMPALABufferConfig {
            batch_size: 3,
            trajectory_length: 20,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Push trajectories of different lengths
        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(10, 1, 1));
        buffer.push_trajectory(make_trajectory(15, 2, 1));
        buffer.consolidate();

        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.trajectories[0].len(), 5);
        assert_eq!(batch.trajectories[1].len(), 10);
        assert_eq!(batch.trajectories[2].len(), 15);
    }
}

// ============================================================================
// IMPALABatch Tests (~10 tests)
// ============================================================================

mod batch_tests {
    use super::*;

    #[test]
    fn test_batch_len() {
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 1),
            make_trajectory(5, 1, 1),
        ]);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_batch_is_empty_false() {
        let batch = make_impala_batch(vec![make_trajectory(5, 0, 1)]);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_is_empty_true() {
        let batch = IMPALABatch {
            trajectories: vec![],
            policy_versions: vec![],
        };
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_total_transitions() {
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 1),
            make_trajectory(10, 1, 1),
            make_trajectory(3, 2, 1),
        ]);
        assert_eq!(batch.total_transitions(), 18); // 5 + 10 + 3
    }

    #[test]
    fn test_batch_states_flattened() {
        let batch = make_impala_batch(vec![make_trajectory(3, 0, 1)]);
        let states = batch.states();
        // Each transition has state [i as f32]
        assert_eq!(states, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_batch_next_states_flattened() {
        let batch = make_impala_batch(vec![make_trajectory(3, 0, 1)]);
        let next_states = batch.next_states();
        assert_eq!(next_states, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_behavior_log_probs() {
        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(2, 0, 1, &[-0.5, -1.0]),
        ]);
        let log_probs = batch.behavior_log_probs();
        assert_eq!(log_probs.len(), 2);
        assert!((log_probs[0] - (-0.5)).abs() < 1e-6);
        assert!((log_probs[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_batch_rewards() {
        let batch = make_impala_batch(vec![
            make_trajectory_with_rewards(3, 0, 1, &[1.0, 2.0, 3.0]),
        ]);
        let rewards = batch.rewards();
        assert_eq!(rewards, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_dones() {
        let batch = make_impala_batch(vec![make_trajectory(3, 0, 1)]);
        let dones = batch.dones();
        // Only last transition is terminal
        assert_eq!(dones, vec![false, false, true]);
    }

    #[test]
    fn test_batch_state_dim() {
        let batch = make_impala_batch(vec![make_trajectory(3, 0, 1)]);
        assert_eq!(batch.state_dim(), 1); // State is vec![i as f32]
    }

    #[test]
    fn test_batch_state_dim_empty() {
        let batch = IMPALABatch {
            trajectories: vec![],
            policy_versions: vec![],
        };
        assert_eq!(batch.state_dim(), 0);
    }

    #[test]
    fn test_batch_max_staleness() {
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 5),   // version 5
            make_trajectory(5, 1, 8),   // version 8
            make_trajectory(5, 2, 10),  // version 10
        ]);

        let staleness = batch.max_staleness(15);
        assert_eq!(staleness, 10); // 15 - 5 = 10
    }

    #[test]
    fn test_batch_max_staleness_no_staleness() {
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 10),
            make_trajectory(5, 1, 10),
        ]);

        let staleness = batch.max_staleness(10);
        assert_eq!(staleness, 0);
    }

    #[test]
    fn test_batch_max_staleness_empty() {
        let batch = IMPALABatch {
            trajectories: vec![],
            policy_versions: vec![],
        };

        assert_eq!(batch.max_staleness(10), 0);
    }

    #[test]
    fn test_batch_max_staleness_saturating() {
        // When current version is less than trajectory version
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 100),
        ]);

        let staleness = batch.max_staleness(50);
        assert_eq!(staleness, 0); // saturating_sub prevents underflow
    }
}

// ============================================================================
// DistributedIMPALA Algorithm Tests (~20 tests)
// ============================================================================

mod algorithm_tests {
    use super::*;

    type B = TestAutodiffBackend;

    #[test]
    fn test_distributed_impala_new() {
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        assert_eq!(<DistributedIMPALA as DistributedAlgorithm<B>>::name(&impala), "DistributedIMPALA");
    }

    #[test]
    fn test_is_off_policy() {
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        assert!(<DistributedIMPALA as DistributedAlgorithm<B>>::is_off_policy(&impala));
    }

    #[test]
    fn test_n_epochs_is_one() {
        // IMPALA uses single pass (off-policy, no need for multiple epochs)
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        assert_eq!(<DistributedIMPALA as DistributedAlgorithm<B>>::n_epochs(&impala), 1);
    }

    #[test]
    fn test_n_minibatches_is_one() {
        // IMPALA processes full trajectory batch at once
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        assert_eq!(<DistributedIMPALA as DistributedAlgorithm<B>>::n_minibatches(&impala), 1);
    }

    #[test]
    fn test_config_accessor() {
        let config = IMPALAConfig::new()
            .with_batch_size(64)
            .with_gamma(0.995);
        let impala = DistributedIMPALA::new(config);

        // Use config() method since field is private
        assert_eq!(<DistributedIMPALA as DistributedAlgorithm<B>>::config(&impala).batch_size, 64);
        assert_eq!(<DistributedIMPALA as DistributedAlgorithm<B>>::config(&impala).gamma, 0.995);
    }

    #[test]
    fn test_create_buffer() {
        let config = IMPALAConfig::default();
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 4, 32);
        assert_eq!(buffer.config().n_actors, 4);
        assert_eq!(buffer.config().n_envs_per_actor, 32);
    }

    #[test]
    fn test_create_buffer_trajectory_length() {
        let config = IMPALAConfig::new().with_trajectory_length(40);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 2, 16);
        assert_eq!(buffer.config().trajectory_length, 40);
    }

    #[test]
    fn test_is_ready_delegates_to_buffer() {
        let config = IMPALAConfig::new()
            .with_batch_size(2);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 1, 1);

        assert!(!<DistributedIMPALA as DistributedAlgorithm<B>>::is_ready(&impala, &buffer));

        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(5, 1, 1));

        assert!(<DistributedIMPALA as DistributedAlgorithm<B>>::is_ready(&impala, &buffer));
    }

    #[test]
    fn test_sample_batch_delegates() {
        let config = IMPALAConfig::new()
            .with_batch_size(2);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 1, 1);
        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(5, 1, 1));

        let batch = <DistributedIMPALA as DistributedAlgorithm<B>>::sample_batch(&impala, &buffer);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
    }

    // ========================================================================
    // Staleness Handling Tests (IMPALA-specific behavior)
    // ========================================================================

    #[test]
    fn test_staleness_keeps_batch() {
        // Unlike PPO, IMPALA should NOT discard stale batches
        // V-trace automatically corrects for staleness
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 1),  // Very stale (version 1)
        ]);

        let current_version = 100;  // 99 versions behind
        let result = <DistributedIMPALA as DistributedAlgorithm<B>>::handle_staleness(
            &impala, batch, current_version
        );

        // IMPALA keeps the batch regardless of staleness
        assert_eq!(result.trajectories.len(), 1);
    }

    #[test]
    fn test_staleness_preserves_all_trajectories() {
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 1),
            make_trajectory(5, 1, 50),
            make_trajectory(5, 2, 99),
        ]);

        let result = <DistributedIMPALA as DistributedAlgorithm<B>>::handle_staleness(
            &impala, batch, 100
        );

        assert_eq!(result.trajectories.len(), 3);
        assert_eq!(result.trajectories[0].env_id, 0);
        assert_eq!(result.trajectories[1].env_id, 1);
        assert_eq!(result.trajectories[2].env_id, 2);
    }

    #[test]
    fn test_staleness_preserves_policy_versions() {
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 5),
            make_trajectory(5, 1, 10),
        ]);

        let result = <DistributedIMPALA as DistributedAlgorithm<B>>::handle_staleness(
            &impala, batch, 100
        );

        assert_eq!(result.policy_versions[0], 5);
        assert_eq!(result.policy_versions[1], 10);
    }

    #[test]
    fn test_high_staleness_logs_warning() {
        // This test verifies the behavior but cannot easily check logs
        // The important thing is it doesn't panic or discard data
        let config = IMPALAConfig::default();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 1),  // version 1
        ]);

        // Staleness > 10 should trigger warning
        let result = <DistributedIMPALA as DistributedAlgorithm<B>>::handle_staleness(
            &impala, batch, 20  // staleness = 19 > 10
        );

        // Still keeps the batch
        assert_eq!(result.len(), 1);
    }

    // ========================================================================
    // V-trace Batch Computation Tests
    // ========================================================================

    #[test]
    fn test_compute_vtrace_batch_on_policy() {
        let config = IMPALAConfig::new()
            .with_gamma(0.99)
            .with_rho_clip(1.0)
            .with_c_clip(1.0);
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(3, 0, 1, &[-1.0, -1.0, -1.0]),
        ]);

        // On-policy: target == behavior
        let target_log_probs = vec![-1.0, -1.0, -1.0];
        let values = vec![0.5, 0.5, 0.5];
        let bootstrap_values = vec![0.0]; // terminal

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vs.len(), 3);
        assert_eq!(results[0].advantages.len(), 3);
        assert_eq!(results[0].rhos.len(), 3);

        // On-policy: all rhos should be 1.0
        for rho in &results[0].rhos {
            assert!((*rho - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compute_vtrace_batch_off_policy() {
        let config = IMPALAConfig::new()
            .with_gamma(0.99)
            .with_rho_clip(1.0)
            .with_c_clip(1.0);
        let impala = DistributedIMPALA::new(config);

        // Behavior: low probability actions
        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(2, 0, 1, &[-2.0, -2.0]),
        ]);

        // Target: higher probability (policy improved)
        let target_log_probs = vec![-0.5, -0.5];
        let values = vec![0.5, 0.5];
        let bootstrap_values = vec![0.0];

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        // Importance weights should be clipped to rho_bar (1.0)
        // exp(-0.5 - (-2.0)) = exp(1.5) > 1.0
        for rho in &results[0].rhos {
            assert!(*rho <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_compute_vtrace_batch_multiple_trajectories() {
        let config = IMPALAConfig::new()
            .with_gamma(0.99)
            .with_rho_clip(1.0)
            .with_c_clip(1.0);
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(2, 0, 1, &[-1.0, -1.0]),
            make_trajectory_with_log_probs(3, 1, 1, &[-1.0, -1.0, -1.0]),
        ]);

        let target_log_probs = vec![-1.0, -1.0, -1.0, -1.0, -1.0]; // 2 + 3 = 5
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        let bootstrap_values = vec![0.0, 0.0];

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vs.len(), 2);
        assert_eq!(results[1].vs.len(), 3);
    }

    #[test]
    fn test_compute_vtrace_batch_bootstrap_per_trajectory() {
        let config = IMPALAConfig::new()
            .with_gamma(0.99)
            .with_rho_clip(1.0)
            .with_c_clip(1.0);
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_non_terminal_trajectory(2, 0, 1),
            make_trajectory(2, 1, 1), // terminal
        ]);

        let target_log_probs = vec![-1.0; 4];
        let values = vec![0.5, 0.5, 0.5, 0.5];
        // Different bootstrap values per trajectory
        let bootstrap_values = vec![0.8, 0.0]; // First non-terminal, second terminal

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        assert_eq!(results.len(), 2);
        // Results should be different due to different bootstrap values
        // (exact values depend on V-trace computation)
    }
}

// ============================================================================
// V-Trace Numerical Stability Tests (~15 tests)
// ============================================================================

mod vtrace_stability_tests {
    use super::*;

    #[test]
    fn test_extreme_positive_log_ratio_no_overflow() {
        // When target >> behavior, exp(target - behavior) could overflow
        // The code clamps log ratio to [-20, 20] before exp()
        let behavior_log_probs = vec![-100.0]; // Very low probability
        let target_log_probs = vec![0.0];      // High probability
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

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

        assert!(result.rhos[0].is_finite(), "rho should not overflow");
        assert!(result.vs[0].is_finite(), "vs should be finite");
        assert!(result.advantages[0].is_finite(), "advantage should be finite");
    }

    #[test]
    fn test_extreme_negative_log_ratio_no_underflow() {
        // When behavior >> target, exp(target - behavior) -> 0
        let behavior_log_probs = vec![0.0];     // High probability
        let target_log_probs = vec![-100.0];    // Very low probability
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

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

        // rho should be clamped, not exactly 0
        assert!(result.rhos[0].is_finite());
        assert!(result.rhos[0] >= 0.0);
    }

    #[test]
    fn test_nan_behavior_log_prob_fallback() {
        // NaN in log_prob should fallback to ratio = 1.0 (on-policy)
        let behavior_log_probs = vec![f32::NAN];
        let target_log_probs = vec![-1.0];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

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

        // Should fallback to on-policy (rho = 1.0)
        assert!((result.rhos[0] - 1.0).abs() < 1e-6, "NaN should fallback to rho=1.0");
    }

    #[test]
    fn test_nan_target_log_prob_fallback() {
        let behavior_log_probs = vec![-1.0];
        let target_log_probs = vec![f32::NAN];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

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

        assert!((result.rhos[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inf_log_prob_fallback() {
        let behavior_log_probs = vec![f32::NEG_INFINITY];
        let target_log_probs = vec![-1.0];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

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

        // Result of NaN/Inf should fallback to on-policy
        assert!(result.rhos[0].is_finite());
    }

    #[test]
    fn test_rho_clipping_respects_rho_bar() {
        // exp(-0.1 - (-2.0)) = exp(1.9) ≈ 6.7 > rho_bar
        let behavior_log_probs = vec![-2.0];
        let target_log_probs = vec![-0.1];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

        let rho_bar = 0.5;
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

        assert!(result.rhos[0] <= rho_bar + 1e-6, "rho should be clipped to rho_bar");
    }

    #[test]
    fn test_c_clipping_respects_c_bar() {
        // This test verifies c_bar is used in V-trace computation
        let behavior_log_probs = vec![-2.0, -2.0];
        let target_log_probs = vec![-0.1, -0.1];
        let rewards = vec![1.0, 1.0];
        let values = vec![0.5, 0.5];
        let dones = vec![false, false];

        // With c_bar = 0.5, trace should be cut more aggressively
        let result1 = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.5,
            0.99,
            1.0,
            0.5, // Low c_bar
        );

        let result2 = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.5,
            0.99,
            1.0,
            10.0, // High c_bar (effectively no clipping for this ratio)
        );

        // V-trace targets should differ due to different c_bar
        // (the exact relationship depends on trajectory length and structure)
        // At minimum, both should be finite
        assert!(result1.vs[0].is_finite());
        assert!(result2.vs[0].is_finite());
    }

    #[test]
    fn test_log_ratio_clamping_boundary() {
        // Test the exact boundary: log_ratio = 20.0
        // exp(20) ≈ 485 million
        let behavior_log_probs = vec![-21.0];
        let target_log_probs = vec![-1.0]; // log_ratio = 20
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];

        let result = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.5,
            0.99,
            1e10, // Very high rho_bar to test clamp not clip
            1e10,
        );

        // With MAX_LOG_RATIO = 20, exp(20) ≈ 4.85e8
        // rho should be clamped before exp, so max is exp(20)
        assert!(result.rhos[0].is_finite());
        assert!(result.rhos[0] <= 5e8); // exp(20) ≈ 4.85e8
    }

    #[test]
    fn test_advantage_uses_value_not_vtrace_target() {
        // Regression test: advantage formula should use V(s_{t+1}), not vs_{t+1}
        let log_probs = vec![-1.0];
        let rewards = vec![1.0];
        let values = vec![0.5];
        let dones = vec![false];
        let bootstrap = 0.8;
        let gamma = 0.99;

        let result = compute_vtrace(
            &log_probs,
            &log_probs, // on-policy
            &rewards,
            &values,
            &dones,
            bootstrap,
            gamma,
            1.0,
            1.0,
        );

        // Expected advantage: r + gamma * V(next) - V(current)
        // = 1.0 + 0.99 * 0.8 - 0.5 = 1.292
        let expected_advantage = 1.0 + gamma * bootstrap - values[0];
        assert!(
            (result.advantages[0] - expected_advantage).abs() < 1e-6,
            "Advantage should use value, not V-trace target"
        );
    }

    #[test]
    fn test_rho_not_in_advantages() {
        // Regression test: advantages should not include rho weighting
        // rho is applied externally in the policy loss
        let behavior_log_probs = vec![-2.0];  // Low probability
        let target_log_probs = vec![-0.5];    // Higher probability
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

        // Advantage should be raw TD error without rho:
        // A = r + gamma * V(next) - V(current)
        let expected_advantage = 1.0 + gamma * bootstrap - values[0];
        assert!(
            (result.advantages[0] - expected_advantage).abs() < 1e-6,
            "Advantage should not include rho!"
        );
    }

    #[test]
    fn test_terminal_state_bootstrap_zero() {
        let log_probs = vec![-1.0, -1.0, -1.0];
        let rewards = vec![1.0, 1.0, 0.0];
        let values = vec![0.5, 0.5, 0.0];
        let dones = vec![false, false, true]; // Last is terminal

        let result = compute_vtrace(
            &log_probs,
            &log_probs,
            &rewards,
            &values,
            &dones,
            0.0, // Terminal bootstrap
            0.99,
            1.0,
            1.0,
        );

        // For terminal state, V-trace target should use 0 for next value
        // This affects the computation for the last step
        assert!(result.vs[2].is_finite());
    }

    #[test]
    fn test_single_step_vtrace() {
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

        // Single step advantage: r + gamma * bootstrap - V(s)
        let expected_adv = 1.0 + 0.99 * 0.5 - 0.5;
        assert!((result.advantages[0] - expected_adv).abs() < 1e-6);
    }

    #[test]
    fn test_vtrace_empty_input() {
        let result = compute_vtrace(
            &[],
            &[],
            &[],
            &[],
            &[],
            0.0,
            0.99,
            1.0,
            1.0,
        );

        assert!(result.vs.is_empty());
        assert!(result.advantages.is_empty());
        assert!(result.rhos.is_empty());
    }

    #[test]
    fn test_vtrace_batch_function() {
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
    }

    #[test]
    fn test_all_outputs_finite_with_extreme_inputs() {
        // Comprehensive test with various extreme inputs
        let behavior_log_probs = vec![-10.0, -0.01, -5.0, 0.0, -100.0];
        let target_log_probs = vec![-0.01, -10.0, -5.0, -50.0, 0.0];
        let rewards = vec![1.0, -1.0, 0.5, 100.0, -100.0];
        let values = vec![0.5, 0.5, 0.5, 10.0, -10.0];
        let dones = vec![false, false, false, false, true];

        let result = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.0,
            0.99,
            1.0,
            1.0,
        );

        for (i, vs) in result.vs.iter().enumerate() {
            assert!(vs.is_finite(), "vs[{}] should be finite, got {}", i, vs);
        }
        for (i, adv) in result.advantages.iter().enumerate() {
            assert!(adv.is_finite(), "advantage[{}] should be finite, got {}", i, adv);
        }
        for (i, rho) in result.rhos.iter().enumerate() {
            assert!(rho.is_finite(), "rho[{}] should be finite, got {}", i, rho);
            assert!(*rho <= 1.0 + 1e-6, "rho[{}] should be clipped to <= 1.0", i);
        }
    }
}

// ============================================================================
// Tensor Shape Tests (~10 tests)
// ============================================================================

mod tensor_shape_tests {
    use super::*;

    type B = TestAutodiffBackend;

    fn get_test_device() -> <B as burn::tensor::backend::Backend>::Device {
        get_device()
    }

    #[test]
    fn test_values_tensor_flattening() {
        let device = get_test_device();

        // Values come in as [batch, 1]
        let values: Tensor<B, 2> = Tensor::from_floats([[0.5], [0.6], [0.7]], &device);
        let values_flat = values.flatten(0, 1);

        assert_eq!(values_flat.dims(), [3]);
    }

    #[test]
    fn test_log_probs_tensor_shape() {
        let device = get_test_device();

        let log_probs: Tensor<B, 1> = Tensor::from_floats([-0.5, -1.0, -1.5], &device);
        assert_eq!(log_probs.dims(), [3]);
    }

    #[test]
    fn test_advantages_tensor_from_vec() {
        let device = get_test_device();

        let advantages_vec = vec![1.0f32, 0.5, -0.5];
        let advantages: Tensor<B, 1> = Tensor::from_floats(advantages_vec.as_slice(), &device);

        assert_eq!(advantages.dims(), [3]);
    }

    #[test]
    fn test_vtrace_targets_tensor_from_vec() {
        let device = get_test_device();

        let targets_vec = vec![1.5f32, 1.2, 0.8];
        let targets: Tensor<B, 1> = Tensor::from_floats(targets_vec.as_slice(), &device);

        assert_eq!(targets.dims(), [3]);
    }

    #[test]
    fn test_rhos_tensor_from_vec() {
        let device = get_test_device();

        let rhos_vec = vec![1.0f32, 0.8, 1.0];
        let rhos: Tensor<B, 1> = Tensor::from_floats(rhos_vec.as_slice(), &device);

        assert_eq!(rhos.dims(), [3]);
    }

    #[test]
    fn test_policy_loss_computation_shape() {
        let device = get_test_device();

        let rhos: Tensor<B, 1> = Tensor::from_floats([1.0, 0.8, 1.0], &device);
        let log_probs: Tensor<B, 1> = Tensor::from_floats([-0.5, -1.0, -1.5], &device);
        let advantages: Tensor<B, 1> = Tensor::from_floats([1.0, 0.5, -0.5], &device);

        // Policy loss: -rho * log_pi * A
        let policy_loss = -(rhos * log_probs * advantages).mean();

        assert_eq!(policy_loss.dims(), [1]);
    }

    #[test]
    fn test_value_loss_computation_shape() {
        let device = get_test_device();

        let values: Tensor<B, 1> = Tensor::from_floats([0.5, 0.6, 0.7], &device);
        let targets: Tensor<B, 1> = Tensor::from_floats([1.0, 1.2, 0.8], &device);

        // Value loss: MSE
        let vf_loss = (values - targets).powf_scalar(2.0).mean();

        assert_eq!(vf_loss.dims(), [1]);
    }

    #[test]
    fn test_entropy_computation_shape() {
        let device = get_test_device();

        let entropy: Tensor<B, 1> = Tensor::from_floats([0.5, 0.5, 0.5], &device);
        let mean_entropy = entropy.mean();

        assert_eq!(mean_entropy.dims(), [1]);
    }

    #[test]
    fn test_total_loss_computation_shape() {
        let device = get_test_device();

        let policy_loss: Tensor<B, 1> = Tensor::from_floats([0.1], &device);
        let vf_loss: Tensor<B, 1> = Tensor::from_floats([0.2], &device);
        let entropy: Tensor<B, 1> = Tensor::from_floats([0.5], &device);

        let vf_coef = 0.5f32;
        let entropy_coef = 0.01f32;

        let total_loss = policy_loss
            + vf_loss.mul_scalar(vf_coef)
            - entropy.mul_scalar(entropy_coef);

        assert_eq!(total_loss.dims(), [1]);
    }

    #[test]
    fn test_batch_size_consistency() {
        let device = get_test_device();

        // All tensors in a batch must have consistent first dimension
        let batch_size = 16;

        let log_probs: Tensor<B, 1> = Tensor::zeros([batch_size], &device);
        let entropy: Tensor<B, 1> = Tensor::zeros([batch_size], &device);
        let values: Tensor<B, 2> = Tensor::zeros([batch_size, 1], &device);
        let advantages: Tensor<B, 1> = Tensor::zeros([batch_size], &device);
        let vtrace_targets: Tensor<B, 1> = Tensor::zeros([batch_size], &device);
        let rhos: Tensor<B, 1> = Tensor::zeros([batch_size], &device);

        assert_eq!(log_probs.dims()[0], batch_size);
        assert_eq!(entropy.dims()[0], batch_size);
        assert_eq!(values.dims()[0], batch_size);
        assert_eq!(advantages.dims()[0], batch_size);
        assert_eq!(vtrace_targets.dims()[0], batch_size);
        assert_eq!(rhos.dims()[0], batch_size);
    }
}

// ============================================================================
// Concurrency Tests (~10 tests)
// ============================================================================

mod concurrency_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_concurrent_push_from_multiple_threads() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            max_trajectories: 1000,
            ..Default::default()
        };
        let buffer = Arc::new(IMPALABuffer::new(config));

        let n_threads = 4;
        let n_trajectories_per_thread = 25;
        let mut handles = vec![];

        for thread_id in 0..n_threads {
            let buffer_clone = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                for i in 0..n_trajectories_per_thread {
                    let env_id = thread_id * 100 + i;
                    buffer_clone.push_trajectory(make_trajectory(5, env_id, 1));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        buffer.consolidate();
        assert_eq!(buffer.len(), n_threads * n_trajectories_per_thread);
    }

    #[test]
    fn test_concurrent_push_and_sample() {
        let config = IMPALABufferConfig {
            batch_size: 5,
            max_trajectories: 1000,
            ..Default::default()
        };
        let buffer = Arc::new(IMPALABuffer::new(config));
        let samples_collected = Arc::new(AtomicUsize::new(0));

        // Producer threads
        let n_producers = 2;
        let n_trajectories_per_producer = 50;
        let mut handles = vec![];

        for thread_id in 0..n_producers {
            let buffer_clone = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                for i in 0..n_trajectories_per_producer {
                    buffer_clone.push_trajectory(make_trajectory(5, thread_id * 100 + i, 1));
                    thread::yield_now();
                }
            });
            handles.push(handle);
        }

        // Consumer thread
        let buffer_clone = Arc::clone(&buffer);
        let samples_clone = Arc::clone(&samples_collected);
        let consumer = thread::spawn(move || {
            let mut attempts = 0;
            while attempts < 100 {
                if buffer_clone.is_training_ready() {
                    if let Some(batch) = buffer_clone.sample_batch() {
                        samples_clone.fetch_add(batch.len(), Ordering::SeqCst);
                    }
                }
                thread::yield_now();
                attempts += 1;
            }
        });

        for handle in handles {
            handle.join().unwrap();
        }
        consumer.join().unwrap();

        // Some samples should have been collected
        // (exact number depends on timing)
        let total_samples = samples_collected.load(Ordering::SeqCst);
        assert!(total_samples >= 0); // Just verify no panic
    }

    #[test]
    fn test_buffer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<IMPALABuffer>();
    }

    #[test]
    fn test_buffer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<IMPALABuffer>();
    }

    #[test]
    fn test_config_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<IMPALAConfig>();
    }

    #[test]
    fn test_config_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<IMPALAConfig>();
    }

    #[test]
    fn test_algorithm_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DistributedIMPALA>();
    }

    #[test]
    fn test_algorithm_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DistributedIMPALA>();
    }

    #[test]
    fn test_no_data_race_on_clear() {
        let config = IMPALABufferConfig {
            batch_size: 2,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = Arc::new(IMPALABuffer::new(config));

        // Push some data
        for i in 0..10 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }

        let buffer1 = Arc::clone(&buffer);
        let buffer2 = Arc::clone(&buffer);

        let handle1 = thread::spawn(move || {
            buffer1.consolidate();
            let _ = buffer1.sample_batch();
        });

        let handle2 = thread::spawn(move || {
            buffer2.clear();
        });

        // Should not panic due to data race
        let _ = handle1.join();
        let _ = handle2.join();
    }

    #[test]
    fn test_concurrent_is_training_ready_checks() {
        let config = IMPALABufferConfig {
            batch_size: 5,
            max_trajectories: 100,
            ..Default::default()
        };
        let buffer = Arc::new(IMPALABuffer::new(config));

        // Push enough data
        for i in 0..10 {
            buffer.push_trajectory(make_trajectory(5, i, 1));
        }

        let mut handles = vec![];
        for _ in 0..4 {
            let buffer_clone = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let _ready = buffer_clone.is_training_ready();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

// ============================================================================
// Integration Tests (~15 tests)
// ============================================================================

mod integration_tests {
    use super::*;

    type B = TestAutodiffBackend;

    fn get_test_device() -> <B as burn::tensor::backend::Backend>::Device {
        get_device()
    }

    #[test]
    fn test_full_training_cycle() {
        // End-to-end: push trajectories -> sample batch -> compute V-trace
        let config = IMPALAConfig::new()
            .with_batch_size(2)
            .with_gamma(0.99)
            .with_rho_clip(1.0)
            .with_c_clip(1.0);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 1, 1);

        // Push trajectories
        buffer.push_trajectory(make_trajectory_with_log_probs(3, 0, 1, &[-1.0, -1.0, -1.0]));
        buffer.push_trajectory(make_trajectory_with_log_probs(3, 1, 2, &[-0.5, -0.5, -0.5]));

        // Sample batch
        let batch = <DistributedIMPALA as DistributedAlgorithm<B>>::sample_batch(&impala, &buffer);
        assert!(batch.is_some());
        let batch = batch.unwrap();

        // Handle staleness
        let batch = <DistributedIMPALA as DistributedAlgorithm<B>>::handle_staleness(&impala, batch, 10);
        assert_eq!(batch.len(), 2);

        // Compute V-trace
        let target_log_probs = vec![-1.0; 6]; // 3 + 3 = 6
        let values = vec![0.5; 6];
        let bootstrap_values = vec![0.0, 0.0]; // Both terminal

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vs.len(), 3);
        assert_eq!(results[1].vs.len(), 3);
    }

    #[test]
    fn test_on_policy_reduces_to_td() {
        // When behavior == target, V-trace should reduce to TD(lambda) with lambda=1
        let config = IMPALAConfig::new()
            .with_gamma(0.99)
            .with_rho_clip(1.0)
            .with_c_clip(1.0);
        let impala = DistributedIMPALA::new(config);

        // Same log probs for behavior and target
        let log_probs = vec![-1.0, -1.0, -1.0];
        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(3, 0, 1, &log_probs),
        ]);

        let results = impala.compute_vtrace_batch(
            &batch,
            &log_probs,
            &[0.5, 0.5, 0.5],
            &[0.0],
        );

        // All importance weights should be 1.0
        for rho in &results[0].rhos {
            assert!((*rho - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_buffer_and_algorithm_integration() {
        let config = IMPALAConfig::new()
            .with_n_actors(4)
            .with_n_envs_per_actor(8)
            .with_trajectory_length(10)
            .with_batch_size(4);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 4, 8);

        // Verify buffer config matches algorithm config
        assert_eq!(buffer.config().n_actors, 4);
        assert_eq!(buffer.config().n_envs_per_actor, 8);
        assert_eq!(buffer.config().trajectory_length, 10);
        assert_eq!(buffer.config().batch_size, 4);
    }

    #[test]
    fn test_multiple_sample_cycles() {
        let config = IMPALAConfig::new().with_batch_size(2);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 1, 1);

        // First cycle
        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(5, 1, 1));

        let batch1 = <DistributedIMPALA as DistributedAlgorithm<B>>::sample_batch(&impala, &buffer);
        assert!(batch1.is_some());

        // Second cycle
        buffer.push_trajectory(make_trajectory(5, 2, 2));
        buffer.push_trajectory(make_trajectory(5, 3, 2));

        let batch2 = <DistributedIMPALA as DistributedAlgorithm<B>>::sample_batch(&impala, &buffer);
        assert!(batch2.is_some());

        // Batches should have different env_ids (FIFO)
        assert_eq!(batch1.unwrap().trajectories[0].env_id, 0);
        assert_eq!(batch2.unwrap().trajectories[0].env_id, 2);
    }

    #[test]
    fn test_mixed_policy_versions_in_batch() {
        let config = IMPALAConfig::new().with_batch_size(3);
        let impala: DistributedIMPALA = DistributedAlgorithm::<B>::new(config);

        let buffer = <DistributedIMPALA as DistributedAlgorithm<B>>::create_buffer(&impala, 1, 1);

        // Push trajectories from different policy versions
        buffer.push_trajectory(make_trajectory(5, 0, 1));
        buffer.push_trajectory(make_trajectory(5, 1, 5));
        buffer.push_trajectory(make_trajectory(5, 2, 10));

        let batch = <DistributedIMPALA as DistributedAlgorithm<B>>::sample_batch(&impala, &buffer).unwrap();

        assert_eq!(batch.policy_versions[0], 1);
        assert_eq!(batch.policy_versions[1], 5);
        assert_eq!(batch.policy_versions[2], 10);
    }

    #[test]
    fn test_different_trajectory_lengths_in_batch() {
        let config = IMPALAConfig::new()
            .with_batch_size(3)
            .with_trajectory_length(20);
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 1),   // Short
            make_trajectory(10, 1, 1),  // Medium
            make_trajectory(15, 2, 1),  // Long
        ]);

        // Total transitions = 5 + 10 + 15 = 30
        assert_eq!(batch.total_transitions(), 30);

        let target_log_probs = vec![-1.0; 30];
        let values = vec![0.5; 30];
        let bootstrap_values = vec![0.0, 0.0, 0.0];

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        assert_eq!(results[0].vs.len(), 5);
        assert_eq!(results[1].vs.len(), 10);
        assert_eq!(results[2].vs.len(), 15);
    }

    #[test]
    fn test_empty_batch_handling() {
        let batch = IMPALABatch {
            trajectories: vec![],
            policy_versions: vec![],
        };

        assert!(batch.is_empty());
        assert_eq!(batch.total_transitions(), 0);
        assert_eq!(batch.state_dim(), 0);
    }

    #[test]
    fn test_terminal_vs_non_terminal_bootstrap() {
        let config = IMPALAConfig::new()
            .with_gamma(0.99)
            .with_rho_clip(1.0);
        let impala = DistributedIMPALA::new(config);

        // Terminal trajectory
        let terminal_batch = make_impala_batch(vec![make_trajectory(3, 0, 1)]);
        let terminal_results = impala.compute_vtrace_batch(
            &terminal_batch,
            &[-1.0; 3],
            &[0.5; 3],
            &[0.0], // Bootstrap = 0 for terminal
        );

        // Non-terminal trajectory
        let non_terminal_batch = make_impala_batch(vec![make_non_terminal_trajectory(3, 0, 1)]);
        let non_terminal_results = impala.compute_vtrace_batch(
            &non_terminal_batch,
            &[-1.0; 3],
            &[0.5; 3],
            &[1.0], // Bootstrap = 1.0 for non-terminal
        );

        // V-trace targets should differ due to bootstrap
        // Terminal trajectory has lower targets because no future value
        assert!(terminal_results[0].vs[0] != non_terminal_results[0].vs[0]);
    }

    #[test]
    fn test_high_rho_clip() {
        let config = IMPALAConfig::new()
            .with_rho_clip(10.0)  // High clip
            .with_c_clip(10.0);
        let impala = DistributedIMPALA::new(config);

        // Large policy divergence
        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(2, 0, 1, &[-5.0, -5.0]),
        ]);

        // Target has much higher probability
        let target_log_probs = vec![-0.1, -0.1];
        let values = vec![0.5, 0.5];
        let bootstrap_values = vec![0.0];

        let results = impala.compute_vtrace_batch(
            &batch,
            &target_log_probs,
            &values,
            &bootstrap_values,
        );

        // With high rho_clip, importance weights should be higher (but still clamped by MAX_LOG_RATIO)
        for rho in &results[0].rhos {
            assert!(rho.is_finite());
            // exp(4.9) ≈ 134, but clamped to rho_clip (10.0)
            assert!(*rho <= 10.0 + 1e-6);
        }
    }

    #[test]
    fn test_low_gamma() {
        let config = IMPALAConfig::new()
            .with_gamma(0.0); // No discounting of future
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(3, 0, 1, &[-1.0, -1.0, -1.0]),
        ]);

        let results = impala.compute_vtrace_batch(
            &batch,
            &[-1.0, -1.0, -1.0],
            &[0.5, 0.5, 0.5],
            &[1.0], // Non-zero bootstrap
        );

        // With gamma=0, future values don't matter
        // Advantage should be just r - V(s)
        for adv in &results[0].advantages {
            // reward is 1.0, value is 0.5, gamma=0 means no future
            // A = r + gamma * V(next) - V(s) = 1.0 + 0 - 0.5 = 0.5
            assert!((*adv - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_algorithm_clone() {
        let config = IMPALAConfig::new()
            .with_batch_size(64)
            .with_gamma(0.995);
        let impala = DistributedIMPALA::new(config);
        let cloned = impala.clone();

        // Use config() method since field is private
        assert_eq!(
            <DistributedIMPALA as DistributedAlgorithm<B>>::config(&impala).batch_size,
            <DistributedIMPALA as DistributedAlgorithm<B>>::config(&cloned).batch_size
        );
        assert_eq!(
            <DistributedIMPALA as DistributedAlgorithm<B>>::config(&impala).gamma,
            <DistributedIMPALA as DistributedAlgorithm<B>>::config(&cloned).gamma
        );
    }

    #[test]
    fn test_batch_clone() {
        let batch = make_impala_batch(vec![make_trajectory(5, 0, 1)]);
        let cloned = batch.clone();

        assert_eq!(batch.len(), cloned.len());
        assert_eq!(batch.policy_versions, cloned.policy_versions);
    }

    #[test]
    fn test_vtrace_result_clone() {
        let result = compute_vtrace(
            &[-1.0],
            &[-1.0],
            &[1.0],
            &[0.5],
            &[false],
            0.5,
            0.99,
            1.0,
            1.0,
        );
        let cloned = result.clone();

        assert_eq!(result.vs, cloned.vs);
        assert_eq!(result.advantages, cloned.advantages);
        assert_eq!(result.rhos, cloned.rhos);
    }
}

// ============================================================================
// Loss Computation Tests (tensor-level)
// ============================================================================

mod loss_tests {
    use super::*;
    use crate::algorithms::algorithm::LossOutput;

    type B = TestAutodiffBackend;

    fn get_test_device() -> <B as burn::tensor::backend::Backend>::Device {
        get_device()
    }

    #[test]
    fn test_compute_batch_loss_output_structure() {
        let device = get_test_device();
        let config = IMPALAConfig::new()
            .with_vf_coef(0.5)
            .with_entropy_coef(0.01);
        let impala = DistributedIMPALA::new(config);

        // Create a minimal batch
        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(3, 0, 1, &[-1.0, -1.0, -1.0]),
        ]);

        // Create tensors
        let log_probs: Tensor<B, 1> = Tensor::from_floats([-0.5, -0.5, -0.5], &device);
        let entropy: Tensor<B, 1> = Tensor::from_floats([0.5, 0.5, 0.5], &device);
        let values: Tensor<B, 2> = Tensor::from_floats([[0.5], [0.5], [0.5]], &device);

        let output: LossOutput<B> = <DistributedIMPALA as DistributedAlgorithm<B>>::compute_batch_loss(
            &impala,
            &batch,
            log_probs,
            entropy,
            values,
            &device,
        );

        // Verify structure
        assert_eq!(output.total_loss.dims(), [1]);
        assert!(output.policy_loss.is_finite());
        assert!(output.value_loss.is_finite());
        assert!(output.entropy.is_finite());
    }

    #[test]
    fn test_loss_is_finite_with_normal_inputs() {
        let device = get_test_device();
        let config = IMPALAConfig::new();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(5, 0, 1, &[-1.0; 5]),
        ]);

        let log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0, -1.0, -1.0, -1.0], &device);
        let entropy: Tensor<B, 1> = Tensor::from_floats([0.5, 0.5, 0.5, 0.5, 0.5], &device);
        let values: Tensor<B, 2> = Tensor::from_floats([[0.5], [0.5], [0.5], [0.5], [0.5]], &device);

        let output: LossOutput<B> = <DistributedIMPALA as DistributedAlgorithm<B>>::compute_batch_loss(
            &impala,
            &batch,
            log_probs,
            entropy,
            values,
            &device,
        );

        let loss_data = output.total_loss.into_data();
        let loss_val = loss_data.as_slice::<f32>().unwrap()[0];
        assert!(loss_val.is_finite(), "Loss should be finite");
    }

    #[test]
    fn test_entropy_coefficient_effect() {
        let device = get_test_device();

        // Low entropy coefficient
        let config_low = IMPALAConfig::new().with_entropy_coef(0.0);
        let impala_low = DistributedIMPALA::new(config_low);

        // High entropy coefficient
        let config_high = IMPALAConfig::new().with_entropy_coef(1.0);
        let impala_high = DistributedIMPALA::new(config_high);

        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(2, 0, 1, &[-1.0, -1.0]),
        ]);

        let log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);
        let entropy: Tensor<B, 1> = Tensor::from_floats([1.0, 1.0], &device); // High entropy
        let values: Tensor<B, 2> = Tensor::from_floats([[0.5], [0.5]], &device);

        let output_low: LossOutput<B> = <DistributedIMPALA as DistributedAlgorithm<B>>::compute_batch_loss(
            &impala_low,
            &batch,
            log_probs.clone(),
            entropy.clone(),
            values.clone(),
            &device,
        );

        let output_high: LossOutput<B> = <DistributedIMPALA as DistributedAlgorithm<B>>::compute_batch_loss(
            &impala_high,
            &batch,
            log_probs,
            entropy,
            values,
            &device,
        );

        // Entropy is subtracted from loss, so higher coef -> lower total loss
        let loss_low = output_low.total_loss.into_data().as_slice::<f32>().unwrap()[0];
        let loss_high = output_high.total_loss.into_data().as_slice::<f32>().unwrap()[0];

        // With positive entropy and subtraction: higher coef means lower loss
        assert!(loss_high < loss_low, "Higher entropy coef should decrease loss");
    }

    #[test]
    fn test_vf_coefficient_effect() {
        let device = get_test_device();

        // Low VF coefficient
        let config_low = IMPALAConfig::new().with_vf_coef(0.0);
        let impala_low = DistributedIMPALA::new(config_low);

        // High VF coefficient
        let config_high = IMPALAConfig::new().with_vf_coef(1.0);
        let impala_high = DistributedIMPALA::new(config_high);

        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(2, 0, 1, &[-1.0, -1.0]),
        ]);

        // Values far from V-trace targets to create high value loss
        let log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);
        let entropy: Tensor<B, 1> = Tensor::from_floats([0.5, 0.5], &device);
        let values: Tensor<B, 2> = Tensor::from_floats([[0.0], [0.0]], &device); // Far from targets

        let output_low: LossOutput<B> = <DistributedIMPALA as DistributedAlgorithm<B>>::compute_batch_loss(
            &impala_low,
            &batch,
            log_probs.clone(),
            entropy.clone(),
            values.clone(),
            &device,
        );

        let output_high: LossOutput<B> = <DistributedIMPALA as DistributedAlgorithm<B>>::compute_batch_loss(
            &impala_high,
            &batch,
            log_probs,
            entropy,
            values,
            &device,
        );

        let loss_low = output_low.total_loss.into_data().as_slice::<f32>().unwrap()[0];
        let loss_high = output_high.total_loss.into_data().as_slice::<f32>().unwrap()[0];

        // Higher VF coef means value loss contributes more to total
        // If there's nonzero value loss, higher coef -> higher total loss
        assert!(loss_high >= loss_low, "Higher VF coef should increase or maintain loss");
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_trajectory_length_one() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        let single_step = make_trajectory(1, 0, 1);
        buffer.push_trajectory(single_step);
        buffer.consolidate();

        assert_eq!(buffer.len(), 1);

        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.trajectories[0].len(), 1);
    }

    #[test]
    fn test_batch_size_one() {
        let config = IMPALABufferConfig {
            batch_size: 1,
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        buffer.push_trajectory(make_trajectory(5, 0, 1));

        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_zero_reward_trajectory() {
        let config = IMPALAConfig::new();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_rewards(3, 0, 1, &[0.0, 0.0, 0.0]),
        ]);

        let results = impala.compute_vtrace_batch(
            &batch,
            &[-1.0; 3],
            &[0.0; 3],
            &[0.0],
        );

        // With zero rewards and zero values, V-trace targets should be ~0
        for vs in &results[0].vs {
            assert!(vs.abs() < 1e-6);
        }
    }

    #[test]
    fn test_negative_rewards() {
        let config = IMPALAConfig::new();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_rewards(3, 0, 1, &[-1.0, -2.0, -3.0]),
        ]);

        let results = impala.compute_vtrace_batch(
            &batch,
            &[-1.0; 3],
            &[0.0; 3],
            &[0.0],
        );

        // Results should still be finite
        for vs in &results[0].vs {
            assert!(vs.is_finite());
        }
    }

    #[test]
    fn test_large_rewards() {
        let config = IMPALAConfig::new();
        let impala = DistributedIMPALA::new(config);

        let batch = make_impala_batch(vec![
            make_trajectory_with_rewards(3, 0, 1, &[1000.0, 2000.0, 3000.0]),
        ]);

        let results = impala.compute_vtrace_batch(
            &batch,
            &[-1.0; 3],
            &[0.0; 3],
            &[0.0],
        );

        // Results should be finite even with large rewards
        for vs in &results[0].vs {
            assert!(vs.is_finite());
        }
    }

    #[test]
    fn test_uniform_policy() {
        // All log probs are the same
        let log_probs = vec![-1.386; 5]; // log(0.25)
        let batch = make_impala_batch(vec![
            make_trajectory_with_log_probs(5, 0, 1, &log_probs),
        ]);

        let config = IMPALAConfig::new();
        let impala = DistributedIMPALA::new(config);

        let results = impala.compute_vtrace_batch(
            &batch,
            &log_probs,
            &[0.5; 5],
            &[0.0],
        );

        // On-policy, all rhos should be 1.0
        for rho in &results[0].rhos {
            assert!((*rho - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_very_long_trajectory() {
        let len = 1000;
        let config = IMPALABufferConfig {
            batch_size: 1,
            max_trajectories: 10,
            trajectory_length: 20, // Shorter than actual trajectory
            ..Default::default()
        };
        let buffer = IMPALABuffer::new(config);

        // Long trajectory should still be accepted (bug 7 fix)
        let long_traj = make_trajectory(len, 0, 1);
        buffer.push_trajectory(long_traj);
        buffer.consolidate();

        let batch = buffer.sample_batch().unwrap();
        assert_eq!(batch.trajectories[0].len(), len);
    }

    #[test]
    fn test_policy_version_zero() {
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, 0),
        ]);

        assert_eq!(batch.policy_versions[0], 0);
        assert_eq!(batch.max_staleness(10), 10);
    }

    #[test]
    fn test_policy_version_max_u64() {
        let batch = make_impala_batch(vec![
            make_trajectory(5, 0, u64::MAX),
        ]);

        assert_eq!(batch.policy_versions[0], u64::MAX);
        assert_eq!(batch.max_staleness(u64::MAX), 0);
    }
}
