//! Integration tests defining complete environment behavior scenarios.
//!
//! These tests verify:
//! - Complete episode rollouts
//! - Realistic training loop patterns
//! - Multi-environment consistency
//! - End-to-end behavior from reset to termination

use crate::adapter::QuadcopterEnvWrapper;
use crate::config::{InitConfig, ObsConfig, QuadcopterConfig, TerminationConfig, TaskMode};
use crate::constants::*;
use crate::env::Quadcopter;
use operant_core::{Environment, ResetMask};

// ============================================================================
// Complete Episode Rollout Tests
// ============================================================================

#[test]
fn should_complete_episode_to_truncation() {
    let max_steps = 50;
    let config = QuadcopterConfig::new(4)
        .with_init(InitConfig::fixed_start())
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(max_steps)
                .without_position_bounds()
                .without_attitude_bounds()
                .with_ground_collision(false),
        )
        .with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Hover action to avoid early termination
    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    let mut steps_until_done = vec![0usize; 4];
    let mut done = vec![false; 4];

    for step in 1..=max_steps as usize {
        let result = env.step_no_reset_with_result(&actions);

        for idx in 0..4 {
            if !done[idx] && (result.truncations[idx] != 0 || result.terminals[idx] != 0) {
                steps_until_done[idx] = step;
                done[idx] = true;
            }
        }

        if done.iter().all(|&d| d) {
            break;
        }
    }

    // All environments should complete at max_steps
    for idx in 0..4 {
        assert!(
            steps_until_done[idx] == max_steps as usize,
            "Env {} should truncate at step {} but got {}",
            idx,
            max_steps,
            steps_until_done[idx]
        );
    }
}

#[test]
fn should_complete_episode_with_ground_collision() {
    let config = QuadcopterConfig::new(4)
        .with_init(InitConfig::fixed_start())
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(1000)
                .with_ground_collision(true),
        )
        .with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Motors off - will fall
    let actions = vec![-1.0; 4 * 4];

    let mut terminal_count = 0;
    for _ in 0..100 {
        let result = env.step_no_reset_with_result(&actions);
        terminal_count += result.terminals.iter().filter(|&&t| t != 0).count();

        if terminal_count >= 4 {
            break;
        }
    }

    assert!(
        terminal_count >= 4,
        "All environments should terminate from ground collision"
    );
}

#[test]
fn should_accumulate_realistic_rewards_over_episode() {
    let config = QuadcopterConfig::new(1)
        .with_init(InitConfig::fixed_start())
        .with_termination(TerminationConfig::default().with_max_steps(100))
        .with_hover_target([0.0, 0.0, 1.0])
        .with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4];

    let mut total_reward = 0.0;
    let mut rewards = vec![0.0; 1];

    for _ in 0..100 {
        env.step_no_reset(&actions);
        env.write_rewards(&mut rewards);
        total_reward += rewards[0];
    }

    // Hover at target should give positive rewards
    assert!(
        total_reward > 0.0,
        "Hovering at target should accumulate positive reward: {}",
        total_reward
    );
}

// ============================================================================
// Training Loop Pattern Tests
// ============================================================================

#[test]
fn should_support_typical_ppo_training_pattern() {
    let num_envs = 8;
    let rollout_length = 32;

    let config = QuadcopterConfig::new(num_envs)
        .with_init(InitConfig::default())
        .with_termination(TerminationConfig::default().with_max_steps(1000))
        .with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let obs_size = env.observation_size();

    // Storage for rollout
    let mut observations = vec![vec![0.0f32; num_envs * obs_size]; rollout_length + 1];
    let mut rewards = vec![vec![0.0f32; num_envs]; rollout_length];
    let mut dones = vec![vec![false; num_envs]; rollout_length];

    // Collect initial observations
    env.write_observations(&mut observations[0]);

    // Simulate rollout collection
    for step in 0..rollout_length {
        // In real training, actions would come from policy
        let hover_action = rpm_to_action(HOVER_RPM);
        let actions = vec![hover_action; num_envs * 4];

        let result = env.step_no_reset_with_result(&actions);

        // Store data
        observations[step + 1].copy_from_slice(result.observations);
        rewards[step].copy_from_slice(result.rewards);
        for idx in 0..num_envs {
            dones[step][idx] = result.terminals[idx] != 0 || result.truncations[idx] != 0;
        }

        // Reset done environments
        let done_indices: Vec<usize> = (0..num_envs)
            .filter(|&idx| dones[step][idx])
            .collect();
        if !done_indices.is_empty() {
            let mut mask = ResetMask::new(num_envs);
            for idx in &done_indices {
                mask.set(*idx);
            }
            env.reset_envs(&mask, step as u64);
        }
    }

    // Verify rollout was collected
    let total_rewards: f32 = rewards.iter().flatten().sum();
    let total_dones: usize = dones.iter().flatten().filter(|&&d| d).count();

    assert!(
        total_rewards.is_finite(),
        "Rollout rewards should be finite"
    );
    // Some environments may have been reset
}

#[test]
fn should_support_typical_dqn_training_pattern() {
    // DQN uses step_no_reset for bootstrap value estimation
    let num_envs = 4;

    let config = QuadcopterConfig::new(num_envs)
        .with_init(InitConfig::default())
        .with_termination(TerminationConfig::default().with_max_steps(100))
        .with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let obs_size = env.observation_size();

    // Simulate DQN training loop
    for episode in 0..5 {
        let mut current_obs = vec![0.0f32; num_envs * obs_size];
        env.write_observations(&mut current_obs);

        for step in 0..50 {
            // Take action
            let actions = vec![rpm_to_action(HOVER_RPM); num_envs * 4];
            let result = env.step_no_reset_with_result(&actions);

            // Copy result data we need before potentially resetting
            let terminals: Vec<u8> = result.terminals.to_vec();
            let truncations: Vec<u8> = result.truncations.to_vec();
            let obs_copy: Vec<f32> = result.observations.to_vec();

            // Check for terminal states to get bootstrap values
            for idx in 0..num_envs {
                let is_terminal = terminals[idx] != 0;
                let is_truncated = truncations[idx] != 0;

                if is_terminal {
                    // Terminal: bootstrap value = 0 (already in result.observations)
                } else if is_truncated {
                    // Truncated: need terminal observation for bootstrap
                    // The observation is the terminal observation before any reset
                    let _terminal_obs = &obs_copy[idx * obs_size..(idx + 1) * obs_size];
                    // In real DQN, would compute V(terminal_obs) for bootstrap
                }
            }

            // Reset done environments
            let done_mask: Vec<usize> = (0..num_envs)
                .filter(|&idx| terminals[idx] != 0 || truncations[idx] != 0)
                .collect();

            if !done_mask.is_empty() {
                let mut mask = ResetMask::new(num_envs);
                for idx in &done_mask {
                    mask.set(*idx);
                }
                env.reset_envs(&mask, (episode * 1000 + step) as u64);
            }

            current_obs.copy_from_slice(&obs_copy);
        }
    }
}

#[test]
fn should_support_auto_reset_pattern_for_policy_gradient() {
    // Policy gradient with auto-reset using step()
    let num_envs = 4;
    let config = QuadcopterConfig::new(num_envs)
        .with_init(InitConfig::fixed_start())
        .with_termination(TerminationConfig::default().with_max_steps(50));
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let obs_size = env.observation_size();

    // Collect trajectories with auto-reset
    let mut episode_rewards = vec![0.0f32; num_envs];
    let mut completed_episodes = 0;

    for _ in 0..200 {
        let actions = vec![rpm_to_action(HOVER_RPM); num_envs * 4];

        // step() auto-resets done environments
        env.step(&actions);

        let mut rewards = vec![0.0f32; num_envs];
        let mut terminals = vec![0u8; num_envs];
        let mut truncations = vec![0u8; num_envs];

        env.write_rewards(&mut rewards);
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        for idx in 0..num_envs {
            episode_rewards[idx] += rewards[idx];

            if terminals[idx] != 0 || truncations[idx] != 0 {
                completed_episodes += 1;
                episode_rewards[idx] = 0.0; // Reset for new episode
            }
        }
    }

    assert!(
        completed_episodes > 0,
        "Should complete some episodes with auto-reset"
    );
}

// ============================================================================
// Multi-Environment Consistency Tests
// ============================================================================

#[test]
fn should_produce_same_trajectory_with_same_seed_and_actions() {
    let config = QuadcopterConfig::new(4)
        .with_init(InitConfig::fixed_start())
        .with_termination(TerminationConfig::default().with_max_steps(100));

    // First run
    let mut env1 = Quadcopter::from_config(config.clone()).unwrap();
    env1.reset(42);

    let obs_size = env1.observation_size();
    let mut obs1 = vec![vec![0.0f32; 4 * obs_size]; 10];
    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    for step in 0..10 {
        env1.step_no_reset(&actions);
        env1.write_observations(&mut obs1[step]);
    }

    // Second run with same seed
    let mut env2 = Quadcopter::from_config(config).unwrap();
    env2.reset(42);

    let mut obs2 = vec![vec![0.0f32; 4 * obs_size]; 10];

    for step in 0..10 {
        env2.step_no_reset(&actions);
        env2.write_observations(&mut obs2[step]);
    }

    // Trajectories should be identical
    for step in 0..10 {
        for i in 0..obs1[step].len() {
            assert!(
                (obs1[step][i] - obs2[step][i]).abs() < 1e-5,
                "Trajectories should match at step {}, index {}: {} vs {}",
                step,
                i,
                obs1[step][i],
                obs2[step][i]
            );
        }
    }
}

#[test]
fn should_produce_different_trajectories_with_different_seeds() {
    let config = QuadcopterConfig::new(4)
        .with_init(InitConfig::default()) // Uses randomization
        .with_termination(TerminationConfig::default().with_max_steps(100));

    let mut env1 = Quadcopter::from_config(config.clone()).unwrap();
    env1.reset(42);

    let mut env2 = Quadcopter::from_config(config).unwrap();
    env2.reset(123);

    let obs_size = env1.observation_size();
    let mut obs1 = vec![0.0f32; 4 * obs_size];
    let mut obs2 = vec![0.0f32; 4 * obs_size];

    env1.write_observations(&mut obs1);
    env2.write_observations(&mut obs2);

    // Initial observations should differ
    let different = obs1.iter().zip(obs2.iter()).any(|(a, b)| (a - b).abs() > 1e-5);
    assert!(
        different,
        "Different seeds should produce different initial states"
    );
}

// ============================================================================
// Wrapper Integration Tests
// ============================================================================

#[test]
fn should_use_wrapper_for_simplified_api() {
    let mut wrapper = QuadcopterEnvWrapper::new(4).unwrap();
    wrapper.reset_all(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    let result = wrapper.step(&actions);

    assert_eq!(result.rewards.len(), 4);
    assert_eq!(result.dones.len(), 4);
    assert_eq!(result.terminals.len(), 4);
}

#[test]
fn should_reset_specific_envs_through_wrapper() {
    let mut wrapper = QuadcopterEnvWrapper::new(4).unwrap();
    wrapper.reset_all(42);

    // Step to change state
    let actions = vec![rpm_to_action(HOVER_RPM); 4 * 4];
    wrapper.step(&actions);

    // Record state of env 1
    let mut obs_before = vec![0.0f32; 4 * wrapper.obs_size];
    wrapper.write_observations(&mut obs_before);
    let pos_1_before = &obs_before[wrapper.obs_size..2 * wrapper.obs_size].to_vec();

    // Reset only env 0 and 2
    wrapper.reset_envs(&[0, 2]);

    let mut obs_after = vec![0.0f32; 4 * wrapper.obs_size];
    wrapper.write_observations(&mut obs_after);
    let pos_1_after = &obs_after[wrapper.obs_size..2 * wrapper.obs_size];

    // Env 1 should be unchanged
    for i in 0..wrapper.obs_size {
        assert!(
            (pos_1_before[i] - pos_1_after[i]).abs() < 1e-5,
            "Env 1 should not change when resetting 0 and 2"
        );
    }
}

// ============================================================================
// Tracking Mode Tests
// ============================================================================

#[test]
fn should_support_tracking_mode_with_moving_target() {
    let config = QuadcopterConfig::new(2)
        .with_init(InitConfig::fixed_start())
        .with_task_mode(TaskMode::Tracking)
        .with_observation(ObsConfig::tracking())
        .with_termination(TerminationConfig::default().with_max_steps(100));
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let actions = vec![rpm_to_action(HOVER_RPM); 2 * 4];

    for step in 0..10 {
        // Update moving target
        let t = step as f32 * 0.1;
        let targets = vec![
            t.sin(),
            t.cos(),
            1.0, // env 0
            -t.sin(),
            -t.cos(),
            1.5, // env 1
        ];
        env.set_targets(&targets);

        env.step_no_reset(&actions);
    }

    // Verify targets were updated
    let target0 = env.state().get_target_position(0);
    let target1 = env.state().get_target_position(1);

    assert!(
        (target0[2] - 1.0).abs() < 0.1,
        "Target 0 z should be ~1.0"
    );
    assert!(
        (target1[2] - 1.5).abs() < 0.1,
        "Target 1 z should be ~1.5"
    );
}

// ============================================================================
// Long-Running Stability Tests
// ============================================================================

#[test]
fn should_remain_stable_over_long_episode() {
    let config = QuadcopterConfig::new(1)
        .with_init(InitConfig::fixed_start())
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(10000)
                .without_position_bounds()
                .without_attitude_bounds()
                .with_ground_collision(false),
        );
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4];

    // Run for many steps
    for step in 0..1000 {
        env.step_no_reset(&actions);

        // Periodically check state validity
        if step % 100 == 0 {
            let pos = env.state().get_position(0);
            let vel = env.state().get_velocity(0);
            let q = env.state().get_quaternion(0);

            assert!(
                pos.iter().all(|p| p.is_finite()),
                "Position should remain finite at step {}",
                step
            );
            assert!(
                vel.iter().all(|v| v.is_finite()),
                "Velocity should remain finite at step {}",
                step
            );
            assert!(
                q.iter().all(|qi| qi.is_finite()),
                "Quaternion should remain finite at step {}",
                step
            );

            let q_norm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();
            assert!(
                (q_norm - 1.0).abs() < 0.01,
                "Quaternion should stay normalized at step {}: {}",
                step,
                q_norm
            );
        }
    }
}

#[test]
fn should_handle_chaotic_action_sequence() {
    let config = QuadcopterConfig::new(4)
        .with_init(InitConfig::fixed_start())
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(1000)
                .without_position_bounds(),
        );
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Use random-ish but deterministic actions
    let mut rng = fastrand::Rng::with_seed(12345);

    for step in 0..200 {
        let actions: Vec<f32> = (0..4 * 4)
            .map(|_| rng.f32() * 2.0 - 1.0)
            .collect();

        env.step_no_reset(&actions);

        // Check for NaN
        for idx in 0..4 {
            let pos = env.state().get_position(idx);
            assert!(
                pos.iter().all(|p| p.is_finite()),
                "Position should remain finite at step {} env {}",
                step,
                idx
            );
        }
    }
}

// ============================================================================
// Performance Sanity Tests
// ============================================================================

#[test]
fn should_complete_many_steps_quickly() {
    use std::time::Instant;

    let config = QuadcopterConfig::new(64);
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let actions = vec![rpm_to_action(HOVER_RPM); 64 * 4];

    let start = Instant::now();
    for _ in 0..1000 {
        env.step_no_reset(&actions);
    }
    let elapsed = start.elapsed();

    // Should complete 1000 steps in reasonable time (< 5 seconds)
    assert!(
        elapsed.as_secs() < 5,
        "1000 steps should complete in <5s: {:?}",
        elapsed
    );
}
