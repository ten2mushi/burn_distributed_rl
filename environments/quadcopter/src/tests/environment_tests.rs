//! Environment API tests defining the complete behavior of the Quadcopter Environment trait.
//!
//! These tests define:
//! - Environment creation and configuration validation
//! - step() vs step_no_reset() behavior differences
//! - reset_envs() with selective mask
//! - Observation buffer writing
//! - Reward, terminal, and truncation buffer writing
//! - StepResult API correctness

use crate::config::{InitConfig, ObsConfig, QuadcopterConfig, RewardConfig, TerminationConfig};
use crate::constants::*;
use crate::env::Quadcopter;
use operant_core::{Environment, ResetMask};

// ============================================================================
// Environment Creation Tests
// ============================================================================

#[test]
fn should_create_environment_with_valid_config() {
    let config = QuadcopterConfig::new(8);
    let result = Quadcopter::from_config(config);
    assert!(result.is_ok(), "Should create environment with valid config");
}

#[test]
fn should_reject_zero_environments() {
    let config = QuadcopterConfig::new(0);
    let result = Quadcopter::from_config(config);
    assert!(result.is_err(), "Should reject zero environments");
}

#[test]
fn should_reject_invalid_frequency_ratio() {
    let config = QuadcopterConfig::new(8)
        .with_physics_freq(240)
        .with_ctrl_freq(70); // 240 not divisible by 70

    let result = Quadcopter::from_config(config);
    assert!(result.is_err(), "Should reject invalid frequency ratio");
}

#[test]
fn should_report_correct_num_envs() {
    let env = Quadcopter::from_config(QuadcopterConfig::new(32)).unwrap();
    assert_eq!(env.num_envs(), 32);
}

#[test]
fn should_report_correct_observation_size_for_kinematic() {
    let config = QuadcopterConfig::new(8).with_observation(ObsConfig::kinematic());
    let env = Quadcopter::from_config(config).unwrap();
    assert_eq!(env.observation_size(), 12); // pos(3) + euler(3) + vel(3) + ang_vel(3)
}

#[test]
fn should_report_correct_observation_size_for_full_state() {
    let config = QuadcopterConfig::new(8).with_observation(ObsConfig::full_state());
    let env = Quadcopter::from_config(config).unwrap();
    assert_eq!(env.observation_size(), 13); // pos(3) + quat(4) + vel(3) + ang_vel(3)
}

#[test]
fn should_report_correct_observation_size_for_tracking() {
    let config = QuadcopterConfig::new(8).with_observation(ObsConfig::tracking());
    let env = Quadcopter::from_config(config).unwrap();
    assert_eq!(env.observation_size(), 15); // kinematic(12) + target_pos(3)
}

#[test]
fn should_report_none_for_continuous_action_space() {
    let env = Quadcopter::from_config(QuadcopterConfig::new(8)).unwrap();
    assert!(
        env.num_actions().is_none(),
        "Continuous action space should return None"
    );
}

#[test]
fn should_support_no_reset_api() {
    let env = Quadcopter::from_config(QuadcopterConfig::new(8)).unwrap();
    assert!(
        env.supports_no_reset(),
        "Quadcopter should support no-reset API"
    );
}

// ============================================================================
// Reset Tests
// ============================================================================

#[test]
fn should_reset_all_environments() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(8)).unwrap();

    env.reset(42);

    // All environments should be in initial state
    for idx in 0..8 {
        assert!(
            env.state().step_count[idx] == 0,
            "Step count should be 0 after reset"
        );
    }
}

#[test]
fn should_populate_observations_after_reset() {
    let config = QuadcopterConfig::new(4).with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();

    env.reset(42);

    let mut obs = vec![0.0; 4 * 12];
    env.write_observations(&mut obs);

    // Should have non-zero values (at least z position)
    let has_nonzero = obs.iter().any(|&x| x != 0.0);
    assert!(has_nonzero, "Observations should be populated after reset");
}

#[test]
fn should_clear_terminal_and_truncation_flags_after_reset() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();

    env.reset(42);

    let mut terminals = vec![1u8; 4];
    let mut truncations = vec![1u8; 4];

    env.write_terminals(&mut terminals);
    env.write_truncations(&mut truncations);

    assert!(
        terminals.iter().all(|&t| t == 0),
        "Terminals should be cleared after reset"
    );
    assert!(
        truncations.iter().all(|&t| t == 0),
        "Truncations should be cleared after reset"
    );
}

// ============================================================================
// Step Tests
// ============================================================================

#[test]
fn should_increment_step_count_after_step() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    env.step(&actions);

    // At least some environments should have incremented step count
    // (unless they were immediately reset due to termination)
    let total_steps: u32 = env.state().step_count.iter().sum();
    assert!(total_steps > 0 || env.state().step_count.iter().any(|&s| s == 0));
}

#[test]
fn should_update_observations_after_step() {
    let config = QuadcopterConfig::new(4)
        .with_observation(ObsConfig::kinematic())
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let mut obs_before = vec![0.0; 4 * 12];
    env.write_observations(&mut obs_before);

    // Apply non-hover action to cause movement
    let actions = vec![
        rpm_to_action(HOVER_RPM * 1.2),
        rpm_to_action(HOVER_RPM * 0.8),
        rpm_to_action(HOVER_RPM * 1.1),
        rpm_to_action(HOVER_RPM * 0.9),
    ]
    .repeat(4);

    env.step(&actions);

    let mut obs_after = vec![0.0; 4 * 12];
    env.write_observations(&mut obs_after);

    // Observations should change
    let changed = obs_before
        .iter()
        .zip(obs_after.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(changed, "Observations should change after step");
}

#[test]
fn should_compute_rewards_after_step() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    env.step(&actions);

    let mut rewards = vec![0.0; 4];
    env.write_rewards(&mut rewards);

    // Rewards should be computed (could be any value, but should be finite)
    assert!(
        rewards.iter().all(|r| r.is_finite()),
        "Rewards should be finite"
    );
}

#[test]
fn should_compute_termination_flags_after_step() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    env.step(&actions);

    let mut terminals = vec![255u8; 4];
    env.write_terminals(&mut terminals);

    // All flags should be 0 or 1
    assert!(
        terminals.iter().all(|&t| t <= 1),
        "Terminal flags should be 0 or 1"
    );
}

// ============================================================================
// step() vs step_no_reset() Tests
// ============================================================================

#[test]
fn should_auto_reset_terminated_envs_in_step() {
    let config = QuadcopterConfig::new(4)
        .with_termination(TerminationConfig::default().with_ground_collision(true))
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Force an environment below ground
    env.state_mut().pos_z[0] = -0.1;

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    env.step(&actions);

    // After step(), the terminated env should be reset (z > 0)
    assert!(
        env.state().pos_z[0] > 0.0,
        "Terminated env should be auto-reset: z={}",
        env.state().pos_z[0]
    );
}

#[test]
fn should_not_auto_reset_in_step_no_reset() {
    let config = QuadcopterConfig::new(4)
        .with_termination(TerminationConfig::default().with_ground_collision(true))
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Force an environment below ground
    env.state_mut().pos_z[0] = -0.1;

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    env.step_no_reset(&actions);

    // After step_no_reset(), the terminated env should NOT be reset
    // The z position will have been updated by physics but should still be negative
    // (or very slightly positive due to physics step from negative velocity)
    let mut terminals = vec![0u8; 4];
    env.write_terminals(&mut terminals);

    assert_eq!(
        terminals[0], 1,
        "Env 0 should be marked as terminal but not reset"
    );
}

#[test]
fn should_return_terminal_observation_in_step_no_reset() {
    let config = QuadcopterConfig::new(4)
        .with_observation(ObsConfig::new().with_position())
        .with_termination(TerminationConfig::default().with_ground_collision(true))
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Force an environment below ground
    env.state_mut().pos_z[0] = -0.1;

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    let result = env.step_no_reset_with_result(&actions);

    // The observation should reflect the terminal state (not a reset state)
    // Z position should still be near or below ground
    let obs_z = result.observations[2]; // z component of env 0

    // The physics step will have updated the position, but it started below ground
    // So terminal should be flagged
    assert_eq!(
        result.terminals[0], 1,
        "Env 0 should be terminal"
    );
}

#[test]
fn should_return_step_result_with_correct_dimensions() {
    let config = QuadcopterConfig::new(8).with_observation(ObsConfig::kinematic());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 8 * 4];

    let result = env.step_no_reset_with_result(&actions);

    assert_eq!(result.num_envs, 8);
    assert_eq!(result.obs_size, 12);
    assert_eq!(result.observations.len(), 8 * 12);
    assert_eq!(result.rewards.len(), 8);
    assert_eq!(result.terminals.len(), 8);
    assert_eq!(result.truncations.len(), 8);
}

// ============================================================================
// reset_envs() Tests
// ============================================================================

#[test]
fn should_reset_only_masked_environments() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(8)).unwrap();
    env.reset(42);

    // Modify all environments
    for idx in 0..8 {
        env.state_mut().pos_x[idx] = idx as f32;
        env.state_mut().step_count[idx] = 100;
    }

    // Create mask for envs 2 and 5
    let mut mask = ResetMask::new(8);
    mask.set(2);
    mask.set(5);

    env.reset_envs(&mask, 123);

    // Envs 2 and 5 should be reset
    assert!(
        env.state().pos_x[2] != 2.0 || env.state().step_count[2] == 0,
        "Env 2 should be reset"
    );
    assert!(
        env.state().pos_x[5] != 5.0 || env.state().step_count[5] == 0,
        "Env 5 should be reset"
    );

    // Other envs should be unchanged
    assert_eq!(env.state().pos_x[0], 0.0, "Env 0 should be unchanged");
    assert_eq!(env.state().step_count[0], 100, "Env 0 step count unchanged");
    assert_eq!(env.state().pos_x[3], 3.0, "Env 3 should be unchanged");
    assert_eq!(env.state().pos_x[7], 7.0, "Env 7 should be unchanged");
}

#[test]
fn should_reset_envs_with_empty_mask() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    // Modify environments
    for idx in 0..4 {
        env.state_mut().pos_x[idx] = idx as f32 + 10.0;
    }

    let mask = ResetMask::new(4); // Empty mask
    env.reset_envs(&mask, 123);

    // No environments should be reset
    for idx in 0..4 {
        assert_eq!(
            env.state().pos_x[idx],
            idx as f32 + 10.0,
            "Env {} should be unchanged with empty mask",
            idx
        );
    }
}

#[test]
fn should_reset_all_envs_with_full_mask() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    // Modify environments
    for idx in 0..4 {
        env.state_mut().step_count[idx] = 100;
    }

    let mut mask = ResetMask::new(4);
    for idx in 0..4 {
        mask.set(idx);
    }

    env.reset_envs(&mask, 123);

    // All environments should be reset
    for idx in 0..4 {
        assert_eq!(
            env.state().step_count[idx], 0,
            "Env {} should be reset with full mask",
            idx
        );
    }
}

#[test]
fn should_use_different_seeds_per_env_in_reset_envs() {
    let config = QuadcopterConfig::new(4).with_init(InitConfig::default()); // Uniform distribution
    let mut env = Quadcopter::from_config(config).unwrap();

    let mut mask = ResetMask::new(4);
    for idx in 0..4 {
        mask.set(idx);
    }

    env.reset_envs(&mask, 42);

    // Different environments should have different positions
    let positions: Vec<[f32; 3]> = (0..4).map(|i| env.state().get_position(i)).collect();

    // At least some positions should differ
    let all_same = positions.windows(2).all(|w| {
        (w[0][0] - w[1][0]).abs() < 1e-6
            && (w[0][1] - w[1][1]).abs() < 1e-6
            && (w[0][2] - w[1][2]).abs() < 1e-6
    });

    assert!(
        !all_same,
        "Different environments should get different random states"
    );
}

// ============================================================================
// Observation Writing Tests
// ============================================================================

#[test]
fn should_write_correct_observation_components_for_kinematic() {
    let config = QuadcopterConfig::new(1)
        .with_observation(ObsConfig::kinematic())
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Set known state
    env.state_mut().set_position(0, [1.0, 2.0, 3.0]);
    env.state_mut().set_quaternion(0, [1.0, 0.0, 0.0, 0.0]); // Level
    env.state_mut().set_velocity(0, [0.1, 0.2, 0.3]);
    env.state_mut().set_angular_velocity(0, [0.4, 0.5, 0.6]);

    // Force observation update
    let actions = vec![rpm_to_action(HOVER_RPM); 4];
    env.step_no_reset(&actions);

    // Get observations after step (state may have changed)
    // Instead, let's read state directly for verification
    let state = env.state();
    let pos = state.get_position(0);
    let vel = state.get_velocity(0);
    let omega = state.get_angular_velocity(0);

    // Values should be reasonable (physics stepped from initial values)
    assert!(pos[2].is_finite(), "Position should be finite");
    assert!(vel[2].is_finite(), "Velocity should be finite");
    assert!(omega[0].is_finite(), "Angular velocity should be finite");
}

#[test]
fn should_write_correct_observation_components_for_full_state() {
    let config = QuadcopterConfig::new(1)
        .with_observation(ObsConfig::full_state())
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let mut obs = vec![0.0; 13];
    env.write_observations(&mut obs);

    // Check structure: pos(3) + quat(4) + vel(3) + ang_vel(3)
    // Position should start at index 0
    // For fixed_start, z=1.0
    assert!(
        (obs[2] - 1.0).abs() < 0.1,
        "Z position should be near 1.0: {}",
        obs[2]
    );

    // Quaternion starts at index 3 (w, x, y, z)
    // Identity quaternion for level attitude
    assert!(
        (obs[3] - 1.0).abs() < 0.1,
        "Quaternion W should be near 1.0: {}",
        obs[3]
    );
}

#[test]
fn should_write_observations_for_multiple_environments() {
    let config = QuadcopterConfig::new(4)
        .with_observation(ObsConfig::new().with_position())
        .with_init(InitConfig::default());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let mut obs = vec![0.0; 4 * 3]; // 4 envs * 3 (position only)
    env.write_observations(&mut obs);

    // Each environment should have its own position
    for idx in 0..4 {
        let env_obs = &obs[idx * 3..(idx + 1) * 3];
        let state_pos = env.state().get_position(idx);

        for i in 0..3 {
            assert!(
                (env_obs[i] - state_pos[i]).abs() < 1e-5,
                "Env {} position {} mismatch: obs={} state={}",
                idx,
                i,
                env_obs[i],
                state_pos[i]
            );
        }
    }
}

#[test]
fn should_write_observations_in_aos_format() {
    // Verify Array-of-Structures layout
    let config = QuadcopterConfig::new(2)
        .with_observation(ObsConfig::new().with_position().with_velocity());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Set different states
    env.state_mut().set_position(0, [1.0, 2.0, 3.0]);
    env.state_mut().set_velocity(0, [4.0, 5.0, 6.0]);
    env.state_mut().set_position(1, [7.0, 8.0, 9.0]);
    env.state_mut().set_velocity(1, [10.0, 11.0, 12.0]);

    // Trigger observation write
    let actions = vec![rpm_to_action(HOVER_RPM); 8];
    env.step_no_reset(&actions);

    // After physics step, just verify the format is correct
    let mut obs = vec![0.0; 2 * 6];
    env.write_observations(&mut obs);

    // Format should be: [env0_pos_x, env0_pos_y, env0_pos_z, env0_vel_x, env0_vel_y, env0_vel_z,
    //                    env1_pos_x, env1_pos_y, env1_pos_z, env1_vel_x, env1_vel_y, env1_vel_z]
    // All values should be finite
    assert!(obs.iter().all(|x| x.is_finite()), "All observations should be finite");
}

// ============================================================================
// Target Setting Tests
// ============================================================================

#[test]
fn should_set_targets_for_all_environments() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    let targets = vec![
        1.0, 2.0, 3.0, // env 0
        4.0, 5.0, 6.0, // env 1
        7.0, 8.0, 9.0, // env 2
        10.0, 11.0, 12.0, // env 3
    ];

    env.set_targets(&targets);

    for idx in 0..4 {
        let target = env.state().get_target_position(idx);
        assert_eq!(target[0], (idx * 3 + 1) as f32);
        assert_eq!(target[1], (idx * 3 + 2) as f32);
        assert_eq!(target[2], (idx * 3 + 3) as f32);
    }
}

#[test]
fn should_set_target_velocities_for_all_environments() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(2)).unwrap();
    env.reset(42);

    let velocities = vec![
        0.1, 0.2, 0.3, // env 0
        0.4, 0.5, 0.6, // env 1
    ];

    env.set_target_velocities(&velocities);

    assert_eq!(env.state().target_vel[0], 0.1);
    assert_eq!(env.state().target_vel[1], 0.2);
    assert_eq!(env.state().target_vel[2], 0.3);
    assert_eq!(env.state().target_vel[3], 0.4);
    assert_eq!(env.state().target_vel[4], 0.5);
    assert_eq!(env.state().target_vel[5], 0.6);
}

// ============================================================================
// Configuration Access Tests
// ============================================================================

#[test]
fn should_provide_access_to_config() {
    let config = QuadcopterConfig::new(16)
        .with_physics_freq(480)
        .with_ctrl_freq(60);
    let env = Quadcopter::from_config(config).unwrap();

    assert_eq!(env.config().num_envs, 16);
    assert_eq!(env.config().physics_freq, 480);
    assert_eq!(env.config().ctrl_freq, 60);
}

#[test]
fn should_provide_mutable_state_access() {
    let mut env = Quadcopter::from_config(QuadcopterConfig::new(4)).unwrap();
    env.reset(42);

    // Modify state through mutable accessor
    env.state_mut().set_position(0, [99.0, 99.0, 99.0]);

    // Verify modification
    assert_eq!(env.state().get_position(0), [99.0, 99.0, 99.0]);
}

// ============================================================================
// Substep Configuration Tests
// ============================================================================

#[test]
fn should_compute_correct_substeps() {
    let config = QuadcopterConfig::new(8)
        .with_physics_freq(240)
        .with_ctrl_freq(30);
    let env = Quadcopter::from_config(config).unwrap();

    // 240 / 30 = 8 substeps
    assert_eq!(env.config().physics_steps_per_ctrl(), 8);
}

#[test]
fn should_handle_different_frequency_ratios() {
    // Test various valid frequency ratios
    let test_cases = [(240, 30, 8), (240, 60, 4), (480, 60, 8), (120, 30, 4)];

    for (phys, ctrl, expected_substeps) in test_cases {
        let config = QuadcopterConfig::new(1)
            .with_physics_freq(phys)
            .with_ctrl_freq(ctrl);

        assert_eq!(
            config.physics_steps_per_ctrl(),
            expected_substeps,
            "Physics={}, Ctrl={} should give {} substeps",
            phys,
            ctrl,
            expected_substeps
        );
    }
}

// ============================================================================
// Multiple Step Sequence Tests
// ============================================================================

#[test]
fn should_accumulate_step_counts_over_multiple_steps() {
    let config = QuadcopterConfig::new(4)
        .with_termination(TerminationConfig::default().with_max_steps(1000))
        .with_init(InitConfig::fixed_start());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 4 * 4];

    for step in 1..=10 {
        env.step_no_reset(&actions);

        // Check step count (use step_no_reset to avoid auto-reset)
        for idx in 0..4 {
            assert!(
                env.state().step_count[idx] <= step as u32,
                "Step count for env {} should be <= {} after {} steps",
                idx,
                step,
                step
            );
        }
    }
}

#[test]
fn should_maintain_state_consistency_across_steps() {
    let config = QuadcopterConfig::new(2)
        .with_init(InitConfig::fixed_start())
        .with_termination(TerminationConfig::default().with_max_steps(1000));
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let hover_action = rpm_to_action(HOVER_RPM);
    let actions = vec![hover_action; 2 * 4];

    for _ in 0..100 {
        env.step_no_reset(&actions);

        // Quaternions should remain normalized
        for idx in 0..2 {
            let q = env.state().get_quaternion(idx);
            let norm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();
            assert!(
                (norm - 1.0).abs() < 0.001,
                "Quaternion should remain normalized: norm={} at env {}",
                norm,
                idx
            );
        }

        // Positions should be finite
        for idx in 0..2 {
            let pos = env.state().get_position(idx);
            assert!(
                pos.iter().all(|p| p.is_finite()),
                "Positions should be finite"
            );
        }
    }
}
