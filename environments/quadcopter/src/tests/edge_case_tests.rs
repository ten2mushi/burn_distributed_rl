//! Edge case tests defining behavior at system boundaries and extreme inputs.
//!
//! These tests define:
//! - Zero RPM behavior (motors off)
//! - Maximum RPM behavior
//! - Action space boundaries [-1, 1]
//! - Very small and very large timesteps
//! - Multiple environments with mixed states
//! - Numerical precision edge cases
//! - Singular and degenerate states

use crate::config::{InitConfig, ObsConfig, QuadcopterConfig, RewardConfig, TerminationConfig};
use crate::constants::*;
use crate::env::Quadcopter;
use crate::physics::aerodynamics::*;
use crate::physics::dynamics::*;
use crate::physics::quaternion::*;
use crate::state::QuadcopterState;
use operant_core::Environment;
use std::f32::consts::PI;

// ============================================================================
// Zero RPM Edge Cases
// ============================================================================

#[test]
fn should_produce_zero_thrust_with_zero_rpm() {
    let rpms = [0.0; 4];
    let thrusts = compute_thrusts(rpms);
    let total = compute_total_thrust(thrusts);

    assert!(
        total.abs() < 1e-10,
        "Zero RPM should produce zero thrust: {}",
        total
    );
}

#[test]
fn should_produce_zero_torques_with_zero_rpm() {
    let rpms = [0.0; 4];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    for (i, tau) in torques.iter().enumerate() {
        assert!(
            tau.abs() < 1e-10,
            "Zero RPM should produce zero torque[{}]: {:e}",
            i,
            tau
        );
    }
}

#[test]
fn should_fall_freely_with_zero_rpm() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let initial_z = state.pos_z[0];
    let initial_vz = state.vel_z[0];

    let rpms = [0.0; 4];
    let dt = 0.1; // Larger timestep to see effect

    physics_step_scalar(&mut state, 0, rpms, dt);

    // Should accelerate downward at g
    let expected_vz = initial_vz - G * dt;
    let expected_z = initial_z + expected_vz * dt;

    assert!(
        state.vel_z[0] < initial_vz,
        "Should accelerate downward: {} vs {}",
        state.vel_z[0],
        initial_vz
    );
    assert!(
        state.pos_z[0] < initial_z,
        "Should fall: {} vs {}",
        state.pos_z[0],
        initial_z
    );
}

#[test]
fn should_convert_action_minus_one_to_zero_rpm() {
    let rpm = action_to_rpm(-1.0);
    assert!(
        rpm.abs() < 1e-10,
        "Action -1 should map to 0 RPM: {}",
        rpm
    );
}

#[test]
fn should_handle_zero_rpm_in_full_environment_step() {
    let config = QuadcopterConfig::new(4)
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(10)
                .with_ground_collision(false),
        );
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // All motors off
    let actions = vec![-1.0; 4 * 4];

    // Should complete without panic
    for _ in 0..5 {
        env.step_no_reset(&actions);
    }

    // Environments should be falling (velocity negative)
    for idx in 0..4 {
        assert!(
            env.state().vel_z[idx] < 0.0,
            "Env {} should be falling with motors off",
            idx
        );
    }
}

// ============================================================================
// Maximum RPM Edge Cases
// ============================================================================

#[test]
fn should_clamp_rpm_at_max() {
    let rpm = action_to_rpm(2.0); // Above valid range
    assert!(
        (rpm - MAX_RPM).abs() < 1.0,
        "Action >1 should clamp to MAX_RPM: {}",
        rpm
    );
}

#[test]
fn should_convert_action_one_to_max_rpm() {
    let rpm = action_to_rpm(1.0);
    assert!(
        (rpm - MAX_RPM).abs() < 1.0,
        "Action 1 should map to MAX_RPM: {} vs {}",
        rpm,
        MAX_RPM
    );
}

#[test]
fn should_produce_high_thrust_at_max_rpm() {
    let rpms = [MAX_RPM; 4];
    let thrusts = compute_thrusts(rpms);
    let total = compute_total_thrust(thrusts);

    // Expected thrust-to-weight ratio of 2.25
    let weight = M * G;
    let ratio = total / weight;

    assert!(
        (ratio - 2.25).abs() < 0.1,
        "Max RPM should give TWR ~2.25: {}",
        ratio
    );
}

#[test]
fn should_accelerate_strongly_upward_at_max_rpm() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let initial_vz = state.vel_z[0];
    let rpms = [MAX_RPM; 4];
    let dt = 0.01;

    physics_step_scalar(&mut state, 0, rpms, dt);

    // Should accelerate upward strongly (TWR > 2)
    assert!(
        state.vel_z[0] > initial_vz + G * dt,
        "Should accelerate faster than gravity: vz_new={} vs vz_old+g*dt={}",
        state.vel_z[0],
        initial_vz + G * dt
    );
}

#[test]
fn should_handle_max_rpm_in_full_environment_step() {
    let config = QuadcopterConfig::new(4)
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(100)
                .without_position_bounds(),
        );
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // All motors at max
    let actions = vec![1.0; 4 * 4];

    // Should complete without panic
    for _ in 0..10 {
        env.step_no_reset(&actions);
    }

    // Environments should be accelerating upward
    for idx in 0..4 {
        assert!(
            env.state().vel_z[idx] > 0.0,
            "Env {} should be rising with max thrust",
            idx
        );
    }
}

// ============================================================================
// Action Space Boundary Tests
// ============================================================================

#[test]
fn should_clamp_actions_below_minus_one() {
    let rpm = action_to_rpm(-5.0);
    assert!(
        rpm >= 0.0 && rpm <= 1.0,
        "Actions below -1 should clamp to 0 RPM: {}",
        rpm
    );
}

#[test]
fn should_clamp_actions_above_one() {
    let rpm = action_to_rpm(5.0);
    assert!(
        (rpm - MAX_RPM).abs() < 1.0,
        "Actions above 1 should clamp to MAX_RPM: {}",
        rpm
    );
}

#[test]
fn should_handle_nan_action_gracefully() {
    let rpm = action_to_rpm(f32::NAN);
    // NaN handling may vary, but should not crash
    assert!(rpm.is_nan() || rpm >= 0.0, "NaN action should be handled");
}

#[test]
fn should_handle_inf_action_gracefully() {
    let rpm_pos = action_to_rpm(f32::INFINITY);
    let rpm_neg = action_to_rpm(f32::NEG_INFINITY);

    // Should clamp to boundaries
    assert!(
        (rpm_pos - MAX_RPM).abs() < 1.0 || rpm_pos.is_nan(),
        "+Inf should clamp to MAX_RPM: {}",
        rpm_pos
    );
    assert!(
        rpm_neg.abs() < 1.0 || rpm_neg.is_nan(),
        "-Inf should clamp to 0: {}",
        rpm_neg
    );
}

#[test]
fn should_provide_linear_action_to_rpm_mapping() {
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let expected_rpms = [0.0, MAX_RPM * 0.25, MAX_RPM * 0.5, MAX_RPM * 0.75, MAX_RPM];

    for (&action, &expected) in test_points.iter().zip(expected_rpms.iter()) {
        let rpm = action_to_rpm(action);
        assert!(
            (rpm - expected).abs() < 10.0,
            "action_to_rpm({}) = {} (expected {})",
            action,
            rpm,
            expected
        );
    }
}

#[test]
fn should_roundtrip_rpm_to_action_to_rpm() {
    for rpm in [0.0, HOVER_RPM, MAX_RPM, MAX_RPM * 0.5] {
        let action = rpm_to_action(rpm);
        let rpm_back = action_to_rpm(action);
        assert!(
            (rpm - rpm_back).abs() < 1.0,
            "Roundtrip failed: {} -> {} -> {}",
            rpm,
            action,
            rpm_back
        );
    }
}

// ============================================================================
// Timestep Edge Cases
// ============================================================================

#[test]
fn should_handle_very_small_timestep() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let initial_pos = state.get_position(0);
    let rpms = [HOVER_RPM; 4];
    let dt = 1e-9; // Very small timestep

    physics_step_scalar(&mut state, 0, rpms, dt);

    let final_pos = state.get_position(0);

    // Position should barely change
    let diff = ((final_pos[0] - initial_pos[0]).powi(2)
        + (final_pos[1] - initial_pos[1]).powi(2)
        + (final_pos[2] - initial_pos[2]).powi(2))
    .sqrt();

    assert!(
        diff < 1e-6,
        "Very small timestep should produce tiny changes: {}",
        diff
    );
}

#[test]
fn should_handle_large_timestep_without_explosion() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let rpms = [HOVER_RPM; 4];
    let dt = 1.0; // Large timestep

    physics_step_scalar(&mut state, 0, rpms, dt);

    // Values should remain finite
    assert!(state.pos_z[0].is_finite(), "Position should be finite");
    assert!(state.vel_z[0].is_finite(), "Velocity should be finite");
    assert!(state.quat_w[0].is_finite(), "Quaternion should be finite");
}

#[test]
fn should_maintain_quaternion_normalization_with_many_small_steps() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);
    state.set_angular_velocity(0, [1.0, 0.5, 0.3]); // Add rotation

    let rpms = [HOVER_RPM * 1.1, HOVER_RPM * 0.9, HOVER_RPM * 1.05, HOVER_RPM * 0.95];
    let dt = 1e-4; // Very small timestep

    for _ in 0..10000 {
        physics_step_scalar(&mut state, 0, rpms, dt);
    }

    let q = state.get_quaternion(0);
    let norm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();

    assert!(
        (norm - 1.0).abs() < 0.01,
        "Quaternion should stay normalized: {}",
        norm
    );
}

// ============================================================================
// Mixed Environment State Tests
// ============================================================================

#[test]
fn should_handle_environments_with_different_termination_states() {
    let config = QuadcopterConfig::new(8)
        .with_init(InitConfig::fixed_start())
        .with_termination(TerminationConfig::default());
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Set up different states:
    // Env 0: Normal
    // Env 1: About to collide with ground
    env.state_mut().pos_z[1] = 0.001;
    env.state_mut().vel_z[1] = -1.0;
    // Env 2: Out of bounds
    env.state_mut().pos_x[2] = 10.0;
    // Env 3: Flipped
    let flipped_quat = euler_to_quat([PI / 2.0, 0.0, 0.0]);
    env.state_mut().set_quaternion(3, flipped_quat);
    // Env 4-7: Normal

    let actions = vec![rpm_to_action(HOVER_RPM); 8 * 4];

    // Step should handle mixed states
    let result = env.step_no_reset_with_result(&actions);

    // Check termination flags
    let mut terminal_count = 0;
    for &t in result.terminals.iter() {
        if t != 0 {
            terminal_count += 1;
        }
    }

    assert!(
        terminal_count >= 2,
        "At least 2 environments should be terminal: {}",
        terminal_count
    );
}

#[test]
fn should_maintain_independent_state_during_mixed_updates() {
    let config = QuadcopterConfig::new(4)
        .with_init(InitConfig::fixed_start())
        .with_termination(
            TerminationConfig::default()
                .with_max_steps(1000)
                .without_position_bounds()
                .without_attitude_bounds()
                .with_ground_collision(false),
        );
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    // Set different initial velocities
    env.state_mut().set_velocity(0, [1.0, 0.0, 0.0]);
    env.state_mut().set_velocity(1, [0.0, 1.0, 0.0]);
    env.state_mut().set_velocity(2, [0.0, 0.0, 1.0]);
    env.state_mut().set_velocity(3, [-1.0, -1.0, -1.0]);

    // All hover
    let actions = vec![rpm_to_action(HOVER_RPM); 4 * 4];

    env.step_no_reset(&actions);

    // Velocities should remain distinct (though modified by physics)
    let vels: Vec<[f32; 3]> = (0..4).map(|i| env.state().get_velocity(i)).collect();

    // At least some should be different
    let all_same = (1..4).all(|i| {
        (vels[i][0] - vels[0][0]).abs() < 1e-6
            && (vels[i][1] - vels[0][1]).abs() < 1e-6
            && (vels[i][2] - vels[0][2]).abs() < 1e-6
    });

    assert!(!all_same, "Different initial conditions should lead to different states");
}

// ============================================================================
// Numerical Precision Edge Cases
// ============================================================================

#[test]
fn should_handle_near_singular_quaternion() {
    let mut state = QuadcopterState::new(1, 0);

    // Near-zero quaternion (should normalize to identity)
    state.set_quaternion(0, [1e-10, 1e-10, 1e-10, 1e-10]);

    // Try to normalize
    let q = quat_normalize(state.get_quaternion(0));

    // Should be identity or close to it
    let norm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5 || q[0] == 1.0,
        "Should handle near-zero quaternion: {:?}",
        q
    );
}

#[test]
fn should_handle_very_high_angular_velocity() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    // Very high angular velocity (10 rad/s on all axes)
    state.set_angular_velocity(0, [10.0, 10.0, 10.0]);

    let rpms = [HOVER_RPM; 4];
    let dt = 1.0 / 240.0;

    // Should complete without panic
    physics_step_scalar(&mut state, 0, rpms, dt);

    // State should remain finite
    assert!(
        state.get_quaternion(0).iter().all(|x| x.is_finite()),
        "Quaternion should remain finite"
    );
}

#[test]
fn should_handle_gimbal_lock_near_pitch_90() {
    // Gimbal lock occurs at pitch = +/- 90 degrees
    let quat = euler_to_quat([0.0, PI / 2.0 - 0.01, 0.0]); // Just under 90 deg pitch

    let euler = quat_to_euler(quat);

    // Should still produce valid angles
    assert!(euler[0].is_finite(), "Roll should be finite near gimbal lock");
    assert!(euler[1].is_finite(), "Pitch should be finite near gimbal lock");
    assert!(euler[2].is_finite(), "Yaw should be finite near gimbal lock");
}

// ============================================================================
// Ground Effect Edge Cases
// ============================================================================

#[test]
fn should_handle_negative_height_in_ground_effect() {
    // Negative height should be clamped
    let mult = ground_effect_multiplier(-1.0);

    // Should be clamped and return valid multiplier
    assert!(mult.is_finite() && mult >= 1.0, "Ground effect at negative height should be clamped: {}", mult);
}

#[test]
fn should_handle_zero_height_in_ground_effect() {
    let mult = ground_effect_multiplier(0.0);

    // Should use clipped height and return large multiplier
    assert!(
        mult > 1.0 && mult.is_finite(),
        "Ground effect at zero height should be large but finite: {}",
        mult
    );
}

#[test]
fn should_transition_smoothly_through_ground_effect_boundary() {
    // Check continuity around GND_EFF_MAX_HEIGHT
    let heights = [
        GND_EFF_MAX_HEIGHT - 0.01,
        GND_EFF_MAX_HEIGHT,
        GND_EFF_MAX_HEIGHT + 0.01,
    ];

    let mults: Vec<f32> = heights.iter().map(|&h| ground_effect_multiplier(h)).collect();

    // Should be continuous (no jumps)
    let jump1 = (mults[1] - mults[0]).abs();
    let jump2 = (mults[2] - mults[1]).abs();

    assert!(
        jump1 < 0.5 && jump2 < 0.5,
        "Ground effect should be continuous: {:?}",
        mults
    );
}

// ============================================================================
// Drag Edge Cases
// ============================================================================

#[test]
fn should_handle_zero_rpm_in_drag() {
    let velocity = [1.0, 1.0, 1.0];
    let total_rpm = 0.0;
    let drag = compute_drag_force(velocity, total_rpm);

    // Zero RPM should give zero drag
    for d in &drag {
        assert!(d.abs() < 1e-10, "Zero RPM should give zero drag: {:?}", drag);
    }
}

#[test]
fn should_handle_high_velocity_in_drag() {
    let velocity = [100.0, 100.0, 100.0]; // Very high velocity
    let total_rpm = 4.0 * HOVER_RPM;
    let drag = compute_drag_force(velocity, total_rpm);

    // Should produce large but finite drag
    for d in &drag {
        assert!(d.is_finite(), "Drag should be finite at high velocity");
        assert!(d.abs() > 0.0, "Drag should be non-zero at high velocity");
    }
}

// ============================================================================
// Environment Configuration Edge Cases
// ============================================================================

#[test]
fn should_handle_single_environment() {
    let config = QuadcopterConfig::new(1);
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let actions = vec![rpm_to_action(HOVER_RPM); 4];

    for _ in 0..100 {
        env.step_no_reset(&actions);
    }

    // Should complete without issues
    assert!(env.state().pos_z[0].is_finite());
}

#[test]
fn should_handle_non_multiple_of_eight_environments() {
    // SIMD processes 8 at a time; test with remainder
    for num_envs in [1, 3, 7, 9, 15, 17, 63, 65] {
        let config = QuadcopterConfig::new(num_envs);
        let mut env = Quadcopter::from_config(config).unwrap();
        env.reset(42);

        let actions = vec![rpm_to_action(HOVER_RPM); num_envs * 4];

        env.step_no_reset(&actions);

        assert_eq!(
            env.num_envs(),
            num_envs,
            "Should support {} environments",
            num_envs
        );
    }
}

#[test]
fn should_handle_large_number_of_environments() {
    let config = QuadcopterConfig::new(256);
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let actions = vec![rpm_to_action(HOVER_RPM); 256 * 4];

    env.step_no_reset(&actions);

    // Should complete without memory issues
    assert_eq!(env.num_envs(), 256);
}

#[test]
fn should_handle_empty_observation_config() {
    let config = QuadcopterConfig::new(4).with_observation(ObsConfig::new());

    // Empty observation size should be rejected
    let result = Quadcopter::from_config(config);
    assert!(result.is_err(), "Should reject empty observation config");
}

#[test]
fn should_handle_very_short_episode_length() {
    let config = QuadcopterConfig::new(4).with_termination(TerminationConfig::new().with_max_steps(1));
    let mut env = Quadcopter::from_config(config).unwrap();
    env.reset(42);

    let actions = vec![rpm_to_action(HOVER_RPM); 4 * 4];
    let result = env.step_no_reset_with_result(&actions);

    // All environments should be truncated after 1 step
    assert!(
        result.truncations.iter().all(|&t| t == 1),
        "All envs should be truncated at max_steps=1"
    );
}

// ============================================================================
// Observation Buffer Edge Cases
// ============================================================================

#[test]
fn should_handle_action_buffer_of_length_one() {
    let config = QuadcopterConfig::new(2).with_observation(
        ObsConfig::new()
            .with_position()
            .with_action_buffer(1),
    );
    let env = Quadcopter::from_config(config).unwrap();

    assert_eq!(
        env.observation_size(),
        3 + 4,
        "Position + 1 action = 7"
    );
}

#[test]
fn should_handle_large_action_buffer() {
    let config = QuadcopterConfig::new(2).with_observation(
        ObsConfig::new()
            .with_position()
            .with_action_buffer(100),
    );
    let env = Quadcopter::from_config(config).unwrap();

    assert_eq!(
        env.observation_size(),
        3 + 100 * 4,
        "Position + 100 actions"
    );
}
