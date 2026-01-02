//! State management tests defining the complete behavior of QuadcopterState.
//!
//! These tests define:
//! - State initialization and memory layout
//! - Reset functionality with various distributions
//! - Position, quaternion, velocity, and angular velocity accessors
//! - Action buffer management
//! - Target position and velocity management
//! - Episode tracking (step count, cumulative reward)

use crate::config::{DistributionType, InitConfig};
use crate::constants::*;
use crate::state::QuadcopterState;
use std::f32::consts::PI;

// ============================================================================
// State Initialization Tests
// ============================================================================

#[test]
fn should_allocate_correct_array_sizes_on_creation() {
    let num_envs = 64;
    let action_buffer_len = 5;
    let state = QuadcopterState::new(num_envs, action_buffer_len);

    assert_eq!(state.pos_x.len(), num_envs);
    assert_eq!(state.pos_y.len(), num_envs);
    assert_eq!(state.pos_z.len(), num_envs);
    assert_eq!(state.quat_w.len(), num_envs);
    assert_eq!(state.quat_x.len(), num_envs);
    assert_eq!(state.quat_y.len(), num_envs);
    assert_eq!(state.quat_z.len(), num_envs);
    assert_eq!(state.vel_x.len(), num_envs);
    assert_eq!(state.vel_y.len(), num_envs);
    assert_eq!(state.vel_z.len(), num_envs);
    assert_eq!(state.ang_vel_x.len(), num_envs);
    assert_eq!(state.ang_vel_y.len(), num_envs);
    assert_eq!(state.ang_vel_z.len(), num_envs);
    assert_eq!(state.last_rpm.len(), num_envs * 4);
    assert_eq!(state.prev_action.len(), num_envs * 4);
    assert_eq!(state.action_buffer.len(), num_envs * action_buffer_len * 4);
    assert_eq!(state.action_buffer_idx.len(), num_envs);
    assert_eq!(state.target_pos.len(), num_envs * 3);
    assert_eq!(state.target_vel.len(), num_envs * 3);
    assert_eq!(state.step_count.len(), num_envs);
    assert_eq!(state.episode_reward.len(), num_envs);
}

#[test]
fn should_initialize_position_at_one_meter_height() {
    let state = QuadcopterState::new(8, 0);

    for idx in 0..8 {
        assert_eq!(state.pos_x[idx], 0.0);
        assert_eq!(state.pos_y[idx], 0.0);
        assert_eq!(
            state.pos_z[idx], 1.0,
            "Default z position should be 1.0 m"
        );
    }
}

#[test]
fn should_initialize_with_identity_quaternion() {
    let state = QuadcopterState::new(8, 0);

    for idx in 0..8 {
        assert_eq!(state.quat_w[idx], 1.0, "Quaternion W should be 1.0");
        assert_eq!(state.quat_x[idx], 0.0, "Quaternion X should be 0.0");
        assert_eq!(state.quat_y[idx], 0.0, "Quaternion Y should be 0.0");
        assert_eq!(state.quat_z[idx], 0.0, "Quaternion Z should be 0.0");
    }
}

#[test]
fn should_initialize_with_zero_velocity() {
    let state = QuadcopterState::new(8, 0);

    for idx in 0..8 {
        assert_eq!(state.vel_x[idx], 0.0);
        assert_eq!(state.vel_y[idx], 0.0);
        assert_eq!(state.vel_z[idx], 0.0);
    }
}

#[test]
fn should_initialize_with_zero_angular_velocity() {
    let state = QuadcopterState::new(8, 0);

    for idx in 0..8 {
        assert_eq!(state.ang_vel_x[idx], 0.0);
        assert_eq!(state.ang_vel_y[idx], 0.0);
        assert_eq!(state.ang_vel_z[idx], 0.0);
    }
}

#[test]
fn should_initialize_with_hover_rpm() {
    let state = QuadcopterState::new(4, 0);

    for idx in 0..4 {
        for motor in 0..4 {
            assert_eq!(
                state.last_rpm[idx * 4 + motor],
                HOVER_RPM,
                "Motor {} of env {} should be at hover RPM",
                motor,
                idx
            );
        }
    }
}

#[test]
fn should_initialize_prev_action_to_normalized_hover() {
    let state = QuadcopterState::new(4, 0);
    let expected = rpm_to_action(HOVER_RPM);

    for idx in 0..4 {
        for motor in 0..4 {
            assert!(
                (state.prev_action[idx * 4 + motor] - expected).abs() < 1e-6,
                "Previous action should be normalized hover"
            );
        }
    }
}

#[test]
fn should_initialize_step_count_to_zero() {
    let state = QuadcopterState::new(8, 0);

    for idx in 0..8 {
        assert_eq!(state.step_count[idx], 0);
    }
}

#[test]
fn should_store_metadata_correctly() {
    let state = QuadcopterState::new(32, 3);

    assert_eq!(state.num_envs, 32);
    assert_eq!(state.action_buffer_len, 3);
}

// ============================================================================
// Reset Tests with Fixed Distribution
// ============================================================================

#[test]
fn should_reset_to_fixed_position() {
    let mut state = QuadcopterState::new(4, 0);
    let config = InitConfig::fixed_start();

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    // Fixed start: position at (0, 0, 1)
    assert_eq!(state.pos_x[0], 0.0);
    assert_eq!(state.pos_y[0], 0.0);
    assert_eq!(state.pos_z[0], 1.0);
}

#[test]
fn should_reset_to_level_attitude_with_fixed_distribution() {
    let mut state = QuadcopterState::new(4, 0);
    let config = InitConfig::fixed_start();

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    // Identity quaternion = level
    assert_eq!(state.quat_w[0], 1.0);
    assert!(state.quat_x[0].abs() < 1e-6);
    assert!(state.quat_y[0].abs() < 1e-6);
    assert!(state.quat_z[0].abs() < 1e-6);
}

#[test]
fn should_reset_to_zero_velocity_with_fixed_distribution() {
    let mut state = QuadcopterState::new(4, 0);
    let config = InitConfig::fixed_start();

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    assert_eq!(state.vel_x[0], 0.0);
    assert_eq!(state.vel_y[0], 0.0);
    assert_eq!(state.vel_z[0], 0.0);
}

#[test]
fn should_reset_to_zero_angular_velocity_with_fixed_distribution() {
    let mut state = QuadcopterState::new(4, 0);
    let config = InitConfig::fixed_start();

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    assert_eq!(state.ang_vel_x[0], 0.0);
    assert_eq!(state.ang_vel_y[0], 0.0);
    assert_eq!(state.ang_vel_z[0], 0.0);
}

#[test]
fn should_reset_step_count_to_zero() {
    let mut state = QuadcopterState::new(4, 0);
    state.step_count[0] = 100;

    let config = InitConfig::fixed_start();
    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    assert_eq!(state.step_count[0], 0);
}

#[test]
fn should_reset_episode_reward_to_zero() {
    let mut state = QuadcopterState::new(4, 0);
    state.episode_reward[0] = 50.0;

    let config = InitConfig::fixed_start();
    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    assert_eq!(state.episode_reward[0], 0.0);
}

#[test]
fn should_reset_target_to_hover_target() {
    let mut state = QuadcopterState::new(4, 0);
    let config = InitConfig::fixed_start();
    let hover_target = [1.5, 2.5, 3.5];

    state.reset_env(0, 42, &config, hover_target);

    assert_eq!(state.target_pos[0], 1.5);
    assert_eq!(state.target_pos[1], 2.5);
    assert_eq!(state.target_pos[2], 3.5);
}

#[test]
fn should_reset_target_velocity_to_zero() {
    let mut state = QuadcopterState::new(4, 0);
    state.target_vel[0] = 1.0;
    state.target_vel[1] = 2.0;
    state.target_vel[2] = 3.0;

    let config = InitConfig::fixed_start();
    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    assert_eq!(state.target_vel[0], 0.0);
    assert_eq!(state.target_vel[1], 0.0);
    assert_eq!(state.target_vel[2], 0.0);
}

// ============================================================================
// Reset Tests with Uniform Distribution
// ============================================================================

#[test]
fn should_reset_position_within_uniform_bounds() {
    let mut state = QuadcopterState::new(100, 0);
    let config = InitConfig::default(); // Uses uniform distribution

    for seed in 0..100 {
        state.reset_env(0, seed, &config, [0.0, 0.0, 1.0]);

        let range = &config.position_range;
        assert!(
            state.pos_x[0] >= range[0] && state.pos_x[0] <= range[1],
            "X position {} out of bounds [{}, {}]",
            state.pos_x[0],
            range[0],
            range[1]
        );
        assert!(
            state.pos_y[0] >= range[2] && state.pos_y[0] <= range[3],
            "Y position {} out of bounds [{}, {}]",
            state.pos_y[0],
            range[2],
            range[3]
        );
        assert!(
            state.pos_z[0] >= range[4] && state.pos_z[0] <= range[5],
            "Z position {} out of bounds [{}, {}]",
            state.pos_z[0],
            range[4],
            range[5]
        );
    }
}

#[test]
fn should_reset_attitude_within_uniform_bounds() {
    let mut state = QuadcopterState::new(1, 0);
    let config = InitConfig::default();

    // Sample many resets to check bounds
    for seed in 0..100 {
        state.reset_env(0, seed, &config, [0.0, 0.0, 1.0]);

        // Convert quaternion to Euler to check bounds
        let quat = state.get_quaternion(0);
        let euler = crate::physics::quaternion::quat_to_euler(quat);

        let range = &config.attitude_range;
        // Roll and pitch should be within small bounds
        assert!(
            euler[0] >= range[0] - 0.01 && euler[0] <= range[1] + 0.01,
            "Roll {} out of bounds [{}, {}]",
            euler[0],
            range[0],
            range[1]
        );
        assert!(
            euler[1] >= range[2] - 0.01 && euler[1] <= range[3] + 0.01,
            "Pitch {} out of bounds [{}, {}]",
            euler[1],
            range[2],
            range[3]
        );
        // Yaw can be full range
        assert!(
            euler[2] >= range[4] - 0.01 && euler[2] <= range[5] + 0.01,
            "Yaw {} out of bounds [{}, {}]",
            euler[2],
            range[4],
            range[5]
        );
    }
}

#[test]
fn should_produce_different_states_with_different_seeds() {
    let mut state = QuadcopterState::new(1, 0);
    let config = InitConfig::default();

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);
    let pos1 = state.get_position(0);

    state.reset_env(0, 123, &config, [0.0, 0.0, 1.0]);
    let pos2 = state.get_position(0);

    // Different seeds should produce different positions
    let different = (pos1[0] - pos2[0]).abs() > 1e-6
        || (pos1[1] - pos2[1]).abs() > 1e-6
        || (pos1[2] - pos2[2]).abs() > 1e-6;

    assert!(
        different,
        "Different seeds should produce different positions: {:?} vs {:?}",
        pos1,
        pos2
    );
}

#[test]
fn should_produce_same_state_with_same_seed() {
    let mut state = QuadcopterState::new(1, 0);
    let config = InitConfig::default();

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);
    let pos1 = state.get_position(0);
    let quat1 = state.get_quaternion(0);

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);
    let pos2 = state.get_position(0);
    let quat2 = state.get_quaternion(0);

    for i in 0..3 {
        assert!(
            (pos1[i] - pos2[i]).abs() < 1e-10,
            "Same seed should produce same position"
        );
    }
    for i in 0..4 {
        assert!(
            (quat1[i] - quat2[i]).abs() < 1e-10,
            "Same seed should produce same quaternion"
        );
    }
}

// ============================================================================
// Reset Tests with Gaussian Distribution
// ============================================================================

#[test]
fn should_reset_position_with_gaussian_centered_in_range() {
    let mut state = QuadcopterState::new(1, 0);
    let mut config = InitConfig::default();
    config.position_dist = DistributionType::Gaussian;

    // Collect samples
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    let n = 1000;

    for seed in 0..n {
        state.reset_env(0, seed as u64, &config, [0.0, 0.0, 1.0]);
        sum_x += state.pos_x[0];
        sum_y += state.pos_y[0];
        sum_z += state.pos_z[0];
    }

    let mean_x = sum_x / n as f32;
    let mean_y = sum_y / n as f32;
    let mean_z = sum_z / n as f32;

    // Mean should be near center of range
    let range = &config.position_range;
    let expected_x = (range[0] + range[1]) / 2.0;
    let expected_y = (range[2] + range[3]) / 2.0;
    let expected_z = (range[4] + range[5]) / 2.0;

    assert!(
        (mean_x - expected_x).abs() < 0.1,
        "Gaussian X mean {} should be near center {}",
        mean_x,
        expected_x
    );
    assert!(
        (mean_y - expected_y).abs() < 0.1,
        "Gaussian Y mean {} should be near center {}",
        mean_y,
        expected_y
    );
    assert!(
        (mean_z - expected_z).abs() < 0.1,
        "Gaussian Z mean {} should be near center {}",
        mean_z,
        expected_z
    );
}

// ============================================================================
// Reset Tests with Hover Initialization
// ============================================================================

#[test]
fn should_initialize_motors_at_hover_rpm_when_enabled() {
    let mut state = QuadcopterState::new(4, 0);
    let mut config = InitConfig::fixed_start();
    config.hover_init = true;

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    for motor in 0..4 {
        assert!(
            (state.last_rpm[motor] - HOVER_RPM).abs() < 1.0,
            "Motor {} should be at hover RPM: {} vs {}",
            motor,
            state.last_rpm[motor],
            HOVER_RPM
        );
    }
}

#[test]
fn should_initialize_motors_at_zero_rpm_when_disabled() {
    let mut state = QuadcopterState::new(4, 0);
    let mut config = InitConfig::fixed_start();
    config.hover_init = false;

    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    for motor in 0..4 {
        assert_eq!(
            state.last_rpm[motor], 0.0,
            "Motor {} should be at 0 RPM when hover_init=false",
            motor
        );
    }
}

// ============================================================================
// State Accessor Tests
// ============================================================================

#[test]
fn should_get_and_set_position_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    state.set_position(2, [1.0, 2.0, 3.0]);
    let pos = state.get_position(2);

    assert_eq!(pos, [1.0, 2.0, 3.0]);

    // Verify internal arrays
    assert_eq!(state.pos_x[2], 1.0);
    assert_eq!(state.pos_y[2], 2.0);
    assert_eq!(state.pos_z[2], 3.0);
}

#[test]
fn should_get_and_set_quaternion_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    state.set_quaternion(1, [0.7, 0.3, 0.5, 0.4]);
    let quat = state.get_quaternion(1);

    assert_eq!(quat, [0.7, 0.3, 0.5, 0.4]);

    // Verify internal arrays
    assert_eq!(state.quat_w[1], 0.7);
    assert_eq!(state.quat_x[1], 0.3);
    assert_eq!(state.quat_y[1], 0.5);
    assert_eq!(state.quat_z[1], 0.4);
}

#[test]
fn should_get_and_set_velocity_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    state.set_velocity(0, [-1.0, 0.5, 2.0]);
    let vel = state.get_velocity(0);

    assert_eq!(vel, [-1.0, 0.5, 2.0]);
}

#[test]
fn should_get_and_set_angular_velocity_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    state.set_angular_velocity(3, [0.1, -0.2, 0.3]);
    let omega = state.get_angular_velocity(3);

    assert_eq!(omega, [0.1, -0.2, 0.3]);
}

#[test]
fn should_get_rpms_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    // Manually set RPMs
    let base = 2 * 4;
    state.last_rpm[base] = 10000.0;
    state.last_rpm[base + 1] = 11000.0;
    state.last_rpm[base + 2] = 12000.0;
    state.last_rpm[base + 3] = 13000.0;

    let rpms = state.get_rpms(2);

    assert_eq!(rpms, [10000.0, 11000.0, 12000.0, 13000.0]);
}

#[test]
fn should_get_and_set_target_position_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    state.set_target_position(1, [5.0, 6.0, 7.0]);
    let target = state.get_target_position(1);

    assert_eq!(target, [5.0, 6.0, 7.0]);

    // Verify internal array layout
    let base = 1 * 3;
    assert_eq!(state.target_pos[base], 5.0);
    assert_eq!(state.target_pos[base + 1], 6.0);
    assert_eq!(state.target_pos[base + 2], 7.0);
}

#[test]
fn should_set_target_velocity_correctly() {
    let mut state = QuadcopterState::new(4, 0);

    state.set_target_velocity(2, [1.0, 2.0, 3.0]);

    let base = 2 * 3;
    assert_eq!(state.target_vel[base], 1.0);
    assert_eq!(state.target_vel[base + 1], 2.0);
    assert_eq!(state.target_vel[base + 2], 3.0);
}

// ============================================================================
// Action Buffer Tests
// ============================================================================

#[test]
fn should_create_empty_action_buffer_when_length_is_zero() {
    let state = QuadcopterState::new(4, 0);
    assert_eq!(state.action_buffer.len(), 0);
}

#[test]
fn should_create_correct_size_action_buffer() {
    let state = QuadcopterState::new(4, 5);
    // 4 envs * 5 buffer slots * 4 motors = 80
    assert_eq!(state.action_buffer.len(), 80);
}

#[test]
fn should_push_action_to_buffer() {
    let mut state = QuadcopterState::new(1, 3);

    state.push_action_buffer(0, &[0.1, 0.2, 0.3, 0.4]);

    // First slot should have the action
    assert_eq!(state.action_buffer[0], 0.1);
    assert_eq!(state.action_buffer[1], 0.2);
    assert_eq!(state.action_buffer[2], 0.3);
    assert_eq!(state.action_buffer[3], 0.4);
}

#[test]
fn should_update_buffer_index_after_push() {
    let mut state = QuadcopterState::new(1, 3);

    assert_eq!(state.action_buffer_idx[0], 0);

    state.push_action_buffer(0, &[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(state.action_buffer_idx[0], 1);

    state.push_action_buffer(0, &[0.5, 0.6, 0.7, 0.8]);
    assert_eq!(state.action_buffer_idx[0], 2);
}

#[test]
fn should_wrap_buffer_index() {
    let mut state = QuadcopterState::new(1, 3);

    // Push 4 actions to a buffer of length 3
    state.push_action_buffer(0, &[1.0, 1.0, 1.0, 1.0]);
    state.push_action_buffer(0, &[2.0, 2.0, 2.0, 2.0]);
    state.push_action_buffer(0, &[3.0, 3.0, 3.0, 3.0]);
    state.push_action_buffer(0, &[4.0, 4.0, 4.0, 4.0]); // Should wrap to index 0

    assert_eq!(state.action_buffer_idx[0], 1);

    // Slot 0 should now have action 4
    assert_eq!(state.action_buffer[0], 4.0);
}

#[test]
fn should_retrieve_actions_in_order_oldest_to_newest() {
    let mut state = QuadcopterState::new(1, 3);

    state.push_action_buffer(0, &[1.0, 1.1, 1.2, 1.3]);
    state.push_action_buffer(0, &[2.0, 2.1, 2.2, 2.3]);

    let mut output = vec![0.0; 12]; // 3 slots * 4 motors
    state.get_action_buffer(0, &mut output);

    // First 4 values should be oldest remaining action (from slot 2, which is zeros)
    // Then action 1, then action 2
    // Buffer contains: [1,1,1,1] at idx 0, [2,2,2,2] at idx 1, [0,0,0,0] at idx 2
    // Read starts from write position (2), so: zeros, action1, action2
    assert_eq!(&output[0..4], &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(&output[4..8], &[1.0, 1.1, 1.2, 1.3]);
    assert_eq!(&output[8..12], &[2.0, 2.1, 2.2, 2.3]);
}

#[test]
fn should_not_push_to_empty_buffer() {
    let mut state = QuadcopterState::new(1, 0);

    // This should be a no-op
    state.push_action_buffer(0, &[1.0, 2.0, 3.0, 4.0]);

    // Buffer is empty, no crash
    assert!(state.action_buffer.is_empty());
}

#[test]
fn should_clear_action_buffer_on_reset() {
    let mut state = QuadcopterState::new(1, 3);

    // Fill buffer
    state.push_action_buffer(0, &[1.0, 1.0, 1.0, 1.0]);
    state.push_action_buffer(0, &[2.0, 2.0, 2.0, 2.0]);

    let config = InitConfig::fixed_start();
    state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

    // Buffer should be cleared
    for i in 0..12 {
        assert_eq!(
            state.action_buffer[i], 0.0,
            "Buffer should be cleared on reset"
        );
    }
    assert_eq!(state.action_buffer_idx[0], 0);
}

// ============================================================================
// Multi-Environment Isolation Tests
// ============================================================================

#[test]
fn should_isolate_state_between_environments() {
    let mut state = QuadcopterState::new(4, 0);

    // Set different states for each environment
    state.set_position(0, [1.0, 0.0, 0.0]);
    state.set_position(1, [0.0, 1.0, 0.0]);
    state.set_position(2, [0.0, 0.0, 1.0]);
    state.set_position(3, [1.0, 1.0, 1.0]);

    // Verify isolation
    assert_eq!(state.get_position(0), [1.0, 0.0, 0.0]);
    assert_eq!(state.get_position(1), [0.0, 1.0, 0.0]);
    assert_eq!(state.get_position(2), [0.0, 0.0, 1.0]);
    assert_eq!(state.get_position(3), [1.0, 1.0, 1.0]);
}

#[test]
fn should_reset_single_environment_without_affecting_others() {
    let mut state = QuadcopterState::new(4, 0);
    let config = InitConfig::fixed_start();

    // Set custom positions
    state.set_position(0, [1.0, 2.0, 3.0]);
    state.set_position(1, [4.0, 5.0, 6.0]);
    state.set_position(2, [7.0, 8.0, 9.0]);
    state.set_position(3, [10.0, 11.0, 12.0]);

    // Reset only environment 1
    state.reset_env(1, 42, &config, [0.0, 0.0, 1.0]);

    // Environment 1 should be reset
    assert_eq!(state.get_position(1), [0.0, 0.0, 1.0]);

    // Others should be unchanged
    assert_eq!(state.get_position(0), [1.0, 2.0, 3.0]);
    assert_eq!(state.get_position(2), [7.0, 8.0, 9.0]);
    assert_eq!(state.get_position(3), [10.0, 11.0, 12.0]);
}

// ============================================================================
// Quaternion Normalization on Reset Tests
// ============================================================================

#[test]
fn should_produce_normalized_quaternion_on_reset() {
    let mut state = QuadcopterState::new(1, 0);
    let config = InitConfig::default(); // Uses uniform attitude distribution

    for seed in 0..100 {
        state.reset_env(0, seed, &config, [0.0, 0.0, 1.0]);

        let q = state.get_quaternion(0);
        let norm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();

        assert!(
            (norm - 1.0).abs() < 1e-5,
            "Quaternion should be normalized on reset: norm={} for seed {}",
            norm,
            seed
        );
    }
}
