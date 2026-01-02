//! Reward computation tests defining the complete behavior of the reward function.
//!
//! These tests define:
//! - Position error penalty computation
//! - Velocity error penalty computation
//! - Attitude (roll/pitch) penalty computation
//! - Angular velocity penalty computation
//! - Action magnitude penalty computation
//! - Action rate penalty computation
//! - Alive bonus application
//! - Combined reward formula correctness

use crate::constants::*;
use crate::physics::quaternion::euler_to_quat;
use crate::reward::components::*;
use crate::reward::{compute_reward, presets, RewardComponent};
use crate::state::QuadcopterState;
use std::f32::consts::PI;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_basic_state() -> QuadcopterState {
    let mut state = QuadcopterState::new(1, 0);
    // Target at origin, height 1
    state.target_pos[0] = 0.0;
    state.target_pos[1] = 0.0;
    state.target_pos[2] = 1.0;
    // Target velocity = 0
    state.target_vel[0] = 0.0;
    state.target_vel[1] = 0.0;
    state.target_vel[2] = 0.0;
    state
}

fn set_position(state: &mut QuadcopterState, pos: [f32; 3]) {
    state.set_position(0, pos);
}

fn set_attitude(state: &mut QuadcopterState, roll: f32, pitch: f32, yaw: f32) {
    let quat = euler_to_quat([roll, pitch, yaw]);
    state.set_quaternion(0, quat);
}

fn set_velocity(state: &mut QuadcopterState, vel: [f32; 3]) {
    state.set_velocity(0, vel);
}

fn set_angular_velocity(state: &mut QuadcopterState, omega: [f32; 3]) {
    state.set_angular_velocity(0, omega);
}

fn set_rpms(state: &mut QuadcopterState, rpms: [f32; 4]) {
    for i in 0..4 {
        state.last_rpm[i] = rpms[i];
    }
}

fn set_prev_action(state: &mut QuadcopterState, actions: [f32; 4]) {
    for i in 0..4 {
        state.prev_action[i] = actions[i];
    }
}

// ============================================================================
// Alive Bonus Tests
// ============================================================================

#[test]
fn should_include_alive_bonus_in_reward() {
    let state = create_basic_state();
    let reward_fn = AliveBonus { bonus: 0.5 };

    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        (reward - 0.5).abs() < 0.01,
        "Reward should be alive bonus: {}",
        reward
    );
}

#[test]
fn should_have_zero_alive_bonus_when_zero() {
    let state = create_basic_state();
    let reward_fn = AliveBonus { bonus: 0.0 };

    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-6,
        "Reward should be 0 with no bonus: {}",
        reward
    );
}

#[test]
fn should_apply_alive_bonus_regardless_of_position() {
    let mut state = create_basic_state();
    set_position(&mut state, [5.0, 5.0, 5.0]); // Far from target

    let reward_fn = AliveBonus { bonus: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        (reward - 1.0).abs() < 1e-5,
        "Alive bonus should be 1.0: {}",
        reward
    );
}

// ============================================================================
// Position Error Penalty Tests
// ============================================================================

#[test]
fn should_have_zero_position_penalty_at_target() {
    let mut state = create_basic_state();
    set_position(&mut state, [0.0, 0.0, 1.0]); // At target

    let reward_fn = PositionError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-6,
        "Position penalty should be 0 at target: {}",
        reward
    );
}

#[test]
fn should_penalize_x_displacement() {
    let mut state = create_basic_state();
    set_position(&mut state, [1.0, 0.0, 1.0]); // 1m x error

    let reward_fn = PositionError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = weight * dx^2 = 1.0 * 1.0 = 1.0
    assert!(
        (reward + 1.0).abs() < 1e-5,
        "X displacement penalty should be -1.0: {}",
        reward
    );
}

#[test]
fn should_penalize_y_displacement() {
    let mut state = create_basic_state();
    set_position(&mut state, [0.0, 2.0, 1.0]); // 2m y error

    let reward_fn = PositionError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = weight * dy^2 = 1.0 * 4.0 = 4.0
    assert!(
        (reward + 4.0).abs() < 1e-5,
        "Y displacement penalty should be -4.0: {}",
        reward
    );
}

#[test]
fn should_penalize_z_displacement() {
    let mut state = create_basic_state();
    set_position(&mut state, [0.0, 0.0, 2.0]); // 1m z error (target at z=1)

    let reward_fn = PositionError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = weight * dz^2 = 1.0 * 1.0 = 1.0
    assert!(
        (reward + 1.0).abs() < 1e-5,
        "Z displacement penalty should be -1.0: {}",
        reward
    );
}

#[test]
fn should_compute_squared_euclidean_position_error() {
    let mut state = create_basic_state();
    set_position(&mut state, [1.0, 2.0, 3.0]); // Target at (0, 0, 1)
    // Error: (1, 2, 2), squared = 1 + 4 + 4 = 9

    let reward_fn = PositionError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        (reward + 9.0).abs() < 1e-5,
        "Squared Euclidean error should be -9.0: {}",
        reward
    );
}

#[test]
fn should_scale_position_penalty_by_weight() {
    let mut state = create_basic_state();
    set_position(&mut state, [1.0, 0.0, 1.0]); // 1m error

    let reward_w1 = compute_reward(&PositionError { weight: 1.0 }, &state, 0);
    let reward_w2 = compute_reward(&PositionError { weight: 2.0 }, &state, 0);

    assert!(
        (reward_w2 / reward_w1 - 2.0).abs() < 1e-5,
        "Penalty should scale with weight: {} vs {}",
        reward_w1,
        reward_w2
    );
}

#[test]
fn should_not_apply_position_penalty_when_weight_is_zero() {
    let mut state = create_basic_state();
    set_position(&mut state, [10.0, 10.0, 10.0]); // Far from target

    let reward_fn = PositionError { weight: 0.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-6,
        "No penalty with zero weight: {}",
        reward
    );
}

// ============================================================================
// Velocity Error Penalty Tests
// ============================================================================

#[test]
fn should_have_zero_velocity_penalty_at_target_velocity() {
    let mut state = create_basic_state();
    set_velocity(&mut state, [0.0, 0.0, 0.0]); // Target velocity is 0

    let reward_fn = VelocityError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-6,
        "Velocity penalty should be 0 at target: {}",
        reward
    );
}

#[test]
fn should_penalize_velocity_error() {
    let mut state = create_basic_state();
    set_velocity(&mut state, [1.0, 2.0, 3.0]); // Target is (0,0,0)

    let reward_fn = VelocityError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = 1^2 + 2^2 + 3^2 = 14
    assert!(
        (reward + 14.0).abs() < 1e-5,
        "Velocity error penalty should be -14.0: {}",
        reward
    );
}

#[test]
fn should_compute_velocity_error_relative_to_target() {
    let mut state = create_basic_state();
    // Set target velocity
    state.target_vel[0] = 1.0;
    state.target_vel[1] = 1.0;
    state.target_vel[2] = 0.0;
    // Set actual velocity same as target
    set_velocity(&mut state, [1.0, 1.0, 0.0]);

    let reward_fn = VelocityError { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-6,
        "No penalty when at target velocity: {}",
        reward
    );
}

#[test]
fn should_scale_velocity_penalty_by_weight() {
    let mut state = create_basic_state();
    set_velocity(&mut state, [1.0, 0.0, 0.0]);

    let reward_w1 = compute_reward(&VelocityError { weight: 0.5 }, &state, 0);
    let reward_w2 = compute_reward(&VelocityError { weight: 1.0 }, &state, 0);

    assert!(
        (reward_w2 / reward_w1 - 2.0).abs() < 1e-5,
        "Velocity penalty should scale with weight"
    );
}

// ============================================================================
// Attitude Penalty Tests
// ============================================================================

#[test]
fn should_have_zero_attitude_penalty_when_level() {
    let mut state = create_basic_state();
    set_attitude(&mut state, 0.0, 0.0, 0.0); // Level

    let reward_fn = AttitudePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-5,
        "Attitude penalty should be 0 when level: {}",
        reward
    );
}

#[test]
fn should_penalize_roll() {
    let mut state = create_basic_state();
    let roll = 0.3; // ~17 degrees
    set_attitude(&mut state, roll, 0.0, 0.0);

    let reward_fn = AttitudePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = roll^2 = 0.09
    assert!(
        (reward + 0.09).abs() < 0.01,
        "Roll penalty should be ~-0.09: {}",
        reward
    );
}

#[test]
fn should_penalize_pitch() {
    let mut state = create_basic_state();
    let pitch = 0.2; // ~11 degrees
    set_attitude(&mut state, 0.0, pitch, 0.0);

    let reward_fn = AttitudePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = pitch^2 = 0.04
    assert!(
        (reward + 0.04).abs() < 0.01,
        "Pitch penalty should be ~-0.04: {}",
        reward
    );
}

#[test]
fn should_not_penalize_yaw() {
    let mut state = create_basic_state();
    set_attitude(&mut state, 0.0, 0.0, PI); // 180 degree yaw

    let reward_fn = AttitudePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // Yaw should not be penalized
    assert!(
        reward.abs() < 0.01,
        "Yaw should not be penalized: {}",
        reward
    );
}

#[test]
fn should_combine_roll_and_pitch_penalties() {
    let mut state = create_basic_state();
    set_attitude(&mut state, 0.2, 0.3, 0.0);

    let reward_fn = AttitudePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = roll^2 + pitch^2 = 0.04 + 0.09 = 0.13
    assert!(
        (reward + 0.13).abs() < 0.02,
        "Combined attitude penalty should be ~-0.13: {}",
        reward
    );
}

#[test]
fn should_scale_attitude_penalty_by_weight() {
    let mut state = create_basic_state();
    set_attitude(&mut state, 0.5, 0.0, 0.0);

    let reward_w1 = compute_reward(&AttitudePenalty { weight: 1.0 }, &state, 0);
    let reward_w2 = compute_reward(&AttitudePenalty { weight: 0.5 }, &state, 0);

    assert!(
        (reward_w1 / reward_w2 - 2.0).abs() < 0.1,
        "Attitude penalty should scale with weight"
    );
}

// ============================================================================
// Angular Velocity Penalty Tests
// ============================================================================

#[test]
fn should_have_zero_angular_velocity_penalty_at_rest() {
    let mut state = create_basic_state();
    set_angular_velocity(&mut state, [0.0, 0.0, 0.0]);

    let reward_fn = AngularVelocityPenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-6,
        "Angular velocity penalty should be 0 at rest: {}",
        reward
    );
}

#[test]
fn should_penalize_roll_rate() {
    let mut state = create_basic_state();
    set_angular_velocity(&mut state, [2.0, 0.0, 0.0]);

    let reward_fn = AngularVelocityPenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = wx^2 = 4.0
    assert!(
        (reward + 4.0).abs() < 1e-5,
        "Roll rate penalty should be -4.0: {}",
        reward
    );
}

#[test]
fn should_penalize_pitch_rate() {
    let mut state = create_basic_state();
    set_angular_velocity(&mut state, [0.0, 1.5, 0.0]);

    let reward_fn = AngularVelocityPenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = wy^2 = 2.25
    assert!(
        (reward + 2.25).abs() < 1e-5,
        "Pitch rate penalty should be -2.25: {}",
        reward
    );
}

#[test]
fn should_penalize_yaw_rate() {
    let mut state = create_basic_state();
    set_angular_velocity(&mut state, [0.0, 0.0, 3.0]);

    let reward_fn = AngularVelocityPenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = wz^2 = 9.0
    assert!(
        (reward + 9.0).abs() < 1e-5,
        "Yaw rate penalty should be -9.0: {}",
        reward
    );
}

#[test]
fn should_compute_squared_magnitude_of_angular_velocity() {
    let mut state = create_basic_state();
    set_angular_velocity(&mut state, [1.0, 2.0, 3.0]);

    let reward_fn = AngularVelocityPenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    // penalty = 1 + 4 + 9 = 14
    assert!(
        (reward + 14.0).abs() < 1e-5,
        "Angular velocity magnitude^2 should be -14.0: {}",
        reward
    );
}

// ============================================================================
// Action Magnitude Penalty Tests
// ============================================================================

#[test]
fn should_have_zero_action_penalty_at_hover() {
    let mut state = create_basic_state();
    set_rpms(&mut state, [HOVER_RPM; 4]);

    let reward_fn = ActionMagnitudePenalty {
        weight: 1.0,
        reference_rpm: HOVER_RPM,
    };
    let reward = compute_reward(&reward_fn, &state, 0);

    // At hover RPM, deviation from hover is 0
    assert!(
        reward.abs() < 1e-5,
        "Action penalty should be 0 at hover: {}",
        reward
    );
}

#[test]
fn should_penalize_deviation_from_hover() {
    let mut state = create_basic_state();
    set_rpms(&mut state, [HOVER_RPM * 1.2; 4]); // 20% above hover

    let reward_fn = ActionMagnitudePenalty {
        weight: 1.0,
        reference_rpm: HOVER_RPM,
    };
    let reward = compute_reward(&reward_fn, &state, 0);

    // Should be negative (penalty)
    assert!(reward < 0.0, "Should penalize non-hover actions: {}", reward);
}

#[test]
fn should_penalize_both_high_and_low_rpms() {
    let mut state = create_basic_state();
    let reward_fn = ActionMagnitudePenalty {
        weight: 1.0,
        reference_rpm: HOVER_RPM,
    };

    // High RPM
    set_rpms(&mut state, [MAX_RPM; 4]);
    let reward_high = compute_reward(&reward_fn, &state, 0);

    // Low RPM
    set_rpms(&mut state, [0.0; 4]);
    let reward_low = compute_reward(&reward_fn, &state, 0);

    assert!(reward_high < 0.0, "High RPM should be penalized");
    assert!(reward_low < 0.0, "Low RPM should be penalized");
}

#[test]
fn should_scale_action_penalty_by_weight() {
    let mut state = create_basic_state();
    set_rpms(&mut state, [MAX_RPM; 4]);

    let reward_w1 = compute_reward(
        &ActionMagnitudePenalty {
            weight: 0.001,
            reference_rpm: HOVER_RPM,
        },
        &state,
        0,
    );
    let reward_w2 = compute_reward(
        &ActionMagnitudePenalty {
            weight: 0.002,
            reference_rpm: HOVER_RPM,
        },
        &state,
        0,
    );

    assert!(
        (reward_w2 / reward_w1 - 2.0).abs() < 0.1,
        "Action penalty should scale with weight"
    );
}

// ============================================================================
// Action Rate Penalty Tests
// ============================================================================

#[test]
fn should_have_zero_action_rate_penalty_with_no_change() {
    let mut state = create_basic_state();
    let action = rpm_to_action(HOVER_RPM);
    set_rpms(&mut state, [HOVER_RPM; 4]);
    set_prev_action(&mut state, [action; 4]);

    let reward_fn = ActionRatePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward.abs() < 1e-5,
        "Action rate penalty should be 0 with no change: {}",
        reward
    );
}

#[test]
fn should_penalize_action_change() {
    let mut state = create_basic_state();
    set_rpms(&mut state, [MAX_RPM; 4]); // Current action = normalized MAX
    set_prev_action(&mut state, [rpm_to_action(HOVER_RPM); 4]); // Previous = hover

    let reward_fn = ActionRatePenalty { weight: 1.0 };
    let reward = compute_reward(&reward_fn, &state, 0);

    assert!(reward < 0.0, "Action rate should be penalized: {}", reward);
}

#[test]
fn should_penalize_larger_action_changes_more() {
    let mut state = create_basic_state();
    let hover_action = rpm_to_action(HOVER_RPM);
    let reward_fn = ActionRatePenalty { weight: 1.0 };

    // Small change
    set_rpms(&mut state, [HOVER_RPM * 1.1; 4]);
    set_prev_action(&mut state, [hover_action; 4]);
    let reward_small = compute_reward(&reward_fn, &state, 0);

    // Large change
    set_rpms(&mut state, [MAX_RPM; 4]);
    set_prev_action(&mut state, [hover_action; 4]);
    let reward_large = compute_reward(&reward_fn, &state, 0);

    assert!(
        reward_large < reward_small,
        "Larger changes should be penalized more: {} vs {}",
        reward_large,
        reward_small
    );
}

// ============================================================================
// Combined Reward Tests (Tuple Composition)
// ============================================================================

#[test]
fn should_combine_all_reward_components() {
    let mut state = create_basic_state();

    // Set non-zero values for all components
    set_position(&mut state, [1.0, 0.0, 1.0]); // 1m x error
    set_velocity(&mut state, [1.0, 0.0, 0.0]); // 1 m/s velocity error
    set_attitude(&mut state, 0.1, 0.0, 0.0); // 0.1 rad roll
    set_angular_velocity(&mut state, [1.0, 0.0, 0.0]); // 1 rad/s roll rate
    set_rpms(&mut state, [HOVER_RPM * 1.5; 4]); // Non-hover RPM

    // Use the hover preset (7-tuple of all components)
    let reward_fn = presets::hover();
    let reward = compute_reward(&reward_fn, &state, 0);

    // reward = alive_bonus - pos_penalty - vel_penalty - att_penalty - ang_penalty - action_penalty
    // Should be negative overall due to penalties
    assert!(
        reward.is_finite(),
        "Combined reward should be finite: {}",
        reward
    );
}

#[test]
fn should_compose_two_components_via_tuple() {
    let state = create_basic_state();

    let pos_only = PositionError { weight: 1.0 };
    let bonus_only = AliveBonus { bonus: 0.5 };
    let combined = (pos_only, bonus_only);

    let reward = compute_reward(&combined, &state, 0);

    // At target position: pos_penalty = 0, alive_bonus = 0.5
    assert!(
        (reward - 0.5).abs() < 1e-5,
        "Combined should be alive bonus: {}",
        reward
    );
}

#[test]
fn should_match_hover_preset_formula() {
    let state = create_basic_state();
    let reward_fn = presets::hover();

    let reward = compute_reward(&reward_fn, &state, 0);

    // At target position with level attitude and zero velocity, reward should be positive
    assert!(
        reward > 0.0,
        "Hover at target should have positive reward: {}",
        reward
    );
}

#[test]
fn should_match_tracking_preset_formula() {
    let state = create_basic_state();
    let reward_fn = presets::tracking();

    let reward = compute_reward(&reward_fn, &state, 0);

    // At target position with level attitude and zero velocity
    assert!(
        reward > 0.0,
        "Tracking at target should have positive reward: {}",
        reward
    );
}

// ============================================================================
// Reward Symmetry Tests
// ============================================================================

#[test]
fn should_have_symmetric_position_penalty() {
    let reward_fn = PositionError { weight: 1.0 };
    let mut state = create_basic_state();

    set_position(&mut state, [1.0, 0.0, 1.0]);
    let reward_pos = compute_reward(&reward_fn, &state, 0);

    set_position(&mut state, [-1.0, 0.0, 1.0]);
    let reward_neg = compute_reward(&reward_fn, &state, 0);

    assert!(
        (reward_pos - reward_neg).abs() < 1e-5,
        "Position penalty should be symmetric: {} vs {}",
        reward_pos,
        reward_neg
    );
}

#[test]
fn should_have_symmetric_velocity_penalty() {
    let reward_fn = VelocityError { weight: 1.0 };
    let mut state = create_basic_state();

    set_velocity(&mut state, [2.0, 0.0, 0.0]);
    let reward_pos = compute_reward(&reward_fn, &state, 0);

    set_velocity(&mut state, [-2.0, 0.0, 0.0]);
    let reward_neg = compute_reward(&reward_fn, &state, 0);

    assert!(
        (reward_pos - reward_neg).abs() < 1e-5,
        "Velocity penalty should be symmetric: {} vs {}",
        reward_pos,
        reward_neg
    );
}

#[test]
fn should_have_symmetric_attitude_penalty() {
    let reward_fn = AttitudePenalty { weight: 1.0 };
    let mut state = create_basic_state();

    set_attitude(&mut state, 0.3, 0.0, 0.0);
    let reward_pos = compute_reward(&reward_fn, &state, 0);

    set_attitude(&mut state, -0.3, 0.0, 0.0);
    let reward_neg = compute_reward(&reward_fn, &state, 0);

    assert!(
        (reward_pos - reward_neg).abs() < 0.01,
        "Attitude penalty should be symmetric: {} vs {}",
        reward_pos,
        reward_neg
    );
}

// ============================================================================
// Multiple Environment Tests
// ============================================================================

#[test]
fn should_compute_independent_rewards_per_environment() {
    let mut state = QuadcopterState::new(4, 0);

    // Set up targets and positions
    for idx in 0..4 {
        let base = idx * 3;
        state.target_pos[base] = 0.0;
        state.target_pos[base + 1] = 0.0;
        state.target_pos[base + 2] = 1.0;
    }

    // Env 0: at target
    state.set_position(0, [0.0, 0.0, 1.0]);

    // Env 1: 1m away
    state.set_position(1, [1.0, 0.0, 1.0]);

    // Env 2: 2m away
    state.set_position(2, [2.0, 0.0, 1.0]);

    // Env 3: 3m away
    state.set_position(3, [3.0, 0.0, 1.0]);

    let reward_fn = PositionError { weight: 1.0 };
    let rewards: Vec<f32> = (0..4)
        .map(|idx| compute_reward(&reward_fn, &state, idx))
        .collect();

    // Rewards should decrease with distance
    assert!(rewards[0] > rewards[1], "Closer should have higher reward");
    assert!(rewards[1] > rewards[2], "Closer should have higher reward");
    assert!(rewards[2] > rewards[3], "Closer should have higher reward");

    // Check specific values
    assert!((rewards[0] - 0.0).abs() < 1e-5, "At target: {}", rewards[0]);
    assert!(
        (rewards[1] - (-1.0)).abs() < 1e-5,
        "1m away: {}",
        rewards[1]
    );
    assert!(
        (rewards[2] - (-4.0)).abs() < 1e-5,
        "2m away: {}",
        rewards[2]
    );
    assert!(
        (rewards[3] - (-9.0)).abs() < 1e-5,
        "3m away: {}",
        rewards[3]
    );
}
