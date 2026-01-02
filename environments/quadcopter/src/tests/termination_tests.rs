//! Termination condition tests defining the complete behavior of episode endings.
//!
//! These tests define:
//! - Ground collision detection
//! - Position bounds checking
//! - Attitude (roll/pitch) bounds checking
//! - Velocity bounds checking
//! - Time limit truncation
//! - Distinction between terminal (failure) and truncated (time limit)

use crate::config::{InitConfig, TerminationConfig};
use crate::physics::quaternion::euler_to_quat;
use crate::state::QuadcopterState;
use crate::termination::{check_terminal, check_termination, check_truncated, TerminationResult};
use std::f32::consts::PI;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_state_with_position(pos: [f32; 3]) -> QuadcopterState {
    let mut state = QuadcopterState::new(1, 0);
    state.set_position(0, pos);
    state.set_quaternion(0, [1.0, 0.0, 0.0, 0.0]); // Level
    state
}

fn create_state_with_attitude(roll: f32, pitch: f32, yaw: f32) -> QuadcopterState {
    let mut state = QuadcopterState::new(1, 0);
    state.set_position(0, [0.0, 0.0, 1.0]);
    let quat = euler_to_quat([roll, pitch, yaw]);
    state.set_quaternion(0, quat);
    state
}

fn create_state_with_velocity(vel: [f32; 3]) -> QuadcopterState {
    let mut state = QuadcopterState::new(1, 0);
    state.set_position(0, [0.0, 0.0, 1.0]);
    state.set_quaternion(0, [1.0, 0.0, 0.0, 0.0]);
    state.set_velocity(0, vel);
    state
}

// ============================================================================
// Ground Collision Tests
// ============================================================================

#[test]
fn should_terminate_when_below_ground() {
    let state = create_state_with_position([0.0, 0.0, -0.1]);
    let config = TerminationConfig::default();

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when z < 0");
}

#[test]
fn should_terminate_at_exactly_zero_height() {
    // z < 0 is the condition, so z = 0 should NOT terminate
    let state = create_state_with_position([0.0, 0.0, 0.0]);
    let config = TerminationConfig::default();

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate at z = 0 (boundary)");
}

#[test]
fn should_not_terminate_above_ground() {
    let state = create_state_with_position([0.0, 0.0, 0.01]);
    let config = TerminationConfig::default();

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate when z > 0");
}

#[test]
fn should_not_terminate_with_ground_collision_disabled() {
    // Position at z=-0.05, which is below ground (z<0)
    // but above the default position_bounds z_min (0.0)
    // We need to disable BOTH ground collision AND position bounds
    // to test ground collision logic specifically
    let state = create_state_with_position([0.0, 0.0, -0.05]);
    let config = TerminationConfig::default()
        .with_ground_collision(false)
        .without_position_bounds(); // Also disable position bounds

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate with ground collision disabled and no position bounds");
}

#[test]
fn should_mark_ground_collision_as_terminal_not_truncated() {
    let state = create_state_with_position([0.0, 0.0, -0.1]);
    let config = TerminationConfig::default();

    let result = check_termination(&state, 0, &config);

    assert!(result.terminal, "Ground collision should be terminal");
    assert!(!result.truncated, "Ground collision should not be truncated");
}

// ============================================================================
// Position Bounds Tests
// ============================================================================

#[test]
fn should_terminate_when_x_below_min() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];
    let state = create_state_with_position([-2.1, 0.0, 1.0]);
    let config = TerminationConfig::default().with_position_bounds(bounds);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when x < x_min");
}

#[test]
fn should_terminate_when_x_above_max() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];
    let state = create_state_with_position([2.1, 0.0, 1.0]);
    let config = TerminationConfig::default().with_position_bounds(bounds);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when x > x_max");
}

#[test]
fn should_terminate_when_y_below_min() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];
    let state = create_state_with_position([0.0, -2.1, 1.0]);
    let config = TerminationConfig::default().with_position_bounds(bounds);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when y < y_min");
}

#[test]
fn should_terminate_when_y_above_max() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];
    let state = create_state_with_position([0.0, 2.1, 1.0]);
    let config = TerminationConfig::default().with_position_bounds(bounds);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when y > y_max");
}

#[test]
fn should_terminate_when_z_below_min_via_bounds() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.5, 3.0];
    let mut state = create_state_with_position([0.0, 0.0, 0.3]);
    // Disable ground collision to test bounds only
    let config = TerminationConfig::default()
        .with_position_bounds(bounds)
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when z < z_min bounds");
}

#[test]
fn should_terminate_when_z_above_max() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];
    let state = create_state_with_position([0.0, 0.0, 3.1]);
    let config = TerminationConfig::default().with_position_bounds(bounds);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when z > z_max");
}

#[test]
fn should_not_terminate_within_position_bounds() {
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];
    let state = create_state_with_position([1.5, -1.5, 2.0]);
    let config = TerminationConfig::default().with_position_bounds(bounds);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate within bounds");
}

#[test]
fn should_not_terminate_at_position_boundaries() {
    // Exactly at boundary should be within bounds (using <= and >=)
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.0, 3.0];

    // Test each boundary
    let test_positions = [
        [-2.0, 0.0, 1.0], // x at min
        [2.0, 0.0, 1.0],  // x at max
        [0.0, -2.0, 1.0], // y at min
        [0.0, 2.0, 1.0],  // y at max
        [0.0, 0.0, 0.0],  // z at min (but ground collision will trigger)
        [0.0, 0.0, 3.0],  // z at max
    ];

    let config = TerminationConfig::default()
        .with_position_bounds(bounds)
        .with_ground_collision(false); // Disable to test bounds only

    for pos in &test_positions {
        let state = create_state_with_position(*pos);
        let terminal = check_terminal(&state, 0, &config);

        assert!(
            !terminal,
            "Should not terminate at boundary position {:?}",
            pos
        );
    }
}

#[test]
fn should_not_check_position_bounds_when_disabled() {
    let state = create_state_with_position([100.0, 100.0, 100.0]);
    let config = TerminationConfig::default()
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate with bounds disabled");
}

// ============================================================================
// Attitude Bounds Tests
// ============================================================================

#[test]
fn should_terminate_when_roll_exceeds_bounds() {
    let max_angle = 0.5; // ~28 degrees
    let state = create_state_with_attitude(0.6, 0.0, 0.0);
    let config = TerminationConfig::default().with_attitude_bounds(max_angle);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when |roll| > max_angle");
}

#[test]
fn should_terminate_when_negative_roll_exceeds_bounds() {
    let max_angle = 0.5;
    let state = create_state_with_attitude(-0.6, 0.0, 0.0);
    let config = TerminationConfig::default().with_attitude_bounds(max_angle);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when roll < -max_angle");
}

#[test]
fn should_terminate_when_pitch_exceeds_bounds() {
    let max_angle = 0.5;
    let state = create_state_with_attitude(0.0, 0.6, 0.0);
    let config = TerminationConfig::default().with_attitude_bounds(max_angle);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when |pitch| > max_angle");
}

#[test]
fn should_terminate_when_negative_pitch_exceeds_bounds() {
    let max_angle = 0.5;
    let state = create_state_with_attitude(0.0, -0.6, 0.0);
    let config = TerminationConfig::default().with_attitude_bounds(max_angle);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when pitch < -max_angle");
}

#[test]
fn should_not_terminate_for_yaw_regardless_of_magnitude() {
    let max_angle = 0.5;
    let state = create_state_with_attitude(0.0, 0.0, PI); // 180 degree yaw
    let config = TerminationConfig::default()
        .with_attitude_bounds(max_angle)
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Yaw should not trigger termination");
}

#[test]
fn should_not_terminate_within_attitude_bounds() {
    let max_angle = 0.8; // ~45 degrees
    let state = create_state_with_attitude(0.7, -0.7, 0.0);
    let config = TerminationConfig::default()
        .with_attitude_bounds(max_angle)
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate within attitude bounds");
}

#[test]
fn should_not_check_attitude_when_disabled() {
    let state = create_state_with_attitude(PI / 2.0, 0.0, 0.0); // 90 degree roll
    let config = TerminationConfig::default()
        .without_attitude_bounds()
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate with attitude bounds disabled");
}

#[test]
fn should_terminate_for_full_flip() {
    let max_angle = 0.8; // ~45 degrees
    // 90 degree roll = flipped on side
    let state = create_state_with_attitude(PI / 2.0, 0.0, 0.0);
    let config = TerminationConfig::default().with_attitude_bounds(max_angle);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Full flip should terminate");
}

// ============================================================================
// Velocity Bounds Tests
// ============================================================================

#[test]
fn should_terminate_when_velocity_exceeds_bounds() {
    let max_vel = 5.0;
    let state = create_state_with_velocity([6.0, 0.0, 0.0]);
    let config = TerminationConfig::default()
        .with_velocity_bounds(max_vel)
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should terminate when |vel| > max_vel");
}

#[test]
fn should_check_velocity_magnitude_not_components() {
    let max_vel = 5.0;
    // Each component < 5, but magnitude = sqrt(3^2 + 3^2 + 3^2) = 5.19 > 5
    let state = create_state_with_velocity([3.0, 3.0, 3.0]);
    let config = TerminationConfig::default()
        .with_velocity_bounds(max_vel)
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Should check velocity magnitude, not components");
}

#[test]
fn should_not_terminate_within_velocity_bounds() {
    let max_vel = 5.0;
    let state = create_state_with_velocity([2.0, 2.0, 2.0]); // magnitude = 3.46
    let config = TerminationConfig::default()
        .with_velocity_bounds(max_vel)
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate within velocity bounds");
}

#[test]
fn should_not_check_velocity_when_disabled() {
    let state = create_state_with_velocity([100.0, 100.0, 100.0]);
    let config = TerminationConfig::default()
        .without_velocity_bounds()
        .without_position_bounds()
        .with_ground_collision(false);

    let terminal = check_terminal(&state, 0, &config);

    assert!(!terminal, "Should not terminate with velocity bounds disabled");
}

// ============================================================================
// Truncation Tests
// ============================================================================

#[test]
fn should_truncate_at_max_steps() {
    let mut state = QuadcopterState::new(1, 0);
    state.step_count[0] = 240;

    let config = TerminationConfig::default().with_max_steps(240);

    let truncated = check_truncated(&state, 0, &config);

    assert!(truncated, "Should truncate at max_steps");
}

#[test]
fn should_truncate_above_max_steps() {
    let mut state = QuadcopterState::new(1, 0);
    state.step_count[0] = 300;

    let config = TerminationConfig::default().with_max_steps(240);

    let truncated = check_truncated(&state, 0, &config);

    assert!(truncated, "Should truncate above max_steps");
}

#[test]
fn should_not_truncate_below_max_steps() {
    let mut state = QuadcopterState::new(1, 0);
    state.step_count[0] = 239;

    let config = TerminationConfig::default().with_max_steps(240);

    let truncated = check_truncated(&state, 0, &config);

    assert!(!truncated, "Should not truncate below max_steps");
}

#[test]
fn should_not_truncate_at_step_zero() {
    let state = QuadcopterState::new(1, 0);
    let config = TerminationConfig::default().with_max_steps(240);

    let truncated = check_truncated(&state, 0, &config);

    assert!(!truncated, "Should not truncate at step 0");
}

#[test]
fn should_mark_truncation_as_truncated_not_terminal() {
    let mut state = QuadcopterState::new(1, 0);
    state.set_position(0, [0.0, 0.0, 1.0]); // Valid position
    state.step_count[0] = 300;

    let config = TerminationConfig::default()
        .with_max_steps(240)
        .without_position_bounds()
        .without_attitude_bounds()
        .with_ground_collision(false);

    let result = check_termination(&state, 0, &config);

    assert!(!result.terminal, "Time limit should not be terminal");
    assert!(result.truncated, "Time limit should be truncated");
}

// ============================================================================
// Combined Termination Tests
// ============================================================================

#[test]
fn should_detect_both_terminal_and_truncated() {
    let mut state = QuadcopterState::new(1, 0);
    state.set_position(0, [0.0, 0.0, -0.1]); // Ground collision
    state.step_count[0] = 300; // Also past max steps

    let config = TerminationConfig::default().with_max_steps(240);

    let result = check_termination(&state, 0, &config);

    assert!(result.terminal, "Should detect ground collision");
    assert!(result.truncated, "Should detect time limit");
}

#[test]
fn should_report_done_when_terminal() {
    let state = create_state_with_position([0.0, 0.0, -0.1]);
    let config = TerminationConfig::default();

    let result = check_termination(&state, 0, &config);

    assert!(result.done(), "done() should be true when terminal");
}

#[test]
fn should_report_done_when_truncated() {
    let mut state = QuadcopterState::new(1, 0);
    state.set_position(0, [0.0, 0.0, 1.0]);
    state.step_count[0] = 300;

    let config = TerminationConfig::default()
        .with_max_steps(240)
        .without_position_bounds()
        .without_attitude_bounds()
        .with_ground_collision(false);

    let result = check_termination(&state, 0, &config);

    assert!(result.done(), "done() should be true when truncated");
}

#[test]
fn should_not_report_done_when_neither() {
    let state = create_state_with_position([0.0, 0.0, 1.0]);
    let config = TerminationConfig::default()
        .without_position_bounds()
        .without_attitude_bounds()
        .with_ground_collision(false)
        .with_max_steps(1000);

    let result = check_termination(&state, 0, &config);

    assert!(!result.done(), "done() should be false when neither terminal nor truncated");
}

// ============================================================================
// Multiple Environment Termination Tests
// ============================================================================

#[test]
fn should_check_termination_independently_per_environment() {
    let mut state = QuadcopterState::new(4, 0);

    // Env 0: valid
    state.set_position(0, [0.0, 0.0, 1.0]);
    state.step_count[0] = 10;

    // Env 1: ground collision
    state.set_position(1, [0.0, 0.0, -0.1]);
    state.step_count[1] = 10;

    // Env 2: out of bounds
    state.set_position(2, [10.0, 0.0, 1.0]);
    state.step_count[2] = 10;

    // Env 3: truncated
    state.set_position(3, [0.0, 0.0, 1.0]);
    state.step_count[3] = 300;

    let config = TerminationConfig::default().with_max_steps(240);

    let results: Vec<TerminationResult> =
        (0..4).map(|idx| check_termination(&state, idx, &config)).collect();

    assert!(!results[0].terminal && !results[0].truncated, "Env 0 should be valid");
    assert!(results[1].terminal, "Env 1 should be terminal (ground)");
    assert!(results[2].terminal, "Env 2 should be terminal (bounds)");
    assert!(results[3].truncated && !results[3].terminal, "Env 3 should be truncated only");
}

// ============================================================================
// Configuration Priority Tests
// ============================================================================

#[test]
fn should_check_ground_collision_before_position_bounds() {
    // Ground collision is z < 0, position bounds might have z_min > 0
    let bounds = [-2.0, 2.0, -2.0, 2.0, 0.5, 3.0]; // z_min = 0.5
    let state = create_state_with_position([0.0, 0.0, -0.1]); // Below both thresholds

    let config = TerminationConfig::default()
        .with_position_bounds(bounds)
        .with_ground_collision(true);

    let terminal = check_terminal(&state, 0, &config);

    assert!(terminal, "Ground collision should trigger termination");
}

#[test]
fn should_allow_all_checks_to_be_disabled() {
    let mut state = QuadcopterState::new(1, 0);

    // Set everything to extreme values
    state.set_position(0, [1000.0, 1000.0, -1000.0]);
    state.set_quaternion(0, euler_to_quat([PI, PI / 2.0, 0.0])); // Fully flipped
    state.set_velocity(0, [1000.0, 1000.0, 1000.0]);
    state.step_count[0] = 100000;

    let config = TerminationConfig::default()
        .without_position_bounds()
        .without_attitude_bounds()
        .without_velocity_bounds()
        .with_ground_collision(false)
        .with_max_steps(u32::MAX);

    let result = check_termination(&state, 0, &config);

    assert!(!result.terminal, "No terminal with all checks disabled");
    assert!(!result.truncated, "No truncation with max steps at max");
}
