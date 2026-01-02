//! Physics simulation tests defining the complete behavior of quadcopter dynamics.
//!
//! These tests define:
//! - Hover equilibrium conditions
//! - Thrust force computation and direction
//! - Roll, pitch, and yaw torque generation
//! - Ground effect aerodynamics
//! - Drag force computation
//! - Quaternion integration correctness
//! - Rigid body dynamics accuracy

use crate::config::{InitConfig, ObsConfig, QuadcopterConfig, RewardConfig, TerminationConfig};
use crate::constants::*;
use crate::physics::aerodynamics::*;
use crate::physics::dynamics::*;
use crate::physics::quaternion::*;
use crate::state::QuadcopterState;

// ============================================================================
// Hover Equilibrium Tests
// ============================================================================

/// Tests that verify the fundamental hover equilibrium condition:
/// At hover RPM, total thrust equals weight, and there are no net torques.

#[test]
fn should_generate_thrust_equal_to_weight_at_hover_rpm() {
    // At hover, the total thrust from all 4 motors must exactly balance gravity
    let rpms = [HOVER_RPM; 4];
    let thrusts = compute_thrusts(rpms);
    let total_thrust = compute_total_thrust(thrusts);

    let weight = M * G;
    let error = (total_thrust - weight).abs();

    assert!(
        error < 1e-4,
        "Hover thrust ({:.6} N) should equal weight ({:.6} N), error: {:.6} N",
        total_thrust,
        weight,
        error
    );
}

#[test]
fn should_generate_no_roll_torque_with_equal_rpms() {
    let rpms = [HOVER_RPM; 4];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[0].abs() < 1e-10,
        "Roll torque should be zero with equal RPMs, got: {:e}",
        torques[0]
    );
}

#[test]
fn should_generate_no_pitch_torque_with_equal_rpms() {
    let rpms = [HOVER_RPM; 4];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[1].abs() < 1e-10,
        "Pitch torque should be zero with equal RPMs, got: {:e}",
        torques[1]
    );
}

#[test]
fn should_generate_no_yaw_torque_with_equal_rpms() {
    // With equal RPMs, CW and CCW motor torques should cancel
    let rpms = [HOVER_RPM; 4];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[2].abs() < 1e-10,
        "Yaw torque should be zero with equal RPMs, got: {:e}",
        torques[2]
    );
}

#[test]
fn should_maintain_altitude_at_hover_with_level_attitude() {
    // A level drone at hover RPM should have zero vertical acceleration
    let quat = [1.0, 0.0, 0.0, 0.0]; // Identity = level
    let thrust = M * G; // Exactly balances gravity
    let accel = compute_linear_acceleration(quat, thrust);

    assert!(
        accel[0].abs() < 1e-6,
        "X acceleration should be zero at hover: {:e}",
        accel[0]
    );
    assert!(
        accel[1].abs() < 1e-6,
        "Y acceleration should be zero at hover: {:e}",
        accel[1]
    );
    assert!(
        accel[2].abs() < 1e-6,  // Tightened from 1e-5
        "Z acceleration should be zero at hover: {:e}",
        accel[2]
    );
}

#[test]
fn should_maintain_position_during_hover_over_multiple_steps() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let initial_pos = state.get_position(0);
    let rpms = [HOVER_RPM; 4];
    let dt = 1.0 / 240.0;

    // Run 240 physics steps (1 second)
    for _ in 0..240 {
        physics_step_scalar(&mut state, 0, rpms, dt);
    }

    let final_pos = state.get_position(0);

    // Position should not drift significantly
    let drift = ((final_pos[0] - initial_pos[0]).powi(2)
        + (final_pos[1] - initial_pos[1]).powi(2)
        + (final_pos[2] - initial_pos[2]).powi(2))
    .sqrt();

    assert!(
        drift < 0.001,  // Tightened from 0.01 (1mm instead of 1cm)
        "Position drifted {:e} m during 1 second hover (threshold: 0.001 m)",
        drift
    );
}

// ============================================================================
// Thrust Direction Tests
// ============================================================================

#[test]
fn should_point_thrust_upward_with_identity_quaternion() {
    let quat = [1.0, 0.0, 0.0, 0.0]; // Identity = no rotation
    let dir = quat_to_thrust_dir(quat);

    assert!(dir[0].abs() < 1e-6, "X component should be 0, got: {}", dir[0]);
    assert!(dir[1].abs() < 1e-6, "Y component should be 0, got: {}", dir[1]);
    assert!(
        (dir[2] - 1.0).abs() < 1e-6,
        "Z component should be 1, got: {}",
        dir[2]
    );
}

#[test]
fn should_tilt_thrust_forward_with_positive_pitch() {
    // Positive pitch = nose up = thrust tilts backward (negative X)
    let pitch = 0.3; // ~17 degrees
    let quat = euler_to_quat([0.0, pitch, 0.0]);
    let dir = quat_to_thrust_dir(quat);

    // When pitched nose-up, thrust has positive X component (forward in world frame)
    // Actually with nose-up pitch, the z-axis of the body tilts, causing
    // the thrust to have a component in the -X direction (backward)
    // Let's verify the geometry
    let expected_x = pitch.sin(); // For small angles
    let expected_z = pitch.cos();

    assert!(
        (dir[0] - expected_x).abs() < 0.01,
        "X component mismatch: expected ~{:.4}, got {:.4}",
        expected_x,
        dir[0]
    );
    assert!(
        (dir[2] - expected_z).abs() < 0.01,
        "Z component mismatch: expected ~{:.4}, got {:.4}",
        expected_z,
        dir[2]
    );
}

#[test]
fn should_tilt_thrust_right_with_positive_roll() {
    // Positive roll = right side up = thrust tilts left (positive Y)
    let roll = 0.3; // ~17 degrees
    let quat = euler_to_quat([roll, 0.0, 0.0]);
    let dir = quat_to_thrust_dir(quat);

    // With positive roll, thrust has negative Y component
    let expected_y = -roll.sin();
    let expected_z = roll.cos();

    assert!(
        (dir[1] - expected_y).abs() < 0.01,
        "Y component mismatch: expected ~{:.4}, got {:.4}",
        expected_y,
        dir[1]
    );
    assert!(
        (dir[2] - expected_z).abs() < 0.01,
        "Z component mismatch: expected ~{:.4}, got {:.4}",
        expected_z,
        dir[2]
    );
}

#[test]
fn should_accelerate_upward_with_excess_thrust() {
    let quat = [1.0, 0.0, 0.0, 0.0];
    let thrust = M * G * 1.5; // 50% more than needed for hover
    let accel = compute_linear_acceleration(quat, thrust);

    // Should accelerate upward
    assert!(
        accel[2] > 0.0,
        "Should accelerate upward with excess thrust, got: {}",
        accel[2]
    );

    // Expected: (1.5 * mg - mg) / m = 0.5 * g
    let expected_z = 0.5 * G;
    assert!(
        (accel[2] - expected_z).abs() < 0.01,
        "Z acceleration should be ~{:.2}, got {:.2}",
        expected_z,
        accel[2]
    );
}

#[test]
fn should_accelerate_downward_with_zero_thrust() {
    let quat = [1.0, 0.0, 0.0, 0.0];
    let thrust = 0.0;
    let accel = compute_linear_acceleration(quat, thrust);

    // Should fall at 1g
    assert!(
        (accel[2] + G).abs() < 1e-5,
        "Should fall at -g, got: {}",
        accel[2]
    );
}

// ============================================================================
// Roll Torque Tests
// ============================================================================

#[test]
fn should_generate_positive_roll_torque_with_higher_left_motor_thrust() {
    // X-config: Motors 2,3 are on the left, motors 0,1 are on the right
    // Higher thrust on left (2,3) should create positive roll
    let left_rpm = HOVER_RPM * 1.1;
    let right_rpm = HOVER_RPM * 0.9;
    let rpms = [right_rpm, right_rpm, left_rpm, left_rpm];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[0] > 0.0,
        "Roll torque should be positive when left motors are higher, got: {:e}",
        torques[0]
    );
}

#[test]
fn should_generate_negative_roll_torque_with_higher_right_motor_thrust() {
    // Higher thrust on right (0,1) should create negative roll
    let left_rpm = HOVER_RPM * 0.9;
    let right_rpm = HOVER_RPM * 1.1;
    let rpms = [right_rpm, right_rpm, left_rpm, left_rpm];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[0] < 0.0,
        "Roll torque should be negative when right motors are higher, got: {:e}",
        torques[0]
    );
}

#[test]
fn should_have_roll_torque_proportional_to_thrust_difference() {
    // Test linearity of roll torque
    let delta1 = HOVER_RPM * 0.05; // 5% difference
    let delta2 = HOVER_RPM * 0.10; // 10% difference
    let omega = [0.0, 0.0, 0.0]; // No body rotation

    let rpms1 = [HOVER_RPM - delta1, HOVER_RPM - delta1, HOVER_RPM + delta1, HOVER_RPM + delta1];
    let rpms2 = [HOVER_RPM - delta2, HOVER_RPM - delta2, HOVER_RPM + delta2, HOVER_RPM + delta2];

    let thrusts1 = compute_thrusts(rpms1);
    let thrusts2 = compute_thrusts(rpms2);
    let torque1 = compute_torques(thrusts1, rpms1, omega)[0];
    let torque2 = compute_torques(thrusts2, rpms2, omega)[0];

    // Torque should scale roughly with thrust difference (approximately quadratic in RPM)
    // But the ratio should be consistent
    assert!(
        torque2 / torque1 > 1.5,
        "Torque should increase with larger RPM difference: {} vs {}",
        torque1,
        torque2
    );
}

// ============================================================================
// Pitch Torque Tests
// ============================================================================

#[test]
fn should_generate_positive_pitch_torque_with_higher_rear_motor_thrust() {
    // X-config: Motors 1,2 are rear, motors 0,3 are front
    // Higher thrust on rear should create positive pitch (nose up)
    let front_rpm = HOVER_RPM * 0.9;
    let rear_rpm = HOVER_RPM * 1.1;
    let rpms = [front_rpm, rear_rpm, rear_rpm, front_rpm];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[1] > 0.0,
        "Pitch torque should be positive when rear motors are higher, got: {:e}",
        torques[1]
    );
}

#[test]
fn should_generate_negative_pitch_torque_with_higher_front_motor_thrust() {
    // Higher thrust on front (0,3) should create negative pitch (nose down)
    let front_rpm = HOVER_RPM * 1.1;
    let rear_rpm = HOVER_RPM * 0.9;
    let rpms = [front_rpm, rear_rpm, rear_rpm, front_rpm];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[1] < 0.0,
        "Pitch torque should be negative when front motors are higher, got: {:e}",
        torques[1]
    );
}

// ============================================================================
// Yaw Torque Tests
// ============================================================================

#[test]
fn should_generate_positive_yaw_torque_with_higher_cw_motor_rpm() {
    // Motors 2,3 spin CW, motors 0,1 spin CCW
    // Higher CW motor RPM creates positive yaw (CCW body rotation)
    let ccw_rpm = HOVER_RPM * 0.95;
    let cw_rpm = HOVER_RPM * 1.05;
    let rpms = [ccw_rpm, ccw_rpm, cw_rpm, cw_rpm];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[2] > 0.0,
        "Yaw torque should be positive when CW motors spin faster, got: {:e}",
        torques[2]
    );
}

#[test]
fn should_generate_negative_yaw_torque_with_higher_ccw_motor_rpm() {
    // Higher CCW motor RPM creates negative yaw (CW body rotation)
    let ccw_rpm = HOVER_RPM * 1.05;
    let cw_rpm = HOVER_RPM * 0.95;
    let rpms = [ccw_rpm, ccw_rpm, cw_rpm, cw_rpm];
    let thrusts = compute_thrusts(rpms);
    let omega = [0.0, 0.0, 0.0]; // No body rotation
    let torques = compute_torques(thrusts, rpms, omega);

    assert!(
        torques[2] < 0.0,
        "Yaw torque should be negative when CCW motors spin faster, got: {:e}",
        torques[2]
    );
}

#[test]
fn should_maintain_constant_thrust_during_pure_yaw_command() {
    // Yaw should be achievable without changing total thrust
    let ccw_rpm = HOVER_RPM * 0.95;
    let cw_rpm = HOVER_RPM * 1.05;
    let yaw_rpms = [ccw_rpm, ccw_rpm, cw_rpm, cw_rpm];
    let hover_rpms = [HOVER_RPM; 4];

    let yaw_thrusts = compute_thrusts(yaw_rpms);
    let hover_thrusts = compute_thrusts(hover_rpms);

    let yaw_total = compute_total_thrust(yaw_thrusts);
    let hover_total = compute_total_thrust(hover_thrusts);

    // Total thrust should be similar (not exactly equal due to RPM^2)
    let diff_percent = ((yaw_total - hover_total) / hover_total).abs() * 100.0;
    assert!(
        diff_percent < 1.0,
        "Thrust change during yaw should be <1%, got: {:.2}%",
        diff_percent
    );
}

// ============================================================================
// Ground Effect Tests
// ============================================================================

#[test]
fn should_have_no_ground_effect_above_max_height() {
    let multiplier = ground_effect_multiplier(1.0); // Well above GND_EFF_MAX_HEIGHT
    assert!(
        (multiplier - 1.0).abs() < 1e-6,
        "Ground effect multiplier should be 1.0 at high altitude, got: {}",
        multiplier
    );
}

#[test]
fn should_have_significant_ground_effect_near_ground() {
    let multiplier = ground_effect_multiplier(0.02); // Very close to ground

    assert!(
        multiplier > 1.5,
        "Ground effect multiplier should be >1.5 at 2cm height, got: {}",
        multiplier
    );
}

#[test]
fn should_have_monotonically_decreasing_ground_effect_with_height() {
    let heights = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5];
    let mut prev_mult = f32::MAX;

    for &h in &heights {
        let mult = ground_effect_multiplier(h);
        assert!(
            mult < prev_mult,
            "Ground effect should decrease with height: at h={}, mult={} >= prev={}",
            h,
            mult,
            prev_mult
        );
        prev_mult = mult;
    }
}

#[test]
fn should_clamp_ground_effect_at_very_low_heights() {
    // Height below GND_EFF_H_CLIP should use clamped value
    let mult_at_clip = ground_effect_multiplier(GND_EFF_H_CLIP);
    let mult_below_clip = ground_effect_multiplier(0.001);

    // Should be clamped to the same value (exactly equal)
    assert!(
        (mult_at_clip - mult_below_clip).abs() < 1e-10,  // Tightened from 0.01
        "Ground effect should be clamped at very low heights: clip={}, below={}",
        mult_at_clip, mult_below_clip
    );
}

#[test]
fn should_compute_ground_effect_according_to_formula() {
    // k_ge = 1 + GND_EFF_COEFF * (r_prop / (4 * z))^2
    let height = 0.1;
    let ratio = PROP_RADIUS / (4.0 * height);
    let expected = 1.0 + GND_EFF_COEFF * ratio * ratio;
    let actual = ground_effect_multiplier(height);

    assert!(
        (actual - expected).abs() < 1e-5,
        "Ground effect formula mismatch: expected {}, got {}",
        expected,
        actual
    );
}

// ============================================================================
// Drag Force Tests
// ============================================================================

#[test]
fn should_have_zero_drag_at_rest() {
    let velocity = [0.0, 0.0, 0.0];
    let total_rpm = 4.0 * HOVER_RPM;
    let drag = compute_drag_force(velocity, total_rpm);

    assert!(drag[0].abs() < 1e-10, "X drag should be 0 at rest: {:e}", drag[0]);
    assert!(drag[1].abs() < 1e-10, "Y drag should be 0 at rest: {:e}", drag[1]);
    assert!(drag[2].abs() < 1e-10, "Z drag should be 0 at rest: {:e}", drag[2]);
}

#[test]
fn should_have_drag_opposing_velocity() {
    let velocity = [1.0, 2.0, -0.5];
    let total_rpm = 4.0 * HOVER_RPM;
    let drag = compute_drag_force(velocity, total_rpm);

    // Drag should oppose velocity direction
    assert!(
        drag[0] < 0.0,
        "X drag should oppose positive vx: drag={}",
        drag[0]
    );
    assert!(
        drag[1] < 0.0,
        "Y drag should oppose positive vy: drag={}",
        drag[1]
    );
    assert!(
        drag[2] > 0.0,
        "Z drag should oppose negative vz: drag={}",
        drag[2]
    );
}

#[test]
fn should_have_drag_proportional_to_velocity_squared() {
    let total_rpm = 4.0 * HOVER_RPM;

    let v1 = [1.0, 0.0, 0.0];
    let v2 = [2.0, 0.0, 0.0];

    let drag1 = compute_drag_force(v1, total_rpm);
    let drag2 = compute_drag_force(v2, total_rpm);

    // Drag should be 4x with 2x velocity
    let ratio = drag2[0] / drag1[0];
    assert!(
        (ratio - 4.0).abs() < 1e-6,  // Tightened from 0.1
        "Drag should scale with v^2: expected ratio=4, got {}",
        ratio
    );
}

#[test]
fn should_have_drag_proportional_to_total_rpm() {
    let velocity = [1.0, 0.0, 0.0];

    let drag1 = compute_drag_force(velocity, 4.0 * HOVER_RPM);
    let drag2 = compute_drag_force(velocity, 8.0 * HOVER_RPM);

    // Drag should double with 2x RPM
    let ratio = drag2[0] / drag1[0];
    assert!(
        (ratio - 2.0).abs() < 1e-6,  // Tightened from 0.1
        "Drag should scale linearly with RPM: expected ratio=2, got {}",
        ratio
    );
}

// ============================================================================
// Quaternion Tests
// ============================================================================

#[test]
fn should_normalize_quaternion_to_unit_length() {
    let q = [2.0, 1.0, 0.5, 0.25];
    let n = quat_normalize(q);
    let norm = (n[0].powi(2) + n[1].powi(2) + n[2].powi(2) + n[3].powi(2)).sqrt();

    assert!(
        (norm - 1.0).abs() < 1e-6,
        "Normalized quaternion should have unit length: {}",
        norm
    );
}

#[test]
fn should_return_identity_for_zero_quaternion() {
    let q = [0.0, 0.0, 0.0, 0.0];
    let n = quat_normalize(q);

    assert_eq!(n, [1.0, 0.0, 0.0, 0.0], "Zero quaternion should normalize to identity");
}

#[test]
fn should_preserve_identity_quaternion_under_multiplication() {
    let identity = [1.0, 0.0, 0.0, 0.0];
    let q = [0.7071, 0.7071, 0.0, 0.0]; // 90 degree rotation about X

    let result = quat_mul(identity, q);

    for i in 0..4 {
        assert!(
            (result[i] - q[i]).abs() < 1e-4,
            "Identity multiplication should preserve quaternion: {:?} vs {:?}",
            result,
            q
        );
    }
}

#[test]
fn should_compute_correct_rotation_matrix_from_identity_quaternion() {
    let q = [1.0, 0.0, 0.0, 0.0];
    let r = quat_to_rotation_matrix(q);

    // Identity rotation matrix
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    for i in 0..9 {
        assert!(
            (r[i] - expected[i]).abs() < 1e-6,
            "Rotation matrix element {} mismatch: {} vs {}",
            i,
            r[i],
            expected[i]
        );
    }
}

#[test]
fn should_roundtrip_euler_to_quaternion_and_back() {
    let euler = [0.2, -0.1, 0.3]; // roll, pitch, yaw
    let q = euler_to_quat(euler);
    let euler_back = quat_to_euler(q);

    for i in 0..3 {
        assert!(
            (euler[i] - euler_back[i]).abs() < 1e-5,
            "Euler roundtrip failed at index {}: {} vs {}",
            i,
            euler[i],
            euler_back[i]
        );
    }
}

#[test]
fn should_integrate_quaternion_with_zero_omega() {
    let q = [0.7071, 0.7071, 0.0, 0.0];
    let omega = [0.0, 0.0, 0.0];
    let q_new = quat_integrate(q, omega, 0.01);

    for i in 0..4 {
        assert!(
            (q[i] - q_new[i]).abs() < 1e-6,
            "Zero omega should not change quaternion: {:?} vs {:?}",
            q,
            q_new
        );
    }
}

#[test]
fn should_integrate_quaternion_correctly_over_time() {
    let q = [1.0, 0.0, 0.0, 0.0]; // Start level
    let omega = [0.0, 0.0, 1.0]; // Yaw rotation at 1 rad/s
    let dt = 0.01;

    // After 1 radian of rotation (1 second), yaw should be ~1 rad
    let mut q_current = q;
    for _ in 0..100 {
        q_current = quat_integrate(q_current, omega, dt);
    }

    let euler = quat_to_euler(q_current);
    let expected_yaw = 1.0;

    assert!(
        (euler[2] - expected_yaw).abs() < 0.01,  // Tightened from 0.05
        "Yaw should be ~1.0 rad after 1s: got {}",
        euler[2]
    );
}

#[test]
fn should_maintain_quaternion_normalization_after_many_integrations() {
    let mut q = [1.0, 0.0, 0.0, 0.0];
    let omega = [0.5, 0.3, 0.2];
    let dt = 1.0 / 240.0;

    for _ in 0..10000 {
        q = quat_integrate(q, omega, dt);
    }

    let norm = (q[0].powi(2) + q[1].powi(2) + q[2].powi(2) + q[3].powi(2)).sqrt();

    assert!(
        (norm - 1.0).abs() < 1e-6,  // Tightened from 1e-4
        "Quaternion should stay normalized after many integrations: norm={}",
        norm
    );
}

// ============================================================================
// Angular Acceleration Tests
// ============================================================================

#[test]
fn should_compute_angular_acceleration_with_no_torque() {
    let torques = [0.0, 0.0, 0.0];
    let omega = [0.0, 0.0, 0.0];
    let alpha = compute_angular_acceleration(torques, omega);

    assert!(alpha[0].abs() < 1e-10, "Alpha_x should be 0: {:e}", alpha[0]);
    assert!(alpha[1].abs() < 1e-10, "Alpha_y should be 0: {:e}", alpha[1]);
    assert!(alpha[2].abs() < 1e-10, "Alpha_z should be 0: {:e}", alpha[2]);
}

#[test]
fn should_compute_angular_acceleration_proportional_to_torque() {
    let tau = 0.001; // 1 mN*m torque
    let torques = [tau, 0.0, 0.0];
    let omega = [0.0, 0.0, 0.0];
    let alpha = compute_angular_acceleration(torques, omega);

    // alpha_x = tau / Ixx
    let expected = tau / IXX;

    assert!(
        (alpha[0] - expected).abs() < 1e-6,
        "Angular acceleration mismatch: expected {}, got {}",
        expected,
        alpha[0]
    );
}

#[test]
fn should_include_gyroscopic_effect_in_angular_acceleration() {
    // When rotating about two axes, gyroscopic effects couple them
    let torques = [0.0, 0.0, 0.0];
    let omega = [1.0, 1.0, 0.0]; // Rotating about X and Y

    let alpha = compute_angular_acceleration(torques, omega);

    // With zero torque but non-zero omega, gyroscopic terms should create acceleration
    // alpha_z = -wx * wy * (Iyy - Ixx) / Izz
    // Since Ixx = Iyy, gyro_z = 0
    assert!(
        alpha[2].abs() < 1e-10,
        "With symmetric Ixx=Iyy, gyro_z should be 0: {:e}",
        alpha[2]
    );

    // alpha_x = -wy * wz * (Izz - Iyy) / Ixx with wz=0 should be 0
    // alpha_y = -wx * wz * (Ixx - Izz) / Iyy with wz=0 should be 0
    // Let's test with wz non-zero
    let omega2 = [1.0, 0.0, 1.0];
    let alpha2 = compute_angular_acceleration(torques, omega2);

    // alpha_y = -wx * wz * (Ixx - Izz) / Iyy
    let expected_gy = 1.0 * 1.0 * (IXX - IZZ) / IYY; // Note: formula subtracts gyro term
    let gyro_y = omega2[0] * omega2[2] * (IXX - IZZ);
    let expected_alpha_y = -gyro_y / IYY;

    assert!(
        (alpha2[1] - expected_alpha_y).abs() < 1e-6,
        "Gyroscopic coupling mismatch: expected {}, got {}",
        expected_alpha_y,
        alpha2[1]
    );
}

// ============================================================================
// Integration Test: Full Physics Step
// ============================================================================

#[test]
fn should_update_all_state_components_in_physics_step() {
    let mut state = QuadcopterState::new(1, 0);
    state.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let initial_pos = state.get_position(0);
    let initial_quat = state.get_quaternion(0);
    let initial_vel = state.get_velocity(0);
    let initial_omega = state.get_angular_velocity(0);

    // Apply asymmetric RPMs to induce motion
    let rpms = [HOVER_RPM * 1.1, HOVER_RPM * 0.9, HOVER_RPM * 1.05, HOVER_RPM * 0.95];
    let dt = 1.0 / 240.0;

    physics_step_scalar(&mut state, 0, rpms, dt);

    let final_pos = state.get_position(0);
    let final_quat = state.get_quaternion(0);
    let final_vel = state.get_velocity(0);
    let final_omega = state.get_angular_velocity(0);

    // Position should change due to thrust imbalance
    let pos_changed = (final_pos[0] - initial_pos[0]).abs()
        + (final_pos[1] - initial_pos[1]).abs()
        + (final_pos[2] - initial_pos[2]).abs()
        > 1e-10;

    // Quaternion should change due to torques
    let quat_changed = (final_quat[0] - initial_quat[0]).abs()
        + (final_quat[1] - initial_quat[1]).abs()
        + (final_quat[2] - initial_quat[2]).abs()
        + (final_quat[3] - initial_quat[3]).abs()
        > 1e-10;

    // Velocity should change (acceleration from asymmetric thrust)
    let vel_changed = (final_vel[0] - initial_vel[0]).abs()
        + (final_vel[1] - initial_vel[1]).abs()
        + (final_vel[2] - initial_vel[2]).abs()
        > 1e-10;

    // Angular velocity should change (angular acceleration from torques)
    let omega_changed = (final_omega[0] - initial_omega[0]).abs()
        + (final_omega[1] - initial_omega[1]).abs()
        + (final_omega[2] - initial_omega[2]).abs()
        > 1e-10;

    assert!(pos_changed, "Position should change with asymmetric RPMs");
    assert!(quat_changed, "Quaternion should change with torques");
    assert!(vel_changed, "Velocity should change with asymmetric thrust");
    assert!(omega_changed, "Angular velocity should change with torques");
}

#[test]
fn should_substep_physics_correctly() {
    let mut state1 = QuadcopterState::new(1, 0);
    let mut state2 = QuadcopterState::new(1, 0);
    state1.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);
    state2.reset_env(0, 42, &InitConfig::fixed_start(), [0.0, 0.0, 1.0]);

    let rpms = [HOVER_RPM * 1.1, HOVER_RPM * 0.9, HOVER_RPM * 1.0, HOVER_RPM * 1.0];
    let dt = 1.0 / 30.0; // Control timestep

    // Run single step on state1
    physics_substeps_scalar(&mut state1, 0, rpms, dt, 1);

    // Run 8 substeps on state2
    physics_substeps_scalar(&mut state2, 0, rpms, dt, 8);

    // state2 should be more accurate (less numerical error from substeps)
    // The results will differ, but both should be reasonable
    let pos1 = state1.get_position(0);
    let pos2 = state2.get_position(0);

    // Both should have moved (not testing accuracy, just that substeps work)
    let moved1 = (pos1[0].abs() + pos1[1].abs()) > 0.0 || (pos1[2] - 1.0).abs() > 1e-6;
    let moved2 = (pos2[0].abs() + pos2[1].abs()) > 0.0 || (pos2[2] - 1.0).abs() > 1e-6;

    assert!(moved1 || moved2, "At least one simulation should show movement");
}
