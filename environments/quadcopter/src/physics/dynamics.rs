//! Core rigid body dynamics for quadcopter simulation.
//!
//! Implements the physics equations for a Crazyflie 2.0 X-configuration drone:
//! - Thrust and torque computation from motor RPMs
//! - Rigid body dynamics with quaternion rotation
//! - Euler integration for state update
//!
//! Provides both scalar and SIMD implementations.

use crate::constants::*;
use crate::physics::quaternion::*;
use crate::state::QuadcopterState;
use crate::types::{BodyFrame, UnitQuaternion, Vec3, WorldFrame};

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat, StdFloat};

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Compute thrust force from 4 motor RPMs.
/// F_i = KF * rpm_i^2
#[inline(always)]
pub fn compute_thrusts(rpms: [f32; 4]) -> [f32; 4] {
    [
        KF * rpms[0] * rpms[0],
        KF * rpms[1] * rpms[1],
        KF * rpms[2] * rpms[2],
        KF * rpms[3] * rpms[3],
    ]
}

/// Compute total thrust magnitude (sum of all motor thrusts).
#[inline(always)]
pub fn compute_total_thrust(thrusts: [f32; 4]) -> f32 {
    thrusts[0] + thrusts[1] + thrusts[2] + thrusts[3]
}

/// Compute body-frame torques from motor thrusts and reaction torques.
///
/// X-configuration motor layout (top view):
/// ```text
///      Front
///        ^
///    3-------0
///     \     /
///      \ X /  ← Center
///       / \
///      /   \
///    2-------1
///       Rear
/// ```
///
/// Motor positions relative to CoM:
/// - Motor 0 (FR): (+L/√2, -L/√2, 0) CCW
/// - Motor 1 (RR): (-L/√2, -L/√2, 0) CCW
/// - Motor 2 (RL): (-L/√2, +L/√2, 0) CW
/// - Motor 3 (FL): (+L/√2, +L/√2, 0) CW
#[inline(always)]
pub fn compute_thrust_torques(thrusts: [f32; 4], rpms: [f32; 4]) -> [f32; 3] {
    // Roll torque (x-axis): positive = right side up
    // Motor 2,3 (left) push up → positive roll
    // Motor 0,1 (right) push up → negative roll
    let tau_x = L_SQRT2 * (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1]);

    // Pitch torque (y-axis): positive = nose up
    // Motor 0,3 (front) push up → negative pitch
    // Motor 1,2 (rear) push up → positive pitch
    let tau_y = L_SQRT2 * (thrusts[1] + thrusts[2] - thrusts[0] - thrusts[3]);

    // Yaw torque (z-axis): from motor reaction torques
    // CCW motors (0,1) create CW reaction → negative yaw
    // CW motors (2,3) create CCW reaction → positive yaw
    let tau_z = KM
        * (MOTOR_DIRS[0] * rpms[0] * rpms[0]
            + MOTOR_DIRS[1] * rpms[1] * rpms[1]
            + MOTOR_DIRS[2] * rpms[2] * rpms[2]
            + MOTOR_DIRS[3] * rpms[3] * rpms[3]);

    [tau_x, tau_y, tau_z]
}

/// Compute gyroscopic precession torque from spinning propellers.
///
/// Physics: τ_gyro = J_prop × ω_prop_net × ω_body
///
/// When the quadcopter rotates, the angular momentum of the spinning propellers
/// creates a gyroscopic torque perpendicular to both the propeller axis and
/// the body rotation axis.
///
/// For propellers spinning about the Z-axis with net angular velocity ω_prop:
/// τ_gyro = J_prop * ω_prop * [ω_y, -ω_x, 0]
///
/// The net propeller angular velocity accounts for CW/CCW directions:
/// ω_prop_net = Σ(dir_i * rpm_i) * (π/30)
#[inline(always)]
pub fn compute_gyroscopic_torque(rpms: [f32; 4], omega: [f32; 3]) -> [f32; 3] {
    // Net propeller angular velocity (rad/s), accounting for rotation directions
    let omega_prop_net = (MOTOR_DIRS[0] * rpms[0]
        + MOTOR_DIRS[1] * rpms[1]
        + MOTOR_DIRS[2] * rpms[2]
        + MOTOR_DIRS[3] * rpms[3])
        * PROP_VEL_COEFF;

    // Angular momentum of propellers about Z-axis
    let h_prop = PROP_INERTIA * omega_prop_net;

    // Gyroscopic torque: H × ω = [0, 0, H_z] × [ω_x, ω_y, ω_z]
    // = [H_z * ω_y, -H_z * ω_x, 0]
    [h_prop * omega[1], -h_prop * omega[0], 0.0]
}

/// Compute total body-frame torques including gyroscopic precession.
///
/// Combines thrust-based torques with gyroscopic precession effects.
#[inline(always)]
pub fn compute_torques(thrusts: [f32; 4], rpms: [f32; 4], omega: [f32; 3]) -> [f32; 3] {
    let thrust_torques = compute_thrust_torques(thrusts, rpms);
    let gyro_torques = compute_gyroscopic_torque(rpms, omega);

    [
        thrust_torques[0] + gyro_torques[0],
        thrust_torques[1] + gyro_torques[1],
        thrust_torques[2] + gyro_torques[2],
    ]
}

/// Compute linear acceleration in world frame.
///
/// a = (R * [0, 0, F_total] - [0, 0, m*g]) / m
///
/// # Arguments
/// * `quat` - Current orientation quaternion [w, x, y, z]
/// * `total_thrust` - Total thrust from all motors
///
/// # Returns
/// Linear acceleration [ax, ay, az] in world frame
#[inline(always)]
pub fn compute_linear_acceleration(quat: [f32; 4], total_thrust: f32) -> [f32; 3] {
    // Get thrust direction in world frame (Z column of rotation matrix)
    let thrust_dir = quat_to_thrust_dir(quat);

    // Acceleration = (thrust_direction * F_total - gravity) / mass
    [
        thrust_dir[0] * total_thrust * M_INV,
        thrust_dir[1] * total_thrust * M_INV,
        thrust_dir[2] * total_thrust * M_INV - G,
    ]
}

/// Compute angular acceleration in body frame.
///
/// α = J⁻¹ * (τ - ω × Jω)
///
/// For a symmetric quadcopter with diagonal inertia matrix:
/// α_x = (τ_x - ω_y * ω_z * (I_zz - I_yy)) / I_xx
/// α_y = (τ_y - ω_x * ω_z * (I_xx - I_zz)) / I_yy
/// α_z = (τ_z - ω_x * ω_y * (I_yy - I_xx)) / I_zz
#[inline(always)]
pub fn compute_angular_acceleration(torques: [f32; 3], omega: [f32; 3]) -> [f32; 3] {
    let (tau_x, tau_y, tau_z) = (torques[0], torques[1], torques[2]);
    let (wx, wy, wz) = (omega[0], omega[1], omega[2]);

    // Gyroscopic terms (ω × Jω)
    let gyro_x = wy * wz * (IZZ - IYY);
    let gyro_y = wx * wz * (IXX - IZZ);
    let gyro_z = wx * wy * (IYY - IXX);

    [
        (tau_x - gyro_x) * IXX_INV,
        (tau_y - gyro_y) * IYY_INV,
        (tau_z - gyro_z) * IZZ_INV,
    ]
}

// ============================================================================
// Type-Safe Wrappers (Public API)
// ============================================================================

/// Compute linear acceleration in world frame (typed version).
///
/// a = (R * [0, 0, F_total] - [0, 0, m*g]) / m
#[inline]
pub fn compute_linear_acceleration_typed(
    orientation: UnitQuaternion,
    total_thrust: f32,
) -> Vec3<WorldFrame> {
    let accel = compute_linear_acceleration(orientation.as_array(), total_thrust);
    Vec3::from_array(accel)
}

/// Compute angular acceleration in body frame (typed version).
///
/// α = J⁻¹ * (τ - ω × Jω)
#[inline]
pub fn compute_angular_acceleration_typed(
    torques: Vec3<BodyFrame>,
    omega: Vec3<BodyFrame>,
) -> Vec3<BodyFrame> {
    let accel = compute_angular_acceleration(torques.as_array(), omega.as_array());
    Vec3::from_array(accel)
}

/// Compute total body-frame torques including gyroscopic precession (typed version).
#[inline]
pub fn compute_torques_typed(
    thrusts: [f32; 4],
    rpms: [f32; 4],
    omega: Vec3<BodyFrame>,
) -> Vec3<BodyFrame> {
    let torques = compute_torques(thrusts, rpms, omega.as_array());
    Vec3::from_array(torques)
}

/// Compute body-frame torques from motor thrusts and reaction torques (typed version).
#[inline]
pub fn compute_thrust_torques_typed(thrusts: [f32; 4], rpms: [f32; 4]) -> Vec3<BodyFrame> {
    let torques = compute_thrust_torques(thrusts, rpms);
    Vec3::from_array(torques)
}

/// Compute gyroscopic precession torque (typed version).
#[inline]
pub fn compute_gyroscopic_torque_typed(rpms: [f32; 4], omega: Vec3<BodyFrame>) -> Vec3<BodyFrame> {
    let torque = compute_gyroscopic_torque(rpms, omega.as_array());
    Vec3::from_array(torque)
}

/// Perform one physics step using Euler integration.
///
/// Updates position, velocity, quaternion, and angular velocity for a single environment.
///
/// # Arguments
/// * `state` - Mutable reference to the state buffer
/// * `idx` - Environment index
/// * `rpms` - Motor RPMs [motor0, motor1, motor2, motor3]
/// * `dt` - Time step in seconds
#[inline]
pub fn physics_step_scalar(state: &mut QuadcopterState, idx: usize, rpms: [f32; 4], dt: f32) {
    // Get current state
    let pos = state.get_position(idx);
    let quat = state.get_quaternion(idx);
    let vel = state.get_velocity(idx);
    let omega = state.get_angular_velocity(idx);

    // Compute forces and torques (including gyroscopic precession)
    let thrusts = compute_thrusts(rpms);
    let total_thrust = compute_total_thrust(thrusts);
    let torques = compute_torques(thrusts, rpms, omega);

    // Compute accelerations
    let lin_accel = compute_linear_acceleration(quat, total_thrust);
    let ang_accel = compute_angular_acceleration(torques, omega);

    // Euler integration for velocity
    let new_vel = [
        vel[0] + lin_accel[0] * dt,
        vel[1] + lin_accel[1] * dt,
        vel[2] + lin_accel[2] * dt,
    ];

    // Euler integration for position
    let new_pos = [
        pos[0] + new_vel[0] * dt,
        pos[1] + new_vel[1] * dt,
        pos[2] + new_vel[2] * dt,
    ];

    // Euler integration for angular velocity
    let new_omega = [
        omega[0] + ang_accel[0] * dt,
        omega[1] + ang_accel[1] * dt,
        omega[2] + ang_accel[2] * dt,
    ];

    // Quaternion integration using Rodrigues formula
    let new_quat = quat_integrate(quat, new_omega, dt);

    // Store updated state
    state.set_position(idx, new_pos);
    state.set_quaternion(idx, new_quat);
    state.set_velocity(idx, new_vel);
    state.set_angular_velocity(idx, new_omega);
}

/// Perform multiple physics substeps for a single environment.
#[inline]
pub fn physics_substeps_scalar(
    state: &mut QuadcopterState,
    idx: usize,
    rpms: [f32; 4],
    dt: f32,
    num_substeps: u32,
) {
    let substep_dt = dt / num_substeps as f32;
    for _ in 0..num_substeps {
        physics_step_scalar(state, idx, rpms, substep_dt);
    }
}

// ============================================================================
// SIMD Implementation (process 8 environments at once)
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    use crate::physics::quaternion::simd::*;
    use crate::physics::simd_helpers::*;

    /// Compute thrusts for 8 environments × 4 motors = 32 values.
    /// Input: 4 f32x8 vectors (one per motor, each containing 8 environments' RPMs)
    /// Output: 4 f32x8 vectors of thrust values
    #[inline(always)]
    pub fn compute_thrusts_simd(
        rpm0: f32x8,
        rpm1: f32x8,
        rpm2: f32x8,
        rpm3: f32x8,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        let kf = f32x8::splat(KF);
        (
            kf * rpm0 * rpm0,
            kf * rpm1 * rpm1,
            kf * rpm2 * rpm2,
            kf * rpm3 * rpm3,
        )
    }

    /// Compute total thrust for 8 environments.
    #[inline(always)]
    pub fn compute_total_thrust_simd(t0: f32x8, t1: f32x8, t2: f32x8, t3: f32x8) -> f32x8 {
        t0 + t1 + t2 + t3
    }

    /// Compute thrust-based torques for 8 environments.
    #[inline(always)]
    pub fn compute_thrust_torques_simd(
        t0: f32x8,
        t1: f32x8,
        t2: f32x8,
        t3: f32x8,
        rpm0: f32x8,
        rpm1: f32x8,
        rpm2: f32x8,
        rpm3: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let l_sqrt2 = f32x8::splat(L_SQRT2);
        let km = f32x8::splat(KM);

        // Roll torque
        let tau_x = l_sqrt2 * (t2 + t3 - t0 - t1);

        // Pitch torque
        let tau_y = l_sqrt2 * (t1 + t2 - t0 - t3);

        // Yaw torque
        let dir0 = f32x8::splat(MOTOR_DIRS[0]);
        let dir1 = f32x8::splat(MOTOR_DIRS[1]);
        let dir2 = f32x8::splat(MOTOR_DIRS[2]);
        let dir3 = f32x8::splat(MOTOR_DIRS[3]);

        let tau_z =
            km * (dir0 * rpm0 * rpm0 + dir1 * rpm1 * rpm1 + dir2 * rpm2 * rpm2 + dir3 * rpm3 * rpm3);

        (tau_x, tau_y, tau_z)
    }

    /// Compute gyroscopic precession torque for 8 environments.
    ///
    /// τ_gyro = J_prop × ω_prop_net × ω_body
    #[inline(always)]
    pub fn compute_gyroscopic_torque_simd(
        rpm0: f32x8,
        rpm1: f32x8,
        rpm2: f32x8,
        rpm3: f32x8,
        omega_x: f32x8,
        omega_y: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let dir0 = f32x8::splat(MOTOR_DIRS[0]);
        let dir1 = f32x8::splat(MOTOR_DIRS[1]);
        let dir2 = f32x8::splat(MOTOR_DIRS[2]);
        let dir3 = f32x8::splat(MOTOR_DIRS[3]);
        let prop_vel_coeff = f32x8::splat(PROP_VEL_COEFF);
        let prop_inertia = f32x8::splat(PROP_INERTIA);

        // Net propeller angular velocity
        let omega_prop_net =
            (dir0 * rpm0 + dir1 * rpm1 + dir2 * rpm2 + dir3 * rpm3) * prop_vel_coeff;

        // Angular momentum of propellers
        let h_prop = prop_inertia * omega_prop_net;

        // Gyroscopic torque: [H * ω_y, -H * ω_x, 0]
        (h_prop * omega_y, -h_prop * omega_x, f32x8::splat(0.0))
    }

    /// Compute total torques (thrust + gyroscopic) for 8 environments.
    #[inline(always)]
    pub fn compute_torques_simd(
        t0: f32x8,
        t1: f32x8,
        t2: f32x8,
        t3: f32x8,
        rpm0: f32x8,
        rpm1: f32x8,
        rpm2: f32x8,
        rpm3: f32x8,
        omega_x: f32x8,
        omega_y: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let (thrust_x, thrust_y, thrust_z) =
            compute_thrust_torques_simd(t0, t1, t2, t3, rpm0, rpm1, rpm2, rpm3);
        let (gyro_x, gyro_y, _) =
            compute_gyroscopic_torque_simd(rpm0, rpm1, rpm2, rpm3, omega_x, omega_y);

        (thrust_x + gyro_x, thrust_y + gyro_y, thrust_z)
    }

    /// Compute linear acceleration for 8 environments.
    #[inline(always)]
    pub fn compute_linear_acceleration_simd(
        qw: f32x8,
        qx: f32x8,
        qy: f32x8,
        qz: f32x8,
        total_thrust: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        // Get thrust direction in world frame
        let (tx, ty, tz) = quat_to_thrust_dir_simd(qw, qx, qy, qz);

        let m_inv = f32x8::splat(M_INV);
        let g = f32x8::splat(G);

        (
            tx * total_thrust * m_inv,
            ty * total_thrust * m_inv,
            tz * total_thrust * m_inv - g,
        )
    }

    /// Compute angular acceleration for 8 environments.
    #[inline(always)]
    pub fn compute_angular_acceleration_simd(
        tau_x: f32x8,
        tau_y: f32x8,
        tau_z: f32x8,
        wx: f32x8,
        wy: f32x8,
        wz: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let izz_iyy = f32x8::splat(IZZ - IYY);
        let ixx_izz = f32x8::splat(IXX - IZZ);
        let iyy_ixx = f32x8::splat(IYY - IXX);

        let ixx_inv = f32x8::splat(IXX_INV);
        let iyy_inv = f32x8::splat(IYY_INV);
        let izz_inv = f32x8::splat(IZZ_INV);

        // Gyroscopic terms
        let gyro_x = wy * wz * izz_iyy;
        let gyro_y = wx * wz * ixx_izz;
        let gyro_z = wx * wy * iyy_ixx;

        (
            (tau_x - gyro_x) * ixx_inv,
            (tau_y - gyro_y) * iyy_inv,
            (tau_z - gyro_z) * izz_inv,
        )
    }

    /// Perform one physics step for 8 environments simultaneously.
    ///
    /// # Arguments
    /// * `state` - Mutable reference to the state buffer
    /// * `base_idx` - Starting environment index (must be aligned to 8)
    /// * `rpm0-3` - Motor RPMs for each motor across 8 environments
    /// * `dt` - Time step
    #[inline]
    pub fn physics_step_simd(
        state: &mut QuadcopterState,
        base_idx: usize,
        rpm0: f32x8,
        rpm1: f32x8,
        rpm2: f32x8,
        rpm3: f32x8,
        dt: f32,
    ) {
        let dt_vec = f32x8::splat(dt);

        // Load current state (SoA layout enables efficient SIMD loads)
        let pos_x = f32x8::from_slice(&state.pos_x[base_idx..]);
        let pos_y = f32x8::from_slice(&state.pos_y[base_idx..]);
        let pos_z = f32x8::from_slice(&state.pos_z[base_idx..]);

        let qw = f32x8::from_slice(&state.quat_w[base_idx..]);
        let qx = f32x8::from_slice(&state.quat_x[base_idx..]);
        let qy = f32x8::from_slice(&state.quat_y[base_idx..]);
        let qz = f32x8::from_slice(&state.quat_z[base_idx..]);

        let vel_x = f32x8::from_slice(&state.vel_x[base_idx..]);
        let vel_y = f32x8::from_slice(&state.vel_y[base_idx..]);
        let vel_z = f32x8::from_slice(&state.vel_z[base_idx..]);

        let omega_x = f32x8::from_slice(&state.ang_vel_x[base_idx..]);
        let omega_y = f32x8::from_slice(&state.ang_vel_y[base_idx..]);
        let omega_z = f32x8::from_slice(&state.ang_vel_z[base_idx..]);

        // Compute forces and torques (including gyroscopic precession)
        let (t0, t1, t2, t3) = compute_thrusts_simd(rpm0, rpm1, rpm2, rpm3);
        let total_thrust = compute_total_thrust_simd(t0, t1, t2, t3);
        let (tau_x, tau_y, tau_z) =
            compute_torques_simd(t0, t1, t2, t3, rpm0, rpm1, rpm2, rpm3, omega_x, omega_y);

        // Compute accelerations
        let (ax, ay, az) = compute_linear_acceleration_simd(qw, qx, qy, qz, total_thrust);
        let (alpha_x, alpha_y, alpha_z) =
            compute_angular_acceleration_simd(tau_x, tau_y, tau_z, omega_x, omega_y, omega_z);

        // Euler integration for velocity
        let new_vel_x = vel_x + ax * dt_vec;
        let new_vel_y = vel_y + ay * dt_vec;
        let new_vel_z = vel_z + az * dt_vec;

        // Euler integration for position
        let new_pos_x = pos_x + new_vel_x * dt_vec;
        let new_pos_y = pos_y + new_vel_y * dt_vec;
        let new_pos_z = pos_z + new_vel_z * dt_vec;

        // Euler integration for angular velocity
        let new_omega_x = omega_x + alpha_x * dt_vec;
        let new_omega_y = omega_y + alpha_y * dt_vec;
        let new_omega_z = omega_z + alpha_z * dt_vec;

        // Quaternion integration
        let (new_qw, new_qx, new_qy, new_qz) = quat_integrate_simd(
            qw, qx, qy, qz, new_omega_x, new_omega_y, new_omega_z, dt,
        );

        // Store updated state
        new_pos_x.copy_to_slice(&mut state.pos_x[base_idx..]);
        new_pos_y.copy_to_slice(&mut state.pos_y[base_idx..]);
        new_pos_z.copy_to_slice(&mut state.pos_z[base_idx..]);

        new_qw.copy_to_slice(&mut state.quat_w[base_idx..]);
        new_qx.copy_to_slice(&mut state.quat_x[base_idx..]);
        new_qy.copy_to_slice(&mut state.quat_y[base_idx..]);
        new_qz.copy_to_slice(&mut state.quat_z[base_idx..]);

        new_vel_x.copy_to_slice(&mut state.vel_x[base_idx..]);
        new_vel_y.copy_to_slice(&mut state.vel_y[base_idx..]);
        new_vel_z.copy_to_slice(&mut state.vel_z[base_idx..]);

        new_omega_x.copy_to_slice(&mut state.ang_vel_x[base_idx..]);
        new_omega_y.copy_to_slice(&mut state.ang_vel_y[base_idx..]);
        new_omega_z.copy_to_slice(&mut state.ang_vel_z[base_idx..]);
    }

    /// Perform multiple physics substeps for 8 environments.
    #[inline]
    pub fn physics_substeps_simd(
        state: &mut QuadcopterState,
        base_idx: usize,
        rpm0: f32x8,
        rpm1: f32x8,
        rpm2: f32x8,
        rpm3: f32x8,
        dt: f32,
        num_substeps: u32,
    ) {
        let substep_dt = dt / num_substeps as f32;
        for _ in 0..num_substeps {
            physics_step_simd(state, base_idx, rpm0, rpm1, rpm2, rpm3, substep_dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_thrust() {
        // At hover, total thrust should equal weight
        let hover_rpm = HOVER_RPM;
        let rpms = [hover_rpm; 4];
        let thrusts = compute_thrusts(rpms);
        let total = compute_total_thrust(thrusts);

        let weight = M * G;
        assert!((total - weight).abs() < 0.01, "Hover thrust {} != weight {}", total, weight);
    }

    #[test]
    fn test_hover_no_torque() {
        // Equal RPMs should produce no torque (at zero body rotation)
        let rpms = [HOVER_RPM; 4];
        let thrusts = compute_thrusts(rpms);
        let omega = [0.0, 0.0, 0.0]; // No body rotation
        let torques = compute_torques(thrusts, rpms, omega);

        assert!(torques[0].abs() < 1e-10, "Roll torque not zero: {}", torques[0]);
        assert!(torques[1].abs() < 1e-10, "Pitch torque not zero: {}", torques[1]);
        assert!(torques[2].abs() < 1e-10, "Yaw torque not zero: {}", torques[2]);
    }

    #[test]
    fn test_roll_torque_direction() {
        // Increase left motors (2,3), decrease right motors (0,1) → positive roll
        let left_rpm = HOVER_RPM * 1.1;
        let right_rpm = HOVER_RPM * 0.9;
        let rpms = [right_rpm, right_rpm, left_rpm, left_rpm];
        let thrusts = compute_thrusts(rpms);
        let omega = [0.0, 0.0, 0.0];
        let torques = compute_torques(thrusts, rpms, omega);

        assert!(torques[0] > 0.0, "Roll torque should be positive: {}", torques[0]);
    }

    #[test]
    fn test_pitch_torque_direction() {
        // Increase rear motors (1,2), decrease front motors (0,3) → positive pitch
        let front_rpm = HOVER_RPM * 0.9;
        let rear_rpm = HOVER_RPM * 1.1;
        let rpms = [front_rpm, rear_rpm, rear_rpm, front_rpm];
        let thrusts = compute_thrusts(rpms);
        let omega = [0.0, 0.0, 0.0];
        let torques = compute_torques(thrusts, rpms, omega);

        assert!(torques[1] > 0.0, "Pitch torque should be positive: {}", torques[1]);
    }

    #[test]
    fn test_yaw_torque_direction() {
        // Increase CW motors (2,3), decrease CCW motors (0,1) → positive yaw
        let ccw_rpm = HOVER_RPM * 0.95;
        let cw_rpm = HOVER_RPM * 1.05;
        let rpms = [ccw_rpm, ccw_rpm, cw_rpm, cw_rpm];
        let thrusts = compute_thrusts(rpms);
        let omega = [0.0, 0.0, 0.0];
        let torques = compute_torques(thrusts, rpms, omega);

        assert!(torques[2] > 0.0, "Yaw torque should be positive: {}", torques[2]);
    }

    #[test]
    fn test_gyroscopic_torque_zero_when_not_rotating() {
        // When body is not rotating, gyroscopic torque should be zero
        let rpms = [HOVER_RPM; 4];
        let omega = [0.0, 0.0, 0.0];
        let gyro = compute_gyroscopic_torque(rpms, omega);

        assert!(gyro[0].abs() < 1e-15, "Gyro X not zero: {}", gyro[0]);
        assert!(gyro[1].abs() < 1e-15, "Gyro Y not zero: {}", gyro[1]);
        assert_eq!(gyro[2], 0.0, "Gyro Z should always be zero");
    }

    #[test]
    fn test_gyroscopic_torque_direction() {
        // With positive pitch rate (ω_y > 0) and net positive prop rotation,
        // gyroscopic torque should create positive roll torque
        let rpms = [HOVER_RPM; 4]; // Motors 0,1 CCW (-1), 2,3 CW (+1) → net ~0
        let omega = [0.0, 1.0, 0.0]; // Pitching forward

        // Use asymmetric RPMs to get net propeller rotation
        // More CW (2,3) than CCW (0,1) → positive net ω_prop
        let rpms_asym = [HOVER_RPM * 0.9, HOVER_RPM * 0.9, HOVER_RPM * 1.1, HOVER_RPM * 1.1];
        let gyro = compute_gyroscopic_torque(rpms_asym, omega);

        // Net positive ω_prop × positive ω_y → positive τ_x
        assert!(gyro[0] > 0.0, "Expected positive roll gyro torque, got: {}", gyro[0]);
    }

    #[test]
    fn test_gyroscopic_torque_z_always_zero() {
        // Gyroscopic torque about Z should always be zero (props spin about Z)
        let rpms = [20000.0, 15000.0, 18000.0, 12000.0]; // Asymmetric
        let omega = [5.0, -3.0, 10.0]; // Arbitrary rotation
        let gyro = compute_gyroscopic_torque(rpms, omega);

        assert_eq!(gyro[2], 0.0, "Gyro Z should always be zero");
    }

    #[test]
    fn test_identity_quat_thrust_up() {
        // Identity quaternion should have thrust pointing up (world +Z)
        let quat = [1.0, 0.0, 0.0, 0.0];
        let thrust = 1.0;
        let accel = compute_linear_acceleration(quat, thrust);

        // Should have no horizontal acceleration, positive Z minus gravity
        assert!(accel[0].abs() < 1e-6);
        assert!(accel[1].abs() < 1e-6);
        assert!((accel[2] - (thrust * M_INV - G)).abs() < 1e-6);
    }

    #[test]
    fn test_hover_acceleration() {
        // At hover with identity orientation, vertical acceleration should be ~0
        let quat = [1.0, 0.0, 0.0, 0.0];
        let thrust = M * G; // Exactly balances gravity
        let accel = compute_linear_acceleration(quat, thrust);

        assert!(accel[2].abs() < 1e-5, "Hover should have zero vertical acceleration");
    }

    #[test]
    fn test_physics_step() {
        let mut state = QuadcopterState::new(1, 0);
        state.reset_env(0, 42, &crate::config::InitConfig::default(), [0.0, 0.0, 1.0]);

        // Set hover RPMs
        let rpms = [HOVER_RPM; 4];

        // Initial position
        let initial_z = state.pos_z[0];

        // Take a physics step
        physics_step_scalar(&mut state, 0, rpms, 1.0 / 240.0);

        // Should stay approximately at same height (hover)
        let delta_z = (state.pos_z[0] - initial_z).abs();
        assert!(delta_z < 0.01, "Position changed too much during hover: {}", delta_z);
    }
}
