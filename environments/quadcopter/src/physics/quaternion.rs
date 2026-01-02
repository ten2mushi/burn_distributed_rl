//! Quaternion operations for rigid body rotation.
//!
//! Provides both scalar and SIMD implementations for:
//! - Quaternion multiplication and normalization
//! - Quaternion integration with angular velocity
//! - Quaternion to rotation matrix conversion
//! - Quaternion to/from Euler angle conversion

use std::f32::consts::PI;

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat, StdFloat};

// ============================================================================
// Scalar Operations
// ============================================================================

/// Normalize a quaternion to unit length.
#[inline(always)]
pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-10 {
        return [1.0, 0.0, 0.0, 0.0]; // Identity quaternion
    }
    let inv_norm = 1.0 / norm;
    [
        q[0] * inv_norm,
        q[1] * inv_norm,
        q[2] * inv_norm,
        q[3] * inv_norm,
    ]
}

/// Multiply two quaternions: q_result = q1 * q2.
/// Quaternion format: [w, x, y, z]
#[inline(always)]
pub fn quat_mul(q1: [f32; 4], q2: [f32; 4]) -> [f32; 4] {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);

    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// Integrate quaternion with angular velocity using Rodrigues formula.
///
/// q_new = exp(dt/2 * omega) * q
///
/// # Arguments
/// * `q` - Current quaternion [w, x, y, z]
/// * `omega` - Angular velocity [wx, wy, wz] in body frame
/// * `dt` - Time step
#[inline(always)]
pub fn quat_integrate(q: [f32; 4], omega: [f32; 3], dt: f32) -> [f32; 4] {
    let (wx, wy, wz) = (omega[0], omega[1], omega[2]);
    let omega_norm = (wx * wx + wy * wy + wz * wz).sqrt();

    if omega_norm < 1e-10 {
        return q; // No rotation
    }

    let half_theta = omega_norm * dt * 0.5;
    let s = half_theta.sin() / omega_norm;
    let c = half_theta.cos();

    // Quaternion representing rotation: [cos(θ/2), sin(θ/2) * axis]
    let dq = [c, s * wx, s * wy, s * wz];

    // q_new = dq * q (rotation applied in body frame)
    let q_new = quat_mul(dq, q);

    // Normalize to prevent drift
    quat_normalize(q_new)
}

/// Convert quaternion to 3x3 rotation matrix.
/// Returns row-major [r00, r01, r02, r10, r11, r12, r20, r21, r22]
#[inline(always)]
pub fn quat_to_rotation_matrix(q: [f32; 4]) -> [f32; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        1.0 - 2.0 * (yy + zz), // r00
        2.0 * (xy - wz),       // r01
        2.0 * (xz + wy),       // r02
        2.0 * (xy + wz),       // r10
        1.0 - 2.0 * (xx + zz), // r11
        2.0 * (yz - wx),       // r12
        2.0 * (xz - wy),       // r20
        2.0 * (yz + wx),       // r21
        1.0 - 2.0 * (xx + yy), // r22
    ]
}

/// Get the Z column of the rotation matrix (thrust direction in world frame).
/// This is the direction the drone's thrust points.
#[inline(always)]
pub fn quat_to_thrust_dir(q: [f32; 4]) -> [f32; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        2.0 * (x * z + w * y), // r02
        2.0 * (y * z - w * x), // r12
        1.0 - 2.0 * (x * x + y * y), // r22
    ]
}

/// Convert quaternion to Euler angles (roll, pitch, yaw).
/// Uses ZYX convention (yaw-pitch-roll).
#[inline(always)]
pub fn quat_to_euler(q: [f32; 4]) -> [f32; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);

    // Roll (x-axis rotation)
    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr_cosp.atan2(cosr_cosp);

    // Pitch (y-axis rotation)
    let sinp = 2.0 * (w * y - z * x);
    let pitch = if sinp.abs() >= 1.0 {
        (PI / 2.0).copysign(sinp) // Gimbal lock
    } else {
        sinp.asin()
    };

    // Yaw (z-axis rotation)
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny_cosp.atan2(cosy_cosp);

    [roll, pitch, yaw]
}

/// Convert Euler angles (roll, pitch, yaw) to quaternion.
/// Uses ZYX convention (yaw-pitch-roll).
#[inline(always)]
pub fn euler_to_quat(rpy: [f32; 3]) -> [f32; 4] {
    let (roll, pitch, yaw) = (rpy[0], rpy[1], rpy[2]);

    let cr = (roll * 0.5).cos();
    let sr = (roll * 0.5).sin();
    let cp = (pitch * 0.5).cos();
    let sp = (pitch * 0.5).sin();
    let cy = (yaw * 0.5).cos();
    let sy = (yaw * 0.5).sin();

    [
        cr * cp * cy + sr * sp * sy, // w
        sr * cp * cy - cr * sp * sy, // x
        cr * sp * cy + sr * cp * sy, // y
        cr * cp * sy - sr * sp * cy, // z
    ]
}

// ============================================================================
// Type-Safe Wrappers (Public API)
// ============================================================================

use crate::types::{BodyFrame, Rotation, UnitQuaternion, Vec3, WorldFrame};

/// Integrate a unit quaternion with angular velocity using Rodrigues formula.
///
/// # Arguments
/// * `q` - Current orientation (unit quaternion)
/// * `omega` - Angular velocity in body frame
/// * `dt` - Time step
///
/// # Returns
/// New orientation after integration (normalized)
#[inline]
pub fn integrate_quaternion(
    q: UnitQuaternion,
    omega: Vec3<BodyFrame>,
    dt: f32,
) -> UnitQuaternion {
    let result = quat_integrate(q.as_array(), omega.as_array(), dt);
    // SAFETY: quat_integrate always returns a normalized quaternion
    unsafe { UnitQuaternion::new_unchecked(result[0], result[1], result[2], result[3]) }
}

/// Get thrust direction (body Z-axis in world frame) from quaternion.
///
/// # Returns
/// Unit vector pointing in the thrust direction (world frame)
#[inline]
pub fn thrust_direction_typed(q: UnitQuaternion) -> Vec3<WorldFrame> {
    let dir = quat_to_thrust_dir(q.as_array());
    Vec3::from_array(dir)
}

/// Convert quaternion to Euler angles [roll, pitch, yaw].
///
/// Uses ZYX convention (yaw-pitch-roll).
#[inline]
pub fn quaternion_to_euler(q: UnitQuaternion) -> [f32; 3] {
    quat_to_euler(q.as_array())
}

/// Convert Euler angles to unit quaternion.
///
/// Uses ZYX convention (yaw-pitch-roll).
#[inline]
pub fn euler_to_quaternion(roll: f32, pitch: f32, yaw: f32) -> UnitQuaternion {
    let q = euler_to_quat([roll, pitch, yaw]);
    // Euler to quat produces normalized quaternion by construction
    UnitQuaternion::from_array(q)
}

/// Convert quaternion to rotation matrix.
///
/// # Returns
/// Rotation matrix for body-to-world transformations
#[inline]
pub fn quaternion_to_rotation(q: UnitQuaternion) -> Rotation {
    let matrix = quat_to_rotation_matrix(q.as_array());
    Rotation::from_matrix_unchecked(matrix)
}

// ============================================================================
// SIMD Operations (process 8 quaternions at a time)
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    use crate::physics::simd_helpers::*;

    /// Normalize 8 quaternions simultaneously.
    #[inline(always)]
    pub fn quat_normalize_simd(
        qw: f32x8,
        qx: f32x8,
        qy: f32x8,
        qz: f32x8,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        let norm_sq = qw * qw + qx * qx + qy * qy + qz * qz;
        let inv_norm = simd_rsqrt(norm_sq);

        (qw * inv_norm, qx * inv_norm, qy * inv_norm, qz * inv_norm)
    }

    /// Integrate 8 quaternions with angular velocities.
    #[inline(always)]
    pub fn quat_integrate_simd(
        qw: f32x8,
        qx: f32x8,
        qy: f32x8,
        qz: f32x8,
        wx: f32x8,
        wy: f32x8,
        wz: f32x8,
        dt: f32,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        let dt_vec = f32x8::splat(dt);
        let half = f32x8::splat(0.5);

        // Compute omega magnitude
        let omega_sq = wx * wx + wy * wy + wz * wz;
        let omega_norm = omega_sq.sqrt();

        // Half angle
        let half_theta = omega_norm * dt_vec * half;

        // sin(half_theta) / omega_norm, with protection for small omega
        let epsilon = f32x8::splat(1e-10);
        let safe_omega = omega_norm.simd_max(epsilon);
        let s = simd_sin_small(half_theta) / safe_omega;
        let c = simd_cos_small(half_theta);

        // Delta quaternion
        let dqw = c;
        let dqx = s * wx;
        let dqy = s * wy;
        let dqz = s * wz;

        // Quaternion multiplication: dq * q
        let new_qw = dqw * qw - dqx * qx - dqy * qy - dqz * qz;
        let new_qx = dqw * qx + dqx * qw + dqy * qz - dqz * qy;
        let new_qy = dqw * qy - dqx * qz + dqy * qw + dqz * qx;
        let new_qz = dqw * qz + dqx * qy - dqy * qx + dqz * qw;

        // Normalize
        quat_normalize_simd(new_qw, new_qx, new_qy, new_qz)
    }

    /// Get thrust direction (Z column of rotation matrix) for 8 quaternions.
    #[inline(always)]
    pub fn quat_to_thrust_dir_simd(
        qw: f32x8,
        qx: f32x8,
        qy: f32x8,
        qz: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let two = f32x8::splat(2.0);
        let one = f32x8::splat(1.0);

        let r02 = two * (qx * qz + qw * qy);
        let r12 = two * (qy * qz - qw * qx);
        let r22 = one - two * (qx * qx + qy * qy);

        (r02, r12, r22)
    }

    /// Convert 8 quaternions to Euler angles.
    #[inline(always)]
    pub fn quat_to_euler_simd(
        qw: f32x8,
        qx: f32x8,
        qy: f32x8,
        qz: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let two = f32x8::splat(2.0);
        let one = f32x8::splat(1.0);

        // Roll
        let sinr_cosp = two * (qw * qx + qy * qz);
        let cosr_cosp = one - two * (qx * qx + qy * qy);
        let roll = simd_atan2(sinr_cosp, cosr_cosp);

        // Pitch
        let sinp = two * (qw * qy - qz * qx);
        let pitch = simd_asin_clamped(sinp);

        // Yaw
        let siny_cosp = two * (qw * qz + qx * qy);
        let cosy_cosp = one - two * (qy * qy + qz * qz);
        let yaw = simd_atan2(siny_cosp, cosy_cosp);

        (roll, pitch, yaw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_normalize() {
        let q = [2.0, 0.0, 0.0, 0.0];
        let n = quat_normalize(q);
        assert!((n[0] - 1.0).abs() < 1e-6);
        assert!(n[1].abs() < 1e-6);
    }

    #[test]
    fn test_quat_identity_multiply() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let q = [0.7071, 0.7071, 0.0, 0.0];
        let result = quat_mul(identity, q);
        for i in 0..4 {
            assert!((result[i] - q[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_quat_integrate_small_omega() {
        let q = [1.0, 0.0, 0.0, 0.0];
        let omega = [0.001, 0.0, 0.0];
        let q_new = quat_integrate(q, omega, 0.01);
        // Should be very close to identity
        assert!((q_new[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_euler_quat_roundtrip() {
        let rpy = [0.1, 0.2, 0.3];
        let q = euler_to_quat(rpy);
        let rpy_back = quat_to_euler(q);
        for i in 0..3 {
            assert!((rpy[i] - rpy_back[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_thrust_dir_identity() {
        let q = [1.0, 0.0, 0.0, 0.0];
        let dir = quat_to_thrust_dir(q);
        // Identity should give [0, 0, 1] (thrust up)
        assert!(dir[0].abs() < 1e-6);
        assert!(dir[1].abs() < 1e-6);
        assert!((dir[2] - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // Type-Safe Wrapper Tests
    // ========================================================================

    #[test]
    fn test_integrate_quaternion_typed() {
        let q = UnitQuaternion::identity();
        let omega: Vec3<BodyFrame> = Vec3::new(0.1, 0.0, 0.0);
        let q_new = integrate_quaternion(q, omega, 0.01);
        // Should still be normalized
        assert!((q_new.norm_squared() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_thrust_direction_typed() {
        let q = UnitQuaternion::identity();
        let thrust = thrust_direction_typed(q);
        // Identity should give [0, 0, 1] (thrust up in world frame)
        assert!(thrust.x().abs() < 1e-6);
        assert!(thrust.y().abs() < 1e-6);
        assert!((thrust.z() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euler_quaternion_roundtrip_typed() {
        let q = euler_to_quaternion(0.1, 0.2, 0.3);
        let euler = quaternion_to_euler(q);
        assert!((euler[0] - 0.1).abs() < 1e-5);
        assert!((euler[1] - 0.2).abs() < 1e-5);
        assert!((euler[2] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_quaternion_to_rotation() {
        let q = UnitQuaternion::identity();
        let rot = quaternion_to_rotation(q);
        // Identity quaternion should give identity rotation
        let matrix = rot.as_matrix();
        assert!((matrix[0] - 1.0).abs() < 1e-6);
        assert!((matrix[4] - 1.0).abs() < 1e-6);
        assert!((matrix[8] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_frame_transforms() {
        let q = euler_to_quaternion(0.0, 0.0, std::f32::consts::FRAC_PI_2); // 90 deg yaw
        let rot = quaternion_to_rotation(q);

        let v_world: Vec3<WorldFrame> = Vec3::new(1.0, 0.0, 0.0);
        let v_body = rot.world_to_body(v_world);

        // 90 deg CCW yaw: world X maps to body +Y (body rotated means world X appears as body +Y)
        assert!(v_body.x().abs() < 1e-5, "x: {}", v_body.x());
        assert!((v_body.y().abs() - 1.0).abs() < 1e-5, "y: {}", v_body.y());
        assert!(v_body.z().abs() < 1e-5, "z: {}", v_body.z());
    }
}
