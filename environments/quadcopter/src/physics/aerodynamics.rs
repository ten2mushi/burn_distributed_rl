//! Aerodynamic effects for quadcopter simulation.
//!
//! Implements:
//! - Ground effect (increased thrust efficiency near ground)
//! - Velocity-dependent drag forces
//!
//! Based on PyBullet gym-pybullet-drones implementation.

use crate::constants::*;
use crate::types::{BodyFrame, Rotation, Vec3, WorldFrame};

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat, StdFloat};

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Compute ground effect thrust multiplier.
///
/// Ground effect increases thrust efficiency when the drone is close to the ground.
/// Based on PyBullet gym-pybullet-drones:
/// k_ge = 1 + GND_EFF_COEFF * (r_prop / (4 * z))^2
///
/// # Arguments
/// * `height` - Height above ground (m)
///
/// # Returns
/// Thrust multiplier (>= 1.0)
#[inline(always)]
pub fn ground_effect_multiplier(height: f32) -> f32 {
    if height > GND_EFF_MAX_HEIGHT {
        return 1.0;
    }

    // Clamp height to prevent division by zero
    let h = height.max(GND_EFF_H_CLIP);

    // Ground effect formula from PyBullet
    let ratio = PROP_RADIUS / (4.0 * h);
    let ratio_sq = ratio * ratio;

    // Multiplier: 1 + GND_EFF_COEFF * ratio^2
    1.0 + GND_EFF_COEFF * ratio_sq
}

/// Compute extra thrust from ground effect for all 4 motors.
///
/// # Arguments
/// * `height` - Height above ground (m)
/// * `thrusts` - Nominal thrust from each motor [F0, F1, F2, F3]
///
/// # Returns
/// Additional thrust per motor due to ground effect
#[inline(always)]
pub fn ground_effect_thrust(height: f32, thrusts: [f32; 4]) -> [f32; 4] {
    let multiplier = ground_effect_multiplier(height);
    let factor = multiplier - 1.0; // Extra factor (0 means no extra thrust)

    [
        thrusts[0] * factor,
        thrusts[1] * factor,
        thrusts[2] * factor,
        thrusts[3] * factor,
    ]
}

/// Compute drag force in body frame.
///
/// Drag is proportional to velocity squared and motor RPMs.
/// F_drag = -C_d * v^2 * sign(v) * (sum of RPM)
///
/// # Arguments
/// * `velocity` - Velocity in body frame [vx, vy, vz]
/// * `total_rpm` - Sum of all motor RPMs
///
/// # Returns
/// Drag force [Fx, Fy, Fz] in body frame (opposing velocity)
#[inline(always)]
pub fn compute_drag_force(velocity: [f32; 3], total_rpm: f32) -> [f32; 3] {
    let (vx, vy, vz) = (velocity[0], velocity[1], velocity[2]);

    // XY plane drag (lateral movement)
    let drag_xy = DRAG_COEFF_XY * total_rpm;

    // Z axis drag (vertical movement)
    let drag_z = DRAG_COEFF_Z * total_rpm;

    [
        -drag_xy * vx * vx.abs(),
        -drag_xy * vy * vy.abs(),
        -drag_z * vz * vz.abs(),
    ]
}

/// Transform velocity from world frame to body frame.
///
/// v_body = R^T * v_world
///
/// # Arguments
/// * `v_world` - Velocity in world frame
/// * `rotation` - Rotation matrix (row-major, 9 elements)
///
/// # Returns
/// Velocity in body frame
#[inline(always)]
pub fn world_to_body_velocity(v_world: [f32; 3], rotation: [f32; 9]) -> [f32; 3] {
    // R^T * v = [R[0,0] R[1,0] R[2,0]] [vx]
    //           [R[0,1] R[1,1] R[2,1]] [vy]
    //           [R[0,2] R[1,2] R[2,2]] [vz]
    [
        rotation[0] * v_world[0] + rotation[3] * v_world[1] + rotation[6] * v_world[2],
        rotation[1] * v_world[0] + rotation[4] * v_world[1] + rotation[7] * v_world[2],
        rotation[2] * v_world[0] + rotation[5] * v_world[1] + rotation[8] * v_world[2],
    ]
}

/// Transform drag force from body frame to world frame.
///
/// F_world = R * F_body
///
/// # Arguments
/// * `f_body` - Force in body frame
/// * `rotation` - Rotation matrix (row-major, 9 elements)
///
/// # Returns
/// Force in world frame
#[inline(always)]
pub fn body_to_world_force(f_body: [f32; 3], rotation: [f32; 9]) -> [f32; 3] {
    [
        rotation[0] * f_body[0] + rotation[1] * f_body[1] + rotation[2] * f_body[2],
        rotation[3] * f_body[0] + rotation[4] * f_body[1] + rotation[5] * f_body[2],
        rotation[6] * f_body[0] + rotation[7] * f_body[1] + rotation[8] * f_body[2],
    ]
}

/// Compute drag acceleration in world frame.
///
/// # Arguments
/// * `v_world` - Velocity in world frame
/// * `rotation` - Rotation matrix from body to world
/// * `total_rpm` - Sum of all motor RPMs
///
/// # Returns
/// Drag acceleration [ax, ay, az] in world frame
#[inline(always)]
pub fn compute_drag_acceleration(
    v_world: [f32; 3],
    rotation: [f32; 9],
    total_rpm: f32,
) -> [f32; 3] {
    // Transform velocity to body frame
    let v_body = world_to_body_velocity(v_world, rotation);

    // Compute drag in body frame
    let f_drag_body = compute_drag_force(v_body, total_rpm);

    // Transform back to world frame
    let f_drag_world = body_to_world_force(f_drag_body, rotation);

    // Convert to acceleration (a = F/m)
    [
        f_drag_world[0] * M_INV,
        f_drag_world[1] * M_INV,
        f_drag_world[2] * M_INV,
    ]
}

// ============================================================================
// Type-Safe Wrappers (Public API)
// ============================================================================

/// Compute drag force in body frame (typed version).
///
/// # Arguments
/// * `velocity` - Velocity in body frame
/// * `total_rpm` - Sum of all motor RPMs
///
/// # Returns
/// Drag force in body frame (opposing velocity)
#[inline]
pub fn compute_drag_force_typed(velocity: Vec3<BodyFrame>, total_rpm: f32) -> Vec3<BodyFrame> {
    let drag = compute_drag_force(velocity.as_array(), total_rpm);
    Vec3::from_array(drag)
}

/// Compute drag acceleration in world frame (typed version).
///
/// This performs the full transformation pipeline:
/// 1. Transform velocity from world to body frame
/// 2. Compute drag force in body frame
/// 3. Transform drag force back to world frame
/// 4. Convert to acceleration
///
/// # Arguments
/// * `v_world` - Velocity in world frame
/// * `rotation` - Rotation from body to world frame
/// * `total_rpm` - Sum of all motor RPMs
///
/// # Returns
/// Drag acceleration in world frame
#[inline]
pub fn compute_drag_acceleration_typed(
    v_world: Vec3<WorldFrame>,
    rotation: &Rotation,
    total_rpm: f32,
) -> Vec3<WorldFrame> {
    // Transform velocity to body frame
    let v_body = rotation.world_to_body(v_world);

    // Compute drag in body frame
    let f_drag_body = compute_drag_force_typed(v_body, total_rpm);

    // Transform back to world frame and convert to acceleration
    let f_drag_world = rotation.body_to_world(f_drag_body);
    f_drag_world * M_INV
}

// ============================================================================
// SIMD Implementation
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    use std::simd::cmp::SimdPartialOrd;

    /// Compute ground effect multiplier for 8 environments.
    /// Uses PyBullet formula: 1 + GND_EFF_COEFF * (r_prop / (4 * z))^2
    #[inline(always)]
    pub fn ground_effect_multiplier_simd(height: f32x8) -> f32x8 {
        let max_height = f32x8::splat(GND_EFF_MAX_HEIGHT);
        let min_height = f32x8::splat(GND_EFF_H_CLIP);
        let one = f32x8::splat(1.0);
        let prop_radius = f32x8::splat(PROP_RADIUS);
        let four = f32x8::splat(4.0);
        let gnd_eff_coeff = f32x8::splat(GND_EFF_COEFF);

        // Clamp height
        let h = height.simd_clamp(min_height, max_height);

        // Check if above max height
        let above_max = height.simd_gt(max_height);

        // Compute ratio^2
        let ratio = prop_radius / (four * h);
        let ratio_sq = ratio * ratio;

        // Multiplier: 1 + GND_EFF_COEFF * ratio^2
        let mult = one + gnd_eff_coeff * ratio_sq;

        // No effect above max height
        above_max.select(one, mult)
    }

    /// Compute extra ground effect thrust for 8 environments × 4 motors.
    #[inline(always)]
    pub fn ground_effect_thrust_simd(
        height: f32x8,
        t0: f32x8,
        t1: f32x8,
        t2: f32x8,
        t3: f32x8,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        let mult = ground_effect_multiplier_simd(height);
        let factor = mult - f32x8::splat(1.0);

        (t0 * factor, t1 * factor, t2 * factor, t3 * factor)
    }

    /// Compute drag force in body frame for 8 environments.
    #[inline(always)]
    pub fn compute_drag_force_simd(
        vx: f32x8,
        vy: f32x8,
        vz: f32x8,
        total_rpm: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        let drag_xy = f32x8::splat(DRAG_COEFF_XY) * total_rpm;
        let drag_z = f32x8::splat(DRAG_COEFF_Z) * total_rpm;

        (
            -drag_xy * vx * vx.abs(),
            -drag_xy * vy * vy.abs(),
            -drag_z * vz * vz.abs(),
        )
    }

    /// Transform velocity from world to body frame for 8 environments.
    /// Rotation matrix elements are passed as separate SIMD vectors.
    #[inline(always)]
    pub fn world_to_body_velocity_simd(
        vx: f32x8,
        vy: f32x8,
        vz: f32x8,
        r00: f32x8,
        r01: f32x8,
        r02: f32x8,
        r10: f32x8,
        r11: f32x8,
        r12: f32x8,
        r20: f32x8,
        r21: f32x8,
        r22: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        // R^T * v
        (
            r00 * vx + r10 * vy + r20 * vz,
            r01 * vx + r11 * vy + r21 * vz,
            r02 * vx + r12 * vy + r22 * vz,
        )
    }

    /// Transform force from body to world frame for 8 environments.
    #[inline(always)]
    pub fn body_to_world_force_simd(
        fx: f32x8,
        fy: f32x8,
        fz: f32x8,
        r00: f32x8,
        r01: f32x8,
        r02: f32x8,
        r10: f32x8,
        r11: f32x8,
        r12: f32x8,
        r20: f32x8,
        r21: f32x8,
        r22: f32x8,
    ) -> (f32x8, f32x8, f32x8) {
        // R * f
        (
            r00 * fx + r01 * fy + r02 * fz,
            r10 * fx + r11 * fy + r12 * fz,
            r20 * fx + r21 * fy + r22 * fz,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_effect_at_ground() {
        // Very close to ground should have large multiplier
        let mult = ground_effect_multiplier(0.02);
        assert!(mult > 1.5, "Ground effect should be strong near ground: {}", mult);
    }

    #[test]
    fn test_ground_effect_high() {
        // High above ground should have no effect
        let mult = ground_effect_multiplier(1.0);
        assert!((mult - 1.0).abs() < 1e-6, "No ground effect when high: {}", mult);
    }

    #[test]
    fn test_ground_effect_monotonic() {
        // Multiplier should decrease as height increases
        let mut prev_mult = ground_effect_multiplier(0.01);
        for h in [0.05, 0.1, 0.2, 0.3, 0.5] {
            let mult = ground_effect_multiplier(h);
            assert!(mult < prev_mult, "Ground effect should decrease with height");
            prev_mult = mult;
        }
    }

    #[test]
    fn test_drag_opposes_velocity() {
        let velocity = [1.0, 0.5, -0.3];
        let total_rpm = 4.0 * HOVER_RPM;
        let drag = compute_drag_force(velocity, total_rpm);

        // Drag should oppose velocity
        assert!(drag[0] < 0.0, "Drag should oppose positive vx");
        assert!(drag[1] < 0.0, "Drag should oppose positive vy");
        assert!(drag[2] > 0.0, "Drag should oppose negative vz");
    }

    #[test]
    fn test_drag_zero_at_rest() {
        let velocity = [0.0, 0.0, 0.0];
        let total_rpm = 4.0 * HOVER_RPM;
        let drag = compute_drag_force(velocity, total_rpm);

        assert!(drag[0].abs() < 1e-10);
        assert!(drag[1].abs() < 1e-10);
        assert!(drag[2].abs() < 1e-10);
    }

    #[test]
    fn test_world_body_transform_identity() {
        // Identity rotation should not change velocity
        let v_world = [1.0, 2.0, 3.0];
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let v_body = world_to_body_velocity(v_world, identity);

        for i in 0..3 {
            assert!((v_world[i] - v_body[i]).abs() < 1e-6);
        }
    }
}
