//! Stylized quadcopter shape rendering primitives.

use crate::physics::quaternion::quat_to_rotation_matrix;

/// Quadcopter shape definition.
/// Uses X-configuration with motors at 45° angles.
pub struct DroneShape {
    /// Arm length (half of wingspan).
    pub arm_length: f32,
    /// Body radius.
    pub body_radius: f32,
    /// Motor radius.
    pub motor_radius: f32,
}

impl Default for DroneShape {
    fn default() -> Self {
        Self {
            arm_length: 0.15,
            body_radius: 0.05,
            motor_radius: 0.03,
        }
    }
}

impl DroneShape {
    /// Create a drone shape with specified wingspan.
    pub fn with_wingspan(wingspan: f32) -> Self {
        let arm = wingspan * 0.5;
        Self {
            arm_length: arm,
            body_radius: arm * 0.25,
            motor_radius: arm * 0.2,
        }
    }
}

/// Motor positions in body frame (X-configuration).
/// Layout: Front-Right, Rear-Right, Rear-Left, Front-Left
/// Returns positions relative to center of mass.
pub fn motor_positions_body(arm_length: f32) -> [[f32; 3]; 4] {
    let d = arm_length * std::f32::consts::FRAC_1_SQRT_2; // 45° offset
    [
        [d, -d, 0.0],  // Motor 0: Front-Right (+X, -Y)
        [-d, -d, 0.0], // Motor 1: Rear-Right (-X, -Y)
        [-d, d, 0.0],  // Motor 2: Rear-Left (-X, +Y)
        [d, d, 0.0],   // Motor 3: Front-Left (+X, +Y)
    ]
}

/// Transform motor positions from body frame to world frame.
pub fn motor_positions_world(
    center: [f32; 3],
    quaternion: [f32; 4],
    arm_length: f32,
) -> [[f32; 3]; 4] {
    let body_positions = motor_positions_body(arm_length);
    let rot = quat_to_rotation_matrix(quaternion);

    let mut world_positions = [[0.0f32; 3]; 4];
    for (i, bp) in body_positions.iter().enumerate() {
        let rotated = rotate_vector(&rot, bp);
        world_positions[i] = [
            center[0] + rotated[0],
            center[1] + rotated[1],
            center[2] + rotated[2],
        ];
    }
    world_positions
}

/// Get body frame axes in world coordinates.
/// Returns (right, forward, up) vectors.
pub fn body_axes_world(quaternion: [f32; 4]) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let rot = quat_to_rotation_matrix(quaternion);

    // X axis (right in body frame)
    let right = [rot[0], rot[3], rot[6]];
    // Y axis (forward in body frame)
    let forward = [rot[1], rot[4], rot[7]];
    // Z axis (up in body frame)
    let up = [rot[2], rot[5], rot[8]];

    (right, forward, up)
}

/// Get heading direction (forward) in world frame.
pub fn heading_direction(quaternion: [f32; 4]) -> [f32; 3] {
    let rot = quat_to_rotation_matrix(quaternion);
    [rot[1], rot[4], rot[7]] // Y column = forward
}

/// Get thrust direction (up) in world frame.
pub fn thrust_direction(quaternion: [f32; 4]) -> [f32; 3] {
    let rot = quat_to_rotation_matrix(quaternion);
    [rot[2], rot[5], rot[8]] // Z column = up
}

/// Rotate a vector by rotation matrix.
fn rotate_vector(rot: &[f32; 9], v: &[f32; 3]) -> [f32; 3] {
    [
        rot[0] * v[0] + rot[1] * v[1] + rot[2] * v[2],
        rot[3] * v[0] + rot[4] * v[1] + rot[5] * v[2],
        rot[6] * v[0] + rot[7] * v[1] + rot[8] * v[2],
    ]
}

/// Arm line endpoints for rendering.
/// Returns pairs of (start, end) points for the 4 arms.
pub fn arm_lines_world(
    center: [f32; 3],
    quaternion: [f32; 4],
    arm_length: f32,
) -> [([f32; 3], [f32; 3]); 4] {
    let motors = motor_positions_world(center, quaternion, arm_length);
    [
        (center, motors[0]),
        (center, motors[1]),
        (center, motors[2]),
        (center, motors[3]),
    ]
}

/// Compute visual scale factor based on distance.
/// Ensures drones don't become too small or too large.
pub fn distance_scale(distance: f32) -> f32 {
    // Clamp scale between 0.5 and 2.0
    let scale = 5.0 / distance.max(2.5);
    scale.clamp(0.5, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_positions() {
        let positions = motor_positions_body(1.0);
        // All motors should be at same distance from center
        for pos in &positions {
            let dist = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
            assert!((dist - 1.0).abs() < 0.01);
        }
        // All at z=0
        for pos in &positions {
            assert!(pos[2].abs() < 0.001);
        }
    }

    #[test]
    fn test_identity_quaternion() {
        let identity = [1.0, 0.0, 0.0, 0.0];
        let (right, forward, up) = body_axes_world(identity);

        // Identity should give standard axes
        assert!((right[0] - 1.0).abs() < 0.01);
        assert!((forward[1] - 1.0).abs() < 0.01);
        assert!((up[2] - 1.0).abs() < 0.01);
    }
}
