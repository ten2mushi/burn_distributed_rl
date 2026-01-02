//! Crazyflie 2.0 X-configuration physical constants.
//!
//! All values derived from cf2x.urdf and PyBullet gym-pybullet-drones.

use std::f32::consts::PI;

// ============================================================================
// Physical Properties
// ============================================================================

/// Mass of the drone (kg)
pub const M: f32 = 0.027;

/// Arm length - distance from center to motor (m)
pub const L: f32 = 0.0397;

/// Gravitational acceleration (m/s^2)
pub const G: f32 = 9.8;

/// Moment of inertia around X axis (kg*m^2)
pub const IXX: f32 = 1.4e-5;

/// Moment of inertia around Y axis (kg*m^2)
pub const IYY: f32 = 1.4e-5;

/// Moment of inertia around Z axis (kg*m^2)
pub const IZZ: f32 = 2.17e-5;

/// Thrust coefficient: F = KF * rpm^2 (N/RPM^2)
pub const KF: f32 = 3.16e-10;

/// Torque coefficient: tau = KM * rpm^2 (N*m/RPM^2)
pub const KM: f32 = 7.94e-12;

/// Propeller radius (m)
pub const PROP_RADIUS: f32 = 0.0231348;

/// Propeller + rotor moment of inertia (kg*m^2)
/// Used for gyroscopic precession computation.
/// Estimated from Crazyflie propeller (~0.3g, 46mm span) plus motor rotor.
/// This is intentionally conservative to avoid numerical instability.
pub const PROP_INERTIA: f32 = 3.0e-7;

/// RPM to rad/s conversion: omega = RPM * PI / 30
pub const PROP_VEL_COEFF: f32 = PI / 30.0;

/// Motor time constant for first-order dynamics (seconds)
/// Typical value for brushless micro motors: 10-20ms
pub const MOTOR_TIME_CONSTANT: f32 = 0.015;

// ============================================================================
// Derived Constants
// ============================================================================

/// Weight force (N) = M * G
pub const GRAVITY_FORCE: f32 = M * G;

/// Hover RPM: sqrt(GRAVITY_FORCE / (4 * KF)) ≈ 14469
pub const HOVER_RPM: f32 = 14468.5;

/// Maximum RPM for thrust-to-weight ratio of 2.25 ≈ 21703
pub const MAX_RPM: f32 = 21702.75;

/// Arm length divided by sqrt(2) for X-configuration torque computation
pub const L_SQRT2: f32 = 0.02808; // L / sqrt(2) = 0.0397 / 1.4142

/// Inverse of moment of inertia X
pub const IXX_INV: f32 = 1.0 / IXX; // ≈ 71428.57

/// Inverse of moment of inertia Y
pub const IYY_INV: f32 = 1.0 / IYY; // ≈ 71428.57

/// Inverse of moment of inertia Z
pub const IZZ_INV: f32 = 1.0 / IZZ; // ≈ 46082.95

/// Inverse of mass
pub const M_INV: f32 = 1.0 / M; // ≈ 37.037

// ============================================================================
// Aerodynamic Coefficients
// ============================================================================

/// Ground effect coefficient
pub const GND_EFF_COEFF: f32 = 11.36859;

/// XY plane drag coefficient
pub const DRAG_COEFF_XY: f32 = 9.1785e-7;

/// Z axis drag coefficient
pub const DRAG_COEFF_Z: f32 = 10.311e-7;

/// Downwash amplitude coefficient (for multi-drone)
pub const DW_COEFF_1: f32 = 2267.18;

/// Downwash vertical decay rate
pub const DW_COEFF_2: f32 = 0.16;

/// Downwash spatial spread offset
pub const DW_COEFF_3: f32 = -0.11;

// ============================================================================
// Default Simulation Parameters
// ============================================================================

/// Default physics frequency (Hz)
pub const DEFAULT_PHYSICS_FREQ: u32 = 240;

/// Default control frequency (Hz)
pub const DEFAULT_CTRL_FREQ: u32 = 30;

/// Default maximum episode steps at 30 Hz control (8 seconds)
pub const DEFAULT_MAX_STEPS: u32 = 240;

/// Minimum height for ground effect calculation (prevents singularity)
pub const GND_EFF_H_CLIP: f32 = 0.01;

/// Height above which ground effect is negligible (m)
pub const GND_EFF_MAX_HEIGHT: f32 = 0.5;

// ============================================================================
// Action Space Bounds
// ============================================================================

/// Minimum RPM (motors can't run negative)
pub const MIN_RPM: f32 = 0.0;

/// Two times PI for quaternion integration
pub const TWO_PI: f32 = 2.0 * PI;

// ============================================================================
// Motor Configuration (X-configuration)
// ============================================================================
// Motor layout (top view):
//
//      Front
//        ^
//        |
//    3-------0
//     \     /
//      \ X /  ← Center
//       / \
//      /   \
//    2-------1
//       Rear
//
// Motor 0 (Front-Right): CCW
// Motor 1 (Rear-Right):  CCW
// Motor 2 (Rear-Left):   CW
// Motor 3 (Front-Left):  CW
//
// Propeller positions relative to CoM:
// Motor 0: (+L/sqrt(2), -L/sqrt(2), 0)
// Motor 1: (-L/sqrt(2), -L/sqrt(2), 0)
// Motor 2: (-L/sqrt(2), +L/sqrt(2), 0)
// Motor 3: (+L/sqrt(2), +L/sqrt(2), 0)

/// Motor reaction torque directions for yaw computation.
/// CCW motor creates CW body reaction (-1), CW motor creates CCW body reaction (+1).
/// Motors 0,1 are CCW, Motors 2,3 are CW.
pub const MOTOR_DIRS: [f32; 4] = [-1.0, -1.0, 1.0, 1.0];

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert normalized action [-1, 1] to RPM [0, MAX_RPM]
#[inline(always)]
pub fn action_to_rpm(action: f32) -> f32 {
    ((action + 1.0) * 0.5 * MAX_RPM).clamp(MIN_RPM, MAX_RPM)
}

/// Convert RPM to normalized action [-1, 1]
#[inline(always)]
pub fn rpm_to_action(rpm: f32) -> f32 {
    (rpm / MAX_RPM) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hover_rpm_calculation() {
        // Verify HOVER_RPM is correct: at hover, total thrust = weight
        // 4 * KF * HOVER_RPM^2 = M * G
        let computed_hover = ((M * G) / (4.0 * KF)).sqrt();
        assert!((computed_hover - HOVER_RPM).abs() < 10.0);
    }

    #[test]
    fn test_max_rpm_calculation() {
        // Verify MAX_RPM gives thrust-to-weight ratio of 2.25
        let max_thrust = 4.0 * KF * MAX_RPM * MAX_RPM;
        let thrust_to_weight = max_thrust / GRAVITY_FORCE;
        assert!((thrust_to_weight - 2.25).abs() < 0.1);
    }

    #[test]
    fn test_action_to_rpm_bounds() {
        assert_eq!(action_to_rpm(-1.0), 0.0);
        assert!((action_to_rpm(0.0) - MAX_RPM * 0.5).abs() < 1.0);
        assert_eq!(action_to_rpm(1.0), MAX_RPM);
    }

    #[test]
    fn test_rpm_to_action_inverse() {
        for rpm in [0.0, HOVER_RPM, MAX_RPM] {
            let action = rpm_to_action(rpm);
            let rpm_back = action_to_rpm(action);
            assert!((rpm - rpm_back).abs() < 1.0);
        }
    }

    #[test]
    fn test_l_sqrt2() {
        let expected = L / 2.0_f32.sqrt();
        assert!((L_SQRT2 - expected).abs() < 1e-4);
    }
}
