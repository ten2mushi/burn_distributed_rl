//! Motor dynamics simulation with first-order response.
//!
//! Real motors cannot change RPM instantaneously due to inertia.
//! This module implements first-order dynamics:
//!
//! d(RPM)/dt = (RPM_cmd - RPM_actual) / tau
//!
//! Discretized as:
//! RPM(t+dt) = alpha * RPM_cmd + (1-alpha) * RPM(t)
//! where alpha = dt / (dt + tau)

use crate::constants::{MAX_RPM, MIN_RPM, MOTOR_TIME_CONSTANT};

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

// ============================================================================
// Configuration
// ============================================================================

/// Motor dynamics configuration.
///
/// Set `time_constant = 0.0` for instantaneous response (legacy behavior).
/// Default is realistic ~15ms response time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorDynamicsConfig {
    /// Time constant in seconds.
    /// - 0.0 = instantaneous (no dynamics)
    /// - 0.015 = typical brushless micro motor (~15ms)
    pub time_constant: f32,
}

impl Default for MotorDynamicsConfig {
    fn default() -> Self {
        Self::realistic()
    }
}

impl MotorDynamicsConfig {
    /// Instantaneous motor response (no dynamics, legacy behavior).
    #[inline]
    pub const fn instantaneous() -> Self {
        Self { time_constant: 0.0 }
    }

    /// Realistic Crazyflie motor response (~15ms time constant).
    #[inline]
    pub const fn realistic() -> Self {
        Self {
            time_constant: MOTOR_TIME_CONSTANT,
        }
    }

    /// Custom time constant.
    #[inline]
    pub const fn with_time_constant(time_constant: f32) -> Self {
        Self { time_constant }
    }

    /// Check if motor dynamics are enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.time_constant > 0.0
    }

    /// Compute alpha coefficient for given timestep.
    /// alpha = dt / (dt + tau)
    #[inline]
    pub fn compute_alpha(&self, dt: f32) -> f32 {
        if self.time_constant <= 0.0 {
            1.0 // Instantaneous: alpha = 1 means new_rpm = cmd_rpm
        } else {
            dt / (dt + self.time_constant)
        }
    }
}

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Apply first-order motor dynamics to a single motor.
///
/// Returns the new actual RPM after applying dynamics.
///
/// # Arguments
/// * `cmd_rpm` - Commanded RPM from action
/// * `actual_rpm` - Current actual RPM
/// * `alpha` - Precomputed coefficient: dt / (dt + tau)
#[inline(always)]
pub fn apply_motor_dynamics(cmd_rpm: f32, actual_rpm: f32, alpha: f32) -> f32 {
    let new_rpm = alpha * cmd_rpm + (1.0 - alpha) * actual_rpm;
    new_rpm.clamp(MIN_RPM, MAX_RPM)
}

/// Apply motor dynamics to all 4 motors of one environment.
///
/// # Arguments
/// * `cmd_rpms` - Commanded RPMs [motor0, motor1, motor2, motor3]
/// * `actual_rpms` - Current actual RPMs (mutated in place)
/// * `alpha` - Precomputed coefficient: dt / (dt + tau)
#[inline]
pub fn apply_motor_dynamics_quad(cmd_rpms: [f32; 4], actual_rpms: &mut [f32; 4], alpha: f32) {
    for i in 0..4 {
        actual_rpms[i] = apply_motor_dynamics(cmd_rpms[i], actual_rpms[i], alpha);
    }
}

/// Get effective RPMs for physics computation.
///
/// If motor dynamics are enabled, returns actual RPMs after applying dynamics.
/// If disabled (tau = 0), returns commanded RPMs directly.
#[inline]
pub fn get_effective_rpms(
    cmd_rpms: [f32; 4],
    actual_rpms: &mut [f32; 4],
    config: &MotorDynamicsConfig,
    dt: f32,
) -> [f32; 4] {
    if config.is_enabled() {
        let alpha = config.compute_alpha(dt);
        apply_motor_dynamics_quad(cmd_rpms, actual_rpms, alpha);
        *actual_rpms
    } else {
        cmd_rpms
    }
}

// ============================================================================
// SIMD Implementation
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;

    /// Apply motor dynamics for 8 environments (one motor position at a time).
    ///
    /// # Arguments
    /// * `cmd_rpm` - Commanded RPMs for 8 environments
    /// * `actual_rpm` - Current actual RPMs for 8 environments
    /// * `alpha` - Precomputed coefficient (broadcast to SIMD)
    #[inline(always)]
    pub fn apply_motor_dynamics_simd(cmd_rpm: f32x8, actual_rpm: f32x8, alpha: f32x8) -> f32x8 {
        let one = f32x8::splat(1.0);
        let min_rpm = f32x8::splat(MIN_RPM);
        let max_rpm = f32x8::splat(MAX_RPM);

        let new_rpm = alpha * cmd_rpm + (one - alpha) * actual_rpm;
        new_rpm.simd_clamp(min_rpm, max_rpm)
    }

    /// Apply motor dynamics for all 4 motors across 8 environments.
    ///
    /// # Arguments
    /// * `cmd_rpm0-3` - Commanded RPMs for each motor (8 environments each)
    /// * `actual_rpm0-3` - Current actual RPMs (mutated in place)
    /// * `alpha` - Precomputed coefficient
    #[inline]
    pub fn apply_motor_dynamics_quad_simd(
        cmd_rpm0: f32x8,
        cmd_rpm1: f32x8,
        cmd_rpm2: f32x8,
        cmd_rpm3: f32x8,
        actual_rpm0: &mut f32x8,
        actual_rpm1: &mut f32x8,
        actual_rpm2: &mut f32x8,
        actual_rpm3: &mut f32x8,
        alpha: f32x8,
    ) {
        *actual_rpm0 = apply_motor_dynamics_simd(cmd_rpm0, *actual_rpm0, alpha);
        *actual_rpm1 = apply_motor_dynamics_simd(cmd_rpm1, *actual_rpm1, alpha);
        *actual_rpm2 = apply_motor_dynamics_simd(cmd_rpm2, *actual_rpm2, alpha);
        *actual_rpm3 = apply_motor_dynamics_simd(cmd_rpm3, *actual_rpm3, alpha);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = MotorDynamicsConfig::default();
        assert!(config.is_enabled());
        assert!((config.time_constant - MOTOR_TIME_CONSTANT).abs() < 1e-6);
    }

    #[test]
    fn test_config_instantaneous() {
        let config = MotorDynamicsConfig::instantaneous();
        assert!(!config.is_enabled());
        assert_eq!(config.time_constant, 0.0);
    }

    #[test]
    fn test_alpha_computation() {
        let config = MotorDynamicsConfig::with_time_constant(0.015);
        let dt = 1.0 / 240.0; // 240 Hz physics

        let alpha = config.compute_alpha(dt);

        // alpha = dt / (dt + tau) = (1/240) / (1/240 + 0.015) ≈ 0.217
        assert!(alpha > 0.0 && alpha < 1.0);
        assert!((alpha - (dt / (dt + 0.015))).abs() < 1e-6);
    }

    #[test]
    fn test_alpha_instantaneous() {
        let config = MotorDynamicsConfig::instantaneous();
        let alpha = config.compute_alpha(1.0 / 240.0);
        assert_eq!(alpha, 1.0); // Instantaneous: alpha = 1
    }

    #[test]
    fn test_motor_dynamics_steady_state() {
        // At steady state (cmd == actual), output should equal input
        let cmd = 15000.0;
        let actual = 15000.0;
        let alpha = 0.5;

        let result = apply_motor_dynamics(cmd, actual, alpha);
        assert!((result - cmd).abs() < 1e-6);
    }

    #[test]
    fn test_motor_dynamics_step_response() {
        // Step from hover to max RPM
        let mut actual = 14468.5; // HOVER_RPM
        let cmd = MAX_RPM;
        let config = MotorDynamicsConfig::realistic();
        let dt = 1.0 / 240.0;
        let alpha = config.compute_alpha(dt);

        // Run for ~5 time constants (should reach ~99% of target)
        let steps = (5.0 * config.time_constant / dt) as usize;
        for _ in 0..steps {
            actual = apply_motor_dynamics(cmd, actual, alpha);
        }

        // Should be very close to commanded
        assert!((actual - cmd).abs() / cmd < 0.02, "Actual: {}, Cmd: {}", actual, cmd);
    }

    #[test]
    fn test_motor_dynamics_63_percent_at_tau() {
        // After one time constant, should reach ~63.2% of the change
        let actual_initial = 0.0;
        let cmd = 10000.0;
        let tau = 0.015;
        let config = MotorDynamicsConfig::with_time_constant(tau);

        // Run for exactly tau seconds
        let dt = 0.001; // 1ms
        let steps = (tau / dt) as usize;
        let alpha = config.compute_alpha(dt);

        let mut actual = actual_initial;
        for _ in 0..steps {
            actual = apply_motor_dynamics(cmd, actual, alpha);
        }

        // 1 - e^(-1) ≈ 0.632
        let expected_fraction = 1.0 - (-1.0_f32).exp();
        let actual_fraction = actual / cmd;

        assert!(
            (actual_fraction - expected_fraction).abs() < 0.05,
            "Expected ~63.2%, got {:.1}%",
            actual_fraction * 100.0
        );
    }

    #[test]
    fn test_motor_dynamics_clamping() {
        // Test that output is clamped to valid range
        let result_low = apply_motor_dynamics(-1000.0, 0.0, 1.0);
        assert_eq!(result_low, MIN_RPM);

        let result_high = apply_motor_dynamics(100000.0, 0.0, 1.0);
        assert_eq!(result_high, MAX_RPM);
    }

    #[test]
    fn test_quad_dynamics() {
        let cmd = [15000.0, 15500.0, 14500.0, 15000.0];
        let mut actual = [14468.5; 4];
        let alpha = 0.3;

        apply_motor_dynamics_quad(cmd, &mut actual, alpha);

        // Each motor should have moved toward its command
        for i in 0..4 {
            assert!(actual[i] != 14468.5);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_scalar() {
        use std::simd::f32x8;

        let cmd_scalar = 15000.0;
        let actual_scalar = 14000.0;
        let alpha = 0.3;

        let result_scalar = apply_motor_dynamics(cmd_scalar, actual_scalar, alpha);

        let cmd_simd = f32x8::splat(cmd_scalar);
        let actual_simd = f32x8::splat(actual_scalar);
        let alpha_simd = f32x8::splat(alpha);

        let result_simd = simd::apply_motor_dynamics_simd(cmd_simd, actual_simd, alpha_simd);

        for i in 0..8 {
            assert!((result_simd[i] - result_scalar).abs() < 1e-6);
        }
    }
}
