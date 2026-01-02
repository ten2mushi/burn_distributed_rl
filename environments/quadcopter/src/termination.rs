//! Episode termination and truncation checking.
//!
//! Provides functions to check various termination conditions:
//! - Position bounds violation
//! - Attitude bounds (flip detection)
//! - Velocity bounds
//! - Ground collision
//! - Maximum episode length

use crate::config::TerminationConfig;
use crate::state::QuadcopterState;

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat, Mask, StdFloat};

// ============================================================================
// Termination Result
// ============================================================================

/// Result of termination check for a single environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerminationResult {
    /// True if episode ended due to failure (e.g., crash, flip)
    pub terminal: bool,
    /// True if episode ended due to time limit
    pub truncated: bool,
}

impl TerminationResult {
    /// Episode is done (either terminal or truncated).
    #[inline]
    pub fn done(&self) -> bool {
        self.terminal || self.truncated
    }
}

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Check if environment is terminated due to failure.
#[inline]
pub fn check_terminal(
    state: &QuadcopterState,
    idx: usize,
    config: &TerminationConfig,
) -> bool {
    // Check ground collision
    if config.ground_collision && state.pos_z[idx] < 0.0 {
        return true;
    }

    // Check position bounds
    if let Some(bounds) = &config.position_bounds {
        let x = state.pos_x[idx];
        let y = state.pos_y[idx];
        let z = state.pos_z[idx];

        if x < bounds[0] || x > bounds[1]
            || y < bounds[2] || y > bounds[3]
            || z < bounds[4] || z > bounds[5]
        {
            return true;
        }
    }

    // Check attitude bounds (roll/pitch only, not yaw)
    if let Some(max_angle) = config.attitude_bounds {
        let quat = state.get_quaternion_typed(idx);
        let euler = crate::physics::quaternion::quaternion_to_euler(quat);
        let roll = euler[0];
        let pitch = euler[1];

        if roll.abs() > max_angle || pitch.abs() > max_angle {
            return true;
        }
    }

    // Check velocity bounds
    if let Some(max_vel) = config.velocity_bounds {
        let vel_sq = state.vel_x[idx].powi(2)
            + state.vel_y[idx].powi(2)
            + state.vel_z[idx].powi(2);

        if vel_sq > max_vel * max_vel {
            return true;
        }
    }

    false
}

/// Check if environment is truncated due to time limit.
#[inline]
pub fn check_truncated(
    state: &QuadcopterState,
    idx: usize,
    config: &TerminationConfig,
) -> bool {
    state.step_count[idx] >= config.max_steps
}

/// Check both terminal and truncated conditions.
#[inline]
pub fn check_termination(
    state: &QuadcopterState,
    idx: usize,
    config: &TerminationConfig,
) -> TerminationResult {
    TerminationResult {
        terminal: check_terminal(state, idx, config),
        truncated: check_truncated(state, idx, config),
    }
}

/// Check termination for all environments (scalar version).
pub fn check_termination_all(
    state: &QuadcopterState,
    config: &TerminationConfig,
    terminals: &mut [bool],
    truncateds: &mut [bool],
) {
    for idx in 0..state.num_envs {
        let result = check_termination(state, idx, config);
        terminals[idx] = result.terminal;
        truncateds[idx] = result.truncated;
    }
}

// ============================================================================
// SIMD Implementation
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    use crate::physics::simd_helpers::simd_asin_clamped;
    use std::simd::cmp::SimdPartialOrd;

    /// Check termination for 8 environments at once.
    ///
    /// Returns (terminal_mask, truncated_mask) as packed u8 bitmasks.
    #[inline]
    pub fn check_termination_simd(
        state: &QuadcopterState,
        base_idx: usize,
        config: &TerminationConfig,
    ) -> (u8, u8) {
        let zero = f32x8::splat(0.0);
        let mut terminal_mask = Mask::<i32, 8>::from_array([false; 8]);

        // Load state
        let pos_x = f32x8::from_slice(&state.pos_x[base_idx..]);
        let pos_y = f32x8::from_slice(&state.pos_y[base_idx..]);
        let pos_z = f32x8::from_slice(&state.pos_z[base_idx..]);

        // Check ground collision
        if config.ground_collision {
            let ground_hit = pos_z.simd_lt(zero);
            terminal_mask |= ground_hit;
        }

        // Check position bounds
        if let Some(bounds) = &config.position_bounds {
            let x_min = f32x8::splat(bounds[0]);
            let x_max = f32x8::splat(bounds[1]);
            let y_min = f32x8::splat(bounds[2]);
            let y_max = f32x8::splat(bounds[3]);
            let z_min = f32x8::splat(bounds[4]);
            let z_max = f32x8::splat(bounds[5]);

            let x_oob = pos_x.simd_lt(x_min) | pos_x.simd_gt(x_max);
            let y_oob = pos_y.simd_lt(y_min) | pos_y.simd_gt(y_max);
            let z_oob = pos_z.simd_lt(z_min) | pos_z.simd_gt(z_max);

            terminal_mask |= x_oob | y_oob | z_oob;
        }

        // Check attitude bounds
        if let Some(max_angle) = config.attitude_bounds {
            let qw = f32x8::from_slice(&state.quat_w[base_idx..]);
            let qx = f32x8::from_slice(&state.quat_x[base_idx..]);
            let qy = f32x8::from_slice(&state.quat_y[base_idx..]);
            let qz = f32x8::from_slice(&state.quat_z[base_idx..]);

            // Compute roll and pitch from quaternion
            let two = f32x8::splat(2.0);
            let one = f32x8::splat(1.0);

            // Roll
            let sinr_cosp = two * (qw * qx + qy * qz);
            let cosr_cosp = one - two * (qx * qx + qy * qy);
            // Approximate atan2 magnitude check: |roll| > max_angle
            // For small angles, |atan2(s,c)| ≈ |s/c| when c > 0
            // More robust: check if sin^2 > sin^2(max_angle) * (sin^2 + cos^2)
            let sin_max = max_angle.sin();
            let sin_max_sq = f32x8::splat(sin_max * sin_max);
            let roll_sq = sinr_cosp * sinr_cosp;
            let roll_total = roll_sq + cosr_cosp * cosr_cosp;
            let roll_exceeded = roll_sq.simd_gt(sin_max_sq * roll_total);

            // Pitch
            let sinp = two * (qw * qy - qz * qx);
            let sinp_clamped = sinp.simd_clamp(-one, one);
            let pitch_exceeded = sinp_clamped.abs().simd_gt(f32x8::splat(sin_max));

            terminal_mask |= roll_exceeded | pitch_exceeded;
        }

        // Check velocity bounds
        if let Some(max_vel) = config.velocity_bounds {
            let vel_x = f32x8::from_slice(&state.vel_x[base_idx..]);
            let vel_y = f32x8::from_slice(&state.vel_y[base_idx..]);
            let vel_z = f32x8::from_slice(&state.vel_z[base_idx..]);

            let vel_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
            let max_vel_sq = f32x8::splat(max_vel * max_vel);

            let vel_exceeded = vel_sq.simd_gt(max_vel_sq);
            terminal_mask |= vel_exceeded;
        }

        // Check truncation (step count >= max_steps)
        let max_steps = config.max_steps;
        let mut truncated_array = [false; 8];
        for i in 0..8 {
            truncated_array[i] = state.step_count[base_idx + i] >= max_steps;
        }
        let truncated_mask = Mask::<i32, 8>::from_array(truncated_array);

        // Convert masks to u8 bitmaps
        (terminal_mask.to_bitmask() as u8, truncated_mask.to_bitmask() as u8)
    }

    /// Check termination for all environments using SIMD.
    pub fn check_termination_all_simd(
        state: &QuadcopterState,
        config: &TerminationConfig,
        terminals: &mut [bool],
        truncateds: &mut [bool],
    ) {
        let chunks = state.num_envs / 8;
        let remainder = state.num_envs % 8;

        // Process full SIMD chunks
        for chunk in 0..chunks {
            let base_idx = chunk * 8;
            let (term_bits, trunc_bits) = check_termination_simd(state, base_idx, config);

            for i in 0..8 {
                terminals[base_idx + i] = (term_bits >> i) & 1 != 0;
                truncateds[base_idx + i] = (trunc_bits >> i) & 1 != 0;
            }
        }

        // Handle remainder with scalar
        let base = chunks * 8;
        for i in 0..remainder {
            let idx = base + i;
            let result = check_termination(state, idx, config);
            terminals[idx] = result.terminal;
            truncateds[idx] = result.truncated;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InitConfig;

    fn create_test_state() -> QuadcopterState {
        QuadcopterState::new(8, 0)
    }

    #[test]
    fn test_no_termination_at_start() {
        let state = create_test_state();
        let config = TerminationConfig::default();

        for idx in 0..state.num_envs {
            let result = check_termination(&state, idx, &config);
            assert!(!result.terminal, "Should not be terminal at start");
            assert!(!result.truncated, "Should not be truncated at start");
        }
    }

    #[test]
    fn test_ground_collision() {
        let mut state = create_test_state();
        state.pos_z[0] = -0.1; // Below ground

        let config = TerminationConfig::default();
        let result = check_termination(&state, 0, &config);

        assert!(result.terminal, "Should be terminal on ground collision");
    }

    #[test]
    fn test_position_bounds() {
        let mut state = create_test_state();
        state.pos_x[0] = 10.0; // Out of default bounds

        let config = TerminationConfig::default();
        let result = check_termination(&state, 0, &config);

        assert!(result.terminal, "Should be terminal when out of bounds");
    }

    #[test]
    fn test_flip_detection() {
        let mut state = create_test_state();
        // Set quaternion for 90 degree roll
        state.quat_w[0] = 0.7071;
        state.quat_x[0] = 0.7071;
        state.quat_y[0] = 0.0;
        state.quat_z[0] = 0.0;

        let config = TerminationConfig::default().with_attitude_bounds(0.5); // ~28 degrees
        let result = check_termination(&state, 0, &config);

        assert!(result.terminal, "Should be terminal when flipped");
    }

    #[test]
    fn test_truncation() {
        let mut state = create_test_state();
        state.step_count[0] = 300; // Exceeds default max_steps

        let config = TerminationConfig::default();
        let result = check_termination(&state, 0, &config);

        assert!(result.truncated, "Should be truncated at max steps");
    }

    #[test]
    fn test_no_bounds_disabled() {
        let mut state = create_test_state();
        state.pos_x[0] = 100.0; // Way out of typical bounds

        let config = TerminationConfig::default()
            .without_position_bounds()
            .with_ground_collision(false);

        let result = check_termination(&state, 0, &config);
        assert!(!result.terminal, "Should not terminate with bounds disabled");
    }
}
