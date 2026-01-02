//! Built-in reward components for quadcopter control.
//!
//! Each component is a zero-cost struct implementing [`RewardComponent`].

use super::RewardComponent;
use crate::constants::rpm_to_action;
#[cfg(test)]
use crate::constants::HOVER_RPM;
use crate::state::QuadcopterState;

#[cfg(feature = "simd")]
use crate::physics::simd_helpers::*;
#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat, StdFloat};

// ============================================================================
// Position Error Component
// ============================================================================

/// Penalizes squared distance from target position.
///
/// `reward -= weight * ||pos - target||^2`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionError {
    /// Weight for position error penalty (typically 1.0)
    pub weight: f32,
}

impl RewardComponent for PositionError {
    const NAME: &'static str = "PositionError";

    #[inline]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        if self.weight == 0.0 {
            return 0.0;
        }

        let target_base = idx * 3;
        let dx = state.pos_x[idx] - state.target_pos[target_base];
        let dy = state.pos_y[idx] - state.target_pos[target_base + 1];
        let dz = state.pos_z[idx] - state.target_pos[target_base + 2];
        let pos_error_sq = dx * dx + dy * dy + dz * dz;

        -self.weight * pos_error_sq
    }

    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        if self.weight == 0.0 {
            return f32x8::splat(0.0);
        }

        let pos_x = f32x8::from_slice(&state.pos_x[base_idx..]);
        let pos_y = f32x8::from_slice(&state.pos_y[base_idx..]);
        let pos_z = f32x8::from_slice(&state.pos_z[base_idx..]);

        // Gather targets
        let mut target_x = [0.0f32; 8];
        let mut target_y = [0.0f32; 8];
        let mut target_z = [0.0f32; 8];
        for i in 0..8 {
            let tgt_base = (base_idx + i) * 3;
            target_x[i] = state.target_pos[tgt_base];
            target_y[i] = state.target_pos[tgt_base + 1];
            target_z[i] = state.target_pos[tgt_base + 2];
        }
        let target_x = f32x8::from_array(target_x);
        let target_y = f32x8::from_array(target_y);
        let target_z = f32x8::from_array(target_z);

        let dx = pos_x - target_x;
        let dy = pos_y - target_y;
        let dz = pos_z - target_z;
        let pos_error_sq = dx * dx + dy * dy + dz * dz;

        -f32x8::splat(self.weight) * pos_error_sq
    }
}

// ============================================================================
// Velocity Error Component
// ============================================================================

/// Penalizes squared velocity difference from target velocity.
///
/// `reward -= weight * ||vel - target_vel||^2`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityError {
    /// Weight for velocity error penalty
    pub weight: f32,
}

impl RewardComponent for VelocityError {
    const NAME: &'static str = "VelocityError";

    #[inline]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        if self.weight == 0.0 {
            return 0.0;
        }

        let target_base = idx * 3;
        let dvx = state.vel_x[idx] - state.target_vel[target_base];
        let dvy = state.vel_y[idx] - state.target_vel[target_base + 1];
        let dvz = state.vel_z[idx] - state.target_vel[target_base + 2];
        let vel_error_sq = dvx * dvx + dvy * dvy + dvz * dvz;

        -self.weight * vel_error_sq
    }

    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        if self.weight == 0.0 {
            return f32x8::splat(0.0);
        }

        let vel_x = f32x8::from_slice(&state.vel_x[base_idx..]);
        let vel_y = f32x8::from_slice(&state.vel_y[base_idx..]);
        let vel_z = f32x8::from_slice(&state.vel_z[base_idx..]);

        let mut target_vx = [0.0f32; 8];
        let mut target_vy = [0.0f32; 8];
        let mut target_vz = [0.0f32; 8];
        for i in 0..8 {
            let tgt_base = (base_idx + i) * 3;
            target_vx[i] = state.target_vel[tgt_base];
            target_vy[i] = state.target_vel[tgt_base + 1];
            target_vz[i] = state.target_vel[tgt_base + 2];
        }
        let target_vx = f32x8::from_array(target_vx);
        let target_vy = f32x8::from_array(target_vy);
        let target_vz = f32x8::from_array(target_vz);

        let dvx = vel_x - target_vx;
        let dvy = vel_y - target_vy;
        let dvz = vel_z - target_vz;
        let vel_error_sq = dvx * dvx + dvy * dvy + dvz * dvz;

        -f32x8::splat(self.weight) * vel_error_sq
    }
}

// ============================================================================
// Attitude Penalty Component
// ============================================================================

/// Penalizes roll and pitch deviation from level.
///
/// `reward -= weight * (roll^2 + pitch^2)`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttitudePenalty {
    /// Weight for attitude penalty
    pub weight: f32,
}

impl RewardComponent for AttitudePenalty {
    const NAME: &'static str = "AttitudePenalty";

    #[inline]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        if self.weight == 0.0 {
            return 0.0;
        }

        let quat = state.get_quaternion_typed(idx);
        let euler = crate::physics::quaternion::quaternion_to_euler(quat);
        let roll = euler[0];
        let pitch = euler[1];
        let attitude_penalty = roll * roll + pitch * pitch;

        -self.weight * attitude_penalty
    }

    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        if self.weight == 0.0 {
            return f32x8::splat(0.0);
        }

        let qw = f32x8::from_slice(&state.quat_w[base_idx..]);
        let qx = f32x8::from_slice(&state.quat_x[base_idx..]);
        let qy = f32x8::from_slice(&state.quat_y[base_idx..]);
        let qz = f32x8::from_slice(&state.quat_z[base_idx..]);

        let two = f32x8::splat(2.0);
        let one = f32x8::splat(1.0);

        // Roll: atan2(2(qw*qx + qy*qz), 1 - 2(qx^2 + qy^2))
        let sinr_cosp = two * (qw * qx + qy * qz);
        let cosr_cosp = one - two * (qx * qx + qy * qy);
        let roll = simd_atan2(sinr_cosp, cosr_cosp);

        // Pitch: asin(2(qw*qy - qz*qx))
        let sinp = two * (qw * qy - qz * qx);
        let pitch = simd_asin_clamped(sinp);

        let attitude_penalty = roll * roll + pitch * pitch;
        -f32x8::splat(self.weight) * attitude_penalty
    }
}

// ============================================================================
// Angular Velocity Penalty Component
// ============================================================================

/// Penalizes angular velocity magnitude.
///
/// `reward -= weight * ||omega||^2`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularVelocityPenalty {
    /// Weight for angular velocity penalty
    pub weight: f32,
}

impl RewardComponent for AngularVelocityPenalty {
    const NAME: &'static str = "AngularVelocityPenalty";

    #[inline]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        if self.weight == 0.0 {
            return 0.0;
        }

        let wx = state.ang_vel_x[idx];
        let wy = state.ang_vel_y[idx];
        let wz = state.ang_vel_z[idx];
        let ang_vel_sq = wx * wx + wy * wy + wz * wz;

        -self.weight * ang_vel_sq
    }

    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        if self.weight == 0.0 {
            return f32x8::splat(0.0);
        }

        let wx = f32x8::from_slice(&state.ang_vel_x[base_idx..]);
        let wy = f32x8::from_slice(&state.ang_vel_y[base_idx..]);
        let wz = f32x8::from_slice(&state.ang_vel_z[base_idx..]);

        let ang_vel_sq = wx * wx + wy * wy + wz * wz;
        -f32x8::splat(self.weight) * ang_vel_sq
    }
}

// ============================================================================
// Action Magnitude Penalty Component
// ============================================================================

/// Penalizes deviation from hover thrust.
///
/// `reward -= weight * sum((action_i - hover_action)^2)`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionMagnitudePenalty {
    /// Weight for action magnitude penalty
    pub weight: f32,
    /// Reference RPM to penalize deviation from (typically HOVER_RPM)
    pub reference_rpm: f32,
}

impl RewardComponent for ActionMagnitudePenalty {
    const NAME: &'static str = "ActionMagnitudePenalty";

    #[inline]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        if self.weight == 0.0 {
            return 0.0;
        }

        let rpm_base = idx * 4;
        let hover_action = rpm_to_action(self.reference_rpm);
        let mut action_penalty = 0.0;

        for i in 0..4 {
            let action = rpm_to_action(state.last_rpm[rpm_base + i]);
            let diff = action - hover_action;
            action_penalty += diff * diff;
        }

        -self.weight * action_penalty
    }

    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        if self.weight == 0.0 {
            return f32x8::splat(0.0);
        }

        let hover_action = rpm_to_action(self.reference_rpm);
        let hover_vec = f32x8::splat(hover_action);
        let mut action_penalty = f32x8::splat(0.0);

        for motor in 0..4 {
            let mut rpms = [0.0f32; 8];
            for i in 0..8 {
                rpms[i] = rpm_to_action(state.last_rpm[(base_idx + i) * 4 + motor]);
            }
            let action = f32x8::from_array(rpms);
            let diff = action - hover_vec;
            action_penalty += diff * diff;
        }

        -f32x8::splat(self.weight) * action_penalty
    }
}

// ============================================================================
// Action Rate Penalty Component
// ============================================================================

/// Penalizes action changes (encourages smooth control).
///
/// `reward -= weight * sum((action_i - prev_action_i)^2)`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionRatePenalty {
    /// Weight for action rate penalty
    pub weight: f32,
}

impl RewardComponent for ActionRatePenalty {
    const NAME: &'static str = "ActionRatePenalty";

    #[inline]
    fn compute(&self, state: &QuadcopterState, idx: usize) -> f32 {
        if self.weight == 0.0 {
            return 0.0;
        }

        let base = idx * 4;
        let mut rate_penalty = 0.0;

        for i in 0..4 {
            let current = rpm_to_action(state.last_rpm[base + i]);
            let prev = state.prev_action[base + i];
            let diff = current - prev;
            rate_penalty += diff * diff;
        }

        -self.weight * rate_penalty
    }

    #[cfg(feature = "simd")]
    fn compute_simd(&self, state: &QuadcopterState, base_idx: usize) -> f32x8 {
        if self.weight == 0.0 {
            return f32x8::splat(0.0);
        }

        let mut rate_penalty = f32x8::splat(0.0);

        for motor in 0..4 {
            let mut current = [0.0f32; 8];
            let mut prev = [0.0f32; 8];
            for i in 0..8 {
                let idx = (base_idx + i) * 4 + motor;
                current[i] = rpm_to_action(state.last_rpm[idx]);
                prev[i] = state.prev_action[idx];
            }
            let current_vec = f32x8::from_array(current);
            let prev_vec = f32x8::from_array(prev);
            let diff = current_vec - prev_vec;
            rate_penalty += diff * diff;
        }

        -f32x8::splat(self.weight) * rate_penalty
    }
}

// ============================================================================
// Alive Bonus Component
// ============================================================================

/// Constant positive reward for surviving.
///
/// `reward += bonus`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AliveBonus {
    /// Bonus reward per step (typically 0.1)
    pub bonus: f32,
}

impl RewardComponent for AliveBonus {
    const NAME: &'static str = "AliveBonus";

    #[inline(always)]
    fn compute(&self, _state: &QuadcopterState, _idx: usize) -> f32 {
        self.bonus
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    fn compute_simd(&self, _state: &QuadcopterState, _base_idx: usize) -> f32x8 {
        f32x8::splat(self.bonus)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::QuadcopterState;

    fn create_test_state() -> QuadcopterState {
        let mut state = QuadcopterState::new(1, 0);
        // Set target at [0, 0, 1]
        state.target_pos[0] = 0.0;
        state.target_pos[1] = 0.0;
        state.target_pos[2] = 1.0;
        state
    }

    #[test]
    fn test_position_error_at_target() {
        let mut state = create_test_state();
        state.pos_x[0] = 0.0;
        state.pos_y[0] = 0.0;
        state.pos_z[0] = 1.0;

        let component = PositionError { weight: 1.0 };
        let reward = component.compute(&state, 0);

        assert!(reward.abs() < 1e-6, "At target, position error should be 0");
    }

    #[test]
    fn test_position_error_away_from_target() {
        let mut state = create_test_state();
        state.pos_x[0] = 1.0; // 1m away in x
        state.pos_y[0] = 0.0;
        state.pos_z[0] = 1.0;

        let component = PositionError { weight: 1.0 };
        let reward = component.compute(&state, 0);

        assert!((reward - (-1.0)).abs() < 1e-6, "Expected -1.0, got {}", reward);
    }

    #[test]
    fn test_position_error_zero_weight() {
        let state = create_test_state();
        let component = PositionError { weight: 0.0 };
        let reward = component.compute(&state, 0);

        assert_eq!(reward, 0.0, "Zero weight should return 0");
    }

    #[test]
    fn test_alive_bonus() {
        let state = create_test_state();
        let component = AliveBonus { bonus: 0.5 };
        let reward = component.compute(&state, 0);

        assert_eq!(reward, 0.5);
    }

    #[test]
    fn test_attitude_penalty_level() {
        let state = create_test_state(); // identity quaternion = level
        let component = AttitudePenalty { weight: 1.0 };
        let reward = component.compute(&state, 0);

        assert!(reward.abs() < 1e-5, "Level attitude should have ~0 penalty");
    }

    #[test]
    fn test_attitude_penalty_tilted() {
        use crate::physics::quaternion::euler_to_quat;

        let mut state = create_test_state();
        // 45 degree roll - use euler_to_quat for proper normalization
        let q = euler_to_quat([std::f32::consts::FRAC_PI_4, 0.0, 0.0]);
        state.quat_w[0] = q[0];
        state.quat_x[0] = q[1];
        state.quat_y[0] = q[2];
        state.quat_z[0] = q[3];

        let component = AttitudePenalty { weight: 1.0 };
        let reward = component.compute(&state, 0);

        assert!(reward < -0.1, "Tilted attitude should have negative reward");
    }

    #[test]
    fn test_angular_velocity_penalty() {
        let mut state = create_test_state();
        state.ang_vel_z[0] = 1.0; // 1 rad/s yaw rate

        let component = AngularVelocityPenalty { weight: 1.0 };
        let reward = component.compute(&state, 0);

        assert!((reward - (-1.0)).abs() < 1e-6, "Expected -1.0, got {}", reward);
    }

    #[test]
    fn test_action_magnitude_penalty_at_hover() {
        let state = create_test_state(); // RPMs initialized to HOVER_RPM
        let component = ActionMagnitudePenalty {
            weight: 1.0,
            reference_rpm: HOVER_RPM,
        };
        let reward = component.compute(&state, 0);

        assert!(
            reward.abs() < 1e-5,
            "At hover RPM, action penalty should be ~0"
        );
    }
}
