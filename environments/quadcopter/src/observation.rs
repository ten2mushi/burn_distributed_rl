//! Observation generation from SoA state to AoS observation buffer.
//!
//! Converts internal state representation to observation vectors
//! based on the configured observation components.

use crate::config::ObsConfig;
use crate::constants::rpm_to_action;
use crate::state::QuadcopterState;

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat, StdFloat};

// ============================================================================
// Scalar Implementation
// ============================================================================

/// Write observation for a single environment to a buffer.
///
/// # Arguments
/// * `state` - Current state
/// * `idx` - Environment index
/// * `config` - Observation configuration
/// * `output` - Output buffer (must be at least `config.observation_size()` long)
///
/// # Returns
/// Number of elements written
#[inline]
pub fn write_observation(
    state: &QuadcopterState,
    idx: usize,
    config: &ObsConfig,
    output: &mut [f32],
) -> usize {
    let mut offset = 0;

    // Position [x, y, z]
    if config.position {
        output[offset] = state.pos_x[idx];
        output[offset + 1] = state.pos_y[idx];
        output[offset + 2] = state.pos_z[idx];
        offset += 3;
    }

    // Quaternion [w, x, y, z]
    if config.quaternion {
        output[offset] = state.quat_w[idx];
        output[offset + 1] = state.quat_x[idx];
        output[offset + 2] = state.quat_y[idx];
        output[offset + 3] = state.quat_z[idx];
        offset += 4;
    }

    // Euler angles [roll, pitch, yaw]
    if config.euler {
        let quat = state.get_quaternion_typed(idx);
        let euler = crate::physics::quaternion::quaternion_to_euler(quat);
        output[offset] = euler[0];
        output[offset + 1] = euler[1];
        output[offset + 2] = euler[2];
        offset += 3;
    }

    // Velocity [vx, vy, vz]
    if config.velocity {
        output[offset] = state.vel_x[idx];
        output[offset + 1] = state.vel_y[idx];
        output[offset + 2] = state.vel_z[idx];
        offset += 3;
    }

    // Angular velocity [wx, wy, wz]
    if config.angular_velocity {
        output[offset] = state.ang_vel_x[idx];
        output[offset + 1] = state.ang_vel_y[idx];
        output[offset + 2] = state.ang_vel_z[idx];
        offset += 3;
    }

    // Last action (normalized RPMs)
    if config.last_action {
        let base = idx * 4;
        for i in 0..4 {
            output[offset + i] = rpm_to_action(state.last_rpm[base + i]);
        }
        offset += 4;
    }

    // Target position [tx, ty, tz]
    if config.target_position {
        let base = idx * 3;
        output[offset] = state.target_pos[base];
        output[offset + 1] = state.target_pos[base + 1];
        output[offset + 2] = state.target_pos[base + 2];
        offset += 3;
    }

    // Target velocity [tvx, tvy, tvz]
    if config.target_velocity {
        let base = idx * 3;
        output[offset] = state.target_vel[base];
        output[offset + 1] = state.target_vel[base + 1];
        output[offset + 2] = state.target_vel[base + 2];
        offset += 3;
    }

    // Action buffer (history)
    if config.action_buffer_len > 0 {
        let buf_size = config.action_buffer_len * 4;
        state.get_action_buffer(idx, &mut output[offset..offset + buf_size]);
        offset += buf_size;
    }

    offset
}

/// Write observations for all environments.
///
/// # Arguments
/// * `state` - Current state
/// * `config` - Observation configuration
/// * `output` - Output buffer [num_envs * obs_size]
pub fn write_observations_all(
    state: &QuadcopterState,
    config: &ObsConfig,
    output: &mut [f32],
) {
    let obs_size = config.observation_size();

    for idx in 0..state.num_envs {
        let start = idx * obs_size;
        let end = start + obs_size;
        write_observation(state, idx, config, &mut output[start..end]);
    }
}

// ============================================================================
// SIMD Implementation
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;
    use crate::physics::simd_helpers::*;

    /// Write position observations for 8 environments.
    #[inline]
    fn write_position_simd(
        state: &QuadcopterState,
        base_idx: usize,
        output: &mut [f32],
        obs_size: usize,
    ) {
        let pos_x = f32x8::from_slice(&state.pos_x[base_idx..]);
        let pos_y = f32x8::from_slice(&state.pos_y[base_idx..]);
        let pos_z = f32x8::from_slice(&state.pos_z[base_idx..]);

        // Scatter to output (AoS format)
        for i in 0..8 {
            let out_base = i * obs_size;
            output[out_base] = pos_x.as_array()[i];
            output[out_base + 1] = pos_y.as_array()[i];
            output[out_base + 2] = pos_z.as_array()[i];
        }
    }

    /// Write quaternion observations for 8 environments.
    #[inline]
    fn write_quaternion_simd(
        state: &QuadcopterState,
        base_idx: usize,
        output: &mut [f32],
        obs_size: usize,
        offset: usize,
    ) {
        let qw = f32x8::from_slice(&state.quat_w[base_idx..]);
        let qx = f32x8::from_slice(&state.quat_x[base_idx..]);
        let qy = f32x8::from_slice(&state.quat_y[base_idx..]);
        let qz = f32x8::from_slice(&state.quat_z[base_idx..]);

        for i in 0..8 {
            let out_base = i * obs_size + offset;
            output[out_base] = qw.as_array()[i];
            output[out_base + 1] = qx.as_array()[i];
            output[out_base + 2] = qy.as_array()[i];
            output[out_base + 3] = qz.as_array()[i];
        }
    }

    /// Write Euler angle observations for 8 environments.
    #[inline]
    fn write_euler_simd(
        state: &QuadcopterState,
        base_idx: usize,
        output: &mut [f32],
        obs_size: usize,
        offset: usize,
    ) {
        use crate::physics::quaternion::simd::quat_to_euler_simd;

        let qw = f32x8::from_slice(&state.quat_w[base_idx..]);
        let qx = f32x8::from_slice(&state.quat_x[base_idx..]);
        let qy = f32x8::from_slice(&state.quat_y[base_idx..]);
        let qz = f32x8::from_slice(&state.quat_z[base_idx..]);

        let (roll, pitch, yaw) = quat_to_euler_simd(qw, qx, qy, qz);

        for i in 0..8 {
            let out_base = i * obs_size + offset;
            output[out_base] = roll.as_array()[i];
            output[out_base + 1] = pitch.as_array()[i];
            output[out_base + 2] = yaw.as_array()[i];
        }
    }

    /// Write velocity observations for 8 environments.
    #[inline]
    fn write_velocity_simd(
        state: &QuadcopterState,
        base_idx: usize,
        output: &mut [f32],
        obs_size: usize,
        offset: usize,
    ) {
        let vx = f32x8::from_slice(&state.vel_x[base_idx..]);
        let vy = f32x8::from_slice(&state.vel_y[base_idx..]);
        let vz = f32x8::from_slice(&state.vel_z[base_idx..]);

        for i in 0..8 {
            let out_base = i * obs_size + offset;
            output[out_base] = vx.as_array()[i];
            output[out_base + 1] = vy.as_array()[i];
            output[out_base + 2] = vz.as_array()[i];
        }
    }

    /// Write angular velocity observations for 8 environments.
    #[inline]
    fn write_angular_velocity_simd(
        state: &QuadcopterState,
        base_idx: usize,
        output: &mut [f32],
        obs_size: usize,
        offset: usize,
    ) {
        let wx = f32x8::from_slice(&state.ang_vel_x[base_idx..]);
        let wy = f32x8::from_slice(&state.ang_vel_y[base_idx..]);
        let wz = f32x8::from_slice(&state.ang_vel_z[base_idx..]);

        for i in 0..8 {
            let out_base = i * obs_size + offset;
            output[out_base] = wx.as_array()[i];
            output[out_base + 1] = wy.as_array()[i];
            output[out_base + 2] = wz.as_array()[i];
        }
    }

    /// Write observations for 8 environments using SIMD loads where possible.
    ///
    /// Note: Scatter operations still require scalar writes due to AoS output format.
    #[inline]
    pub fn write_observations_simd(
        state: &QuadcopterState,
        base_idx: usize,
        config: &ObsConfig,
        output: &mut [f32],
    ) {
        let obs_size = config.observation_size();
        let mut offset = 0;

        // Position
        if config.position {
            write_position_simd(state, base_idx, output, obs_size);
            offset += 3;
        }

        // Quaternion
        if config.quaternion {
            write_quaternion_simd(state, base_idx, output, obs_size, offset);
            offset += 4;
        }

        // Euler angles
        if config.euler {
            write_euler_simd(state, base_idx, output, obs_size, offset);
            offset += 3;
        }

        // Velocity
        if config.velocity {
            write_velocity_simd(state, base_idx, output, obs_size, offset);
            offset += 3;
        }

        // Angular velocity
        if config.angular_velocity {
            write_angular_velocity_simd(state, base_idx, output, obs_size, offset);
            offset += 3;
        }

        // Last action, targets, and action buffer use scalar for simplicity
        // (these are less performance-critical)
        for i in 0..8 {
            let idx = base_idx + i;
            let out_base = i * obs_size + offset;
            let mut local_offset = 0;

            if config.last_action {
                let rpm_base = idx * 4;
                for j in 0..4 {
                    output[out_base + local_offset + j] =
                        rpm_to_action(state.last_rpm[rpm_base + j]);
                }
                local_offset += 4;
            }

            if config.target_position {
                let tgt_base = idx * 3;
                output[out_base + local_offset] = state.target_pos[tgt_base];
                output[out_base + local_offset + 1] = state.target_pos[tgt_base + 1];
                output[out_base + local_offset + 2] = state.target_pos[tgt_base + 2];
                local_offset += 3;
            }

            if config.target_velocity {
                let tgt_base = idx * 3;
                output[out_base + local_offset] = state.target_vel[tgt_base];
                output[out_base + local_offset + 1] = state.target_vel[tgt_base + 1];
                output[out_base + local_offset + 2] = state.target_vel[tgt_base + 2];
                local_offset += 3;
            }

            if config.action_buffer_len > 0 {
                let buf_size = config.action_buffer_len * 4;
                state.get_action_buffer(
                    idx,
                    &mut output[out_base + local_offset..out_base + local_offset + buf_size],
                );
            }
        }
    }

    /// Write observations for all environments using SIMD.
    pub fn write_observations_all_simd(
        state: &QuadcopterState,
        config: &ObsConfig,
        output: &mut [f32],
    ) {
        let obs_size = config.observation_size();
        let chunks = state.num_envs / 8;
        let remainder = state.num_envs % 8;

        // Process full SIMD chunks
        for chunk in 0..chunks {
            let base_idx = chunk * 8;
            let out_start = base_idx * obs_size;
            write_observations_simd(
                state,
                base_idx,
                config,
                &mut output[out_start..out_start + 8 * obs_size],
            );
        }

        // Handle remainder with scalar
        let base = chunks * 8;
        for i in 0..remainder {
            let idx = base + i;
            let start = idx * obs_size;
            write_observation(state, idx, config, &mut output[start..start + obs_size]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::InitConfig;

    #[test]
    fn test_observation_size() {
        let config = ObsConfig::kinematic(); // pos + euler + vel + ang_vel = 12
        assert_eq!(config.observation_size(), 12);
    }

    #[test]
    fn test_write_observation_position() {
        let mut state = QuadcopterState::new(1, 0);
        state.pos_x[0] = 1.0;
        state.pos_y[0] = 2.0;
        state.pos_z[0] = 3.0;

        let config = ObsConfig::new().with_position();
        let mut output = vec![0.0; 3];
        write_observation(&state, 0, &config, &mut output);

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 3.0);
    }

    #[test]
    fn test_write_observation_euler() {
        let mut state = QuadcopterState::new(1, 0);
        // Identity quaternion
        state.quat_w[0] = 1.0;
        state.quat_x[0] = 0.0;
        state.quat_y[0] = 0.0;
        state.quat_z[0] = 0.0;

        let config = ObsConfig::new().with_euler();
        let mut output = vec![0.0; 3];
        write_observation(&state, 0, &config, &mut output);

        // Identity quaternion should give zero Euler angles
        assert!(output[0].abs() < 1e-6, "Roll should be ~0");
        assert!(output[1].abs() < 1e-6, "Pitch should be ~0");
        assert!(output[2].abs() < 1e-6, "Yaw should be ~0");
    }

    #[test]
    fn test_write_observation_kinematic() {
        let state = QuadcopterState::new(1, 0);
        let config = ObsConfig::kinematic();
        let mut output = vec![0.0; config.observation_size()];

        let written = write_observation(&state, 0, &config, &mut output);
        assert_eq!(written, 12);
    }

    #[test]
    fn test_write_observations_all() {
        let state = QuadcopterState::new(4, 0);
        let config = ObsConfig::new().with_position();
        let mut output = vec![0.0; 4 * 3];

        write_observations_all(&state, &config, &mut output);

        // Check each environment's z position (should be 1.0)
        for i in 0..4 {
            assert_eq!(output[i * 3 + 2], 1.0);
        }
    }
}
