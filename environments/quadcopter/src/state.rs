//! Struct-of-Arrays (SoA) state layout for SIMD-efficient quadcopter simulation.
//!
//! All state variables are stored in separate contiguous arrays, enabling
//! efficient SIMD vectorization where 8 environments can be processed per
//! instruction (f32x8).

use crate::config::{DistributionType, InitConfig};
use crate::constants::*;
use crate::physics::quaternion::euler_to_quat;
use crate::types::{BodyFrame, UnitQuaternion, Vec3, WorldFrame};

/// SoA state storage for all parallel quadcopter environments.
///
/// Each field is a vector with one element per environment, enabling
/// SIMD operations to process 8 environments simultaneously.
pub struct QuadcopterState {
    // ========================================================================
    // Position [num_envs]
    // ========================================================================
    /// X position (m) in world frame
    pub pos_x: Vec<f32>,
    /// Y position (m) in world frame
    pub pos_y: Vec<f32>,
    /// Z position (m) in world frame (up is positive)
    pub pos_z: Vec<f32>,

    // ========================================================================
    // Quaternion [num_envs] - stored as [w, x, y, z]
    // ========================================================================
    /// Quaternion W component
    pub quat_w: Vec<f32>,
    /// Quaternion X component
    pub quat_x: Vec<f32>,
    /// Quaternion Y component
    pub quat_y: Vec<f32>,
    /// Quaternion Z component
    pub quat_z: Vec<f32>,

    // ========================================================================
    // Linear Velocity [num_envs]
    // ========================================================================
    /// X velocity (m/s) in world frame
    pub vel_x: Vec<f32>,
    /// Y velocity (m/s) in world frame
    pub vel_y: Vec<f32>,
    /// Z velocity (m/s) in world frame
    pub vel_z: Vec<f32>,

    // ========================================================================
    // Angular Velocity (body frame) [num_envs]
    // ========================================================================
    /// Roll rate (rad/s)
    pub ang_vel_x: Vec<f32>,
    /// Pitch rate (rad/s)
    pub ang_vel_y: Vec<f32>,
    /// Yaw rate (rad/s)
    pub ang_vel_z: Vec<f32>,

    // ========================================================================
    // Motor RPMs [num_envs * 4]
    // Layout: [env0_m0, env0_m1, env0_m2, env0_m3, env1_m0, ...]
    // ========================================================================
    /// Commanded motor RPMs (from action)
    pub last_rpm: Vec<f32>,
    /// Actual motor RPMs after dynamics (may lag commanded)
    pub actual_rpm: Vec<f32>,

    // ========================================================================
    // Previous Action (for action rate penalty) [num_envs * 4]
    // ========================================================================
    /// Previous normalized actions [-1, 1]
    pub prev_action: Vec<f32>,

    // ========================================================================
    // Action Buffer (for observation) [num_envs * buffer_len * 4]
    // Circular buffer of previous actions
    // ========================================================================
    /// Action history buffer
    pub action_buffer: Vec<f32>,
    /// Current write position in action buffer for each environment
    pub action_buffer_idx: Vec<usize>,

    // ========================================================================
    // External Targets [num_envs * 3]
    // ========================================================================
    /// Target position [x, y, z]
    pub target_pos: Vec<f32>,
    /// Target velocity [vx, vy, vz]
    pub target_vel: Vec<f32>,

    // ========================================================================
    // Episode Tracking [num_envs]
    // ========================================================================
    /// Current step count in episode
    pub step_count: Vec<u32>,
    /// Accumulated reward in current episode
    pub episode_reward: Vec<f32>,

    // ========================================================================
    // Metadata
    // ========================================================================
    /// Number of parallel environments
    pub num_envs: usize,
    /// Action buffer length (number of previous actions stored)
    pub action_buffer_len: usize,
}

impl QuadcopterState {
    /// Create new state storage for the given number of environments.
    pub fn new(num_envs: usize, action_buffer_len: usize) -> Self {
        Self {
            // Position
            pos_x: vec![0.0; num_envs],
            pos_y: vec![0.0; num_envs],
            pos_z: vec![1.0; num_envs], // Start 1m above ground

            // Quaternion (identity = level orientation)
            quat_w: vec![1.0; num_envs],
            quat_x: vec![0.0; num_envs],
            quat_y: vec![0.0; num_envs],
            quat_z: vec![0.0; num_envs],

            // Velocity (at rest)
            vel_x: vec![0.0; num_envs],
            vel_y: vec![0.0; num_envs],
            vel_z: vec![0.0; num_envs],

            // Angular velocity (at rest)
            ang_vel_x: vec![0.0; num_envs],
            ang_vel_y: vec![0.0; num_envs],
            ang_vel_z: vec![0.0; num_envs],

            // Motor RPMs (at hover)
            last_rpm: vec![HOVER_RPM; num_envs * 4],
            actual_rpm: vec![HOVER_RPM; num_envs * 4],

            // Previous action (normalized hover)
            prev_action: vec![rpm_to_action(HOVER_RPM); num_envs * 4],

            // Action buffer
            action_buffer: vec![0.0; num_envs * action_buffer_len * 4],
            action_buffer_idx: vec![0; num_envs],

            // Targets (hover at origin by default)
            target_pos: vec![0.0; num_envs * 3],
            target_vel: vec![0.0; num_envs * 3],

            // Episode tracking
            step_count: vec![0; num_envs],
            episode_reward: vec![0.0; num_envs],

            // Metadata
            num_envs,
            action_buffer_len,
        }
    }

    /// Reset a single environment to initial state.
    ///
    /// # Arguments
    /// * `idx` - Environment index to reset
    /// * `seed` - Random seed for this environment
    /// * `config` - Initialization configuration
    /// * `hover_target` - Default target position for hover mode
    pub fn reset_env(
        &mut self,
        idx: usize,
        seed: u64,
        config: &InitConfig,
        hover_target: [f32; 3],
    ) {
        let mut rng = fastrand::Rng::with_seed(seed);

        // Reset position
        let (px, py, pz) = Self::sample_range_3d(
            &config.position_range,
            config.position_dist,
            &mut rng,
        );
        self.pos_x[idx] = px;
        self.pos_y[idx] = py;
        self.pos_z[idx] = pz;

        // Reset velocity
        let (vx, vy, vz) = Self::sample_range_3d(
            &config.velocity_range,
            config.velocity_dist,
            &mut rng,
        );
        self.vel_x[idx] = vx;
        self.vel_y[idx] = vy;
        self.vel_z[idx] = vz;

        // Reset attitude (Euler to quaternion)
        let (roll, pitch, yaw) = Self::sample_range_3d(
            &config.attitude_range,
            config.attitude_dist,
            &mut rng,
        );
        let q = euler_to_quat([roll, pitch, yaw]);
        self.quat_w[idx] = q[0];
        self.quat_x[idx] = q[1];
        self.quat_y[idx] = q[2];
        self.quat_z[idx] = q[3];

        // Reset angular velocity
        let (wx, wy, wz) = Self::sample_range_3d(
            &config.angular_vel_range,
            config.angular_vel_dist,
            &mut rng,
        );
        self.ang_vel_x[idx] = wx;
        self.ang_vel_y[idx] = wy;
        self.ang_vel_z[idx] = wz;

        // Reset motor RPMs
        let base = idx * 4;
        let init_rpm = if config.hover_init { HOVER_RPM } else { 0.0 };
        for i in 0..4 {
            self.last_rpm[base + i] = init_rpm;
            self.actual_rpm[base + i] = init_rpm;
            self.prev_action[base + i] = rpm_to_action(init_rpm);
        }

        // Reset action buffer
        if self.action_buffer_len > 0 {
            let buf_base = idx * self.action_buffer_len * 4;
            let buf_end = buf_base + self.action_buffer_len * 4;
            self.action_buffer[buf_base..buf_end].fill(0.0);
            self.action_buffer_idx[idx] = 0;
        }

        // Reset target to hover position
        let tgt_base = idx * 3;
        self.target_pos[tgt_base] = hover_target[0];
        self.target_pos[tgt_base + 1] = hover_target[1];
        self.target_pos[tgt_base + 2] = hover_target[2];
        self.target_vel[tgt_base] = 0.0;
        self.target_vel[tgt_base + 1] = 0.0;
        self.target_vel[tgt_base + 2] = 0.0;

        // Reset episode tracking
        self.step_count[idx] = 0;
        self.episode_reward[idx] = 0.0;
    }

    /// Sample a value from range based on distribution type.
    fn sample_range(min: f32, max: f32, dist: DistributionType, rng: &mut fastrand::Rng) -> f32 {
        match dist {
            DistributionType::Fixed => (min + max) * 0.5,
            DistributionType::Uniform => {
                let t = rng.f32();
                min + t * (max - min)
            }
            DistributionType::Gaussian => {
                // Box-Muller transform for Gaussian
                let mean = (min + max) * 0.5;
                let std = (max - min) / 6.0; // 3-sigma rule
                let u1 = rng.f32().max(1e-10);
                let u2 = rng.f32();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                (mean + std * z).clamp(min, max)
            }
        }
    }

    /// Sample 3D values from a 6-element range array.
    fn sample_range_3d(
        range: &[f32; 6],
        dist: DistributionType,
        rng: &mut fastrand::Rng,
    ) -> (f32, f32, f32) {
        (
            Self::sample_range(range[0], range[1], dist, rng),
            Self::sample_range(range[2], range[3], dist, rng),
            Self::sample_range(range[4], range[5], dist, rng),
        )
    }

    /// Get position for an environment.
    #[inline]
    pub fn get_position(&self, idx: usize) -> [f32; 3] {
        [self.pos_x[idx], self.pos_y[idx], self.pos_z[idx]]
    }

    /// Get quaternion for an environment.
    #[inline]
    pub fn get_quaternion(&self, idx: usize) -> [f32; 4] {
        [
            self.quat_w[idx],
            self.quat_x[idx],
            self.quat_y[idx],
            self.quat_z[idx],
        ]
    }

    /// Get velocity for an environment.
    #[inline]
    pub fn get_velocity(&self, idx: usize) -> [f32; 3] {
        [self.vel_x[idx], self.vel_y[idx], self.vel_z[idx]]
    }

    /// Get angular velocity for an environment.
    #[inline]
    pub fn get_angular_velocity(&self, idx: usize) -> [f32; 3] {
        [self.ang_vel_x[idx], self.ang_vel_y[idx], self.ang_vel_z[idx]]
    }

    /// Set position for an environment.
    #[inline]
    pub fn set_position(&mut self, idx: usize, pos: [f32; 3]) {
        self.pos_x[idx] = pos[0];
        self.pos_y[idx] = pos[1];
        self.pos_z[idx] = pos[2];
    }

    /// Set quaternion for an environment.
    #[inline]
    pub fn set_quaternion(&mut self, idx: usize, quat: [f32; 4]) {
        self.quat_w[idx] = quat[0];
        self.quat_x[idx] = quat[1];
        self.quat_y[idx] = quat[2];
        self.quat_z[idx] = quat[3];
    }

    /// Set velocity for an environment.
    #[inline]
    pub fn set_velocity(&mut self, idx: usize, vel: [f32; 3]) {
        self.vel_x[idx] = vel[0];
        self.vel_y[idx] = vel[1];
        self.vel_z[idx] = vel[2];
    }

    /// Set angular velocity for an environment.
    #[inline]
    pub fn set_angular_velocity(&mut self, idx: usize, ang_vel: [f32; 3]) {
        self.ang_vel_x[idx] = ang_vel[0];
        self.ang_vel_y[idx] = ang_vel[1];
        self.ang_vel_z[idx] = ang_vel[2];
    }

    /// Get commanded motor RPMs for an environment.
    #[inline]
    pub fn get_rpms(&self, idx: usize) -> [f32; 4] {
        let base = idx * 4;
        [
            self.last_rpm[base],
            self.last_rpm[base + 1],
            self.last_rpm[base + 2],
            self.last_rpm[base + 3],
        ]
    }

    /// Get actual motor RPMs for an environment (after motor dynamics).
    #[inline]
    pub fn get_actual_rpm(&self, idx: usize) -> [f32; 4] {
        let base = idx * 4;
        [
            self.actual_rpm[base],
            self.actual_rpm[base + 1],
            self.actual_rpm[base + 2],
            self.actual_rpm[base + 3],
        ]
    }

    /// Set actual motor RPMs for an environment.
    #[inline]
    pub fn set_actual_rpm(&mut self, idx: usize, rpms: [f32; 4]) {
        let base = idx * 4;
        self.actual_rpm[base] = rpms[0];
        self.actual_rpm[base + 1] = rpms[1];
        self.actual_rpm[base + 2] = rpms[2];
        self.actual_rpm[base + 3] = rpms[3];
    }

    /// Get target position for an environment.
    #[inline]
    pub fn get_target_position(&self, idx: usize) -> [f32; 3] {
        let base = idx * 3;
        [
            self.target_pos[base],
            self.target_pos[base + 1],
            self.target_pos[base + 2],
        ]
    }

    /// Set target position for an environment.
    #[inline]
    pub fn set_target_position(&mut self, idx: usize, target: [f32; 3]) {
        let base = idx * 3;
        self.target_pos[base] = target[0];
        self.target_pos[base + 1] = target[1];
        self.target_pos[base + 2] = target[2];
    }

    /// Set target velocity for an environment.
    #[inline]
    pub fn set_target_velocity(&mut self, idx: usize, target: [f32; 3]) {
        let base = idx * 3;
        self.target_vel[base] = target[0];
        self.target_vel[base + 1] = target[1];
        self.target_vel[base + 2] = target[2];
    }

    /// Push action to the action buffer for an environment.
    pub fn push_action_buffer(&mut self, idx: usize, action: &[f32; 4]) {
        if self.action_buffer_len == 0 {
            return;
        }

        let buf_base = idx * self.action_buffer_len * 4;
        let write_idx = self.action_buffer_idx[idx];
        let action_base = buf_base + write_idx * 4;

        self.action_buffer[action_base] = action[0];
        self.action_buffer[action_base + 1] = action[1];
        self.action_buffer[action_base + 2] = action[2];
        self.action_buffer[action_base + 3] = action[3];

        self.action_buffer_idx[idx] = (write_idx + 1) % self.action_buffer_len;
    }

    /// Get flattened action buffer for an environment (oldest to newest).
    pub fn get_action_buffer(&self, idx: usize, output: &mut [f32]) {
        if self.action_buffer_len == 0 {
            return;
        }

        let buf_base = idx * self.action_buffer_len * 4;
        let start_idx = self.action_buffer_idx[idx];

        for i in 0..self.action_buffer_len {
            let read_idx = (start_idx + i) % self.action_buffer_len;
            let src_base = buf_base + read_idx * 4;
            let dst_base = i * 4;
            output[dst_base..dst_base + 4]
                .copy_from_slice(&self.action_buffer[src_base..src_base + 4]);
        }
    }

    // ========================================================================
    // Type-Safe Getters/Setters
    // ========================================================================

    /// Get position as typed Vec3<WorldFrame>.
    #[inline]
    pub fn get_position_typed(&self, idx: usize) -> Vec3<WorldFrame> {
        Vec3::new(self.pos_x[idx], self.pos_y[idx], self.pos_z[idx])
    }

    /// Get velocity as typed Vec3<WorldFrame>.
    #[inline]
    pub fn get_velocity_typed(&self, idx: usize) -> Vec3<WorldFrame> {
        Vec3::new(self.vel_x[idx], self.vel_y[idx], self.vel_z[idx])
    }

    /// Get angular velocity as typed Vec3<BodyFrame>.
    #[inline]
    pub fn get_angular_velocity_typed(&self, idx: usize) -> Vec3<BodyFrame> {
        Vec3::new(self.ang_vel_x[idx], self.ang_vel_y[idx], self.ang_vel_z[idx])
    }

    /// Get quaternion as UnitQuaternion.
    #[inline]
    pub fn get_quaternion_typed(&self, idx: usize) -> UnitQuaternion {
        // SAFETY: Quaternions in state are always kept normalized
        unsafe {
            UnitQuaternion::new_unchecked(
                self.quat_w[idx],
                self.quat_x[idx],
                self.quat_y[idx],
                self.quat_z[idx],
            )
        }
    }

    /// Set position from typed Vec3<WorldFrame>.
    #[inline]
    pub fn set_position_typed(&mut self, idx: usize, pos: Vec3<WorldFrame>) {
        let arr = pos.as_array();
        self.pos_x[idx] = arr[0];
        self.pos_y[idx] = arr[1];
        self.pos_z[idx] = arr[2];
    }

    /// Set velocity from typed Vec3<WorldFrame>.
    #[inline]
    pub fn set_velocity_typed(&mut self, idx: usize, vel: Vec3<WorldFrame>) {
        let arr = vel.as_array();
        self.vel_x[idx] = arr[0];
        self.vel_y[idx] = arr[1];
        self.vel_z[idx] = arr[2];
    }

    /// Set angular velocity from typed Vec3<BodyFrame>.
    #[inline]
    pub fn set_angular_velocity_typed(&mut self, idx: usize, omega: Vec3<BodyFrame>) {
        let arr = omega.as_array();
        self.ang_vel_x[idx] = arr[0];
        self.ang_vel_y[idx] = arr[1];
        self.ang_vel_z[idx] = arr[2];
    }

    /// Set quaternion from UnitQuaternion.
    #[inline]
    pub fn set_quaternion_typed(&mut self, idx: usize, q: UnitQuaternion) {
        let arr = q.as_array();
        self.quat_w[idx] = arr[0];
        self.quat_x[idx] = arr[1];
        self.quat_y[idx] = arr[2];
        self.quat_z[idx] = arr[3];
    }

    /// Get target position as typed Vec3<WorldFrame>.
    #[inline]
    pub fn get_target_position_typed(&self, idx: usize) -> Vec3<WorldFrame> {
        let base = idx * 3;
        Vec3::new(
            self.target_pos[base],
            self.target_pos[base + 1],
            self.target_pos[base + 2],
        )
    }

    /// Get target velocity as typed Vec3<WorldFrame>.
    #[inline]
    pub fn get_target_velocity_typed(&self, idx: usize) -> Vec3<WorldFrame> {
        let base = idx * 3;
        Vec3::new(
            self.target_vel[base],
            self.target_vel[base + 1],
            self.target_vel[base + 2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = QuadcopterState::new(64, 0);
        assert_eq!(state.num_envs, 64);
        assert_eq!(state.pos_x.len(), 64);
        assert_eq!(state.quat_w.len(), 64);
        assert_eq!(state.last_rpm.len(), 64 * 4);
    }

    #[test]
    fn test_state_with_action_buffer() {
        let state = QuadcopterState::new(8, 5);
        assert_eq!(state.action_buffer.len(), 8 * 5 * 4);
        assert_eq!(state.action_buffer_idx.len(), 8);
    }

    #[test]
    fn test_reset_env() {
        let mut state = QuadcopterState::new(4, 0);
        let config = InitConfig::fixed_start();

        state.reset_env(0, 42, &config, [0.0, 0.0, 1.0]);

        assert_eq!(state.pos_x[0], 0.0);
        assert_eq!(state.pos_y[0], 0.0);
        assert_eq!(state.pos_z[0], 1.0);
        assert_eq!(state.quat_w[0], 1.0);
        assert_eq!(state.step_count[0], 0);
    }

    #[test]
    fn test_euler_to_quat_identity() {
        let q = euler_to_quat([0.0, 0.0, 0.0]);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!(q[1].abs() < 1e-6);
        assert!(q[2].abs() < 1e-6);
        assert!(q[3].abs() < 1e-6);
    }

    #[test]
    fn test_typed_getters_setters() {
        let mut state = QuadcopterState::new(1, 0);

        // Test position
        let pos: Vec3<WorldFrame> = Vec3::new(1.0, 2.0, 3.0);
        state.set_position_typed(0, pos);
        let pos_back = state.get_position_typed(0);
        assert_eq!(pos.as_array(), pos_back.as_array());

        // Test velocity
        let vel: Vec3<WorldFrame> = Vec3::new(0.1, 0.2, 0.3);
        state.set_velocity_typed(0, vel);
        let vel_back = state.get_velocity_typed(0);
        assert_eq!(vel.as_array(), vel_back.as_array());

        // Test angular velocity
        let omega: Vec3<BodyFrame> = Vec3::new(0.01, 0.02, 0.03);
        state.set_angular_velocity_typed(0, omega);
        let omega_back = state.get_angular_velocity_typed(0);
        assert_eq!(omega.as_array(), omega_back.as_array());

        // Test quaternion
        let q = UnitQuaternion::new(0.5, 0.5, 0.5, 0.5);
        state.set_quaternion_typed(0, q);
        let q_back = state.get_quaternion_typed(0);
        assert!((q.norm_squared() - q_back.norm_squared()).abs() < 1e-6);
    }

    #[test]
    fn test_action_buffer() {
        let mut state = QuadcopterState::new(1, 3);

        state.push_action_buffer(0, &[1.0, 2.0, 3.0, 4.0]);
        state.push_action_buffer(0, &[5.0, 6.0, 7.0, 8.0]);

        let mut output = vec![0.0; 12];
        state.get_action_buffer(0, &mut output);

        // Buffer should contain: [0,0,0,0], [1,2,3,4], [5,6,7,8]
        // Read order starts from oldest (index after current write position)
    }
}
