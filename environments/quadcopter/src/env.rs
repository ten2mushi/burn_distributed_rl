//! Main Quadcopter environment implementing the operant Environment trait.
//!
//! Provides a high-performance SIMD-optimized quadcopter RL environment
//! with support for both auto-reset and non-auto-reset APIs.

use operant_core::{Environment, ResetMask, StepResult};

use crate::config::{QuadcopterConfig, TaskMode};
use crate::constants::action_to_rpm;
use crate::noise::{apply_observation_noise, XorShiftRng};
use crate::normalization::RewardProcessor;
use crate::observation::write_observations_all;
use crate::physics::dynamics::physics_substeps_scalar;
use crate::physics::motor::apply_motor_dynamics_quad;
use crate::reward::{compute_rewards_all, presets, RewardComponent};
use crate::state::QuadcopterState;
use crate::termination::check_termination_all;

#[cfg(feature = "simd")]
use crate::observation::simd::write_observations_all_simd;
#[cfg(feature = "simd")]
use crate::physics::dynamics::simd::physics_substeps_simd;
#[cfg(feature = "simd")]
use crate::physics::motor::simd::apply_motor_dynamics_simd;
#[cfg(feature = "simd")]
use crate::reward::compute_rewards_all_simd;
#[cfg(feature = "simd")]
use crate::termination::simd::check_termination_all_simd;
#[cfg(feature = "simd")]
use std::simd::f32x8;

/// SIMD-optimized quadcopter environment for reinforcement learning.
///
/// Generic over reward type `R` for compile-time reward composition.
/// Defaults to [`presets::HoverReward`] for standard hover tasks.
pub struct Quadcopter<R: RewardComponent = presets::HoverReward> {
    /// Environment configuration
    config: QuadcopterConfig<R>,

    /// State storage (SoA layout)
    state: QuadcopterState,

    // Internal buffers for StepResult
    obs_buffer: Vec<f32>,
    reward_buffer: Vec<f32>,
    terminal_buffer: Vec<u8>,
    truncation_buffer: Vec<u8>,

    /// Temporary bool buffers for termination checks
    terminal_bool: Vec<bool>,
    truncation_bool: Vec<bool>,

    /// Reward processor for normalization/clipping
    reward_processor: RewardProcessor,

    /// RNG for observation noise injection
    noise_rng: XorShiftRng,

    /// Physics substeps per control step
    substeps: u32,
}

impl<R: RewardComponent> Quadcopter<R> {
    /// Create a new quadcopter environment from configuration.
    pub fn from_config(config: QuadcopterConfig<R>) -> Result<Self, String> {
        config.validate()?;

        let num_envs = config.num_envs;
        let obs_size = config.observation_size();
        let substeps = config.physics_steps_per_ctrl();
        let reward_processor = RewardProcessor::new(config.normalization.clone());
        let noise_rng = XorShiftRng::new(0xDEADBEEF);

        Ok(Self {
            state: QuadcopterState::new(num_envs, config.obs.action_buffer_len),
            obs_buffer: vec![0.0; num_envs * obs_size],
            reward_buffer: vec![0.0; num_envs],
            terminal_buffer: vec![0; num_envs],
            truncation_buffer: vec![0; num_envs],
            terminal_bool: vec![false; num_envs],
            truncation_bool: vec![false; num_envs],
            reward_processor,
            noise_rng,
            substeps,
            config,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &QuadcopterConfig<R> {
        &self.config
    }

    /// Get read access to the state.
    pub fn state(&self) -> &QuadcopterState {
        &self.state
    }

    /// Get mutable access to the state.
    pub fn state_mut(&mut self) -> &mut QuadcopterState {
        &mut self.state
    }

    /// Set target positions for all environments (for tracking mode).
    ///
    /// # Arguments
    /// * `positions` - Flat array of target positions [x0, y0, z0, x1, y1, z1, ...]
    pub fn set_targets(&mut self, positions: &[f32]) {
        self.state.target_pos.copy_from_slice(positions);
    }

    /// Set target velocities for all environments (for tracking mode).
    ///
    /// # Arguments
    /// * `velocities` - Flat array of target velocities [vx0, vy0, vz0, vx1, vy1, vz1, ...]
    pub fn set_target_velocities(&mut self, velocities: &[f32]) {
        self.state.target_vel.copy_from_slice(velocities);
    }

    /// Reset a single environment.
    fn reset_single_env(&mut self, idx: usize, seed: u64) {
        self.state.reset_env(
            idx,
            seed.wrapping_add(idx as u64),
            &self.config.init,
            self.config.hover_target,
        );
    }

    /// Step a single environment (scalar implementation).
    fn step_single_env(&mut self, idx: usize, actions: &[f32]) {
        // Convert normalized actions to commanded RPMs
        let base = idx * 4;
        let cmd_rpms = [
            action_to_rpm(actions[0]),
            action_to_rpm(actions[1]),
            action_to_rpm(actions[2]),
            action_to_rpm(actions[3]),
        ];

        // Store previous action for action rate penalty
        for i in 0..4 {
            self.state.prev_action[base + i] = actions[i];
        }

        // Apply motor dynamics (first-order lag response)
        let effective_rpms = if self.config.motor_dynamics.is_enabled() {
            let alpha = self.config.motor_dynamics.compute_alpha(self.config.dt_ctrl());
            let mut actual = self.state.get_actual_rpm(idx);
            apply_motor_dynamics_quad(cmd_rpms, &mut actual, alpha);
            self.state.set_actual_rpm(idx, actual);
            actual
        } else {
            // Instantaneous: commanded = actual
            self.state.set_actual_rpm(idx, cmd_rpms);
            cmd_rpms
        };

        // Run physics substeps with effective RPMs
        let dt = self.config.dt_ctrl();
        physics_substeps_scalar(&mut self.state, idx, effective_rpms, dt, self.substeps);

        // Store current RPMs (commanded, for observation/reward)
        for i in 0..4 {
            self.state.last_rpm[base + i] = cmd_rpms[i];
        }

        // Increment step count
        self.state.step_count[idx] += 1;
    }

    /// Apply observation noise to all environments if enabled.
    fn apply_observation_noise_all(&mut self) {
        if !self.config.obs.noise.is_enabled() {
            return;
        }

        let obs_size = self.config.observation_size();
        let (pos_offset, euler_offset, vel_offset, ang_vel_offset) = self.config.obs.noise_offsets();

        for idx in 0..self.config.num_envs {
            let base = idx * obs_size;
            apply_observation_noise(
                &mut self.obs_buffer[base..base + obs_size],
                &self.config.obs.noise,
                pos_offset,
                euler_offset,
                vel_offset,
                ang_vel_offset,
                &mut self.noise_rng,
            );
        }
    }

    /// Step all environments (scalar path).
    fn step_all_scalar(&mut self, actions: &[f32]) {
        for idx in 0..self.config.num_envs {
            let action_base = idx * 4;
            self.step_single_env(idx, &actions[action_base..action_base + 4]);
        }
    }

    /// Step all environments (SIMD path).
    #[cfg(feature = "simd")]
    fn step_all_simd(&mut self, actions: &[f32]) {
        let chunks = self.config.num_envs / 8;
        let remainder = self.config.num_envs % 8;
        let dt = self.config.dt_ctrl();
        let motor_dynamics_enabled = self.config.motor_dynamics.is_enabled();
        let alpha = self.config.motor_dynamics.compute_alpha(dt);
        let alpha_vec = f32x8::splat(alpha);

        // Process full SIMD chunks
        for chunk in 0..chunks {
            let base_idx = chunk * 8;

            // Load actions and convert to commanded RPMs
            let mut cmd_rpm0 = [0.0f32; 8];
            let mut cmd_rpm1 = [0.0f32; 8];
            let mut cmd_rpm2 = [0.0f32; 8];
            let mut cmd_rpm3 = [0.0f32; 8];

            // Load current actual RPMs for motor dynamics
            let mut actual_rpm0 = [0.0f32; 8];
            let mut actual_rpm1 = [0.0f32; 8];
            let mut actual_rpm2 = [0.0f32; 8];
            let mut actual_rpm3 = [0.0f32; 8];

            for i in 0..8 {
                let idx = base_idx + i;
                let action_base = idx * 4;

                cmd_rpm0[i] = action_to_rpm(actions[action_base]);
                cmd_rpm1[i] = action_to_rpm(actions[action_base + 1]);
                cmd_rpm2[i] = action_to_rpm(actions[action_base + 2]);
                cmd_rpm3[i] = action_to_rpm(actions[action_base + 3]);

                // Load current actual RPMs
                actual_rpm0[i] = self.state.actual_rpm[idx * 4];
                actual_rpm1[i] = self.state.actual_rpm[idx * 4 + 1];
                actual_rpm2[i] = self.state.actual_rpm[idx * 4 + 2];
                actual_rpm3[i] = self.state.actual_rpm[idx * 4 + 3];

                // Store previous action for action rate penalty
                for j in 0..4 {
                    self.state.prev_action[idx * 4 + j] = actions[action_base + j];
                }
            }

            // Convert to SIMD vectors
            let cmd_rpm0_vec = f32x8::from_array(cmd_rpm0);
            let cmd_rpm1_vec = f32x8::from_array(cmd_rpm1);
            let cmd_rpm2_vec = f32x8::from_array(cmd_rpm2);
            let cmd_rpm3_vec = f32x8::from_array(cmd_rpm3);

            // Apply motor dynamics
            let (eff_rpm0, eff_rpm1, eff_rpm2, eff_rpm3) = if motor_dynamics_enabled {
                let actual_rpm0_vec = f32x8::from_array(actual_rpm0);
                let actual_rpm1_vec = f32x8::from_array(actual_rpm1);
                let actual_rpm2_vec = f32x8::from_array(actual_rpm2);
                let actual_rpm3_vec = f32x8::from_array(actual_rpm3);

                let eff0 = apply_motor_dynamics_simd(cmd_rpm0_vec, actual_rpm0_vec, alpha_vec);
                let eff1 = apply_motor_dynamics_simd(cmd_rpm1_vec, actual_rpm1_vec, alpha_vec);
                let eff2 = apply_motor_dynamics_simd(cmd_rpm2_vec, actual_rpm2_vec, alpha_vec);
                let eff3 = apply_motor_dynamics_simd(cmd_rpm3_vec, actual_rpm3_vec, alpha_vec);

                // Store updated actual RPMs
                for i in 0..8 {
                    let idx = base_idx + i;
                    self.state.actual_rpm[idx * 4] = eff0[i];
                    self.state.actual_rpm[idx * 4 + 1] = eff1[i];
                    self.state.actual_rpm[idx * 4 + 2] = eff2[i];
                    self.state.actual_rpm[idx * 4 + 3] = eff3[i];
                }

                (eff0, eff1, eff2, eff3)
            } else {
                // Instantaneous: commanded = actual
                for i in 0..8 {
                    let idx = base_idx + i;
                    self.state.actual_rpm[idx * 4] = cmd_rpm0[i];
                    self.state.actual_rpm[idx * 4 + 1] = cmd_rpm1[i];
                    self.state.actual_rpm[idx * 4 + 2] = cmd_rpm2[i];
                    self.state.actual_rpm[idx * 4 + 3] = cmd_rpm3[i];
                }
                (cmd_rpm0_vec, cmd_rpm1_vec, cmd_rpm2_vec, cmd_rpm3_vec)
            };

            // Run physics substeps with effective RPMs
            physics_substeps_simd(
                &mut self.state,
                base_idx,
                eff_rpm0,
                eff_rpm1,
                eff_rpm2,
                eff_rpm3,
                dt,
                self.substeps,
            );

            // Store commanded RPMs (for observation/reward)
            for i in 0..8 {
                let idx = base_idx + i;
                self.state.last_rpm[idx * 4] = cmd_rpm0[i];
                self.state.last_rpm[idx * 4 + 1] = cmd_rpm1[i];
                self.state.last_rpm[idx * 4 + 2] = cmd_rpm2[i];
                self.state.last_rpm[idx * 4 + 3] = cmd_rpm3[i];
                self.state.step_count[idx] += 1;
            }
        }

        // Handle remainder with scalar
        let base = chunks * 8;
        for i in 0..remainder {
            let idx = base + i;
            let action_base = idx * 4;
            self.step_single_env(idx, &actions[action_base..action_base + 4]);
        }
    }
}

impl<R: RewardComponent> Environment for Quadcopter<R> {
    fn num_envs(&self) -> usize {
        self.config.num_envs
    }

    fn observation_size(&self) -> usize {
        self.config.observation_size()
    }

    fn num_actions(&self) -> Option<usize> {
        None // Continuous action space (4 motor commands)
    }

    fn reset(&mut self, seed: u64) {
        for idx in 0..self.config.num_envs {
            self.reset_single_env(idx, seed);
        }

        // Set initial targets based on task mode
        match self.config.task_mode {
            TaskMode::Hover => {
                for idx in 0..self.config.num_envs {
                    self.state.set_target_position(idx, self.config.hover_target);
                    self.state.set_target_velocity(idx, [0.0, 0.0, 0.0]);
                }
            }
            TaskMode::Tracking => {
                // Targets will be set externally via set_targets()
            }
        }

        // Clear buffers
        self.reward_buffer.fill(0.0);
        self.terminal_buffer.fill(0);
        self.truncation_buffer.fill(0);

        // Populate initial observations
        #[cfg(feature = "simd")]
        {
            write_observations_all_simd(&self.state, &self.config.obs, &mut self.obs_buffer);
        }
        #[cfg(not(feature = "simd"))]
        {
            write_observations_all(&self.state, &self.config.obs, &mut self.obs_buffer);
        }

        // Apply observation noise if enabled
        self.apply_observation_noise_all();
    }

    fn step(&mut self, actions: &[f32]) {
        // Step physics
        #[cfg(feature = "simd")]
        {
            self.step_all_simd(actions);
        }
        #[cfg(not(feature = "simd"))]
        {
            self.step_all_scalar(actions);
        }

        // Compute rewards using generic reward function
        #[cfg(feature = "simd")]
        {
            compute_rewards_all_simd(&self.config.reward, &self.state, &mut self.reward_buffer);
        }
        #[cfg(not(feature = "simd"))]
        {
            compute_rewards_all(&self.config.reward, &self.state, &mut self.reward_buffer);
        }

        // Apply reward normalization/clipping
        self.reward_processor.process_batch(&mut self.reward_buffer);

        // Check termination
        #[cfg(feature = "simd")]
        {
            check_termination_all_simd(&self.state, &self.config.termination, &mut self.terminal_bool, &mut self.truncation_bool);
        }
        #[cfg(not(feature = "simd"))]
        {
            check_termination_all(&self.state, &self.config.termination, &mut self.terminal_bool, &mut self.truncation_bool);
        }

        // Convert bool to u8 and auto-reset done environments
        for idx in 0..self.config.num_envs {
            self.terminal_buffer[idx] = self.terminal_bool[idx] as u8;
            self.truncation_buffer[idx] = self.truncation_bool[idx] as u8;

            if self.terminal_bool[idx] || self.truncation_bool[idx] {
                // Auto-reset
                let seed = (idx as u64).wrapping_mul(0x9e3779b97f4a7c15);
                self.reset_single_env(idx, seed);
            }
        }

        // Write observations
        #[cfg(feature = "simd")]
        {
            write_observations_all_simd(&self.state, &self.config.obs, &mut self.obs_buffer);
        }
        #[cfg(not(feature = "simd"))]
        {
            write_observations_all(&self.state, &self.config.obs, &mut self.obs_buffer);
        }

        // Apply observation noise if enabled
        self.apply_observation_noise_all();
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        buffer.copy_from_slice(&self.obs_buffer);
    }

    fn write_rewards(&self, buffer: &mut [f32]) {
        buffer.copy_from_slice(&self.reward_buffer);
    }

    fn write_terminals(&self, buffer: &mut [u8]) {
        buffer.copy_from_slice(&self.terminal_buffer);
    }

    fn write_truncations(&self, buffer: &mut [u8]) {
        buffer.copy_from_slice(&self.truncation_buffer);
    }

    // ========================================================================
    // Non-auto-reset API for value-based RL
    // ========================================================================

    fn step_no_reset(&mut self, actions: &[f32]) {
        // Step physics
        #[cfg(feature = "simd")]
        {
            self.step_all_simd(actions);
        }
        #[cfg(not(feature = "simd"))]
        {
            self.step_all_scalar(actions);
        }

        // Compute rewards using generic reward function
        #[cfg(feature = "simd")]
        {
            compute_rewards_all_simd(&self.config.reward, &self.state, &mut self.reward_buffer);
        }
        #[cfg(not(feature = "simd"))]
        {
            compute_rewards_all(&self.config.reward, &self.state, &mut self.reward_buffer);
        }

        // Apply reward normalization/clipping
        self.reward_processor.process_batch(&mut self.reward_buffer);

        // Check termination (but don't reset)
        #[cfg(feature = "simd")]
        {
            check_termination_all_simd(&self.state, &self.config.termination, &mut self.terminal_bool, &mut self.truncation_bool);
        }
        #[cfg(not(feature = "simd"))]
        {
            check_termination_all(&self.state, &self.config.termination, &mut self.terminal_bool, &mut self.truncation_bool);
        }

        // Convert bool to u8
        for idx in 0..self.config.num_envs {
            self.terminal_buffer[idx] = self.terminal_bool[idx] as u8;
            self.truncation_buffer[idx] = self.truncation_bool[idx] as u8;
        }

        // Write observations (terminal observations, before any reset)
        #[cfg(feature = "simd")]
        {
            write_observations_all_simd(&self.state, &self.config.obs, &mut self.obs_buffer);
        }
        #[cfg(not(feature = "simd"))]
        {
            write_observations_all(&self.state, &self.config.obs, &mut self.obs_buffer);
        }

        // Apply observation noise if enabled
        self.apply_observation_noise_all();
    }

    fn step_no_reset_with_result(&mut self, actions: &[f32]) -> StepResult<'_> {
        self.step_no_reset(actions);

        StepResult {
            observations: &self.obs_buffer,
            rewards: &self.reward_buffer,
            terminals: &self.terminal_buffer,
            truncations: &self.truncation_buffer,
            num_envs: self.config.num_envs,
            obs_size: self.config.observation_size(),
        }
    }

    fn reset_envs(&mut self, mask: &ResetMask, seed: u64) {
        for idx in mask.iter_set() {
            self.reset_single_env(idx, seed.wrapping_add(idx as u64));
        }
    }

    fn supports_no_reset(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ObsConfig, TerminationConfig};

    fn create_test_env(num_envs: usize) -> Quadcopter {
        QuadcopterConfig::new(num_envs)
            .with_observation(ObsConfig::kinematic())
            .with_termination(TerminationConfig::default())
            .build()
            .unwrap()
    }

    #[test]
    fn test_environment_creation() {
        let env = create_test_env(64);
        assert_eq!(env.num_envs(), 64);
        assert_eq!(env.observation_size(), 12); // kinematic observation
        assert!(env.num_actions().is_none()); // continuous
    }

    #[test]
    fn test_reset() {
        let mut env = create_test_env(8);
        env.reset(42);

        // All environments should be at valid initial state
        for idx in 0..8 {
            assert!(env.state.pos_z[idx] > 0.0, "Drone should be above ground");
            assert!(env.state.step_count[idx] == 0, "Step count should be reset");
        }
    }

    #[test]
    fn test_step() {
        let mut env = create_test_env(4);
        env.reset(42);

        // Hover action (normalized)
        let hover_action = crate::constants::rpm_to_action(crate::constants::HOVER_RPM);
        let actions = vec![hover_action; 4 * 4]; // 4 envs × 4 motors

        env.step(&actions);

        // Step count should increase
        for idx in 0..4 {
            assert!(env.state.step_count[idx] > 0 || env.terminal_buffer[idx] != 0);
        }
    }

    #[test]
    fn test_write_buffers() {
        let mut env = create_test_env(4);
        env.reset(42);

        let obs_size = env.observation_size();
        let mut obs = vec![0.0; 4 * obs_size];
        let mut rewards = vec![0.0; 4];
        let mut terminals = vec![0u8; 4];
        let mut truncations = vec![0u8; 4];

        env.write_observations(&mut obs);
        env.write_rewards(&mut rewards);
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        // Check observations are written
        let has_nonzero = obs.iter().any(|&x| x != 0.0);
        assert!(has_nonzero, "Observations should have non-zero values");
    }

    #[test]
    fn test_step_no_reset() {
        let mut env = create_test_env(4);
        env.reset(42);

        let hover_action = crate::constants::rpm_to_action(crate::constants::HOVER_RPM);
        let actions = vec![hover_action; 4 * 4];

        let result = env.step_no_reset_with_result(&actions);

        assert_eq!(result.num_envs, 4);
        assert_eq!(result.obs_size, 12);
        assert_eq!(result.observations.len(), 4 * 12);
    }

    #[test]
    fn test_reset_mask() {
        let mut env = create_test_env(8);
        env.reset(42);

        // Force some environments to need reset
        env.state.pos_z[0] = -1.0; // Ground collision
        env.state.pos_z[3] = -1.0;

        // Create mask and reset
        let mut mask = ResetMask::new(8);
        mask.set(0);
        mask.set(3);

        let old_z_1 = env.state.pos_z[1]; // This should not change

        env.reset_envs(&mask, 123);

        // Only masked environments should be reset
        assert!(env.state.pos_z[0] > 0.0, "Env 0 should be reset");
        assert!(env.state.pos_z[3] > 0.0, "Env 3 should be reset");
        assert_eq!(env.state.pos_z[1], old_z_1, "Env 1 should not change");
    }

    #[test]
    fn test_supports_no_reset() {
        let env = create_test_env(4);
        assert!(env.supports_no_reset());
    }

    #[test]
    fn test_set_targets() {
        let mut env = create_test_env(2);
        env.reset(42);

        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 envs × 3 components
        env.set_targets(&targets);

        assert_eq!(env.state.target_pos[0], 1.0);
        assert_eq!(env.state.target_pos[1], 2.0);
        assert_eq!(env.state.target_pos[2], 3.0);
        assert_eq!(env.state.target_pos[3], 4.0);
    }
}
