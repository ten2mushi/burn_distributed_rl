//! Adapter for integrating Quadcopter with distributed_rl training system.
//!
//! Provides wrappers that implement the distributed_rl VectorizedEnv trait
//! for use with PPOContinuous and other continuous action learners.

use crate::config::QuadcopterConfig;
use crate::env::Quadcopter;
use operant_core::Environment;

// ============================================================================
// Wrapper for simplified interface
// ============================================================================

/// Simple step result for the Quadcopter environment.
#[derive(Debug, Clone)]
pub struct QuadcopterStepResult {
    /// Rewards for each environment
    pub rewards: Vec<f32>,
    /// Done flags (terminal or truncated)
    pub dones: Vec<bool>,
    /// True terminal flags (not truncation)
    pub terminals: Vec<bool>,
}

/// Wrapper for the Quadcopter environment with simplified interface.
///
/// This provides a simple interface for the quadcopter environment
/// that can be used directly or adapted to the generic VectorizedEnv trait.
pub struct QuadcopterEnvWrapper {
    env: Quadcopter,
    /// Number of environments
    pub n_envs: usize,
    /// Observation size
    pub obs_size: usize,
    /// Action dimension (4 motors)
    pub action_dim: usize,
}

impl QuadcopterEnvWrapper {
    /// Create a new Quadcopter wrapper from configuration.
    /// Automatically resets all environments on creation.
    pub fn from_config(config: QuadcopterConfig) -> Result<Self, String> {
        let n_envs = config.num_envs;
        let obs_size = config.observation_size();
        let mut env = Quadcopter::from_config(config)?;

        // Auto-reset on creation so observations are immediately available
        env.reset(0);

        Ok(Self {
            env,
            n_envs,
            obs_size,
            action_dim: 4,
        })
    }

    /// Create with default configuration.
    pub fn new(n_envs: usize) -> Result<Self, String> {
        Self::from_config(QuadcopterConfig::new(n_envs))
    }

    /// Get reference to the underlying environment.
    pub fn inner(&self) -> &Quadcopter {
        &self.env
    }

    /// Get mutable reference to the underlying environment.
    pub fn inner_mut(&mut self) -> &mut Quadcopter {
        &mut self.env
    }

    /// Write current observations to buffer.
    pub fn write_observations(&self, buffer: &mut [f32]) {
        self.env.write_observations(buffer);
    }

    /// Step with continuous actions.
    ///
    /// # Arguments
    /// * `actions` - Flat array of motor commands [m0, m1, m2, m3] per env
    ///   Each motor command should be in [-1, 1] range.
    pub fn step(&mut self, actions: &[f32]) -> QuadcopterStepResult {
        let result = self.env.step_no_reset_with_result(actions);

        let terminals: Vec<bool> = result.terminals.iter().map(|&t| t != 0).collect();
        let truncations: Vec<bool> = result.truncations.iter().map(|&t| t != 0).collect();
        let dones: Vec<bool> = terminals
            .iter()
            .zip(truncations.iter())
            .map(|(&t, &tr)| t || tr)
            .collect();

        QuadcopterStepResult {
            rewards: result.rewards.to_vec(),
            dones,
            terminals,
        }
    }

    /// Reset specific environments.
    pub fn reset_envs(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
        let mut terminals = vec![0u8; self.n_envs];
        for &idx in indices {
            terminals[idx] = 1;
        }
        let truncations = vec![0u8; self.n_envs];
        let mask = operant_core::ResetMask::from_done_flags(&terminals, &truncations);
        self.env.reset_envs(&mask, fastrand::u64(..));
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: u64) {
        self.env.reset(seed);
    }

    /// Set target positions for all environments (for tracking mode).
    ///
    /// # Arguments
    /// * `positions` - Flat array [x0, y0, z0, x1, y1, z1, ...]
    pub fn set_targets(&mut self, positions: &[f32]) {
        self.env.set_targets(positions);
    }

    /// Set target velocities for all environments (for tracking mode).
    ///
    /// # Arguments
    /// * `velocities` - Flat array [vx0, vy0, vz0, vx1, vy1, vz1, ...]
    pub fn set_target_velocities(&mut self, velocities: &[f32]) {
        self.env.set_target_velocities(velocities);
    }
}

// ============================================================================
// LearnerVectorizedEnv Implementation
// ============================================================================

#[cfg(feature = "distributed_rl")]
pub mod distributed {
    use super::*;

    // Re-export the action types from distributed_rl
    use distributed_rl::algorithms::action_policy::ContinuousAction;
    use distributed_rl::runners::learner::{
        StepResult as LearnerStepResult, VectorizedEnv as LearnerVectorizedEnv,
    };

    /// Quadcopter environment adapter for the distributed_rl Learner.
    ///
    /// Implements `VectorizedEnv<ContinuousAction>` so it can be used directly
    /// with `PPOContinuous` or other continuous action learners.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use quadcopter_env::adapter::distributed::QuadcopterEnv;
    /// use quadcopter_env::config::QuadcopterConfig;
    ///
    /// let config = QuadcopterConfig::new(64)
    ///     .with_observation(ObsConfig::kinematic());
    ///
    /// let env = QuadcopterEnv::from_config(config).unwrap();
    /// let learner = PPOContinuous::new(config, algorithm);
    /// learner.run(model, env, optimizer, callback);
    /// ```
    pub struct QuadcopterEnv {
        inner: QuadcopterEnvWrapper,
    }

    impl QuadcopterEnv {
        /// Action dimension: 4 motor commands.
        pub const ACTION_DIM: usize = 4;
        /// Action bounds: [-1, 1] for normalized motor commands.
        pub const ACTION_LOW: f32 = -1.0;
        pub const ACTION_HIGH: f32 = 1.0;

        /// Create a new Quadcopter environment from configuration.
        pub fn from_config(config: QuadcopterConfig) -> Result<Self, String> {
            let mut inner = QuadcopterEnvWrapper::from_config(config)?;
            inner.reset_all(42); // Required: reset before use
            Ok(Self { inner })
        }

        /// Create with default configuration.
        pub fn new(n_envs: usize) -> Result<Self, String> {
            Self::from_config(QuadcopterConfig::new(n_envs))
        }

        /// Number of parallel environments.
        pub fn n_envs(&self) -> usize {
            self.inner.n_envs
        }

        /// Observation size.
        pub fn obs_size(&self) -> usize {
            self.inner.obs_size
        }

        /// Get action bounds as (low, high) vectors.
        pub fn action_bounds() -> (Vec<f32>, Vec<f32>) {
            (
                vec![Self::ACTION_LOW; Self::ACTION_DIM],
                vec![Self::ACTION_HIGH; Self::ACTION_DIM],
            )
        }

        /// Set target positions for tracking mode.
        pub fn set_targets(&mut self, positions: &[f32]) {
            self.inner.set_targets(positions);
        }

        /// Set target velocities for tracking mode.
        pub fn set_target_velocities(&mut self, velocities: &[f32]) {
            self.inner.set_target_velocities(velocities);
        }

        /// Get reference to the inner wrapper.
        pub fn inner(&self) -> &QuadcopterEnvWrapper {
            &self.inner
        }

        /// Get mutable reference to the inner wrapper.
        pub fn inner_mut(&mut self) -> &mut QuadcopterEnvWrapper {
            &mut self.inner
        }
    }

    impl LearnerVectorizedEnv<ContinuousAction> for QuadcopterEnv {
        fn n_envs(&self) -> usize {
            self.inner.n_envs
        }

        fn obs_size(&self) -> usize {
            self.inner.obs_size
        }

        fn write_observations(&self, buffer: &mut [f32]) {
            self.inner.write_observations(buffer);
        }

        fn step(&mut self, actions: &[ContinuousAction]) -> LearnerStepResult {
            // Flatten ContinuousAction Vec<f32> to single f32 slice
            let action_floats: Vec<f32> = actions.iter().flat_map(|a| a.0.clone()).collect();
            let result = self.inner.step(&action_floats);

            LearnerStepResult {
                rewards: result.rewards,
                dones: result.dones,
                terminals: result.terminals,
            }
        }

        fn reset_envs(&mut self, indices: &[usize]) {
            self.inner.reset_envs(indices);
        }
    }
}

// ============================================================================
// Standalone wrapper (without distributed_rl dependency)
// ============================================================================

/// Convenience function to create a Quadcopter wrapper with default configuration.
pub fn create_quadcopter(n_envs: usize) -> Result<QuadcopterEnvWrapper, String> {
    QuadcopterEnvWrapper::new(n_envs)
}

/// Convenience function to create a Quadcopter wrapper from configuration.
pub fn create_quadcopter_from_config(config: QuadcopterConfig) -> Result<QuadcopterEnvWrapper, String> {
    QuadcopterEnvWrapper::from_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::rpm_to_action;

    #[test]
    fn test_wrapper_creation() {
        let wrapper = QuadcopterEnvWrapper::new(8).unwrap();
        assert_eq!(wrapper.n_envs, 8);
        assert_eq!(wrapper.action_dim, 4);
    }

    #[test]
    fn test_wrapper_step() {
        let mut wrapper = QuadcopterEnvWrapper::new(4).unwrap();
        wrapper.reset_all(42);

        // Hover actions
        let hover_action = rpm_to_action(crate::constants::HOVER_RPM);
        let actions = vec![hover_action; 4 * 4];

        let result = wrapper.step(&actions);

        assert_eq!(result.rewards.len(), 4);
        assert_eq!(result.dones.len(), 4);
        assert_eq!(result.terminals.len(), 4);
    }

    #[test]
    fn test_wrapper_observations() {
        let wrapper = QuadcopterEnvWrapper::new(4).unwrap();
        let mut buffer = vec![0.0; 4 * wrapper.obs_size];
        wrapper.write_observations(&mut buffer);

        // Should have some non-zero values
        let has_nonzero = buffer.iter().any(|&x| x != 0.0);
        assert!(has_nonzero);
    }

    #[test]
    fn test_wrapper_reset_envs() {
        let mut wrapper = QuadcopterEnvWrapper::new(8).unwrap();
        wrapper.reset_all(42);

        // Get initial observation for env 0
        let mut buffer = vec![0.0; 8 * wrapper.obs_size];
        wrapper.write_observations(&mut buffer);
        let initial_obs_0 = buffer[0..wrapper.obs_size].to_vec();

        // Step to change state
        let hover_action = rpm_to_action(crate::constants::HOVER_RPM);
        let actions = vec![hover_action; 8 * 4];
        wrapper.step(&actions);

        // Get observation after step
        wrapper.write_observations(&mut buffer);
        let stepped_obs_0 = buffer[0..wrapper.obs_size].to_vec();

        // Reset env 0 only
        wrapper.reset_envs(&[0]);

        // Observation should be different from stepped observation
        // (though may be same as initial due to deterministic reset)
    }

    #[test]
    fn test_set_targets() {
        let mut wrapper = QuadcopterEnvWrapper::new(2).unwrap();
        wrapper.reset_all(42);

        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        wrapper.set_targets(&targets);

        // Verify targets are set in the underlying environment
        let state = wrapper.inner().state();
        assert_eq!(state.target_pos[0], 1.0);
        assert_eq!(state.target_pos[3], 4.0);
    }
}
