//! Environment abstraction for distributed training.
//!
//! Provides adapters for vectorized environments that can be used with
//! the generic Learner.

/// Result from stepping vectorized environments.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Observations after step [n_envs * obs_size] (flattened)
    pub observations: Vec<f32>,
    /// Rewards received [n_envs]
    pub rewards: Vec<f32>,
    /// Terminal flags (episode ended due to goal/failure) [n_envs]
    pub terminals: Vec<bool>,
    /// Truncation flags (episode ended due to time limit) [n_envs]
    pub truncations: Vec<bool>,
}

impl StepResult {
    /// Create a new step result.
    pub fn new(
        observations: Vec<f32>,
        rewards: Vec<f32>,
        terminals: Vec<bool>,
        truncations: Vec<bool>,
    ) -> Self {
        Self {
            observations,
            rewards,
            terminals,
            truncations,
        }
    }

    /// Get done flags (terminal OR truncated).
    pub fn dones(&self) -> Vec<bool> {
        self.terminals
            .iter()
            .zip(self.truncations.iter())
            .map(|(&t, &tr)| t || tr)
            .collect()
    }
}

/// Mask indicating which environments need reset.
pub struct ResetMask {
    mask: Vec<bool>,
}

impl ResetMask {
    /// Create from done flags.
    pub fn from_dones(dones: &[bool]) -> Self {
        Self {
            mask: dones.to_vec(),
        }
    }

    /// Create from step result.
    pub fn from_step_result(result: &StepResult) -> Self {
        Self::from_dones(&result.dones())
    }

    /// Check if any environment needs reset.
    pub fn any(&self) -> bool {
        self.mask.iter().any(|&x| x)
    }

    /// Get the underlying mask.
    pub fn as_slice(&self) -> &[bool] {
        &self.mask
    }

    /// Number of environments that need reset.
    pub fn count(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }
}

// ============================================================================
// Discrete VectorizedEnv Trait (for IMPALA)
// ============================================================================

/// Trait for vectorized discrete environments (used by IMPALA runner).
///
/// Implementations can wrap any environment system (operant, gymnasium, etc.)
/// and provide a uniform interface for the distributed training runners.
pub trait VectorizedEnv: Send {
    /// Number of parallel environments.
    fn n_envs(&self) -> usize;

    /// Size of observation vector for single environment.
    fn obs_size(&self) -> usize;

    /// Number of discrete actions (for discrete action spaces).
    fn n_actions(&self) -> usize;

    /// Write current observations to buffer.
    ///
    /// Buffer must have size `n_envs * obs_size`.
    /// Observations are written in flat layout: [env0_obs, env1_obs, ...].
    fn write_observations(&self, buffer: &mut [f32]);

    /// Step all environments with given actions.
    ///
    /// Actions are discrete indices as f32 for compatibility with operant.
    /// Returns observations, rewards, terminals, and truncations.
    ///
    /// Note: This does NOT auto-reset terminated environments.
    /// Call `reset_envs` separately to reset specific environments.
    fn step(&mut self, actions: &[f32]) -> StepResult;

    /// Reset specific environments indicated by mask.
    fn reset_envs(&mut self, mask: &ResetMask, seed: u64);

    /// Reset all environments.
    fn reset_all(&mut self, seed: u64);

    /// Get current observations as a new vector.
    fn get_observations(&self) -> Vec<f32> {
        let mut buffer = vec![0.0f32; self.n_envs() * self.obs_size()];
        self.write_observations(&mut buffer);
        buffer
    }
}

// ============================================================================
// Operant Adapter for VectorizedEnv
// ============================================================================


use operant_core::Environment as OperantEnvironment;

use operant_core::ResetMask as OperantResetMask;

/// Adapter for operant environments implementing VectorizedEnv.

pub struct OperantAdapter<E> {
    env: E,
    n_envs: usize,
    obs_size: usize,
    n_actions: usize,
}


impl<E> OperantAdapter<E> {
    /// Create a new adapter wrapping an operant environment.
    pub fn new(env: E, n_envs: usize, obs_size: usize, n_actions: usize) -> Self {
        Self {
            env,
            n_envs,
            obs_size,
            n_actions,
        }
    }

    /// Get a reference to the underlying environment.
    pub fn inner(&self) -> &E {
        &self.env
    }

    /// Get a mutable reference to the underlying environment.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.env
    }

    /// Consume the adapter and return the underlying environment.
    pub fn into_inner(self) -> E {
        self.env
    }
}


impl<E: OperantEnvironment + Send> VectorizedEnv for OperantAdapter<E> {
    fn n_envs(&self) -> usize {
        self.n_envs
    }

    fn obs_size(&self) -> usize {
        self.obs_size
    }

    fn n_actions(&self) -> usize {
        self.n_actions
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        self.env.write_observations(buffer);
    }

    fn step(&mut self, actions: &[f32]) -> StepResult {
        let result = self.env.step_no_reset_with_result(actions);

        // Convert operant's u8 flags to bool and clone owned data
        let terminals: Vec<bool> = result.terminals.iter().map(|&t| t != 0).collect();
        let truncations: Vec<bool> = result.truncations.iter().map(|&t| t != 0).collect();

        StepResult {
            observations: result.observations.to_vec(),
            rewards: result.rewards.to_vec(),
            terminals,
            truncations,
        }
    }

    fn reset_envs(&mut self, mask: &ResetMask, seed: u64) {
        // Convert our bool mask to operant's ResetMask format
        let terminals: Vec<u8> = mask.as_slice().iter().map(|&b| if b { 1 } else { 0 }).collect();
        let truncations: Vec<u8> = vec![0; terminals.len()];
        let operant_mask = OperantResetMask::from_done_flags(&terminals, &truncations);
        self.env.reset_envs(&operant_mask, seed);
    }

    fn reset_all(&mut self, seed: u64) {
        self.env.reset(seed);
    }
}

// ============================================================================
// Simplified Environment Wrappers
// ============================================================================

/// Simple step result for wrapper environments.
#[derive(Debug, Clone)]
pub struct WrapperStepResult {
    /// Rewards for each environment
    pub rewards: Vec<f32>,
    /// Done flags (terminal or truncated)
    pub dones: Vec<bool>,
    /// True terminal flags (not truncation)
    pub terminals: Vec<bool>,
}

/// Wrapper for operant CartPole environment.
///
/// This provides a simple interface that can be adapted to the generic
/// VectorizedEnv trait in examples.

pub struct CartPoleEnvWrapper {
    env: operant::CartPole,
    /// Number of environments
    pub n_envs: usize,
}


impl CartPoleEnvWrapper {
    /// Create a new CartPole wrapper.
    pub fn new(env: operant::CartPole) -> Self {
        let n_envs = env.num_envs();
        Self { env, n_envs }
    }

    /// Write current observations to buffer.
    pub fn write_observations(&self, buffer: &mut [f32]) {
        self.env.write_observations(buffer);
    }

    /// Step with discrete actions (as u32 indices).
    pub fn step(&mut self, actions: &[u32]) -> WrapperStepResult {
        // Convert u32 to f32 for operant
        let actions_f32: Vec<f32> = actions.iter().map(|&a| a as f32).collect();
        let result = self.env.step_no_reset_with_result(&actions_f32);

        let terminals: Vec<bool> = result.terminals.iter().map(|&t| t != 0).collect();
        let truncations: Vec<bool> = result.truncations.iter().map(|&t| t != 0).collect();
        let dones: Vec<bool> = terminals
            .iter()
            .zip(truncations.iter())
            .map(|(&t, &tr)| t || tr)
            .collect();

        WrapperStepResult {
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
        let mask = OperantResetMask::from_done_flags(&terminals, &truncations);
        self.env.reset_envs(&mask, 0);
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: u64) {
        self.env.reset(seed);
    }
}

/// Wrapper for operant Pendulum environment.
///
/// This provides a simple interface for continuous action environments.

pub struct PendulumEnvWrapper {
    env: operant::Pendulum,
    /// Number of environments
    pub n_envs: usize,
}


impl PendulumEnvWrapper {
    /// Create a new Pendulum wrapper.
    pub fn new(env: operant::Pendulum) -> Self {
        let n_envs = env.num_envs();
        Self { env, n_envs }
    }

    /// Write current observations to buffer.
    pub fn write_observations(&self, buffer: &mut [f32]) {
        self.env.write_observations(buffer);
    }

    /// Step with continuous actions.
    pub fn step(&mut self, actions: &[f32]) -> WrapperStepResult {
        let result = self.env.step_no_reset_with_result(actions);

        let terminals: Vec<bool> = result.terminals.iter().map(|&t| t != 0).collect();
        let truncations: Vec<bool> = result.truncations.iter().map(|&t| t != 0).collect();
        let dones: Vec<bool> = terminals
            .iter()
            .zip(truncations.iter())
            .map(|(&t, &tr)| t || tr)
            .collect();

        WrapperStepResult {
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
        let mask = OperantResetMask::from_done_flags(&terminals, &truncations);
        self.env.reset_envs(&mask, 0);
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: u64) {
        self.env.reset(seed);
    }
}

/// Convenience function to wrap operant CartPole.

pub fn wrap_cartpole(env: operant::CartPole) -> CartPoleEnvWrapper {
    CartPoleEnvWrapper::new(env)
}

/// Convenience function to wrap operant Pendulum.

pub fn wrap_pendulum(env: operant::Pendulum) -> PendulumEnvWrapper {
    PendulumEnvWrapper::new(env)
}

// ============================================================================
// Ready-to-Use Environment Adapters for Generic Learner
// ============================================================================

use crate::algorithms::action_policy::{ContinuousAction, DiscreteAction};
use crate::runners::learner::{
    StepResult as LearnerStepResult, VectorizedEnv as LearnerVectorizedEnv,
};

/// CartPole environment adapter for the generic Learner.
///
/// Implements `VectorizedEnv<DiscreteAction>` so it can be used directly
/// with `PPODiscrete` or `RecurrentPPODiscrete`.
///
/// # Example
///
/// ```ignore
/// use burn_rl::distributed::environment::CartPoleEnv;
/// use operant::CartPole;
///
/// let env = CartPoleEnv::new(64).unwrap();
/// let learner = PPODiscrete::new(config, algorithm);
/// learner.run(model, env, optimizer, callback);
/// ```

pub struct CartPoleEnv {
    inner: CartPoleEnvWrapper,
}


impl CartPoleEnv {
    /// Observation size for CartPole.
    pub const OBS_SIZE: usize = 4;
    /// Number of discrete actions for CartPole.
    pub const N_ACTIONS: usize = 2;

    /// Create a new CartPole environment with the given number of parallel environments.
    ///
    /// The environment is automatically reset with a random seed.
    pub fn new(n_envs: usize) -> Result<Self, String> {
        let mut env = operant::CartPole::with_defaults(n_envs)
            .map_err(|e| format!("Failed to create CartPole: {:?}", e))?;
        env.reset(42); // Required: operant environments must be reset before use
        Ok(Self {
            inner: CartPoleEnvWrapper::new(env),
        })
    }

    /// Create from an existing operant CartPole environment.
    pub fn from_operant(env: operant::CartPole) -> Self {
        Self {
            inner: CartPoleEnvWrapper::new(env),
        }
    }

    /// Number of parallel environments.
    pub fn n_envs(&self) -> usize {
        self.inner.n_envs
    }
}


impl LearnerVectorizedEnv<DiscreteAction> for CartPoleEnv {
    fn n_envs(&self) -> usize {
        self.inner.n_envs
    }

    fn obs_size(&self) -> usize {
        Self::OBS_SIZE
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        self.inner.write_observations(buffer);
    }

    fn step(&mut self, actions: &[DiscreteAction]) -> LearnerStepResult {
        let action_indices: Vec<u32> = actions.iter().map(|a| a.0).collect();
        let result = self.inner.step(&action_indices);
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

/// Pendulum environment adapter for the generic Learner.
///
/// Implements `VectorizedEnv<ContinuousAction>` so it can be used directly
/// with `PPOContinuous` or `RecurrentPPOContinuous`.
///
/// # Example
///
/// ```ignore
/// use burn_rl::distributed::environment::PendulumEnv;
/// use operant::Pendulum;
///
/// let env = PendulumEnv::new(64).unwrap();
/// let learner = PPOContinuous::new(config, algorithm);
/// learner.run(model, env, optimizer, callback);
/// ```

pub struct PendulumEnv {
    inner: PendulumEnvWrapper,
}


impl PendulumEnv {
    /// Observation size for Pendulum: [cos(θ), sin(θ), θ_dot].
    pub const OBS_SIZE: usize = 3;
    /// Action dimension for Pendulum (single torque).
    pub const ACTION_DIM: usize = 1;
    /// Action bounds for Pendulum torque.
    pub const ACTION_LOW: f32 = -2.0;
    pub const ACTION_HIGH: f32 = 2.0;

    /// Create a new Pendulum environment with the given number of parallel environments.
    ///
    /// The environment is automatically reset with a random seed.
    pub fn new(n_envs: usize) -> Result<Self, String> {
        let mut env = operant::Pendulum::with_defaults(n_envs)
            .map_err(|e| format!("Failed to create Pendulum: {:?}", e))?;
        env.reset(42); // Required: operant environments must be reset before use
        Ok(Self {
            inner: PendulumEnvWrapper::new(env),
        })
    }

    /// Create from an existing operant Pendulum environment.
    pub fn from_operant(env: operant::Pendulum) -> Self {
        Self {
            inner: PendulumEnvWrapper::new(env),
        }
    }

    /// Number of parallel environments.
    pub fn n_envs(&self) -> usize {
        self.inner.n_envs
    }

    /// Get action bounds as (low, high) vectors.
    pub fn action_bounds() -> (Vec<f32>, Vec<f32>) {
        (vec![Self::ACTION_LOW], vec![Self::ACTION_HIGH])
    }
}


impl LearnerVectorizedEnv<ContinuousAction> for PendulumEnv {
    fn n_envs(&self) -> usize {
        self.inner.n_envs
    }

    fn obs_size(&self) -> usize {
        Self::OBS_SIZE
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        self.inner.write_observations(buffer);
    }

    fn step(&mut self, actions: &[ContinuousAction]) -> LearnerStepResult {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_result_dones() {
        let result = StepResult {
            observations: vec![],
            rewards: vec![1.0, 2.0, 3.0],
            terminals: vec![true, false, false],
            truncations: vec![false, false, true],
        };

        let dones = result.dones();
        assert_eq!(dones, vec![true, false, true]);
    }

    #[test]
    fn test_reset_mask() {
        let dones = vec![true, false, true, false];
        let mask = ResetMask::from_dones(&dones);

        assert!(mask.any());
        assert_eq!(mask.count(), 2);
        assert_eq!(mask.as_slice(), &[true, false, true, false]);
    }

    #[test]
    fn test_reset_mask_none() {
        let dones = vec![false, false, false];
        let mask = ResetMask::from_dones(&dones);

        assert!(!mask.any());
        assert_eq!(mask.count(), 0);
    }
}
