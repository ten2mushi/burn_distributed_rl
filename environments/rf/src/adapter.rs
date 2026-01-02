//! Adapter for distributed_rl integration
//!
//! Provides adapters to use RFWorld with the distributed_rl framework,
//! implementing both the IMPALA-style `VectorizedEnv` trait and the
//! generic `LearnerVectorizedEnv` trait.

use crate::config::RFWorldConfig;
use crate::env::RFWorld;
use crate::observation::{MultiAgentObservations, ObservationConfig};

// ============================================================================
// StepResult (matches distributed_rl::environment::StepResult)
// ============================================================================

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
    /// Get done flags (terminal OR truncated).
    pub fn dones(&self) -> Vec<bool> {
        self.terminals
            .iter()
            .zip(self.truncations.iter())
            .map(|(&t, &tr)| t || tr)
            .collect()
    }
}

// ============================================================================
// ResetMask
// ============================================================================

/// Mask indicating which environments need reset.
#[derive(Debug, Clone)]
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

    /// Get indices that need reset.
    pub fn indices(&self) -> Vec<usize> {
        self.mask
            .iter()
            .enumerate()
            .filter_map(|(i, &reset)| if reset { Some(i) } else { None })
            .collect()
    }
}

// ============================================================================
// VectorizedEnv Trait (for IMPALA/distributed training)
// ============================================================================

/// Trait for vectorized discrete environments.
///
/// This matches the distributed_rl VectorizedEnv trait interface.
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
    fn write_observations(&self, buffer: &mut [f32]);

    /// Step all environments with given actions.
    ///
    /// Actions are discrete indices as f32 for compatibility.
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
// RF Environment Adapter
// ============================================================================

/// Agent view configuration for the adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentView {
    /// Single jammer perspective (for jammer training)
    Jammer(usize),
    /// Single CR perspective (for CR training)
    CognitiveRadio(usize),
    /// Combined view (aggregates all agent rewards)
    Combined,
}

impl Default for AgentView {
    fn default() -> Self {
        AgentView::Jammer(0)
    }
}

/// RF Environment adapter implementing VectorizedEnv.
///
/// Wraps RFWorld to provide the VectorizedEnv interface required
/// by the distributed_rl framework.
///
/// # Example
///
/// ```ignore
/// use rf_environment::{RFEnvAdapter, RFWorldConfig, AgentView};
///
/// let config = RFWorldConfig::new()
///     .with_num_envs(64)
///     .with_num_jammers(1)
///     .with_num_crs(1);
///
/// let mut env = RFEnvAdapter::new(config, AgentView::Jammer(0));
/// env.reset_all(42);
///
/// let obs = env.get_observations();
/// let actions = vec![0.0; 64]; // Discrete action indices
/// let result = env.step(&actions);
/// ```
pub struct RFEnvAdapter {
    /// Inner RF world
    world: RFWorld,
    /// Which agent's perspective to use
    agent_view: AgentView,
    /// Observation configuration
    obs_config: ObservationConfig,
    /// Cached observation buffer
    obs_buffer: Vec<f32>,
    /// Number of discrete actions (if using discrete action space)
    n_discrete_actions: usize,
}

impl RFEnvAdapter {
    /// Create a new RF environment adapter.
    ///
    /// # Arguments
    /// * `config` - RF world configuration
    /// * `agent_view` - Which agent's perspective to use for observations/rewards
    pub fn new(config: RFWorldConfig, agent_view: AgentView) -> Self {
        let obs_config = ObservationConfig::default()
            .with_psd_bins(64);

        let obs_size = match agent_view {
            AgentView::Jammer(_) => obs_config.jammer_observation_size(
                config.num_jammers,
                config.num_crs,
            ),
            AgentView::CognitiveRadio(_) => obs_config.cr_observation_size(
                config.num_jammers,
                config.num_crs,
            ),
            AgentView::Combined => {
                // Combined: use jammer observation size
                obs_config.jammer_observation_size(
                    config.num_jammers,
                    config.num_crs,
                )
            }
        };

        let n_envs = config.num_envs;
        let obs_buffer = vec![0.0; n_envs * obs_size];

        // Calculate number of discrete actions based on discretization
        // Default: 10 frequency bins × 5 bandwidth options × 5 power levels = 250
        let n_discrete_actions = Self::compute_discrete_action_count(&config);

        let world = RFWorld::new(config);

        Self {
            world,
            agent_view,
            obs_config,
            obs_buffer,
            n_discrete_actions,
        }
    }

    /// Create with custom observation config.
    pub fn with_obs_config(mut self, obs_config: ObservationConfig) -> Self {
        let obs_size = match self.agent_view {
            AgentView::Jammer(_) => obs_config.jammer_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
            AgentView::CognitiveRadio(_) => obs_config.cr_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
            AgentView::Combined => obs_config.jammer_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
        };
        self.obs_buffer = vec![0.0; self.world.num_envs() * obs_size];
        self.obs_config = obs_config;
        self
    }

    /// Set the number of discrete actions.
    pub fn with_discrete_actions(mut self, n_actions: usize) -> Self {
        self.n_discrete_actions = n_actions;
        self
    }

    /// Get a reference to the inner RFWorld.
    pub fn inner(&self) -> &RFWorld {
        &self.world
    }

    /// Get a mutable reference to the inner RFWorld.
    pub fn inner_mut(&mut self) -> &mut RFWorld {
        &mut self.world
    }

    /// Compute number of discrete actions from config.
    fn compute_discrete_action_count(config: &RFWorldConfig) -> usize {
        // Discretization scheme:
        // - 20 frequency choices (across band)
        // - 5 bandwidth options
        // - 5 power levels
        // For jammer: 20 × 5 × 5 = 500
        // We'll use a more reasonable 100 for now
        let _ = config; // Config can influence this
        100
    }

    /// Convert discrete action index to continuous action.
    ///
    /// The action space is factored as:
    /// action = freq_idx * (n_bw * n_power) + bw_idx * n_power + power_idx
    fn discrete_to_continuous(&self, action: u32) -> [f32; 4] {
        let n_freq = 20;
        let n_bw = 5;
        let n_power = 5;
        // Total: 500, but we use modular arithmetic

        let action = action as usize % (n_freq * n_bw * n_power);

        let freq_idx = action / (n_bw * n_power);
        let remainder = action % (n_bw * n_power);
        let bw_idx = remainder / n_power;
        let power_idx = remainder % n_power;

        // Normalize to [0, 1]
        let freq_norm = freq_idx as f32 / (n_freq - 1) as f32;
        let bw_norm = bw_idx as f32 / (n_bw - 1) as f32;
        let power_norm = power_idx as f32 / (n_power - 1) as f32;
        let mod_norm = 0.0; // Default modulation

        [freq_norm, bw_norm, power_norm, mod_norm]
    }

    /// Build observations for current state.
    fn build_observations(&self) -> Vec<f32> {
        let observations = MultiAgentObservations::build(
            self.world.state(),
            self.world.agent_state(),
            &self.obs_config,
        );

        match self.agent_view {
            AgentView::Jammer(idx) => {
                // Extract observations for this specific jammer
                observations.jammer_observations_flat(idx)
            }
            AgentView::CognitiveRadio(idx) => {
                observations.cr_observations_flat(idx)
            }
            AgentView::Combined => {
                // Use jammer 0 observations for combined view
                observations.jammer_observations_flat(0)
            }
        }
    }

    /// Extract rewards for the agent view.
    fn extract_rewards(&self, rewards: &crate::agents::MultiAgentRewards) -> Vec<f32> {
        let n_envs = self.world.num_envs();
        let mut result = vec![0.0; n_envs];

        for env in 0..n_envs {
            result[env] = match self.agent_view {
                AgentView::Jammer(idx) => rewards.jammer_reward(env, idx),
                AgentView::CognitiveRadio(idx) => rewards.cr_reward(env, idx),
                AgentView::Combined => {
                    // Sum all jammer rewards for combined view
                    rewards.total_jammer_reward(env)
                }
            };
        }

        result
    }
}

impl VectorizedEnv for RFEnvAdapter {
    fn n_envs(&self) -> usize {
        self.world.num_envs()
    }

    fn obs_size(&self) -> usize {
        match self.agent_view {
            AgentView::Jammer(_) => self.obs_config.jammer_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
            AgentView::CognitiveRadio(_) => self.obs_config.cr_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
            AgentView::Combined => self.obs_config.jammer_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
        }
    }

    fn n_actions(&self) -> usize {
        self.n_discrete_actions
    }

    fn write_observations(&self, buffer: &mut [f32]) {
        let obs = self.build_observations();
        buffer[..obs.len()].copy_from_slice(&obs);
    }

    fn step(&mut self, actions: &[f32]) -> StepResult {
        let n_envs = self.world.num_envs();
        let num_jammers = self.world.num_jammers();
        let num_crs = self.world.num_crs();

        // Convert discrete actions to continuous
        // Each env gets one discrete action that we expand
        let action_dim = num_jammers * 4 + num_crs * 3;
        let mut continuous_actions = vec![0.0f32; n_envs * action_dim];

        for env in 0..n_envs {
            let discrete_action = actions[env] as u32;
            let [freq, bw, power, modulation] = self.discrete_to_continuous(discrete_action);

            // Set action for jammer 0 (or according to agent_view)
            let base = env * action_dim;
            match self.agent_view {
                AgentView::Jammer(j) => {
                    let jammer_base = base + j * 4;
                    continuous_actions[jammer_base] = freq;
                    continuous_actions[jammer_base + 1] = bw;
                    continuous_actions[jammer_base + 2] = power;
                    continuous_actions[jammer_base + 3] = modulation;
                }
                AgentView::CognitiveRadio(c) => {
                    let cr_base = base + num_jammers * 4 + c * 3;
                    continuous_actions[cr_base] = freq;
                    continuous_actions[cr_base + 1] = power;
                    continuous_actions[cr_base + 2] = bw;
                }
                AgentView::Combined => {
                    // Apply to jammer 0
                    continuous_actions[base] = freq;
                    continuous_actions[base + 1] = bw;
                    continuous_actions[base + 2] = power;
                    continuous_actions[base + 3] = modulation;
                }
            }
        }

        // Step the world
        let (_state, rewards) = self.world.step_multi_agent(&continuous_actions);

        // Build observations
        let observations = self.build_observations();

        // Extract state info
        let state = self.world.state();
        let terminals: Vec<bool> = (0..n_envs).map(|i| state.is_terminal(i)).collect();
        let truncations: Vec<bool> = (0..n_envs).map(|i| state.is_truncated(i)).collect();

        StepResult {
            observations,
            rewards: self.extract_rewards(&rewards),
            terminals,
            truncations,
        }
    }

    fn reset_envs(&mut self, mask: &ResetMask, seed: u64) {
        let _ = seed; // Could be used for reproducibility

        // Reset environments marked in mask
        for (env, &should_reset) in mask.as_slice().iter().enumerate() {
            if should_reset {
                self.world.state_mut().reset_env(env);
                self.world.entities_mut().reset_env(env);
                self.world.agent_state_mut().reset_env(env);
            }
        }
    }

    fn reset_all(&mut self, seed: u64) {
        self.world.reset_with_seed(seed);
    }
}

// ============================================================================
// Continuous Action Adapter
// ============================================================================

/// RF Environment adapter with continuous action space.
///
/// This variant is for algorithms like PPO with continuous actions
/// where actions are directly the normalized parameters.
pub struct RFContinuousEnvAdapter {
    /// Inner RF world
    world: RFWorld,
    /// Which agent's perspective to use
    agent_view: AgentView,
    /// Observation configuration
    obs_config: ObservationConfig,
}

impl RFContinuousEnvAdapter {
    /// Create a new continuous action RF environment adapter.
    pub fn new(config: RFWorldConfig, agent_view: AgentView) -> Self {
        let obs_config = ObservationConfig::default().with_psd_bins(64);
        let world = RFWorld::new(config);

        Self {
            world,
            agent_view,
            obs_config,
        }
    }

    /// Create with custom observation config.
    pub fn with_obs_config(mut self, obs_config: ObservationConfig) -> Self {
        self.obs_config = obs_config;
        self
    }

    /// Get reference to inner world.
    pub fn inner(&self) -> &RFWorld {
        &self.world
    }

    /// Get mutable reference to inner world.
    pub fn inner_mut(&mut self) -> &mut RFWorld {
        &mut self.world
    }

    /// Number of environments.
    pub fn n_envs(&self) -> usize {
        self.world.num_envs()
    }

    /// Observation size.
    pub fn obs_size(&self) -> usize {
        match self.agent_view {
            AgentView::Jammer(_) => self.obs_config.jammer_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
            AgentView::CognitiveRadio(_) => self.obs_config.cr_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
            AgentView::Combined => self.obs_config.jammer_observation_size(
                self.world.num_jammers(),
                self.world.num_crs(),
            ),
        }
    }

    /// Action dimension per environment.
    pub fn action_dim(&self) -> usize {
        match self.agent_view {
            AgentView::Jammer(_) => 4, // freq, bw, power, modulation
            AgentView::CognitiveRadio(_) => 3, // freq, power, bw
            AgentView::Combined => self.world.action_dim(),
        }
    }

    /// Write observations to buffer.
    pub fn write_observations(&self, buffer: &mut [f32]) {
        let obs = MultiAgentObservations::build(
            self.world.state(),
            self.world.agent_state(),
            &self.obs_config,
        );

        let flat = match self.agent_view {
            AgentView::Jammer(idx) => obs.jammer_observations_flat(idx),
            AgentView::CognitiveRadio(idx) => obs.cr_observations_flat(idx),
            AgentView::Combined => obs.jammer_observations_flat(0),
        };

        buffer[..flat.len()].copy_from_slice(&flat);
    }

    /// Step with continuous actions.
    ///
    /// Actions are [n_envs, action_dim] in flattened row-major order.
    pub fn step(&mut self, actions: &[f32]) -> StepResult {
        let n_envs = self.world.num_envs();
        let num_jammers = self.world.num_jammers();
        let num_crs = self.world.num_crs();
        let action_dim = self.action_dim();
        let full_action_dim = num_jammers * 4 + num_crs * 3;

        // Build full action array
        let mut full_actions = vec![0.5f32; n_envs * full_action_dim];

        for env in 0..n_envs {
            let input_base = env * action_dim;
            let output_base = env * full_action_dim;

            match self.agent_view {
                AgentView::Jammer(j) => {
                    let jammer_base = output_base + j * 4;
                    for i in 0..4.min(action_dim) {
                        full_actions[jammer_base + i] = actions[input_base + i];
                    }
                }
                AgentView::CognitiveRadio(c) => {
                    let cr_base = output_base + num_jammers * 4 + c * 3;
                    for i in 0..3.min(action_dim) {
                        full_actions[cr_base + i] = actions[input_base + i];
                    }
                }
                AgentView::Combined => {
                    for i in 0..full_action_dim.min(action_dim) {
                        full_actions[output_base + i] = actions[input_base + i];
                    }
                }
            }
        }

        // Step
        let (_state, rewards) = self.world.step_multi_agent(&full_actions);

        // Build result
        let state = self.world.state();
        let terminals: Vec<bool> = (0..n_envs).map(|i| state.is_terminal(i)).collect();
        let truncations: Vec<bool> = (0..n_envs).map(|i| state.is_truncated(i)).collect();

        let obs = MultiAgentObservations::build(
            self.world.state(),
            self.world.agent_state(),
            &self.obs_config,
        );

        let observations = match self.agent_view {
            AgentView::Jammer(idx) => obs.jammer_observations_flat(idx),
            AgentView::CognitiveRadio(idx) => obs.cr_observations_flat(idx),
            AgentView::Combined => obs.jammer_observations_flat(0),
        };

        let reward_vec: Vec<f32> = (0..n_envs)
            .map(|env| match self.agent_view {
                AgentView::Jammer(idx) => rewards.jammer_reward(env, idx),
                AgentView::CognitiveRadio(idx) => rewards.cr_reward(env, idx),
                AgentView::Combined => rewards.total_jammer_reward(env),
            })
            .collect();

        StepResult {
            observations,
            rewards: reward_vec,
            terminals,
            truncations,
        }
    }

    /// Reset specific environments.
    pub fn reset_envs(&mut self, indices: &[usize]) {
        for &env in indices {
            self.world.state_mut().reset_env(env);
            self.world.entities_mut().reset_env(env);
            self.world.agent_state_mut().reset_env(env);
        }
    }

    /// Reset all environments.
    pub fn reset_all(&mut self, seed: u64) {
        self.world.reset_with_seed(seed);
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RFWorldConfig {
        RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(256)
            .with_num_jammers(1)
            .with_num_crs(1)
            .with_max_steps(100)
            .build()
    }

    #[test]
    fn test_adapter_creation() {
        let config = test_config();
        let adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        assert_eq!(adapter.n_envs(), 8);
        assert!(adapter.obs_size() > 0);
        assert!(adapter.n_actions() > 0);
    }

    #[test]
    fn test_adapter_reset() {
        let config = test_config();
        let mut adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        adapter.reset_all(42);

        let obs = adapter.get_observations();
        assert_eq!(obs.len(), 8 * adapter.obs_size());
    }

    #[test]
    fn test_adapter_step() {
        let config = test_config();
        let mut adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        adapter.reset_all(42);

        let actions = vec![0.0f32; 8];
        let result = adapter.step(&actions);

        assert_eq!(result.observations.len(), 8 * adapter.obs_size());
        assert_eq!(result.rewards.len(), 8);
        assert_eq!(result.terminals.len(), 8);
        assert_eq!(result.truncations.len(), 8);
    }

    #[test]
    fn test_discrete_to_continuous() {
        let config = test_config();
        let adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        // Action 0 should give low values
        let [freq, bw, power, _] = adapter.discrete_to_continuous(0);
        assert_eq!(freq, 0.0);
        assert_eq!(bw, 0.0);
        assert_eq!(power, 0.0);

        // Action 99 (last in first set) should give higher values
        let [freq, _, _, _] = adapter.discrete_to_continuous(99);
        assert!(freq > 0.0);
    }

    #[test]
    fn test_reset_mask() {
        let dones = vec![true, false, true, false, false, false, true, false];
        let mask = ResetMask::from_dones(&dones);

        assert!(mask.any());
        assert_eq!(mask.count(), 3);
        assert_eq!(mask.indices(), vec![0, 2, 6]);
    }

    #[test]
    fn test_continuous_adapter() {
        let config = test_config();
        let mut adapter = RFContinuousEnvAdapter::new(config, AgentView::Jammer(0));

        adapter.reset_all(42);

        assert_eq!(adapter.action_dim(), 4); // Jammer: freq, bw, power, mod

        let actions = vec![0.5f32; 8 * 4]; // 8 envs × 4 actions
        let result = adapter.step(&actions);

        assert_eq!(result.rewards.len(), 8);
    }

    #[test]
    fn test_agent_views() {
        let config = test_config();

        // Test Jammer view
        let jammer_adapter = RFEnvAdapter::new(config.clone(), AgentView::Jammer(0));
        assert!(jammer_adapter.obs_size() > 0);

        // Test CR view
        let cr_adapter = RFEnvAdapter::new(config.clone(), AgentView::CognitiveRadio(0));
        assert!(cr_adapter.obs_size() > 0);

        // Test Combined view
        let combined_adapter = RFEnvAdapter::new(config, AgentView::Combined);
        assert!(combined_adapter.obs_size() > 0);
    }
}
