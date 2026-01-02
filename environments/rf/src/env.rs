//! RF World Environment
//!
//! Main environment structure for RF spectrum simulation.
//! This provides the step/reset interface for RL training.
//!
//! ## Type-Safe API
//!
//! This module uses the Curry-Howard compliant type system for RF parameters:
//! - `ValidatedFrequencyGrid` ensures valid frequency ranges at compile time
//! - Type-safe noise functions guarantee non-negative power values
//!
//! Access the validated grid via `validated_grid()` method.

use crate::config::RFWorldConfig;
use crate::state::RFWorldState;
use crate::entities::{EntitySoA, EntitySpawner};
use crate::entities::behaviors::render_all_entities;
use crate::spectrum::noise::{add_thermal_noise, add_man_made_noise};
use crate::types::ValidatedFrequencyGrid;
use crate::agents::{
    ActionConfig, MultiAgentState, MultiAgentActions,
    MultiAgentRewards, RewardConfig, compute_all_interference,
};
use crate::observation::ObservationConfig;

use crate::simd_rf::random::SimdRng;
use crate::entities::behaviors::update_all_entities;

/// RF World Environment
///
/// A vectorized RF spectrum simulation environment for training
/// RL agents in cognitive radio and jamming scenarios.
///
/// # Example (Phase 1 - Basic Shell)
///
/// ```ignore
/// use rf_environment::{RFWorld, RFWorldConfig};
///
/// let config = RFWorldConfig::new()
///     .with_num_envs(8)
///     .with_freq_bins(256)
///     .build();
///
/// let mut env = RFWorld::new(config);
/// env.reset();
///
/// for _ in 0..100 {
///     env.step();
/// }
/// ```
pub struct RFWorld {
    /// Configuration
    config: RFWorldConfig,

    /// Environment state
    state: RFWorldState,

    /// Entity state (SoA layout)
    entities: EntitySoA,

    /// Validated frequency grid (type-safe)
    validated_grid: ValidatedFrequencyGrid,

    /// Entity spawner
    spawner: EntitySpawner,

    /// SIMD Random number generator
    rng: SimdRng,

    // ========================================================================
    // Multi-Agent State (Phase 4)
    // ========================================================================
    /// Multi-agent state
    agent_state: MultiAgentState,

    /// Action configuration for agents
    action_config: ActionConfig,

    /// Reward configuration
    reward_config: RewardConfig,

    /// Observation configuration
    obs_config: ObservationConfig,
}

impl RFWorld {
    /// Create a new RF World environment with the given configuration
    pub fn new(config: RFWorldConfig) -> Self {
        let state = RFWorldState::new(&config);
        let entities = EntitySoA::new(config.num_envs, config.max_entities);

        // Create validated frequency grid (type-safe)
        let validated_grid = config.validated_grid()
            .expect("RFWorld requires valid frequency configuration");

        let spawner = EntitySpawner::new(
            config.entity_config.clone(),
            config.world_size,
            config.seed,
        );

        // Initialize multi-agent state
        let agent_state = MultiAgentState::new(
            config.num_envs,
            config.num_jammers,
            config.num_crs,
            8, // Default history length
        );

        // Set up action config from world config
        let action_config = ActionConfig::new()
            .with_freq_range(config.freq_min, config.freq_max)
            .with_power_range(-10.0, 30.0)
            .with_bandwidth_range(1e6, 40e6);

        let rng = SimdRng::new(config.seed);

        Self {
            config,
            state,
            entities,
            validated_grid,
            spawner,
            rng,
            agent_state,
            action_config,
            reward_config: RewardConfig::default(),
            obs_config: ObservationConfig::default(),
        }
    }

    /// Reset all environments to initial state
    ///
    /// Returns a reference to the state after reset.
    pub fn reset(&mut self) -> &RFWorldState {
        self.state.reset_all();

        // Re-seed RNG for reproducibility
        self.rng = SimdRng::new(self.config.seed);

        // Reset and respawn entities
        self.entities.reset_all();
        self.spawner = EntitySpawner::new(
            self.config.entity_config.clone(),
            self.config.world_size,
            self.config.seed,
        );
        for env in 0..self.config.num_envs {
            self.spawner.spawn_all(&mut self.entities, env);
        }

        // Reset agent state
        self.agent_state.reset_all();

        // Render initial PSD
        self.render_psd();

        &self.state
    }

    /// Reset with a new seed
    ///
    /// Useful for creating diverse training scenarios.
    pub fn reset_with_seed(&mut self, seed: u64) -> &RFWorldState {
        self.state.reset_all_with_seed(seed);
        self.rng = SimdRng::new(seed);

        // Reset and respawn entities with new seed
        self.entities.reset_all();
        self.spawner = EntitySpawner::new(
            self.config.entity_config.clone(),
            self.config.world_size,
            seed,
        );
        for env in 0..self.config.num_envs {
            self.spawner.spawn_all(&mut self.entities, env);
        }

        // Reset agent state
        self.agent_state.reset_all();

        // Render initial PSD
        self.render_psd();

        &self.state
    }

    /// Reset specific environments that are done
    ///
    /// Returns indices of environments that were reset.
    pub fn reset_done_envs(&mut self) -> Vec<usize> {
        let mut reset_envs = Vec::new();

        for env in 0..self.config.num_envs {
            if self.state.is_done(env) {
                self.state.reset_env(env);
                // Re-spawn entities for this environment
                self.entities.reset_env(env);
                self.spawner.spawn_all(&mut self.entities, env);
                // Reset agent state for this environment
                self.agent_state.reset_env(env);
                reset_envs.push(env);
            }
        }

        // Re-render PSD for reset environments
        if !reset_envs.is_empty() {
            self.render_psd();
        }

        reset_envs
    }

    /// Advance the simulation by one control step
    ///
    /// Updates entity behaviors and renders the PSD for each environment.
    ///
    /// # Returns
    /// Reference to the updated state
    pub fn step(&mut self) -> &RFWorldState {
        let max_steps = self.config.max_steps;
        let dt = 1.0 / self.config.ctrl_freq as f32;

        // Update entities for each environment
        for env in 0..self.config.num_envs {
            if !self.state.is_done(env) {
                // Update entity behaviors
                update_all_entities(
                    &mut self.entities,
                    env,
                    dt,
                    &mut self.rng,
                    &self.config,
                );

                let step = self.state.increment_step(env);

                // Check for truncation (max steps reached)
                if step >= max_steps {
                    self.state.set_truncated(env, true);
                }
            }
        }

        // Render PSD for all environments
        self.render_psd();

        &self.state
    }

    /// Advance the simulation with actions (legacy, calls step_multi_agent)
    pub fn step_with_actions(&mut self, actions: &[f32]) -> &RFWorldState {
        let _ = self.step_multi_agent(actions);
        &self.state
    }

    /// Advance the simulation with multi-agent actions
    ///
    /// This is the primary step function for multi-agent RL training.
    ///
    /// # Arguments
    /// * `actions` - Flat array of actions for all agents in all environments.
    ///   Layout: [env0_j0, env0_j1, ..., env0_cr0, env0_cr1, ..., env1_j0, ...]
    ///   Each jammer has 4 values (freq, bw, power, modulation) normalized to [0, 1]
    ///   Each CR has 3 values (freq, power, bw) normalized to [0, 1]
    ///
    /// # Returns
    /// Tuple of (state reference, rewards)
    pub fn step_multi_agent(&mut self, actions: &[f32]) -> (&RFWorldState, MultiAgentRewards) {
        let max_steps = self.config.max_steps;
        let dt = 1.0 / self.config.ctrl_freq as f32;
        let num_jammers = self.config.num_jammers;
        let num_crs = self.config.num_crs;

        // Parse actions
        let parsed_actions = MultiAgentActions::from_flat(
            actions,
            self.config.num_envs,
            num_jammers,
            num_crs,
        );

        // Process each environment
        for env in 0..self.config.num_envs {
            if !self.state.is_done(env) {
                // Apply jammer actions
                for j in 0..num_jammers {
                    let action = parsed_actions.jammer_action(env, j);
                    self.agent_state.apply_jammer_action(env, j, action, &self.action_config);
                }

                // Apply CR actions
                for c in 0..num_crs {
                    let action = parsed_actions.cr_action(env, c);
                    self.agent_state.apply_cr_action(env, c, action, &self.action_config);
                }

                // Update entity behaviors
                update_all_entities(
                    &mut self.entities,
                    env,
                    dt,
                    &mut self.rng,
                    &self.config,
                );

                let step = self.state.increment_step(env);

                // Check for truncation (max steps reached)
                if step >= max_steps {
                    self.state.set_truncated(env, true);
                }
            }
        }

        // Render PSD for all environments
        self.render_psd();

        // Compute interference for all environments
        let interference_matrices = compute_all_interference(
            &self.agent_state,
            self.config.noise_figure - 174.0 + 10.0 * (self.config.bandwidth().log10()), // Noise floor in dBm
            self.config.sinr_threshold_db,
        );

        // Update agent SINR based on interference
        for env in 0..self.config.num_envs {
            if !self.state.is_done(env) {
                let matrix = &interference_matrices[env];

                // Update CR SINR and throughput
                for c in 0..num_crs {
                    let sinr = matrix.sinr(c);
                    self.agent_state.update_cr_sinr(env, c, sinr, self.config.sinr_threshold_db);
                }

                // Update jammer success rates
                for j in 0..num_jammers {
                    let victim_count = matrix.jammer_victim_count(j);
                    let success = victim_count > 0;
                    self.agent_state.update_jammer_success(env, j, success, 0.1);
                }
            }
        }

        // Compute rewards
        let rewards = MultiAgentRewards::compute(
            &self.agent_state,
            &interference_matrices,
            &self.reward_config,
        );

        // Update episode returns
        for env in 0..self.config.num_envs {
            let total_reward = rewards.total_jammer_reward(env) + rewards.total_cr_reward(env);
            self.state.episode_returns[env] += total_reward;
        }

        (&self.state, rewards)
    }

    /// Render PSD for all environments
    ///
    /// Clears the PSD, adds noise floor, then renders all entity contributions.
    fn render_psd(&mut self) {
        let num_envs = self.config.num_envs;
        let num_bins = self.config.num_freq_bins;

        // Clear PSD
        self.state.psd.fill(0.0);

        // Add noise floor and entity contributions for each environment
        for env in 0..num_envs {
            let psd_offset = env * num_bins;
            let psd_slice = &mut self.state.psd[psd_offset..psd_offset + num_bins];

            // Add thermal noise floor
            add_thermal_noise(psd_slice, &self.validated_grid, self.config.noise_figure);

            // Add man-made noise
            add_man_made_noise(psd_slice, &self.validated_grid, self.config.noise_environment);

            // Render entity contributions
            render_all_entities(
                &self.entities,
                &mut self.state.psd,
                &self.validated_grid,
                env,
                &self.config,
            );
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get a reference to the current state
    #[inline]
    pub fn state(&self) -> &RFWorldState {
        &self.state
    }

    /// Get a mutable reference to the state
    ///
    /// Use with caution - direct state manipulation may break invariants.
    #[inline]
    pub fn state_mut(&mut self) -> &mut RFWorldState {
        &mut self.state
    }

    /// Get a reference to the entity state
    #[inline]
    pub fn entities(&self) -> &EntitySoA {
        &self.entities
    }

    /// Get a mutable reference to the entity state
    #[inline]
    pub fn entities_mut(&mut self) -> &mut EntitySoA {
        &mut self.entities
    }

    /// Get a reference to the validated frequency grid
    ///
    /// This grid provides compile-time guarantees about frequency bounds:
    /// - `freq_min > 0`
    /// - `freq_max > freq_min`
    /// - `num_bins > 0`
    #[inline]
    pub fn validated_grid(&self) -> &ValidatedFrequencyGrid {
        &self.validated_grid
    }

    /// Get a reference to the configuration
    #[inline]
    pub fn config(&self) -> &RFWorldConfig {
        &self.config
    }

    /// Get the number of environments
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.config.num_envs
    }

    /// Get the number of frequency bins
    #[inline]
    pub fn num_freq_bins(&self) -> usize {
        self.config.num_freq_bins
    }

    /// Get the SIMD RNG (for testing or advanced usage)
    #[inline]
    pub fn rng(&mut self) -> &mut SimdRng {
        &mut self.rng
    }

    // ========================================================================
    // Multi-Agent Accessors
    // ========================================================================

    /// Get a reference to the multi-agent state
    #[inline]
    pub fn agent_state(&self) -> &MultiAgentState {
        &self.agent_state
    }

    /// Get a mutable reference to the multi-agent state
    #[inline]
    pub fn agent_state_mut(&mut self) -> &mut MultiAgentState {
        &mut self.agent_state
    }

    /// Get a reference to the action configuration
    #[inline]
    pub fn action_config(&self) -> &ActionConfig {
        &self.action_config
    }

    /// Set the action configuration
    pub fn set_action_config(&mut self, config: ActionConfig) {
        self.action_config = config;
    }

    /// Get a reference to the reward configuration
    #[inline]
    pub fn reward_config(&self) -> &RewardConfig {
        &self.reward_config
    }

    /// Set the reward configuration
    pub fn set_reward_config(&mut self, config: RewardConfig) {
        self.reward_config = config;
    }

    /// Get a reference to the observation configuration
    #[inline]
    pub fn obs_config(&self) -> &ObservationConfig {
        &self.obs_config
    }

    /// Set the observation configuration
    pub fn set_obs_config(&mut self, config: ObservationConfig) {
        self.obs_config = config;
    }

    /// Get the number of jammers per environment
    #[inline]
    pub fn num_jammers(&self) -> usize {
        self.config.num_jammers
    }

    /// Get the number of cognitive radios per environment
    #[inline]
    pub fn num_crs(&self) -> usize {
        self.config.num_crs
    }

    // ========================================================================
    // Observation Helpers
    // ========================================================================

    /// Get the jammer observation dimension
    pub fn jammer_observation_dim(&self) -> usize {
        self.obs_config.jammer_observation_size(
            self.config.num_jammers,
            self.config.num_crs,
        )
    }

    /// Get the CR observation dimension
    pub fn cr_observation_dim(&self) -> usize {
        self.obs_config.cr_observation_size(
            self.config.num_jammers,
            self.config.num_crs,
        )
    }

    /// Get the observation dimension (for legacy compatibility)
    ///
    /// Returns the PSD size for basic observations.
    /// Use `jammer_observation_dim()` or `cr_observation_dim()` for multi-agent.
    pub fn observation_dim(&self) -> usize {
        self.config.num_freq_bins
    }

    /// Get observations for all environments
    ///
    /// Returns a flat array of [num_envs × observation_dim].
    /// For Phase 1, this is just the normalized PSD.
    pub fn get_observations(&self) -> Vec<f32> {
        // For Phase 1, just return the raw PSD
        // Normalization and additional features will be added later
        self.state.psd.clone()
    }

    /// Get observation for a single environment
    pub fn get_observation(&self, env: usize) -> &[f32] {
        self.state.psd_slice(env)
    }

    // ========================================================================
    // Action Space Helpers
    // ========================================================================

    /// Get the action dimension per environment
    ///
    /// Returns num_jammers * 4 + num_crs * 3
    /// (4 values per jammer: freq, bw, power, modulation)
    /// (3 values per CR: freq, power, bw)
    pub fn action_dim(&self) -> usize {
        self.config.num_jammers * 4 + self.config.num_crs * 3
    }

    /// Get the jammer action dimension
    pub fn jammer_action_dim(&self) -> usize {
        4 // freq, bw, power, modulation
    }

    /// Get the CR action dimension
    pub fn cr_action_dim(&self) -> usize {
        3 // freq, power, bw
    }

    /// Get total action size for all environments
    pub fn total_action_size(&self) -> usize {
        self.config.num_envs * self.action_dim()
    }

    // ========================================================================
    // Episode Info
    // ========================================================================

    /// Get episode lengths for all environments
    pub fn episode_lengths(&self) -> &[u32] {
        &self.state.step_count
    }

    /// Get episode returns for all environments
    pub fn episode_returns(&self) -> &[f32] {
        &self.state.episode_returns
    }

    /// Check if any environment is done
    pub fn any_done(&self) -> bool {
        (0..self.config.num_envs).any(|env| self.state.is_done(env))
    }

    /// Check if all environments are done
    pub fn all_done(&self) -> bool {
        (0..self.config.num_envs).all(|env| self.state.is_done(env))
    }

    /// Get indices of done environments
    pub fn done_envs(&self) -> Vec<usize> {
        (0..self.config.num_envs)
            .filter(|&env| self.state.is_done(env))
            .collect()
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
            .with_max_steps(100)
            .build()
    }

    #[test]
    fn test_env_creation() {
        let config = test_config();
        let env = RFWorld::new(config);

        assert_eq!(env.num_envs(), 8);
        assert_eq!(env.num_freq_bins(), 256);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut env = RFWorld::new(config);

        // Step a bit
        for _ in 0..10 {
            env.step();
        }

        // Verify steps incremented
        assert!(env.state().step_count[0] > 0);

        // Reset
        env.reset();

        // Verify reset
        assert_eq!(env.state().step_count[0], 0);
    }

    #[test]
    fn test_step_increments() {
        let config = test_config();
        let mut env = RFWorld::new(config);

        env.reset();

        for i in 1..=10 {
            env.step();
            assert_eq!(env.state().step_count[0], i);
        }
    }

    #[test]
    fn test_truncation() {
        let config = RFWorldConfig::new()
            .with_num_envs(8)
            .with_max_steps(10)
            .build();

        let mut env = RFWorld::new(config);
        env.reset();

        // Step until truncation
        for _ in 0..10 {
            env.step();
        }

        // All environments should be truncated
        assert!(env.all_done());
        for i in 0..8 {
            assert!(env.state().is_truncated(i));
        }
    }

    #[test]
    fn test_reset_done_envs() {
        let config = RFWorldConfig::new()
            .with_num_envs(8)
            .with_max_steps(10)
            .build();

        let mut env = RFWorld::new(config);
        env.reset();

        // Manually set some envs as done
        env.state_mut().set_terminal(0, true);
        env.state_mut().set_truncated(2, true);

        let reset_envs = env.reset_done_envs();

        assert!(reset_envs.contains(&0));
        assert!(reset_envs.contains(&2));
        assert!(!reset_envs.contains(&1));
    }

    #[test]
    fn test_observations() {
        let config = test_config();
        let env = RFWorld::new(config);

        let obs = env.get_observations();
        assert_eq!(obs.len(), 8 * 256);

        let single_obs = env.get_observation(0);
        assert_eq!(single_obs.len(), 256);
    }

    #[test]
    fn test_done_envs() {
        let config = test_config();
        let mut env = RFWorld::new(config);
        env.reset();

        env.state_mut().set_terminal(1, true);
        env.state_mut().set_terminal(5, true);

        let done = env.done_envs();
        assert_eq!(done.len(), 2);
        assert!(done.contains(&1));
        assert!(done.contains(&5));
    }

    #[test]
    fn test_any_all_done() {
        let config = test_config();
        let mut env = RFWorld::new(config);
        env.reset();

        assert!(!env.any_done());
        assert!(!env.all_done());

        env.state_mut().set_terminal(0, true);
        assert!(env.any_done());
        assert!(!env.all_done());

        for i in 0..8 {
            env.state_mut().set_terminal(i, true);
        }
        assert!(env.all_done());
    }

    #[test]
    fn test_reset_with_seed() {
        let config = test_config();
        let mut env = RFWorld::new(config);

        env.reset_with_seed(12345);
        assert_eq!(env.state().step_count[0], 0);
    }

    #[test]
    fn test_validated_grid() {
        let config = RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(256)
            .with_freq_range(1e9, 2e9)
            .build();

        let env = RFWorld::new(config);
        let grid = env.validated_grid();

        assert_eq!(grid.num_bins(), 256);
        assert!((grid.freq_min().as_hz() - 1e9).abs() < 1.0);
        assert!((grid.freq_max().as_hz() - 2e9).abs() < 1.0);
    }
}
