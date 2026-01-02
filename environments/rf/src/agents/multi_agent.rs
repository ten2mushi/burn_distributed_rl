//! Multi-Agent State Management
//!
//! Struct-of-Arrays (SoA) layout for efficient multi-agent state storage.
//! Supports parallel environments with configurable numbers of jammers
//! and cognitive radios.

use super::actions::{ActionConfig, CognitiveRadioAction, JammerAction, JammerModulation};

// ============================================================================
// Agent Type
// ============================================================================

/// Type of agent in the environment
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentType {
    /// Jammer agent
    Jammer,
    /// Cognitive Radio agent
    CognitiveRadio,
}

// ============================================================================
// Multi-Agent State (SoA Layout)
// ============================================================================

/// Multi-agent state with Struct-of-Arrays layout
///
/// All arrays are organized as [num_envs × num_agents] for efficient
/// SIMD processing across environments.
#[derive(Clone, Debug)]
pub struct MultiAgentState {
    // ========================================================================
    // Configuration
    // ========================================================================
    /// Number of parallel environments
    pub num_envs: usize,
    /// Number of jammer agents per environment
    pub num_jammers: usize,
    /// Number of cognitive radio agents per environment
    pub num_crs: usize,
    /// History length for frequency tracking
    pub history_length: usize,

    // ========================================================================
    // Jammer State: [num_envs × num_jammers]
    // ========================================================================
    /// Jammer X position (meters)
    pub jammer_x: Vec<f32>,
    /// Jammer Y position (meters)
    pub jammer_y: Vec<f32>,
    /// Jammer current frequency (Hz)
    pub jammer_freq: Vec<f32>,
    /// Jammer current bandwidth (Hz)
    pub jammer_bandwidth: Vec<f32>,
    /// Jammer current power (dBm)
    pub jammer_power: Vec<f32>,
    /// Jammer modulation type
    pub jammer_modulation: Vec<u8>,
    /// Jammer target CR index (-1 = no target)
    pub jammer_target: Vec<i32>,
    /// Jammer success rate (exponential moving average)
    pub jammer_success_rate: Vec<f32>,

    // ========================================================================
    // Cognitive Radio State: [num_envs × num_crs]
    // ========================================================================
    /// CR X position (meters)
    pub cr_x: Vec<f32>,
    /// CR Y position (meters)
    pub cr_y: Vec<f32>,
    /// CR current frequency (Hz)
    pub cr_freq: Vec<f32>,
    /// CR current bandwidth (Hz)
    pub cr_bandwidth: Vec<f32>,
    /// CR current power (dBm)
    pub cr_power: Vec<f32>,
    /// CR current throughput (normalized 0-1)
    pub cr_throughput: Vec<f32>,
    /// CR collision count in current episode
    pub cr_collisions: Vec<u32>,
    /// CR hop count in current episode
    pub cr_hop_count: Vec<u32>,
    /// CR current SINR (dB)
    pub cr_sinr: Vec<f32>,
    /// CR alive status (1 = active, 0 = jammed/failed)
    pub cr_alive: Vec<u8>,

    // ========================================================================
    // History Tracking: [num_envs × num_agents × history_length]
    // ========================================================================
    /// Jammer frequency history
    pub jammer_freq_history: Vec<f32>,
    /// CR frequency history
    pub cr_freq_history: Vec<f32>,
}

impl MultiAgentState {
    /// Create new multi-agent state
    pub fn new(
        num_envs: usize,
        num_jammers: usize,
        num_crs: usize,
        history_length: usize,
    ) -> Self {
        let jammer_total = num_envs * num_jammers;
        let cr_total = num_envs * num_crs;
        let jammer_history_total = jammer_total * history_length;
        let cr_history_total = cr_total * history_length;

        Self {
            num_envs,
            num_jammers,
            num_crs,
            history_length,

            // Jammer state
            jammer_x: vec![0.0; jammer_total],
            jammer_y: vec![0.0; jammer_total],
            jammer_freq: vec![0.0; jammer_total],
            jammer_bandwidth: vec![0.0; jammer_total],
            jammer_power: vec![0.0; jammer_total],
            jammer_modulation: vec![0; jammer_total],
            jammer_target: vec![-1; jammer_total],
            jammer_success_rate: vec![0.0; jammer_total],

            // CR state
            cr_x: vec![0.0; cr_total],
            cr_y: vec![0.0; cr_total],
            cr_freq: vec![0.0; cr_total],
            cr_bandwidth: vec![0.0; cr_total],
            cr_power: vec![0.0; cr_total],
            cr_throughput: vec![0.0; cr_total],
            cr_collisions: vec![0; cr_total],
            cr_hop_count: vec![0; cr_total],
            cr_sinr: vec![0.0; cr_total],
            cr_alive: vec![1; cr_total],

            // History
            jammer_freq_history: vec![0.0; jammer_history_total],
            cr_freq_history: vec![0.0; cr_history_total],
        }
    }

    /// Create with default history length of 8
    pub fn with_defaults(num_envs: usize, num_jammers: usize, num_crs: usize) -> Self {
        Self::new(num_envs, num_jammers, num_crs, 8)
    }

    // ========================================================================
    // Indexing Functions
    // ========================================================================

    /// Get index into jammer arrays for (env, jammer)
    #[inline]
    pub fn jammer_idx(&self, env: usize, jammer: usize) -> usize {
        debug_assert!(env < self.num_envs);
        debug_assert!(jammer < self.num_jammers);
        env * self.num_jammers + jammer
    }

    /// Get index into CR arrays for (env, cr)
    #[inline]
    pub fn cr_idx(&self, env: usize, cr: usize) -> usize {
        debug_assert!(env < self.num_envs);
        debug_assert!(cr < self.num_crs);
        env * self.num_crs + cr
    }

    /// Get index into jammer history for (env, jammer, history_slot)
    #[inline]
    pub fn jammer_history_idx(&self, env: usize, jammer: usize, slot: usize) -> usize {
        debug_assert!(slot < self.history_length);
        (env * self.num_jammers + jammer) * self.history_length + slot
    }

    /// Get index into CR history for (env, cr, history_slot)
    #[inline]
    pub fn cr_history_idx(&self, env: usize, cr: usize, slot: usize) -> usize {
        debug_assert!(slot < self.history_length);
        (env * self.num_crs + cr) * self.history_length + slot
    }

    // ========================================================================
    // Reset Functions
    // ========================================================================

    /// Reset a single environment
    pub fn reset_env(&mut self, env: usize) {
        // Reset jammer state
        for j in 0..self.num_jammers {
            let idx = self.jammer_idx(env, j);
            self.jammer_x[idx] = 0.0;
            self.jammer_y[idx] = 0.0;
            self.jammer_freq[idx] = 0.0;
            self.jammer_bandwidth[idx] = 0.0;
            self.jammer_power[idx] = 0.0;
            self.jammer_modulation[idx] = 0;
            self.jammer_target[idx] = -1;
            self.jammer_success_rate[idx] = 0.0;

            // Reset jammer history
            for h in 0..self.history_length {
                let hist_idx = self.jammer_history_idx(env, j, h);
                self.jammer_freq_history[hist_idx] = 0.0;
            }
        }

        // Reset CR state
        for c in 0..self.num_crs {
            let idx = self.cr_idx(env, c);
            self.cr_x[idx] = 0.0;
            self.cr_y[idx] = 0.0;
            self.cr_freq[idx] = 0.0;
            self.cr_bandwidth[idx] = 0.0;
            self.cr_power[idx] = 0.0;
            self.cr_throughput[idx] = 0.0;
            self.cr_collisions[idx] = 0;
            self.cr_hop_count[idx] = 0;
            self.cr_sinr[idx] = 0.0;
            self.cr_alive[idx] = 1;

            // Reset CR history
            for h in 0..self.history_length {
                let hist_idx = self.cr_history_idx(env, c, h);
                self.cr_freq_history[hist_idx] = 0.0;
            }
        }
    }

    /// Reset all environments
    pub fn reset_all(&mut self) {
        for env in 0..self.num_envs {
            self.reset_env(env);
        }
    }

    // ========================================================================
    // State Update Functions
    // ========================================================================

    /// Apply jammer action to state
    pub fn apply_jammer_action(
        &mut self,
        env: usize,
        jammer: usize,
        action: &JammerAction,
        config: &ActionConfig,
    ) {
        let idx = self.jammer_idx(env, jammer);

        // Update history (shift old values)
        self.push_jammer_history(env, jammer, self.jammer_freq[idx]);

        // Apply new action
        self.jammer_freq[idx] = action.actual_frequency(config);
        self.jammer_bandwidth[idx] = action.actual_bandwidth(config);
        self.jammer_power[idx] = action.actual_power_dbm(config);
        self.jammer_modulation[idx] = action.modulation_type() as u8;
    }

    /// Apply CR action to state
    pub fn apply_cr_action(
        &mut self,
        env: usize,
        cr: usize,
        action: &CognitiveRadioAction,
        config: &ActionConfig,
    ) {
        let idx = self.cr_idx(env, cr);

        // Track frequency hop
        let old_freq = self.cr_freq[idx];
        let new_freq = action.actual_frequency(config);

        // Update history
        self.push_cr_history(env, cr, old_freq);

        // Increment hop count if frequency changed significantly
        if (new_freq - old_freq).abs() > config.min_hop_distance {
            self.cr_hop_count[idx] += 1;
        }

        // Apply new action
        self.cr_freq[idx] = new_freq;
        self.cr_bandwidth[idx] = action.actual_bandwidth(config);
        self.cr_power[idx] = action.actual_power_dbm(config);
    }

    /// Push new frequency to jammer history
    fn push_jammer_history(&mut self, env: usize, jammer: usize, freq: f32) {
        // Shift history values
        for h in (1..self.history_length).rev() {
            let src = self.jammer_history_idx(env, jammer, h - 1);
            let dst = self.jammer_history_idx(env, jammer, h);
            self.jammer_freq_history[dst] = self.jammer_freq_history[src];
        }
        // Insert new value at front
        let idx = self.jammer_history_idx(env, jammer, 0);
        self.jammer_freq_history[idx] = freq;
    }

    /// Push new frequency to CR history
    fn push_cr_history(&mut self, env: usize, cr: usize, freq: f32) {
        // Shift history values
        for h in (1..self.history_length).rev() {
            let src = self.cr_history_idx(env, cr, h - 1);
            let dst = self.cr_history_idx(env, cr, h);
            self.cr_freq_history[dst] = self.cr_freq_history[src];
        }
        // Insert new value at front
        let idx = self.cr_history_idx(env, cr, 0);
        self.cr_freq_history[idx] = freq;
    }

    /// Update jammer success rate with exponential moving average
    pub fn update_jammer_success(&mut self, env: usize, jammer: usize, success: bool, alpha: f32) {
        let idx = self.jammer_idx(env, jammer);
        let current = self.jammer_success_rate[idx];
        let new_value = if success { 1.0 } else { 0.0 };
        self.jammer_success_rate[idx] = alpha * new_value + (1.0 - alpha) * current;
    }

    /// Update CR SINR and throughput
    pub fn update_cr_sinr(&mut self, env: usize, cr: usize, sinr_db: f32, threshold_db: f32) {
        let idx = self.cr_idx(env, cr);
        self.cr_sinr[idx] = sinr_db;

        // Calculate throughput based on SINR (simplified Shannon capacity)
        if sinr_db > threshold_db {
            let sinr_linear = 10.0_f32.powf(sinr_db / 10.0);
            // Normalized throughput: log2(1 + SINR) / max_capacity
            let capacity = (1.0 + sinr_linear).log2();
            let max_capacity = (1.0 + 10.0_f32.powf(30.0 / 10.0)).log2(); // 30 dB reference
            self.cr_throughput[idx] = (capacity / max_capacity).min(1.0);
            self.cr_alive[idx] = 1;
        } else {
            self.cr_throughput[idx] = 0.0;
            self.cr_collisions[idx] += 1;
            // Mark as jammed if SINR is very low
            if sinr_db < threshold_db - 10.0 {
                self.cr_alive[idx] = 0;
            }
        }
    }

    // ========================================================================
    // Getters
    // ========================================================================

    /// Get jammer position
    pub fn jammer_position(&self, env: usize, jammer: usize) -> (f32, f32) {
        let idx = self.jammer_idx(env, jammer);
        (self.jammer_x[idx], self.jammer_y[idx])
    }

    /// Get jammer modulation type
    pub fn jammer_mod(&self, env: usize, jammer: usize) -> JammerModulation {
        let idx = self.jammer_idx(env, jammer);
        JammerModulation::from(self.jammer_modulation[idx])
    }

    /// Get CR position
    pub fn cr_position(&self, env: usize, cr: usize) -> (f32, f32) {
        let idx = self.cr_idx(env, cr);
        (self.cr_x[idx], self.cr_y[idx])
    }

    /// Check if CR is alive
    pub fn cr_is_alive(&self, env: usize, cr: usize) -> bool {
        let idx = self.cr_idx(env, cr);
        self.cr_alive[idx] != 0
    }

    /// Get CR frequency history
    pub fn cr_freq_history_slice(&self, env: usize, cr: usize) -> Vec<f32> {
        (0..self.history_length)
            .map(|h| {
                let idx = self.cr_history_idx(env, cr, h);
                self.cr_freq_history[idx]
            })
            .collect()
    }

    /// Get jammer frequency history
    pub fn jammer_freq_history_slice(&self, env: usize, jammer: usize) -> Vec<f32> {
        (0..self.history_length)
            .map(|h| {
                let idx = self.jammer_history_idx(env, jammer, h);
                self.jammer_freq_history[idx]
            })
            .collect()
    }

    /// Count alive CRs in an environment
    pub fn count_alive_crs(&self, env: usize) -> usize {
        (0..self.num_crs)
            .filter(|&cr| self.cr_is_alive(env, cr))
            .count()
    }

    /// Get total CR throughput for an environment
    pub fn total_cr_throughput(&self, env: usize) -> f32 {
        (0..self.num_crs)
            .map(|cr| {
                let idx = self.cr_idx(env, cr);
                self.cr_throughput[idx]
            })
            .sum()
    }

    /// Get average jammer success rate for an environment
    pub fn avg_jammer_success(&self, env: usize) -> f32 {
        if self.num_jammers == 0 {
            return 0.0;
        }
        let sum: f32 = (0..self.num_jammers)
            .map(|j| {
                let idx = self.jammer_idx(env, j);
                self.jammer_success_rate[idx]
            })
            .sum();
        sum / self.num_jammers as f32
    }
}

// ============================================================================
// Multi-Agent Actions Container
// ============================================================================

/// Container for all agent actions in a step
#[derive(Clone, Debug)]
pub struct MultiAgentActions {
    /// Jammer actions per environment: [num_envs × num_jammers]
    pub jammer_actions: Vec<JammerAction>,
    /// CR actions per environment: [num_envs × num_crs]
    pub cr_actions: Vec<CognitiveRadioAction>,
    /// Number of environments
    pub num_envs: usize,
    /// Number of jammers per environment
    pub num_jammers: usize,
    /// Number of CRs per environment
    pub num_crs: usize,
}

impl MultiAgentActions {
    /// Create new actions container
    pub fn new(num_envs: usize, num_jammers: usize, num_crs: usize) -> Self {
        Self {
            jammer_actions: vec![JammerAction::default(); num_envs * num_jammers],
            cr_actions: vec![CognitiveRadioAction::default(); num_envs * num_crs],
            num_envs,
            num_jammers,
            num_crs,
        }
    }

    /// Get total action dimension per environment
    #[inline]
    pub fn action_dim(&self) -> usize {
        self.num_jammers * JammerAction::DIM + self.num_crs * CognitiveRadioAction::DIM
    }

    /// Get total flat action size for all environments
    #[inline]
    pub fn total_size(&self) -> usize {
        self.num_envs * self.action_dim()
    }

    /// Get jammer action index
    #[inline]
    pub fn jammer_idx(&self, env: usize, jammer: usize) -> usize {
        env * self.num_jammers + jammer
    }

    /// Get CR action index
    #[inline]
    pub fn cr_idx(&self, env: usize, cr: usize) -> usize {
        env * self.num_crs + cr
    }

    /// Get jammer action
    pub fn jammer_action(&self, env: usize, jammer: usize) -> &JammerAction {
        &self.jammer_actions[self.jammer_idx(env, jammer)]
    }

    /// Get CR action
    pub fn cr_action(&self, env: usize, cr: usize) -> &CognitiveRadioAction {
        &self.cr_actions[self.cr_idx(env, cr)]
    }

    /// Set jammer action
    pub fn set_jammer_action(&mut self, env: usize, jammer: usize, action: JammerAction) {
        let idx = self.jammer_idx(env, jammer);
        self.jammer_actions[idx] = action;
    }

    /// Set CR action
    pub fn set_cr_action(&mut self, env: usize, cr: usize, action: CognitiveRadioAction) {
        let idx = self.cr_idx(env, cr);
        self.cr_actions[idx] = action;
    }

    /// Create from flat action array
    ///
    /// Expected layout per environment:
    /// [J0_freq, J0_bw, J0_power, J0_mod, ..., CR0_freq, CR0_power, CR0_bw, ...]
    pub fn from_flat(flat: &[f32], num_envs: usize, num_jammers: usize, num_crs: usize) -> Self {
        let action_dim = num_jammers * JammerAction::DIM + num_crs * CognitiveRadioAction::DIM;
        let expected_size = num_envs * action_dim;

        assert!(
            flat.len() >= expected_size,
            "Expected {} actions, got {}",
            expected_size,
            flat.len()
        );

        let mut actions = Self::new(num_envs, num_jammers, num_crs);

        for env in 0..num_envs {
            let env_offset = env * action_dim;

            // Parse jammer actions
            for j in 0..num_jammers {
                let j_offset = env_offset + j * JammerAction::DIM;
                let jammer_action = JammerAction::from_flat(&flat[j_offset..]);
                actions.set_jammer_action(env, j, jammer_action);
            }

            // Parse CR actions
            let cr_start = env_offset + num_jammers * JammerAction::DIM;
            for c in 0..num_crs {
                let c_offset = cr_start + c * CognitiveRadioAction::DIM;
                let cr_action = CognitiveRadioAction::from_flat(&flat[c_offset..]);
                actions.set_cr_action(env, c, cr_action);
            }
        }

        actions
    }

    /// Flatten to array
    ///
    /// Output layout: [env0_j0, env0_j1, ..., env0_cr0, env0_cr1, ..., env1_j0, ...]
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.total_size());

        for env in 0..self.num_envs {
            // Jammer actions
            for j in 0..self.num_jammers {
                flat.extend_from_slice(&self.jammer_action(env, j).to_flat());
            }
            // CR actions
            for c in 0..self.num_crs {
                flat.extend_from_slice(&self.cr_action(env, c).to_flat());
            }
        }

        flat
    }
}

// ============================================================================
// Multi-Agent Configuration
// ============================================================================

/// Configuration for multi-agent setup
#[derive(Clone, Debug)]
pub struct MultiAgentConfig {
    /// Number of jammer agents per environment
    pub num_jammers: usize,
    /// Number of cognitive radio agents per environment
    pub num_crs: usize,
    /// History length for frequency tracking
    pub history_length: usize,
    /// Action configuration
    pub action_config: ActionConfig,
    /// SINR threshold for successful communication (dB)
    pub sinr_threshold_db: f32,
    /// Exponential moving average alpha for success rate
    pub success_rate_alpha: f32,
}

impl Default for MultiAgentConfig {
    fn default() -> Self {
        Self {
            num_jammers: 1,
            num_crs: 1,
            history_length: 8,
            action_config: ActionConfig::default(),
            sinr_threshold_db: 10.0,
            success_rate_alpha: 0.1,
        }
    }
}

impl MultiAgentConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of jammers
    pub fn with_jammers(mut self, num: usize) -> Self {
        self.num_jammers = num;
        self
    }

    /// Set number of CRs
    pub fn with_crs(mut self, num: usize) -> Self {
        self.num_crs = num;
        self
    }

    /// Set history length
    pub fn with_history_length(mut self, len: usize) -> Self {
        self.history_length = len;
        self
    }

    /// Set action configuration
    pub fn with_action_config(mut self, config: ActionConfig) -> Self {
        self.action_config = config;
        self
    }

    /// Set SINR threshold
    pub fn with_sinr_threshold(mut self, threshold_db: f32) -> Self {
        self.sinr_threshold_db = threshold_db;
        self
    }

    /// Total agents per environment
    #[inline]
    pub fn total_agents(&self) -> usize {
        self.num_jammers + self.num_crs
    }

    /// Action dimension per environment
    #[inline]
    pub fn action_dim(&self) -> usize {
        self.num_jammers * JammerAction::DIM + self.num_crs * CognitiveRadioAction::DIM
    }

    /// Create multi-agent state
    pub fn create_state(&self, num_envs: usize) -> MultiAgentState {
        MultiAgentState::new(num_envs, self.num_jammers, self.num_crs, self.history_length)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_agent_state_indexing() {
        let state = MultiAgentState::new(8, 2, 3, 4);

        // Test jammer indexing
        assert_eq!(state.jammer_idx(0, 0), 0);
        assert_eq!(state.jammer_idx(0, 1), 1);
        assert_eq!(state.jammer_idx(1, 0), 2);
        assert_eq!(state.jammer_idx(7, 1), 15);

        // Test CR indexing
        assert_eq!(state.cr_idx(0, 0), 0);
        assert_eq!(state.cr_idx(0, 2), 2);
        assert_eq!(state.cr_idx(1, 0), 3);
        assert_eq!(state.cr_idx(7, 2), 23);

        // Test history indexing
        assert_eq!(state.jammer_history_idx(0, 0, 0), 0);
        assert_eq!(state.jammer_history_idx(0, 0, 3), 3);
        assert_eq!(state.jammer_history_idx(0, 1, 0), 4);
        assert_eq!(state.jammer_history_idx(1, 0, 0), 8);
    }

    #[test]
    fn test_multi_agent_state_reset() {
        let mut state = MultiAgentState::new(8, 2, 3, 4);

        // Modify state
        let idx = state.jammer_idx(0, 0);
        state.jammer_freq[idx] = 2.4e9;
        state.jammer_power[idx] = 20.0;

        let cr_idx = state.cr_idx(0, 0);
        state.cr_freq[cr_idx] = 2.45e9;
        state.cr_collisions[cr_idx] = 5;

        // Reset
        state.reset_env(0);

        // Verify reset
        assert!((state.jammer_freq[idx]).abs() < 1e-6);
        assert!((state.jammer_power[idx]).abs() < 1e-6);
        assert!((state.cr_freq[cr_idx]).abs() < 1e-6);
        assert_eq!(state.cr_collisions[cr_idx], 0);
    }

    #[test]
    fn test_apply_jammer_action() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);
        let config = ActionConfig::default();

        let action = JammerAction::new(0.5, 0.3, 0.8, 0.1);
        state.apply_jammer_action(0, 0, &action, &config);

        let idx = state.jammer_idx(0, 0);
        assert!((state.jammer_freq[idx] - action.actual_frequency(&config)).abs() < 1.0);
        assert!((state.jammer_bandwidth[idx] - action.actual_bandwidth(&config)).abs() < 1.0);
        assert!((state.jammer_power[idx] - action.actual_power_dbm(&config)).abs() < 0.1);
    }

    #[test]
    fn test_apply_cr_action_with_hop() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);
        let config = ActionConfig::default();

        // Set initial frequency
        let idx = state.cr_idx(0, 0);
        state.cr_freq[idx] = 2.4e9;

        // Apply action with significant frequency change
        let action = CognitiveRadioAction::new(0.9, 0.5, 0.5); // High frequency
        state.apply_cr_action(0, 0, &action, &config);

        // Should increment hop count
        assert_eq!(state.cr_hop_count[idx], 1);
    }

    #[test]
    fn test_cr_sinr_update() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);
        let threshold = 10.0;

        // Good SINR
        state.update_cr_sinr(0, 0, 20.0, threshold);
        let idx = state.cr_idx(0, 0);
        assert!(state.cr_throughput[idx] > 0.0);
        assert_eq!(state.cr_alive[idx], 1);

        // Bad SINR
        state.update_cr_sinr(0, 0, 5.0, threshold);
        assert!((state.cr_throughput[idx]).abs() < 1e-6);
        assert_eq!(state.cr_collisions[idx], 1);
    }

    #[test]
    fn test_history_tracking() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);
        let config = ActionConfig::default();

        // Apply multiple actions
        for i in 0..6 {
            let action = CognitiveRadioAction::new(i as f32 / 10.0, 0.5, 0.5);
            state.apply_cr_action(0, 0, &action, &config);
        }

        // Check history
        let history = state.cr_freq_history_slice(0, 0);
        assert_eq!(history.len(), 4);
    }

    #[test]
    fn test_multi_agent_actions_from_flat() {
        let num_envs = 2;
        let num_jammers = 1;
        let num_crs = 1;

        // [J0_f, J0_b, J0_p, J0_m, CR0_f, CR0_p, CR0_b] × 2 envs
        let flat = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, // env 0
            0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, // env 1
        ];

        let actions = MultiAgentActions::from_flat(&flat, num_envs, num_jammers, num_crs);

        // Check env 0
        assert!((actions.jammer_action(0, 0).frequency - 0.1).abs() < 1e-6);
        assert!((actions.cr_action(0, 0).frequency - 0.5).abs() < 1e-6);

        // Check env 1
        assert!((actions.jammer_action(1, 0).frequency - 0.8).abs() < 1e-6);
        assert!((actions.cr_action(1, 0).frequency - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_multi_agent_actions_roundtrip() {
        let num_envs = 2;
        let num_jammers = 2;
        let num_crs = 3;

        let mut actions = MultiAgentActions::new(num_envs, num_jammers, num_crs);

        // Set some actions
        actions.set_jammer_action(0, 0, JammerAction::new(0.1, 0.2, 0.3, 0.4));
        actions.set_jammer_action(1, 1, JammerAction::new(0.5, 0.6, 0.7, 0.8));
        actions.set_cr_action(0, 2, CognitiveRadioAction::new(0.9, 0.8, 0.7));

        // Flatten and recover
        let flat = actions.to_flat();
        let recovered = MultiAgentActions::from_flat(&flat, num_envs, num_jammers, num_crs);

        // Verify
        assert!(
            (recovered.jammer_action(0, 0).frequency - 0.1).abs() < 1e-6
        );
        assert!(
            (recovered.jammer_action(1, 1).frequency - 0.5).abs() < 1e-6
        );
        assert!(
            (recovered.cr_action(0, 2).frequency - 0.9).abs() < 1e-6
        );
    }

    #[test]
    fn test_action_dim_calculation() {
        let config = MultiAgentConfig::new()
            .with_jammers(2)
            .with_crs(3);

        // 2 × 4 + 3 × 3 = 8 + 9 = 17
        assert_eq!(config.action_dim(), 17);
    }

    #[test]
    fn test_multi_agent_config_create_state() {
        let config = MultiAgentConfig::new()
            .with_jammers(2)
            .with_crs(3)
            .with_history_length(10);

        let state = config.create_state(8);

        assert_eq!(state.num_envs, 8);
        assert_eq!(state.num_jammers, 2);
        assert_eq!(state.num_crs, 3);
        assert_eq!(state.history_length, 10);
    }

    #[test]
    fn test_aggregate_functions() {
        let mut state = MultiAgentState::new(1, 2, 3, 4);

        // Set CR throughputs
        for c in 0..3 {
            let idx = state.cr_idx(0, c);
            state.cr_throughput[idx] = (c + 1) as f32 * 0.1;
        }

        let total = state.total_cr_throughput(0);
        assert!((total - 0.6).abs() < 1e-6); // 0.1 + 0.2 + 0.3

        // Set jammer success rates
        for j in 0..2 {
            let idx = state.jammer_idx(0, j);
            state.jammer_success_rate[idx] = (j + 1) as f32 * 0.2;
        }

        let avg = state.avg_jammer_success(0);
        assert!((avg - 0.3).abs() < 1e-6); // (0.2 + 0.4) / 2
    }
}
