//! RF World State Structures
//!
//! Struct-of-Arrays (SoA) layout for cache-efficient SIMD processing.
//! All state arrays are organized as [num_envs × feature_size] for
//! optimal memory access patterns when processing 8 environments in parallel.

use crate::config::RFWorldConfig;

/// State of the RF World environment
///
/// Uses Struct-of-Arrays (SoA) layout where each field is a flat array
/// containing data for all environments, enabling efficient SIMD processing.
#[derive(Clone)]
pub struct RFWorldState {
    // ========================================================================
    // Spectrum State
    // ========================================================================
    /// Power Spectral Density: [num_envs × num_freq_bins]
    /// Linear power values (not dB)
    pub psd: Vec<f32>,

    /// Noise floor per environment: [num_envs]
    /// Linear power value
    pub noise_floor: Vec<f32>,

    // ========================================================================
    // Episode State
    // ========================================================================
    /// Current step count per environment: [num_envs]
    pub step_count: Vec<u32>,

    /// Terminal flags (1 = terminal, 0 = not): [num_envs]
    pub terminals: Vec<u8>,

    /// Truncation flags (1 = truncated, 0 = not): [num_envs]
    pub truncations: Vec<u8>,

    /// Episode return accumulators: [num_envs]
    pub episode_returns: Vec<f32>,

    // ========================================================================
    // Configuration Cache
    // ========================================================================
    /// Number of environments
    num_envs: usize,

    /// Number of frequency bins
    num_freq_bins: usize,
}

impl RFWorldState {
    /// Create a new state structure based on configuration
    pub fn new(config: &RFWorldConfig) -> Self {
        let num_envs = config.num_envs;
        let num_freq_bins = config.num_freq_bins;
        let psd_size = num_envs * num_freq_bins;

        // Convert noise floor from dBm/Hz to linear, accounting for bandwidth per bin
        let noise_floor_linear = db_to_linear_power(
            config.effective_noise_floor() + 10.0 * (config.freq_resolution()).log10(),
        );

        Self {
            // Spectrum state
            psd: vec![noise_floor_linear; psd_size],
            noise_floor: vec![noise_floor_linear; num_envs],

            // Episode state
            step_count: vec![0; num_envs],
            terminals: vec![0; num_envs],
            truncations: vec![0; num_envs],
            episode_returns: vec![0.0; num_envs],

            // Cache
            num_envs,
            num_freq_bins,
        }
    }

    /// Reset a single environment to initial state
    pub fn reset_env(&mut self, env: usize) {
        debug_assert!(env < self.num_envs, "Environment index out of bounds");

        // Reset PSD to noise floor
        let psd_start = self.psd_idx(env, 0);
        let psd_end = psd_start + self.num_freq_bins;
        let noise = self.noise_floor[env];
        self.psd[psd_start..psd_end].fill(noise);

        // Reset episode state
        self.step_count[env] = 0;
        self.terminals[env] = 0;
        self.truncations[env] = 0;
        self.episode_returns[env] = 0.0;
    }

    /// Reset all environments to initial state
    pub fn reset_all(&mut self) {
        for env in 0..self.num_envs {
            self.reset_env(env);
        }
    }

    /// Reset all environments with a new seed
    ///
    /// This is useful for ensuring reproducibility across training runs.
    pub fn reset_all_with_seed(&mut self, _seed: u64) {
        // For Phase 1, just reset without seed-dependent initialization
        // Full entity spawning will be added in Phase 2
        self.reset_all();
    }

    // ========================================================================
    // Index Functions
    // ========================================================================

    /// Get the index into PSD array for a specific (env, bin) pair
    ///
    /// Layout: env-major, i.e., all bins for env 0, then env 1, etc.
    #[inline]
    pub fn psd_idx(&self, env: usize, bin: usize) -> usize {
        debug_assert!(env < self.num_envs);
        debug_assert!(bin < self.num_freq_bins);
        env * self.num_freq_bins + bin
    }

    /// Get a slice of the PSD for a single environment
    #[inline]
    pub fn psd_slice(&self, env: usize) -> &[f32] {
        let start = self.psd_idx(env, 0);
        &self.psd[start..start + self.num_freq_bins]
    }

    /// Get a mutable slice of the PSD for a single environment
    #[inline]
    pub fn psd_slice_mut(&mut self, env: usize) -> &mut [f32] {
        let start = self.psd_idx(env, 0);
        &mut self.psd[start..start + self.num_freq_bins]
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get number of environments
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get number of frequency bins
    #[inline]
    pub fn num_freq_bins(&self) -> usize {
        self.num_freq_bins
    }

    /// Check if an environment is terminal
    #[inline]
    pub fn is_terminal(&self, env: usize) -> bool {
        self.terminals[env] != 0
    }

    /// Check if an environment is truncated
    #[inline]
    pub fn is_truncated(&self, env: usize) -> bool {
        self.truncations[env] != 0
    }

    /// Check if an environment is done (terminal or truncated)
    #[inline]
    pub fn is_done(&self, env: usize) -> bool {
        self.is_terminal(env) || self.is_truncated(env)
    }

    /// Mark an environment as terminal
    #[inline]
    pub fn set_terminal(&mut self, env: usize, terminal: bool) {
        self.terminals[env] = if terminal { 1 } else { 0 };
    }

    /// Mark an environment as truncated
    #[inline]
    pub fn set_truncated(&mut self, env: usize, truncated: bool) {
        self.truncations[env] = if truncated { 1 } else { 0 };
    }

    /// Increment step count and return new value
    #[inline]
    pub fn increment_step(&mut self, env: usize) -> u32 {
        self.step_count[env] += 1;
        self.step_count[env]
    }

    /// Add reward to episode return
    #[inline]
    pub fn add_reward(&mut self, env: usize, reward: f32) {
        self.episode_returns[env] += reward;
    }

    // ========================================================================
    // PSD Operations
    // ========================================================================

    /// Clear PSD to noise floor for all environments
    pub fn clear_psd_all(&mut self) {
        for env in 0..self.num_envs {
            let noise = self.noise_floor[env];
            let psd_start = self.psd_idx(env, 0);
            let psd_end = psd_start + self.num_freq_bins;
            self.psd[psd_start..psd_end].fill(noise);
        }
    }

    /// Add power to a specific frequency bin
    ///
    /// Power is additive (interference accumulates)
    #[inline]
    pub fn add_psd_power(&mut self, env: usize, bin: usize, power: f32) {
        let idx = self.psd_idx(env, bin);
        self.psd[idx] += power;
    }

    /// Set power at a specific frequency bin
    #[inline]
    pub fn set_psd_power(&mut self, env: usize, bin: usize, power: f32) {
        let idx = self.psd_idx(env, bin);
        self.psd[idx] = power;
    }

    /// Get power at a specific frequency bin
    #[inline]
    pub fn get_psd_power(&self, env: usize, bin: usize) -> f32 {
        let idx = self.psd_idx(env, bin);
        self.psd[idx]
    }

    /// Get the total power in the PSD for an environment
    pub fn total_psd_power(&self, env: usize) -> f32 {
        self.psd_slice(env).iter().sum()
    }

    /// Get the peak power in the PSD for an environment
    pub fn peak_psd_power(&self, env: usize) -> f32 {
        self.psd_slice(env)
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert dBm to linear power (mW)
#[inline]
fn db_to_linear_power(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

/// Convert linear power to dBm
#[allow(dead_code)]
#[inline]
fn linear_to_db_power(linear: f32) -> f32 {
    10.0 * linear.log10()
}

// ============================================================================
// Public Index Function (for external use without state instance)
// ============================================================================

/// Calculate frequency bin index: env-major ordering
///
/// This is the standalone version for use in contexts where you don't
/// have a state instance.
#[inline]
pub fn freq_bin_idx(env: usize, bin: usize, num_bins: usize) -> usize {
    env * num_bins + bin
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
            .build()
    }

    #[test]
    fn test_state_creation() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        assert_eq!(state.num_envs(), 8);
        assert_eq!(state.num_freq_bins(), 256);
        assert_eq!(state.psd.len(), 8 * 256);
        assert_eq!(state.step_count.len(), 8);
    }

    #[test]
    fn test_psd_indexing() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        // Check index calculation
        assert_eq!(state.psd_idx(0, 0), 0);
        assert_eq!(state.psd_idx(0, 255), 255);
        assert_eq!(state.psd_idx(1, 0), 256);
        assert_eq!(state.psd_idx(7, 255), 7 * 256 + 255);
    }

    #[test]
    fn test_reset_env() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Modify state
        state.step_count[0] = 100;
        state.terminals[0] = 1;
        state.episode_returns[0] = 50.0;

        // Reset
        state.reset_env(0);

        // Verify reset
        assert_eq!(state.step_count[0], 0);
        assert_eq!(state.terminals[0], 0);
        assert_eq!(state.episode_returns[0], 0.0);
    }

    #[test]
    fn test_psd_operations() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        let initial_power = state.get_psd_power(0, 100);

        // Add power
        state.add_psd_power(0, 100, 1.0);
        assert!((state.get_psd_power(0, 100) - initial_power - 1.0).abs() < 1e-6);

        // Set power
        state.set_psd_power(0, 100, 5.0);
        assert!((state.get_psd_power(0, 100) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_terminal_flags() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        assert!(!state.is_terminal(0));
        assert!(!state.is_truncated(0));
        assert!(!state.is_done(0));

        state.set_terminal(0, true);
        assert!(state.is_terminal(0));
        assert!(state.is_done(0));

        state.set_terminal(0, false);
        state.set_truncated(0, true);
        assert!(state.is_truncated(0));
        assert!(state.is_done(0));
    }

    #[test]
    fn test_step_count() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        assert_eq!(state.step_count[0], 0);

        let new_count = state.increment_step(0);
        assert_eq!(new_count, 1);
        assert_eq!(state.step_count[0], 1);
    }

    #[test]
    fn test_episode_returns() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        state.add_reward(0, 1.0);
        state.add_reward(0, 2.0);
        assert!((state.episode_returns[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_psd_slice() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        let slice = state.psd_slice(0);
        assert_eq!(slice.len(), 256);
    }

    #[test]
    fn test_standalone_index_function() {
        assert_eq!(freq_bin_idx(0, 0, 256), 0);
        assert_eq!(freq_bin_idx(1, 0, 256), 256);
        assert_eq!(freq_bin_idx(2, 100, 256), 2 * 256 + 100);
    }
}
