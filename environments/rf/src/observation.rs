//! Observation Building
//!
//! Constructs observations for jammer and cognitive radio agents
//! from environment state and agent state.

use crate::agents::{AgentType, MultiAgentState};
use crate::state::RFWorldState;

// ============================================================================
// Observation Configuration
// ============================================================================

/// Configuration for observation building
#[derive(Clone, Debug)]
pub struct ObservationConfig {
    // PSD observation options
    /// Whether to include PSD in observations
    pub include_psd: bool,
    /// Number of frequency bins in compressed PSD (0 = use raw)
    pub num_freq_bins: usize,
    /// PSD compression mode: "avg", "max", or "subsample"
    pub psd_compression: PsdCompression,
    /// Whether to normalize PSD values
    pub normalize_psd: bool,
    /// Minimum PSD value for normalization (dBm)
    pub psd_min_db: f32,
    /// Maximum PSD value for normalization (dBm)
    pub psd_max_db: f32,

    // Agent state options
    /// Whether to include own agent state
    pub include_agent_state: bool,
    /// Whether to include team state (all agents of same type)
    pub include_team_state: bool,
    /// Whether to include opponent state
    pub include_opponent_state: bool,
    /// Whether to include frequency history
    pub include_history: bool,

    // Normalization bounds
    /// Maximum frequency for normalization (Hz)
    pub freq_max: f32,
    /// Maximum power for normalization (dBm)
    pub power_max: f32,
    /// Maximum bandwidth for normalization (Hz)
    pub bandwidth_max: f32,
    /// World size for position normalization (m)
    pub world_size: f32,

    // History options
    /// Length of frequency history to include
    pub history_length: usize,
}

/// PSD compression modes
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PsdCompression {
    /// Average pooling
    Average,
    /// Max pooling
    Max,
    /// Subsampling (take every Nth bin)
    Subsample,
}

impl Default for ObservationConfig {
    fn default() -> Self {
        Self {
            include_psd: true,
            num_freq_bins: 64,
            psd_compression: PsdCompression::Max,
            normalize_psd: true,
            psd_min_db: -120.0,
            psd_max_db: 0.0,

            include_agent_state: true,
            include_team_state: true,
            include_opponent_state: false,
            include_history: true,

            freq_max: 6e9,
            power_max: 40.0,
            bandwidth_max: 100e6,
            world_size: 1000.0,

            history_length: 8,
        }
    }
}

impl ObservationConfig {
    /// Create new observation config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set PSD compression target size
    pub fn with_psd_bins(mut self, bins: usize) -> Self {
        self.num_freq_bins = bins;
        self
    }

    /// Disable PSD in observations
    pub fn without_psd(mut self) -> Self {
        self.include_psd = false;
        self
    }

    /// Include opponent state in observations
    pub fn with_opponent_state(mut self) -> Self {
        self.include_opponent_state = true;
        self
    }

    /// Set history length
    pub fn with_history_length(mut self, len: usize) -> Self {
        self.history_length = len;
        self
    }

    /// Calculate observation size for a jammer
    pub fn jammer_observation_size(&self, num_jammers: usize, num_crs: usize) -> usize {
        let mut size = 0;

        // PSD
        if self.include_psd {
            size += self.num_freq_bins;
        }

        // Own state: x, y, freq, bandwidth, power, modulation, target, success_rate
        if self.include_agent_state {
            size += 8;
        }

        // Team state: other jammers
        if self.include_team_state && num_jammers > 1 {
            size += (num_jammers - 1) * 6; // x, y, freq, bw, power, mod
        }

        // Opponent state: CRs
        if self.include_opponent_state {
            size += num_crs * 5; // x, y, freq, bw, power
        }

        // History
        if self.include_history {
            size += self.history_length;
        }

        size
    }

    /// Calculate observation size for a CR
    pub fn cr_observation_size(&self, num_jammers: usize, num_crs: usize) -> usize {
        let mut size = 0;

        // PSD
        if self.include_psd {
            size += self.num_freq_bins;
        }

        // Own state: x, y, freq, bandwidth, power, throughput, sinr, alive, collisions, hops
        if self.include_agent_state {
            size += 10;
        }

        // Team state: other CRs
        if self.include_team_state && num_crs > 1 {
            size += (num_crs - 1) * 5; // x, y, freq, bw, power
        }

        // Opponent state: jammers
        if self.include_opponent_state {
            size += num_jammers * 6; // x, y, freq, bw, power, mod
        }

        // History
        if self.include_history {
            size += self.history_length;
        }

        size
    }
}

// ============================================================================
// PSD Compression
// ============================================================================

/// Compress PSD to target number of bins
pub fn compress_psd(psd: &[f32], target_size: usize, mode: PsdCompression) -> Vec<f32> {
    if psd.len() == target_size {
        return psd.to_vec();
    }

    if target_size >= psd.len() || target_size == 0 {
        return psd.to_vec();
    }

    let mut result = Vec::with_capacity(target_size);
    let bins_per_output = psd.len() / target_size;
    let remainder = psd.len() % target_size;

    match mode {
        PsdCompression::Average => {
            let mut start = 0;
            for i in 0..target_size {
                let extra = if i < remainder { 1 } else { 0 };
                let end = start + bins_per_output + extra;
                let sum: f32 = psd[start..end].iter().sum();
                result.push(sum / (end - start) as f32);
                start = end;
            }
        }
        PsdCompression::Max => {
            let mut start = 0;
            for i in 0..target_size {
                let extra = if i < remainder { 1 } else { 0 };
                let end = start + bins_per_output + extra;
                let max = psd[start..end]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                result.push(max);
                start = end;
            }
        }
        PsdCompression::Subsample => {
            let step = psd.len() / target_size;
            for i in 0..target_size {
                result.push(psd[i * step]);
            }
        }
    }

    result
}

/// Normalize PSD values to [0, 1] range
pub fn normalize_psd(psd: &[f32], min_db: f32, max_db: f32) -> Vec<f32> {
    let db_range = max_db - min_db;
    psd.iter()
        .map(|&power| {
            // Convert linear power to dB
            let power_db = 10.0 * power.max(1e-20).log10();
            // Normalize to [0, 1]
            ((power_db - min_db) / db_range).clamp(0.0, 1.0)
        })
        .collect()
}

// ============================================================================
// Multi-Agent Observations
// ============================================================================

/// Container for all agent observations
#[derive(Clone, Debug)]
pub struct MultiAgentObservations {
    /// Jammer observations: [num_envs × num_jammers × obs_size]
    pub jammer_obs: Vec<f32>,
    /// CR observations: [num_envs × num_crs × obs_size]
    pub cr_obs: Vec<f32>,
    /// Observation size for jammers
    pub jammer_obs_size: usize,
    /// Observation size for CRs
    pub cr_obs_size: usize,
    /// Number of environments
    pub num_envs: usize,
    /// Number of jammers
    pub num_jammers: usize,
    /// Number of CRs
    pub num_crs: usize,
}

impl MultiAgentObservations {
    /// Create new observations container
    pub fn new(
        num_envs: usize,
        num_jammers: usize,
        num_crs: usize,
        jammer_obs_size: usize,
        cr_obs_size: usize,
    ) -> Self {
        Self {
            jammer_obs: vec![0.0; num_envs * num_jammers * jammer_obs_size],
            cr_obs: vec![0.0; num_envs * num_crs * cr_obs_size],
            jammer_obs_size,
            cr_obs_size,
            num_envs,
            num_jammers,
            num_crs,
        }
    }

    /// Get start index for jammer observation
    #[inline]
    pub fn jammer_start(&self, env: usize, jammer: usize) -> usize {
        (env * self.num_jammers + jammer) * self.jammer_obs_size
    }

    /// Get start index for CR observation
    #[inline]
    pub fn cr_start(&self, env: usize, cr: usize) -> usize {
        (env * self.num_crs + cr) * self.cr_obs_size
    }

    /// Get jammer observation slice
    pub fn jammer_observation(&self, env: usize, jammer: usize) -> &[f32] {
        let start = self.jammer_start(env, jammer);
        &self.jammer_obs[start..start + self.jammer_obs_size]
    }

    /// Get mutable jammer observation slice
    pub fn jammer_observation_mut(&mut self, env: usize, jammer: usize) -> &mut [f32] {
        let start = self.jammer_start(env, jammer);
        &mut self.jammer_obs[start..start + self.jammer_obs_size]
    }

    /// Get CR observation slice
    pub fn cr_observation(&self, env: usize, cr: usize) -> &[f32] {
        let start = self.cr_start(env, cr);
        &self.cr_obs[start..start + self.cr_obs_size]
    }

    /// Get mutable CR observation slice
    pub fn cr_observation_mut(&mut self, env: usize, cr: usize) -> &mut [f32] {
        let start = self.cr_start(env, cr);
        &mut self.cr_obs[start..start + self.cr_obs_size]
    }

    /// Build all observations
    pub fn build(
        rf_state: &RFWorldState,
        agent_state: &MultiAgentState,
        config: &ObservationConfig,
    ) -> Self {
        let jammer_obs_size = config.jammer_observation_size(agent_state.num_jammers, agent_state.num_crs);
        let cr_obs_size = config.cr_observation_size(agent_state.num_jammers, agent_state.num_crs);

        let mut obs = Self::new(
            agent_state.num_envs,
            agent_state.num_jammers,
            agent_state.num_crs,
            jammer_obs_size,
            cr_obs_size,
        );

        for env in 0..agent_state.num_envs {
            // Build jammer observations
            for j in 0..agent_state.num_jammers {
                let obs_slice = obs.jammer_observation_mut(env, j);
                build_jammer_observation(rf_state, agent_state, env, j, config, obs_slice);
            }

            // Build CR observations
            for c in 0..agent_state.num_crs {
                let obs_slice = obs.cr_observation_mut(env, c);
                build_cr_observation(rf_state, agent_state, env, c, config, obs_slice);
            }
        }

        obs
    }

    /// Flatten all observations for training
    ///
    /// Layout: [all jammer obs, all CR obs]
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.jammer_obs.len() + self.cr_obs.len());
        flat.extend_from_slice(&self.jammer_obs);
        flat.extend_from_slice(&self.cr_obs);
        flat
    }

    /// Get observations by agent type
    pub fn by_type(&self, agent_type: AgentType) -> &[f32] {
        match agent_type {
            AgentType::Jammer => &self.jammer_obs,
            AgentType::CognitiveRadio => &self.cr_obs,
        }
    }

    /// Get flat observations for a specific jammer across all environments.
    ///
    /// Returns [n_envs × jammer_obs_size] in row-major order.
    /// This is used when training a single jammer agent across all environments.
    pub fn jammer_observations_flat(&self, jammer_idx: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.num_envs * self.jammer_obs_size);

        for env in 0..self.num_envs {
            let obs = self.jammer_observation(env, jammer_idx);
            result.extend_from_slice(obs);
        }

        result
    }

    /// Get flat observations for a specific CR across all environments.
    ///
    /// Returns [n_envs × cr_obs_size] in row-major order.
    /// This is used when training a single CR agent across all environments.
    pub fn cr_observations_flat(&self, cr_idx: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.num_envs * self.cr_obs_size);

        for env in 0..self.num_envs {
            let obs = self.cr_observation(env, cr_idx);
            result.extend_from_slice(obs);
        }

        result
    }

    /// Get all jammer observations flat for environment-major layout.
    ///
    /// Returns [n_envs × n_jammers × jammer_obs_size].
    pub fn all_jammer_observations_flat(&self) -> &[f32] {
        &self.jammer_obs
    }

    /// Get all CR observations flat for environment-major layout.
    ///
    /// Returns [n_envs × n_crs × cr_obs_size].
    pub fn all_cr_observations_flat(&self) -> &[f32] {
        &self.cr_obs
    }
}

// ============================================================================
// Observation Building Functions
// ============================================================================

/// Build observation for a single jammer
fn build_jammer_observation(
    rf_state: &RFWorldState,
    agent_state: &MultiAgentState,
    env: usize,
    jammer: usize,
    config: &ObservationConfig,
    output: &mut [f32],
) {
    let mut idx = 0;

    // PSD
    if config.include_psd {
        let psd_slice = rf_state.psd_slice(env);
        let compressed = compress_psd(psd_slice, config.num_freq_bins, config.psd_compression);
        let normalized = if config.normalize_psd {
            normalize_psd(&compressed, config.psd_min_db, config.psd_max_db)
        } else {
            compressed
        };
        for &val in &normalized {
            output[idx] = val;
            idx += 1;
        }
    }

    // Own state
    if config.include_agent_state {
        let j_idx = agent_state.jammer_idx(env, jammer);

        output[idx] = agent_state.jammer_x[j_idx] / config.world_size;
        idx += 1;
        output[idx] = agent_state.jammer_y[j_idx] / config.world_size;
        idx += 1;
        output[idx] = agent_state.jammer_freq[j_idx] / config.freq_max;
        idx += 1;
        output[idx] = agent_state.jammer_bandwidth[j_idx] / config.bandwidth_max;
        idx += 1;
        output[idx] = (agent_state.jammer_power[j_idx] / config.power_max).clamp(0.0, 1.0);
        idx += 1;
        output[idx] = agent_state.jammer_modulation[j_idx] as f32 / 3.0;
        idx += 1;
        output[idx] = if agent_state.jammer_target[j_idx] >= 0 {
            agent_state.jammer_target[j_idx] as f32 / agent_state.num_crs as f32
        } else {
            0.0
        };
        idx += 1;
        output[idx] = agent_state.jammer_success_rate[j_idx];
        idx += 1;
    }

    // Team state (other jammers)
    if config.include_team_state && agent_state.num_jammers > 1 {
        for other_j in 0..agent_state.num_jammers {
            if other_j != jammer {
                let j_idx = agent_state.jammer_idx(env, other_j);
                output[idx] = agent_state.jammer_x[j_idx] / config.world_size;
                idx += 1;
                output[idx] = agent_state.jammer_y[j_idx] / config.world_size;
                idx += 1;
                output[idx] = agent_state.jammer_freq[j_idx] / config.freq_max;
                idx += 1;
                output[idx] = agent_state.jammer_bandwidth[j_idx] / config.bandwidth_max;
                idx += 1;
                output[idx] = (agent_state.jammer_power[j_idx] / config.power_max).clamp(0.0, 1.0);
                idx += 1;
                output[idx] = agent_state.jammer_modulation[j_idx] as f32 / 3.0;
                idx += 1;
            }
        }
    }

    // Opponent state (CRs)
    if config.include_opponent_state {
        for c in 0..agent_state.num_crs {
            let c_idx = agent_state.cr_idx(env, c);
            output[idx] = agent_state.cr_x[c_idx] / config.world_size;
            idx += 1;
            output[idx] = agent_state.cr_y[c_idx] / config.world_size;
            idx += 1;
            output[idx] = agent_state.cr_freq[c_idx] / config.freq_max;
            idx += 1;
            output[idx] = agent_state.cr_bandwidth[c_idx] / config.bandwidth_max;
            idx += 1;
            output[idx] = (agent_state.cr_power[c_idx] / config.power_max).clamp(0.0, 1.0);
            idx += 1;
        }
    }

    // History
    if config.include_history {
        let history = agent_state.jammer_freq_history_slice(env, jammer);
        for h in 0..config.history_length.min(history.len()) {
            output[idx] = history[h] / config.freq_max;
            idx += 1;
        }
        // Pad with zeros if history is shorter
        for _ in history.len()..config.history_length {
            output[idx] = 0.0;
            idx += 1;
        }
    }
}

/// Build observation for a single CR
fn build_cr_observation(
    rf_state: &RFWorldState,
    agent_state: &MultiAgentState,
    env: usize,
    cr: usize,
    config: &ObservationConfig,
    output: &mut [f32],
) {
    let mut idx = 0;

    // PSD
    if config.include_psd {
        let psd_slice = rf_state.psd_slice(env);
        let compressed = compress_psd(psd_slice, config.num_freq_bins, config.psd_compression);
        let normalized = if config.normalize_psd {
            normalize_psd(&compressed, config.psd_min_db, config.psd_max_db)
        } else {
            compressed
        };
        for &val in &normalized {
            output[idx] = val;
            idx += 1;
        }
    }

    // Own state
    if config.include_agent_state {
        let c_idx = agent_state.cr_idx(env, cr);

        output[idx] = agent_state.cr_x[c_idx] / config.world_size;
        idx += 1;
        output[idx] = agent_state.cr_y[c_idx] / config.world_size;
        idx += 1;
        output[idx] = agent_state.cr_freq[c_idx] / config.freq_max;
        idx += 1;
        output[idx] = agent_state.cr_bandwidth[c_idx] / config.bandwidth_max;
        idx += 1;
        output[idx] = (agent_state.cr_power[c_idx] / config.power_max).clamp(0.0, 1.0);
        idx += 1;
        output[idx] = agent_state.cr_throughput[c_idx];
        idx += 1;
        // Normalize SINR: assume range [-20, 40] dB -> [0, 1]
        output[idx] = ((agent_state.cr_sinr[c_idx] + 20.0) / 60.0).clamp(0.0, 1.0);
        idx += 1;
        output[idx] = agent_state.cr_alive[c_idx] as f32;
        idx += 1;
        // Normalize collisions: cap at 100
        output[idx] = (agent_state.cr_collisions[c_idx] as f32 / 100.0).min(1.0);
        idx += 1;
        // Normalize hop count: cap at 100
        output[idx] = (agent_state.cr_hop_count[c_idx] as f32 / 100.0).min(1.0);
        idx += 1;
    }

    // Team state (other CRs)
    if config.include_team_state && agent_state.num_crs > 1 {
        for other_c in 0..agent_state.num_crs {
            if other_c != cr {
                let c_idx = agent_state.cr_idx(env, other_c);
                output[idx] = agent_state.cr_x[c_idx] / config.world_size;
                idx += 1;
                output[idx] = agent_state.cr_y[c_idx] / config.world_size;
                idx += 1;
                output[idx] = agent_state.cr_freq[c_idx] / config.freq_max;
                idx += 1;
                output[idx] = agent_state.cr_bandwidth[c_idx] / config.bandwidth_max;
                idx += 1;
                output[idx] = (agent_state.cr_power[c_idx] / config.power_max).clamp(0.0, 1.0);
                idx += 1;
            }
        }
    }

    // Opponent state (jammers)
    if config.include_opponent_state {
        for j in 0..agent_state.num_jammers {
            let j_idx = agent_state.jammer_idx(env, j);
            output[idx] = agent_state.jammer_x[j_idx] / config.world_size;
            idx += 1;
            output[idx] = agent_state.jammer_y[j_idx] / config.world_size;
            idx += 1;
            output[idx] = agent_state.jammer_freq[j_idx] / config.freq_max;
            idx += 1;
            output[idx] = agent_state.jammer_bandwidth[j_idx] / config.bandwidth_max;
            idx += 1;
            output[idx] = (agent_state.jammer_power[j_idx] / config.power_max).clamp(0.0, 1.0);
            idx += 1;
            output[idx] = agent_state.jammer_modulation[j_idx] as f32 / 3.0;
            idx += 1;
        }
    }

    // History
    if config.include_history {
        let history = agent_state.cr_freq_history_slice(env, cr);
        for h in 0..config.history_length.min(history.len()) {
            output[idx] = history[h] / config.freq_max;
            idx += 1;
        }
        // Pad with zeros if history is shorter
        for _ in history.len()..config.history_length {
            output[idx] = 0.0;
            idx += 1;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psd_compression_average() {
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Average);

        assert_eq!(compressed.len(), 4);
        assert!((compressed[0] - 1.5).abs() < 1e-6); // avg(1, 2)
        assert!((compressed[1] - 3.5).abs() < 1e-6); // avg(3, 4)
        assert!((compressed[2] - 5.5).abs() < 1e-6); // avg(5, 6)
        assert!((compressed[3] - 7.5).abs() < 1e-6); // avg(7, 8)
    }

    #[test]
    fn test_psd_compression_max() {
        let psd = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Max);

        assert_eq!(compressed.len(), 4);
        assert!((compressed[0] - 5.0).abs() < 1e-6); // max(1, 5)
        assert!((compressed[1] - 7.0).abs() < 1e-6); // max(3, 7)
        assert!((compressed[2] - 6.0).abs() < 1e-6); // max(2, 6)
        assert!((compressed[3] - 8.0).abs() < 1e-6); // max(4, 8)
    }

    #[test]
    fn test_psd_compression_subsample() {
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Subsample);

        assert_eq!(compressed.len(), 4);
        assert!((compressed[0] - 1.0).abs() < 1e-6);
        assert!((compressed[1] - 3.0).abs() < 1e-6);
        assert!((compressed[2] - 5.0).abs() < 1e-6);
        assert!((compressed[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_psd() {
        // 1e-10 in linear = -100 dBm
        // 1e-5 in linear = -50 dBm
        let psd = vec![1e-10, 1e-8, 1e-6, 1e-4];
        let normalized = normalize_psd(&psd, -100.0, -40.0);

        assert_eq!(normalized.len(), 4);
        assert!((normalized[0] - 0.0).abs() < 0.01); // -100 dB -> 0
        assert!(normalized[3] > 0.99); // -40 dB -> 1
    }

    #[test]
    fn test_observation_size_calculation() {
        let config = ObservationConfig {
            num_freq_bins: 64,
            include_psd: true,
            include_agent_state: true,
            include_team_state: true,
            include_opponent_state: false,
            include_history: true,
            history_length: 8,
            ..ObservationConfig::default()
        };

        // 1 jammer, 1 CR
        let jammer_size = config.jammer_observation_size(1, 1);
        // PSD (64) + own state (8) + history (8) = 80
        assert_eq!(jammer_size, 64 + 8 + 8);

        let cr_size = config.cr_observation_size(1, 1);
        // PSD (64) + own state (10) + history (8) = 82
        assert_eq!(cr_size, 64 + 10 + 8);

        // 2 jammers, 3 CRs with team state
        let jammer_size = config.jammer_observation_size(2, 3);
        // PSD (64) + own (8) + 1 teammate (6) + history (8) = 86
        assert_eq!(jammer_size, 64 + 8 + 6 + 8);

        let cr_size = config.cr_observation_size(2, 3);
        // PSD (64) + own (10) + 2 teammates (10) + history (8) = 92
        assert_eq!(cr_size, 64 + 10 + 10 + 8);
    }

    #[test]
    fn test_multi_agent_observations_indexing() {
        let obs = MultiAgentObservations::new(8, 2, 3, 100, 120);

        // Check sizes
        assert_eq!(obs.jammer_obs.len(), 8 * 2 * 100);
        assert_eq!(obs.cr_obs.len(), 8 * 3 * 120);

        // Check indexing
        assert_eq!(obs.jammer_start(0, 0), 0);
        assert_eq!(obs.jammer_start(0, 1), 100);
        assert_eq!(obs.jammer_start(1, 0), 200);

        assert_eq!(obs.cr_start(0, 0), 0);
        assert_eq!(obs.cr_start(0, 2), 240);
        assert_eq!(obs.cr_start(1, 0), 360);
    }

    #[test]
    fn test_observation_slices() {
        let mut obs = MultiAgentObservations::new(2, 1, 1, 10, 15);

        // Write to jammer observation
        {
            let slice = obs.jammer_observation_mut(0, 0);
            slice[0] = 1.0;
            slice[9] = 2.0;
        }

        // Read back
        let slice = obs.jammer_observation(0, 0);
        assert_eq!(slice.len(), 10);
        assert!((slice[0] - 1.0).abs() < 1e-6);
        assert!((slice[9] - 2.0).abs() < 1e-6);

        // Different env should be separate
        let slice_other = obs.jammer_observation(1, 0);
        assert!((slice_other[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_observation_config_builder() {
        let config = ObservationConfig::new()
            .with_psd_bins(128)
            .with_opponent_state()
            .with_history_length(16);

        assert_eq!(config.num_freq_bins, 128);
        assert!(config.include_opponent_state);
        assert_eq!(config.history_length, 16);
    }
}
