//! Reward Functions
//!
//! Implements reward calculations for both jammer and cognitive radio agents
//! in the adversarial multi-agent setting.

use super::interference::InterferenceMatrix;
use super::multi_agent::MultiAgentState;

// ============================================================================
// Jammer Reward Configuration
// ============================================================================

/// Configuration for jammer reward calculation
#[derive(Clone, Debug)]
pub struct JammerRewardConfig {
    /// Weight for disruption reward (higher = more emphasis on jamming)
    pub disruption_weight: f32,
    /// Weight for target matching (bonus for jamming intended target)
    pub target_match_weight: f32,
    /// Penalty weight for power usage (energy efficiency)
    pub power_penalty_weight: f32,
    /// Penalty for causing collateral damage (unintended CR disruption)
    pub collateral_penalty_weight: f32,
    /// Bonus for complete CR shutdown
    pub shutdown_bonus: f32,
    /// Maximum power for normalization (dBm)
    pub max_power_dbm: f32,
}

impl Default for JammerRewardConfig {
    fn default() -> Self {
        Self {
            disruption_weight: 1.0,
            target_match_weight: 0.5,
            power_penalty_weight: 0.01,
            collateral_penalty_weight: 0.1,
            shutdown_bonus: 2.0,
            max_power_dbm: 30.0,
        }
    }
}

impl JammerRewardConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set disruption weight
    pub fn with_disruption_weight(mut self, weight: f32) -> Self {
        self.disruption_weight = weight;
        self
    }

    /// Set power penalty weight
    pub fn with_power_penalty(mut self, weight: f32) -> Self {
        self.power_penalty_weight = weight;
        self
    }

    /// Aggressive config (maximize disruption)
    pub fn aggressive() -> Self {
        Self {
            disruption_weight: 2.0,
            target_match_weight: 0.0,
            power_penalty_weight: 0.0,
            collateral_penalty_weight: 0.0,
            shutdown_bonus: 5.0,
            max_power_dbm: 30.0,
        }
    }

    /// Efficient config (balance disruption and power)
    pub fn efficient() -> Self {
        Self {
            disruption_weight: 1.0,
            target_match_weight: 0.5,
            power_penalty_weight: 0.05,
            collateral_penalty_weight: 0.0,
            shutdown_bonus: 1.0,
            max_power_dbm: 30.0,
        }
    }
}

// ============================================================================
// Cognitive Radio Reward Configuration
// ============================================================================

/// Configuration for CR reward calculation
#[derive(Clone, Debug)]
pub struct CRRewardConfig {
    /// Weight for throughput reward
    pub throughput_weight: f32,
    /// Penalty for collisions/interference
    pub collision_penalty: f32,
    /// Penalty for power usage
    pub power_penalty_weight: f32,
    /// Penalty for frequency switching
    pub switching_penalty: f32,
    /// Bonus for staying alive (not jammed)
    pub alive_bonus: f32,
    /// SINR threshold for successful communication (dB)
    pub sinr_threshold_db: f32,
    /// Maximum normalized throughput
    pub max_throughput: f32,
    /// Maximum power for normalization (dBm)
    pub max_power_dbm: f32,
}

impl Default for CRRewardConfig {
    fn default() -> Self {
        Self {
            throughput_weight: 1.0,
            collision_penalty: 0.5,
            power_penalty_weight: 0.01,
            switching_penalty: 0.05,
            alive_bonus: 0.1,
            sinr_threshold_db: 10.0,
            max_throughput: 1.0,
            max_power_dbm: 30.0,
        }
    }
}

impl CRRewardConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set throughput weight
    pub fn with_throughput_weight(mut self, weight: f32) -> Self {
        self.throughput_weight = weight;
        self
    }

    /// Set SINR threshold
    pub fn with_sinr_threshold(mut self, threshold_db: f32) -> Self {
        self.sinr_threshold_db = threshold_db;
        self
    }

    /// Aggressive config (maximize throughput regardless of energy)
    pub fn aggressive() -> Self {
        Self {
            throughput_weight: 2.0,
            collision_penalty: 0.2,
            power_penalty_weight: 0.0,
            switching_penalty: 0.0,
            alive_bonus: 0.5,
            sinr_threshold_db: 10.0,
            max_throughput: 1.0,
            max_power_dbm: 30.0,
        }
    }

    /// Cautious config (avoid collisions at all costs)
    pub fn cautious() -> Self {
        Self {
            throughput_weight: 0.5,
            collision_penalty: 2.0,
            power_penalty_weight: 0.02,
            switching_penalty: 0.1,
            alive_bonus: 0.2,
            sinr_threshold_db: 15.0,
            max_throughput: 1.0,
            max_power_dbm: 30.0,
        }
    }
}

// ============================================================================
// Combined Reward Configuration
// ============================================================================

/// Combined reward configuration for all agents
#[derive(Clone, Debug)]
pub struct RewardConfig {
    /// Jammer reward configuration
    pub jammer: JammerRewardConfig,
    /// CR reward configuration
    pub cr: CRRewardConfig,
    /// Whether to use zero-sum rewards (jammer gain = -CR loss)
    pub zero_sum: bool,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            jammer: JammerRewardConfig::default(),
            cr: CRRewardConfig::default(),
            zero_sum: false,
        }
    }
}

impl RewardConfig {
    /// Create new configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set jammer configuration
    pub fn with_jammer(mut self, config: JammerRewardConfig) -> Self {
        self.jammer = config;
        self
    }

    /// Set CR configuration
    pub fn with_cr(mut self, config: CRRewardConfig) -> Self {
        self.cr = config;
        self
    }

    /// Enable zero-sum rewards
    pub fn with_zero_sum(mut self, enabled: bool) -> Self {
        self.zero_sum = enabled;
        self
    }
}

// ============================================================================
// Multi-Agent Rewards
// ============================================================================

/// Rewards for all agents in a step
#[derive(Clone, Debug)]
pub struct MultiAgentRewards {
    /// Jammer rewards: [num_envs × num_jammers]
    pub jammer_rewards: Vec<f32>,
    /// CR rewards: [num_envs × num_crs]
    pub cr_rewards: Vec<f32>,
    /// Number of environments
    pub num_envs: usize,
    /// Number of jammers per environment
    pub num_jammers: usize,
    /// Number of CRs per environment
    pub num_crs: usize,
}

impl MultiAgentRewards {
    /// Create new rewards structure
    pub fn new(num_envs: usize, num_jammers: usize, num_crs: usize) -> Self {
        Self {
            jammer_rewards: vec![0.0; num_envs * num_jammers],
            cr_rewards: vec![0.0; num_envs * num_crs],
            num_envs,
            num_jammers,
            num_crs,
        }
    }

    /// Get jammer reward index
    #[inline]
    pub fn jammer_idx(&self, env: usize, jammer: usize) -> usize {
        env * self.num_jammers + jammer
    }

    /// Get CR reward index
    #[inline]
    pub fn cr_idx(&self, env: usize, cr: usize) -> usize {
        env * self.num_crs + cr
    }

    /// Get jammer reward
    #[inline]
    pub fn jammer_reward(&self, env: usize, jammer: usize) -> f32 {
        self.jammer_rewards[self.jammer_idx(env, jammer)]
    }

    /// Set jammer reward
    #[inline]
    pub fn set_jammer_reward(&mut self, env: usize, jammer: usize, reward: f32) {
        let idx = self.jammer_idx(env, jammer);
        self.jammer_rewards[idx] = reward;
    }

    /// Get CR reward
    #[inline]
    pub fn cr_reward(&self, env: usize, cr: usize) -> f32 {
        self.cr_rewards[self.cr_idx(env, cr)]
    }

    /// Set CR reward
    #[inline]
    pub fn set_cr_reward(&mut self, env: usize, cr: usize, reward: f32) {
        let idx = self.cr_idx(env, cr);
        self.cr_rewards[idx] = reward;
    }

    /// Total jammer rewards for an environment
    pub fn total_jammer_reward(&self, env: usize) -> f32 {
        (0..self.num_jammers)
            .map(|j| self.jammer_reward(env, j))
            .sum()
    }

    /// Total CR rewards for an environment
    pub fn total_cr_reward(&self, env: usize) -> f32 {
        (0..self.num_crs).map(|c| self.cr_reward(env, c)).sum()
    }

    /// Flatten all rewards to a single vector
    ///
    /// Layout: [env0_j0, env0_j1, ..., env0_cr0, env0_cr1, ..., env1_j0, ...]
    pub fn to_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.num_envs * (self.num_jammers + self.num_crs));

        for env in 0..self.num_envs {
            for j in 0..self.num_jammers {
                flat.push(self.jammer_reward(env, j));
            }
            for c in 0..self.num_crs {
                flat.push(self.cr_reward(env, c));
            }
        }

        flat
    }

    /// Get rewards as separate jammer and CR vectors
    pub fn split(&self) -> (Vec<f32>, Vec<f32>) {
        (self.jammer_rewards.clone(), self.cr_rewards.clone())
    }

    /// Compute rewards for a single environment
    pub fn compute_env(
        state: &MultiAgentState,
        interference: &InterferenceMatrix,
        env: usize,
        config: &RewardConfig,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut jammer_rewards = Vec::with_capacity(state.num_jammers);
        let mut cr_rewards = Vec::with_capacity(state.num_crs);

        // Compute jammer rewards
        for j in 0..state.num_jammers {
            let reward = compute_jammer_reward(state, interference, env, j, &config.jammer);
            jammer_rewards.push(reward);
        }

        // Compute CR rewards
        for c in 0..state.num_crs {
            let reward = compute_cr_reward(state, interference, env, c, &config.cr);
            cr_rewards.push(reward);
        }

        // Apply zero-sum adjustment if enabled
        if config.zero_sum {
            let total_cr_loss: f32 = cr_rewards.iter().filter(|&&r| r < 0.0).sum();
            let jammer_bonus = -total_cr_loss / state.num_jammers.max(1) as f32;

            for r in &mut jammer_rewards {
                *r += jammer_bonus;
            }
        }

        (jammer_rewards, cr_rewards)
    }

    /// Compute rewards for all environments
    pub fn compute(
        state: &MultiAgentState,
        interference_matrices: &[InterferenceMatrix],
        config: &RewardConfig,
    ) -> Self {
        let mut rewards = Self::new(state.num_envs, state.num_jammers, state.num_crs);

        for env in 0..state.num_envs {
            let (jammer_rewards, cr_rewards) =
                Self::compute_env(state, &interference_matrices[env], env, config);

            for (j, r) in jammer_rewards.into_iter().enumerate() {
                rewards.set_jammer_reward(env, j, r);
            }
            for (c, r) in cr_rewards.into_iter().enumerate() {
                rewards.set_cr_reward(env, c, r);
            }
        }

        rewards
    }
}

// ============================================================================
// Individual Reward Functions
// ============================================================================

/// Compute reward for a single jammer
pub fn compute_jammer_reward(
    state: &MultiAgentState,
    interference: &InterferenceMatrix,
    env: usize,
    jammer: usize,
    config: &JammerRewardConfig,
) -> f32 {
    let j_idx = state.jammer_idx(env, jammer);
    let mut reward = 0.0;

    // Disruption reward: based on number of CRs jammed
    let victims = interference.jammer_victim_count(jammer);
    reward += config.disruption_weight * victims as f32;

    // Target matching bonus
    let target = state.jammer_target[j_idx];
    if target >= 0 && (target as usize) < state.num_crs {
        if interference.is_jammed(jammer, target as usize) {
            reward += config.target_match_weight;
        }
    }

    // Check for shutdown bonus (CR marked as not alive)
    for c in 0..state.num_crs {
        if interference.is_jammed(jammer, c) && !state.cr_is_alive(env, c) {
            reward += config.shutdown_bonus;
        }
    }

    // Power penalty
    let power = state.jammer_power[j_idx];
    let normalized_power = (power / config.max_power_dbm).clamp(0.0, 1.0);
    reward -= config.power_penalty_weight * normalized_power;

    // Collateral penalty (if targeting specific CRs)
    if target >= 0 {
        let unintended_victims = victims.saturating_sub(1);
        reward -= config.collateral_penalty_weight * unintended_victims as f32;
    }

    reward
}

/// Compute reward for a single cognitive radio
pub fn compute_cr_reward(
    state: &MultiAgentState,
    interference: &InterferenceMatrix,
    env: usize,
    cr: usize,
    config: &CRRewardConfig,
) -> f32 {
    let c_idx = state.cr_idx(env, cr);
    let mut reward = 0.0;

    // Throughput reward
    let throughput = state.cr_throughput[c_idx];
    reward += config.throughput_weight * throughput / config.max_throughput;

    // Alive bonus
    if state.cr_is_alive(env, cr) {
        reward += config.alive_bonus;
    }

    // SINR-based quality
    let sinr = interference.sinr(cr);
    if sinr >= config.sinr_threshold_db {
        // Bonus for exceeding threshold
        let sinr_margin = (sinr - config.sinr_threshold_db) / 10.0;
        reward += 0.1 * sinr_margin.min(1.0);
    }

    // Collision penalty
    if interference.cr_is_jammed(cr) {
        reward -= config.collision_penalty;
    }

    // Switching penalty (based on hop count)
    let hops_this_step = if state.cr_hop_count[c_idx] > 0 { 1.0 } else { 0.0 };
    reward -= config.switching_penalty * hops_this_step;

    // Power penalty
    let power = state.cr_power[c_idx];
    let normalized_power = (power / config.max_power_dbm).clamp(0.0, 1.0);
    reward -= config.power_penalty_weight * normalized_power;

    reward
}

// ============================================================================
// Shaped Rewards (for curriculum learning)
// ============================================================================

/// Compute shaping reward for CR based on distance to nearest interference
pub fn cr_shaping_reward(
    state: &MultiAgentState,
    env: usize,
    cr: usize,
    previous_freq: f32,
) -> f32 {
    let c_idx = state.cr_idx(env, cr);
    let current_freq = state.cr_freq[c_idx];

    // Find nearest jammer frequency
    let mut min_jammer_dist = f32::INFINITY;
    for j in 0..state.num_jammers {
        let j_idx = state.jammer_idx(env, j);
        let jammer_freq = state.jammer_freq[j_idx];
        let jammer_bw = state.jammer_bandwidth[j_idx];

        // Distance to jammer band edges
        let low = jammer_freq - jammer_bw / 2.0;
        let high = jammer_freq + jammer_bw / 2.0;

        let dist = if current_freq < low {
            low - current_freq
        } else if current_freq > high {
            current_freq - high
        } else {
            0.0 // Inside jammer band
        };

        min_jammer_dist = min_jammer_dist.min(dist);
    }

    // Previous distance
    let mut prev_min_dist = f32::INFINITY;
    for j in 0..state.num_jammers {
        let j_idx = state.jammer_idx(env, j);
        let jammer_freq = state.jammer_freq[j_idx];
        let jammer_bw = state.jammer_bandwidth[j_idx];

        let low = jammer_freq - jammer_bw / 2.0;
        let high = jammer_freq + jammer_bw / 2.0;

        let dist = if previous_freq < low {
            low - previous_freq
        } else if previous_freq > high {
            previous_freq - high
        } else {
            0.0
        };

        prev_min_dist = prev_min_dist.min(dist);
    }

    // Reward for moving away from interference
    let improvement = min_jammer_dist - prev_min_dist;

    // Normalize by bandwidth for scale-invariance
    let scale = 1e7; // 10 MHz normalization
    (improvement / scale).clamp(-1.0, 1.0) * 0.1
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_state() -> (MultiAgentState, InterferenceMatrix) {
        let mut state = MultiAgentState::new(1, 2, 2, 4);

        // Set jammer state
        for j in 0..2 {
            let idx = state.jammer_idx(0, j);
            state.jammer_freq[idx] = 2.45e9;
            state.jammer_bandwidth[idx] = 20e6;
            state.jammer_power[idx] = 20.0;
        }

        // Set CR state
        for c in 0..2 {
            let idx = state.cr_idx(0, c);
            state.cr_freq[idx] = if c == 0 { 2.45e9 } else { 2.48e9 }; // CR0 overlaps, CR1 doesn't
            state.cr_bandwidth[idx] = 10e6;
            state.cr_power[idx] = 15.0;
            state.cr_throughput[idx] = if c == 0 { 0.0 } else { 0.8 };
            state.cr_alive[idx] = 1;
        }

        let mut interference = InterferenceMatrix::new(2, 2);

        // Jammer 0 jams CR 0
        interference.set_jammed(0, 0, true);
        interference.set_jammer_interference(0, 0, -50.0);
        interference.cr_sinr[0] = 5.0; // Below threshold
        interference.cr_sinr[1] = 25.0; // Above threshold

        (state, interference)
    }

    #[test]
    fn test_jammer_disruption_reward() {
        let (state, interference) = setup_test_state();
        let config = JammerRewardConfig::default();

        let reward = compute_jammer_reward(&state, &interference, 0, 0, &config);

        // Should have positive reward for jamming CR 0
        assert!(reward > 0.0, "Jammer reward = {}", reward);
    }

    #[test]
    fn test_jammer_no_victims() {
        let (state, interference) = setup_test_state();
        let config = JammerRewardConfig::default();

        // Jammer 1 has no victims
        let reward = compute_jammer_reward(&state, &interference, 0, 1, &config);

        // Should be near zero or negative (power penalty only)
        assert!(reward <= 0.1, "Jammer reward = {}", reward);
    }

    #[test]
    fn test_cr_throughput_reward() {
        let (state, interference) = setup_test_state();
        let config = CRRewardConfig::default();

        // CR 1 has throughput
        let reward = compute_cr_reward(&state, &interference, 0, 1, &config);
        assert!(reward > 0.0, "CR reward = {}", reward);

        // CR 0 has no throughput
        let reward0 = compute_cr_reward(&state, &interference, 0, 0, &config);
        assert!(reward0 < reward, "CR0 reward {} >= CR1 reward {}", reward0, reward);
    }

    #[test]
    fn test_cr_collision_penalty() {
        let (state, interference) = setup_test_state();
        let config = CRRewardConfig::default();

        // CR 0 is jammed, should have penalty
        let reward = compute_cr_reward(&state, &interference, 0, 0, &config);

        // Should have negative component from collision
        let reward_no_penalty = compute_cr_reward(&state, &interference, 0, 1, &config);
        assert!(reward < reward_no_penalty);
    }

    #[test]
    fn test_multi_agent_rewards_compute() {
        let (state, interference) = setup_test_state();
        let config = RewardConfig::default();

        let matrices = vec![interference];
        let rewards = MultiAgentRewards::compute(&state, &matrices, &config);

        assert_eq!(rewards.num_envs, 1);
        assert_eq!(rewards.num_jammers, 2);
        assert_eq!(rewards.num_crs, 2);

        // Verify we got some rewards
        let j0 = rewards.jammer_reward(0, 0);
        let c1 = rewards.cr_reward(0, 1);

        assert!(j0.is_finite());
        assert!(c1.is_finite());
    }

    #[test]
    fn test_rewards_flatten() {
        let mut rewards = MultiAgentRewards::new(2, 1, 1);

        rewards.set_jammer_reward(0, 0, 1.0);
        rewards.set_cr_reward(0, 0, 2.0);
        rewards.set_jammer_reward(1, 0, 3.0);
        rewards.set_cr_reward(1, 0, 4.0);

        let flat = rewards.to_flat();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_power_penalty() {
        let (mut state, interference) = setup_test_state();

        let config = JammerRewardConfig {
            power_penalty_weight: 0.5,
            disruption_weight: 0.0,
            target_match_weight: 0.0,
            shutdown_bonus: 0.0,
            collateral_penalty_weight: 0.0,
            max_power_dbm: 30.0,
        };

        // High power jammer
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_power[j_idx] = 30.0; // Max power

        let reward = compute_jammer_reward(&state, &interference, 0, 0, &config);

        // Should have negative reward from power penalty
        assert!(reward < 0.0, "High power reward = {}", reward);

        // Low power jammer
        state.jammer_power[j_idx] = 0.0;
        let reward_low = compute_jammer_reward(&state, &interference, 0, 0, &config);

        assert!(reward_low > reward, "Low power reward {} <= high power {}", reward_low, reward);
    }

    #[test]
    fn test_shaping_reward() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);

        // Set jammer at 2.45 GHz
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_bandwidth[j_idx] = 20e6;

        // CR moved away from jammer
        let c_idx = state.cr_idx(0, 0);
        state.cr_freq[c_idx] = 2.48e9; // Far from jammer

        let previous_freq = 2.46e9; // Was closer

        let shaping = cr_shaping_reward(&state, 0, 0, previous_freq);

        // Should be positive for moving away
        assert!(shaping > 0.0, "Shaping reward = {}", shaping);

        // Moving closer should be negative
        let shaping_bad = cr_shaping_reward(&state, 0, 0, 2.49e9);
        assert!(shaping_bad < 0.0, "Bad shaping reward = {}", shaping_bad);
    }

    #[test]
    fn test_zero_sum_rewards() {
        let (state, interference) = setup_test_state();

        let config = RewardConfig {
            jammer: JammerRewardConfig::default(),
            cr: CRRewardConfig {
                collision_penalty: 2.0,
                ..CRRewardConfig::default()
            },
            zero_sum: true,
        };

        let matrices = vec![interference];
        let rewards = MultiAgentRewards::compute(&state, &matrices, &config);

        // With zero-sum, jammer should benefit from CR losses
        let total_j = rewards.total_jammer_reward(0);
        let total_cr = rewards.total_cr_reward(0);

        // Not strictly zero-sum due to other reward terms, but should show correlation
        assert!(total_j.is_finite());
        assert!(total_cr.is_finite());
    }

    #[test]
    fn test_reward_config_presets() {
        let jammer_aggressive = JammerRewardConfig::aggressive();
        let jammer_efficient = JammerRewardConfig::efficient();

        assert!(jammer_aggressive.disruption_weight > jammer_efficient.disruption_weight);
        assert!(jammer_aggressive.power_penalty_weight < jammer_efficient.power_penalty_weight);

        let cr_aggressive = CRRewardConfig::aggressive();
        let cr_cautious = CRRewardConfig::cautious();

        assert!(cr_aggressive.collision_penalty < cr_cautious.collision_penalty);
    }
}
