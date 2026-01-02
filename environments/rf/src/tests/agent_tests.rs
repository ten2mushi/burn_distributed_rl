//! Agent Tests
//!
//! Tests for multi-agent actions, state, interference, observations, and rewards.

use crate::agents::{
    ActionConfig, ActionSpace, AgentType, CognitiveRadioAction, JammerAction, JammerModulation,
    MultiAgentActions, MultiAgentConfig, MultiAgentState,
    InterferenceMatrix, frequency_overlap,
    MultiAgentRewards, RewardConfig, JammerRewardConfig, CRRewardConfig,
    compute_jammer_reward, compute_cr_reward,
};
use crate::observation::{
    compress_psd, normalize_psd, MultiAgentObservations, ObservationConfig, PsdCompression,
};

// ============================================================================
// Action Tests
// ============================================================================

mod action_tests {
    use super::*;

    #[test]
    fn test_action_denormalization() {
        let config = ActionConfig::new()
            .with_freq_range(2.4e9, 2.5e9)
            .with_power_range(-10.0, 30.0)
            .with_bandwidth_range(1e6, 20e6);

        // Test frequency denormalization
        let freq_0 = config.denorm_freq(0.0);
        let freq_1 = config.denorm_freq(1.0);
        let freq_mid = config.denorm_freq(0.5);

        assert!((freq_0 - 2.4e9).abs() < 1.0);
        assert!((freq_1 - 2.5e9).abs() < 1.0);
        assert!((freq_mid - 2.45e9).abs() < 1.0);

        // Test power denormalization
        let power_0 = config.denorm_power(0.0);
        let power_1 = config.denorm_power(1.0);

        assert!((power_0 - (-10.0)).abs() < 0.01);
        assert!((power_1 - 30.0).abs() < 0.01);

        // Test bandwidth denormalization
        let bw_0 = config.denorm_bandwidth(0.0);
        let bw_1 = config.denorm_bandwidth(1.0);

        assert!((bw_0 - 1e6).abs() < 1.0);
        assert!((bw_1 - 20e6).abs() < 1.0);
    }

    #[test]
    fn test_action_flatten_roundtrip() {
        let num_envs = 4;
        let num_jammers = 2;
        let num_crs = 3;

        let mut original = MultiAgentActions::new(num_envs, num_jammers, num_crs);

        // Set some specific actions
        original.set_jammer_action(0, 0, JammerAction::new(0.1, 0.2, 0.3, 0.4));
        original.set_jammer_action(1, 1, JammerAction::new(0.5, 0.6, 0.7, 0.8));
        original.set_cr_action(2, 0, CognitiveRadioAction::new(0.9, 0.8, 0.7));
        original.set_cr_action(3, 2, CognitiveRadioAction::new(0.1, 0.2, 0.3));

        // Flatten
        let flat = original.to_flat();

        // Recover
        let recovered = MultiAgentActions::from_flat(&flat, num_envs, num_jammers, num_crs);

        // Verify
        for env in 0..num_envs {
            for j in 0..num_jammers {
                let orig = original.jammer_action(env, j);
                let recov = recovered.jammer_action(env, j);
                assert!((orig.frequency - recov.frequency).abs() < 1e-6);
                assert!((orig.bandwidth - recov.bandwidth).abs() < 1e-6);
                assert!((orig.power - recov.power).abs() < 1e-6);
                assert!((orig.modulation - recov.modulation).abs() < 1e-6);
            }
            for c in 0..num_crs {
                let orig = original.cr_action(env, c);
                let recov = recovered.cr_action(env, c);
                assert!((orig.frequency - recov.frequency).abs() < 1e-6);
                assert!((orig.power - recov.power).abs() < 1e-6);
                assert!((orig.bandwidth - recov.bandwidth).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_action_dim_calculation() {
        // 4 per jammer + 3 per CR
        let space = ActionSpace::new(2, 3, ActionConfig::default());
        assert_eq!(space.total_dim(), 2 * 4 + 3 * 3); // 8 + 9 = 17
        assert_eq!(space.jammer_dim(), 8);
        assert_eq!(space.cr_dim(), 9);

        let space2 = ActionSpace::new(0, 5, ActionConfig::default());
        assert_eq!(space2.total_dim(), 0 * 4 + 5 * 3); // 0 + 15 = 15

        let space3 = ActionSpace::new(1, 1, ActionConfig::default());
        assert_eq!(space3.total_dim(), 1 * 4 + 1 * 3); // 4 + 3 = 7
    }

    #[test]
    fn test_jammer_modulation_roundtrip() {
        for mod_type in JammerModulation::all() {
            let normalized = mod_type.to_normalized();
            let recovered = JammerModulation::from_normalized(normalized);
            assert_eq!(mod_type, recovered);
        }
    }
}

// ============================================================================
// Multi-Agent State Tests
// ============================================================================

mod state_tests {
    use super::*;

    #[test]
    fn test_agent_state_indexing() {
        let state = MultiAgentState::new(8, 3, 4, 8);

        // Jammer indexing: env-major, jammer-minor
        assert_eq!(state.jammer_idx(0, 0), 0);
        assert_eq!(state.jammer_idx(0, 1), 1);
        assert_eq!(state.jammer_idx(0, 2), 2);
        assert_eq!(state.jammer_idx(1, 0), 3);
        assert_eq!(state.jammer_idx(7, 2), 7 * 3 + 2);

        // CR indexing: env-major, cr-minor
        assert_eq!(state.cr_idx(0, 0), 0);
        assert_eq!(state.cr_idx(0, 3), 3);
        assert_eq!(state.cr_idx(1, 0), 4);
        assert_eq!(state.cr_idx(7, 3), 7 * 4 + 3);
    }

    #[test]
    fn test_agent_state_reset() {
        let mut state = MultiAgentState::new(2, 2, 2, 4);

        // Set some values
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_power[j_idx] = 20.0;
        state.jammer_success_rate[j_idx] = 0.75;

        let c_idx = state.cr_idx(0, 0);
        state.cr_freq[c_idx] = 2.48e9;
        state.cr_throughput[c_idx] = 0.9;
        state.cr_collisions[c_idx] = 5;
        state.cr_hop_count[c_idx] = 10;
        state.cr_alive[c_idx] = 0;

        // Reset env 0
        state.reset_env(0);

        // Check jammer reset
        assert!((state.jammer_freq[j_idx]).abs() < 1e-6);
        assert!((state.jammer_power[j_idx]).abs() < 1e-6);
        assert!((state.jammer_success_rate[j_idx]).abs() < 1e-6);

        // Check CR reset
        assert!((state.cr_freq[c_idx]).abs() < 1e-6);
        assert!((state.cr_throughput[c_idx]).abs() < 1e-6);
        assert_eq!(state.cr_collisions[c_idx], 0);
        assert_eq!(state.cr_hop_count[c_idx], 0);
        assert_eq!(state.cr_alive[c_idx], 1); // Should be alive after reset

        // Env 1 should be unchanged
        let j_idx_1 = state.jammer_idx(1, 0);
        let c_idx_1 = state.cr_idx(1, 0);
        // These should still be at initial values (0), not affected by our changes to env 0
        assert!((state.jammer_freq[j_idx_1]).abs() < 1e-6);
    }

    #[test]
    fn test_agent_state_history_tracking() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);
        let config = ActionConfig::default();

        // Apply multiple CR actions
        let freqs = [0.1, 0.3, 0.5, 0.7, 0.9];
        for &f in &freqs {
            let action = CognitiveRadioAction::new(f, 0.5, 0.5);
            state.apply_cr_action(0, 0, &action, &config);
        }

        // Check history
        let history = state.cr_freq_history_slice(0, 0);
        assert_eq!(history.len(), 4);

        // Most recent should be at index 0 (after apply, current freq is pushed to history)
        // The history contains the PREVIOUS frequencies (before the current action was applied)
        // So history[0] = freq from action 4 (0.7 normalized)
        // history[1] = freq from action 3 (0.5 normalized)
        // etc.
    }

    #[test]
    fn test_count_alive_crs() {
        let mut state = MultiAgentState::new(1, 1, 4, 4);

        // All CRs start alive
        assert_eq!(state.count_alive_crs(0), 4);

        // Mark some as dead
        let c1_idx = state.cr_idx(0, 1);
        let c3_idx = state.cr_idx(0, 3);
        state.cr_alive[c1_idx] = 0;
        state.cr_alive[c3_idx] = 0;

        assert_eq!(state.count_alive_crs(0), 2);
    }
}

// ============================================================================
// Interference Tests
// ============================================================================

mod interference_tests {
    use super::*;

    #[test]
    fn test_frequency_overlap_complete() {
        // Same frequency and bandwidth
        let overlap = frequency_overlap(100.0, 10.0, 100.0, 10.0);
        assert!((overlap - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_overlap_none() {
        // Completely disjoint
        let overlap = frequency_overlap(100.0, 10.0, 200.0, 10.0);
        assert!(overlap.abs() < 1e-6);
    }

    #[test]
    fn test_frequency_overlap_partial() {
        // Signal 1: 95-105, Signal 2: 100-110
        // Overlap: 100-105 = 5 out of 10 = 50%
        let overlap = frequency_overlap(100.0, 10.0, 105.0, 10.0);
        assert!((overlap - 0.5).abs() < 1e-6, "overlap = {}", overlap);
    }

    #[test]
    fn test_interference_matrix_jam_success() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);

        // Set up jammer on same frequency as CR
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_bandwidth[j_idx] = 20e6;
        state.jammer_power[j_idx] = 30.0; // High power
        state.jammer_x[j_idx] = 0.0;
        state.jammer_y[j_idx] = 0.0;

        let c_idx = state.cr_idx(0, 0);
        state.cr_freq[c_idx] = 2.45e9; // Same frequency!
        state.cr_bandwidth[c_idx] = 10e6;
        state.cr_power[c_idx] = 10.0; // Lower power
        state.cr_x[c_idx] = 50.0;
        state.cr_y[c_idx] = 0.0;

        let matrix = InterferenceMatrix::compute(&state, 0, -100.0, 10.0);

        // Should detect interference
        assert!(matrix.jammer_interference(0, 0) > -100.0);

        // SINR should be low due to jammer
        assert!(matrix.sinr(0).is_finite());
    }

    #[test]
    fn test_jam_success_threshold() {
        let mut matrix = InterferenceMatrix::new(2, 2);

        // Initially no one jammed
        assert!(!matrix.is_jammed(0, 0));
        assert!(!matrix.is_jammed(0, 1));
        assert!(!matrix.is_jammed(1, 0));
        assert!(!matrix.is_jammed(1, 1));

        // Set some jams
        matrix.set_jammed(0, 0, true);
        matrix.set_jammed(1, 1, true);

        assert!(matrix.is_jammed(0, 0));
        assert!(!matrix.is_jammed(0, 1));
        assert!(!matrix.is_jammed(1, 0));
        assert!(matrix.is_jammed(1, 1));

        // Test victim counts
        assert_eq!(matrix.jammer_victim_count(0), 1);
        assert_eq!(matrix.jammer_victim_count(1), 1);

        // Test CR jammed status
        assert!(matrix.cr_is_jammed(0));
        assert!(matrix.cr_is_jammed(1));
    }
}

// ============================================================================
// Observation Tests
// ============================================================================

mod observation_tests {
    use super::*;

    #[test]
    fn test_psd_compression() {
        let psd: Vec<f32> = (0..128).map(|i| i as f32).collect();

        // Average compression
        let compressed = compress_psd(&psd, 32, PsdCompression::Average);
        assert_eq!(compressed.len(), 32);

        // Each bin should be average of 4 original bins
        // First bin: avg(0, 1, 2, 3) = 1.5
        assert!((compressed[0] - 1.5).abs() < 1e-6);

        // Max compression
        let compressed_max = compress_psd(&psd, 32, PsdCompression::Max);
        // First bin: max(0, 1, 2, 3) = 3
        assert!((compressed_max[0] - 3.0).abs() < 1e-6);

        // Subsample compression
        let compressed_sub = compress_psd(&psd, 32, PsdCompression::Subsample);
        // First bin: psd[0] = 0, second bin: psd[4] = 4
        assert!((compressed_sub[0] - 0.0).abs() < 1e-6);
        assert!((compressed_sub[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_observation_normalization() {
        // PSD values in linear power
        let psd = vec![1e-10, 1e-8, 1e-6, 1e-4]; // -100, -80, -60, -40 dBm

        let normalized = normalize_psd(&psd, -100.0, -40.0);

        assert_eq!(normalized.len(), 4);
        // -100 dBm should normalize to 0
        assert!(normalized[0] < 0.05);
        // -40 dBm should normalize to 1
        assert!(normalized[3] > 0.95);
        // Values should be monotonically increasing
        for i in 1..normalized.len() {
            assert!(normalized[i] >= normalized[i - 1]);
        }
    }

    #[test]
    fn test_observation_size_calculation() {
        let config = ObservationConfig {
            include_psd: true,
            num_freq_bins: 64,
            include_agent_state: true,
            include_team_state: true,
            include_opponent_state: false,
            include_history: true,
            history_length: 8,
            ..ObservationConfig::default()
        };

        // 1 jammer, 1 CR - no team state
        let jammer_size = config.jammer_observation_size(1, 1);
        // PSD (64) + own state (8) + no team + history (8) = 80
        assert_eq!(jammer_size, 64 + 8 + 8);

        // 2 jammers - includes 1 teammate
        let jammer_size_2 = config.jammer_observation_size(2, 1);
        // PSD (64) + own state (8) + 1 teammate (6) + history (8) = 86
        assert_eq!(jammer_size_2, 64 + 8 + 6 + 8);

        // CR observation
        let cr_size = config.cr_observation_size(1, 1);
        // PSD (64) + own state (10) + no team + history (8) = 82
        assert_eq!(cr_size, 64 + 10 + 8);
    }

    #[test]
    fn test_observation_indexing() {
        let obs = MultiAgentObservations::new(4, 2, 3, 100, 120);

        // Check jammer observation slices
        assert_eq!(obs.jammer_observation(0, 0).len(), 100);
        assert_eq!(obs.jammer_observation(0, 1).len(), 100);
        assert_eq!(obs.jammer_observation(3, 1).len(), 100);

        // Check CR observation slices
        assert_eq!(obs.cr_observation(0, 0).len(), 120);
        assert_eq!(obs.cr_observation(2, 2).len(), 120);
    }
}

// ============================================================================
// Reward Tests
// ============================================================================

mod reward_tests {
    use super::*;

    fn setup_test_scenario() -> (MultiAgentState, InterferenceMatrix) {
        let mut state = MultiAgentState::new(1, 1, 1, 4);

        // Set up jammer
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_bandwidth[j_idx] = 20e6;
        state.jammer_power[j_idx] = 20.0;

        // Set up CR
        let c_idx = state.cr_idx(0, 0);
        state.cr_freq[c_idx] = 2.45e9; // Same frequency
        state.cr_bandwidth[c_idx] = 10e6;
        state.cr_power[c_idx] = 10.0;
        state.cr_throughput[c_idx] = 0.0; // Jammed
        state.cr_alive[c_idx] = 1;

        // Create interference matrix
        let mut interference = InterferenceMatrix::new(1, 1);
        interference.set_jammed(0, 0, true);
        interference.set_jammer_interference(0, 0, -50.0);
        interference.cr_sinr[0] = 5.0; // Low SINR

        (state, interference)
    }

    #[test]
    fn test_sinr_calculation() {
        use crate::agents::interference::calculate_sinr_db;

        // Signal at -50 dBm, no interference, noise at -100 dBm
        let sinr = calculate_sinr_db(-50.0, &[], -100.0);
        // Should be about 50 dB
        assert!((sinr - 50.0).abs() < 1.0, "SINR = {}", sinr);

        // Signal at -50 dBm, interference at -50 dBm, noise at -100 dBm
        let sinr2 = calculate_sinr_db(-50.0, &[-50.0], -100.0);
        // Signal and interference equal, so SINR should be about 0 dB
        assert!((sinr2 - 0.0).abs() < 1.0, "SINR = {}", sinr2);
    }

    #[test]
    fn test_jammer_reward_disruption() {
        let (state, interference) = setup_test_scenario();
        let config = JammerRewardConfig::default();

        let reward = compute_jammer_reward(&state, &interference, 0, 0, &config);

        // Jammer successfully jams CR, should have positive reward
        assert!(reward > 0.0, "Jammer reward = {}", reward);
    }

    #[test]
    fn test_cr_reward_throughput() {
        let (mut state, mut interference) = setup_test_scenario();

        // Set high throughput for CR
        let c_idx = state.cr_idx(0, 0);
        state.cr_throughput[c_idx] = 0.9;
        interference.cr_sinr[0] = 25.0; // Good SINR

        // Not jammed
        interference.set_jammed(0, 0, false);

        let config = CRRewardConfig::default();
        let reward = compute_cr_reward(&state, &interference, 0, 0, &config);

        // High throughput, no collision should give positive reward
        assert!(reward > 0.0, "CR reward = {}", reward);
    }

    #[test]
    fn test_cr_reward_collision_penalty() {
        let (state, interference) = setup_test_scenario();
        let config = CRRewardConfig::default();

        let reward_jammed = compute_cr_reward(&state, &interference, 0, 0, &config);

        // Create unjammed scenario
        let mut interference_clear = InterferenceMatrix::new(1, 1);
        interference_clear.cr_sinr[0] = 25.0;

        let mut state_clear = state.clone();
        let c_idx = state_clear.cr_idx(0, 0);
        state_clear.cr_throughput[c_idx] = 0.8;

        let reward_clear = compute_cr_reward(&state_clear, &interference_clear, 0, 0, &config);

        // Unjammed CR should have higher reward
        assert!(reward_clear > reward_jammed, "clear={}, jammed={}", reward_clear, reward_jammed);
    }

    #[test]
    fn test_power_penalty() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);
        let interference = InterferenceMatrix::new(1, 1);

        // High power jammer
        let config_high_penalty = JammerRewardConfig {
            power_penalty_weight: 0.5,
            disruption_weight: 0.0,
            target_match_weight: 0.0,
            shutdown_bonus: 0.0,
            collateral_penalty_weight: 0.0,
            max_power_dbm: 30.0,
        };

        let j_idx = state.jammer_idx(0, 0);
        state.jammer_power[j_idx] = 30.0; // Max power

        let reward_high = compute_jammer_reward(&state, &interference, 0, 0, &config_high_penalty);

        state.jammer_power[j_idx] = 0.0; // Zero power
        let reward_low = compute_jammer_reward(&state, &interference, 0, 0, &config_high_penalty);

        // Low power should give higher reward (less penalty)
        assert!(reward_low > reward_high, "low={}, high={}", reward_low, reward_high);
    }

    #[test]
    fn test_multi_agent_rewards_compute() {
        let (state, interference) = setup_test_scenario();
        let config = RewardConfig::default();

        let matrices = vec![interference];
        let rewards = MultiAgentRewards::compute(&state, &matrices, &config);

        assert_eq!(rewards.num_envs, 1);
        assert_eq!(rewards.num_jammers, 1);
        assert_eq!(rewards.num_crs, 1);

        let j_reward = rewards.jammer_reward(0, 0);
        let c_reward = rewards.cr_reward(0, 0);

        assert!(j_reward.is_finite());
        assert!(c_reward.is_finite());
    }

    #[test]
    fn test_rewards_flatten() {
        let mut rewards = MultiAgentRewards::new(2, 2, 3);

        rewards.set_jammer_reward(0, 0, 1.0);
        rewards.set_jammer_reward(0, 1, 2.0);
        rewards.set_cr_reward(0, 0, 3.0);
        rewards.set_cr_reward(0, 1, 4.0);
        rewards.set_cr_reward(0, 2, 5.0);

        let flat = rewards.to_flat();

        // Layout: [J00, J01, CR00, CR01, CR02, J10, J11, CR10, CR11, CR12]
        assert!((flat[0] - 1.0).abs() < 1e-6);
        assert!((flat[1] - 2.0).abs() < 1e-6);
        assert!((flat[2] - 3.0).abs() < 1e-6);
        assert!((flat[3] - 4.0).abs() < 1e-6);
        assert!((flat[4] - 5.0).abs() < 1e-6);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_agent_pipeline() {
        // Create agent state
        let mut state = MultiAgentState::new(8, 2, 4, 8);
        let config = ActionConfig::default();

        // Create and apply actions
        let mut actions = MultiAgentActions::new(8, 2, 4);

        for env in 0..8 {
            // Set random-ish actions
            for j in 0..2 {
                actions.set_jammer_action(
                    env,
                    j,
                    JammerAction::new(
                        (env as f32 + j as f32) / 10.0,
                        0.3,
                        0.5,
                        j as f32 / 4.0,
                    ),
                );
            }
            for c in 0..4 {
                actions.set_cr_action(
                    env,
                    c,
                    CognitiveRadioAction::new(
                        (env as f32 + c as f32) / 12.0,
                        0.4,
                        0.2,
                    ),
                );
            }
        }

        // Apply actions to state
        for env in 0..8 {
            for j in 0..2 {
                state.apply_jammer_action(env, j, actions.jammer_action(env, j), &config);
            }
            for c in 0..4 {
                state.apply_cr_action(env, c, actions.cr_action(env, c), &config);
            }
        }

        // Verify state was updated
        for env in 0..8 {
            for j in 0..2 {
                let j_idx = state.jammer_idx(env, j);
                assert!(state.jammer_freq[j_idx] > 0.0);
                assert!(state.jammer_bandwidth[j_idx] > 0.0);
            }
            for c in 0..4 {
                let c_idx = state.cr_idx(env, c);
                assert!(state.cr_freq[c_idx] > 0.0);
            }
        }

        // Compute interference for each environment
        let matrices: Vec<_> = (0..8)
            .map(|env| InterferenceMatrix::compute(&state, env, -100.0, 10.0))
            .collect();

        // Compute rewards
        let reward_config = RewardConfig::default();
        let rewards = MultiAgentRewards::compute(&state, &matrices, &reward_config);

        // Verify rewards computed
        for env in 0..8 {
            for j in 0..2 {
                assert!(rewards.jammer_reward(env, j).is_finite());
            }
            for c in 0..4 {
                assert!(rewards.cr_reward(env, c).is_finite());
            }
        }
    }

    #[test]
    fn test_jammer_disrupts_cr() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);

        // Place jammer directly on CR's frequency
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_bandwidth[j_idx] = 20e6;
        state.jammer_power[j_idx] = 30.0;
        state.jammer_x[j_idx] = 0.0;
        state.jammer_y[j_idx] = 0.0;

        let c_idx = state.cr_idx(0, 0);
        state.cr_freq[c_idx] = 2.45e9;
        state.cr_bandwidth[c_idx] = 10e6;
        state.cr_power[c_idx] = 10.0;
        state.cr_x[c_idx] = 100.0; // 100m away
        state.cr_y[c_idx] = 0.0;

        // Compute interference
        let matrix = InterferenceMatrix::compute(&state, 0, -100.0, 10.0);

        // CR should have low SINR due to jammer
        let sinr = matrix.sinr(0);
        assert!(sinr < 10.0, "SINR should be low when jammed: {}", sinr);
    }

    #[test]
    fn test_cr_avoids_jammer() {
        let mut state = MultiAgentState::new(1, 1, 1, 4);

        // Jammer at 2.45 GHz
        let j_idx = state.jammer_idx(0, 0);
        state.jammer_freq[j_idx] = 2.45e9;
        state.jammer_bandwidth[j_idx] = 20e6;
        state.jammer_power[j_idx] = 30.0;

        let c_idx = state.cr_idx(0, 0);
        state.cr_power[c_idx] = 10.0;
        state.cr_bandwidth[c_idx] = 10e6;

        // CR on jammer frequency
        state.cr_freq[c_idx] = 2.45e9;
        let matrix_jammed = InterferenceMatrix::compute(&state, 0, -100.0, 10.0);
        let sinr_jammed = matrix_jammed.sinr(0);

        // CR avoids to 2.48 GHz (outside jammer bandwidth)
        state.cr_freq[c_idx] = 2.48e9;
        let matrix_clear = InterferenceMatrix::compute(&state, 0, -100.0, 10.0);
        let sinr_clear = matrix_clear.sinr(0);

        // SINR should be higher when CR avoids jammer
        assert!(sinr_clear > sinr_jammed, "clear={}, jammed={}", sinr_clear, sinr_jammed);
    }

    #[test]
    fn test_reset_preserves_config() {
        let config = MultiAgentConfig::new()
            .with_jammers(3)
            .with_crs(5)
            .with_history_length(16);

        let mut state = config.create_state(8);

        // Modify state
        for env in 0..8 {
            for j in 0..3 {
                let j_idx = state.jammer_idx(env, j);
                state.jammer_freq[j_idx] = 2.4e9 + (j as f32) * 1e7;
            }
        }

        // Reset
        state.reset_all();

        // Config should be preserved
        assert_eq!(state.num_envs, 8);
        assert_eq!(state.num_jammers, 3);
        assert_eq!(state.num_crs, 5);
        assert_eq!(state.history_length, 16);

        // State should be reset
        for env in 0..8 {
            for j in 0..3 {
                let j_idx = state.jammer_idx(env, j);
                assert!((state.jammer_freq[j_idx]).abs() < 1e-6);
            }
        }
    }
}
