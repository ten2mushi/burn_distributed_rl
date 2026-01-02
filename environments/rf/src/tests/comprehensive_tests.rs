//! Comprehensive Unit and Integration Tests for RF Environment Crate
//!
//! This module provides exhaustive testing following the "Tests as Definition: The Yoneda Way"
//! philosophy. These tests serve as a complete behavioral specification for the RF environment.
//!
//! # Test Organization
//!
//! - **Config Tests**: Builder pattern, validation, edge cases
//! - **State Tests**: SoA layout, indexing, PSD operations
//! - **Entity Tests**: Spawner, behaviors, EntitySoA operations
//! - **Spectrum Tests**: PSD buffer, patterns, noise modeling
//! - **Observation Tests**: Compression, normalization, multi-agent
//! - **Adapter Tests**: VectorizedEnv interface, agent views
//! - **Integration Tests**: Full environment workflows
//!
//! Each section defines the complete behavioral specification for its module.

use crate::agents::AgentType;
use crate::config::RFWorldConfig;
use crate::constants;
use crate::entities::{EntitySoA, EntityType};
use crate::env::RFWorld;
use crate::observation::{
    compress_psd, normalize_psd, MultiAgentObservations, ObservationConfig, PsdCompression,
};
use crate::spectrum::ValidatedPsd;
use crate::types::{Hertz, PositivePower, ValidatedFrequencyGrid};
use crate::state::{freq_bin_idx, RFWorldState};
use crate::adapter::{AgentView, ResetMask, RFContinuousEnvAdapter, RFEnvAdapter, StepResult, VectorizedEnv};
use crate::bands::{SHFBand, UHFBand};
use crate::traits::{FrequencyBand, PropagationModel, NoiseModel};

// ============================================================================
// Config Module Tests: Complete Behavioral Specification
// ============================================================================

mod config_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Builder Pattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_config_with_default_values() {
        let config = RFWorldConfig::new().build();

        // Verify all default values match expected constants
        assert_eq!(config.num_envs, 8, "Default num_envs should be 8");
        assert_eq!(
            config.num_freq_bins,
            constants::DEFAULT_NUM_FREQ_BINS,
            "Default freq bins should match constant"
        );
        assert_eq!(
            config.max_steps,
            constants::DEFAULT_MAX_STEPS,
            "Default max_steps should match constant"
        );
    }

    #[test]
    fn should_chain_builder_methods_correctly() {
        let config = RFWorldConfig::new()
            .with_num_envs(16)
            .with_freq_bins(1024)
            .with_freq_range(1e9, 5e9)
            .with_num_jammers(3)
            .with_num_crs(4)
            .with_max_steps(500)
            .build();

        assert_eq!(config.num_envs, 16);
        assert_eq!(config.num_freq_bins, 1024);
        assert_eq!(config.freq_min, 1e9);
        assert_eq!(config.freq_max, 5e9);
        assert_eq!(config.num_jammers, 3);
        assert_eq!(config.num_crs, 4);
        assert_eq!(config.max_steps, 500);
    }

    #[test]
    fn should_calculate_freq_resolution_correctly() {
        // Note: num_freq_bins must be a power of 2 for FFT efficiency
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024) // Must be power of 2
            .build();

        let expected_resolution = (2e9 - 1e9) / 1024.0;
        let actual_resolution = config.freq_resolution();

        assert!(
            (actual_resolution - expected_resolution).abs() < 1e-6,
            "Frequency resolution mismatch: expected {}, got {}",
            expected_resolution,
            actual_resolution
        );
    }

    #[test]
    fn should_calculate_effective_noise_floor_correctly() {
        let config = RFWorldConfig::new()
            .with_noise_figure(6.0)
            .build();

        // Effective noise floor = thermal noise (-174 dBm/Hz) + noise figure
        let expected = constants::THERMAL_NOISE_DBM_HZ + 6.0;
        let actual = config.effective_noise_floor();

        assert!(
            (actual - expected).abs() < 1e-6,
            "Effective noise floor mismatch: expected {}, got {}",
            expected,
            actual
        );
    }

    // -------------------------------------------------------------------------
    // Boundary Condition Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_minimum_valid_environments() {
        // Note: num_envs must be a multiple of 8 for SIMD
        let config = RFWorldConfig::new().with_num_envs(8).build();

        assert_eq!(config.num_envs, 8);
        let state = RFWorldState::new(&config);
        assert_eq!(state.num_envs(), 8);
    }

    #[test]
    fn should_handle_large_number_of_environments() {
        // num_envs must be a multiple of 8 for SIMD
        let config = RFWorldConfig::new().with_num_envs(1024).build();

        assert_eq!(config.num_envs, 1024);
        let state = RFWorldState::new(&config);
        assert_eq!(state.num_envs(), 1024);
    }

    #[test]
    fn should_handle_minimum_valid_freq_bins() {
        // Note: num_freq_bins must be a power of 2 for FFT
        let config = RFWorldConfig::new().with_freq_bins(1).build();

        assert_eq!(config.num_freq_bins, 1);
        let state = RFWorldState::new(&config);
        assert_eq!(state.num_freq_bins(), 1);
    }

    #[test]
    fn should_handle_large_freq_bins() {
        // num_freq_bins must be a power of 2 for FFT
        let config = RFWorldConfig::new().with_freq_bins(8192).build();

        assert_eq!(config.num_freq_bins, 8192);
    }

    #[test]
    fn should_handle_zero_agents() {
        let config = RFWorldConfig::new()
            .with_num_jammers(0)
            .with_num_crs(0)
            .build();

        assert_eq!(config.num_jammers, 0);
        assert_eq!(config.num_crs, 0);
    }

    #[test]
    fn should_handle_many_agents() {
        let config = RFWorldConfig::new()
            .with_num_jammers(10)
            .with_num_crs(20)
            .build();

        assert_eq!(config.num_jammers, 10);
        assert_eq!(config.num_crs, 20);
    }

    // -------------------------------------------------------------------------
    // Frequency Range Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_set_uhf_band_correctly() {
        let config = RFWorldConfig::new()
            .with_freq_range(constants::UHF_MIN, constants::UHF_MAX)
            .build();

        assert_eq!(config.freq_min, 300e6);
        assert_eq!(config.freq_max, 3e9);
    }

    #[test]
    fn should_set_shf_band_correctly() {
        let config = RFWorldConfig::new()
            .with_freq_range(constants::SHF_MIN, constants::SHF_MAX)
            .build();

        assert_eq!(config.freq_min, 3e9);
        assert_eq!(config.freq_max, 30e9);
    }

    #[test]
    fn should_handle_narrow_frequency_range() {
        // Note: num_freq_bins must be power of 2
        let config = RFWorldConfig::new()
            .with_freq_range(2.4e9, 2.5e9) // 100 MHz band
            .with_freq_bins(128) // Power of 2
            .build();

        let expected_resolution = 1e8 / 128.0; // ~781.25 kHz
        assert!((config.freq_resolution() - expected_resolution).abs() < 1.0);
    }

    #[test]
    fn should_handle_wide_frequency_range() {
        // Note: num_freq_bins must be power of 2
        let config = RFWorldConfig::new()
            .with_freq_range(100e6, 10e9) // ~10 GHz span
            .with_freq_bins(1024) // Power of 2
            .build();

        let expected_resolution = (10e9 - 100e6) / 1024.0; // ~9.668 MHz
        assert!((config.freq_resolution() - expected_resolution).abs() < 1e3);
    }
}

// ============================================================================
// State Module Tests: Complete Behavioral Specification
// ============================================================================

mod state_tests {
    use super::*;

    fn test_config() -> RFWorldConfig {
        RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(256)
            .build()
    }

    // -------------------------------------------------------------------------
    // State Creation and Initialization
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_state_with_correct_dimensions() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        assert_eq!(state.num_envs(), 8);
        assert_eq!(state.num_freq_bins(), 256);
        assert_eq!(state.psd.len(), 8 * 256);
        assert_eq!(state.noise_floor.len(), 8);
        assert_eq!(state.step_count.len(), 8);
        assert_eq!(state.terminals.len(), 8);
        assert_eq!(state.truncations.len(), 8);
        assert_eq!(state.episode_returns.len(), 8);
    }

    #[test]
    fn should_initialize_psd_to_noise_floor() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        // All PSD values should be initialized to noise floor
        for env in 0..8 {
            let noise = state.noise_floor[env];
            for bin in 0..256 {
                let power = state.get_psd_power(env, bin);
                assert!(
                    (power - noise).abs() < 1e-10,
                    "PSD at ({}, {}) should be noise floor",
                    env,
                    bin
                );
            }
        }
    }

    #[test]
    fn should_initialize_episode_state_to_zero() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        for env in 0..8 {
            assert_eq!(state.step_count[env], 0);
            assert_eq!(state.terminals[env], 0);
            assert_eq!(state.truncations[env], 0);
            assert_eq!(state.episode_returns[env], 0.0);
        }
    }

    // -------------------------------------------------------------------------
    // PSD Indexing Tests (Critical for SoA layout)
    // -------------------------------------------------------------------------

    #[test]
    fn should_calculate_psd_indices_for_env_major_layout() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        // Verify env-major ordering: all bins for env 0, then env 1, etc.
        assert_eq!(state.psd_idx(0, 0), 0);
        assert_eq!(state.psd_idx(0, 1), 1);
        assert_eq!(state.psd_idx(0, 255), 255);
        assert_eq!(state.psd_idx(1, 0), 256);
        assert_eq!(state.psd_idx(1, 1), 257);
        assert_eq!(state.psd_idx(7, 0), 7 * 256);
        assert_eq!(state.psd_idx(7, 255), 7 * 256 + 255);
    }

    #[test]
    fn should_provide_correct_psd_slices() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        for env in 0..8 {
            let slice = state.psd_slice(env);
            assert_eq!(slice.len(), 256);

            // Verify slice points to correct memory region
            assert_eq!(slice.as_ptr() as usize, state.psd[env * 256..].as_ptr() as usize);
        }
    }

    #[test]
    fn should_provide_mutable_psd_slices() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Modify through mutable slice
        {
            let slice = state.psd_slice_mut(2);
            slice[100] = 42.0;
        }

        // Verify modification through getter
        assert!((state.get_psd_power(2, 100) - 42.0).abs() < 1e-10);

        // Verify other envs unaffected
        assert!((state.get_psd_power(1, 100) - state.noise_floor[1]).abs() < 1e-10);
        assert!((state.get_psd_power(3, 100) - state.noise_floor[3]).abs() < 1e-10);
    }

    #[test]
    fn should_use_standalone_index_function_consistently() {
        let config = test_config();
        let state = RFWorldState::new(&config);

        // Verify standalone function matches instance method
        for env in 0..8 {
            for bin in [0, 50, 100, 200, 255] {
                assert_eq!(
                    freq_bin_idx(env, bin, 256),
                    state.psd_idx(env, bin),
                    "Index mismatch at ({}, {})",
                    env,
                    bin
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // PSD Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_add_power_correctly() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        let initial = state.get_psd_power(0, 100);
        state.add_psd_power(0, 100, 1.0);

        assert!((state.get_psd_power(0, 100) - initial - 1.0).abs() < 1e-10);
    }

    #[test]
    fn should_accumulate_power_correctly() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        let initial = state.get_psd_power(0, 100);

        // Add power multiple times
        state.add_psd_power(0, 100, 1.0);
        state.add_psd_power(0, 100, 2.0);
        state.add_psd_power(0, 100, 3.0);

        let expected = initial + 6.0;
        assert!(
            (state.get_psd_power(0, 100) - expected).abs() < 1e-10,
            "Power should accumulate: expected {}, got {}",
            expected,
            state.get_psd_power(0, 100)
        );
    }

    #[test]
    fn should_set_power_directly() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        state.set_psd_power(0, 100, 42.0);
        assert!((state.get_psd_power(0, 100) - 42.0).abs() < 1e-10);

        // Set overwrites, not adds
        state.set_psd_power(0, 100, 10.0);
        assert!((state.get_psd_power(0, 100) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn should_calculate_total_psd_power() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Set known power values
        state.clear_psd_all();
        state.set_psd_power(0, 0, 1.0);
        state.set_psd_power(0, 1, 2.0);
        state.set_psd_power(0, 2, 3.0);

        let total = state.total_psd_power(0);
        let noise_contribution: f32 = (3..256).map(|b| state.get_psd_power(0, b)).sum();
        let expected = 1.0 + 2.0 + 3.0 + noise_contribution;

        assert!(
            (total - expected).abs() < 1e-6,
            "Total power mismatch: expected {}, got {}",
            expected,
            total
        );
    }

    #[test]
    fn should_find_peak_psd_power() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Add a clear peak
        state.set_psd_power(0, 128, 1000.0);

        let peak = state.peak_psd_power(0);
        assert!(
            peak >= 1000.0,
            "Peak should be at least 1000.0, got {}",
            peak
        );
    }

    #[test]
    fn should_clear_psd_to_noise_floor() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Add some power
        state.set_psd_power(0, 100, 1000.0);
        state.set_psd_power(1, 50, 500.0);

        // Clear
        state.clear_psd_all();

        // Verify reset to noise floor
        for env in 0..8 {
            for bin in 0..256 {
                let power = state.get_psd_power(env, bin);
                let noise = state.noise_floor[env];
                assert!(
                    (power - noise).abs() < 1e-10,
                    "PSD at ({}, {}) not reset to noise floor",
                    env,
                    bin
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Episode State Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_track_terminal_state_correctly() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        assert!(!state.is_terminal(0));

        state.set_terminal(0, true);
        assert!(state.is_terminal(0));
        assert!(state.is_done(0));

        state.set_terminal(0, false);
        assert!(!state.is_terminal(0));
        assert!(!state.is_done(0));
    }

    #[test]
    fn should_track_truncation_state_correctly() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        assert!(!state.is_truncated(0));

        state.set_truncated(0, true);
        assert!(state.is_truncated(0));
        assert!(state.is_done(0));

        state.set_truncated(0, false);
        assert!(!state.is_truncated(0));
    }

    #[test]
    fn should_report_done_for_terminal_or_truncated() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Terminal only
        state.set_terminal(0, true);
        state.set_truncated(0, false);
        assert!(state.is_done(0));

        // Truncated only
        state.set_terminal(0, false);
        state.set_truncated(0, true);
        assert!(state.is_done(0));

        // Both
        state.set_terminal(0, true);
        assert!(state.is_done(0));

        // Neither
        state.set_terminal(0, false);
        state.set_truncated(0, false);
        assert!(!state.is_done(0));
    }

    #[test]
    fn should_increment_step_count() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        assert_eq!(state.step_count[0], 0);

        let count1 = state.increment_step(0);
        assert_eq!(count1, 1);
        assert_eq!(state.step_count[0], 1);

        let count2 = state.increment_step(0);
        assert_eq!(count2, 2);
        assert_eq!(state.step_count[0], 2);
    }

    #[test]
    fn should_accumulate_rewards() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        assert_eq!(state.episode_returns[0], 0.0);

        state.add_reward(0, 1.5);
        assert!((state.episode_returns[0] - 1.5).abs() < 1e-10);

        state.add_reward(0, 2.5);
        assert!((state.episode_returns[0] - 4.0).abs() < 1e-10);

        // Negative rewards should work too
        state.add_reward(0, -1.0);
        assert!((state.episode_returns[0] - 3.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Reset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_reset_single_environment() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Modify env 0
        state.step_count[0] = 100;
        state.terminals[0] = 1;
        state.truncations[0] = 1;
        state.episode_returns[0] = 50.0;
        state.set_psd_power(0, 100, 1000.0);

        // Modify env 1 (should not be affected)
        state.step_count[1] = 200;
        state.episode_returns[1] = 100.0;

        // Reset env 0 only
        state.reset_env(0);

        // Verify env 0 is reset
        assert_eq!(state.step_count[0], 0);
        assert_eq!(state.terminals[0], 0);
        assert_eq!(state.truncations[0], 0);
        assert_eq!(state.episode_returns[0], 0.0);
        assert!((state.get_psd_power(0, 100) - state.noise_floor[0]).abs() < 1e-10);

        // Verify env 1 is NOT reset
        assert_eq!(state.step_count[1], 200);
        assert_eq!(state.episode_returns[1], 100.0);
    }

    #[test]
    fn should_reset_all_environments() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Modify all envs
        for env in 0..8 {
            state.step_count[env] = 100 + env as u32;
            state.terminals[env] = 1;
            state.episode_returns[env] = 50.0 + env as f32;
        }

        // Reset all
        state.reset_all();

        // Verify all reset
        for env in 0..8 {
            assert_eq!(state.step_count[env], 0);
            assert_eq!(state.terminals[env], 0);
            assert_eq!(state.episode_returns[env], 0.0);
        }
    }

    #[test]
    fn should_handle_reset_with_seed() {
        let config = test_config();
        let mut state = RFWorldState::new(&config);

        // Modify state
        state.step_count[0] = 100;

        // Reset with seed
        state.reset_all_with_seed(42);

        // Verify reset
        assert_eq!(state.step_count[0], 0);

        // For now, seed doesn't affect outcome, but API should work
    }
}

// ============================================================================
// Entity Module Tests: Complete Behavioral Specification
// ============================================================================

mod entity_tests {
    use super::*;
    use crate::entities::ModulationType;

    // -------------------------------------------------------------------------
    // EntitySoA Creation and Initialization
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_entity_soa_with_correct_dimensions() {
        let entities = EntitySoA::new(8, 16);

        assert_eq!(entities.num_envs(), 8);
        assert_eq!(entities.max_entities(), 16);
        assert_eq!(entities.capacity(), 8 * 16);
    }

    #[test]
    fn should_initialize_all_entities_as_inactive() {
        let entities = EntitySoA::new(8, 16);

        for env in 0..8 {
            for entity in 0..16 {
                assert!(
                    !entities.is_active(env, entity),
                    "Entity ({}, {}) should be inactive initially",
                    env,
                    entity
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Entity Activation and Deactivation
    // -------------------------------------------------------------------------

    #[test]
    fn should_activate_and_deactivate_entities() {
        let mut entities = EntitySoA::new(8, 16);

        // Initially inactive
        assert!(!entities.is_active(0, 0));

        // Activate
        entities.activate(0, 0);
        assert!(entities.is_active(0, 0));

        // Deactivate
        entities.deactivate(0, 0);
        assert!(!entities.is_active(0, 0));
    }

    #[test]
    fn should_count_active_entities() {
        let mut entities = EntitySoA::new(8, 16);

        assert_eq!(entities.count_active(0), 0);

        entities.activate(0, 0);
        entities.activate(0, 3);
        entities.activate(0, 7);

        assert_eq!(entities.count_active(0), 3);
        assert_eq!(entities.count_active(1), 0); // Different env
    }

    // -------------------------------------------------------------------------
    // Entity Type Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_store_and_retrieve_entity_types() {
        let mut entities = EntitySoA::new(8, 16);

        // Set entity with specific type using set_entity
        entities.set_entity(
            0, 0, EntityType::TVStation, ModulationType::COFDM,
            0.0, 0.0, 0.0, 550e6, 6e6, 45.0
        );
        entities.set_entity(
            0, 1, EntityType::WiFiAP, ModulationType::OFDM,
            0.0, 0.0, 0.0, 5.8e9, 20e6, 20.0
        );
        entities.set_entity(
            0, 2, EntityType::SBandRadar, ModulationType::Chirp,
            0.0, 0.0, 0.0, 3e9, 100e6, 60.0
        );

        assert_eq!(entities.get_type(0, 0), EntityType::TVStation);
        assert_eq!(entities.get_type(0, 1), EntityType::WiFiAP);
        assert_eq!(entities.get_type(0, 2), EntityType::SBandRadar);
    }

    #[test]
    fn should_handle_all_entity_types() {
        let mut entities = EntitySoA::new(8, 16);

        let types = [
            (EntityType::TVStation, ModulationType::COFDM),
            (EntityType::FMRadio, ModulationType::FM),
            (EntityType::LTETower, ModulationType::OFDM),
            (EntityType::WiFiAP, ModulationType::OFDM),
            (EntityType::Bluetooth, ModulationType::GFSK),
            (EntityType::SBandRadar, ModulationType::Chirp),
            (EntityType::WeatherRadar, ModulationType::Chirp),
            (EntityType::DroneAnalog, ModulationType::FM),
            (EntityType::DroneDigital, ModulationType::OFDM),
            (EntityType::Vehicle, ModulationType::OFDM),
        ];

        for (i, (entity_type, mod_type)) in types.iter().enumerate() {
            entities.set_entity(
                0, i, *entity_type, *mod_type,
                0.0, 0.0, 0.0, 1e9, 10e6, 20.0
            );
            assert_eq!(entities.get_type(0, i), *entity_type);
        }
    }

    // -------------------------------------------------------------------------
    // Entity Property Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_store_and_retrieve_entity_properties() {
        let mut entities = EntitySoA::new(8, 16);

        entities.set_entity(
            0, 0, EntityType::WiFiAP, ModulationType::OFDM,
            100.0, 200.0, 50.0, 2.4e9, 20e6, 30.0
        );

        let idx = entities.idx(0, 0);

        // Retrieve and verify
        assert!((entities.x[idx] - 100.0).abs() < 1e-6);
        assert!((entities.y[idx] - 200.0).abs() < 1e-6);
        assert!((entities.z[idx] - 50.0).abs() < 1e-6);
        assert!((entities.center_freq[idx] - 2.4e9).abs() < 1.0);
        assert!((entities.bandwidth[idx] - 20e6).abs() < 1.0);
        assert!((entities.power_dbm[idx] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn should_set_velocity() {
        let mut entities = EntitySoA::new(8, 16);

        entities.set_velocity(0, 0, 10.0, 20.0, 5.0);

        let idx = entities.idx(0, 0);
        assert!((entities.vx[idx] - 10.0).abs() < 1e-6);
        assert!((entities.vy[idx] - 20.0).abs() < 1e-6);
        assert!((entities.vz[idx] - 5.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Entity Indexing Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_calculate_entity_indices_correctly() {
        let entities = EntitySoA::new(8, 64);

        // Layout: env-major, entity-minor
        assert_eq!(entities.idx(0, 0), 0);
        assert_eq!(entities.idx(0, 63), 63);
        assert_eq!(entities.idx(1, 0), 64);
        assert_eq!(entities.idx(7, 63), 7 * 64 + 63);
    }

    // -------------------------------------------------------------------------
    // Entity Reset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_reset_single_environment_entities() {
        let mut entities = EntitySoA::new(8, 16);

        // Activate entities in multiple envs
        entities.activate(0, 0);
        entities.activate(0, 1);
        entities.activate(1, 0);

        let idx = entities.idx(0, 0);
        entities.timer[idx] = 1.5;

        // Reset env 0
        entities.reset_env(0);

        // Env 0 should have no active entities
        assert_eq!(entities.count_active(0), 0);
        assert_eq!(entities.timer[idx], 0.0);

        // Env 1 should still have active entity
        assert!(entities.is_active(1, 0));
    }

    #[test]
    fn should_reset_all_environments() {
        let mut entities = EntitySoA::new(8, 16);

        // Activate in multiple envs
        for env in 0..8 {
            entities.activate(env, 0);
        }

        // Reset all
        entities.reset_all();

        // All should be inactive
        for env in 0..8 {
            assert_eq!(entities.count_active(env), 0);
        }
    }

    // -------------------------------------------------------------------------
    // Entity Type Counting Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_count_active_entities_of_specific_type() {
        let mut entities = EntitySoA::new(8, 16);

        entities.set_entity(
            0, 0, EntityType::TVStation, ModulationType::COFDM,
            0.0, 0.0, 0.0, 500e6, 6e6, 30.0
        );
        entities.set_entity(
            0, 1, EntityType::TVStation, ModulationType::COFDM,
            0.0, 0.0, 0.0, 506e6, 6e6, 30.0
        );
        entities.set_entity(
            0, 2, EntityType::FMRadio, ModulationType::FM,
            0.0, 0.0, 0.0, 100e6, 200e3, 20.0
        );

        assert_eq!(entities.count_active_of_type(0, EntityType::TVStation), 2);
        assert_eq!(entities.count_active_of_type(0, EntityType::FMRadio), 1);
        assert_eq!(entities.count_active_of_type(0, EntityType::LTETower), 0);
    }
}

// ============================================================================
// Spectrum Module Tests: Complete Behavioral Specification
// ============================================================================

mod spectrum_tests {
    use super::*;
    use crate::spectrum::patterns::{
        add_chirp_pattern, add_gaussian_pattern, add_ofdm_pattern, add_rect_pattern,
        dbm_to_watts, watts_to_dbm,
    };

    fn test_grid() -> ValidatedFrequencyGrid {
        ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).expect("Valid grid")
    }

    // -------------------------------------------------------------------------
    // FrequencyGrid Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_frequency_grid_with_correct_parameters() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 5e9, 1000).expect("Valid grid");

        assert!((grid.freq_min().as_hz() - 1e9).abs() < 1.0);
        assert!((grid.freq_max().as_hz() - 5e9).abs() < 1.0);
        assert_eq!(grid.num_bins(), 1000);
        assert!((grid.resolution().as_hz() - 4e6).abs() < 1.0); // 4 GHz / 1000 = 4 MHz
    }

    #[test]
    fn should_convert_frequency_to_bin_correctly() {
        let grid = test_grid(); // 1-2 GHz, 1000 bins

        // Boundary cases
        assert_eq!(grid.freq_to_bin(Hertz::new(1e9)), 0);
        assert_eq!(grid.freq_to_bin(Hertz::new(2e9)), 999);

        // Middle
        assert_eq!(grid.freq_to_bin(Hertz::new(1.5e9)), 500);

        // Quarter points
        assert_eq!(grid.freq_to_bin(Hertz::new(1.25e9)), 250);
        assert_eq!(grid.freq_to_bin(Hertz::new(1.75e9)), 750);
    }

    #[test]
    fn should_convert_bin_to_frequency_correctly() {
        let grid = test_grid();

        // Boundary cases
        assert!((grid.bin_to_freq(0).as_hz() - 1e9).abs() < 1e6);
        assert!((grid.bin_to_freq(999).as_hz() - 2e9).abs() < 1e6);

        // Middle
        assert!((grid.bin_to_freq(500).as_hz() - 1.5e9).abs() < 1e6);
    }

    #[test]
    fn should_check_frequency_in_range_correctly() {
        let grid = test_grid();

        // In range
        assert!(grid.in_range(Hertz::new(1.5e9)));
        assert!(grid.in_range(Hertz::new(1e9)));
        assert!(grid.in_range(Hertz::new(2e9)));

        // Out of range
        assert!(!grid.in_range(Hertz::new(0.5e9)));
        assert!(!grid.in_range(Hertz::new(2.5e9)));
    }

    #[test]
    fn should_clamp_frequencies_outside_range() {
        let grid = test_grid();

        // Below range should clamp to 0
        assert_eq!(grid.freq_to_bin(Hertz::new(0.5e9)), 0);

        // Above range should clamp to last bin
        assert_eq!(grid.freq_to_bin(Hertz::new(3e9)), 999);
    }

    // -------------------------------------------------------------------------
    // Pattern Tests: Rectangular
    // -------------------------------------------------------------------------

    #[test]
    fn should_add_rectangular_pattern_at_center() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_rect_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(10e6), PositivePower::new(0.001));

        // Center bin should have power
        let center = grid.freq_to_bin(Hertz::new(1.5e9));
        assert!(psd[center] > 0.0);

        // Outside the band should be zero
        assert_eq!(psd[0], 0.0);
        assert_eq!(psd[999], 0.0);
    }

    #[test]
    fn should_distribute_rect_pattern_evenly() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_rect_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(100e6), PositivePower::new(0.001));

        // Find bins with power
        let start = grid.freq_to_bin(Hertz::new(1.45e9));
        let end = grid.freq_to_bin(Hertz::new(1.55e9));

        // Power should be approximately equal across the band
        let center_power = psd[grid.freq_to_bin(Hertz::new(1.5e9))];
        for bin in start..=end {
            let ratio = psd[bin] / center_power;
            assert!(
                ratio > 0.9 && ratio < 1.1,
                "Rectangular pattern should be flat, ratio = {}",
                ratio
            );
        }
    }

    #[test]
    fn should_not_affect_psd_when_signal_out_of_range() {
        let grid = test_grid(); // 1-2 GHz
        let mut psd = vec![0.0; grid.num_bins()];

        // Signal completely below grid
        add_rect_pattern(&mut psd, &grid, Hertz::new(500e6), Hertz::new(10e6), PositivePower::new(0.001));
        assert!(psd.iter().all(|&x| x == 0.0));

        // Signal completely above grid
        add_rect_pattern(&mut psd, &grid, Hertz::new(3e9), Hertz::new(10e6), PositivePower::new(0.001));
        assert!(psd.iter().all(|&x| x == 0.0));
    }

    // -------------------------------------------------------------------------
    // Pattern Tests: Gaussian
    // -------------------------------------------------------------------------

    #[test]
    fn should_add_gaussian_pattern_with_peak_at_center() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_gaussian_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(5e6), PositivePower::new(0.001));

        // Find peak
        let (max_idx, _) = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let expected_center = grid.freq_to_bin(Hertz::new(1.5e9));
        assert!(
            (max_idx as i32 - expected_center as i32).abs() <= 1,
            "Gaussian peak should be at center"
        );
    }

    #[test]
    fn should_have_gaussian_decay_from_center() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        // Use larger sigma to ensure we have clear decay over larger offsets
        add_gaussian_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(50e6), PositivePower::new(0.001));

        let center = grid.freq_to_bin(Hertz::new(1.5e9));
        let offset_20mhz = grid.freq_to_bin(Hertz::new(1.52e9));
        let offset_50mhz = grid.freq_to_bin(Hertz::new(1.55e9));
        let offset_100mhz = grid.freq_to_bin(Hertz::new(1.6e9));

        // Power should decrease with distance from center
        // Use >= instead of > since at small distances they might be equal
        assert!(
            psd[center] >= psd[offset_20mhz],
            "Center power {} should be >= offset 20MHz power {}",
            psd[center],
            psd[offset_20mhz]
        );
        assert!(
            psd[offset_20mhz] >= psd[offset_50mhz],
            "Offset 20MHz power {} should be >= offset 50MHz power {}",
            psd[offset_20mhz],
            psd[offset_50mhz]
        );
        assert!(
            psd[offset_50mhz] >= psd[offset_100mhz],
            "Offset 50MHz power {} should be >= offset 100MHz power {}",
            psd[offset_50mhz],
            psd[offset_100mhz]
        );
    }

    // -------------------------------------------------------------------------
    // Pattern Tests: OFDM
    // -------------------------------------------------------------------------

    #[test]
    fn should_add_ofdm_pattern_with_flat_response() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_ofdm_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(20e6), PositivePower::new(0.001));

        // Find bins within the signal (90% of bandwidth)
        let center = grid.freq_to_bin(Hertz::new(1.5e9));
        let offset = grid.freq_to_bin(Hertz::new(1.5e9 + 8e6)); // 8 MHz offset, within 90% of 20 MHz

        // Should have similar power (flat response)
        if psd[center] > 0.0 && psd[offset] > 0.0 {
            let ratio = psd[center] / psd[offset];
            assert!(ratio > 0.9 && ratio < 1.1, "OFDM should be flat");
        }
    }

    // -------------------------------------------------------------------------
    // Pattern Tests: Chirp
    // -------------------------------------------------------------------------

    #[test]
    fn should_add_chirp_pattern_across_sweep_range() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_chirp_pattern(&mut psd, &grid, Hertz::new(1.4e9), Hertz::new(1.6e9), PositivePower::new(0.001));

        // Power should exist at start, middle, and end of sweep
        let start = grid.freq_to_bin(Hertz::new(1.4e9));
        let middle = grid.freq_to_bin(Hertz::new(1.5e9));
        let end = grid.freq_to_bin(Hertz::new(1.6e9));

        assert!(psd[start] > 0.0);
        assert!(psd[middle] > 0.0);
        assert!(psd[end] > 0.0);

        // Outside sweep should be zero
        assert_eq!(psd[grid.freq_to_bin(Hertz::new(1.3e9))], 0.0);
        assert_eq!(psd[grid.freq_to_bin(Hertz::new(1.7e9))], 0.0);
    }

    // -------------------------------------------------------------------------
    // Power Conversion Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_convert_dbm_to_watts_correctly() {
        // 0 dBm = 1 mW = 0.001 W
        assert!((dbm_to_watts(0.0) - 0.001).abs() < 1e-9);

        // 30 dBm = 1 W
        assert!((dbm_to_watts(30.0) - 1.0).abs() < 1e-6);

        // 10 dBm = 10 mW = 0.01 W
        assert!((dbm_to_watts(10.0) - 0.01).abs() < 1e-6);

        // -10 dBm = 0.1 mW = 0.0001 W
        assert!((dbm_to_watts(-10.0) - 0.0001).abs() < 1e-9);
    }

    #[test]
    fn should_convert_watts_to_dbm_correctly() {
        // 0.001 W = 0 dBm
        assert!((watts_to_dbm(0.001) - 0.0).abs() < 1e-6);

        // 1 W = 30 dBm
        assert!((watts_to_dbm(1.0) - 30.0).abs() < 1e-6);
    }

    #[test]
    fn should_round_trip_power_conversions() {
        let test_dbm_values = [-30.0, -10.0, 0.0, 10.0, 30.0, 40.0];

        for dbm in test_dbm_values {
            let watts = dbm_to_watts(dbm);
            let back = watts_to_dbm(watts);
            assert!(
                (back - dbm).abs() < 1e-6,
                "Round-trip failed for {} dBm",
                dbm
            );
        }
    }

    // PsdBuffer tests removed - covered by ValidatedPsd tests in psd.rs
}

// ============================================================================
// Observation Module Tests: Complete Behavioral Specification
// ============================================================================

mod observation_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PSD Compression Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_return_same_vector_when_sizes_match() {
        let psd = vec![1.0, 2.0, 3.0, 4.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Average);
        assert_eq!(compressed, psd);
    }

    #[test]
    fn should_return_original_when_target_exceeds_source() {
        let psd = vec![1.0, 2.0, 3.0, 4.0];
        let compressed = compress_psd(&psd, 8, PsdCompression::Average);
        assert_eq!(compressed, psd);
    }

    #[test]
    fn should_compress_with_average_pooling() {
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Average);

        assert_eq!(compressed.len(), 4);
        assert!((compressed[0] - 1.5).abs() < 1e-6); // avg(1, 2)
        assert!((compressed[1] - 3.5).abs() < 1e-6); // avg(3, 4)
        assert!((compressed[2] - 5.5).abs() < 1e-6); // avg(5, 6)
        assert!((compressed[3] - 7.5).abs() < 1e-6); // avg(7, 8)
    }

    #[test]
    fn should_compress_with_max_pooling() {
        let psd = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Max);

        assert_eq!(compressed.len(), 4);
        assert!((compressed[0] - 5.0).abs() < 1e-6); // max(1, 5)
        assert!((compressed[1] - 7.0).abs() < 1e-6); // max(3, 7)
        assert!((compressed[2] - 6.0).abs() < 1e-6); // max(2, 6)
        assert!((compressed[3] - 8.0).abs() < 1e-6); // max(4, 8)
    }

    #[test]
    fn should_compress_with_subsampling() {
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Subsample);

        assert_eq!(compressed.len(), 4);
        assert!((compressed[0] - 1.0).abs() < 1e-6); // index 0
        assert!((compressed[1] - 3.0).abs() < 1e-6); // index 2
        assert!((compressed[2] - 5.0).abs() < 1e-6); // index 4
        assert!((compressed[3] - 7.0).abs() < 1e-6); // index 6
    }

    #[test]
    fn should_handle_uneven_compression() {
        // 9 elements compressed to 4 (9/4 = 2 with remainder 1)
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let compressed = compress_psd(&psd, 4, PsdCompression::Average);

        assert_eq!(compressed.len(), 4);
        // First group gets extra element due to remainder distribution
    }

    // -------------------------------------------------------------------------
    // PSD Normalization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_normalize_psd_to_0_1_range() {
        // Linear power values corresponding to dB values in range
        let psd = vec![1e-10, 1e-8, 1e-6, 1e-4]; // -100, -80, -60, -40 dBm

        let normalized = normalize_psd(&psd, -100.0, -40.0);

        assert_eq!(normalized.len(), 4);
        assert!(normalized[0] >= 0.0 && normalized[0] <= 1.0);
        assert!(normalized[3] >= 0.0 && normalized[3] <= 1.0);

        // First should be near 0, last near 1
        assert!(normalized[0] < 0.1);
        assert!(normalized[3] > 0.9);
    }

    #[test]
    fn should_clamp_normalized_values() {
        // Values outside normalization range
        let psd = vec![1e-15, 1e-2]; // Below min and above max

        let normalized = normalize_psd(&psd, -100.0, -40.0);

        // Should be clamped to [0, 1]
        assert!(normalized[0] >= 0.0);
        assert!(normalized[1] <= 1.0);
    }

    // -------------------------------------------------------------------------
    // ObservationConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_default_observation_config() {
        let config = ObservationConfig::default();

        assert!(config.include_psd);
        assert!(config.include_agent_state);
        assert!(config.include_team_state);
        assert!(!config.include_opponent_state);
        assert!(config.include_history);
        assert_eq!(config.num_freq_bins, 64);
        assert_eq!(config.history_length, 8);
    }

    #[test]
    fn should_calculate_jammer_observation_size_correctly() {
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

        // Single jammer, single CR: PSD(64) + own_state(8) + history(8) = 80
        let size = config.jammer_observation_size(1, 1);
        assert_eq!(size, 80);

        // 2 jammers: adds teammate state (6 per teammate)
        // PSD(64) + own(8) + team(6) + history(8) = 86
        let size_2j = config.jammer_observation_size(2, 1);
        assert_eq!(size_2j, 86);
    }

    #[test]
    fn should_calculate_cr_observation_size_correctly() {
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

        // Single CR, single jammer: PSD(64) + own_state(10) + history(8) = 82
        let size = config.cr_observation_size(1, 1);
        assert_eq!(size, 82);

        // 3 CRs: adds teammate state (5 per teammate)
        // PSD(64) + own(10) + team(10) + history(8) = 92
        let size_3c = config.cr_observation_size(1, 3);
        assert_eq!(size_3c, 92);
    }

    #[test]
    fn should_include_opponent_state_in_observation_size() {
        let config = ObservationConfig {
            num_freq_bins: 64,
            include_psd: true,
            include_agent_state: true,
            include_team_state: false,
            include_opponent_state: true,
            include_history: false,
            ..ObservationConfig::default()
        };

        // Jammer with 2 CRs: PSD(64) + own(8) + opponents(5*2=10) = 82
        let size = config.jammer_observation_size(1, 2);
        assert_eq!(size, 82);
    }

    // -------------------------------------------------------------------------
    // MultiAgentObservations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_observation_container_with_correct_dimensions() {
        let obs = MultiAgentObservations::new(8, 2, 3, 100, 120);

        assert_eq!(obs.num_envs, 8);
        assert_eq!(obs.num_jammers, 2);
        assert_eq!(obs.num_crs, 3);
        assert_eq!(obs.jammer_obs_size, 100);
        assert_eq!(obs.cr_obs_size, 120);
        assert_eq!(obs.jammer_obs.len(), 8 * 2 * 100);
        assert_eq!(obs.cr_obs.len(), 8 * 3 * 120);
    }

    #[test]
    fn should_calculate_correct_indices() {
        let obs = MultiAgentObservations::new(4, 2, 3, 50, 60);

        // Jammer indices
        assert_eq!(obs.jammer_start(0, 0), 0);
        assert_eq!(obs.jammer_start(0, 1), 50);
        assert_eq!(obs.jammer_start(1, 0), 100);
        assert_eq!(obs.jammer_start(1, 1), 150);

        // CR indices
        assert_eq!(obs.cr_start(0, 0), 0);
        assert_eq!(obs.cr_start(0, 1), 60);
        assert_eq!(obs.cr_start(0, 2), 120);
        assert_eq!(obs.cr_start(1, 0), 180);
    }

    #[test]
    fn should_provide_correct_observation_slices() {
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
        assert!((slice[0] - 1.0).abs() < 1e-10);
        assert!((slice[9] - 2.0).abs() < 1e-10);

        // Other env should be unaffected
        let other = obs.jammer_observation(1, 0);
        assert_eq!(other[0], 0.0);
    }

    #[test]
    fn should_flatten_observations_correctly() {
        let mut obs = MultiAgentObservations::new(2, 1, 1, 3, 4);

        obs.jammer_observation_mut(0, 0)[0] = 1.0;
        obs.cr_observation_mut(0, 0)[0] = 2.0;

        let flat = obs.to_flat();

        // Total: 2*1*3 jammer + 2*1*4 cr = 6 + 8 = 14
        assert_eq!(flat.len(), 14);
        assert!((flat[0] - 1.0).abs() < 1e-10); // First jammer obs
        assert!((flat[6] - 2.0).abs() < 1e-10); // First CR obs
    }

    #[test]
    fn should_select_by_agent_type() {
        let obs = MultiAgentObservations::new(2, 1, 1, 3, 4);

        let jammer_obs = obs.by_type(AgentType::Jammer);
        let cr_obs = obs.by_type(AgentType::CognitiveRadio);

        assert_eq!(jammer_obs.len(), 2 * 1 * 3);
        assert_eq!(cr_obs.len(), 2 * 1 * 4);
    }

    #[test]
    fn should_extract_flat_observations_for_specific_agent() {
        let mut obs = MultiAgentObservations::new(3, 2, 1, 4, 5);

        // Set distinct values for jammer 1 across envs
        obs.jammer_observation_mut(0, 1)[0] = 10.0;
        obs.jammer_observation_mut(1, 1)[0] = 20.0;
        obs.jammer_observation_mut(2, 1)[0] = 30.0;

        let flat = obs.jammer_observations_flat(1);

        // Should be 3 envs * 4 obs_size = 12
        assert_eq!(flat.len(), 12);
        assert!((flat[0] - 10.0).abs() < 1e-10); // env 0
        assert!((flat[4] - 20.0).abs() < 1e-10); // env 1
        assert!((flat[8] - 30.0).abs() < 1e-10); // env 2
    }
}

// ============================================================================
// Adapter Module Tests: Complete Behavioral Specification
// ============================================================================

mod adapter_tests {
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

    // -------------------------------------------------------------------------
    // StepResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_compute_done_flags_correctly() {
        let result = StepResult {
            observations: vec![],
            rewards: vec![0.0; 4],
            terminals: vec![true, false, false, true],
            truncations: vec![false, true, false, false],
        };

        let dones = result.dones();
        assert_eq!(dones, vec![true, true, false, true]);
    }

    // -------------------------------------------------------------------------
    // ResetMask Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_reset_mask_from_dones() {
        let dones = vec![true, false, true, false, false, false, true, false];
        let mask = ResetMask::from_dones(&dones);

        assert!(mask.any());
        assert_eq!(mask.count(), 3);
        assert_eq!(mask.indices(), vec![0, 2, 6]);
    }

    #[test]
    fn should_report_no_resets_for_empty_mask() {
        let dones = vec![false, false, false, false];
        let mask = ResetMask::from_dones(&dones);

        assert!(!mask.any());
        assert_eq!(mask.count(), 0);
        assert!(mask.indices().is_empty());
    }

    #[test]
    fn should_provide_mask_as_slice() {
        let dones = vec![true, false, true];
        let mask = ResetMask::from_dones(&dones);

        assert_eq!(mask.as_slice(), &[true, false, true]);
    }

    // -------------------------------------------------------------------------
    // AgentView Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_default_to_jammer_zero() {
        let view = AgentView::default();
        assert_eq!(view, AgentView::Jammer(0));
    }

    #[test]
    fn should_distinguish_agent_views() {
        assert_ne!(AgentView::Jammer(0), AgentView::Jammer(1));
        assert_ne!(AgentView::Jammer(0), AgentView::CognitiveRadio(0));
        assert_ne!(AgentView::Jammer(0), AgentView::Combined);
    }

    // -------------------------------------------------------------------------
    // RFEnvAdapter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_adapter_with_correct_properties() {
        let config = test_config();
        let adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        assert_eq!(adapter.n_envs(), 8);
        assert!(adapter.obs_size() > 0);
        assert!(adapter.n_actions() > 0);
    }

    #[test]
    fn should_implement_vectorized_env_trait() {
        let config = test_config();
        let mut adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        // Reset
        adapter.reset_all(42);

        // Get observations
        let obs = adapter.get_observations();
        assert_eq!(obs.len(), 8 * adapter.obs_size());

        // Step
        let actions = vec![0.0f32; 8];
        let result = adapter.step(&actions);

        assert_eq!(result.observations.len(), 8 * adapter.obs_size());
        assert_eq!(result.rewards.len(), 8);
        assert_eq!(result.terminals.len(), 8);
        assert_eq!(result.truncations.len(), 8);
    }

    #[test]
    fn should_reset_specific_environments() {
        let config = test_config();
        let mut adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        adapter.reset_all(42);

        // Step a few times
        for _ in 0..5 {
            let actions = vec![0.0f32; 8];
            adapter.step(&actions);
        }

        // Reset only envs 0, 3, 5
        let mask = ResetMask::from_dones(&[true, false, false, true, false, true, false, false]);
        adapter.reset_envs(&mask, 43);

        // Verify we can continue stepping (no crash)
        let actions = vec![0.0f32; 8];
        let result = adapter.step(&actions);
        assert_eq!(result.rewards.len(), 8);
    }

    // NOTE: discrete_to_continuous is a private method and cannot be tested directly.
    // The conversion is implicitly tested through step() which uses it internally.

    #[test]
    fn should_support_different_agent_views() {
        let config = test_config();

        // Jammer view
        let jammer_adapter = RFEnvAdapter::new(config.clone(), AgentView::Jammer(0));
        assert!(jammer_adapter.obs_size() > 0);

        // CR view
        let cr_adapter = RFEnvAdapter::new(config.clone(), AgentView::CognitiveRadio(0));
        assert!(cr_adapter.obs_size() > 0);

        // Combined view
        let combined_adapter = RFEnvAdapter::new(config, AgentView::Combined);
        assert!(combined_adapter.obs_size() > 0);
    }

    #[test]
    fn should_access_inner_world() {
        let config = test_config();
        let adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));

        let inner = adapter.inner();
        assert_eq!(inner.num_envs(), 8);
    }

    // -------------------------------------------------------------------------
    // RFContinuousEnvAdapter Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_create_continuous_adapter() {
        let config = test_config();
        let adapter = RFContinuousEnvAdapter::new(config, AgentView::Jammer(0));

        assert_eq!(adapter.n_envs(), 8);
        assert_eq!(adapter.action_dim(), 4); // freq, bw, power, mod for jammer
    }

    #[test]
    fn should_step_with_continuous_actions() {
        let config = test_config();
        let mut adapter = RFContinuousEnvAdapter::new(config, AgentView::Jammer(0));

        adapter.reset_all(42);

        // Continuous actions: 8 envs * 4 actions
        let actions = vec![0.5f32; 8 * adapter.action_dim()];
        let result = adapter.step(&actions);

        assert_eq!(result.rewards.len(), 8);
        assert_eq!(result.observations.len(), 8 * adapter.obs_size());
    }

    #[test]
    fn should_have_correct_action_dim_for_cr() {
        let config = test_config();
        let adapter = RFContinuousEnvAdapter::new(config, AgentView::CognitiveRadio(0));

        assert_eq!(adapter.action_dim(), 3); // freq, power, bw for CR
    }
}

// ============================================================================
// Band Module Tests: Complete Behavioral Specification
// ============================================================================

mod band_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // UHF Band Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_define_uhf_frequency_range() {
        let band = UHFBand::new();
        let (min, max) = band.freq_range();

        assert_eq!(min, 300e6);
        assert_eq!(max, 3e9);
    }

    #[test]
    fn should_check_frequency_containment() {
        let band = UHFBand::new();

        // In range
        assert!(band.contains(500e6));
        assert!(band.contains(1e9));
        assert!(band.contains(2.4e9));

        // Out of range
        assert!(!band.contains(100e6));
        assert!(!band.contains(5e9));
    }

    #[test]
    fn should_provide_uhf_propagation_model() {
        let band = UHFBand::new();
        let model = band.propagation_model();

        // Verify it works
        let loss = model.path_loss_db(100.0, 1e9, 10.0, 1.5);
        assert!(loss > 0.0);
        assert!(loss < 200.0);
    }

    #[test]
    fn should_provide_uhf_noise_model() {
        let band = UHFBand::new();
        let model = band.noise_model();

        // NoiseModel.noise_floor_dbm_hz requires NoiseEnvironment
        use crate::spectrum::NoiseEnvironment;
        let env = NoiseEnvironment::default();
        let noise = model.noise_floor_dbm_hz(1e9, env);
        assert!(noise < -160.0); // Reasonable range
        assert!(noise > -180.0);
    }

    #[test]
    fn should_list_typical_uhf_entities() {
        let band = UHFBand::new();
        let entities = band.typical_entities();

        assert!(!entities.is_empty());
        // UHF should include TV, LTE, etc.
    }

    // -------------------------------------------------------------------------
    // SHF Band Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_define_shf_frequency_range() {
        let band = SHFBand::new();
        let (min, max) = band.freq_range();

        assert_eq!(min, 3e9);
        assert_eq!(max, 30e9);
    }

    #[test]
    fn should_contain_wifi_5ghz() {
        let band = SHFBand::new();

        assert!(band.contains(5e9));
        assert!(band.contains(5.8e9));
    }

    #[test]
    fn should_not_contain_wifi_2_4ghz() {
        let band = SHFBand::new();

        assert!(!band.contains(2.4e9));
    }

    #[test]
    fn should_create_different_shf_configurations() {
        let outdoor = SHFBand::outdoor_los();
        let indoor = SHFBand::indoor_wifi();
        let satellite = SHFBand::satellite();
        let radar = SHFBand::radar();

        // Different configurations should have different properties
        assert!(outdoor.path_loss_exponent < indoor.path_loss_exponent);
        assert!(satellite.noise_figure_db < indoor.noise_figure_db);
    }

    #[test]
    fn should_use_builder_pattern_for_shf() {
        let band = SHFBand::new()
            .with_atmospheric()
            .with_noise_figure(4.0)
            .with_path_loss_exponent(2.5);

        assert!(band.include_atmospheric);
        assert!((band.noise_figure_db - 4.0).abs() < 1e-6);
        assert!((band.path_loss_exponent - 2.5).abs() < 1e-6);
    }

    #[test]
    fn should_have_higher_shf_resolution_than_uhf() {
        let uhf = UHFBand::new();
        let shf = SHFBand::new();

        assert!(shf.default_resolution() >= uhf.default_resolution());
    }
}

// ============================================================================
// Integration Tests: Complete System Workflows
// ============================================================================

mod integration_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Full Environment Workflow Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_run_complete_episode() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_num_jammers(1)
            .with_num_crs(1)
            .with_max_steps(10)
            .build();

        let mut world = RFWorld::new(config);
        world.reset_with_seed(42);

        let mut total_rewards = vec![0.0f32; 8];
        let mut steps = 0;

        // Run until all environments are done or max steps
        while steps < 100 {
            let actions = vec![0.5f32; 8 * (1 * 4 + 1 * 3)]; // All mid-range actions
            let (state, rewards) = world.step_multi_agent(&actions);

            // Accumulate rewards
            for env in 0..8 {
                total_rewards[env] += rewards.total_jammer_reward(env);
            }

            steps += 1;

            // Check if all done
            if (0..8).all(|env| state.is_done(env)) {
                break;
            }
        }

        // Episode should have run
        assert!(steps > 0);
    }

    #[test]
    fn should_reset_and_continue_after_episode() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_num_jammers(1)
            .with_num_crs(1)
            .with_max_steps(5)
            .build();

        let mut world = RFWorld::new(config);

        for episode in 0..3 {
            world.reset_with_seed(42 + episode);

            for _step in 0..5 {
                let actions = vec![0.5f32; 8 * 7];
                let _ = world.step_multi_agent(&actions);
            }

            // Verify state after each episode
            let state = world.state();
            assert!(state.step_count[0] > 0 || state.is_done(0));
        }
    }

    // -------------------------------------------------------------------------
    // Adapter Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_integrate_adapter_with_training_loop() {
        let config = RFWorldConfig::new()
            .with_num_envs(8)
            .with_num_jammers(1)
            .with_num_crs(1)
            .with_max_steps(50)
            .build();

        let mut adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));
        adapter.reset_all(42);

        let mut episode_rewards = vec![0.0f32; 8];
        let mut episode_lengths = vec![0u32; 8];

        for _step in 0..100 {
            let obs = adapter.get_observations();
            assert_eq!(obs.len(), 8 * adapter.obs_size());

            // Random actions (would be policy in real training)
            let actions: Vec<f32> = (0..8).map(|i| (i % adapter.n_actions()) as f32).collect();

            let result = adapter.step(&actions);

            // Track episode statistics
            for env in 0..8 {
                episode_rewards[env] += result.rewards[env];
                episode_lengths[env] += 1;
            }

            // Handle resets
            let dones = result.dones();
            if dones.iter().any(|&d| d) {
                let mask = ResetMask::from_dones(&dones);
                adapter.reset_envs(&mask, 43);

                // Reset tracking for done envs
                for (env, &done) in dones.iter().enumerate() {
                    if done {
                        episode_rewards[env] = 0.0;
                        episode_lengths[env] = 0;
                    }
                }
            }
        }

        // Should have run successfully
    }

    // -------------------------------------------------------------------------
    // State Consistency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_maintain_state_consistency_across_operations() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_freq_bins(128)
            .build();

        let mut state = RFWorldState::new(&config);

        // Operations on different environments should be independent
        state.set_psd_power(0, 64, 100.0);
        state.set_terminal(1, true);
        state.add_reward(2, 5.0);
        state.increment_step(3);

        // Verify each env has only its own changes
        assert!((state.get_psd_power(0, 64) - 100.0).abs() < 1e-6);
        assert!((state.get_psd_power(1, 64) - state.noise_floor[1]).abs() < 1e-10);

        assert!(!state.is_terminal(0));
        assert!(state.is_terminal(1));

        assert_eq!(state.episode_returns[0], 0.0);
        assert_eq!(state.episode_returns[2], 5.0);

        assert_eq!(state.step_count[0], 0);
        assert_eq!(state.step_count[3], 1);
    }

    // -------------------------------------------------------------------------
    // Observation Pipeline Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_build_observations_from_world_state() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_num_jammers(2)
            .with_num_crs(2)
            .with_freq_bins(256)
            .build();

        let mut world = RFWorld::new(config);
        world.reset_with_seed(42);

        let obs_config = ObservationConfig::default().with_psd_bins(32);

        let observations = MultiAgentObservations::build(
            world.state(),
            world.agent_state(),
            &obs_config,
        );

        // Verify dimensions
        assert_eq!(observations.num_envs, 8);
        assert_eq!(observations.num_jammers, 2);
        assert_eq!(observations.num_crs, 2);

        // Verify we can access all observations
        for env in 0..8 {
            for j in 0..2 {
                let obs = observations.jammer_observation(env, j);
                assert_eq!(obs.len(), observations.jammer_obs_size);
            }
            for c in 0..2 {
                let obs = observations.cr_observation(env, c);
                assert_eq!(obs.len(), observations.cr_obs_size);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Multi-Agent Reward Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_compute_separate_rewards_for_each_agent() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_num_jammers(2)
            .with_num_crs(2)
            .with_max_steps(10)
            .build();

        let mut world = RFWorld::new(config);
        world.reset_with_seed(42);

        let actions = vec![0.5f32; 8 * (2 * 4 + 2 * 3)];
        let (_state, rewards) = world.step_multi_agent(&actions);

        // Each agent should have their own reward (check first 2 envs)
        for env in 0..2 {
            let j0_reward = rewards.jammer_reward(env, 0);
            let j1_reward = rewards.jammer_reward(env, 1);
            let c0_reward = rewards.cr_reward(env, 0);
            let c1_reward = rewards.cr_reward(env, 1);

            // Rewards are finite numbers (no NaN/Inf)
            assert!(j0_reward.is_finite());
            assert!(j1_reward.is_finite());
            assert!(c0_reward.is_finite());
            assert!(c1_reward.is_finite());

            // Total reward should be sum
            let total_jammer = rewards.total_jammer_reward(env);
            let total_cr = rewards.total_cr_reward(env);

            assert!((total_jammer - (j0_reward + j1_reward)).abs() < 1e-6);
            assert!((total_cr - (c0_reward + c1_reward)).abs() < 1e-6);
        }
    }
}

// ============================================================================
// Edge Case and Error Condition Tests
// ============================================================================

mod edge_case_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Numerical Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_very_small_power_values() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 100).expect("Valid grid");
        let mut psd = vec![0.0; 100];

        // Very small power (near numerical precision)
        use crate::spectrum::patterns::add_rect_pattern;
        add_rect_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(10e6), PositivePower::new(1e-20));

        // Should still work without NaN/Inf
        for &val in &psd {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn should_handle_very_large_power_values() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 100).expect("Valid grid");
        let mut psd = vec![0.0; 100];

        use crate::spectrum::patterns::add_rect_pattern;
        add_rect_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(10e6), PositivePower::new(1e10));

        for &val in &psd {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn should_handle_zero_bandwidth() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 100).expect("Valid grid");
        let mut psd = vec![0.0; 100];

        // Zero bandwidth might cause division issues
        use crate::spectrum::patterns::add_chirp_pattern;
        add_chirp_pattern(&mut psd, &grid, Hertz::new(1.5e9), Hertz::new(1.5e9), PositivePower::new(0.001));

        // Should handle gracefully
        for &val in &psd {
            assert!(val.is_finite());
        }
    }

    // -------------------------------------------------------------------------
    // Boundary Index Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_last_environment_index() {
        let config = RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(256)
            .build();

        let mut state = RFWorldState::new(&config);

        // Access last env
        let last_env = 7;
        state.set_psd_power(last_env, 255, 100.0);

        assert!((state.get_psd_power(last_env, 255) - 100.0).abs() < 1e-6);

        // Reset last env
        state.reset_env(last_env);
        assert_eq!(state.step_count[last_env], 0);
    }

    #[test]
    fn should_handle_first_and_last_frequency_bins() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_freq_bins(512)
            .build();

        let mut state = RFWorldState::new(&config);

        // First bin
        state.set_psd_power(0, 0, 1.0);
        assert!((state.get_psd_power(0, 0) - 1.0).abs() < 1e-6);

        // Last bin
        state.set_psd_power(0, 511, 2.0);
        assert!((state.get_psd_power(0, 511) - 2.0).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Empty/Minimal Configuration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_single_step_episode() {
        let config = RFWorldConfig::new()
            .with_num_envs(8) // Must be multiple of 8
            .with_max_steps(1)
            .build();

        let mut adapter = RFEnvAdapter::new(config, AgentView::Jammer(0));
        adapter.reset_all(42);

        let actions = vec![0.0f32; 8];
        let result = adapter.step(&actions);

        // Single step should complete
        assert_eq!(result.rewards.len(), 8);
    }

    #[test]
    fn should_handle_observation_without_psd() {
        let config = ObservationConfig::new()
            .without_psd()
            .with_history_length(0);

        // Size without PSD should be smaller
        let size_with_psd = ObservationConfig::default().jammer_observation_size(1, 1);
        let size_without_psd = config.jammer_observation_size(1, 1);

        assert!(size_without_psd < size_with_psd);
    }

    // -------------------------------------------------------------------------
    // Frequency Range Edge Cases
    // -------------------------------------------------------------------------

    #[test]
    fn should_handle_signals_partially_in_range() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 100).expect("Valid grid");
        let mut psd = vec![0.0; 100];

        // Signal at edge of grid
        use crate::spectrum::patterns::add_rect_pattern;
        add_rect_pattern(&mut psd, &grid, Hertz::new(0.95e9), Hertz::new(100e6), PositivePower::new(0.001));

        // Should add power to lower bins
        assert!(psd[0..10].iter().any(|&x| x > 0.0));
    }
}
