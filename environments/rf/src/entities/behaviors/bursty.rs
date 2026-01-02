//! Bursty Entity Behavior
//!
//! Implements behavior for entities with stochastic (Poisson) activation:
//! - LTE towers
//! - WiFi access points
//! - Bluetooth (combined with FHSS)
//!
//! These entities toggle between active and inactive states based on
//! exponentially distributed timers.

#[cfg(feature = "simd")]
use std::simd::{cmp::SimdPartialOrd, f32x8, Mask};

use super::ValidatedFrequencyGrid;
use crate::types::dimensional::Hertz;
use crate::config::RFWorldConfig;
use crate::entities::{entity_idx, EntitySoA, EntityType};

#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

/// Bursty behavior implementation for Poisson-activated entities.
pub struct BurstyBehavior;

// ============================================================================
// Constants
// ============================================================================

/// Mean active duration for LTE (ms)
const LTE_MEAN_ACTIVE_MS: f32 = 5.0;
/// Mean inactive duration for LTE (ms)
const LTE_MEAN_INACTIVE_MS: f32 = 8.0;

/// Mean active duration for WiFi (ms) - shorter bursts due to CSMA/CA
const WIFI_MEAN_ACTIVE_MS: f32 = 2.0;
/// Mean inactive duration for WiFi (ms)
const WIFI_MEAN_INACTIVE_MS: f32 = 15.0;

/// Mean active duration for Bluetooth (ms)
const BT_MEAN_ACTIVE_MS: f32 = 1.0;
/// Mean inactive duration for Bluetooth (ms)
const BT_MEAN_INACTIVE_MS: f32 = 5.0;

impl BurstyBehavior {
    // ========================================================================
    // Parameter Helpers
    // ========================================================================

    /// Get mean durations for an entity type.
    fn get_durations(entity_type: EntityType) -> (f32, f32) {
        match entity_type {
            EntityType::LTETower => (LTE_MEAN_ACTIVE_MS / 1000.0, LTE_MEAN_INACTIVE_MS / 1000.0),
            EntityType::WiFiAP => (WIFI_MEAN_ACTIVE_MS / 1000.0, WIFI_MEAN_INACTIVE_MS / 1000.0),
            EntityType::Bluetooth => (BT_MEAN_ACTIVE_MS / 1000.0, BT_MEAN_INACTIVE_MS / 1000.0),
            _ => (0.005, 0.010), // Default 5ms on, 10ms off
        }
    }

    // ========================================================================
    // Scalar Updates
    // ========================================================================

    /// Update a bursty entity (scalar version).
    ///
    /// Timer counts down; on expiry, toggles active state and samples
    /// new duration from exponential distribution.
    #[cfg(feature = "simd")]
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        rng: &mut SimdRng,
        _config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());
        let entity_type = EntityType::from(entities.entity_type[idx]);

        // Decrement timer
        entities.timer[idx] -= dt;

        // Check for expiry
        if entities.timer[idx] <= 0.0 {
            // Toggle transmitting state (stored in phase as 0.0 or 1.0)
            let is_transmitting = entities.phase[idx] > 0.5;

            let (mean_active, mean_inactive) = Self::get_durations(entity_type);

            if is_transmitting {
                // Was transmitting, now go inactive
                entities.phase[idx] = 0.0;

                // Sample inactive duration (exponential distribution)
                let u = rng.uniform();
                let u_arr: [f32; 8] = u.into();
                entities.timer[idx] = -mean_inactive * (u_arr[0] + 1e-10).ln();
            } else {
                // Was inactive, now start transmitting
                entities.phase[idx] = 1.0;

                // Sample active duration
                let u = rng.uniform();
                let u_arr: [f32; 8] = u.into();
                entities.timer[idx] = -mean_active * (u_arr[0] + 1e-10).ln();
            }
        }
    }

    // ========================================================================
    // SIMD Updates
    // ========================================================================

    /// Update bursty entities for 8 environments (SIMD version).
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        rng: &mut SimdRng,
        _config: &RFWorldConfig,
    ) {
        let base_env = batch * 8;
        let stride = entities.max_entities();

        // Get entity type (assume same across batch)
        let entity_type = EntityType::from(entities.entity_type[entity_idx(base_env, entity, stride)]);
        let (mean_active, mean_inactive) = Self::get_durations(entity_type);

        // Load timer and transmitting state (stored in phase)
        let timer = entities.load_timer_simd(batch, entity);
        let phase = f32x8::from_array([
            entities.phase[entity_idx(base_env + 0, entity, stride)],
            entities.phase[entity_idx(base_env + 1, entity, stride)],
            entities.phase[entity_idx(base_env + 2, entity, stride)],
            entities.phase[entity_idx(base_env + 3, entity, stride)],
            entities.phase[entity_idx(base_env + 4, entity, stride)],
            entities.phase[entity_idx(base_env + 5, entity, stride)],
            entities.phase[entity_idx(base_env + 6, entity, stride)],
            entities.phase[entity_idx(base_env + 7, entity, stride)],
        ]);

        // Decrement timer
        let new_timer = timer - dt;

        // Check for expiry (timer <= 0)
        let expired: Mask<i32, 8> = new_timer.simd_le(f32x8::splat(0.0));

        // Determine current transmitting state
        let is_transmitting: Mask<i32, 8> = phase.simd_gt(f32x8::splat(0.5));

        // Sample new durations
        let u = rng.uniform();
        let eps = f32x8::splat(1e-10);
        let log_u = crate::simd_rf::math::simd_log(u + eps);

        // Calculate new timer based on state transition
        let mean_active_simd = f32x8::splat(mean_active);
        let mean_inactive_simd = f32x8::splat(mean_inactive);

        // If was transmitting -> use inactive duration; else use active duration
        let new_duration = is_transmitting.select(
            -mean_inactive_simd * log_u,
            -mean_active_simd * log_u,
        );

        // Update timer: if expired use new_duration, else use decremented timer
        let final_timer = expired.select(new_duration, new_timer);

        // Update phase: if expired, toggle; else keep same
        let new_phase_value = is_transmitting.select(f32x8::splat(0.0), f32x8::splat(1.0));
        let final_phase = expired.select(new_phase_value, phase);

        // Store back
        entities.store_timer_simd(batch, entity, final_timer);

        let phase_arr: [f32; 8] = final_phase.into();
        for lane in 0..8 {
            entities.phase[entity_idx(base_env + lane, entity, stride)] = phase_arr[lane];
        }
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render OFDM signal to PSD (scalar version).
    ///
    /// Used for LTE and WiFi - flat power distribution across bandwidth.
    pub fn render_ofdm_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        // Check if transmitting (phase > 0.5)
        if entities.phase[idx] < 0.5 {
            return; // Not transmitting
        }

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx];
        let power_dbm = entities.power_dbm[idx];

        // Get grid bounds using type-safe API
        let freq_min = grid.freq_min().as_hz();
        let freq_max = grid.freq_max().as_hz();
        let resolution = grid.resolution().as_hz();

        // Check if signal is within grid range
        let low_freq = center_freq - bandwidth / 2.0;
        let high_freq = center_freq + bandwidth / 2.0;

        if high_freq < freq_min || low_freq > freq_max {
            return;
        }

        // Convert power from dBm to linear
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;

        // OFDM has relatively flat spectrum with small guard bands
        // Use 90% of bandwidth for subcarriers
        let effective_bw = bandwidth * 0.9;
        let power_per_hz = power_linear / effective_bw;
        let power_per_bin = power_per_hz * resolution;

        // Find affected bins
        let bin_start = grid.freq_to_bin(Hertz::new((center_freq - effective_bw / 2.0).max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new((center_freq + effective_bw / 2.0).min(freq_max)));

        // Add power to PSD (flat OFDM shape)
        let psd_offset = env * config.num_freq_bins;
        for bin in bin_start..=bin_end {
            psd[psd_offset + bin] += power_per_bin;
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::ModulationType;

    fn test_config() -> RFWorldConfig {
        RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(512)
            .with_freq_range(300e6, 3e9)
            .build()
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_bursty_state_toggle() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);
        let mut rng = SimdRng::new(42);

        // Set up an LTE tower
        entities.set_entity(
            0, 0,
            EntityType::LTETower,
            ModulationType::OFDM,
            0.0, 0.0, 0.0,
            2e9, 20e6, 46.0,
        );

        // Start transmitting
        let idx = entities.idx(0, 0);
        entities.phase[idx] = 1.0;
        entities.timer[idx] = 0.001; // 1ms remaining

        // Run updates until state changes
        let dt = 0.0001; // 0.1 ms timestep
        let mut toggled = false;

        for _ in 0..100 {
            let initial_phase = entities.phase[idx];
            BurstyBehavior::update_scalar(&mut entities, 0, 0, dt, &mut rng, &config);
            if entities.phase[idx] != initial_phase {
                toggled = true;
                break;
            }
        }

        assert!(toggled, "Bursty entity should have toggled state");
    }

    #[test]
    fn test_ofdm_render_when_active() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::LTETower,
                ModulationType::OFDM,
                0.0, 0.0, 0.0,
                2e9, 20e6, 46.0,
            );
            // Set as transmitting
            let idx = e.idx(0, 0);
            e.phase[idx] = 1.0;
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        BurstyBehavior::render_ofdm_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that power is added in the correct frequency range
        let center_bin = grid.freq_to_bin(Hertz::new(2e9));
        assert!(psd[center_bin] > 0.0, "OFDM should add power at center frequency");
    }

    #[test]
    fn test_ofdm_no_render_when_inactive() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::LTETower,
                ModulationType::OFDM,
                0.0, 0.0, 0.0,
                2e9, 20e6, 46.0,
            );
            // Set as NOT transmitting
            let idx = e.idx(0, 0);
            e.phase[idx] = 0.0;
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        BurstyBehavior::render_ofdm_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that no power was added
        let total_power: f32 = psd.iter().sum();
        assert_eq!(total_power, 0.0, "Inactive entity should not add power");
    }

    #[test]
    fn test_ofdm_flat_spectrum() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::WiFiAP,
                ModulationType::OFDM,
                0.0, 0.0, 0.0,
                2.4e9, 20e6, 23.0,
            );
            let idx = e.idx(0, 0);
            e.phase[idx] = 1.0; // Transmitting
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        BurstyBehavior::render_ofdm_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that power is relatively flat across the bandwidth
        let center_bin = grid.freq_to_bin(Hertz::new(2.4e9));
        let offset_bin = grid.freq_to_bin(Hertz::new(2.4e9 + 5e6)); // 5 MHz offset

        // Both should have similar power (within 1 dB)
        let center_power = psd[center_bin];
        let offset_power = psd[offset_bin];

        if center_power > 0.0 && offset_power > 0.0 {
            let ratio = center_power / offset_power;
            assert!(ratio > 0.8 && ratio < 1.25, "OFDM spectrum should be flat");
        }
    }
}
