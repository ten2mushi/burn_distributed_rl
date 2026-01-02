//! Continuous Entity Behavior
//!
//! Implements behavior for entities with 100% duty cycle:
//! - TV stations (ATSC/DVB-T)
//! - FM radio stations
//!
//! These entities are always active and only update their phase accumulator.
//! Uses type-safe `ValidatedFrequencyGrid` for all frequency operations.

#[cfg(feature = "simd")]
use std::simd::{f32x8, StdFloat};

use super::ValidatedFrequencyGrid;
use crate::config::RFWorldConfig;
use crate::constants;
use crate::entities::{entity_idx, EntitySoA};
use crate::types::dimensional::Hertz;

/// Continuous behavior implementation for always-on entities.
pub struct ContinuousBehavior;

impl ContinuousBehavior {
    // ========================================================================
    // Scalar Updates
    // ========================================================================

    /// Update a continuous entity (scalar version).
    ///
    /// Only updates phase accumulator with wrap.
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        _config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        // Phase accumulates at carrier frequency
        let freq = entities.center_freq[idx];
        let phase_increment = constants::TWO_PI * freq * dt;

        // Update phase with wrap to prevent overflow
        entities.phase[idx] = (entities.phase[idx] + phase_increment) % constants::TWO_PI;
    }

    // ========================================================================
    // SIMD Updates
    // ========================================================================

    /// Update continuous entities for 8 environments (SIMD version).
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        _config: &RFWorldConfig,
    ) {
        let base_env = batch * 8;
        let stride = entities.max_entities();

        // Load frequencies and phases
        let (freq, _, _) = entities.load_rf_params_simd(batch, entity);
        let phase = entities.load_timer_simd(batch, entity); // Reusing timer for phase

        // Calculate phase increment
        let two_pi = f32x8::splat(constants::TWO_PI);
        let phase_increment = two_pi * freq * dt;

        // Update phase with modulo (approximate using fmod pattern)
        let new_phase = phase + phase_increment;
        let wrapped_phase = new_phase - (new_phase / two_pi).floor() * two_pi;

        // Store back to phase array (stored in timer field for now)
        entities.store_timer_simd(batch, entity, wrapped_phase);

        // Also update the actual phase array
        let phase_arr: [f32; 8] = wrapped_phase.into();
        for lane in 0..8 {
            let idx = entity_idx(base_env + lane, entity, stride);
            entities.phase[idx] = phase_arr[lane];
        }
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render TV station to PSD (scalar version).
    ///
    /// TV signals have a 6 MHz rectangular spectral shape (brick wall).
    pub fn render_tv_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx]; // 6 MHz
        let power_dbm = entities.power_dbm[idx];

        // Check if signal is within grid range
        let low_freq = center_freq - bandwidth / 2.0;
        let high_freq = center_freq + bandwidth / 2.0;

        let freq_min = grid.freq_min().as_hz();
        let freq_max = grid.freq_max().as_hz();

        if high_freq < freq_min || low_freq > freq_max {
            return; // Signal outside grid
        }

        // Convert power from dBm to linear
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0; // mW to W

        // Distribute power across bandwidth
        let power_per_hz = power_linear / bandwidth;
        let resolution = grid.resolution().as_hz();
        let power_per_bin = power_per_hz * resolution;

        // Find affected bins
        let bin_start = grid.freq_to_bin(Hertz::new(low_freq.max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_freq.min(freq_max)));

        // Add power to PSD (rectangular shape)
        let psd_offset = env * config.num_freq_bins;
        for bin in bin_start..=bin_end {
            psd[psd_offset + bin] += power_per_bin;
        }
    }

    /// Render FM radio station to PSD (scalar version).
    ///
    /// FM signals have a Gaussian spectral shape with 200 kHz bandwidth.
    pub fn render_fm_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx]; // 200 kHz
        let power_dbm = entities.power_dbm[idx];

        let freq_min = grid.freq_min().as_hz();
        let freq_max = grid.freq_max().as_hz();

        // Check if signal is within grid range (use 3x bandwidth for Gaussian tails)
        let extend = bandwidth * 1.5;
        if center_freq + extend < freq_min || center_freq - extend > freq_max {
            return;
        }

        // Convert power from dBm to linear
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;

        // Gaussian parameters: sigma = bandwidth / (2 * sqrt(2 * ln(2))) for -3dB bandwidth
        let sigma = bandwidth / 2.355; // Approximately

        // Normalization factor for Gaussian
        let sqrt_2pi = (2.0 * constants::PI).sqrt();
        let gaussian_norm = power_linear / (sigma * sqrt_2pi);

        // Render Gaussian shape
        let psd_offset = env * config.num_freq_bins;
        let resolution = grid.resolution().as_hz();
        let bin_start = grid.freq_to_bin(Hertz::new((center_freq - extend).max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new((center_freq + extend).min(freq_max)));

        for bin in bin_start..=bin_end {
            let bin_freq = grid.bin_to_freq(bin).as_hz();
            let diff = bin_freq - center_freq;
            let exponent = -0.5 * (diff / sigma).powi(2);
            let gaussian_value = gaussian_norm * exponent.exp() * resolution;

            psd[psd_offset + bin] += gaussian_value;
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::EntityType;
    use crate::entities::ModulationType;

    fn test_config() -> RFWorldConfig {
        RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(512)
            .with_freq_range(300e6, 800e6)
            .build()
    }

    #[test]
    fn test_continuous_phase_update() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);

        // Set up a TV station
        entities.set_entity(
            0, 0,
            EntityType::TVStation,
            ModulationType::COFDM,
            0.0, 0.0, 0.0,
            550e6, 6e6, 45.0,
        );
        let idx = entities.idx(0, 0);
        entities.phase[idx] = 0.0;

        // Update
        let dt = 0.001; // 1 ms
        ContinuousBehavior::update_scalar(&mut entities, 0, 0, dt, &config);

        // Phase should have increased
        let idx = entities.idx(0, 0);
        let new_phase = entities.phase[idx];
        assert!(new_phase > 0.0);
        assert!(new_phase < constants::TWO_PI);
    }

    #[test]
    fn test_continuous_phase_wraps() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);

        entities.set_entity(
            0, 0,
            EntityType::TVStation,
            ModulationType::COFDM,
            0.0, 0.0, 0.0,
            550e6, 6e6, 45.0,
        );
        let idx = entities.idx(0, 0);
        entities.phase[idx] = constants::TWO_PI - 0.1;

        // Update with small dt to force wrap
        let dt = 0.0001;
        for _ in 0..1000 {
            ContinuousBehavior::update_scalar(&mut entities, 0, 0, dt, &config);
        }

        // Phase should still be bounded
        let idx = entities.idx(0, 0);
        let phase = entities.phase[idx];
        assert!(phase >= 0.0 && phase < constants::TWO_PI);
    }

    #[test]
    fn test_tv_render_in_band() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::TVStation,
                ModulationType::COFDM,
                0.0, 0.0, 0.0,
                550e6, 6e6, 45.0, // 550 MHz center, 6 MHz BW
            );
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        ContinuousBehavior::render_tv_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that power is added in the correct frequency range
        let center_bin = grid.freq_to_bin(Hertz::new(550e6));
        assert!(psd[center_bin] > 0.0);

        // Check adjacent bins also have power (rectangular shape)
        let low_bin = grid.freq_to_bin(Hertz::new(547e6));
        let high_bin = grid.freq_to_bin(Hertz::new(553e6));
        assert!(psd[low_bin] > 0.0);
        assert!(psd[high_bin] > 0.0);

        // Check outside the band has no power
        let out_of_band_bin = grid.freq_to_bin(Hertz::new(400e6));
        assert_eq!(psd[out_of_band_bin], 0.0);
    }

    #[test]
    fn test_fm_render_gaussian() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::FMRadio,
                ModulationType::FM,
                0.0, 0.0, 0.0,
                500e6, 200e3, 35.0, // 500 MHz center, 200 kHz BW
            );
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        ContinuousBehavior::render_fm_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that power peaks at center
        let center_bin = grid.freq_to_bin(Hertz::new(500e6));
        let peak_power = psd[center_bin];
        assert!(peak_power > 0.0);

        // Check that power decreases away from center (Gaussian shape)
        if center_bin > 5 {
            let side_power = psd[center_bin - 5];
            assert!(side_power < peak_power || side_power == 0.0);
        }
    }
}
