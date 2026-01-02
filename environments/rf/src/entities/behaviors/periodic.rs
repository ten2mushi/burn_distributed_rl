//! Periodic Entity Behavior
//!
//! Implements behavior for entities with deterministic pulse trains:
//! - S-Band radar
//! - Weather radar
//!
//! These entities emit high-power pulses at regular intervals (PRI).

#[cfg(feature = "simd")]
use std::simd::{cmp::SimdPartialOrd, f32x8, StdFloat};

use super::ValidatedFrequencyGrid;
use crate::types::dimensional::Hertz;
use crate::config::RFWorldConfig;
use crate::entities::{entity_idx, EntitySoA, EntityType};

/// Periodic behavior implementation for pulsed entities.
pub struct PeriodicBehavior;

// ============================================================================
// Constants
// ============================================================================

/// S-Band radar PRI (Pulse Repetition Interval) in seconds
const SBAND_PRI: f32 = 0.001; // 1 ms
/// S-Band radar pulse width in seconds
const SBAND_PULSE_WIDTH: f32 = 1e-6; // 1 μs

/// Weather radar PRI in seconds
const WEATHER_PRI: f32 = 0.003; // 3 ms
/// Weather radar pulse width in seconds
const WEATHER_PULSE_WIDTH: f32 = 2e-6; // 2 μs

impl PeriodicBehavior {
    // ========================================================================
    // Parameter Helpers
    // ========================================================================

    /// Get PRI and pulse width for an entity type.
    fn get_pulse_params(entity_type: EntityType) -> (f32, f32) {
        match entity_type {
            EntityType::SBandRadar => (SBAND_PRI, SBAND_PULSE_WIDTH),
            EntityType::WeatherRadar => (WEATHER_PRI, WEATHER_PULSE_WIDTH),
            _ => (0.001, 1e-6), // Default
        }
    }

    // ========================================================================
    // Scalar Updates
    // ========================================================================

    /// Update a periodic entity (scalar version).
    ///
    /// Timer counts down and wraps at PRI. Entity is "active" (pulsing)
    /// when timer is within pulse width of PRI boundary.
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        _config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());
        let entity_type = EntityType::from(entities.entity_type[idx]);

        let (pri, pulse_width) = Self::get_pulse_params(entity_type);

        // Increment timer
        entities.timer[idx] += dt;

        // Wrap at PRI
        if entities.timer[idx] >= pri {
            entities.timer[idx] -= pri;
        }

        // Determine if we're in the pulse window
        // Pulse occurs at the start of each PRI period
        let in_pulse = entities.timer[idx] < pulse_width;

        // Store pulse state in phase (1.0 = pulsing, 0.0 = not pulsing)
        entities.phase[idx] = if in_pulse { 1.0 } else { 0.0 };
    }

    // ========================================================================
    // SIMD Updates
    // ========================================================================

    /// Update periodic entities for 8 environments (SIMD version).
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

        // Get entity type (assume same across batch)
        let entity_type = EntityType::from(entities.entity_type[entity_idx(base_env, entity, stride)]);
        let (pri, pulse_width) = Self::get_pulse_params(entity_type);

        let pri_simd = f32x8::splat(pri);
        let pulse_width_simd = f32x8::splat(pulse_width);

        // Load timer
        let timer = entities.load_timer_simd(batch, entity);

        // Increment timer
        let new_timer = timer + dt;

        // Wrap at PRI using modulo
        let wrapped_timer = new_timer - (new_timer / pri_simd).floor() * pri_simd;

        // Determine if in pulse window
        let in_pulse = wrapped_timer.simd_lt(pulse_width_simd);
        let phase = in_pulse.select(f32x8::splat(1.0), f32x8::splat(0.0));

        // Store back
        entities.store_timer_simd(batch, entity, wrapped_timer);

        let phase_arr: [f32; 8] = phase.into();
        for lane in 0..8 {
            entities.phase[entity_idx(base_env + lane, entity, stride)] = phase_arr[lane];
        }
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render chirp radar signal to PSD (scalar version).
    ///
    /// Radar pulses use linear FM chirp spreading power across bandwidth.
    /// Only renders when the radar is actively pulsing (phase > 0.5).
    ///
    /// # Physics
    ///
    /// Real radar pulses are microseconds long (1-2 μs) with millisecond PRIs,
    /// giving duty cycles of ~0.1%. This means radars emit zero power 99.9%
    /// of the time. This function is physically accurate - it only adds power
    /// to the PSD during the pulse window.
    ///
    /// # Visualization Note
    ///
    /// Since simulation timestep (1ms) >> pulse width (1μs), the probability
    /// of catching a radar pulse is very low. Visualization systems should
    /// implement peak-hold or integration to detect pulsed signals.
    pub fn render_chirp_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx]; // Chirp sweep width
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

        // Convert power from dBm to linear (radar power is HIGH)
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;

        // Check if pulsing (phase > 0.5)
        // Radar only emits power during the pulse window.
        // This is physically accurate: 0.1% duty cycle means 99.9% of the time
        // the radar emits zero power.
        if entities.phase[idx] < 0.5 {
            return; // Not pulsing - zero emission (physically correct)
        }

        // Full instantaneous power during pulse
        let effective_power = power_linear;

        // Chirp spreads power relatively evenly across bandwidth
        let power_per_hz = effective_power / bandwidth;
        let power_per_bin = power_per_hz * resolution;

        // Find affected bins
        let bin_start = grid.freq_to_bin(Hertz::new(low_freq.max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_freq.min(freq_max)));

        // Add power to PSD (flat chirp shape)
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
            .with_freq_range(2e9, 4e9)
            .build()
    }

    #[test]
    fn test_periodic_timer_wraps() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);

        entities.set_entity(
            0, 0,
            EntityType::SBandRadar,
            ModulationType::Chirp,
            0.0, 0.0, 0.0,
            3e9, 10e6, 60.0,
        );
        let idx = entities.idx(0, 0);
        entities.timer[idx] = 0.0;

        // Run for several PRIs
        let dt = 0.0001; // 0.1 ms
        for _ in 0..100 {
            PeriodicBehavior::update_scalar(&mut entities, 0, 0, dt, &config);
        }

        // Timer should have wrapped multiple times and be < PRI
        let timer = entities.timer[idx];
        assert!(timer < SBAND_PRI, "Timer should wrap at PRI");
        assert!(timer >= 0.0, "Timer should be non-negative");
    }

    #[test]
    fn test_periodic_pulse_timing() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);

        entities.set_entity(
            0, 0,
            EntityType::SBandRadar,
            ModulationType::Chirp,
            0.0, 0.0, 0.0,
            3e9, 10e6, 60.0,
        );
        let idx = entities.idx(0, 0);
        entities.timer[idx] = 0.0;

        // At t=0, should be pulsing
        PeriodicBehavior::update_scalar(&mut entities, 0, 0, 0.0001, &config);
        // Timer was 0, now 0.0001 which is > pulse width (1e-6)
        // Actually first update sets timer to dt, so check again

        // Reset to exactly 0
        entities.timer[idx] = 0.0;
        entities.phase[idx] = 0.0;

        // Very small dt to stay in pulse window
        let tiny_dt = 1e-7; // 0.1 μs, less than pulse width
        PeriodicBehavior::update_scalar(&mut entities, 0, 0, tiny_dt, &config);

        // Should be pulsing (timer < pulse_width)
        assert!(entities.phase[idx] > 0.5, "Should be pulsing at start of PRI");

        // Move past pulse window
        entities.timer[idx] = SBAND_PULSE_WIDTH * 2.0;
        PeriodicBehavior::update_scalar(&mut entities, 0, 0, 0.0, &config);

        // Should not be pulsing anymore
        assert!(entities.phase[idx] < 0.5, "Should not be pulsing outside pulse window");
    }

    #[test]
    fn test_chirp_render_when_pulsing() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::SBandRadar,
                ModulationType::Chirp,
                0.0, 0.0, 0.0,
                3e9, 10e6, 60.0,
            );
            // Set as pulsing
            let idx = e.idx(0, 0);
            e.phase[idx] = 1.0;
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        PeriodicBehavior::render_chirp_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that high power is added
        let center_bin = grid.freq_to_bin(Hertz::new(3e9));
        assert!(psd[center_bin] > 0.0, "Chirp should add power at center frequency");

        // Radar power is high - check it's significant
        let power_dbm = 60.0;
        let expected_power = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;
        let total_power: f32 = psd.iter().sum();
        // Should have substantial power (accounting for grid resolution)
        assert!(total_power > expected_power * 0.1, "Radar should add significant power");
    }

    #[test]
    fn test_chirp_no_render_when_not_pulsing() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::SBandRadar,
                ModulationType::Chirp,
                0.0, 0.0, 0.0,
                3e9, 10e6, 60.0,
            );
            // Set as NOT pulsing (physically accurate: zero emission)
            let idx = e.idx(0, 0);
            e.phase[idx] = 0.0;
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        PeriodicBehavior::render_chirp_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Physically accurate: non-pulsing radar emits ZERO power
        let total_power: f32 = psd.iter().sum();
        assert_eq!(total_power, 0.0, "Non-pulsing radar should emit zero power (physically accurate)");
    }

    #[test]
    fn test_duty_cycle() {
        // Calculate expected duty cycle
        let duty_sband = SBAND_PULSE_WIDTH / SBAND_PRI;
        let duty_weather = WEATHER_PULSE_WIDTH / WEATHER_PRI;

        // S-band: 1μs / 1ms = 0.001 = 0.1%
        assert!((duty_sband - 0.001).abs() < 1e-6);

        // Weather: 2μs / 3ms ≈ 0.00067 = 0.067%
        assert!((duty_weather - (2.0/3000.0)).abs() < 1e-6);
    }
}
