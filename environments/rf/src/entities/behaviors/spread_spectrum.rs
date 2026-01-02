//! Spread Spectrum Entity Behavior
//!
//! Implements FHSS (Frequency Hopping Spread Spectrum) and DSSS logic:
//! - Bluetooth: FHSS with 79 channels, 1600 hops/second
//!
//! FHSS entities hop between frequencies using a pseudo-random sequence.

#[cfg(feature = "simd")]
use std::simd::f32x8;

use super::ValidatedFrequencyGrid;
use crate::types::dimensional::Hertz;
use crate::config::RFWorldConfig;
use crate::entities::{entity_idx, EntitySoA, EntityType};

/// Spread spectrum behavior implementation.
pub struct SpreadSpectrumBehavior;

// ============================================================================
// Constants
// ============================================================================

/// Bluetooth FHSS parameters
const BT_NUM_CHANNELS: u32 = 79;
const BT_HOP_RATE: f32 = 1600.0; // hops per second
const BT_HOP_INTERVAL: f32 = 1.0 / BT_HOP_RATE; // ~625 μs
const BT_CHANNEL_SPACING: f32 = 1e6; // 1 MHz between channels
const BT_BASE_FREQ: f32 = 2.402e9; // 2.402 GHz start of Bluetooth band
const BT_CHANNEL_BANDWIDTH: f32 = 1e6; // 1 MHz per channel

impl SpreadSpectrumBehavior {
    // ========================================================================
    // FHSS Channel Calculation
    // ========================================================================

    /// Calculate the current hop frequency for a given hop index.
    ///
    /// Uses a simple pseudo-random sequence: channel = (hop_idx * 17) % num_channels
    /// The multiplier 17 is chosen to provide good channel spreading.
    #[inline]
    fn calc_hop_channel(hop_idx: u32, num_channels: u32) -> u32 {
        // Simple LCG-like hop sequence
        // Using prime multiplier for better spreading
        (hop_idx.wrapping_mul(17)) % num_channels
    }

    /// Convert channel index to actual frequency.
    #[inline]
    fn channel_to_freq(channel: u32, base_freq: f32, spacing: f32) -> f32 {
        base_freq + (channel as f32) * spacing
    }

    // ========================================================================
    // Scalar Updates
    // ========================================================================

    /// Update FHSS entity (scalar version).
    ///
    /// Manages hop timing and updates center frequency based on hop sequence.
    pub fn update_fhss_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        _config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());
        let entity_type = EntityType::from(entities.entity_type[idx]);

        // Get FHSS parameters based on entity type
        let (num_channels, hop_interval, base_freq, spacing) = match entity_type {
            EntityType::Bluetooth => (
                BT_NUM_CHANNELS,
                BT_HOP_INTERVAL,
                BT_BASE_FREQ,
                BT_CHANNEL_SPACING,
            ),
            _ => return, // Not an FHSS entity
        };

        // Decrement hop timer (stored in timer field, separate from bursty timer)
        // Note: For Bluetooth, we reuse the timer field for hop timing
        // since the bursty behavior has already processed it
        let hop_timer = entities.timer[idx];
        let new_hop_timer = hop_timer - dt;

        if new_hop_timer <= 0.0 {
            // Time to hop to next channel
            let hop_idx = entities.hop_idx[idx];
            let new_hop_idx = hop_idx.wrapping_add(1);
            entities.hop_idx[idx] = new_hop_idx;

            // Calculate new channel and frequency
            let channel = Self::calc_hop_channel(new_hop_idx, num_channels);
            let new_freq = Self::channel_to_freq(channel, base_freq, spacing);
            entities.center_freq[idx] = new_freq;

            // Reset hop timer (with wraparound handling)
            entities.timer[idx] = hop_interval + new_hop_timer;
        } else {
            entities.timer[idx] = new_hop_timer;
        }
    }

    // ========================================================================
    // SIMD Updates
    // ========================================================================

    /// Update FHSS entities for 8 environments (SIMD version).
    #[cfg(feature = "simd")]
    pub fn update_fhss_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        _config: &RFWorldConfig,
    ) {
        let base_env = batch * 8;
        let stride = entities.max_entities();

        // Get entity type (assume same across batch)
        let entity_type =
            EntityType::from(entities.entity_type[entity_idx(base_env, entity, stride)]);

        let (num_channels, hop_interval, base_freq, spacing) = match entity_type {
            EntityType::Bluetooth => (
                BT_NUM_CHANNELS,
                BT_HOP_INTERVAL,
                BT_BASE_FREQ,
                BT_CHANNEL_SPACING,
            ),
            _ => return,
        };

        // Load timers
        let timer = entities.load_timer_simd(batch, entity);
        let new_timer = timer - dt;

        // Process each lane individually for hop logic
        // (hop index update is stateful and hard to vectorize cleanly)
        let timer_arr: [f32; 8] = new_timer.into();

        for lane in 0..8 {
            let idx = entity_idx(base_env + lane, entity, stride);

            if timer_arr[lane] <= 0.0 {
                // Time to hop
                let hop_idx = entities.hop_idx[idx];
                let new_hop_idx = hop_idx.wrapping_add(1);
                entities.hop_idx[idx] = new_hop_idx;

                let channel = Self::calc_hop_channel(new_hop_idx, num_channels);
                let new_freq = Self::channel_to_freq(channel, base_freq, spacing);
                entities.center_freq[idx] = new_freq;

                // Reset timer with wraparound
                entities.timer[idx] = hop_interval + timer_arr[lane];
            } else {
                entities.timer[idx] = timer_arr[lane];
            }
        }
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render FHSS signal to PSD (scalar version).
    ///
    /// FHSS signals appear as narrow Gaussian peaks at the current hop frequency.
    pub fn render_fhss_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        // Check if transmitting (phase > 0.5, set by bursty behavior)
        if entities.phase[idx] < 0.5 {
            return; // Not transmitting
        }

        let center_freq = entities.center_freq[idx]; // Current hop frequency
        let bandwidth = BT_CHANNEL_BANDWIDTH;
        let power_dbm = entities.power_dbm[idx];

        // Get grid bounds using type-safe API
        let freq_min = grid.freq_min().as_hz();
        let freq_max = grid.freq_max().as_hz();
        let resolution = grid.resolution().as_hz();

        // Check if signal is within grid range
        let low_freq = center_freq - bandwidth;
        let high_freq = center_freq + bandwidth;

        if high_freq < freq_min || low_freq > freq_max {
            return;
        }

        // Convert power from dBm to linear
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;

        // Bluetooth uses GFSK - approximated as narrow Gaussian
        let sigma = bandwidth / 2.355; // -3dB bandwidth
        let sqrt_2pi = (2.0 * std::f32::consts::PI).sqrt();
        let gaussian_norm = power_linear / (sigma * sqrt_2pi);

        // Render Gaussian shape
        let psd_offset = env * config.num_freq_bins;
        let extend = bandwidth * 1.5;
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

    // ========================================================================
    // DSSS (Direct Sequence Spread Spectrum)
    // ========================================================================

    /// Render DSSS signal to PSD.
    ///
    /// DSSS signals spread power across a wide bandwidth, appearing as
    /// a raised noise floor. Used for GPS/GNSS signals.
    #[allow(dead_code)]
    pub fn render_dsss_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx]; // Spread bandwidth
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

        // DSSS spreads power evenly across bandwidth (sinc^2 envelope)
        // Simplified as flat distribution for now
        let power_per_hz = power_linear / bandwidth;
        let power_per_bin = power_per_hz * resolution;

        // Find affected bins
        let bin_start = grid.freq_to_bin(Hertz::new(low_freq.max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_freq.min(freq_max)));

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
            .with_freq_range(2.4e9, 2.5e9) // Bluetooth band
            .build()
    }

    #[test]
    fn test_hop_channel_distribution() {
        // Test that hop sequence covers all channels reasonably
        let mut channel_hits = vec![0u32; BT_NUM_CHANNELS as usize];

        for hop_idx in 0..1000 {
            let channel = SpreadSpectrumBehavior::calc_hop_channel(hop_idx, BT_NUM_CHANNELS);
            assert!(channel < BT_NUM_CHANNELS);
            channel_hits[channel as usize] += 1;
        }

        // Check that all channels are hit at least once in 1000 hops
        // With 79 channels and 1000 hops, each should be hit ~12.6 times
        for (channel, &hits) in channel_hits.iter().enumerate() {
            assert!(
                hits >= 1,
                "Channel {} was never visited in 1000 hops",
                channel
            );
        }
    }

    #[test]
    fn test_hop_sequence_not_sequential() {
        // Verify that consecutive hops don't produce sequential channels
        let ch0 = SpreadSpectrumBehavior::calc_hop_channel(0, BT_NUM_CHANNELS);
        let ch1 = SpreadSpectrumBehavior::calc_hop_channel(1, BT_NUM_CHANNELS);
        let ch2 = SpreadSpectrumBehavior::calc_hop_channel(2, BT_NUM_CHANNELS);

        // Channels should not be consecutive
        assert!(ch1 != ch0 + 1 || ch2 != ch1 + 1, "Hop sequence is too sequential");
    }

    #[test]
    fn test_channel_to_freq() {
        let freq0 = SpreadSpectrumBehavior::channel_to_freq(0, BT_BASE_FREQ, BT_CHANNEL_SPACING);
        let freq1 = SpreadSpectrumBehavior::channel_to_freq(1, BT_BASE_FREQ, BT_CHANNEL_SPACING);
        let freq78 = SpreadSpectrumBehavior::channel_to_freq(78, BT_BASE_FREQ, BT_CHANNEL_SPACING);

        // Use 1 kHz tolerance for f32 precision at GHz frequencies
        assert!((freq0 - 2.402e9).abs() < 1e3, "freq0 = {}", freq0);
        assert!((freq1 - 2.403e9).abs() < 1e3, "freq1 = {}", freq1);
        assert!((freq78 - 2.480e9).abs() < 1e3, "freq78 = {}", freq78);
    }

    #[test]
    fn test_fhss_update_hops() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);

        entities.set_entity(
            0,
            0,
            EntityType::Bluetooth,
            ModulationType::FHSS,
            0.0,
            0.0,
            0.0,
            BT_BASE_FREQ,
            BT_CHANNEL_BANDWIDTH,
            10.0,
        );

        let idx = entities.idx(0, 0);
        entities.timer[idx] = 0.0001; // Almost expired
        entities.hop_idx[idx] = 0;
        let initial_freq = entities.center_freq[idx];

        // Update with small dt to trigger hop
        SpreadSpectrumBehavior::update_fhss_scalar(&mut entities, 0, 0, 0.0002, &config);

        // Hop index should have incremented
        assert_eq!(entities.hop_idx[idx], 1);

        // Frequency should have changed
        let new_freq = entities.center_freq[idx];
        assert_ne!(initial_freq, new_freq, "Frequency should change after hop");

        // Timer should be reset
        assert!(entities.timer[idx] > 0.0);
    }

    #[test]
    fn test_fhss_render_when_transmitting() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0,
                0,
                EntityType::Bluetooth,
                ModulationType::FHSS,
                0.0,
                0.0,
                0.0,
                2.44e9, // Mid-band
                BT_CHANNEL_BANDWIDTH,
                10.0,
            );
            // Set as transmitting
            let idx = e.idx(0, 0);
            e.phase[idx] = 1.0;
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        SpreadSpectrumBehavior::render_fhss_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that power is added at the hop frequency
        let center_bin = grid.freq_to_bin(Hertz::new(2.44e9));
        assert!(psd[center_bin] > 0.0, "FHSS should add power at hop frequency");
    }

    #[test]
    fn test_fhss_no_render_when_not_transmitting() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0,
                0,
                EntityType::Bluetooth,
                ModulationType::FHSS,
                0.0,
                0.0,
                0.0,
                2.44e9,
                BT_CHANNEL_BANDWIDTH,
                10.0,
            );
            // Set as NOT transmitting
            let idx = e.idx(0, 0);
            e.phase[idx] = 0.0;
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        SpreadSpectrumBehavior::render_fhss_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        let total_power: f32 = psd.iter().sum();
        assert_eq!(total_power, 0.0, "Non-transmitting FHSS should add no power");
    }

    #[test]
    fn test_hop_rate() {
        // Verify hop interval matches 1600 hops/second
        let expected_interval = 1.0 / 1600.0; // ~625 μs
        assert!((BT_HOP_INTERVAL - expected_interval).abs() < 1e-9);
    }
}
