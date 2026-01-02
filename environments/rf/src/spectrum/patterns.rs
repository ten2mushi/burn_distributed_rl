//! Spectral Pattern Functions
//!
//! Functions for adding various spectral shapes to PSD buffers:
//! - Rectangular (brick-wall) for TV, radar
//! - Gaussian for FM radio, FHSS
//! - OFDM for LTE, WiFi
//! - Chirp for radar sweeps
//!
//! All functions use type-safe `PositivePower`, `Hertz`, and
//! `ValidatedFrequencyGrid` types to guarantee correctness at compile time.

#[cfg(feature = "simd")]
use std::simd::f32x8;

use crate::constants;
use crate::types::primitives::PositivePower;
use crate::types::dimensional::Hertz;
use crate::types::frequency::ValidatedFrequencyGrid;

// ============================================================================
// Rectangular Pattern
// ============================================================================

/// Add a rectangular (brick-wall) spectral pattern to PSD.
///
/// Power is distributed evenly across the bandwidth.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center frequency
/// * `bandwidth` - Total bandwidth
/// * `power` - Total power (guaranteed non-negative)
pub fn add_rect_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();
    let low_freq = center - bw / 2.0;
    let high_freq = center + bw / 2.0;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    // Power per Hz, then per bin
    let power_per_hz = power.watts() / bw;
    let power_per_bin = power_per_hz * grid.resolution().as_hz();

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // Add power to each bin in range
    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        psd[bin] += power_per_bin;
    }
}

/// SIMD version of rectangular pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_rect_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    bandwidth: f32x8,
    power_linear: f32x8,
) {
    let half_bw = bandwidth / f32x8::splat(2.0);
    let low_freq = center_freq - half_bw;
    let high_freq = center_freq + half_bw;

    let power_per_hz = power_linear / bandwidth;
    let power_per_bin = power_per_hz * f32x8::splat(grid.resolution().as_hz());

    // Process each lane
    let low_arr: [f32; 8] = low_freq.into();
    let high_arr: [f32; 8] = high_freq.into();
    let power_arr: [f32; 8] = power_per_bin.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();

    for lane in 0..8 {
        let bin_start = grid.freq_to_bin(Hertz::new(low_arr[lane].max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_arr[lane].min(freq_max)));

        let psd_offset = env_offset + lane * num_bins;
        for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
            psd[psd_offset + bin] += power_arr[lane];
        }
    }
}

// ============================================================================
// Gaussian Pattern
// ============================================================================

/// Add a Gaussian spectral pattern to PSD.
///
/// Power distribution follows a Gaussian curve centered at center_freq.
/// The bandwidth parameter corresponds to the -3dB bandwidth.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center frequency
/// * `bandwidth` - -3dB bandwidth
/// * `power` - Total power (guaranteed non-negative)
pub fn add_gaussian_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();

    // sigma = bandwidth / (2 * sqrt(2 * ln(2))) for -3dB bandwidth
    let sigma = bw / 2.355;

    // Extend rendering to 3 sigma for practical cutoff
    let extend = 3.0 * sigma;

    // Check if signal is within grid range
    if center + extend < grid.freq_min().as_hz() || center - extend > grid.freq_max().as_hz() {
        return;
    }

    // Normalization for Gaussian: integral = power_linear
    let sqrt_2pi = (2.0 * constants::PI).sqrt();
    let norm = power.watts() / (sigma * sqrt_2pi);

    // Find affected bins
    let low_hz = (center - extend).max(grid.freq_min().as_hz());
    let high_hz = (center + extend).min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    let resolution = grid.resolution().as_hz();

    // Add Gaussian-shaped power to each bin
    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        let bin_freq = grid.bin_to_freq(bin).as_hz();
        let diff = bin_freq - center;
        let exponent = -0.5 * (diff / sigma).powi(2);
        let gaussian_value = norm * exponent.exp() * resolution;
        psd[bin] += gaussian_value;
    }
}

/// SIMD version of Gaussian pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_gaussian_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    bandwidth: f32x8,
    power_linear: f32x8,
) {
    let sigma = bandwidth / f32x8::splat(2.355);
    let extend = sigma * f32x8::splat(3.0);

    let sqrt_2pi = f32x8::splat((2.0 * constants::PI).sqrt());
    let norm = power_linear / (sigma * sqrt_2pi);

    // Process each lane
    let center_arr: [f32; 8] = center_freq.into();
    let sigma_arr: [f32; 8] = sigma.into();
    let extend_arr: [f32; 8] = extend.into();
    let norm_arr: [f32; 8] = norm.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();
    let resolution = grid.resolution().as_hz();

    for lane in 0..8 {
        let c = center_arr[lane];
        let s = sigma_arr[lane];
        let e = extend_arr[lane];
        let n = norm_arr[lane];

        if c + e < freq_min || c - e > freq_max {
            continue;
        }

        let bin_start = grid.freq_to_bin(Hertz::new((c - e).max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new((c + e).min(freq_max)));

        let psd_offset = env_offset + lane * num_bins;
        for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
            let bin_freq = grid.bin_to_freq(bin).as_hz();
            let diff = bin_freq - c;
            let exponent = -0.5 * (diff / s).powi(2);
            let gaussian_value = n * exponent.exp() * resolution;
            psd[psd_offset + bin] += gaussian_value;
        }
    }
}

// ============================================================================
// OFDM Pattern
// ============================================================================

/// Add an OFDM spectral pattern to PSD.
///
/// OFDM signals have a relatively flat spectrum with small guard bands.
/// This is a simplified model using 90% bandwidth utilization.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center frequency
/// * `bandwidth` - Total bandwidth
/// * `power` - Total power (guaranteed non-negative)
pub fn add_ofdm_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();

    // OFDM uses ~90% of bandwidth for subcarriers (guard bands on edges)
    let effective_bw = bw * 0.9;

    let low_freq = center - effective_bw / 2.0;
    let high_freq = center + effective_bw / 2.0;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    // Power distributed evenly across effective bandwidth
    let power_per_hz = power.watts() / effective_bw;
    let power_per_bin = power_per_hz * grid.resolution().as_hz();

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // Add flat power to each bin
    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        psd[bin] += power_per_bin;
    }
}

// ============================================================================
// Chirp Pattern
// ============================================================================

/// Add a chirp (linear FM) spectral pattern to PSD.
///
/// Chirp signals sweep linearly from start_freq to end_freq.
/// Power is distributed across the sweep range.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `start_freq` - Chirp start frequency
/// * `end_freq` - Chirp end frequency
/// * `power` - Total power (guaranteed non-negative)
pub fn add_chirp_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    start_freq: Hertz,
    end_freq: Hertz,
    power: PositivePower,
) {
    let start = start_freq.as_hz();
    let end = end_freq.as_hz();

    let low_freq = start.min(end);
    let high_freq = start.max(end);
    let chirp_bw = high_freq - low_freq;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    let resolution = grid.resolution().as_hz();

    // Handle zero bandwidth edge case
    if chirp_bw < resolution {
        // Single-tone approximation
        let bin = grid.freq_to_bin(Hertz::new((start + end) / 2.0));
        if bin < psd.len() {
            psd[bin] += power.watts() / resolution;
        }
        return;
    }

    // Power distributed evenly across chirp bandwidth
    let power_per_hz = power.watts() / chirp_bw;
    let power_per_bin = power_per_hz * resolution;

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // Add power to each bin in sweep range
    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        psd[bin] += power_per_bin;
    }
}

/// Add chirp pattern with center frequency and bandwidth specification.
pub fn add_chirp_pattern_centered(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();
    let start_freq = Hertz::new(center - bw / 2.0);
    let end_freq = Hertz::new(center + bw / 2.0);
    add_chirp_pattern(psd, grid, start_freq, end_freq, power);
}

// ============================================================================
// SC-FDM Pattern (Single-Carrier Frequency Division Multiplexing)
// ============================================================================

/// Subband allocation type for SC-FDM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubbandAllocation {
    /// Localized: contiguous subcarriers (LTE uplink default)
    Localized,
    /// Distributed: interleaved subcarriers across bandwidth
    Distributed,
}

/// Add an SC-FDM (Single-Carrier FDM) spectral pattern to PSD.
///
/// SC-FDM uses DFT-spreading before OFDM to reduce PAPR. The spectral shape
/// is similar to OFDM but with:
/// - More concentrated energy (less flat-topped)
/// - Sinc-like roll-off at edges
/// - Lower PAPR making it suitable for power-efficient uplinks
///
/// Used in LTE uplink (SC-FDMA) and some drone telemetry systems.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center frequency
/// * `bandwidth` - Total bandwidth
/// * `num_subcarriers` - Number of active subcarriers
/// * `allocation` - Localized or distributed subband allocation
/// * `power` - Total power (guaranteed non-negative)
pub fn add_scfdm_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    num_subcarriers: usize,
    allocation: SubbandAllocation,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();

    // SC-FDM uses ~90% of bandwidth (similar to OFDM guard bands)
    let effective_bw = bw * 0.9;
    let subcarrier_spacing = effective_bw / num_subcarriers as f32;

    let low_freq = center - effective_bw / 2.0;
    let high_freq = center + effective_bw / 2.0;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    let resolution = grid.resolution().as_hz();
    let power_per_subcarrier = power.watts() / num_subcarriers as f32;

    match allocation {
        SubbandAllocation::Localized => {
            // Localized: concentrated in center with sinc-like roll-off
            // The DFT-spreading causes a sinc^2 envelope over the subcarriers
            let low_hz = low_freq.max(grid.freq_min().as_hz());
            let high_hz = high_freq.min(grid.freq_max().as_hz());
            let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
            let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

            // Apply sinc^2 envelope for more realistic SC-FDM shape
            let num_bins = bin_end.saturating_sub(bin_start) + 1;
            let center_bin_offset = num_bins as f32 / 2.0;

            for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
                let bin_offset = (bin - bin_start) as f32 - center_bin_offset;
                // Normalized position in [-0.5, 0.5]
                let normalized = bin_offset / num_bins as f32;

                // Sinc^2 envelope (approximated for stability at edges)
                let sinc_arg = normalized * constants::PI * 2.0;
                let envelope = if sinc_arg.abs() < 0.01 {
                    1.0 // Near center, sinc(0) = 1
                } else {
                    let sinc_val = sinc_arg.sin() / sinc_arg;
                    sinc_val * sinc_val
                };

                // Normalize so total power is conserved
                let power_density = power_per_subcarrier * envelope / subcarrier_spacing;
                psd[bin] += power_density * resolution;
            }
        }
        SubbandAllocation::Distributed => {
            // Distributed: subcarriers spread across the band (interleaved)
            // Power appears at regular intervals
            let spread_factor = num_subcarriers.max(1);
            let interleave_spacing = effective_bw / spread_factor as f32;

            for i in 0..num_subcarriers {
                let sc_freq = low_freq + (i as f32 + 0.5) * interleave_spacing;

                if sc_freq < grid.freq_min().as_hz() || sc_freq > grid.freq_max().as_hz() {
                    continue;
                }

                // Each subcarrier has narrow bandwidth
                let sc_bw = subcarrier_spacing * 0.8; // 80% of spacing
                let sc_low = (sc_freq - sc_bw / 2.0).max(grid.freq_min().as_hz());
                let sc_high = (sc_freq + sc_bw / 2.0).min(grid.freq_max().as_hz());

                let bin_start = grid.freq_to_bin(Hertz::new(sc_low));
                let bin_end = grid.freq_to_bin(Hertz::new(sc_high));

                let power_per_hz = power_per_subcarrier / sc_bw;
                let power_per_bin = power_per_hz * resolution;

                for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
                    psd[bin] += power_per_bin;
                }
            }
        }
    }
}

/// SIMD version of SC-FDM pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_scfdm_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    bandwidth: f32x8,
    num_subcarriers: usize,
    allocation: SubbandAllocation,
    power_linear: f32x8,
) {
    let effective_bw = bandwidth * f32x8::splat(0.9);
    let half_bw = effective_bw / f32x8::splat(2.0);
    let low_freq = center_freq - half_bw;
    let high_freq = center_freq + half_bw;

    let power_per_sc = power_linear / f32x8::splat(num_subcarriers as f32);
    let sc_spacing = effective_bw / f32x8::splat(num_subcarriers as f32);

    // Extract arrays for per-lane processing
    let _center_arr: [f32; 8] = center_freq.into(); // Used for debugging
    let low_arr: [f32; 8] = low_freq.into();
    let high_arr: [f32; 8] = high_freq.into();
    let power_per_sc_arr: [f32; 8] = power_per_sc.into();
    let sc_spacing_arr: [f32; 8] = sc_spacing.into();
    let effective_bw_arr: [f32; 8] = effective_bw.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();
    let resolution = grid.resolution().as_hz();

    for lane in 0..8 {
        if high_arr[lane] < freq_min || low_arr[lane] > freq_max {
            continue;
        }

        let psd_offset = env_offset + lane * num_bins;

        match allocation {
            SubbandAllocation::Localized => {
                let low_hz = low_arr[lane].max(freq_min);
                let high_hz = high_arr[lane].min(freq_max);
                let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
                let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

                let num_bins_signal = bin_end.saturating_sub(bin_start) + 1;
                let center_offset = num_bins_signal as f32 / 2.0;

                for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
                    let bin_offset = (bin - bin_start) as f32 - center_offset;
                    let normalized = bin_offset / num_bins_signal as f32;

                    let sinc_arg = normalized * constants::PI * 2.0;
                    let envelope = if sinc_arg.abs() < 0.01 {
                        1.0
                    } else {
                        let sinc_val = sinc_arg.sin() / sinc_arg;
                        sinc_val * sinc_val
                    };

                    let power_density = power_per_sc_arr[lane] * envelope / sc_spacing_arr[lane];
                    psd[psd_offset + bin] += power_density * resolution;
                }
            }
            SubbandAllocation::Distributed => {
                let interleave_spacing = effective_bw_arr[lane] / num_subcarriers as f32;

                for i in 0..num_subcarriers {
                    let sc_freq = low_arr[lane] + (i as f32 + 0.5) * interleave_spacing;

                    if sc_freq < freq_min || sc_freq > freq_max {
                        continue;
                    }

                    let sc_bw = sc_spacing_arr[lane] * 0.8;
                    let sc_low = (sc_freq - sc_bw / 2.0).max(freq_min);
                    let sc_high = (sc_freq + sc_bw / 2.0).min(freq_max);

                    let bin_start = grid.freq_to_bin(Hertz::new(sc_low));
                    let bin_end = grid.freq_to_bin(Hertz::new(sc_high));

                    let power_per_hz = power_per_sc_arr[lane] / sc_bw;
                    let power_per_bin = power_per_hz * resolution;

                    for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
                        psd[psd_offset + bin] += power_per_bin;
                    }
                }
            }
        }
    }
}

// ============================================================================
// OOK/ASK Pattern (On-Off Keying / Amplitude Shift Keying)
// ============================================================================

/// Add an OOK (On-Off Keying) spectral pattern to PSD.
///
/// OOK modulation produces a sinc-squared spectral shape due to the
/// rectangular pulse shaping. The bandwidth is approximately 2x the symbol rate.
///
/// Used in key fobs (315/433 MHz), RFID tags, and simple IoT devices.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center/carrier frequency
/// * `symbol_rate` - Symbol rate in symbols/second (determines bandwidth)
/// * `power` - Total power (guaranteed non-negative)
pub fn add_ook_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    symbol_rate: f32,
    power: PositivePower,
) {
    let center = center_freq.as_hz();

    // OOK bandwidth ≈ 2 × symbol rate for first null
    // We extend to 3x for sidelobe capture
    let main_lobe_bw = 2.0 * symbol_rate;
    let extend_bw = 3.0 * symbol_rate;

    let low_freq = center - extend_bw;
    let high_freq = center + extend_bw;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    let resolution = grid.resolution().as_hz();

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // sinc^2 power spectral density: P(f) = sinc^2(f/symbol_rate)
    // Normalize so total power integrates to power_linear
    let norm_factor = power.watts() / main_lobe_bw; // Approximate normalization

    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        let bin_freq = grid.bin_to_freq(bin).as_hz();
        let freq_offset = bin_freq - center;

        // sinc^2 envelope
        let sinc_arg = freq_offset / symbol_rate;
        let sinc_sq = if sinc_arg.abs() < 0.01 {
            1.0
        } else {
            let sinc_val = (constants::PI * sinc_arg).sin() / (constants::PI * sinc_arg);
            sinc_val * sinc_val
        };

        psd[bin] += norm_factor * sinc_sq * resolution;
    }
}

/// SIMD version of OOK pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_ook_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    symbol_rate: f32x8,
    power_linear: f32x8,
) {
    let main_lobe_bw = symbol_rate * f32x8::splat(2.0);
    let extend_bw = symbol_rate * f32x8::splat(3.0);
    let low_freq = center_freq - extend_bw;
    let high_freq = center_freq + extend_bw;

    let norm_factor = power_linear / main_lobe_bw;

    // Extract arrays
    let center_arr: [f32; 8] = center_freq.into();
    let low_arr: [f32; 8] = low_freq.into();
    let high_arr: [f32; 8] = high_freq.into();
    let symbol_rate_arr: [f32; 8] = symbol_rate.into();
    let norm_arr: [f32; 8] = norm_factor.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();
    let resolution = grid.resolution().as_hz();

    for lane in 0..8 {
        if high_arr[lane] < freq_min || low_arr[lane] > freq_max {
            continue;
        }

        let bin_start = grid.freq_to_bin(Hertz::new(low_arr[lane].max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_arr[lane].min(freq_max)));

        let psd_offset = env_offset + lane * num_bins;

        for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
            let bin_freq = grid.bin_to_freq(bin).as_hz();
            let freq_offset = bin_freq - center_arr[lane];
            let sinc_arg = freq_offset / symbol_rate_arr[lane];

            let sinc_sq = if sinc_arg.abs() < 0.01 {
                1.0
            } else {
                let sinc_val = (constants::PI * sinc_arg).sin() / (constants::PI * sinc_arg);
                sinc_val * sinc_val
            };

            psd[psd_offset + bin] += norm_arr[lane] * sinc_sq * resolution;
        }
    }
}

// ============================================================================
// C4FM Pattern (Continuous 4-level FM)
// ============================================================================

/// Add a C4FM (4-level FSK) spectral pattern to PSD.
///
/// C4FM is a constant-envelope modulation used in P25 public safety radios.
/// It produces a spectral shape similar to filtered 4-FSK with specified
/// frequency deviation.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center frequency
/// * `deviation` - Peak frequency deviation (±1.8 kHz typical for P25)
/// * `symbol_rate` - Symbol rate (4800 baud for P25)
/// * `power` - Total power (guaranteed non-negative)
pub fn add_c4fm_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    deviation: f32,
    symbol_rate: f32,
    power: PositivePower,
) {
    let center = center_freq.as_hz();

    // C4FM bandwidth: Carson's rule approximation
    // BW ≈ 2 × (deviation + symbol_rate/2)
    let carson_bw = 2.0 * (deviation + symbol_rate / 2.0);

    // 4 frequency levels at ±deviation and ±deviation/3
    let levels = [-deviation, -deviation / 3.0, deviation / 3.0, deviation];

    let low_freq = center - carson_bw;
    let high_freq = center + carson_bw;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    let resolution = grid.resolution().as_hz();

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // C4FM has 4 spectral peaks corresponding to the 4 deviation levels
    // Each level contributes 1/4 of the power with Gaussian-like spreading
    let power_per_level = power.watts() / 4.0;
    let level_sigma = symbol_rate / 2.0; // Spreading due to data transitions

    let sqrt_2pi = (2.0 * constants::PI).sqrt();
    let norm = power_per_level / (level_sigma * sqrt_2pi);

    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        let bin_freq = grid.bin_to_freq(bin).as_hz();

        let mut bin_power = 0.0;
        for &level in &levels {
            let diff = bin_freq - (center + level);
            let exponent = -0.5 * (diff / level_sigma).powi(2);
            bin_power += norm * exponent.exp();
        }

        psd[bin] += bin_power * resolution;
    }
}

/// SIMD version of C4FM pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_c4fm_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    deviation: f32x8,
    symbol_rate: f32x8,
    power_linear: f32x8,
) {
    let half_symbol_rate = symbol_rate / f32x8::splat(2.0);
    let carson_bw = (deviation + half_symbol_rate) * f32x8::splat(2.0);
    let low_freq = center_freq - carson_bw;
    let high_freq = center_freq + carson_bw;

    let power_per_level = power_linear / f32x8::splat(4.0);
    let level_sigma = half_symbol_rate;
    let sqrt_2pi = f32x8::splat((2.0 * constants::PI).sqrt());
    let norm = power_per_level / (level_sigma * sqrt_2pi);

    // Extract arrays
    let center_arr: [f32; 8] = center_freq.into();
    let deviation_arr: [f32; 8] = deviation.into();
    let low_arr: [f32; 8] = low_freq.into();
    let high_arr: [f32; 8] = high_freq.into();
    let norm_arr: [f32; 8] = norm.into();
    let sigma_arr: [f32; 8] = level_sigma.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();
    let resolution = grid.resolution().as_hz();

    for lane in 0..8 {
        if high_arr[lane] < freq_min || low_arr[lane] > freq_max {
            continue;
        }

        let bin_start = grid.freq_to_bin(Hertz::new(low_arr[lane].max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_arr[lane].min(freq_max)));

        let psd_offset = env_offset + lane * num_bins;
        let dev = deviation_arr[lane];
        let levels = [-dev, -dev / 3.0, dev / 3.0, dev];

        for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
            let bin_freq = grid.bin_to_freq(bin).as_hz();

            let mut bin_power = 0.0;
            for &level in &levels {
                let diff = bin_freq - (center_arr[lane] + level);
                let exponent = -0.5 * (diff / sigma_arr[lane]).powi(2);
                bin_power += norm_arr[lane] * exponent.exp();
            }

            psd[psd_offset + bin] += bin_power * resolution;
        }
    }
}

// ============================================================================
// SSB Pattern (Single Sideband)
// ============================================================================

/// Sideband selection for SSB modulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sideband {
    /// Upper Sideband (USB) - common above 10 MHz
    Upper,
    /// Lower Sideband (LSB) - common below 10 MHz
    Lower,
}

/// Add an SSB (Single Sideband) spectral pattern to PSD.
///
/// SSB modulation suppresses the carrier and one sideband, producing
/// an asymmetric spectral shape. Used extensively in HF amateur radio
/// and marine communications.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Suppressed carrier frequency
/// * `bandwidth` - Audio bandwidth (typically 2.7-3.0 kHz)
/// * `sideband` - Upper or Lower sideband selection
/// * `power` - Total power (guaranteed non-negative)
pub fn add_ssb_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    sideband: Sideband,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();

    // SSB occupies only one side of the carrier
    let (low_freq, high_freq) = match sideband {
        Sideband::Upper => (center, center + bw),
        Sideband::Lower => (center - bw, center),
    };

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    let resolution = grid.resolution().as_hz();

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // SSB spectral shape: voice frequencies peak around 300-3000 Hz
    // Model as asymmetric with roll-off at edges
    // Power concentrated in 300-2700 Hz range (relative to carrier)
    let voice_low = 300.0;
    let voice_high = 2700.0;
    let voice_bw = voice_high - voice_low;

    // Normalization for power conservation
    let norm = power.watts() / (voice_bw * 1.2); // 1.2 factor for roll-off regions

    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        let bin_freq = grid.bin_to_freq(bin).as_hz();

        // Calculate offset from carrier in audio band
        let audio_offset = match sideband {
            Sideband::Upper => bin_freq - center,
            Sideband::Lower => center - bin_freq,
        };

        // Spectral shape: flat in voice band with roll-off
        let envelope = if audio_offset < voice_low {
            // Roll-off below 300 Hz
            let roll = (audio_offset / voice_low).max(0.0);
            roll * roll // Quadratic roll-off
        } else if audio_offset > voice_high {
            // Roll-off above 2700 Hz
            let excess = audio_offset - voice_high;
            let roll_factor = 1.0 - (excess / (bw - voice_high)).min(1.0);
            roll_factor * roll_factor
        } else {
            // Flat in voice band
            1.0
        };

        psd[bin] += norm * envelope * resolution;
    }
}

/// SIMD version of SSB pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_ssb_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    bandwidth: f32x8,
    sideband: Sideband,
    power_linear: f32x8,
) {
    // Voice band parameters (same for all lanes)
    let voice_low = 300.0_f32;
    let voice_high = 2700.0_f32;
    let voice_bw = voice_high - voice_low;

    let norm = power_linear / f32x8::splat(voice_bw * 1.2);

    // Calculate frequency ranges based on sideband
    let (low_freq, high_freq) = match sideband {
        Sideband::Upper => (center_freq, center_freq + bandwidth),
        Sideband::Lower => (center_freq - bandwidth, center_freq),
    };

    // Extract arrays
    let center_arr: [f32; 8] = center_freq.into();
    let bw_arr: [f32; 8] = bandwidth.into();
    let low_arr: [f32; 8] = low_freq.into();
    let high_arr: [f32; 8] = high_freq.into();
    let norm_arr: [f32; 8] = norm.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();
    let resolution = grid.resolution().as_hz();

    for lane in 0..8 {
        if high_arr[lane] < freq_min || low_arr[lane] > freq_max {
            continue;
        }

        let bin_start = grid.freq_to_bin(Hertz::new(low_arr[lane].max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_arr[lane].min(freq_max)));

        let psd_offset = env_offset + lane * num_bins;

        for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
            let bin_freq = grid.bin_to_freq(bin).as_hz();

            let audio_offset = match sideband {
                Sideband::Upper => bin_freq - center_arr[lane],
                Sideband::Lower => center_arr[lane] - bin_freq,
            };

            let envelope = if audio_offset < voice_low {
                let roll = (audio_offset / voice_low).max(0.0);
                roll * roll
            } else if audio_offset > voice_high {
                let excess = audio_offset - voice_high;
                let roll_factor = 1.0 - (excess / (bw_arr[lane] - voice_high)).min(1.0);
                roll_factor * roll_factor
            } else {
                1.0
            };

            psd[psd_offset + bin] += norm_arr[lane] * envelope * resolution;
        }
    }
}

// ============================================================================
// COFDM Pattern (Coded OFDM)
// ============================================================================

/// Add a COFDM (Coded OFDM) spectral pattern to PSD.
///
/// COFDM is OFDM with forward error correction (FEC). Spectrally similar
/// to OFDM but typically uses:
/// - Higher pilot density (1/3 vs 1/12)
/// - Longer guard intervals (1/4 vs 1/8)
///
/// Used in digital video broadcasting (DVB-T), digital radio (DAB),
/// and some enhanced drone video systems.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `center_freq` - Center frequency
/// * `bandwidth` - Total bandwidth
/// * `pilot_density` - Fraction of subcarriers used for pilots (e.g., 0.33)
/// * `power` - Total power (guaranteed non-negative)
pub fn add_cofdm_pattern(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    center_freq: Hertz,
    bandwidth: Hertz,
    pilot_density: f32,
    power: PositivePower,
) {
    let center = center_freq.as_hz();
    let bw = bandwidth.as_hz();

    // COFDM typically uses 85% bandwidth (larger guard bands than OFDM)
    let effective_bw = bw * 0.85;

    let low_freq = center - effective_bw / 2.0;
    let high_freq = center + effective_bw / 2.0;

    // Check if signal is within grid range
    if high_freq < grid.freq_min().as_hz() || low_freq > grid.freq_max().as_hz() {
        return;
    }

    let resolution = grid.resolution().as_hz();

    // Find affected bins
    let low_hz = low_freq.max(grid.freq_min().as_hz());
    let high_hz = high_freq.min(grid.freq_max().as_hz());
    let bin_start = grid.freq_to_bin(Hertz::new(low_hz));
    let bin_end = grid.freq_to_bin(Hertz::new(high_hz));

    // Power split between data and pilots
    // Pilots are typically 3dB higher in power for channel estimation
    let pilot_boost = 2.0; // 3dB = 2x linear
    let data_fraction = 1.0 - pilot_density;
    let effective_power = power.watts() / (data_fraction + pilot_density * pilot_boost);
    let data_power_per_hz = effective_power * data_fraction / effective_bw;
    let pilot_power_per_hz = effective_power * pilot_density * pilot_boost / effective_bw;

    let power_per_bin = (data_power_per_hz + pilot_power_per_hz) * resolution;

    // COFDM is relatively flat within the band
    for bin in bin_start..=bin_end.min(psd.len().saturating_sub(1)) {
        psd[bin] += power_per_bin;
    }
}

/// SIMD version of COFDM pattern for 8 environments.
#[cfg(feature = "simd")]
pub fn add_cofdm_pattern_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    center_freq: f32x8,
    bandwidth: f32x8,
    pilot_density: f32,
    power_linear: f32x8,
) {
    let effective_bw = bandwidth * f32x8::splat(0.85);
    let half_bw = effective_bw / f32x8::splat(2.0);
    let low_freq = center_freq - half_bw;
    let high_freq = center_freq + half_bw;

    // Power calculation
    let pilot_boost = 2.0_f32;
    let data_fraction = 1.0 - pilot_density;
    let effective_power = power_linear / f32x8::splat(data_fraction + pilot_density * pilot_boost);
    let total_power_density = effective_power / effective_bw;
    let power_per_bin = total_power_density * f32x8::splat(grid.resolution().as_hz());

    // Extract arrays
    let low_arr: [f32; 8] = low_freq.into();
    let high_arr: [f32; 8] = high_freq.into();
    let power_arr: [f32; 8] = power_per_bin.into();

    let freq_min = grid.freq_min().as_hz();
    let freq_max = grid.freq_max().as_hz();

    for lane in 0..8 {
        if high_arr[lane] < freq_min || low_arr[lane] > freq_max {
            continue;
        }

        let bin_start = grid.freq_to_bin(Hertz::new(low_arr[lane].max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new(high_arr[lane].min(freq_max)));

        let psd_offset = env_offset + lane * num_bins;

        for bin in bin_start..=bin_end.min(num_bins.saturating_sub(1)) {
            psd[psd_offset + bin] += power_arr[lane];
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Convert power in dBm to linear Watts.
#[inline]
pub fn dbm_to_watts(dbm: f32) -> f32 {
    10.0_f32.powf(dbm / 10.0) / 1000.0
}

/// Convert power in Watts to dBm.
#[inline]
pub fn watts_to_dbm(watts: f32) -> f32 {
    10.0 * (watts * 1000.0).log10()
}

/// Convert dBm to PositivePower.
#[inline]
pub fn dbm_to_power(dbm: f32) -> PositivePower {
    PositivePower::new(dbm_to_watts(dbm))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> ValidatedFrequencyGrid {
        ValidatedFrequencyGrid::from_params(1e9, 2e9, 1000).unwrap()
    }

    #[test]
    fn test_rect_pattern_basic() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        // 10 MHz bandwidth centered at 1.5 GHz, 1 mW total
        add_rect_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(10e6),
            PositivePower::new(0.001),
        );

        // Check center has power
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        assert!(psd[center_bin] > 0.0);

        // Check outside band has no power
        let out_bin = grid.freq_to_bin(Hertz::new(1.2e9));
        assert_eq!(psd[out_bin], 0.0);
    }

    #[test]
    fn test_rect_pattern_power_conservation() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        let total_power = PositivePower::new(0.001);
        add_rect_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(100e6),
            total_power,
        );

        let measured_power: f32 = psd.iter().sum();

        // Should be close to total power (within 10% numerical tolerance)
        assert!(
            (measured_power - total_power.watts()).abs() < total_power.watts() * 0.1,
            "Power not conserved: expected {}, got {}",
            total_power.watts(),
            measured_power
        );
    }

    #[test]
    fn test_gaussian_pattern_basic() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_gaussian_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(1e6),
            PositivePower::new(0.001),
        );

        // Check center has peak power
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        let center_power = psd[center_bin];
        assert!(center_power > 0.0);

        // Check power decreases away from center
        let offset_bin = grid.freq_to_bin(Hertz::new(1.5e9 + 2e6));
        assert!(psd[offset_bin] < center_power);
    }

    #[test]
    fn test_gaussian_pattern_peak_at_center() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_gaussian_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(5e6),
            PositivePower::new(0.001),
        );

        // Find peak
        let (max_idx, _) = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Peak should be at or very near center frequency
        let expected_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        assert!(
            (max_idx as i32 - expected_bin as i32).abs() <= 1,
            "Peak at {} but expected {}",
            max_idx,
            expected_bin
        );
    }

    #[test]
    fn test_ofdm_pattern_flat() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_ofdm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(20e6),
            PositivePower::new(0.001),
        );

        // Check that power is relatively flat in the band
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        let offset_bin = grid.freq_to_bin(Hertz::new(1.5e9 + 5e6));

        let center_power = psd[center_bin];
        let offset_power = psd[offset_bin];

        // Both should have similar power (OFDM is flat-topped)
        if center_power > 0.0 && offset_power > 0.0 {
            let ratio = center_power / offset_power;
            assert!(ratio > 0.9 && ratio < 1.1, "OFDM should be flat");
        }
    }

    #[test]
    fn test_chirp_pattern_coverage() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        // Chirp from 1.4 GHz to 1.6 GHz
        add_chirp_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.4e9),
            Hertz::new(1.6e9),
            PositivePower::new(0.001),
        );

        // Check that power exists across the sweep range
        let start_bin = grid.freq_to_bin(Hertz::new(1.4e9));
        let mid_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        let end_bin = grid.freq_to_bin(Hertz::new(1.6e9));

        assert!(psd[start_bin] > 0.0);
        assert!(psd[mid_bin] > 0.0);
        assert!(psd[end_bin] > 0.0);

        // Check outside range has no power
        let before_bin = grid.freq_to_bin(Hertz::new(1.3e9));
        let after_bin = grid.freq_to_bin(Hertz::new(1.7e9));
        assert_eq!(psd[before_bin], 0.0);
        assert_eq!(psd[after_bin], 0.0);
    }

    #[test]
    fn test_dbm_watts_conversion() {
        // 0 dBm = 1 mW = 0.001 W
        let watts = dbm_to_watts(0.0);
        assert!((watts - 0.001).abs() < 1e-9);

        // 30 dBm = 1 W
        let watts = dbm_to_watts(30.0);
        assert!((watts - 1.0).abs() < 1e-6);

        // Round-trip
        let original = 0.01; // 10 mW
        let dbm = watts_to_dbm(original);
        let back = dbm_to_watts(dbm);
        assert!((back - original).abs() < 1e-9);
    }

    #[test]
    fn test_pattern_out_of_range() {
        let grid = test_grid(); // 1-2 GHz
        let mut psd = vec![0.0; grid.num_bins()];

        // Signal completely out of range
        add_rect_pattern(
            &mut psd,
            &grid,
            Hertz::new(500e6),
            Hertz::new(10e6),
            PositivePower::new(0.001),
        );

        // PSD should be unchanged
        assert!(psd.iter().all(|&x| x == 0.0));
    }

    // ========================================================================
    // SC-FDM Pattern Tests
    // ========================================================================

    #[test]
    fn test_scfdm_pattern_localized() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_scfdm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(10e6),
            12, // 12 subcarriers
            SubbandAllocation::Localized,
            PositivePower::new(0.001),
        );

        // Check center has power
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        assert!(psd[center_bin] > 0.0);

        // Check edge has less power (sinc roll-off)
        let edge_bin = grid.freq_to_bin(Hertz::new(1.5e9 + 4e6));
        assert!(psd[edge_bin] < psd[center_bin] || psd[edge_bin] == 0.0);
    }

    #[test]
    fn test_scfdm_pattern_distributed() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_scfdm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(20e6),
            6, // 6 subcarriers distributed
            SubbandAllocation::Distributed,
            PositivePower::new(0.001),
        );

        // Should have multiple peaks across bandwidth
        let total_power: f32 = psd.iter().sum();
        assert!(total_power > 0.0);
    }

    // ========================================================================
    // OOK Pattern Tests
    // ========================================================================

    #[test]
    fn test_ook_pattern_basic() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        // Key fob style: 433 MHz, 4800 baud - but use grid range
        add_ook_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            10000.0, // 10 kbps symbol rate
            PositivePower::new(0.001),
        );

        // Peak should be at center
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        let max_bin = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert!(
            (max_bin as i32 - center_bin as i32).abs() <= 2,
            "OOK peak should be near center"
        );
    }

    #[test]
    fn test_ook_pattern_sinc_shape() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_ook_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            1e6, // 1 MHz symbol rate for visible BW
            PositivePower::new(0.001),
        );

        // Check that power decreases away from center (sinc shape)
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        let offset_bin = grid.freq_to_bin(Hertz::new(1.5e9 + 1.5e6));

        assert!(
            psd[center_bin] > psd[offset_bin],
            "OOK should have sinc shape with lower power at offset"
        );
    }

    // ========================================================================
    // C4FM Pattern Tests
    // ========================================================================

    #[test]
    fn test_c4fm_pattern_basic() {
        let grid = ValidatedFrequencyGrid::from_params(1.4999e9, 1.5001e9, 1000).unwrap();
        let mut psd = vec![0.0; grid.num_bins()];

        // P25 parameters
        add_c4fm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            1800.0, // ±1.8 kHz deviation
            4800.0, // 4800 baud
            PositivePower::new(0.001),
        );

        // Should have power in the signal band
        let total_power: f32 = psd.iter().sum();
        assert!(total_power > 0.0);

        // C4FM has 4 peaks - check multiple peaks exist
        let non_zero_bins: usize = psd.iter().filter(|&&x| x > 0.0).count();
        assert!(non_zero_bins > 10, "C4FM should occupy multiple bins");
    }

    #[test]
    fn test_c4fm_pattern_four_levels() {
        // Use high-resolution grid to see the 4 spectral peaks
        // C4FM has peaks at ±deviation and ±deviation/3
        // With 1800 Hz deviation: peaks at -1800, -600, +600, +1800 Hz
        let grid = ValidatedFrequencyGrid::from_params(
            1.5e9 - 10000.0,
            1.5e9 + 10000.0,
            2000, // Higher resolution
        ).unwrap();
        let mut psd = vec![0.0; grid.num_bins()];

        add_c4fm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            1800.0,
            4800.0,
            PositivePower::new(0.001),
        );

        // Verify that power exists at the expected deviation frequencies
        let center = 1.5e9;
        let deviation = 1800.0;

        // Check power at each of the 4 expected frequencies
        let expected_freqs = [
            center - deviation,      // -1800 Hz
            center - deviation / 3.0, // -600 Hz
            center + deviation / 3.0, // +600 Hz
            center + deviation,      // +1800 Hz
        ];

        for &freq in &expected_freqs {
            let bin = grid.freq_to_bin(Hertz::new(freq));
            assert!(
                psd[bin] > 0.0,
                "C4FM should have power at offset {} Hz",
                freq - center
            );
        }

        // Verify the spectral shape spans the expected bandwidth
        let non_zero_bins: usize = psd.iter().filter(|&&x| x > 1e-15).count();
        assert!(
            non_zero_bins > 50,
            "C4FM should occupy substantial bandwidth, got {} bins",
            non_zero_bins
        );
    }

    // ========================================================================
    // SSB Pattern Tests
    // ========================================================================

    #[test]
    fn test_ssb_pattern_upper() {
        let grid = ValidatedFrequencyGrid::from_params(
            14.0e6 - 5000.0,
            14.0e6 + 5000.0,
            500,
        ).unwrap();
        let mut psd = vec![0.0; grid.num_bins()];

        add_ssb_pattern(
            &mut psd,
            &grid,
            Hertz::new(14.0e6),
            Hertz::new(3000.0),
            Sideband::Upper,
            PositivePower::new(0.001),
        );

        // USB should have power ABOVE carrier only
        let carrier_bin = grid.freq_to_bin(Hertz::new(14.0e6));
        let above_bin = grid.freq_to_bin(Hertz::new(14.0e6 + 1500.0));
        let below_bin = grid.freq_to_bin(Hertz::new(14.0e6 - 1500.0));

        assert!(psd[above_bin] > 0.0, "USB should have power above carrier");
        assert!(
            psd[below_bin] < psd[above_bin] * 0.1,
            "USB should have minimal power below carrier"
        );
    }

    #[test]
    fn test_ssb_pattern_lower() {
        let grid = ValidatedFrequencyGrid::from_params(
            7.0e6 - 5000.0,
            7.0e6 + 5000.0,
            500,
        ).unwrap();
        let mut psd = vec![0.0; grid.num_bins()];

        add_ssb_pattern(
            &mut psd,
            &grid,
            Hertz::new(7.0e6),
            Hertz::new(3000.0),
            Sideband::Lower,
            PositivePower::new(0.001),
        );

        // LSB should have power BELOW carrier only
        let above_bin = grid.freq_to_bin(Hertz::new(7.0e6 + 1500.0));
        let below_bin = grid.freq_to_bin(Hertz::new(7.0e6 - 1500.0));

        assert!(psd[below_bin] > 0.0, "LSB should have power below carrier");
        assert!(
            psd[above_bin] < psd[below_bin] * 0.1,
            "LSB should have minimal power above carrier"
        );
    }

    // ========================================================================
    // COFDM Pattern Tests
    // ========================================================================

    #[test]
    fn test_cofdm_pattern_basic() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_cofdm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(8e6), // 8 MHz DVB-T style
            0.33, // 1/3 pilot density
            PositivePower::new(0.001),
        );

        // Check center has power
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        assert!(psd[center_bin] > 0.0);
    }

    #[test]
    fn test_cofdm_pattern_flat() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_cofdm_pattern(
            &mut psd,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(20e6),
            0.25,
            PositivePower::new(0.001),
        );

        // COFDM should be relatively flat in-band
        let center_bin = grid.freq_to_bin(Hertz::new(1.5e9));
        let offset_bin = grid.freq_to_bin(Hertz::new(1.5e9 + 5e6));

        let center_power = psd[center_bin];
        let offset_power = psd[offset_bin];

        if center_power > 0.0 && offset_power > 0.0 {
            let ratio = center_power / offset_power;
            assert!(
                ratio > 0.9 && ratio < 1.1,
                "COFDM should be flat, ratio: {}",
                ratio
            );
        }
    }

    #[test]
    fn test_cofdm_narrower_than_ofdm() {
        let grid = test_grid();

        // Compare OFDM and COFDM with same bandwidth
        let mut psd_ofdm = vec![0.0; grid.num_bins()];
        let mut psd_cofdm = vec![0.0; grid.num_bins()];

        add_ofdm_pattern(
            &mut psd_ofdm,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(20e6),
            PositivePower::new(0.001),
        );

        add_cofdm_pattern(
            &mut psd_cofdm,
            &grid,
            Hertz::new(1.5e9),
            Hertz::new(20e6),
            0.33,
            PositivePower::new(0.001),
        );

        // Count occupied bins for each
        let ofdm_bins: usize = psd_ofdm.iter().filter(|&&x| x > 1e-15).count();
        let cofdm_bins: usize = psd_cofdm.iter().filter(|&&x| x > 1e-15).count();

        // COFDM uses 85% BW vs OFDM 90% BW, so should occupy fewer bins
        assert!(
            cofdm_bins <= ofdm_bins,
            "COFDM should occupy fewer bins ({}) than OFDM ({})",
            cofdm_bins,
            ofdm_bins
        );
    }
}
