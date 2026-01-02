//! Noise Floor Models
//!
//! Implements thermal and man-made noise models for realistic RF environments:
//! - Thermal noise: kTB at -174 dBm/Hz + receiver noise figure
//! - Man-made noise: ITU-R P.372 model for various environments
//!
//! All functions use type-safe `PositivePower`, `Hertz`, and
//! `ValidatedFrequencyGrid` types to guarantee correctness at compile time.

use crate::types::primitives::PositivePower;
use crate::types::dimensional::Hertz;
use crate::types::frequency::ValidatedFrequencyGrid;
use crate::types::power::PowerDbm;

#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

// ============================================================================
// Noise Environment Types
// ============================================================================

/// Noise environment classification per ITU-R P.372.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseEnvironment {
    /// City/industrial environment (highest man-made noise)
    City,
    /// Residential/suburban environment
    Residential,
    /// Rural environment (lowest man-made noise)
    Rural,
    /// Quiet rural (minimal man-made noise, mainly galactic)
    QuietRural,
}

impl Default for NoiseEnvironment {
    fn default() -> Self {
        Self::Residential
    }
}

// ============================================================================
// Constants
// ============================================================================

/// Thermal noise power spectral density at room temperature.
/// -174 dBm/Hz at 290K (17°C)
const THERMAL_NOISE_DBM_PER_HZ: f32 = -174.0;

/// Default receiver noise figure in dB (typical SDR)
const DEFAULT_NOISE_FIGURE_DB: f32 = 6.0;

/// Boltzmann constant (J/K)
const K_BOLTZMANN: f32 = 1.380649e-23;

/// Reference temperature (K)
const T_0: f32 = 290.0;

// ============================================================================
// Thermal Noise
// ============================================================================

/// Add thermal noise floor to PSD.
///
/// Thermal noise is frequency-independent (white noise) with power kTB.
/// The effective noise floor includes the receiver noise figure.
///
/// Returns the noise power per bin that was added.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `noise_figure_db` - Receiver noise figure in dB
pub fn add_thermal_noise(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    noise_figure_db: f32,
) -> PositivePower {
    // Effective noise floor in dBm/Hz
    let noise_floor_dbm_per_hz = THERMAL_NOISE_DBM_PER_HZ + noise_figure_db;

    // Convert to linear power per Hz
    let noise_psd_linear = 10.0_f32.powf(noise_floor_dbm_per_hz / 10.0) / 1000.0;

    // Power per frequency bin
    let noise_per_bin = noise_psd_linear * grid.resolution().as_hz();

    // Add to all bins
    for val in psd.iter_mut() {
        *val += noise_per_bin;
    }

    PositivePower::new(noise_per_bin)
}

/// Add thermal noise with default noise figure.
pub fn add_thermal_noise_default(psd: &mut [f32], grid: &ValidatedFrequencyGrid) -> PositivePower {
    add_thermal_noise(psd, grid, DEFAULT_NOISE_FIGURE_DB)
}

// ============================================================================
// Man-Made Noise (ITU-R P.372)
// ============================================================================

/// Calculate man-made noise figure per ITU-R P.372.
///
/// F_am = c - d * log10(f_MHz)
///
/// Where c and d are environment-dependent constants.
fn man_made_noise_figure_db(freq_mhz: f32, environment: NoiseEnvironment) -> f32 {
    // ITU-R P.372-14 Table 2: Man-made noise
    let (c, d) = match environment {
        NoiseEnvironment::City => (76.8, 27.7),
        NoiseEnvironment::Residential => (72.5, 27.7),
        NoiseEnvironment::Rural => (67.2, 27.7),
        NoiseEnvironment::QuietRural => (53.6, 28.6), // Galactic + minimal man-made
    };

    // Clamp frequency to valid range (0.3 MHz to 250 MHz for man-made)
    // Above ~250 MHz, man-made noise becomes negligible
    let freq_clamped = freq_mhz.max(0.3).min(250.0);

    c - d * freq_clamped.log10()
}

/// Add man-made noise to PSD per ITU-R P.372.
///
/// Man-made noise is frequency-dependent, higher at lower frequencies.
/// This adds realistic urban/rural noise characteristics.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `environment` - Noise environment type
pub fn add_man_made_noise(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    environment: NoiseEnvironment,
) {
    let resolution = grid.resolution().as_hz();

    for bin in 0..grid.num_bins() {
        let freq_hz = grid.bin_to_freq(bin).as_hz();
        let freq_mhz = freq_hz / 1e6;

        // Man-made noise is only significant below ~250 MHz
        if freq_mhz > 250.0 {
            continue;
        }

        // Calculate noise figure
        let f_am_db = man_made_noise_figure_db(freq_mhz, environment);

        // Convert to external noise factor
        // F_a = 10^(F_am/10)
        let f_a = 10.0_f32.powf(f_am_db / 10.0);

        // External noise temperature
        // T_a = T_0 * (F_a - 1) where T_0 = 290K
        let t_a = T_0 * (f_a - 1.0);

        // Noise power spectral density
        // N = k * T_a where k = Boltzmann constant
        let noise_psd = K_BOLTZMANN * t_a;

        // Power per bin
        let noise_per_bin = noise_psd * resolution;

        if bin < psd.len() {
            psd[bin] += noise_per_bin;
        }
    }
}

// ============================================================================
// Combined Noise Floor
// ============================================================================

/// Add complete noise floor (thermal + man-made) to PSD.
///
/// Returns the thermal noise power per bin.
///
/// # Arguments
/// * `psd` - PSD slice for a single environment
/// * `grid` - Validated frequency grid
/// * `noise_figure_db` - Receiver noise figure in dB
/// * `environment` - Noise environment type
pub fn add_noise_floor(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    noise_figure_db: f32,
    environment: NoiseEnvironment,
) -> PositivePower {
    let thermal = add_thermal_noise(psd, grid, noise_figure_db);
    add_man_made_noise(psd, grid, environment);
    thermal
}

/// Add noise floor with temporal variation (more realistic).
///
/// Adds small random fluctuations to model real-world noise variance.
#[cfg(feature = "simd")]
pub fn add_noise_floor_simd(
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env_offset: usize,
    num_bins: usize,
    noise_figure_db: f32,
    environment: NoiseEnvironment,
    rng: &mut SimdRng,
) {
    // Calculate base noise floor
    let noise_floor_dbm_per_hz = THERMAL_NOISE_DBM_PER_HZ + noise_figure_db;
    let base_noise_psd = 10.0_f32.powf(noise_floor_dbm_per_hz / 10.0) / 1000.0;
    let resolution = grid.resolution().as_hz();
    let base_noise_per_bin = base_noise_psd * resolution;

    // Add noise with small variance (~1 dB standard deviation)
    let variance_factor = 0.26; // ~1 dB in linear scale

    // Process bins in SIMD chunks
    for bin_base in (0..num_bins).step_by(8) {
        // Generate random variation
        let u = rng.uniform();
        let variation: [f32; 8] = u.into();

        for i in 0..8 {
            let bin = bin_base + i;
            if bin >= num_bins {
                break;
            }

            // Calculate man-made noise for this bin
            let freq_hz = grid.bin_to_freq(bin).as_hz();
            let freq_mhz = freq_hz / 1e6;

            let man_made_noise = if freq_mhz <= 250.0 {
                let f_am_db = man_made_noise_figure_db(freq_mhz, environment);
                let f_a = 10.0_f32.powf(f_am_db / 10.0);
                let t_a = T_0 * (f_a - 1.0);
                K_BOLTZMANN * t_a * resolution
            } else {
                0.0
            };

            // Apply random variation (log-normal distribution approximation)
            let noise_variation = 1.0 + variance_factor * (variation[i] - 0.5) * 2.0;

            let total_noise = (base_noise_per_bin + man_made_noise) * noise_variation;
            psd[env_offset + bin] += total_noise;
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Calculate noise floor level in dBm for a given bandwidth.
pub fn noise_floor_dbm(bandwidth: Hertz, noise_figure_db: f32) -> PowerDbm {
    let dbm = THERMAL_NOISE_DBM_PER_HZ + 10.0 * bandwidth.as_hz().log10() + noise_figure_db;
    PowerDbm::new(dbm)
}

/// Calculate signal-to-noise ratio.
pub fn calculate_snr_db(
    signal_power: PowerDbm,
    bandwidth: Hertz,
    noise_figure_db: f32,
) -> f32 {
    let noise = noise_floor_dbm(bandwidth, noise_figure_db);
    signal_power.as_dbm() - noise.as_dbm()
}

/// Get thermal noise power spectral density per Hz.
///
/// Returns the noise floor in linear power per Hz.
pub fn thermal_noise_psd(noise_figure_db: f32) -> PositivePower {
    let noise_floor_dbm_per_hz = THERMAL_NOISE_DBM_PER_HZ + noise_figure_db;
    let noise_psd_linear = 10.0_f32.powf(noise_floor_dbm_per_hz / 10.0) / 1000.0;
    PositivePower::new(noise_psd_linear)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> ValidatedFrequencyGrid {
        ValidatedFrequencyGrid::from_params(100e6, 1e9, 1000).unwrap()
    }

    #[test]
    fn test_thermal_noise_level() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        let noise_per_bin = add_thermal_noise(&mut psd, &grid, 0.0);

        // Verify noise was added
        assert!(psd[100] > 0.0);

        // Verify returned noise power
        assert!(noise_per_bin.watts() > 0.0);
        assert!((psd[100] - noise_per_bin.watts()).abs() < 1e-30);
    }

    #[test]
    fn test_thermal_noise_with_noise_figure() {
        let grid = test_grid();
        let mut psd_0db = vec![0.0; grid.num_bins()];
        let mut psd_6db = vec![0.0; grid.num_bins()];

        let noise_0 = add_thermal_noise(&mut psd_0db, &grid, 0.0);
        let noise_6 = add_thermal_noise(&mut psd_6db, &grid, 6.0);

        // 6 dB noise figure should increase noise by factor of ~4
        let ratio = noise_6.watts() / noise_0.watts();
        let expected_ratio = 10.0_f32.powf(6.0 / 10.0);

        assert!(
            (ratio - expected_ratio).abs() < 0.1,
            "Noise figure effect incorrect: ratio = {}, expected = {}",
            ratio,
            expected_ratio
        );
    }

    #[test]
    fn test_man_made_noise_frequency_dependence() {
        // Man-made noise should decrease with frequency
        let f_100mhz = man_made_noise_figure_db(100.0, NoiseEnvironment::City);
        let f_10mhz = man_made_noise_figure_db(10.0, NoiseEnvironment::City);

        // Lower frequency should have higher noise figure
        assert!(
            f_10mhz > f_100mhz,
            "Man-made noise should be higher at lower frequencies"
        );
    }

    #[test]
    fn test_man_made_noise_environment() {
        let freq = 50.0; // 50 MHz

        let city = man_made_noise_figure_db(freq, NoiseEnvironment::City);
        let residential = man_made_noise_figure_db(freq, NoiseEnvironment::Residential);
        let rural = man_made_noise_figure_db(freq, NoiseEnvironment::Rural);

        // City > Residential > Rural
        assert!(city > residential, "City noise should be highest");
        assert!(residential > rural, "Residential noise should exceed rural");
    }

    #[test]
    fn test_noise_floor_dbm_calculation() {
        let bandwidth = Hertz::new(1e6);
        let noise = noise_floor_dbm(bandwidth, 6.0);

        // -174 + 60 + 6 = -108 dBm
        let expected = -174.0 + 10.0 * 1e6_f32.log10() + 6.0;

        assert!(
            (noise.as_dbm() - expected).abs() < 0.1,
            "Noise floor calculation: got {}, expected {}",
            noise.as_dbm(),
            expected
        );
    }

    #[test]
    fn test_snr_calculation() {
        let signal = PowerDbm::new(-50.0);
        let bandwidth = Hertz::new(1e6);
        let nf = 6.0;

        let snr = calculate_snr_db(signal, bandwidth, nf);
        let expected_snr = -50.0 - (-174.0 + 60.0 + 6.0);

        assert!(
            (snr - expected_snr).abs() < 0.1,
            "SNR calculation incorrect: got {}, expected {}",
            snr,
            expected_snr
        );
    }

    #[test]
    fn test_combined_noise_floor() {
        let grid = ValidatedFrequencyGrid::from_params(10e6, 100e6, 100).unwrap();
        let mut psd = vec![0.0; grid.num_bins()];

        let thermal = add_noise_floor(&mut psd, &grid, 6.0, NoiseEnvironment::City);

        // All bins should have some noise
        assert!(psd.iter().all(|&x| x > 0.0));

        // Returned thermal power should be positive
        assert!(thermal.watts() > 0.0);

        // Lower frequency bins should have more noise (man-made contribution)
        assert!(psd[0] > psd[grid.num_bins() - 1]);
    }

    #[test]
    fn test_noise_values_reasonable() {
        let grid = test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        add_thermal_noise(&mut psd, &grid, 6.0);

        // Convert to dBm and check reasonable range
        for &val in psd.iter() {
            let dbm = if val > 1e-30 {
                10.0 * (val * 1000.0).log10()
            } else {
                -300.0
            };

            // Noise floor per bin should be very low (sub-picowatt range)
            assert!(
                dbm < -100.0,
                "Noise per bin seems too high: {} dBm",
                dbm
            );
        }
    }

    #[test]
    fn test_thermal_noise_psd() {
        let psd_0db = thermal_noise_psd(0.0);
        let psd_10db = thermal_noise_psd(10.0);

        // 10 dB increase should mean 10x power
        let ratio = psd_10db.watts() / psd_0db.watts();
        assert!(
            (ratio - 10.0).abs() < 0.1,
            "PSD ratio incorrect: got {}, expected 10",
            ratio
        );
    }
}
