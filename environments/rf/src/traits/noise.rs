//! Noise Model Trait
//!
//! Defines the interface for computing noise floors in RF environments.
//! Includes thermal noise and ITU-R P.372 man-made noise models.

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

use crate::spectrum::NoiseEnvironment;

/// Noise model trait for computing RF noise floors.
///
/// Implementations compute the noise power spectral density in dBm/Hz
/// at a given frequency and environment type.
pub trait NoiseModel: Send + Sync {
    /// Compute noise floor in dBm/Hz
    ///
    /// # Arguments
    /// * `frequency_hz` - Carrier frequency in Hz
    /// * `environment` - Noise environment type
    fn noise_floor_dbm_hz(&self, frequency_hz: f32, environment: NoiseEnvironment) -> f32;

    /// Compute noise floor with bandwidth in dBm
    ///
    /// # Arguments
    /// * `frequency_hz` - Carrier frequency in Hz
    /// * `bandwidth_hz` - Receiver bandwidth in Hz
    /// * `environment` - Noise environment type
    fn noise_power_dbm(&self, frequency_hz: f32, bandwidth_hz: f32, environment: NoiseEnvironment) -> f32 {
        self.noise_floor_dbm_hz(frequency_hz, environment) + 10.0 * bandwidth_hz.log10()
    }

    /// Noise figure of the receiver in dB
    fn noise_figure_db(&self) -> f32;

    /// Model name for debugging/logging
    fn name(&self) -> &'static str;

    /// Compute total system noise temperature in Kelvin
    fn system_noise_temperature_k(&self, frequency_hz: f32, environment: NoiseEnvironment) -> f32 {
        // Convert noise floor from dBm/Hz to noise temperature
        // N = kTB => T = 10^(N_dbm/10) * 1e-3 / (k * B)
        // For B=1 Hz: T = 10^(N_dbm_hz/10) * 1e-3 / k
        let n_dbm_hz = self.noise_floor_dbm_hz(frequency_hz, environment);
        let k = 1.380649e-23; // Boltzmann constant
        let power_w = 10.0_f32.powf(n_dbm_hz / 10.0) * 1e-3;
        power_w / k
    }
}

// ============================================================================
// Thermal Noise Model
// ============================================================================

/// Thermal (Johnson-Nyquist) noise model
///
/// The fundamental noise floor due to thermal agitation of electrons.
/// N₀ = kT = -174 dBm/Hz at 290K (room temperature).
///
/// This model ignores man-made noise and represents the theoretical minimum.
#[derive(Debug, Clone, Copy)]
pub struct ThermalNoiseModel {
    /// System noise figure in dB
    noise_figure_db: f32,
    /// Reference temperature in Kelvin
    temperature_k: f32,
}

impl ThermalNoiseModel {
    /// Create with default parameters (NF=6dB, T=290K)
    pub fn new() -> Self {
        Self {
            noise_figure_db: 6.0,
            temperature_k: 290.0,
        }
    }

    /// Create with specific noise figure
    pub fn with_noise_figure(mut self, nf_db: f32) -> Self {
        self.noise_figure_db = nf_db;
        self
    }

    /// Create with specific temperature
    pub fn with_temperature(mut self, temp_k: f32) -> Self {
        self.temperature_k = temp_k;
        self
    }

    /// Low noise amplifier preset (NF=1dB)
    pub fn low_noise() -> Self {
        Self::new().with_noise_figure(1.0)
    }

    /// Consumer receiver preset (NF=10dB)
    pub fn consumer_grade() -> Self {
        Self::new().with_noise_figure(10.0)
    }
}

impl Default for ThermalNoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseModel for ThermalNoiseModel {
    fn noise_floor_dbm_hz(&self, _frequency_hz: f32, _environment: NoiseEnvironment) -> f32 {
        // kTB = -174 dBm/Hz at 290K, adjust for actual temperature
        let thermal_noise_290k = -174.0;
        let temp_correction = 10.0 * (self.temperature_k / 290.0).log10();
        thermal_noise_290k + temp_correction + self.noise_figure_db
    }

    fn noise_figure_db(&self) -> f32 {
        self.noise_figure_db
    }

    fn name(&self) -> &'static str {
        "Thermal Noise"
    }
}

// ============================================================================
// ITU-R P.372 Man-Made Noise Model
// ============================================================================

/// ITU-R P.372 man-made noise model
///
/// Models man-made noise from electrical equipment, motors, ignition systems, etc.
/// Man-made noise typically dominates at lower frequencies (< 1 GHz) in urban areas.
///
/// Reference: ITU-R P.372-15 "Radio noise"
#[derive(Debug, Clone, Copy)]
pub struct ITUNoiseModel {
    /// System noise figure in dB
    noise_figure_db: f32,
    /// Reference temperature for thermal baseline
    temperature_k: f32,
}

impl ITUNoiseModel {
    pub fn new() -> Self {
        Self {
            noise_figure_db: 6.0,
            temperature_k: 290.0,
        }
    }

    pub fn with_noise_figure(mut self, nf_db: f32) -> Self {
        self.noise_figure_db = nf_db;
        self
    }

    /// Compute man-made noise factor in dB above thermal
    ///
    /// From ITU-R P.372, the median man-made noise figure is:
    /// Fam = c - d*log10(f_MHz)
    ///
    /// where c and d depend on environment type
    fn man_made_factor_db(&self, frequency_hz: f32, environment: NoiseEnvironment) -> f32 {
        let f_mhz = frequency_hz / 1e6;

        // ITU-R P.372 coefficients
        let (c, d) = match environment {
            NoiseEnvironment::City => (76.8, 27.7),
            NoiseEnvironment::Residential => (72.5, 27.7),
            NoiseEnvironment::Rural => (67.2, 27.7),
            NoiseEnvironment::QuietRural => (53.6, 28.6), // "Quiet rural"
        };

        // Man-made noise factor above thermal
        let fam_db = c - d * f_mhz.max(0.01).log10();

        // Man-made noise decreases above about 200 MHz
        // and becomes negligible above a few GHz
        if frequency_hz > 3e9 {
            0.0 // Negligible above 3 GHz
        } else if frequency_hz > 1e9 {
            // Transition region
            let ratio = (3e9 - frequency_hz) / 2e9;
            fam_db * ratio
        } else {
            fam_db.max(0.0)
        }
    }
}

impl Default for ITUNoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseModel for ITUNoiseModel {
    fn noise_floor_dbm_hz(&self, frequency_hz: f32, environment: NoiseEnvironment) -> f32 {
        // Start with thermal noise baseline
        let thermal_noise_dbm_hz = -174.0 + 10.0 * (self.temperature_k / 290.0).log10();

        // Add man-made noise contribution
        let man_made_db = self.man_made_factor_db(frequency_hz, environment);

        // Add receiver noise figure
        let system_noise = thermal_noise_dbm_hz + self.noise_figure_db;

        // Combine thermal and man-made (power sum in linear domain)
        if man_made_db > 0.0 {
            let thermal_linear = 10.0_f32.powf(system_noise / 10.0);
            let man_made_linear = 10.0_f32.powf((thermal_noise_dbm_hz + man_made_db) / 10.0);
            10.0 * (thermal_linear + man_made_linear).log10()
        } else {
            system_noise
        }
    }

    fn noise_figure_db(&self) -> f32 {
        self.noise_figure_db
    }

    fn name(&self) -> &'static str {
        "ITU-R P.372 Man-Made"
    }
}

// ============================================================================
// SIMD Support
// ============================================================================

#[cfg(feature = "simd")]
use crate::simd_rf::math::{simd_log, simd_pow};

/// Compute thermal noise floor for 8 frequencies (SIMD)
#[cfg(feature = "simd")]
pub fn thermal_noise_simd(
    _frequency_hz: f32x8,
    bandwidth_hz: f32x8,
    noise_figure_db: f32,
    temperature_k: f32,
) -> f32x8 {
    let thermal_base = f32x8::splat(-174.0);
    let temp_correction = f32x8::splat(10.0 * (temperature_k / 290.0).log10());
    let nf = f32x8::splat(noise_figure_db);
    let bw_correction = f32x8::splat(10.0) * simd_log(bandwidth_hz) / f32x8::splat(std::f32::consts::LN_10);

    thermal_base + temp_correction + nf + bw_correction
}

/// Compute ITU man-made noise for 8 frequencies (SIMD)
///
/// Simplified version using city environment coefficients
#[cfg(feature = "simd")]
pub fn itu_noise_simd(
    frequency_hz: f32x8,
    bandwidth_hz: f32x8,
    noise_figure_db: f32,
    environment: NoiseEnvironment,
) -> f32x8 {
    use std::simd::cmp::SimdPartialOrd;

    let f_mhz = frequency_hz / f32x8::splat(1e6);
    let log_f = simd_log(f_mhz.simd_max(f32x8::splat(0.01))) / f32x8::splat(std::f32::consts::LN_10);

    // ITU coefficients
    let (c, d) = match environment {
        NoiseEnvironment::City => (76.8, 27.7),
        NoiseEnvironment::Residential => (72.5, 27.7),
        NoiseEnvironment::Rural => (67.2, 27.7),
        NoiseEnvironment::QuietRural => (53.6, 28.6),
    };

    let man_made_factor = (f32x8::splat(c) - f32x8::splat(d) * log_f).simd_max(f32x8::splat(0.0));

    // Above 3 GHz, man-made noise is negligible
    let high_freq_mask = frequency_hz.simd_gt(f32x8::splat(3e9));
    let man_made_factor = high_freq_mask.select(f32x8::splat(0.0), man_made_factor);

    // Thermal baseline
    let thermal = f32x8::splat(-174.0 + noise_figure_db);

    // Combine thermal and man-made
    let thermal_linear = simd_pow(f32x8::splat(10.0), thermal / f32x8::splat(10.0));
    let man_made_linear = simd_pow(f32x8::splat(10.0), (f32x8::splat(-174.0) + man_made_factor) / f32x8::splat(10.0));
    let combined = f32x8::splat(10.0) * simd_log(thermal_linear + man_made_linear) / f32x8::splat(std::f32::consts::LN_10);

    // Add bandwidth
    let bw_correction = f32x8::splat(10.0) * simd_log(bandwidth_hz) / f32x8::splat(std::f32::consts::LN_10);

    combined + bw_correction
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_noise_baseline() {
        let model = ThermalNoiseModel::new().with_noise_figure(0.0);
        let noise = model.noise_floor_dbm_hz(1e9, NoiseEnvironment::QuietRural);

        // Should be approximately -174 dBm/Hz at 290K with NF=0
        assert!(
            (noise - (-174.0)).abs() < 0.1,
            "Thermal noise should be -174 dBm/Hz, got {}",
            noise
        );
    }

    #[test]
    fn test_thermal_noise_with_nf() {
        let model = ThermalNoiseModel::new().with_noise_figure(6.0);
        let noise = model.noise_floor_dbm_hz(1e9, NoiseEnvironment::QuietRural);

        // Should be -174 + 6 = -168 dBm/Hz
        assert!(
            (noise - (-168.0)).abs() < 0.1,
            "Thermal + NF should be -168 dBm/Hz, got {}",
            noise
        );
    }

    #[test]
    fn test_thermal_noise_power_with_bandwidth() {
        let model = ThermalNoiseModel::new().with_noise_figure(0.0);
        let noise = model.noise_power_dbm(1e9, 1e6, NoiseEnvironment::QuietRural);

        // -174 + 10*log10(1e6) = -174 + 60 = -114 dBm
        assert!(
            (noise - (-114.0)).abs() < 0.1,
            "1 MHz BW noise should be -114 dBm, got {}",
            noise
        );
    }

    #[test]
    fn test_itu_noise_city_higher() {
        let model = ITUNoiseModel::new();
        let city = model.noise_floor_dbm_hz(100e6, NoiseEnvironment::City);
        let rural = model.noise_floor_dbm_hz(100e6, NoiseEnvironment::Rural);

        assert!(
            city > rural,
            "City noise should exceed rural: {} vs {}",
            city,
            rural
        );
    }

    #[test]
    fn test_itu_noise_decreases_with_frequency() {
        let model = ITUNoiseModel::new();
        let low_freq = model.noise_floor_dbm_hz(100e6, NoiseEnvironment::City);
        let high_freq = model.noise_floor_dbm_hz(1e9, NoiseEnvironment::City);

        assert!(
            low_freq > high_freq,
            "Man-made noise should decrease with frequency: {} vs {}",
            low_freq,
            high_freq
        );
    }

    #[test]
    fn test_itu_noise_approaches_thermal_at_high_freq() {
        let itu = ITUNoiseModel::new().with_noise_figure(6.0);
        let thermal = ThermalNoiseModel::new().with_noise_figure(6.0);

        let itu_5ghz = itu.noise_floor_dbm_hz(5e9, NoiseEnvironment::City);
        let thermal_5ghz = thermal.noise_floor_dbm_hz(5e9, NoiseEnvironment::City);

        // Above 3 GHz, should be nearly thermal
        assert!(
            (itu_5ghz - thermal_5ghz).abs() < 1.0,
            "At 5 GHz, ITU should approach thermal: {} vs {}",
            itu_5ghz,
            thermal_5ghz
        );
    }

    #[test]
    fn test_noise_figure_accessor() {
        let model = ThermalNoiseModel::new().with_noise_figure(8.0);
        assert!((model.noise_figure_db() - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_model_names() {
        assert_eq!(ThermalNoiseModel::new().name(), "Thermal Noise");
        assert_eq!(ITUNoiseModel::new().name(), "ITU-R P.372 Man-Made");
    }

    #[test]
    fn test_system_noise_temperature() {
        let model = ThermalNoiseModel::new().with_noise_figure(0.0).with_temperature(290.0);
        let temp = model.system_noise_temperature_k(1e9, NoiseEnvironment::QuietRural);

        // Should be approximately 290K
        assert!(
            (temp - 290.0).abs() < 10.0,
            "System noise temp should be ~290K, got {}",
            temp
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_thermal_noise_simd() {
        let frequencies = f32x8::splat(1e9);
        let bandwidths = f32x8::splat(1e6);

        let noise = thermal_noise_simd(frequencies, bandwidths, 0.0, 290.0);
        let noise_arr: [f32; 8] = noise.into();

        for n in noise_arr {
            // -174 + 60 (BW) = -114 dBm
            assert!(
                (n - (-114.0)).abs() < 0.5,
                "SIMD thermal noise should be ~-114 dBm, got {}",
                n
            );
        }
    }
}
