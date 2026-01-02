//! SHF Band Implementation
//!
//! Super High Frequency band (3 - 30 GHz), covering:
//! - WiFi 5 GHz (5.15-5.85 GHz)
//! - 5G NR FR1 upper bands
//! - Radar (C, X, Ku bands)
//! - Satellite communications
//! - Point-to-point microwave links
//! - mmWave early bands

use crate::entities::EntityType;
use crate::physics::channel::MultipathModel;
use crate::spectrum::NoiseEnvironment;
use crate::traits::{
    FrequencyBand, PropagationModel, NoiseModel,
    FSPLModel, LogDistanceModel, ThermalNoiseModel,
};

/// SHF Band (3 - 30 GHz)
///
/// This band is characterized by:
/// - Higher path loss than UHF
/// - Atmospheric absorption becomes significant (especially >10 GHz)
/// - Man-made noise is negligible (thermal noise dominates)
/// - Good directivity possible with small antennas
/// - Rain attenuation can be significant
#[derive(Debug, Clone)]
pub struct SHFBand {
    /// Noise environment configuration (QuietRural used since man-made noise negligible)
    pub environment: NoiseEnvironment,
    /// Include atmospheric absorption in propagation
    pub include_atmospheric: bool,
    /// Receiver noise figure in dB
    pub noise_figure_db: f32,
    /// Path loss exponent for indoor/NLOS scenarios
    pub path_loss_exponent: f32,
}

impl SHFBand {
    /// Create SHF band with default parameters
    pub fn new() -> Self {
        Self {
            environment: NoiseEnvironment::QuietRural, // Man-made noise negligible at SHF
            include_atmospheric: false,
            noise_figure_db: 6.0,
            path_loss_exponent: 2.0, // Free space for LOS
        }
    }

    /// Create for outdoor LOS scenarios (point-to-point links)
    pub fn outdoor_los() -> Self {
        Self {
            environment: NoiseEnvironment::QuietRural,
            include_atmospheric: true,
            noise_figure_db: 4.0,
            path_loss_exponent: 2.0,
        }
    }

    /// Create for indoor WiFi 5 GHz scenarios
    pub fn indoor_wifi() -> Self {
        Self {
            environment: NoiseEnvironment::QuietRural,
            include_atmospheric: false,
            noise_figure_db: 8.0,
            path_loss_exponent: 3.5, // Indoor NLOS
        }
    }

    /// Create for satellite communication
    pub fn satellite() -> Self {
        Self {
            environment: NoiseEnvironment::QuietRural,
            include_atmospheric: true,
            noise_figure_db: 2.0, // Low noise LNA
            path_loss_exponent: 2.0,
        }
    }

    /// Create for radar applications
    pub fn radar() -> Self {
        Self {
            environment: NoiseEnvironment::QuietRural,
            include_atmospheric: true,
            noise_figure_db: 3.0,
            path_loss_exponent: 2.0,
        }
    }

    /// Enable atmospheric absorption modeling
    pub fn with_atmospheric(mut self) -> Self {
        self.include_atmospheric = true;
        self
    }

    /// Set noise figure
    pub fn with_noise_figure(mut self, nf_db: f32) -> Self {
        self.noise_figure_db = nf_db;
        self
    }

    /// Set path loss exponent for NLOS scenarios
    pub fn with_path_loss_exponent(mut self, exp: f32) -> Self {
        self.path_loss_exponent = exp;
        self
    }
}

impl Default for SHFBand {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyBand for SHFBand {
    fn freq_range(&self) -> (f32, f32) {
        (3e9, 30e9) // 3 GHz to 30 GHz
    }

    fn default_resolution(&self) -> f32 {
        1e6 // 1 MHz resolution (wider channels at SHF)
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        if (self.path_loss_exponent - 2.0).abs() < 0.1 {
            // Near free-space exponent, use FSPL
            Box::new(FSPLModel::new())
        } else {
            // Custom exponent for NLOS
            Box::new(LogDistanceModel::new(self.path_loss_exponent))
        }
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        // Man-made noise negligible at SHF, use thermal only
        Box::new(ThermalNoiseModel::new().with_noise_figure(self.noise_figure_db))
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![
            EntityType::WiFiAP,       // 5 GHz WiFi
            EntityType::SBandRadar,   // S-band radar
            EntityType::WeatherRadar, // Weather radar
            EntityType::DroneDigital, // 5.8 GHz digital video
        ]
    }

    fn default_channel_model(&self) -> MultipathModel {
        // At SHF, multipath is typically less significant due to higher attenuation
        // of reflected paths, but can still occur in indoor/urban environments
        match self.path_loss_exponent {
            exp if exp > 3.0 => MultipathModel::EVA, // Indoor NLOS
            exp if exp > 2.5 => MultipathModel::EPA, // Mild multipath
            _ => MultipathModel::None,                // LOS
        }
    }

    fn name(&self) -> &'static str {
        "SHF (3 - 30 GHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        self.environment
    }
}

// ============================================================================
// SHF with Atmospheric Propagation Model
// ============================================================================

/// FSPL with atmospheric absorption for SHF
///
/// Combines free-space path loss with gaseous absorption (O2, H2O).
/// Important for frequencies above ~10 GHz.
#[derive(Debug, Clone, Copy)]
pub struct FSPLWithAtmospheric {
    /// Include oxygen absorption
    pub include_o2: bool,
    /// Include water vapor absorption
    pub include_h2o: bool,
    /// Water vapor density in g/m³ (typical: 7.5 for temperate climate)
    pub water_vapor_density: f32,
}

impl FSPLWithAtmospheric {
    pub fn new() -> Self {
        Self {
            include_o2: true,
            include_h2o: true,
            water_vapor_density: 7.5,
        }
    }

    pub fn dry_air() -> Self {
        Self {
            include_o2: true,
            include_h2o: false,
            water_vapor_density: 0.0,
        }
    }

    pub fn with_humidity(mut self, vapor_density: f32) -> Self {
        self.water_vapor_density = vapor_density;
        self.include_h2o = vapor_density > 0.0;
        self
    }

    /// Approximate specific attenuation in dB/km
    /// Based on ITU-R P.676 simplified model
    fn atmospheric_attenuation_db_per_km(&self, frequency_hz: f32) -> f32 {
        let f_ghz = frequency_hz / 1e9;
        let mut attenuation = 0.0;

        if self.include_o2 {
            // O2 absorption peak around 60 GHz, simplified model for SHF
            if f_ghz > 20.0 {
                attenuation += 0.01 * (f_ghz - 20.0).max(0.0);
            }
        }

        if self.include_h2o && self.water_vapor_density > 0.0 {
            // H2O absorption starts becoming significant above 10 GHz
            if f_ghz > 10.0 {
                let rho = self.water_vapor_density;
                attenuation += 0.001 * rho * (f_ghz - 10.0).max(0.0);
            }
        }

        attenuation
    }
}

impl Default for FSPLWithAtmospheric {
    fn default() -> Self {
        Self::new()
    }
}

impl PropagationModel for FSPLWithAtmospheric {
    fn path_loss_db(&self, distance_m: f32, frequency_hz: f32, _tx_height_m: f32, _rx_height_m: f32) -> f32 {
        // Base FSPL
        let d = distance_m.max(0.1);
        let fspl = 20.0 * (4.0 * std::f32::consts::PI * d * frequency_hz / 299_792_458.0).log10();

        // Add atmospheric absorption
        let dist_km = d / 1000.0;
        let atmos = self.atmospheric_attenuation_db_per_km(frequency_hz) * dist_km;

        fspl + atmos
    }

    #[cfg(feature = "simd")]
    fn path_loss_db_simd(
        &self,
        distance_m: std::simd::f32x8,
        frequency_hz: std::simd::f32x8,
        _tx_height_m: std::simd::f32x8,
        _rx_height_m: std::simd::f32x8,
    ) -> std::simd::f32x8 {
        use std::simd::{f32x8, num::SimdFloat};
        use crate::simd_rf::math::simd_log;

        const LOG10_E: f32 = std::f32::consts::LOG10_E;

        let d = distance_m.simd_max(f32x8::splat(0.1));

        // FSPL
        let constant_db = f32x8::splat(-147.55);
        let log10_d = simd_log(d) * f32x8::splat(LOG10_E);
        let log10_f = simd_log(frequency_hz) * f32x8::splat(LOG10_E);
        let fspl = f32x8::splat(20.0) * (log10_d + log10_f) + constant_db;

        // Simplified atmospheric (constant for SIMD efficiency)
        let f_ghz = frequency_hz / f32x8::splat(1e9);
        let dist_km = d / f32x8::splat(1000.0);

        // Approximate atmospheric: 0.01 dB/km/GHz above 10 GHz
        let excess_ghz = (f_ghz - f32x8::splat(10.0)).simd_max(f32x8::splat(0.0));
        let atmos = f32x8::splat(0.01) * excess_ghz * dist_km;

        fspl + atmos
    }

    fn name(&self) -> &'static str {
        "FSPL + Atmospheric"
    }

    fn valid_frequency_range(&self) -> (f32, f32) {
        (3e9, 100e9) // 3 GHz to 100 GHz
    }

    fn valid_distance_range(&self) -> (f32, f32) {
        (0.1, 100_000.0) // 0.1m to 100 km
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shf_freq_range() {
        let band = SHFBand::new();
        let (min, max) = band.freq_range();
        assert_eq!(min, 3e9);
        assert_eq!(max, 30e9);
    }

    #[test]
    fn test_shf_contains_common_frequencies() {
        let band = SHFBand::new();

        // Should contain
        assert!(band.contains(5e9));     // WiFi 5 GHz
        assert!(band.contains(10e9));    // X-band radar
        assert!(band.contains(28e9));    // 5G mmWave

        // Should not contain
        assert!(!band.contains(2.4e9));  // WiFi 2.4 GHz
        assert!(!band.contains(60e9));   // 60 GHz
    }

    #[test]
    fn test_shf_propagation_model() {
        let band_los = SHFBand::outdoor_los();
        let model = band_los.propagation_model();
        assert_eq!(model.name(), "Free Space Path Loss");

        let band_nlos = SHFBand::indoor_wifi();
        let model_nlos = band_nlos.propagation_model();
        assert_eq!(model_nlos.name(), "Log-Distance Path Loss");
    }

    #[test]
    fn test_shf_noise_model() {
        let band = SHFBand::new();
        let model = band.noise_model();
        assert_eq!(model.name(), "Thermal Noise");
    }

    #[test]
    fn test_shf_typical_entities() {
        let band = SHFBand::new();
        let entities = band.typical_entities();
        assert!(!entities.is_empty());
        assert!(entities.contains(&EntityType::WiFiAP));
        assert!(entities.contains(&EntityType::SBandRadar));
    }

    #[test]
    fn test_shf_channel_models() {
        let los = SHFBand::outdoor_los();
        assert_eq!(los.default_channel_model(), MultipathModel::None);

        let indoor = SHFBand::indoor_wifi();
        assert_eq!(indoor.default_channel_model(), MultipathModel::EVA);
    }

    #[test]
    fn test_shf_resolution() {
        let band = SHFBand::new();
        assert_eq!(band.default_resolution(), 1e6);
    }

    #[test]
    fn test_atmospheric_model() {
        let model = FSPLWithAtmospheric::new();

        // At 10 GHz, 1 km, should be close to pure FSPL
        let loss_10ghz = model.path_loss_db(1000.0, 10e9, 10.0, 1.5);

        // At 28 GHz, same distance, should have additional atmospheric loss
        let loss_28ghz = model.path_loss_db(1000.0, 28e9, 10.0, 1.5);

        // Higher frequency should have more loss
        assert!(
            loss_28ghz > loss_10ghz,
            "28 GHz should have more loss than 10 GHz: {} vs {}",
            loss_28ghz,
            loss_10ghz
        );
    }

    #[test]
    fn test_atmospheric_vs_pure_fspl() {
        let atmos = FSPLWithAtmospheric::new();
        let fspl = FSPLModel::new();

        // At 10 km, 28 GHz, atmospheric should add noticeable loss
        let loss_atmos = atmos.path_loss_db(10000.0, 28e9, 10.0, 1.5);
        let loss_fspl = fspl.path_loss_db(10000.0, 28e9, 10.0, 1.5);

        assert!(
            loss_atmos > loss_fspl,
            "Atmospheric model should exceed pure FSPL: {} vs {}",
            loss_atmos,
            loss_fspl
        );
    }
}
