//! UHF Band Implementation
//!
//! Ultra High Frequency band (300 MHz - 3 GHz), covering:
//! - TV broadcast (470-890 MHz)
//! - Cellular (700, 850, 900, 1800, 1900, 2100 MHz)
//! - WiFi 2.4 GHz
//! - Bluetooth
//! - GPS L1 (1.57542 GHz)
//! - Many IoT and industrial applications

use crate::entities::EntityType;
use crate::physics::channel::MultipathModel;
use crate::spectrum::NoiseEnvironment;
use crate::traits::{
    FrequencyBand, PropagationModel, NoiseModel,
    HataUrbanModel, LogDistanceModel, ITUNoiseModel,
};

/// UHF Band (300 MHz - 3 GHz)
///
/// This band is characterized by:
/// - Good balance of range and bandwidth
/// - Significant man-made noise in urban areas
/// - Commonly used Hata/COST-231 propagation models
/// - Multipath fading is significant
#[derive(Debug, Clone)]
pub struct UHFBand {
    /// Noise environment configuration
    pub environment: NoiseEnvironment,
    /// Use Hata model (true) or log-distance (false)
    pub use_hata: bool,
    /// Custom path loss exponent for log-distance model
    pub path_loss_exponent: f32,
}

impl UHFBand {
    /// Create UHF band with default urban environment
    pub fn new() -> Self {
        Self {
            environment: NoiseEnvironment::City,
            use_hata: true,
            path_loss_exponent: 3.5,
        }
    }

    /// Create for urban environment
    pub fn urban() -> Self {
        Self {
            environment: NoiseEnvironment::City,
            use_hata: true,
            path_loss_exponent: 3.5,
        }
    }

    /// Create for suburban/residential environment
    pub fn suburban() -> Self {
        Self {
            environment: NoiseEnvironment::Residential,
            use_hata: true,
            path_loss_exponent: 3.2,
        }
    }

    /// Create for rural environment
    pub fn rural() -> Self {
        Self {
            environment: NoiseEnvironment::Rural,
            use_hata: false, // Log-distance more appropriate
            path_loss_exponent: 2.5,
        }
    }

    /// Create for indoor environment
    pub fn indoor() -> Self {
        Self {
            environment: NoiseEnvironment::QuietRural, // Indoor has minimal man-made noise
            use_hata: false,
            path_loss_exponent: 3.0,
        }
    }

    /// Set noise environment
    pub fn with_environment(mut self, env: NoiseEnvironment) -> Self {
        self.environment = env;
        self
    }

    /// Use log-distance model instead of Hata
    pub fn with_log_distance(mut self, exponent: f32) -> Self {
        self.use_hata = false;
        self.path_loss_exponent = exponent;
        self
    }
}

impl Default for UHFBand {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyBand for UHFBand {
    fn freq_range(&self) -> (f32, f32) {
        (300e6, 3e9) // 300 MHz to 3 GHz
    }

    fn default_resolution(&self) -> f32 {
        100e3 // 100 kHz resolution
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        if self.use_hata {
            Box::new(HataUrbanModel::new())
        } else {
            Box::new(LogDistanceModel::new(self.path_loss_exponent))
        }
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ITUNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![
            EntityType::TVStation,
            EntityType::LTETower,
            EntityType::WiFiAP,
            EntityType::Bluetooth,
        ]
    }

    fn default_channel_model(&self) -> MultipathModel {
        match self.environment {
            NoiseEnvironment::City => MultipathModel::ETU,        // Extended Typical Urban
            NoiseEnvironment::Residential => MultipathModel::EVA, // Extended Vehicular A
            NoiseEnvironment::Rural => MultipathModel::EPA,       // Extended Pedestrian A
            NoiseEnvironment::QuietRural => MultipathModel::None, // Indoor/quiet, less multipath
        }
    }

    fn name(&self) -> &'static str {
        "UHF (300 MHz - 3 GHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        self.environment
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uhf_freq_range() {
        let band = UHFBand::new();
        let (min, max) = band.freq_range();
        assert_eq!(min, 300e6);
        assert_eq!(max, 3e9);
    }

    #[test]
    fn test_uhf_contains_common_frequencies() {
        let band = UHFBand::new();

        // Should contain
        assert!(band.contains(900e6));   // GSM 900
        assert!(band.contains(1.8e9));   // GSM 1800
        assert!(band.contains(2.4e9));   // WiFi 2.4 GHz
        assert!(band.contains(1.57542e9)); // GPS L1

        // Should not contain
        assert!(!band.contains(5e9));    // WiFi 5 GHz
        assert!(!band.contains(100e6));  // FM radio
    }

    #[test]
    fn test_uhf_propagation_model() {
        let band = UHFBand::urban();
        let model = band.propagation_model();
        assert_eq!(model.name(), "Okumura-Hata Urban");

        let band_indoor = UHFBand::indoor();
        let model_indoor = band_indoor.propagation_model();
        assert_eq!(model_indoor.name(), "Log-Distance Path Loss");
    }

    #[test]
    fn test_uhf_noise_model() {
        let band = UHFBand::new();
        let model = band.noise_model();
        assert_eq!(model.name(), "ITU-R P.372 Man-Made");
    }

    #[test]
    fn test_uhf_typical_entities() {
        let band = UHFBand::new();
        let entities = band.typical_entities();
        assert!(!entities.is_empty());
        assert!(entities.contains(&EntityType::LTETower));
        assert!(entities.contains(&EntityType::WiFiAP));
    }

    #[test]
    fn test_uhf_channel_models() {
        let urban = UHFBand::urban();
        assert_eq!(urban.default_channel_model(), MultipathModel::ETU);

        let rural = UHFBand::rural();
        assert_eq!(rural.default_channel_model(), MultipathModel::EPA);
    }

    #[test]
    fn test_uhf_resolution() {
        let band = UHFBand::new();
        assert_eq!(band.default_resolution(), 100e3);
    }

    #[test]
    fn test_uhf_environment() {
        let band = UHFBand::suburban();
        assert_eq!(band.environment(), NoiseEnvironment::Residential);
    }
}
