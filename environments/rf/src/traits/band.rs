//! Frequency Band Trait
//!
//! Defines the interface for frequency band configurations that combine
//! propagation models, noise models, and default parameters.

use crate::entities::EntityType;
use crate::physics::channel::MultipathModel;
use crate::spectrum::NoiseEnvironment;

use super::noise::NoiseModel;
use super::propagation::PropagationModel;

/// Frequency band trait combining models and parameters.
///
/// Each band implementation provides appropriate defaults for
/// its frequency range, including propagation model, noise model,
/// typical entities, and channel characteristics.
pub trait FrequencyBand: Send + Sync {
    /// Frequency range in Hz (min, max)
    fn freq_range(&self) -> (f32, f32);

    /// Default frequency resolution for PSD in Hz
    fn default_resolution(&self) -> f32;

    /// Create appropriate propagation model for this band
    fn propagation_model(&self) -> Box<dyn PropagationModel>;

    /// Create appropriate noise model for this band
    fn noise_model(&self) -> Box<dyn NoiseModel>;

    /// Typical entity types found in this band
    fn typical_entities(&self) -> Vec<EntityType>;

    /// Default multipath channel model
    fn default_channel_model(&self) -> MultipathModel;

    /// Band name for debugging/logging
    fn name(&self) -> &'static str;

    /// Noise environment this band is configured for
    fn environment(&self) -> NoiseEnvironment;

    /// Check if a frequency is within this band
    fn contains(&self, frequency_hz: f32) -> bool {
        let (min, max) = self.freq_range();
        frequency_hz >= min && frequency_hz <= max
    }

    /// Get center frequency of the band
    fn center_frequency(&self) -> f32 {
        let (min, max) = self.freq_range();
        (min + max) / 2.0
    }

    /// Get total bandwidth of the band
    fn bandwidth(&self) -> f32 {
        let (min, max) = self.freq_range();
        max - min
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing the trait
    struct MockBand {
        min_freq: f32,
        max_freq: f32,
    }

    impl FrequencyBand for MockBand {
        fn freq_range(&self) -> (f32, f32) {
            (self.min_freq, self.max_freq)
        }

        fn default_resolution(&self) -> f32 {
            100e3
        }

        fn propagation_model(&self) -> Box<dyn PropagationModel> {
            Box::new(super::super::propagation::FSPLModel::new())
        }

        fn noise_model(&self) -> Box<dyn NoiseModel> {
            Box::new(super::super::noise::ThermalNoiseModel::new())
        }

        fn typical_entities(&self) -> Vec<EntityType> {
            vec![]
        }

        fn default_channel_model(&self) -> MultipathModel {
            MultipathModel::None
        }

        fn name(&self) -> &'static str {
            "Mock Band"
        }

        fn environment(&self) -> NoiseEnvironment {
            NoiseEnvironment::QuietRural
        }
    }

    #[test]
    fn test_contains() {
        let band = MockBand {
            min_freq: 1e9,
            max_freq: 2e9,
        };

        assert!(band.contains(1.5e9));
        assert!(band.contains(1e9));
        assert!(band.contains(2e9));
        assert!(!band.contains(0.5e9));
        assert!(!band.contains(2.5e9));
    }

    #[test]
    fn test_center_frequency() {
        let band = MockBand {
            min_freq: 1e9,
            max_freq: 2e9,
        };

        assert!((band.center_frequency() - 1.5e9).abs() < 1.0);
    }

    #[test]
    fn test_bandwidth() {
        let band = MockBand {
            min_freq: 1e9,
            max_freq: 2e9,
        };

        assert!((band.bandwidth() - 1e9).abs() < 1.0);
    }
}
