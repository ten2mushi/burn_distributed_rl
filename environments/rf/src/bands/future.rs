//! Future Band Stubs
//!
//! Placeholder implementations for frequency bands that are not yet fully implemented.
//! These provide basic structure for future extension.

use crate::entities::EntityType;
use crate::physics::channel::MultipathModel;
use crate::spectrum::NoiseEnvironment;
use crate::traits::{
    FrequencyBand, PropagationModel, NoiseModel,
    FSPLModel, ThermalNoiseModel,
};

// ============================================================================
// VLF Band (3 - 30 kHz)
// ============================================================================

/// Very Low Frequency band (3 - 30 kHz)
///
/// Used for submarine communication, time signals, and navigation.
/// Characterized by ground wave propagation over long distances.
#[derive(Debug, Clone, Default)]
pub struct VLFBand;

impl FrequencyBand for VLFBand {
    fn freq_range(&self) -> (f32, f32) {
        (3e3, 30e3)
    }

    fn default_resolution(&self) -> f32 {
        100.0 // 100 Hz resolution
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        // TODO: Implement ground wave propagation model
        Box::new(FSPLModel::new())
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ThermalNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![] // TODO: Add VLF entity types
    }

    fn default_channel_model(&self) -> MultipathModel {
        MultipathModel::None
    }

    fn name(&self) -> &'static str {
        "VLF (3 - 30 kHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        NoiseEnvironment::QuietRural
    }
}

// ============================================================================
// LF Band (30 - 300 kHz)
// ============================================================================

/// Low Frequency band (30 - 300 kHz)
///
/// Used for navigation (LORAN-C), time signals, AM longwave broadcasting.
#[derive(Debug, Clone, Default)]
pub struct LFBand;

impl FrequencyBand for LFBand {
    fn freq_range(&self) -> (f32, f32) {
        (30e3, 300e3)
    }

    fn default_resolution(&self) -> f32 {
        1e3 // 1 kHz resolution
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        // TODO: Implement ground wave propagation model
        Box::new(FSPLModel::new())
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ThermalNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![] // TODO: Add LF entity types
    }

    fn default_channel_model(&self) -> MultipathModel {
        MultipathModel::None
    }

    fn name(&self) -> &'static str {
        "LF (30 - 300 kHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        NoiseEnvironment::QuietRural
    }
}

// ============================================================================
// MF Band (300 kHz - 3 MHz)
// ============================================================================

/// Medium Frequency band (300 kHz - 3 MHz)
///
/// Used for AM broadcast, maritime communication, aviation NDB.
#[derive(Debug, Clone, Default)]
pub struct MFBand;

impl FrequencyBand for MFBand {
    fn freq_range(&self) -> (f32, f32) {
        (300e3, 3e6)
    }

    fn default_resolution(&self) -> f32 {
        10e3 // 10 kHz resolution
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        // TODO: Implement ground wave + sky wave model
        Box::new(FSPLModel::new())
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ThermalNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![] // TODO: Add MF entity types (AM stations, NDBs)
    }

    fn default_channel_model(&self) -> MultipathModel {
        MultipathModel::None
    }

    fn name(&self) -> &'static str {
        "MF (300 kHz - 3 MHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        NoiseEnvironment::QuietRural
    }
}

// ============================================================================
// HF Band (3 - 30 MHz)
// ============================================================================

/// High Frequency band (3 - 30 MHz)
///
/// Used for shortwave broadcasting, amateur radio, aviation, maritime.
/// Characterized by ionospheric propagation.
#[derive(Debug, Clone, Default)]
pub struct HFBand;

impl FrequencyBand for HFBand {
    fn freq_range(&self) -> (f32, f32) {
        (3e6, 30e6)
    }

    fn default_resolution(&self) -> f32 {
        10e3 // 10 kHz resolution
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        // TODO: Implement ionospheric propagation model
        Box::new(FSPLModel::new())
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ThermalNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![] // TODO: Add HF entity types
    }

    fn default_channel_model(&self) -> MultipathModel {
        MultipathModel::None
    }

    fn name(&self) -> &'static str {
        "HF (3 - 30 MHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        NoiseEnvironment::QuietRural
    }
}

// ============================================================================
// VHF Band (30 - 300 MHz)
// ============================================================================

/// Very High Frequency band (30 - 300 MHz)
///
/// Used for FM broadcast, TV (VHF), aviation, marine VHF, weather satellites.
#[derive(Debug, Clone, Default)]
pub struct VHFBand;

impl FrequencyBand for VHFBand {
    fn freq_range(&self) -> (f32, f32) {
        (30e6, 300e6)
    }

    fn default_resolution(&self) -> f32 {
        25e3 // 25 kHz resolution (common VHF channel spacing)
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        // VHF is primarily line-of-sight, but with some tropospheric effects
        Box::new(FSPLModel::new())
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ThermalNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![
            EntityType::TVStation, // VHF TV
            EntityType::FMRadio,   // FM broadcast
        ]
    }

    fn default_channel_model(&self) -> MultipathModel {
        MultipathModel::EPA // Some multipath possible
    }

    fn name(&self) -> &'static str {
        "VHF (30 - 300 MHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        NoiseEnvironment::QuietRural
    }
}

// ============================================================================
// EHF Band (30 - 300 GHz)
// ============================================================================

/// Extremely High Frequency band (30 - 300 GHz)
///
/// Millimeter wave band. Used for 5G mmWave, point-to-point links,
/// imaging, and radio astronomy.
#[derive(Debug, Clone, Default)]
pub struct EHFBand;

impl FrequencyBand for EHFBand {
    fn freq_range(&self) -> (f32, f32) {
        (30e9, 300e9)
    }

    fn default_resolution(&self) -> f32 {
        10e6 // 10 MHz resolution (very wide channels)
    }

    fn propagation_model(&self) -> Box<dyn PropagationModel> {
        // TODO: Implement mmWave propagation with atmospheric effects
        Box::new(FSPLModel::new())
    }

    fn noise_model(&self) -> Box<dyn NoiseModel> {
        Box::new(ThermalNoiseModel::new())
    }

    fn typical_entities(&self) -> Vec<EntityType> {
        vec![] // TODO: Add mmWave entity types (5G NR FR2, WiGig)
    }

    fn default_channel_model(&self) -> MultipathModel {
        MultipathModel::None // Very directional, limited multipath
    }

    fn name(&self) -> &'static str {
        "EHF (30 - 300 GHz)"
    }

    fn environment(&self) -> NoiseEnvironment {
        NoiseEnvironment::QuietRural
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vlf_band() {
        let band = VLFBand;
        let (min, max) = band.freq_range();
        assert_eq!(min, 3e3);
        assert_eq!(max, 30e3);
        assert_eq!(band.name(), "VLF (3 - 30 kHz)");
    }

    #[test]
    fn test_lf_band() {
        let band = LFBand;
        let (min, max) = band.freq_range();
        assert_eq!(min, 30e3);
        assert_eq!(max, 300e3);
    }

    #[test]
    fn test_mf_band() {
        let band = MFBand;
        let (min, max) = band.freq_range();
        assert_eq!(min, 300e3);
        assert_eq!(max, 3e6);
    }

    #[test]
    fn test_hf_band() {
        let band = HFBand;
        let (min, max) = band.freq_range();
        assert_eq!(min, 3e6);
        assert_eq!(max, 30e6);
    }

    #[test]
    fn test_vhf_band() {
        let band = VHFBand;
        let (min, max) = band.freq_range();
        assert_eq!(min, 30e6);
        assert_eq!(max, 300e6);
    }

    #[test]
    fn test_ehf_band() {
        let band = EHFBand;
        let (min, max) = band.freq_range();
        assert_eq!(min, 30e9);
        assert_eq!(max, 300e9);
    }

    #[test]
    fn test_band_continuity() {
        // Verify the lower frequency bands cover the spectrum without gaps
        // (VLF through VHF - the future stub bands)
        let lower_bands: Vec<(f32, f32)> = vec![
            VLFBand.freq_range(),
            LFBand.freq_range(),
            MFBand.freq_range(),
            HFBand.freq_range(),
            VHFBand.freq_range(),
        ];

        // Check that each band's max equals next band's min
        for i in 0..lower_bands.len() - 1 {
            let current_max = lower_bands[i].1;
            let next_min = lower_bands[i + 1].0;
            assert!(
                (current_max - next_min).abs() < 1.0,
                "Gap between bands at {} Hz",
                current_max
            );
        }

        // Note: UHF (300 MHz - 3 GHz) and SHF (3 - 30 GHz) are implemented
        // separately and connect VHF to EHF
    }
}
