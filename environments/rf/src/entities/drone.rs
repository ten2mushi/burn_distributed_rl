//! Drone Protocol Variants
//!
//! Models realistic drone RF signatures with multiple simultaneous links:
//! - **Video Link**: Analog FM (5.8 GHz) or Digital OFDM (2.4/5.8 GHz)
//! - **Control Link**: DSSS or FHSS (2.4 GHz)
//! - **Telemetry Link**: MAVLink or SC-FDM uplink
//!
//! Also models micro-Doppler signatures from propeller rotation.

#[cfg(feature = "simd")]
use std::simd::f32x8;

#[cfg(feature = "simd")]
use crate::physics::doppler::calc_micro_doppler_simd;
use crate::physics::doppler::calc_micro_doppler_scalar;

// ============================================================================
// Drone Protocol Types
// ============================================================================

/// Video link protocol variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DroneVideoProtocol {
    /// Analog FM video (legacy FPV drones)
    /// Frequency: 5.8 GHz, Bandwidth: ~20 MHz
    #[default]
    AnalogFM,

    /// Digital OFDM video (DJI, modern FPV)
    /// Frequency: 2.4/5.8 GHz, Bandwidth: 10-40 MHz
    DigitalOFDM,

    /// Enhanced Digital HD (DJI Lightbridge, O3)
    /// Frequency: 2.4/5.8 GHz, Bandwidth: 20-40 MHz, higher data rate
    EnhancedHD,

    /// Analog video disabled (telemetry-only drone)
    None,
}

/// Control link protocol variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DroneControlProtocol {
    /// Direct Sequence Spread Spectrum (many toy drones)
    /// Frequency: 2.4 GHz, lower range, simple
    DSSS,

    /// Frequency Hopping Spread Spectrum (FrSky ACCST, FlySky AFHDS)
    /// Frequency: 2.4 GHz, better interference rejection
    #[default]
    FHSS,

    /// ACCESS protocol (FrSky ACCESS)
    /// Frequency: 2.4 GHz, bidirectional telemetry
    ACCESS,

    /// ExpressLRS (ELRS) - Long range FHSS
    /// Frequency: 868/915/2.4 GHz, extreme range
    ELRS,

    /// Crossfire (TBS Crossfire)
    /// Frequency: 868/915 MHz, long range FHSS
    Crossfire,
}

/// Telemetry link protocol variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DroneTelemetryProtocol {
    /// MAVLink over serial/UDP (ArduPilot, PX4)
    /// Low bandwidth, variable update rate
    #[default]
    MAVLink,

    /// Smart Audio / MSP telemetry (Betaflight)
    /// VTX control and FC telemetry
    SmartAudio,

    /// SC-FDM uplink telemetry
    /// Low PAPR, efficient for battery-powered TX
    SCFDM,

    /// Telemetry integrated into control link (ACCESS, ELRS)
    Integrated,

    /// No dedicated telemetry link
    None,
}

/// Complete drone RF configuration
#[derive(Debug, Clone, Copy, Default)]
pub struct DroneRFConfig {
    /// Video downlink protocol
    pub video: DroneVideoProtocol,
    /// Control uplink protocol
    pub control: DroneControlProtocol,
    /// Telemetry protocol
    pub telemetry: DroneTelemetryProtocol,

    // Video link parameters
    /// Video center frequency (Hz)
    pub video_freq_hz: f32,
    /// Video bandwidth (Hz)
    pub video_bw_hz: f32,
    /// Video transmit power (dBm)
    pub video_power_dbm: f32,

    // Control link parameters
    /// Control center frequency (Hz)
    pub control_freq_hz: f32,
    /// Control bandwidth (Hz)
    pub control_bw_hz: f32,
    /// Number of FHSS channels (if applicable)
    pub fhss_channels: u8,
    /// FHSS hop rate (hops/second)
    pub fhss_hop_rate: f32,

    // Physical parameters for micro-Doppler
    /// Number of propeller blades
    pub blade_count: u8,
    /// Propeller blade length (meters)
    pub blade_length_m: f32,
    /// Propeller rotation rate (Hz) - RPM/60
    pub rotation_rate_hz: f32,
}

impl DroneRFConfig {
    /// Create a typical consumer FPV drone configuration
    pub fn consumer_fpv() -> Self {
        Self {
            video: DroneVideoProtocol::DigitalOFDM,
            control: DroneControlProtocol::FHSS,
            telemetry: DroneTelemetryProtocol::MAVLink,
            video_freq_hz: 5.8e9,
            video_bw_hz: 20e6,
            video_power_dbm: 25.0,
            control_freq_hz: 2.4e9,
            control_bw_hz: 2e6,
            fhss_channels: 50,
            fhss_hop_rate: 200.0, // 200 hops/s typical
            blade_count: 4,        // Quadcopter
            blade_length_m: 0.127, // 5-inch props (2.5-inch radius)
            rotation_rate_hz: 250.0, // 15000 RPM typical hover
        }
    }

    /// Create a DJI-style prosumer drone configuration
    pub fn prosumer_dji() -> Self {
        Self {
            video: DroneVideoProtocol::EnhancedHD,
            control: DroneControlProtocol::FHSS,
            telemetry: DroneTelemetryProtocol::Integrated,
            video_freq_hz: 5.8e9,
            video_bw_hz: 40e6,
            video_power_dbm: 20.0, // Lower power, better antennas
            control_freq_hz: 2.4e9,
            control_bw_hz: 5e6,
            fhss_channels: 80,
            fhss_hop_rate: 400.0,
            blade_count: 4,
            blade_length_m: 0.19, // Larger props for efficiency
            rotation_rate_hz: 200.0, // Slower rotation, larger props
        }
    }

    /// Create a long-range racing/exploration drone
    pub fn long_range() -> Self {
        Self {
            video: DroneVideoProtocol::DigitalOFDM,
            control: DroneControlProtocol::ELRS,
            telemetry: DroneTelemetryProtocol::Integrated,
            video_freq_hz: 5.8e9,
            video_bw_hz: 20e6,
            video_power_dbm: 25.0,
            control_freq_hz: 915e6, // Sub-GHz for range
            control_bw_hz: 500e3,   // Narrow for range
            fhss_channels: 100,
            fhss_hop_rate: 500.0, // Fast hopping
            blade_count: 4,
            blade_length_m: 0.178, // 7-inch props
            rotation_rate_hz: 166.0, // Lower RPM, efficiency
        }
    }

    /// Create a legacy analog FPV drone
    pub fn analog_fpv() -> Self {
        Self {
            video: DroneVideoProtocol::AnalogFM,
            control: DroneControlProtocol::DSSS,
            telemetry: DroneTelemetryProtocol::None,
            video_freq_hz: 5.8e9,
            video_bw_hz: 20e6,
            video_power_dbm: 25.0,
            control_freq_hz: 2.4e9,
            control_bw_hz: 1e6,
            fhss_channels: 0, // DSSS, no hopping
            fhss_hop_rate: 0.0,
            blade_count: 4,
            blade_length_m: 0.127,
            rotation_rate_hz: 250.0,
        }
    }

    /// Calculate maximum micro-Doppler shift (Hz) for the video link
    pub fn max_micro_doppler_hz(&self) -> f32 {
        let tip_velocity = 2.0 * std::f32::consts::PI * self.blade_length_m * self.rotation_rate_hz;
        (2.0 * tip_velocity / crate::constants::SPEED_OF_LIGHT) * self.video_freq_hz
    }
}

// ============================================================================
// Micro-Doppler Functions
// ============================================================================

/// Calculate micro-Doppler signature for a multi-blade propeller (scalar)
///
/// For n blades, the effective blade passage rate is n × rotation_rate.
/// This creates a higher-frequency modulation component in the micro-Doppler
/// signature. The spectral signature shows harmonics at n × f_rot.
///
/// # Arguments
/// * `config` - Drone RF configuration
/// * `carrier_freq` - Carrier frequency in Hz
/// * `time` - Current time in seconds
///
/// # Returns
/// Micro-Doppler shift in Hz (models blade passage rate)
pub fn calc_multi_blade_micro_doppler_scalar(
    config: &DroneRFConfig,
    carrier_freq: f32,
    time: f32,
) -> f32 {
    if config.blade_count == 0 || config.rotation_rate_hz <= 0.0 {
        return 0.0;
    }

    // Effective blade passage rate = blade_count × rotation_rate
    // This models the fact that each blade creates a Doppler signature
    // at the blade passage frequency, not the rotation frequency
    let blade_passage_rate = config.blade_count as f32 * config.rotation_rate_hz;

    // Calculate micro-Doppler at blade passage rate
    // The amplitude is reduced because only part of each blade contributes
    // to the instantaneous Doppler shift at any moment
    calc_micro_doppler_scalar(
        blade_passage_rate,
        config.blade_length_m,
        carrier_freq,
        time,
    )
}

/// Calculate micro-Doppler signature for a multi-blade propeller (SIMD)
///
/// For n blades, the effective blade passage rate is n × rotation_rate.
#[cfg(feature = "simd")]
pub fn calc_multi_blade_micro_doppler_simd(
    blade_count: u8,
    rotation_rate_hz: f32x8,
    blade_length_m: f32x8,
    carrier_freq: f32x8,
    time: f32x8,
) -> f32x8 {
    if blade_count == 0 {
        return f32x8::splat(0.0);
    }

    // Effective blade passage rate = blade_count × rotation_rate
    let blade_passage_rate = f32x8::splat(blade_count as f32) * rotation_rate_hz;

    calc_micro_doppler_simd(
        blade_passage_rate,
        blade_length_m,
        carrier_freq,
        time,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drone_config_presets() {
        let consumer = DroneRFConfig::consumer_fpv();
        assert_eq!(consumer.video, DroneVideoProtocol::DigitalOFDM);
        assert_eq!(consumer.control, DroneControlProtocol::FHSS);
        assert!(consumer.video_freq_hz > 5e9);

        let dji = DroneRFConfig::prosumer_dji();
        assert_eq!(dji.video, DroneVideoProtocol::EnhancedHD);
        assert!(dji.video_bw_hz > consumer.video_bw_hz);

        let lr = DroneRFConfig::long_range();
        assert_eq!(lr.control, DroneControlProtocol::ELRS);
        assert!(lr.control_freq_hz < 1e9); // Sub-GHz
    }

    #[test]
    fn test_max_micro_doppler() {
        let config = DroneRFConfig::consumer_fpv();
        let max_md = config.max_micro_doppler_hz();

        // Expected: (2 × 2π × 0.127 × 250 / 3e8) × 5.8e9 ≈ 7700 Hz
        assert!(max_md > 5000.0, "Max micro-Doppler should be >5 kHz: {}", max_md);
        assert!(max_md < 15000.0, "Max micro-Doppler should be <15 kHz: {}", max_md);
    }

    #[test]
    fn test_multi_blade_micro_doppler() {
        let config = DroneRFConfig::consumer_fpv();

        // Sample over one rotation
        let period = 1.0 / config.rotation_rate_hz;
        let mut samples = Vec::new();
        for i in 0..100 {
            let t = i as f32 * period / 100.0;
            samples.push(calc_multi_blade_micro_doppler_scalar(&config, config.video_freq_hz, t));
        }

        // Should have variation (not constant)
        let min = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max - min > 1000.0, "Should have >1 kHz variation: min={}, max={}", min, max);
    }

    #[test]
    fn test_protocol_defaults() {
        assert_eq!(DroneVideoProtocol::default(), DroneVideoProtocol::AnalogFM);
        assert_eq!(DroneControlProtocol::default(), DroneControlProtocol::FHSS);
        assert_eq!(DroneTelemetryProtocol::default(), DroneTelemetryProtocol::MAVLink);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_micro_doppler() {
        let rotation = f32x8::splat(250.0);
        let blade = f32x8::splat(0.127);
        let freq = f32x8::splat(5.8e9);
        let time = f32x8::from_array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]);

        let result = calc_multi_blade_micro_doppler_simd(4, rotation, blade, freq, time);
        let arr: [f32; 8] = result.into();

        // All values should be finite
        for &v in &arr {
            assert!(v.is_finite(), "Micro-Doppler should be finite: {}", v);
        }
    }
}
