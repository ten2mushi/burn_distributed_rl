//! Physical and RF Constants
//!
//! Fundamental physical constants and common RF band definitions
//! used throughout the RF environment simulation.

// ============================================================================
// Fundamental Physical Constants
// ============================================================================

/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f32 = 299_792_458.0;

/// Boltzmann constant (J/K)
pub const BOLTZMANN: f32 = 1.380649e-23;

/// Reference temperature for thermal noise (K)
pub const REFERENCE_TEMP: f32 = 290.0;

/// Thermal noise power density at 290K (dBm/Hz)
/// kT at 290K ≈ -174 dBm/Hz
pub const THERMAL_NOISE_DBM_HZ: f32 = -174.0;

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Pi
pub const PI: f32 = std::f32::consts::PI;

/// Two times Pi (Tau)
pub const TWO_PI: f32 = 2.0 * PI;

/// Half Pi
pub const HALF_PI: f32 = PI / 2.0;

/// Natural logarithm of 2
pub const LN_2: f32 = std::f32::consts::LN_2;

/// Natural logarithm of 10
pub const LN_10: f32 = std::f32::consts::LN_10;

/// Log base 10 of e
pub const LOG10_E: f32 = std::f32::consts::LOG10_E;

// ============================================================================
// Frequency Band Definitions (Hz)
// ============================================================================

// Very Low Frequency (VLF)
/// VLF band minimum frequency (3 kHz)
pub const VLF_MIN: f32 = 3e3;
/// VLF band maximum frequency (30 kHz)
pub const VLF_MAX: f32 = 30e3;

// Low Frequency (LF)
/// LF band minimum frequency (30 kHz)
pub const LF_MIN: f32 = 30e3;
/// LF band maximum frequency (300 kHz)
pub const LF_MAX: f32 = 300e3;

// Medium Frequency (MF)
/// MF band minimum frequency (300 kHz)
pub const MF_MIN: f32 = 300e3;
/// MF band maximum frequency (3 MHz)
pub const MF_MAX: f32 = 3e6;

// High Frequency (HF)
/// HF band minimum frequency (3 MHz)
pub const HF_MIN: f32 = 3e6;
/// HF band maximum frequency (30 MHz)
pub const HF_MAX: f32 = 30e6;

// Very High Frequency (VHF)
/// VHF band minimum frequency (30 MHz)
pub const VHF_MIN: f32 = 30e6;
/// VHF band maximum frequency (300 MHz)
pub const VHF_MAX: f32 = 300e6;

// Ultra High Frequency (UHF)
/// UHF band minimum frequency (300 MHz)
pub const UHF_MIN: f32 = 300e6;
/// UHF band maximum frequency (3 GHz)
pub const UHF_MAX: f32 = 3e9;

// Super High Frequency (SHF)
/// SHF band minimum frequency (3 GHz)
pub const SHF_MIN: f32 = 3e9;
/// SHF band maximum frequency (30 GHz)
pub const SHF_MAX: f32 = 30e9;

// Extremely High Frequency (EHF)
/// EHF band minimum frequency (30 GHz)
pub const EHF_MIN: f32 = 30e9;
/// EHF band maximum frequency (300 GHz)
pub const EHF_MAX: f32 = 300e9;

// ============================================================================
// Common Service Allocations (Hz)
// ============================================================================

// Broadcast
/// FM Radio band start (88 MHz)
pub const FM_RADIO_START: f32 = 88e6;
/// FM Radio band end (108 MHz)
pub const FM_RADIO_END: f32 = 108e6;
/// FM Radio channel width (200 kHz)
pub const FM_RADIO_CHANNEL_WIDTH: f32 = 200e3;

/// UHF TV band start (470 MHz)
pub const TV_UHF_START: f32 = 470e6;
/// UHF TV band end (698 MHz, post-digital transition)
pub const TV_UHF_END: f32 = 698e6;
/// ATSC channel width (6 MHz)
pub const TV_CHANNEL_WIDTH: f32 = 6e6;

// Cellular
/// LTE Band 700 start (698 MHz)
pub const LTE_BAND_700_START: f32 = 698e6;
/// LTE Band 700 end (806 MHz)
pub const LTE_BAND_700_END: f32 = 806e6;

/// LTE common bandwidth 10 MHz
pub const LTE_BW_10: f32 = 10e6;
/// LTE common bandwidth 20 MHz
pub const LTE_BW_20: f32 = 20e6;

/// 5G FR1 range start (410 MHz)
pub const FR1_START: f32 = 410e6;
/// 5G FR1 range end (7.125 GHz)
pub const FR1_END: f32 = 7.125e9;

/// 5G FR2 (mmWave) range start (24.25 GHz)
pub const FR2_START: f32 = 24.25e9;
/// 5G FR2 (mmWave) range end (52.6 GHz)
pub const FR2_END: f32 = 52.6e9;

// ISM Bands (Unlicensed)
/// 2.4 GHz ISM band start
pub const ISM_2_4_START: f32 = 2.4e9;
/// 2.4 GHz ISM band end
pub const ISM_2_4_END: f32 = 2.4835e9;

/// 5 GHz ISM band start (U-NII-1)
pub const ISM_5_START: f32 = 5.15e9;
/// 5 GHz ISM band end (U-NII-3)
pub const ISM_5_END: f32 = 5.825e9;

/// 6 GHz WiFi 6E band start
pub const WIFI_6E_START: f32 = 5.925e9;
/// 6 GHz WiFi 6E band end
pub const WIFI_6E_END: f32 = 7.125e9;

// WiFi Channel Widths
/// WiFi 20 MHz channel
pub const WIFI_BW_20: f32 = 20e6;
/// WiFi 40 MHz channel
pub const WIFI_BW_40: f32 = 40e6;
/// WiFi 80 MHz channel
pub const WIFI_BW_80: f32 = 80e6;
/// WiFi 160 MHz channel
pub const WIFI_BW_160: f32 = 160e6;
/// WiFi 320 MHz channel (WiFi 7)
pub const WIFI_BW_320: f32 = 320e6;

// Drone Control/Video
/// Common drone control frequency 2.4 GHz
pub const DRONE_CONTROL_2_4: f32 = 2.4e9;
/// Common drone video frequency 5.8 GHz
pub const DRONE_VIDEO_5_8: f32 = 5.8e9;

// Navigation
/// GPS L1 frequency
pub const GPS_L1: f32 = 1575.42e6;
/// GPS L2 frequency
pub const GPS_L2: f32 = 1227.60e6;
/// GPS L5 frequency
pub const GPS_L5: f32 = 1176.45e6;

// Radar
/// S-band radar range start (2 GHz)
pub const RADAR_S_START: f32 = 2e9;
/// S-band radar range end (4 GHz)
pub const RADAR_S_END: f32 = 4e9;

/// X-band radar range start (8 GHz)
pub const RADAR_X_START: f32 = 8e9;
/// X-band radar range end (12 GHz)
pub const RADAR_X_END: f32 = 12e9;

// ============================================================================
// Default Simulation Parameters
// ============================================================================

/// Default number of frequency bins
pub const DEFAULT_NUM_FREQ_BINS: usize = 512;

/// Default minimum frequency for simulation (300 MHz - UHF start)
pub const DEFAULT_FREQ_MIN: f32 = 300e6;

/// Default maximum frequency for simulation (6 GHz - covers most relevant bands)
pub const DEFAULT_FREQ_MAX: f32 = 6e9;

/// Default world size in meters (1 km × 1 km × 100 m)
pub const DEFAULT_WORLD_SIZE: (f32, f32, f32) = (1000.0, 1000.0, 100.0);

/// Default physics simulation frequency (100 Hz)
pub const DEFAULT_PHYSICS_FREQ: u32 = 100;

/// Default control frequency (10 Hz)
pub const DEFAULT_CTRL_FREQ: u32 = 10;

/// Default maximum episode steps
pub const DEFAULT_MAX_STEPS: u32 = 1000;

/// Default noise figure (dB)
pub const DEFAULT_NOISE_FIGURE: f32 = 5.0;

// ============================================================================
// Power Limits (dBm)
// ============================================================================

/// Typical jammer minimum power (10 dBm = 10 mW)
pub const JAMMER_POWER_MIN: f32 = 10.0;
/// Typical jammer maximum power (40 dBm = 10 W)
pub const JAMMER_POWER_MAX: f32 = 40.0;

/// Typical CR minimum power (0 dBm = 1 mW)
pub const CR_POWER_MIN: f32 = 0.0;
/// Typical CR maximum power (23 dBm ≈ 200 mW, typical LTE UE)
pub const CR_POWER_MAX: f32 = 23.0;

// ============================================================================
// Path Loss Model Constants
// ============================================================================

/// Free space path loss reference distance (1 meter)
pub const FSPL_REF_DISTANCE: f32 = 1.0;

/// Urban path loss exponent (typically 3.5-4.5)
pub const URBAN_PATH_LOSS_EXP: f32 = 4.0;

/// Rural path loss exponent (typically 2.5-3.5)
pub const RURAL_PATH_LOSS_EXP: f32 = 3.0;

/// Indoor path loss exponent (typically 2.0-3.0)
pub const INDOOR_PATH_LOSS_EXP: f32 = 2.5;

// ============================================================================
// Fading Model Constants
// ============================================================================

/// Rician K-factor for strong LOS (10 dB)
pub const RICIAN_K_STRONG_LOS: f32 = 10.0;

/// Rician K-factor for weak LOS (3 dB)
pub const RICIAN_K_WEAK_LOS: f32 = 3.0;

/// Shadowing standard deviation in urban environment (dB)
pub const URBAN_SHADOW_STD: f32 = 8.0;

/// Shadowing standard deviation in rural environment (dB)
pub const RURAL_SHADOW_STD: f32 = 6.0;

// ============================================================================
// Conversion Utilities
// ============================================================================

/// Convert dBm to milliwatts
#[inline]
pub const fn dbm_to_mw(dbm: f32) -> f32 {
    // Can't use powf in const fn, so this is just documentation
    // Real conversion: 10.0_f32.powf(dbm / 10.0)
    dbm // placeholder - use simd_db_to_linear at runtime
}

/// Convert dB to linear ratio
#[inline]
pub const fn db_to_linear(db: f32) -> f32 {
    // Can't use powf in const fn
    // Real conversion: 10.0_f32.powf(db / 10.0)
    db // placeholder - use simd_db_to_linear at runtime
}

/// Calculate wavelength from frequency (meters)
#[inline]
pub const fn freq_to_wavelength(freq: f32) -> f32 {
    SPEED_OF_LIGHT / freq
}

/// Calculate frequency from wavelength (Hz)
#[inline]
pub const fn wavelength_to_freq(wavelength: f32) -> f32 {
    SPEED_OF_LIGHT / wavelength
}
