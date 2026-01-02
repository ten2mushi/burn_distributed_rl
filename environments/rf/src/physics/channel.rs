//! Composite Channel Model
//!
//! Combines all physics effects into a unified channel model:
//! - Path loss (FSPL, log-distance, Hata, COST-231)
//! - Fading (Rayleigh, Rician, shadowing)
//! - Multipath (ITU EPA, EVA, ETU)
//! - Atmospheric (gaseous absorption, rain)
//! - Doppler effects
//!
//! Provides both SIMD and scalar interfaces for flexible use.

#[cfg(feature = "simd")]
use std::simd::f32x8;

#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

use super::atmospheric::{
    atmospheric_attenuation_db_per_km_scalar, rain_attenuation_db_per_km_scalar,
};
use super::doppler::calc_doppler_shift_scalar;
use super::fading::{
    apply_composite_fading_scalar_raw, apply_rayleigh_fading_scalar_raw, apply_rician_fading_scalar_raw,
};
use super::multipath::{
    apply_multipath_power_scalar, generate_cir_scalar, ChannelImpulseResponse, ITUChannelModel,
};
use super::path_loss::{
    cost231_hata_db_scalar, fspl_db_scalar, ground_wave_db_scalar, hata_urban_db_scalar,
    log_distance_db_scalar,
};

#[cfg(feature = "simd")]
use super::doppler::calc_doppler_shift_simd;
#[cfg(feature = "simd")]
use super::fading::{apply_composite_fading_simd_raw, apply_rayleigh_fading_simd_raw, apply_rician_fading_simd_raw};
#[cfg(feature = "simd")]
use super::path_loss::{fspl_db_simd, log_distance_db_simd};

// ============================================================================
// Channel Configuration Enums
// ============================================================================

/// Path loss model selection
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PathLossModel {
    /// Free Space Path Loss - simplest model
    #[default]
    FreeSpace,
    /// Log-distance model with configurable exponent
    LogDistance {
        path_loss_exponent: f32,
        reference_distance_m: f32,
    },
    /// Okumura-Hata for urban macro cells (150-1500 MHz)
    HataUrban {
        tx_height_m: f32,
        rx_height_m: f32,
    },
    /// COST-231 Hata extension (1500-2000 MHz)
    Cost231 {
        tx_height_m: f32,
        rx_height_m: f32,
        is_large_city: bool,
    },
    /// Ground wave for VLF/LF
    GroundWave { conductivity: f32 },
}

/// Fading type selection
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FadingType {
    /// No fading applied
    #[default]
    None,
    /// Rayleigh fading for NLOS
    Rayleigh,
    /// Rician fading with K-factor (LOS + scattered)
    Rician { k_factor: f32 },
    /// Composite: Rician small-scale + log-normal shadowing
    Composite {
        k_factor: f32,
        shadow_sigma_db: f32,
    },
}

/// Multipath model selection
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum MultipathModel {
    /// No multipath (flat channel)
    #[default]
    None,
    /// ITU Extended Pedestrian A
    EPA,
    /// ITU Extended Vehicular A
    EVA,
    /// ITU Extended Typical Urban
    ETU,
}

impl From<MultipathModel> for Option<ITUChannelModel> {
    fn from(model: MultipathModel) -> Option<ITUChannelModel> {
        match model {
            MultipathModel::None => None,
            MultipathModel::EPA => Some(ITUChannelModel::EPA),
            MultipathModel::EVA => Some(ITUChannelModel::EVA),
            MultipathModel::ETU => Some(ITUChannelModel::ETU),
        }
    }
}

/// Atmospheric effects configuration
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AtmosphericConfig {
    /// Enable gaseous absorption (O2, H2O)
    pub gaseous_absorption: bool,
    /// Rain rate in mm/hour (0 = no rain)
    pub rain_rate_mm_per_hour: f32,
    /// Fog liquid water content in g/m³ (0 = no fog)
    pub fog_lwc_g_per_m3: f32,
}

// ============================================================================
// Channel Parameters
// ============================================================================

/// Complete channel configuration
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelParams {
    /// Path loss model
    pub path_loss: PathLossModel,
    /// Fading type
    pub fading: FadingType,
    /// Multipath model
    pub multipath: MultipathModel,
    /// Atmospheric effects
    pub atmospheric: AtmosphericConfig,
    /// Carrier frequency in Hz
    pub carrier_freq_hz: f32,
    /// Noise floor in dBm
    pub noise_floor_dbm: f32,
    /// Minimum received power threshold in dBm
    pub sensitivity_dbm: f32,
}

impl Default for ChannelParams {
    fn default() -> Self {
        Self {
            path_loss: PathLossModel::FreeSpace,
            fading: FadingType::None,
            multipath: MultipathModel::None,
            atmospheric: AtmosphericConfig::default(),
            carrier_freq_hz: 2.4e9, // 2.4 GHz
            noise_floor_dbm: -100.0,
            sensitivity_dbm: -120.0,
        }
    }
}

impl ChannelParams {
    /// Create new channel params with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set path loss model
    pub fn with_path_loss(mut self, model: PathLossModel) -> Self {
        self.path_loss = model;
        self
    }

    /// Set fading type
    pub fn with_fading(mut self, fading: FadingType) -> Self {
        self.fading = fading;
        self
    }

    /// Set multipath model
    pub fn with_multipath(mut self, multipath: MultipathModel) -> Self {
        self.multipath = multipath;
        self
    }

    /// Set carrier frequency
    pub fn with_carrier_freq(mut self, freq_hz: f32) -> Self {
        self.carrier_freq_hz = freq_hz;
        self
    }

    /// Set noise floor
    pub fn with_noise_floor(mut self, noise_dbm: f32) -> Self {
        self.noise_floor_dbm = noise_dbm;
        self
    }

    /// Enable atmospheric gaseous absorption
    pub fn with_gaseous_absorption(mut self) -> Self {
        self.atmospheric.gaseous_absorption = true;
        self
    }

    /// Set rain rate for attenuation
    pub fn with_rain(mut self, rate_mm_per_hour: f32) -> Self {
        self.atmospheric.rain_rate_mm_per_hour = rate_mm_per_hour;
        self
    }

    /// Create urban channel preset
    pub fn urban() -> Self {
        Self {
            path_loss: PathLossModel::LogDistance {
                path_loss_exponent: 3.5,
                reference_distance_m: 1.0,
            },
            fading: FadingType::Rayleigh,
            multipath: MultipathModel::ETU,
            atmospheric: AtmosphericConfig::default(),
            carrier_freq_hz: 2.4e9,
            noise_floor_dbm: -100.0,
            sensitivity_dbm: -110.0,
        }
    }

    /// Create rural LOS channel preset
    pub fn rural_los() -> Self {
        Self {
            path_loss: PathLossModel::FreeSpace,
            fading: FadingType::Rician { k_factor: 10.0 },
            multipath: MultipathModel::EPA,
            atmospheric: AtmosphericConfig {
                gaseous_absorption: true,
                rain_rate_mm_per_hour: 0.0,
                fog_lwc_g_per_m3: 0.0,
            },
            carrier_freq_hz: 900e6,
            noise_floor_dbm: -110.0,
            sensitivity_dbm: -120.0,
        }
    }

    /// Create vehicular channel preset
    pub fn vehicular() -> Self {
        Self {
            path_loss: PathLossModel::HataUrban {
                tx_height_m: 30.0,
                rx_height_m: 1.5,
            },
            fading: FadingType::Composite {
                k_factor: 0.0,
                shadow_sigma_db: 8.0,
            },
            multipath: MultipathModel::EVA,
            atmospheric: AtmosphericConfig::default(),
            carrier_freq_hz: 1.8e9,
            noise_floor_dbm: -100.0,
            sensitivity_dbm: -110.0,
        }
    }

    /// Create satellite/mmWave channel preset
    pub fn satellite() -> Self {
        Self {
            path_loss: PathLossModel::FreeSpace,
            fading: FadingType::Rician { k_factor: 15.0 },
            multipath: MultipathModel::None,
            atmospheric: AtmosphericConfig {
                gaseous_absorption: true,
                rain_rate_mm_per_hour: 0.0,
                fog_lwc_g_per_m3: 0.0,
            },
            carrier_freq_hz: 28e9, // Ka-band
            noise_floor_dbm: -90.0,
            sensitivity_dbm: -100.0,
        }
    }
}

// ============================================================================
// Channel State (for time-varying effects)
// ============================================================================

/// Time-varying channel state
#[derive(Debug, Clone)]
pub struct ChannelState {
    /// Current simulation time in seconds
    pub time: f32,
    /// Mobile velocity in m/s (for Doppler)
    pub velocity_mps: f32,
    /// Current channel impulse response
    pub cir: Option<ChannelImpulseResponse>,
    /// Last update time for CIR
    pub last_cir_update: f32,
    /// Current shadowing value in dB (slow-varying)
    pub shadow_db: f32,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            time: 0.0,
            velocity_mps: 0.0,
            cir: None,
            last_cir_update: -1.0,
            shadow_db: 0.0,
        }
    }
}

impl ChannelState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_velocity(mut self, velocity: f32) -> Self {
        self.velocity_mps = velocity;
        self
    }

    pub fn advance_time(&mut self, dt: f32) {
        self.time += dt;
    }

    /// Check if CIR needs update based on coherence time
    pub fn needs_cir_update(&self, coherence_time: f32) -> bool {
        self.cir.is_none() || (self.time - self.last_cir_update) > coherence_time * 0.5
    }
}

// ============================================================================
// Channel Output
// ============================================================================

/// Result of channel application
#[derive(Debug, Clone, Copy)]
pub struct ChannelOutput {
    /// Received power in dBm
    pub rx_power_dbm: f32,
    /// Path loss in dB
    pub path_loss_db: f32,
    /// Fading attenuation in dB
    pub fading_db: f32,
    /// Atmospheric attenuation in dB
    pub atmospheric_db: f32,
    /// Doppler shift in Hz
    pub doppler_hz: f32,
    /// Signal is above sensitivity threshold
    pub is_detectable: bool,
}

// ============================================================================
// Scalar Channel Application
// ============================================================================

/// Apply complete channel model (scalar version)
///
/// # Arguments
/// * `tx_power_dbm` - Transmit power in dBm
/// * `distance_m` - Distance between TX and RX in meters
/// * `params` - Channel parameters
/// * `state` - Channel state (for time-varying effects)
/// * `rng` - Random number generator (returns Gaussian pair)
///
/// # Returns
/// Channel output with received power and diagnostics
pub fn apply_channel_scalar(
    tx_power_dbm: f32,
    distance_m: f32,
    params: &ChannelParams,
    state: &ChannelState,
    rng: &mut impl FnMut() -> (f32, f32),
) -> ChannelOutput {
    let freq_hz = params.carrier_freq_hz;

    // 1. Path Loss
    let path_loss_db = match params.path_loss {
        PathLossModel::FreeSpace => fspl_db_scalar(distance_m, freq_hz),
        PathLossModel::LogDistance {
            path_loss_exponent,
            reference_distance_m,
        } => {
            let ref_loss = fspl_db_scalar(reference_distance_m, freq_hz);
            log_distance_db_scalar(distance_m, reference_distance_m, path_loss_exponent, ref_loss)
        }
        PathLossModel::HataUrban {
            tx_height_m,
            rx_height_m,
        } => {
            let freq_mhz = freq_hz / 1e6;
            let dist_km = distance_m / 1000.0;
            hata_urban_db_scalar(dist_km, freq_mhz, tx_height_m, rx_height_m)
        }
        PathLossModel::Cost231 {
            tx_height_m,
            rx_height_m,
            is_large_city,
        } => {
            let freq_mhz = freq_hz / 1e6;
            let dist_km = distance_m / 1000.0;
            cost231_hata_db_scalar(dist_km, freq_mhz, tx_height_m, rx_height_m, is_large_city)
        }
        PathLossModel::GroundWave { conductivity: _ } => {
            // Simplified ground wave
            ground_wave_db_scalar(distance_m, freq_hz)
        }
    };

    let mut power_linear = tx_power_dbm_to_linear(tx_power_dbm - path_loss_db);

    // 2. Atmospheric attenuation
    let mut atmospheric_db = 0.0;
    if params.atmospheric.gaseous_absorption {
        let freq_ghz = freq_hz / 1e9;
        let dist_km = distance_m / 1000.0;
        atmospheric_db += atmospheric_attenuation_db_per_km_scalar(freq_ghz) * dist_km;
    }
    if params.atmospheric.rain_rate_mm_per_hour > 0.0 {
        let freq_ghz = freq_hz / 1e9;
        let dist_km = distance_m / 1000.0;
        atmospheric_db += rain_attenuation_db_per_km_scalar(
            freq_ghz,
            params.atmospheric.rain_rate_mm_per_hour,
        ) * dist_km;
    }
    if atmospheric_db > 0.0 {
        power_linear *= 10.0_f32.powf(-atmospheric_db / 10.0);
    }

    // 3. Fading
    let mut fading_db = 0.0;
    let power_before_fading = power_linear;

    power_linear = match params.fading {
        FadingType::None => power_linear,
        FadingType::Rayleigh => apply_rayleigh_fading_scalar_raw(power_linear, rng),
        FadingType::Rician { k_factor } => {
            apply_rician_fading_scalar_raw(power_linear, k_factor, rng)
        }
        FadingType::Composite {
            k_factor,
            shadow_sigma_db,
        } => apply_composite_fading_scalar_raw(power_linear, k_factor, shadow_sigma_db, rng),
    };

    if power_before_fading > 0.0 && power_linear > 0.0 {
        fading_db = 10.0 * (power_before_fading / power_linear).log10();
    }

    // 4. Multipath (if enabled)
    if let Some(itu_model) = Option::<ITUChannelModel>::from(params.multipath) {
        if let Some(ref cir) = state.cir {
            power_linear = apply_multipath_power_scalar(power_linear, cir);
        } else {
            // Generate CIR on the fly
            let cir = generate_cir_scalar(itu_model, state.velocity_mps, freq_hz, state.time, rng);
            power_linear = apply_multipath_power_scalar(power_linear, &cir);
        }
    }

    // 5. Doppler shift
    let doppler_hz = calc_doppler_shift_scalar(state.velocity_mps, freq_hz);

    // Convert back to dBm
    let rx_power_dbm = linear_to_dbm(power_linear);
    let is_detectable = rx_power_dbm >= params.sensitivity_dbm;

    ChannelOutput {
        rx_power_dbm,
        path_loss_db,
        fading_db,
        atmospheric_db,
        doppler_hz,
        is_detectable,
    }
}

/// Simplified channel application without state
pub fn apply_channel_simple(
    tx_power_dbm: f32,
    distance_m: f32,
    params: &ChannelParams,
    rng: &mut impl FnMut() -> (f32, f32),
) -> f32 {
    let state = ChannelState::default();
    apply_channel_scalar(tx_power_dbm, distance_m, params, &state, rng).rx_power_dbm
}

// ============================================================================
// SIMD Channel Application
// ============================================================================

/// SIMD channel output for 8 parallel channels
#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy)]
pub struct SimdChannelOutput {
    /// Received power in dBm (8 channels)
    pub rx_power_dbm: f32x8,
    /// Path loss in dB (8 channels)
    pub path_loss_db: f32x8,
    /// Fading attenuation in dB (8 channels)
    pub fading_db: f32x8,
    /// Atmospheric attenuation in dB (8 channels)
    pub atmospheric_db: f32x8,
    /// Doppler shift in Hz (8 channels)
    pub doppler_hz: f32x8,
}

/// Apply channel model to 8 parallel transmissions (SIMD)
#[cfg(feature = "simd")]
pub fn apply_channel_simd(
    tx_power_dbm: f32x8,
    distance_m: f32x8,
    frequency_hz: f32x8,
    velocity_mps: f32x8,
    _time: f32x8,
    path_loss_model: PathLossModel,
    fading_type: FadingType,
    rng: &mut SimdRng,
) -> SimdChannelOutput {
    // 1. Path Loss
    let path_loss_db = match path_loss_model {
        PathLossModel::FreeSpace => fspl_db_simd(distance_m, frequency_hz),
        PathLossModel::LogDistance {
            path_loss_exponent,
            reference_distance_m,
        } => {
            let ref_dist = f32x8::splat(reference_distance_m);
            let ref_loss = fspl_db_simd(ref_dist, frequency_hz);
            log_distance_db_simd(
                distance_m,
                ref_dist,
                f32x8::splat(path_loss_exponent),
                ref_loss,
            )
        }
        _ => fspl_db_simd(distance_m, frequency_hz), // Default to FSPL for other models
    };

    let mut power_linear = simd_dbm_to_linear(tx_power_dbm - path_loss_db);

    // 2. Fading
    let power_before = power_linear;
    power_linear = match fading_type {
        FadingType::None => power_linear,
        FadingType::Rayleigh => apply_rayleigh_fading_simd_raw(power_linear, rng),
        FadingType::Rician { k_factor } => {
            apply_rician_fading_simd_raw(power_linear, f32x8::splat(k_factor), rng)
        }
        FadingType::Composite {
            k_factor,
            shadow_sigma_db,
        } => apply_composite_fading_simd_raw(
            power_linear,
            f32x8::splat(k_factor),
            f32x8::splat(shadow_sigma_db),
            rng,
        ),
    };

    // Calculate fading dB
    let fading_db = f32x8::splat(10.0) * simd_log10(power_before / power_linear);

    // 3. Doppler
    let doppler_hz = calc_doppler_shift_simd(velocity_mps, frequency_hz);

    // Convert to dBm
    let rx_power_dbm = simd_linear_to_dbm(power_linear);

    SimdChannelOutput {
        rx_power_dbm,
        path_loss_db,
        fading_db,
        atmospheric_db: f32x8::splat(0.0), // Simplified
        doppler_hz,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert dBm to linear power (mW)
#[inline]
fn tx_power_dbm_to_linear(dbm: f32) -> f32 {
    10.0_f32.powf(dbm / 10.0)
}

/// Convert linear power (mW) to dBm
#[inline]
fn linear_to_dbm(power_mw: f32) -> f32 {
    if power_mw <= 0.0 {
        -200.0 // Very low floor
    } else {
        10.0 * power_mw.log10()
    }
}

#[cfg(feature = "simd")]
#[inline]
fn simd_dbm_to_linear(dbm: f32x8) -> f32x8 {
    use crate::simd_rf::math::simd_pow;
    simd_pow(f32x8::splat(10.0), dbm / f32x8::splat(10.0))
}

#[cfg(feature = "simd")]
#[inline]
fn simd_linear_to_dbm(power: f32x8) -> f32x8 {
    f32x8::splat(10.0) * simd_log10(power)
}

#[cfg(feature = "simd")]
#[inline]
fn simd_log10(x: f32x8) -> f32x8 {
    use crate::simd_rf::math::simd_log;
    // simd_log returns natural log, convert to log10
    simd_log(x) / f32x8::splat(std::f32::consts::LN_10)
}

// ============================================================================
// SINR Calculation
// ============================================================================

/// Calculate Signal-to-Interference-plus-Noise Ratio
///
/// # Arguments
/// * `signal_power_dbm` - Desired signal power in dBm
/// * `interference_powers_dbm` - Slice of interferer powers in dBm
/// * `noise_floor_dbm` - Thermal noise floor in dBm
///
/// # Returns
/// SINR in dB
pub fn calculate_sinr_db(
    signal_power_dbm: f32,
    interference_powers_dbm: &[f32],
    noise_floor_dbm: f32,
) -> f32 {
    let signal_linear = tx_power_dbm_to_linear(signal_power_dbm);
    let noise_linear = tx_power_dbm_to_linear(noise_floor_dbm);

    let interference_linear: f32 = interference_powers_dbm
        .iter()
        .map(|&p| tx_power_dbm_to_linear(p))
        .sum();

    let denominator = noise_linear + interference_linear;
    if denominator <= 0.0 {
        return 100.0; // Very high SINR
    }

    10.0 * (signal_linear / denominator).log10()
}

/// Calculate SNR (no interference)
pub fn calculate_snr_db(signal_power_dbm: f32, noise_floor_dbm: f32) -> f32 {
    signal_power_dbm - noise_floor_dbm
}

// ============================================================================
// Link Budget Calculation
// ============================================================================

/// Link budget calculation result
#[derive(Debug, Clone, Copy)]
pub struct LinkBudget {
    /// Maximum allowable path loss in dB
    pub max_path_loss_db: f32,
    /// Link margin in dB
    pub margin_db: f32,
    /// Estimated maximum range in meters
    pub max_range_m: f32,
}

/// Calculate link budget
///
/// # Arguments
/// * `tx_power_dbm` - Transmit power in dBm
/// * `tx_antenna_gain_dbi` - TX antenna gain in dBi
/// * `rx_antenna_gain_dbi` - RX antenna gain in dBi
/// * `required_snr_db` - Required SNR at receiver
/// * `noise_floor_dbm` - Receiver noise floor in dBm
/// * `frequency_hz` - Operating frequency in Hz
///
/// # Returns
/// Link budget analysis
pub fn calculate_link_budget(
    tx_power_dbm: f32,
    tx_antenna_gain_dbi: f32,
    rx_antenna_gain_dbi: f32,
    required_snr_db: f32,
    noise_floor_dbm: f32,
    frequency_hz: f32,
) -> LinkBudget {
    // EIRP
    let eirp_dbm = tx_power_dbm + tx_antenna_gain_dbi;

    // Required signal at receiver
    let required_signal_dbm = noise_floor_dbm + required_snr_db;

    // Minimum receivable signal with RX gain
    let min_rx_signal = required_signal_dbm - rx_antenna_gain_dbi;

    // Maximum path loss
    let max_path_loss_db = eirp_dbm - min_rx_signal;

    // Estimate range from FSPL
    // FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
    // d = 10^((FSPL - 20*log10(f) - 20*log10(4π/c)) / 20)
    let c = 299_792_458.0_f32;
    let k = 20.0 * (4.0 * std::f32::consts::PI / c).log10();
    let max_range_m =
        10.0_f32.powf((max_path_loss_db - 20.0 * frequency_hz.log10() - k) / 20.0);

    // Margin (assuming some path loss)
    let margin_db = max_path_loss_db - fspl_db_scalar(1000.0, frequency_hz);

    LinkBudget {
        max_path_loss_db,
        margin_db,
        max_range_m,
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_rng() -> impl FnMut() -> (f32, f32) {
        let mut state = 12345u32;
        move || {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let r1 = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let r2 = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            (r1, r2)
        }
    }

    #[test]
    fn test_channel_params_default() {
        let params = ChannelParams::default();
        assert_eq!(params.path_loss, PathLossModel::FreeSpace);
        assert_eq!(params.fading, FadingType::None);
        assert_eq!(params.multipath, MultipathModel::None);
    }

    #[test]
    fn test_channel_params_builder() {
        let params = ChannelParams::new()
            .with_path_loss(PathLossModel::LogDistance {
                path_loss_exponent: 3.0,
                reference_distance_m: 1.0,
            })
            .with_fading(FadingType::Rayleigh)
            .with_carrier_freq(5.8e9);

        assert!(matches!(params.path_loss, PathLossModel::LogDistance { .. }));
        assert_eq!(params.fading, FadingType::Rayleigh);
        assert!((params.carrier_freq_hz - 5.8e9).abs() < 1.0);
    }

    #[test]
    fn test_channel_presets() {
        let urban = ChannelParams::urban();
        assert!(matches!(urban.fading, FadingType::Rayleigh));
        assert_eq!(urban.multipath, MultipathModel::ETU);

        let rural = ChannelParams::rural_los();
        assert!(matches!(rural.fading, FadingType::Rician { .. }));

        let sat = ChannelParams::satellite();
        assert!(sat.atmospheric.gaseous_absorption);
    }

    #[test]
    fn test_apply_channel_fspl() {
        let params = ChannelParams::new()
            .with_path_loss(PathLossModel::FreeSpace)
            .with_carrier_freq(2.4e9);
        let state = ChannelState::default();
        let mut rng = make_test_rng();

        let output = apply_channel_scalar(20.0, 100.0, &params, &state, &mut rng);

        // FSPL at 100m, 2.4 GHz ≈ 80 dB
        assert!(
            output.path_loss_db > 75.0 && output.path_loss_db < 85.0,
            "FSPL should be ~80 dB at 100m, 2.4 GHz: {}",
            output.path_loss_db
        );

        // RX power ≈ 20 - 80 = -60 dBm
        assert!(
            output.rx_power_dbm > -65.0 && output.rx_power_dbm < -55.0,
            "RX power should be ~-60 dBm: {}",
            output.rx_power_dbm
        );
    }

    #[test]
    fn test_apply_channel_with_fading() {
        let params = ChannelParams::new()
            .with_path_loss(PathLossModel::FreeSpace)
            .with_fading(FadingType::Rayleigh)
            .with_carrier_freq(2.4e9);
        let state = ChannelState::default();

        // Run multiple times to check statistical behavior
        let mut powers = Vec::new();
        for seed in 0..100 {
            let mut rng = {
                let mut s = seed as u32;
                move || {
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    let r1 = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    s = s.wrapping_mul(1103515245).wrapping_add(12345);
                    let r2 = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    (r1, r2)
                }
            };
            let output = apply_channel_scalar(20.0, 100.0, &params, &state, &mut rng);
            powers.push(output.rx_power_dbm);
        }

        let mean: f32 = powers.iter().sum::<f32>() / powers.len() as f32;
        let variance: f32 =
            powers.iter().map(|p| (p - mean).powi(2)).sum::<f32>() / powers.len() as f32;

        // Should have some variance due to fading
        assert!(variance > 0.1, "Should have variance from fading: {}", variance);
    }

    #[test]
    fn test_calculate_sinr() {
        // Signal at -60 dBm, no interference, noise at -100 dBm
        let sinr = calculate_sinr_db(-60.0, &[], -100.0);
        assert!(
            (sinr - 40.0).abs() < 0.1,
            "SINR should be 40 dB: {}",
            sinr
        );

        // Add interferer at same level as signal
        let sinr_with_int = calculate_sinr_db(-60.0, &[-60.0], -100.0);
        assert!(
            sinr_with_int < 5.0,
            "SINR with equal interferer should be ~0 dB: {}",
            sinr_with_int
        );
    }

    #[test]
    fn test_calculate_snr() {
        let snr = calculate_snr_db(-60.0, -100.0);
        assert!((snr - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_link_budget() {
        let budget = calculate_link_budget(
            20.0,  // 20 dBm TX power
            0.0,   // 0 dBi TX antenna
            0.0,   // 0 dBi RX antenna
            10.0,  // 10 dB required SNR
            -100.0, // -100 dBm noise floor
            2.4e9, // 2.4 GHz
        );

        // Max path loss = 20 - (-100 + 10) = 110 dB
        assert!(
            (budget.max_path_loss_db - 110.0).abs() < 1.0,
            "Max path loss should be ~110 dB: {}",
            budget.max_path_loss_db
        );

        // Max range should be reasonable for these parameters
        assert!(
            budget.max_range_m > 1000.0,
            "Max range should be > 1km: {}",
            budget.max_range_m
        );
    }

    #[test]
    fn test_channel_state() {
        let mut state = ChannelState::new().with_velocity(30.0);
        assert!((state.velocity_mps - 30.0).abs() < 0.01);

        state.advance_time(0.001);
        assert!((state.time - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_multipath_model_conversion() {
        assert!(Option::<ITUChannelModel>::from(MultipathModel::None).is_none());
        assert_eq!(
            Option::<ITUChannelModel>::from(MultipathModel::EPA),
            Some(ITUChannelModel::EPA)
        );
        assert_eq!(
            Option::<ITUChannelModel>::from(MultipathModel::EVA),
            Some(ITUChannelModel::EVA)
        );
        assert_eq!(
            Option::<ITUChannelModel>::from(MultipathModel::ETU),
            Some(ITUChannelModel::ETU)
        );
    }

    #[test]
    fn test_atmospheric_integration() {
        // Test at 60 GHz where O₂ absorption is significant (~15 dB/km)
        let params_60ghz = ChannelParams::new()
            .with_path_loss(PathLossModel::FreeSpace)
            .with_gaseous_absorption()
            .with_carrier_freq(60e9); // 60 GHz O₂ resonance

        let state = ChannelState::default();
        let mut rng = make_test_rng();

        let output_60 = apply_channel_scalar(30.0, 1000.0, &params_60ghz, &state, &mut rng);

        // At 60 GHz, O₂ absorption alone is ~15 dB/km
        assert!(
            output_60.atmospheric_db > 10.0,
            "60 GHz O₂ resonance should have >10 dB/km loss: {}",
            output_60.atmospheric_db
        );

        // Test at 28 GHz with rain (5G mmWave n261)
        let params_28ghz = ChannelParams::new()
            .with_path_loss(PathLossModel::FreeSpace)
            .with_gaseous_absorption()
            .with_rain(10.0) // Moderate rain
            .with_carrier_freq(28e9);

        let output_28 = apply_channel_scalar(30.0, 1000.0, &params_28ghz, &state, &mut rng);

        // At 28 GHz with 10 mm/h rain: ~0.1-0.3 dB/km (gaseous) + ~0.1 dB/km (rain) = ~0.2-0.4 dB/km
        assert!(
            output_28.atmospheric_db > 0.1,
            "28 GHz with rain should have measurable loss: {}",
            output_28.atmospheric_db
        );
        assert!(
            output_28.atmospheric_db < 2.0,
            "28 GHz atmospheric should be moderate, not extreme: {}",
            output_28.atmospheric_db
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_channel_output() {
        let mut rng = SimdRng::new(42);

        let output = apply_channel_simd(
            f32x8::splat(20.0),      // TX power
            f32x8::splat(100.0),     // Distance
            f32x8::splat(2.4e9),     // Frequency
            f32x8::splat(0.0),       // Velocity
            f32x8::splat(0.0),       // Time
            PathLossModel::FreeSpace,
            FadingType::Rayleigh,
            &mut rng,
        );

        let rx_powers: [f32; 8] = output.rx_power_dbm.into();
        for &p in &rx_powers {
            // Should be around -60 dBm with fading variation
            assert!(
                p > -100.0 && p < 0.0,
                "RX power should be reasonable: {}",
                p
            );
        }
    }
}
