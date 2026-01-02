//! Ionospheric Propagation Model for HF (3-30 MHz)
//!
//! Implements ionospheric skip propagation physics including:
//! - Ionospheric layer models (D, E, F1, F2)
//! - Critical frequency and MUF/LUF calculations
//! - Multi-hop path loss estimation
//! - Diurnal (day/night) variation
//!
//! # Physical Background
//!
//! HF radio waves can be reflected by the ionosphere, enabling long-distance
//! communication beyond the horizon. The ionosphere consists of layers:
//!
//! - **D layer** (60-90 km): Only present during day, absorbs rather than reflects
//! - **E layer** (90-150 km): Reflects up to ~10 MHz during day
//! - **F1 layer** (150-220 km): Present during day, merges with F2 at night
//! - **F2 layer** (220-400+ km): Primary reflector, highest critical frequency
//!
//! The Maximum Usable Frequency (MUF) depends on:
//! - Critical frequency (foF2) of the F2 layer
//! - Solar activity (F10.7 index)
//! - Time of day (diurnal variation)
//! - Season and geographic location
//!
//! # References
//!
//! - ITU-R P.533: Method for the prediction of HF performance
//! - ITU-R P.1239: Propagation data for HF radio planning

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

#[cfg(feature = "simd")]
use crate::simd_rf::math::{simd_log, simd_sqrt};

use crate::constants::PI;
use crate::types::{
    dimensional::{Hertz, Meters},
    power::PathLoss,
};

// ============================================================================
// Physical Constants
// ============================================================================

/// F2 layer height in meters (typical daytime)
const F2_HEIGHT_DAY_M: f32 = 300_000.0;

/// F2 layer height in meters (typical nighttime)
const F2_HEIGHT_NIGHT_M: f32 = 350_000.0;

/// E layer height in meters
const E_HEIGHT_M: f32 = 110_000.0;

/// D layer height in meters
const D_HEIGHT_M: f32 = 75_000.0;

// ============================================================================
// Ionospheric Layer Model
// ============================================================================

/// Ionospheric layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IonosphericLayer {
    /// D layer (60-90 km) - absorbs, doesn't reflect
    D,
    /// E layer (90-150 km) - sporadic Es can enable VHF propagation
    E,
    /// F1 layer (150-220 km) - daytime only
    F1,
    /// F2 layer (220-400 km) - primary HF reflector
    F2,
}

impl IonosphericLayer {
    /// Nominal height of the layer in meters
    #[inline]
    pub const fn height_m(&self) -> f32 {
        match self {
            IonosphericLayer::D => D_HEIGHT_M,
            IonosphericLayer::E => E_HEIGHT_M,
            IonosphericLayer::F1 => 200_000.0,
            IonosphericLayer::F2 => F2_HEIGHT_DAY_M,
        }
    }

    /// Whether the layer is present at night
    #[inline]
    pub const fn present_at_night(&self) -> bool {
        matches!(self, IonosphericLayer::F2)
    }
}

/// Ionospheric conditions structure
#[derive(Debug, Clone, Copy)]
pub struct IonosphericConditions {
    /// Critical frequency of F2 layer (foF2) in Hz
    pub fo_f2: f32,
    /// Solar flux index (F10.7) - typical 70-250
    pub solar_flux: f32,
    /// Time of day in hours (0-24 UTC)
    pub time_utc: f32,
    /// Latitude in degrees (-90 to 90)
    pub latitude: f32,
    /// Whether it's daytime at the reflection point
    pub is_day: bool,
    /// D-layer absorption coefficient (0-1, higher = more absorption)
    pub d_layer_absorption: f32,
}

impl Default for IonosphericConditions {
    fn default() -> Self {
        Self {
            fo_f2: 8e6,  // 8 MHz typical midlatitude daytime
            solar_flux: 100.0,
            time_utc: 12.0,
            latitude: 45.0,
            is_day: true,
            d_layer_absorption: 0.3,
        }
    }
}

impl IonosphericConditions {
    /// Create daytime conditions with specified critical frequency
    pub fn daytime(fo_f2_hz: f32) -> Self {
        Self {
            fo_f2: fo_f2_hz,
            is_day: true,
            d_layer_absorption: 0.4,
            ..Default::default()
        }
    }

    /// Create nighttime conditions with specified critical frequency
    pub fn nighttime(fo_f2_hz: f32) -> Self {
        Self {
            fo_f2: fo_f2_hz,
            is_day: false,
            d_layer_absorption: 0.0, // No D-layer at night
            time_utc: 0.0,
            ..Default::default()
        }
    }

    /// Estimate F2 layer height based on conditions
    #[inline]
    pub fn f2_height_m(&self) -> f32 {
        if self.is_day {
            F2_HEIGHT_DAY_M
        } else {
            F2_HEIGHT_NIGHT_M
        }
    }
}

// ============================================================================
// MUF/LUF Calculations
// ============================================================================

/// Calculate Maximum Usable Frequency (MUF) for a given distance.
///
/// MUF = foF2 * sec(angle_of_incidence)
///
/// For short distances, the incidence angle is small and MUF ≈ foF2.
/// For longer distances, MUF increases as the signal must hit the layer
/// at a more oblique angle.
///
/// # Arguments
/// * `distance_m` - Ground distance in meters (1 hop)
/// * `conditions` - Ionospheric conditions
///
/// # Returns
/// Maximum usable frequency in Hz
#[inline]
pub fn muf(distance_m: f32, conditions: &IonosphericConditions) -> f32 {
    let h = conditions.f2_height_m();
    let fo_f2 = conditions.fo_f2;

    // Calculate elevation angle using spherical geometry
    // For flat Earth approximation: sec(θ) = sqrt(1 + (d/2h)²)
    let d_over_2h = distance_m / (2.0 * h);
    let sec_theta = (1.0 + d_over_2h * d_over_2h).sqrt();

    // MUF factor typically ranges 1.0 (overhead) to ~3.5 (4000 km)
    let muf_factor = sec_theta.min(4.0);

    fo_f2 * muf_factor
}

/// Calculate Lowest Usable Frequency (LUF) considering D-layer absorption.
///
/// The LUF is the frequency below which D-layer absorption becomes too high.
/// Lower frequencies have higher absorption.
///
/// # Arguments
/// * `distance_m` - Ground distance in meters
/// * `required_snr_db` - Required signal-to-noise ratio
/// * `conditions` - Ionospheric conditions
///
/// # Returns
/// Lowest usable frequency in Hz
#[inline]
pub fn luf(distance_m: f32, required_snr_db: f32, conditions: &IonosphericConditions) -> f32 {
    // LUF increases with distance (more absorption over path)
    // LUF decreases with required SNR (can tolerate more absorption)
    // LUF is 0 at night (no D-layer)

    if !conditions.is_day {
        return 1.5e6; // Minimum practical HF frequency
    }

    // Empirical formula: LUF ≈ k * sqrt(distance) * absorption_factor
    let distance_km = distance_m / 1000.0;
    let absorption = conditions.d_layer_absorption;

    // Base LUF in MHz, then convert to Hz
    let luf_mhz = 2.0 + 0.1 * distance_km.sqrt() * absorption * (1.0 + required_snr_db / 20.0);

    luf_mhz.clamp(1.8, 10.0) * 1e6
}

/// Calculate the skip distance for a given frequency.
///
/// The skip distance is the minimum distance at which sky wave propagation
/// is possible. Signals closer than this distance are in the "skip zone"
/// where neither ground wave nor sky wave is received.
///
/// # Arguments
/// * `freq_hz` - Operating frequency in Hz
/// * `conditions` - Ionospheric conditions
///
/// # Returns
/// Skip distance in meters
#[inline]
pub fn skip_distance(freq_hz: f32, conditions: &IonosphericConditions) -> f32 {
    let fo_f2 = conditions.fo_f2;
    let h = conditions.f2_height_m();

    // Skip distance formula: d = 2h * sqrt((f/foF2)² - 1)
    let f_ratio = freq_hz / fo_f2;

    if f_ratio <= 1.0 {
        return 0.0; // No skip zone for f < foF2
    }

    let term = f_ratio * f_ratio - 1.0;
    2.0 * h * term.sqrt()
}

/// Calculate number of hops needed for a given distance.
///
/// # Arguments
/// * `distance_m` - Total ground distance
/// * `max_hop_distance_m` - Maximum single-hop distance
///
/// # Returns
/// Number of hops (1-5 typically)
#[inline]
pub fn num_hops(distance_m: f32, max_hop_distance_m: f32) -> u8 {
    let hops = (distance_m / max_hop_distance_m).ceil() as u8;
    hops.clamp(1, 5)
}

// ============================================================================
// Ionospheric Path Loss
// ============================================================================

/// Calculate ionospheric propagation path loss.
///
/// This models the complete HF path including:
/// - Free space path loss
/// - D-layer absorption (daytime)
/// - Ionospheric reflection losses
/// - Multi-hop losses
///
/// # Arguments
/// * `distance` - Ground distance
/// * `frequency` - Operating frequency
/// * `conditions` - Ionospheric conditions
///
/// # Returns
/// Total path loss in dB
pub fn ionospheric_path_loss(
    distance: Meters,
    frequency: Hertz,
    conditions: &IonosphericConditions,
) -> PathLoss {
    let d = distance.as_f32();
    let f = frequency.as_hz();
    let h = conditions.f2_height_m();

    // Check if frequency is usable
    let current_muf = muf(d, conditions);
    let current_luf = luf(d, 10.0, conditions);

    if f > current_muf || f < current_luf {
        // Signal cannot propagate - return very high loss
        return PathLoss::new(200.0);
    }

    // Calculate slant path length (geometric)
    // For single hop: slant = 2 * sqrt(h² + (d/2)²)
    let half_d = d / 2.0;
    let slant_single = 2.0 * (h * h + half_d * half_d).sqrt();

    // Determine number of hops
    let max_single_hop = 4000e3; // ~4000 km max single hop
    let n_hops = num_hops(d, max_single_hop);
    let total_slant = slant_single * n_hops as f32;

    // Free space path loss for slant distance
    let fspl_db = 20.0 * total_slant.log10() + 20.0 * f.log10() - 147.55;

    // D-layer absorption (inversely proportional to f²)
    // Higher at lower frequencies, especially below 10 MHz
    let f_mhz = f / 1e6;
    let d_layer_db = if conditions.is_day {
        conditions.d_layer_absorption * 100.0 / (f_mhz * f_mhz)
    } else {
        0.0
    };

    // Reflection loss per hop (typically 1-3 dB)
    let reflection_loss_db = 2.0 * n_hops as f32;

    // Auroral absorption (simplified - higher at high latitudes)
    let auroral_db = if conditions.latitude.abs() > 60.0 {
        5.0
    } else {
        0.0
    };

    let total_loss = fspl_db + d_layer_db + reflection_loss_db + auroral_db;
    PathLoss::new(total_loss.max(0.0))
}

/// Scalar ionospheric path loss for raw f32 inputs
#[inline]
pub fn ionospheric_path_loss_scalar(
    distance_m: f32,
    frequency_hz: f32,
    fo_f2_hz: f32,
    is_day: bool,
) -> f32 {
    let conditions = if is_day {
        IonosphericConditions::daytime(fo_f2_hz)
    } else {
        IonosphericConditions::nighttime(fo_f2_hz)
    };

    ionospheric_path_loss(
        Meters::new(distance_m),
        Hertz::new(frequency_hz),
        &conditions,
    )
    .as_db()
}

// ============================================================================
// SIMD Implementations
// ============================================================================

/// SIMD MUF calculation for 8 environments
#[cfg(feature = "simd")]
#[inline]
pub fn muf_simd(distance_m: f32x8, fo_f2: f32x8, layer_height: f32x8) -> f32x8 {
    let two = f32x8::splat(2.0);
    let d_over_2h = distance_m / (two * layer_height);
    let sec_theta = simd_sqrt(f32x8::splat(1.0) + d_over_2h * d_over_2h);

    // Clamp MUF factor
    let muf_factor = sec_theta.simd_min(f32x8::splat(4.0));
    fo_f2 * muf_factor
}

/// SIMD ionospheric path loss calculation
#[cfg(feature = "simd")]
#[inline]
pub fn ionospheric_path_loss_simd(
    distance_m: f32x8,
    frequency_hz: f32x8,
    _fo_f2: f32x8,
    layer_height: f32x8,
    d_layer_absorption: f32x8,
    is_day_mask: [bool; 8],
) -> f32x8 {
    // Calculate slant path (single hop approximation)
    let half_d = distance_m * f32x8::splat(0.5);
    let slant = f32x8::splat(2.0) * simd_sqrt(layer_height * layer_height + half_d * half_d);

    // FSPL for slant distance
    let log10_e = f32x8::splat(std::f32::consts::LOG10_E);
    let fspl_db = f32x8::splat(20.0) * simd_log(slant) * log10_e
        + f32x8::splat(20.0) * simd_log(frequency_hz) * log10_e
        - f32x8::splat(147.55);

    // D-layer absorption
    let f_mhz = frequency_hz * f32x8::splat(1e-6);
    let d_layer_base = d_layer_absorption * f32x8::splat(100.0) / (f_mhz * f_mhz);

    // Apply day/night mask for D-layer
    let mut d_layer_arr = [0.0f32; 8];
    let d_layer_base_arr: [f32; 8] = d_layer_base.into();
    for i in 0..8 {
        d_layer_arr[i] = if is_day_mask[i] { d_layer_base_arr[i] } else { 0.0 };
    }
    let d_layer_db = f32x8::from_array(d_layer_arr);

    // Reflection loss (assume single hop)
    let reflection_db = f32x8::splat(2.0);

    let total = fspl_db + d_layer_db + reflection_db;
    total.simd_max(f32x8::splat(0.0))
}

// ============================================================================
// Diurnal Variation Model
// ============================================================================

/// Model for day/night ionospheric variation.
///
/// The ionosphere changes significantly between day and night:
/// - foF2 is higher during day (solar ionization)
/// - D-layer disappears at night (no absorption)
/// - F1 merges with F2 at night
/// - Propagation conditions can improve or degrade
#[derive(Debug, Clone, Copy)]
pub struct DiurnalModel {
    /// Latitude in degrees
    pub latitude: f32,
    /// Longitude in degrees
    pub longitude: f32,
    /// Day of year (1-365)
    pub day_of_year: u16,
    /// Solar flux index (F10.7)
    pub solar_flux: f32,
}

impl DiurnalModel {
    /// Create a new diurnal model
    pub fn new(latitude: f32, longitude: f32, day_of_year: u16, solar_flux: f32) -> Self {
        Self {
            latitude: latitude.clamp(-90.0, 90.0),
            longitude: longitude.clamp(-180.0, 180.0),
            day_of_year: day_of_year.clamp(1, 365),
            solar_flux: solar_flux.clamp(60.0, 300.0),
        }
    }

    /// Calculate local hour from UTC
    #[inline]
    pub fn local_hour(&self, utc_hour: f32) -> f32 {
        let offset = self.longitude / 15.0; // 15 degrees per hour
        let local = utc_hour + offset;
        if local < 0.0 {
            local + 24.0
        } else if local >= 24.0 {
            local - 24.0
        } else {
            local
        }
    }

    /// Determine if it's daytime at this location
    #[inline]
    pub fn is_daytime(&self, utc_hour: f32) -> bool {
        let local = self.local_hour(utc_hour);
        // Simple sunrise/sunset (6 AM - 6 PM)
        // TODO: Proper solar zenith angle calculation
        local >= 6.0 && local < 18.0
    }

    /// Calculate critical frequency (foF2) for given UTC time
    ///
    /// foF2 follows a diurnal pattern:
    /// - Maximum around local noon
    /// - Minimum around local midnight
    /// - Influenced by solar activity
    ///
    /// Typical values:
    /// - Solar minimum (F10.7 ~ 70): 4-8 MHz daytime
    /// - Solar maximum (F10.7 ~ 200): 8-15 MHz daytime
    /// - Night values are roughly half of daytime values
    #[inline]
    pub fn critical_frequency(&self, utc_hour: f32) -> f32 {
        let local = self.local_hour(utc_hour);

        // Base critical frequency from solar flux
        // Empirical: foF2 ≈ 4 + 0.04 * F10.7 for midlatitude daytime
        // At F10.7=70: 6.8 MHz, at F10.7=200: 12 MHz
        let fo_base_mhz = 4.0 + 0.04 * self.solar_flux;

        // Diurnal variation (sinusoidal model)
        // Peak at local noon (12:00), minimum at midnight (0:00)
        let phase = (local - 12.0) * PI / 12.0;
        let variation = phase.cos();

        // foF2 ranges from base * 0.5 (night) to base * 1.0 (day)
        // Daytime: variation = 1, factor = 0.75 + 0.25 = 1.0
        // Nighttime: variation = -1, factor = 0.75 - 0.25 = 0.5
        let diurnal_factor = 0.75 + 0.25 * variation;
        let fo_f2_mhz = fo_base_mhz * diurnal_factor;

        // Latitude effect (higher at mid-latitudes ~40-50 degrees)
        let lat_effect = 1.0 - 0.1 * ((self.latitude.abs() - 45.0) / 45.0).abs();

        (fo_f2_mhz * lat_effect * 1e6).clamp(2e6, 15e6)
    }

    /// Calculate D-layer absorption coefficient
    ///
    /// D-layer only exists during daytime and is strongest at local noon.
    #[inline]
    pub fn d_layer_absorption(&self, utc_hour: f32) -> f32 {
        if !self.is_daytime(utc_hour) {
            return 0.0;
        }

        let local = self.local_hour(utc_hour);

        // Peak absorption at noon
        let phase = (local - 12.0) * PI / 6.0;
        let base = 0.5 * (1.0 + phase.cos());

        // Solar flux influence
        let flux_factor = (self.solar_flux / 100.0).sqrt();

        (base * flux_factor).clamp(0.0, 1.0)
    }

    /// Get complete ionospheric conditions for a given time
    pub fn conditions(&self, utc_hour: f32) -> IonosphericConditions {
        IonosphericConditions {
            fo_f2: self.critical_frequency(utc_hour),
            solar_flux: self.solar_flux,
            time_utc: utc_hour,
            latitude: self.latitude,
            is_day: self.is_daytime(utc_hour),
            d_layer_absorption: self.d_layer_absorption(utc_hour),
        }
    }
}

impl Default for DiurnalModel {
    fn default() -> Self {
        Self {
            latitude: 45.0,  // Mid-latitude
            longitude: 0.0,  // UTC
            day_of_year: 172, // Summer solstice
            solar_flux: 100.0,
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 0.1;

    #[test]
    fn test_muf_increases_with_distance() {
        let conditions = IonosphericConditions::default();

        let muf_500km = muf(500e3, &conditions);
        let muf_2000km = muf(2000e3, &conditions);
        let muf_4000km = muf(4000e3, &conditions);

        assert!(
            muf_2000km > muf_500km,
            "MUF should increase with distance: {} vs {}",
            muf_500km,
            muf_2000km
        );
        assert!(
            muf_4000km > muf_2000km,
            "MUF should increase with distance: {} vs {}",
            muf_2000km,
            muf_4000km
        );
    }

    #[test]
    fn test_muf_equals_fof2_at_zero_distance() {
        let conditions = IonosphericConditions::daytime(10e6);
        let muf_0 = muf(0.0, &conditions);

        assert!(
            (muf_0 - 10e6).abs() < 1e3,
            "MUF at 0 distance should equal foF2: {} vs 10 MHz",
            muf_0 / 1e6
        );
    }

    #[test]
    fn test_luf_zero_at_night() {
        let conditions = IonosphericConditions::nighttime(8e6);
        let luf_night = luf(2000e3, 10.0, &conditions);

        assert!(
            luf_night < 2e6,
            "LUF at night should be low: {} MHz",
            luf_night / 1e6
        );
    }

    #[test]
    fn test_luf_increases_with_distance_day() {
        let conditions = IonosphericConditions::daytime(8e6);

        let luf_500km = luf(500e3, 10.0, &conditions);
        let luf_2000km = luf(2000e3, 10.0, &conditions);

        assert!(
            luf_2000km > luf_500km,
            "LUF should increase with distance: {} vs {}",
            luf_500km / 1e6,
            luf_2000km / 1e6
        );
    }

    #[test]
    fn test_skip_distance_zero_below_fof2() {
        let conditions = IonosphericConditions::daytime(10e6);

        // Frequency below foF2 should have no skip zone
        let skip = skip_distance(8e6, &conditions);
        assert!(
            skip == 0.0,
            "Skip distance should be 0 for f < foF2: {}",
            skip
        );
    }

    #[test]
    fn test_skip_distance_increases_with_frequency() {
        let conditions = IonosphericConditions::daytime(10e6);

        let skip_12mhz = skip_distance(12e6, &conditions);
        let skip_20mhz = skip_distance(20e6, &conditions);

        assert!(
            skip_20mhz > skip_12mhz,
            "Skip distance should increase with frequency: {} vs {}",
            skip_12mhz / 1e3,
            skip_20mhz / 1e3
        );
    }

    #[test]
    fn test_num_hops() {
        let max_hop = 4000e3;

        assert_eq!(num_hops(1000e3, max_hop), 1);
        assert_eq!(num_hops(4000e3, max_hop), 1);
        assert_eq!(num_hops(4001e3, max_hop), 2);
        assert_eq!(num_hops(8000e3, max_hop), 2);
        assert_eq!(num_hops(12000e3, max_hop), 3);
    }

    #[test]
    fn test_ionospheric_path_loss_reasonable_range() {
        let conditions = IonosphericConditions::daytime(10e6);

        // 1000 km at 14 MHz (within MUF)
        let loss = ionospheric_path_loss(
            Meters::new(1000e3),
            Hertz::new(14e6),
            &conditions,
        );

        // Should be in reasonable range (80-140 dB for HF)
        assert!(
            loss.as_db() > 80.0 && loss.as_db() < 150.0,
            "Path loss should be in range: {} dB",
            loss.as_db()
        );
    }

    #[test]
    fn test_ionospheric_path_loss_high_above_muf() {
        let conditions = IonosphericConditions::daytime(8e6);
        let muf_1000km = muf(1000e3, &conditions);

        // Frequency well above MUF
        let loss = ionospheric_path_loss(
            Meters::new(1000e3),
            Hertz::new(muf_1000km + 10e6),
            &conditions,
        );

        assert!(
            loss.as_db() >= 200.0,
            "Loss above MUF should be very high: {}",
            loss.as_db()
        );
    }

    #[test]
    fn test_diurnal_daytime() {
        let model = DiurnalModel::new(45.0, 0.0, 172, 100.0);

        assert!(model.is_daytime(12.0)); // Noon UTC = noon local
        assert!(!model.is_daytime(0.0)); // Midnight UTC = midnight local
    }

    #[test]
    fn test_diurnal_timezone_offset() {
        // New York: -75 degrees longitude
        let model = DiurnalModel::new(40.0, -75.0, 172, 100.0);

        // 17:00 UTC = 12:00 local (noon)
        assert!(model.is_daytime(17.0));
        // 5:00 UTC = 0:00 local (midnight)
        assert!(!model.is_daytime(5.0));
    }

    #[test]
    fn test_diurnal_critical_frequency_variation() {
        let model = DiurnalModel::new(45.0, 0.0, 172, 100.0);

        let fo_noon = model.critical_frequency(12.0);
        let fo_midnight = model.critical_frequency(0.0);

        assert!(
            fo_noon > fo_midnight,
            "foF2 should be higher at noon: {} vs {}",
            fo_noon / 1e6,
            fo_midnight / 1e6
        );

        // Noon should be roughly 2x midnight
        let ratio = fo_noon / fo_midnight;
        assert!(
            ratio > 1.5 && ratio < 3.0,
            "Noon/midnight ratio: {}",
            ratio
        );
    }

    #[test]
    fn test_diurnal_d_layer_absorption() {
        let model = DiurnalModel::new(45.0, 0.0, 172, 100.0);

        // D-layer only present during day
        assert_eq!(model.d_layer_absorption(0.0), 0.0); // Midnight
        assert!(model.d_layer_absorption(12.0) > 0.0); // Noon

        // Peak at noon
        let abs_10 = model.d_layer_absorption(10.0);
        let abs_12 = model.d_layer_absorption(12.0);
        let abs_14 = model.d_layer_absorption(14.0);

        assert!(abs_12 > abs_10);
        assert!(abs_12 > abs_14);
    }

    #[test]
    fn test_diurnal_conditions() {
        let model = DiurnalModel::new(45.0, 0.0, 172, 150.0);

        let cond_day = model.conditions(12.0);
        let cond_night = model.conditions(0.0);

        assert!(cond_day.is_day);
        assert!(!cond_night.is_day);

        assert!(cond_day.fo_f2 > cond_night.fo_f2);
        assert!(cond_day.d_layer_absorption > cond_night.d_layer_absorption);
    }

    #[test]
    fn test_solar_flux_effect() {
        let model_low = DiurnalModel::new(45.0, 0.0, 172, 70.0);
        let model_high = DiurnalModel::new(45.0, 0.0, 172, 200.0);

        let fo_low = model_low.critical_frequency(12.0);
        let fo_high = model_high.critical_frequency(12.0);

        assert!(
            fo_high > fo_low,
            "Higher solar flux should increase foF2: {} vs {}",
            fo_low / 1e6,
            fo_high / 1e6
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_muf_simd() {
        let distances = f32x8::from_array([
            500e3, 1000e3, 1500e3, 2000e3, 2500e3, 3000e3, 3500e3, 4000e3,
        ]);
        let fo_f2 = f32x8::splat(10e6);
        let height = f32x8::splat(300e3);

        let muf_result = muf_simd(distances, fo_f2, height);
        let muf_arr: [f32; 8] = muf_result.into();

        // Verify increasing MUF with distance
        for i in 1..8 {
            assert!(
                muf_arr[i] >= muf_arr[i - 1],
                "MUF should increase: [{}] {} vs [{}] {}",
                i - 1,
                muf_arr[i - 1],
                i,
                muf_arr[i]
            );
        }

        // Verify against scalar
        for i in 0..8 {
            let d_arr: [f32; 8] = distances.into();
            let conditions = IonosphericConditions {
                fo_f2: 10e6,
                ..Default::default()
            };
            let scalar_muf = muf(d_arr[i], &conditions);
            let rel_err = ((muf_arr[i] - scalar_muf) / scalar_muf).abs();
            assert!(
                rel_err < 0.01,
                "SIMD/scalar mismatch at {}: {} vs {}",
                i,
                muf_arr[i],
                scalar_muf
            );
        }
    }
}
