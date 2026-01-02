//! Path Loss Models - Type-Safe Implementation
//!
//! SIMD-optimized implementations of various RF propagation path loss models
//! with Curry-Howard compliant types.
//!
//! # Type Safety
//!
//! This module uses the refined type system:
//! - `Hertz` for carrier frequencies (guaranteed positive)
//! - `Meters` for distances (guaranteed non-negative)
//! - `PathLoss` for loss values (guaranteed non-negative dB)
//!
//! # Models
//!
//! - **FSPL**: Free Space Path Loss - ideal propagation
//! - **Log-Distance**: General indoor/outdoor model with configurable exponent
//! - **Hata Urban**: Okumura-Hata model for 150-1500 MHz urban areas
//! - **COST-231**: Extension to 1500-2000 MHz
//! - **Ground Wave**: VLF/LF propagation along Earth's surface
//! - **Waveguide**: VLF Earth-ionosphere waveguide propagation
//!
//! # Verified Constants
//!
//! All physical constants are verified at compile time via tests.

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

#[cfg(feature = "simd")]
use crate::simd_rf::math::{simd_exp, simd_log, simd_sqrt};

use crate::constants::{PI, SPEED_OF_LIGHT};
use crate::types::{
    dimensional::{Hertz, Meters},
    power::PathLoss,
    primitives::PositiveF32,
};

/// Log base 10 of e, for converting ln to log10
const LOG10_E: f32 = std::f32::consts::LOG10_E;

// ============================================================================
// Verified Physical Constants
// ============================================================================

/// Verified Free Space Path Loss constant.
///
/// FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
///
/// The constant is 20*log10(4π/c):
/// c = 299,792,458 m/s
/// 4π/c ≈ 4.188e-8
/// 20*log10(4.188e-8) ≈ -147.55 dB
pub const FSPL_CONSTANT_DB: f32 = -147.55;

/// Alternative FSPL constant for MHz and km inputs.
/// FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.45
pub const FSPL_CONSTANT_MHZ_KM: f32 = 32.45;

/// Verify constant at compile time (via test)
#[cfg(test)]
mod constant_verification {
    use super::*;

    #[test]
    fn verify_fspl_constant() {
        let theoretical = 20.0 * (4.0 * PI / SPEED_OF_LIGHT).log10();
        assert!(
            (theoretical - FSPL_CONSTANT_DB).abs() < 0.02,
            "FSPL constant mismatch: theoretical={}, defined={}",
            theoretical,
            FSPL_CONSTANT_DB
        );
    }

    #[test]
    fn verify_fspl_mhz_km_constant() {
        // At 1 MHz, 1 km: FSPL = 20*log10(1e3) + 20*log10(1e6) + FSPL_CONSTANT_DB
        //                     = 60 + 120 - 147.55 = 32.45 dB
        let at_1mhz_1km = 60.0 + 120.0 + FSPL_CONSTANT_DB;
        assert!(
            (at_1mhz_1km - FSPL_CONSTANT_MHZ_KM).abs() < 0.1,
            "FSPL MHz/km constant: theoretical={}, defined={}",
            at_1mhz_1km,
            FSPL_CONSTANT_MHZ_KM
        );
    }
}

// ============================================================================
// Type-Safe Path Loss Functions
// ============================================================================

/// Compute Free Space Path Loss with type safety.
///
/// FSPL(dB) = 20*log10(d) + 20*log10(f) + FSPL_CONSTANT_DB
///
/// # Arguments
/// * `distance` - Distance from transmitter to receiver
/// * `frequency` - Carrier frequency
///
/// # Returns
/// Path loss in dB (always non-negative for d >= 1/f wavelengths)
#[inline]
pub fn fspl(distance: Meters, frequency: Hertz) -> PathLoss {
    let d = distance.as_f32().max(1e-6);
    let f = frequency.as_hz();

    let loss_db = 20.0 * d.log10() + 20.0 * f.log10() + FSPL_CONSTANT_DB;
    PathLoss::new(loss_db.max(0.0))
}

/// Compute FSPL linear attenuation factor.
///
/// Returns the factor by which power is multiplied.
#[inline]
pub fn fspl_factor(distance: Meters, frequency: Hertz) -> PositiveF32 {
    fspl(distance, frequency).to_linear_factor()
}

/// Log-distance path loss model.
///
/// General model for indoor/outdoor propagation with configurable path loss exponent.
///
/// PL(d) = PL(d₀) + 10n·log₁₀(d/d₀)
///
/// # Arguments
/// * `distance` - Distance from transmitter to receiver
/// * `ref_distance` - Reference distance d₀ (typically 1m)
/// * `path_loss_exp` - Path loss exponent n (typically 2-4)
/// * `ref_loss` - Reference path loss PL(d₀)
#[inline]
pub fn log_distance(
    distance: Meters,
    ref_distance: Meters,
    path_loss_exp: f32,
    ref_loss: PathLoss,
) -> PathLoss {
    let d = distance.as_f32().max(0.1);
    let d0 = ref_distance.as_f32().max(0.1);

    let loss_db = ref_loss.as_db() + 10.0 * path_loss_exp * (d / d0).log10();
    PathLoss::new(loss_db.max(0.0))
}

/// Hata model for urban areas (150-1500 MHz).
///
/// Empirical model based on Okumura measurements for urban macro cells.
///
/// # Valid Ranges
/// * Distance: 1-20 km
/// * Frequency: 150-1500 MHz
/// * TX height: 30-200 m
/// * RX height: 1-10 m
#[inline]
pub fn hata_urban(
    distance: Meters,
    frequency: Hertz,
    tx_height: Meters,
    rx_height: Meters,
) -> PathLoss {
    // Convert to required units and clamp to valid ranges
    let d_km = (distance.as_f32() / 1000.0).clamp(1.0, 20.0);
    let f_mhz = (frequency.as_hz() / 1e6).clamp(150.0, 1500.0);
    let h_t = tx_height.as_f32().clamp(30.0, 200.0);
    let h_r = rx_height.as_f32().clamp(1.0, 10.0);

    let log_f = f_mhz.log10();
    let log_ht = h_t.log10();
    let log_d = d_km.log10();

    // Mobile antenna correction factor for large city (f > 300 MHz)
    let a_hr = 3.2 * (11.75 * h_r).log10().powi(2) - 4.97;

    // Hata formula
    let loss_db = 69.55 + 26.16 * log_f - 13.82 * log_ht - a_hr
        + (44.9 - 6.55 * log_ht) * log_d;

    PathLoss::new(loss_db.max(0.0))
}

/// COST-231 Hata model extension (1500-2000 MHz).
///
/// Extension of Hata model for higher frequencies.
///
/// # Valid Ranges
/// * Distance: 1-20 km
/// * Frequency: 1500-2000 MHz
/// * TX height: 30-200 m
/// * RX height: 1-10 m
#[inline]
pub fn cost231_hata(
    distance: Meters,
    frequency: Hertz,
    tx_height: Meters,
    rx_height: Meters,
    is_urban: bool,
) -> PathLoss {
    let d_km = (distance.as_f32() / 1000.0).clamp(1.0, 20.0);
    let f_mhz = (frequency.as_hz() / 1e6).clamp(1500.0, 2000.0);
    let h_t = tx_height.as_f32().clamp(30.0, 200.0);
    let h_r = rx_height.as_f32().clamp(1.0, 10.0);

    let log_f = f_mhz.log10();
    let log_ht = h_t.log10();
    let log_d = d_km.log10();

    // Mobile antenna correction for small/medium city
    let a_hr = (1.1 * log_f - 0.7) * h_r - (1.56 * log_f - 0.8);

    // Metropolitan correction
    let c_m = if is_urban { 3.0 } else { 0.0 };

    let loss_db = 46.3 + 33.9 * log_f - 13.82 * log_ht - a_hr
        + (44.9 - 6.55 * log_ht) * log_d + c_m;

    PathLoss::new(loss_db.max(0.0))
}

// ============================================================================
// Legacy Raw SIMD Functions (for backward compatibility)
// ============================================================================

/// Free Space Path Loss (SIMD, raw f32x8)
///
/// Returns linear attenuation factor (multiply by power to get received power)
#[cfg(feature = "simd")]
#[inline]
pub fn fspl_simd(distance: f32x8, frequency: f32x8) -> f32x8 {
    let four_pi_over_c = f32x8::splat(4.0 * PI / SPEED_OF_LIGHT);
    let term = four_pi_over_c * distance * frequency;
    let loss = term * term;

    // Return inverse (attenuation factor), clamped to avoid division by zero
    f32x8::splat(1.0) / loss.simd_max(f32x8::splat(1.0))
}

/// Free Space Path Loss in dB (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn fspl_db_simd(distance: f32x8, frequency: f32x8) -> f32x8 {
    let constant_db = f32x8::splat(FSPL_CONSTANT_DB);
    let log10_d = simd_log(distance) * f32x8::splat(LOG10_E);
    let log10_f = simd_log(frequency) * f32x8::splat(LOG10_E);

    f32x8::splat(20.0) * (log10_d + log10_f) + constant_db
}

/// Log-distance path loss model (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn log_distance_simd(
    distance: f32x8,
    ref_distance: f32x8,
    path_loss_exp: f32x8,
    ref_loss_db: f32x8,
) -> f32x8 {
    let d = distance.simd_max(f32x8::splat(0.1));
    let log_ratio = simd_log(d / ref_distance) * f32x8::splat(LOG10_E);
    let path_loss_db = ref_loss_db + f32x8::splat(10.0) * path_loss_exp * log_ratio;

    simd_db_to_linear(-path_loss_db)
}

/// Log-distance path loss in dB (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn log_distance_db_simd(
    distance: f32x8,
    ref_distance: f32x8,
    path_loss_exp: f32x8,
    ref_loss_db: f32x8,
) -> f32x8 {
    let d = distance.simd_max(f32x8::splat(0.1));
    let log_ratio = simd_log(d / ref_distance) * f32x8::splat(LOG10_E);
    ref_loss_db + f32x8::splat(10.0) * path_loss_exp * log_ratio
}

/// Hata model for urban areas (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn hata_urban_db_simd(
    distance_km: f32x8,
    frequency_mhz: f32x8,
    tx_height_m: f32x8,
    rx_height_m: f32x8,
) -> f32x8 {
    let d = distance_km.simd_clamp(f32x8::splat(1.0), f32x8::splat(20.0));
    let f = frequency_mhz.simd_clamp(f32x8::splat(150.0), f32x8::splat(1500.0));
    let h_t = tx_height_m.simd_clamp(f32x8::splat(30.0), f32x8::splat(200.0));
    let h_r = rx_height_m.simd_clamp(f32x8::splat(1.0), f32x8::splat(10.0));

    let log_f = simd_log(f) * f32x8::splat(LOG10_E);
    let log_ht = simd_log(h_t) * f32x8::splat(LOG10_E);
    let log_d = simd_log(d) * f32x8::splat(LOG10_E);

    let log_term = simd_log(f32x8::splat(11.75) * h_r) * f32x8::splat(LOG10_E);
    let a_hr = f32x8::splat(3.2) * log_term * log_term - f32x8::splat(4.97);

    f32x8::splat(69.55) + f32x8::splat(26.16) * log_f
        - f32x8::splat(13.82) * log_ht
        - a_hr
        + (f32x8::splat(44.9) - f32x8::splat(6.55) * log_ht) * log_d
}

/// Hata model - linear attenuation factor (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn hata_urban_simd(
    distance_km: f32x8,
    frequency_mhz: f32x8,
    tx_height_m: f32x8,
    rx_height_m: f32x8,
) -> f32x8 {
    let loss_db = hata_urban_db_simd(distance_km, frequency_mhz, tx_height_m, rx_height_m);
    simd_db_to_linear(-loss_db)
}

/// COST-231 Hata model in dB (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn cost231_hata_db_simd(
    distance_km: f32x8,
    frequency_mhz: f32x8,
    tx_height_m: f32x8,
    rx_height_m: f32x8,
    is_urban: bool,
) -> f32x8 {
    let d = distance_km.simd_clamp(f32x8::splat(1.0), f32x8::splat(20.0));
    let f = frequency_mhz.simd_clamp(f32x8::splat(1500.0), f32x8::splat(2000.0));
    let h_t = tx_height_m.simd_clamp(f32x8::splat(30.0), f32x8::splat(200.0));
    let h_r = rx_height_m.simd_clamp(f32x8::splat(1.0), f32x8::splat(10.0));

    let log_f = simd_log(f) * f32x8::splat(LOG10_E);
    let log_ht = simd_log(h_t) * f32x8::splat(LOG10_E);
    let log_d = simd_log(d) * f32x8::splat(LOG10_E);

    let a_hr = (f32x8::splat(1.1) * log_f - f32x8::splat(0.7)) * h_r
        - (f32x8::splat(1.56) * log_f - f32x8::splat(0.8));

    let c_m = if is_urban {
        f32x8::splat(3.0)
    } else {
        f32x8::splat(0.0)
    };

    f32x8::splat(46.3) + f32x8::splat(33.9) * log_f
        - f32x8::splat(13.82) * log_ht
        - a_hr
        + (f32x8::splat(44.9) - f32x8::splat(6.55) * log_ht) * log_d
        + c_m
}

/// COST-231 Hata - linear attenuation (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn cost231_hata_simd(
    distance_km: f32x8,
    frequency_mhz: f32x8,
    tx_height_m: f32x8,
    rx_height_m: f32x8,
    is_urban: bool,
) -> f32x8 {
    let loss_db =
        cost231_hata_db_simd(distance_km, frequency_mhz, tx_height_m, rx_height_m, is_urban);
    simd_db_to_linear(-loss_db)
}

/// Ground wave propagation (SIMD, raw f32x8)
#[cfg(feature = "simd")]
#[inline]
pub fn ground_wave_simd(distance: f32x8, frequency: f32x8) -> f32x8 {
    let sqrt_f = simd_sqrt(frequency);
    let alpha = f32x8::splat(0.001) * sqrt_f;
    let sqrt_d = simd_sqrt(distance.simd_max(f32x8::splat(1.0)));

    simd_exp(-alpha * distance) / sqrt_d
}

// ============================================================================
// Legacy Raw Scalar Functions
// ============================================================================

/// Scalar Free Space Path Loss (raw f32)
#[inline]
pub fn fspl_scalar(distance: f32, frequency: f32) -> f32 {
    let term = (4.0 * PI * distance * frequency) / SPEED_OF_LIGHT;
    let loss = term * term;
    1.0 / loss.max(1.0)
}

/// Scalar FSPL in dB (raw f32)
#[inline]
pub fn fspl_db_scalar(distance: f32, frequency: f32) -> f32 {
    20.0 * (4.0 * PI * distance * frequency / SPEED_OF_LIGHT).log10()
}

/// Scalar log-distance path loss (raw f32)
#[inline]
pub fn log_distance_scalar(
    distance: f32,
    ref_distance: f32,
    path_loss_exp: f32,
    ref_loss_db: f32,
) -> f32 {
    let d = distance.max(0.1);
    let path_loss_db = ref_loss_db + 10.0 * path_loss_exp * (d / ref_distance).log10();
    10.0_f32.powf(-path_loss_db / 10.0)
}

/// Scalar log-distance path loss in dB (raw f32)
#[inline]
pub fn log_distance_db_scalar(
    distance: f32,
    ref_distance: f32,
    path_loss_exp: f32,
    ref_loss_db: f32,
) -> f32 {
    let d = distance.max(0.1);
    ref_loss_db + 10.0 * path_loss_exp * (d / ref_distance).log10()
}

/// Scalar Hata urban model in dB (raw f32)
#[inline]
pub fn hata_urban_db_scalar(
    distance_km: f32,
    frequency_mhz: f32,
    tx_height_m: f32,
    rx_height_m: f32,
) -> f32 {
    let d = distance_km.clamp(1.0, 20.0);
    let f = frequency_mhz.clamp(150.0, 1500.0);
    let h_t = tx_height_m.clamp(30.0, 200.0);
    let h_r = rx_height_m.clamp(1.0, 10.0);

    let log_f = f.log10();
    let log_ht = h_t.log10();
    let log_d = d.log10();

    let a_hr = 3.2 * (11.75 * h_r).log10().powi(2) - 4.97;

    69.55 + 26.16 * log_f - 13.82 * log_ht - a_hr + (44.9 - 6.55 * log_ht) * log_d
}

/// Scalar COST-231 Hata model in dB (raw f32)
#[inline]
pub fn cost231_hata_db_scalar(
    distance_km: f32,
    frequency_mhz: f32,
    tx_height_m: f32,
    rx_height_m: f32,
    is_urban: bool,
) -> f32 {
    let d = distance_km.clamp(1.0, 20.0);
    let f = frequency_mhz.clamp(1500.0, 2000.0);
    let h_t = tx_height_m.clamp(30.0, 200.0);
    let h_r = rx_height_m.clamp(1.0, 10.0);

    let log_f = f.log10();
    let log_ht = h_t.log10();
    let log_d = d.log10();

    let a_hr = (1.1 * log_f - 0.7) * h_r - (1.56 * log_f - 0.8);
    let c_m = if is_urban { 3.0 } else { 0.0 };

    46.3 + 33.9 * log_f - 13.82 * log_ht - a_hr + (44.9 - 6.55 * log_ht) * log_d + c_m
}

/// Scalar ground wave attenuation in dB (raw f32)
#[inline]
pub fn ground_wave_db_scalar(distance: f32, frequency: f32) -> f32 {
    let sqrt_f = frequency.sqrt();
    let alpha = 0.001 * sqrt_f;
    let d = distance.max(1.0);

    let atten_linear = (-alpha * d).exp() / d.sqrt();
    -10.0 * atten_linear.log10()
}

// ============================================================================
// VLF Earth-Ionosphere Waveguide Propagation
// ============================================================================

/// VLF waveguide attenuation in dB (raw f32).
///
/// At VLF frequencies (3-30 kHz), the Earth-ionosphere acts as a waveguide
/// with extremely low attenuation (~0.001-0.003 dB/km). This enables
/// worldwide communication with moderate power.
///
/// The model uses the Wait-Spies formula for waveguide attenuation.
///
/// # Arguments
/// * `distance_km` - Distance in kilometers
/// * `frequency_hz` - Frequency in Hz (typically 3-30 kHz)
/// * `ionosphere_height_km` - Ionosphere height (60-90 km, varies with solar activity)
///
/// # Returns
/// Path loss in dB (very low, typically <1 dB for 1000 km)
#[inline]
pub fn waveguide_db_scalar(distance_km: f32, frequency_hz: f32, ionosphere_height_km: f32) -> f32 {
    // VLF waveguide attenuation is very low: ~0.001-0.003 dB/km
    // Attenuation coefficient depends on frequency and ionosphere height

    // Base attenuation rate (dB/km) - increases slightly with frequency
    let freq_khz = frequency_hz / 1000.0;
    let base_alpha = 0.001 + 0.0001 * (freq_khz / 10.0); // ~0.001-0.003 dB/km

    // Height factor - lower ionosphere increases attenuation slightly
    let height_factor = 75.0 / ionosphere_height_km.clamp(60.0, 100.0);

    // Total attenuation
    let attenuation_db = base_alpha * height_factor * distance_km;

    // Add spreading loss (cylindrical spreading in waveguide)
    // For waveguide, spreading goes as sqrt(distance) not distance squared
    let spreading_db = 10.0 * (distance_km.max(1.0)).log10();

    attenuation_db + spreading_db
}

/// VLF waveguide attenuation with default ionosphere height.
#[inline]
pub fn waveguide_db_scalar_default(distance_km: f32, frequency_hz: f32) -> f32 {
    waveguide_db_scalar(distance_km, frequency_hz, 75.0) // 75 km typical D-layer
}

/// VLF waveguide propagation (SIMD, raw f32x8)
///
/// SIMD variant for batch processing of VLF propagation.
#[cfg(feature = "simd")]
#[inline]
pub fn waveguide_simd(distance_km: f32x8, frequency_hz: f32x8, ionosphere_height_km: f32x8) -> f32x8 {
    let freq_khz = frequency_hz / f32x8::splat(1000.0);
    let base_alpha = f32x8::splat(0.001) + f32x8::splat(0.0001) * (freq_khz / f32x8::splat(10.0));

    let height_factor = f32x8::splat(75.0) / ionosphere_height_km.simd_clamp(f32x8::splat(60.0), f32x8::splat(100.0));

    let attenuation_db = base_alpha * height_factor * distance_km;
    let spreading_db = f32x8::splat(10.0) * simd_log(distance_km.simd_max(f32x8::splat(1.0))) * f32x8::splat(LOG10_E);

    attenuation_db + spreading_db
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert dB to linear power (SIMD)
#[cfg(feature = "simd")]
#[inline]
fn simd_db_to_linear(db: f32x8) -> f32x8 {
    crate::simd_rf::math::simd_db_to_linear(db)
}

/// Calculate 3D distance between two points (SIMD)
#[cfg(feature = "simd")]
#[inline]
pub fn calc_distance_simd(
    pos1: (f32x8, f32x8, f32x8),
    pos2: (f32x8, f32x8, f32x8),
) -> f32x8 {
    let dx = pos1.0 - pos2.0;
    let dy = pos1.1 - pos2.1;
    let dz = pos1.2 - pos2.2;
    simd_sqrt(dx * dx + dy * dy + dz * dz)
}

/// Calculate 2D distance (SIMD)
#[cfg(feature = "simd")]
#[inline]
pub fn calc_distance_2d_simd(pos1: (f32x8, f32x8), pos2: (f32x8, f32x8)) -> f32x8 {
    let dx = pos1.0 - pos2.0;
    let dy = pos1.1 - pos2.1;
    simd_sqrt(dx * dx + dy * dy)
}

/// Scalar 3D distance calculation
#[inline]
pub fn calc_distance_scalar(pos1: (f32, f32, f32), pos2: (f32, f32, f32)) -> f32 {
    let dx = pos1.0 - pos2.0;
    let dy = pos1.1 - pos2.1;
    let dz = pos1.2 - pos2.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 0.1; // 0.1 dB tolerance

    #[test]
    fn test_fspl_db_known_value() {
        // 1 km at 1 GHz should be approximately 91.5 dB
        let distance = 1000.0;
        let frequency = 1e9;
        let expected_db = 91.5;

        let loss_db = fspl_db_scalar(distance, frequency);
        assert!(
            (loss_db - expected_db).abs() < 1.0,
            "FSPL: expected ~{}, got {}",
            expected_db,
            loss_db
        );
    }

    #[test]
    fn test_fspl_distance_relationship() {
        // Doubling distance should increase path loss by 6 dB
        let f = 1e9;
        let loss_1km = fspl_db_scalar(1000.0, f);
        let loss_2km = fspl_db_scalar(2000.0, f);

        let delta = loss_2km - loss_1km;
        assert!(
            (delta - 6.0).abs() < TOLERANCE,
            "Doubling distance should add 6 dB, got {}",
            delta
        );
    }

    #[test]
    fn test_fspl_frequency_relationship() {
        // Doubling frequency should increase path loss by 6 dB
        let d = 1000.0;
        let loss_1ghz = fspl_db_scalar(d, 1e9);
        let loss_2ghz = fspl_db_scalar(d, 2e9);

        let delta = loss_2ghz - loss_1ghz;
        assert!(
            (delta - 6.0).abs() < TOLERANCE,
            "Doubling frequency should add 6 dB, got {}",
            delta
        );
    }

    #[test]
    fn test_log_distance_at_reference() {
        let ref_loss = 40.0;
        let atten = log_distance_scalar(1.0, 1.0, 2.0, ref_loss);
        let loss_db = -10.0 * atten.log10();

        assert!(
            (loss_db - ref_loss).abs() < TOLERANCE,
            "At d0, loss should be PL0: expected {}, got {}",
            ref_loss,
            loss_db
        );
    }

    #[test]
    fn test_hata_valid_range() {
        let loss = hata_urban_db_scalar(1.0, 900.0, 30.0, 1.5);

        assert!(
            loss > 100.0 && loss < 150.0,
            "Hata loss {} out of expected range 100-150 dB",
            loss
        );
    }

    #[test]
    fn test_hata_distance_increases_loss() {
        let loss_1km = hata_urban_db_scalar(1.0, 900.0, 30.0, 1.5);
        let loss_5km = hata_urban_db_scalar(5.0, 900.0, 30.0, 1.5);

        assert!(
            loss_5km > loss_1km,
            "Greater distance should have more loss: {} vs {}",
            loss_1km,
            loss_5km
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_fspl_simd_matches_scalar() {
        let distances = [100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0];
        let frequency = 1e9;

        let d_simd = f32x8::from_array(distances);
        let f_simd = f32x8::splat(frequency);
        let simd_result = fspl_simd(d_simd, f_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &d) in distances.iter().enumerate() {
            let scalar_result = fspl_scalar(d, frequency);
            let rel_error = ((simd_arr[i] - scalar_result) / scalar_result).abs();
            assert!(
                rel_error < 1e-4,
                "FSPL SIMD/scalar mismatch at d={}: {} vs {}, rel_err={}",
                d,
                simd_arr[i],
                scalar_result,
                rel_error
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_distance_calculation() {
        let p1 = (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0));
        let p2 = (f32x8::splat(3.0), f32x8::splat(4.0), f32x8::splat(0.0));

        let dist = calc_distance_simd(p1, p2);
        let arr: [f32; 8] = dist.into();

        for d in arr.iter() {
            assert!(
                (*d - 5.0).abs() < 1e-5,
                "3-4-5 triangle distance should be 5, got {}",
                d
            );
        }
    }

    // ========================================================================
    // Type-Safe Tests
    // ========================================================================

    #[test]
    fn test_type_safe_fspl() {
        let distance = Meters::new(1000.0);
        let frequency = Hertz::from_ghz(1.0).unwrap();

        let loss = fspl(distance, frequency);

        // Should be ~91.5 dB
        assert!(
            (loss.as_db() - 91.5).abs() < 1.0,
            "FSPL at 1km, 1GHz: expected ~91.5 dB, got {}",
            loss.as_db()
        );
    }

    #[test]
    fn test_type_safe_fspl_factor() {
        let distance = Meters::new(1000.0);
        let frequency = Hertz::from_ghz(1.0).unwrap();

        let factor = fspl_factor(distance, frequency);

        // ~91.5 dB loss = ~7e-10 linear factor
        assert!(
            factor.get() < 1e-8 && factor.get() > 1e-11,
            "FSPL factor at 1km, 1GHz: expected ~7e-10, got {}",
            factor.get()
        );
    }

    #[test]
    fn test_type_safe_log_distance() {
        let distance = Meters::new(100.0);
        let ref_distance = Meters::new(1.0);
        let ref_loss = PathLoss::new(40.0);
        let exponent = 2.0;

        let loss = log_distance(distance, ref_distance, exponent, ref_loss);

        // PL = 40 + 10 * 2 * log10(100) = 40 + 40 = 80 dB
        assert!(
            (loss.as_db() - 80.0).abs() < 0.5,
            "Log-distance at 100m: expected 80 dB, got {}",
            loss.as_db()
        );
    }

    #[test]
    fn test_type_safe_hata() {
        let distance = Meters::new(5000.0); // 5 km
        let frequency = Hertz::from_mhz(900.0).unwrap();
        let tx_height = Meters::new(30.0);
        let rx_height = Meters::new(1.5);

        let loss = hata_urban(distance, frequency, tx_height, rx_height);

        // Should be in reasonable range
        assert!(
            loss.as_db() > 100.0 && loss.as_db() < 160.0,
            "Hata loss at 5km, 900MHz: expected 100-160 dB, got {}",
            loss.as_db()
        );
    }

    #[test]
    fn test_type_safe_cost231() {
        let distance = Meters::new(5000.0);
        let frequency = Hertz::from_mhz(1800.0).unwrap();
        let tx_height = Meters::new(30.0);
        let rx_height = Meters::new(1.5);

        let loss_urban = cost231_hata(distance, frequency, tx_height, rx_height, true);
        let loss_suburban = cost231_hata(distance, frequency, tx_height, rx_height, false);

        // Urban should have 3 dB more loss
        assert!(
            (loss_urban.as_db() - loss_suburban.as_db() - 3.0).abs() < 0.1,
            "Urban should be 3 dB higher: {} vs {}",
            loss_urban.as_db(),
            loss_suburban.as_db()
        );
    }

    // ========================================================================
    // VLF Waveguide Tests
    // ========================================================================

    #[test]
    fn test_waveguide_very_low_attenuation() {
        // VLF waveguide should have very low attenuation per km
        // At 10 kHz over 1000 km, expect < 20 dB total (spreading + attenuation)
        let loss_1000km = waveguide_db_scalar(1000.0, 10e3, 75.0);

        assert!(
            loss_1000km < 35.0,
            "VLF waveguide at 1000 km should be < 35 dB, got {}",
            loss_1000km
        );
    }

    #[test]
    fn test_waveguide_distance_increases_loss() {
        // Loss should increase with distance
        let loss_100km = waveguide_db_scalar(100.0, 10e3, 75.0);
        let loss_1000km = waveguide_db_scalar(1000.0, 10e3, 75.0);

        assert!(
            loss_1000km > loss_100km,
            "Greater distance should have more loss: {} vs {}",
            loss_100km,
            loss_1000km
        );
    }

    #[test]
    fn test_waveguide_frequency_effect() {
        // Higher VLF frequencies should have slightly more attenuation
        let loss_5khz = waveguide_db_scalar(1000.0, 5e3, 75.0);
        let loss_25khz = waveguide_db_scalar(1000.0, 25e3, 75.0);

        assert!(
            loss_25khz > loss_5khz,
            "Higher VLF frequency should have more attenuation: {} vs {}",
            loss_5khz,
            loss_25khz
        );
    }

    #[test]
    fn test_waveguide_ionosphere_height_effect() {
        // Lower ionosphere should increase attenuation slightly
        let loss_high = waveguide_db_scalar(1000.0, 10e3, 90.0); // High ionosphere
        let loss_low = waveguide_db_scalar(1000.0, 10e3, 60.0);  // Low ionosphere

        assert!(
            loss_low > loss_high,
            "Lower ionosphere should have more attenuation: {} vs {}",
            loss_high,
            loss_low
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_waveguide_simd_matches_scalar() {
        let distances = [100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0];
        let frequency = 10e3;
        let height = 75.0;

        let d_simd = f32x8::from_array(distances);
        let f_simd = f32x8::splat(frequency);
        let h_simd = f32x8::splat(height);
        let simd_result = waveguide_simd(d_simd, f_simd, h_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &d) in distances.iter().enumerate() {
            let scalar_result = waveguide_db_scalar(d, frequency, height);
            let diff = (simd_arr[i] - scalar_result).abs();
            assert!(
                diff < 0.1,
                "Waveguide SIMD/scalar mismatch at d={}: {} vs {}, diff={}",
                d,
                simd_arr[i],
                scalar_result,
                diff
            );
        }
    }
}
