//! Atmospheric Effects
//!
//! SIMD-optimized implementations of atmospheric absorption models based on
//! ITU-R recommendations:
//! - ITU-R P.676: Gaseous absorption (O2 and H2O)
//! - ITU-R P.838: Rain attenuation
//!
//! These effects become significant above ~10 GHz and are critical for
//! satellite and mmWave communications.

#[cfg(feature = "simd")]
use std::simd::f32x8;

#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;

#[cfg(feature = "simd")]
use crate::simd_rf::math::{simd_exp, simd_pow};

// ============================================================================
// Constants for ITU-R P.676 Absorption Lines
// ============================================================================

/// O₂ absorption resonance frequencies (GHz)
const O2_RESONANCE_60: f32 = 60.0; // Primary O₂ absorption complex
const O2_RESONANCE_119: f32 = 118.75; // Secondary O₂ line

/// H₂O absorption resonance frequencies (GHz)
const H2O_RESONANCE_22: f32 = 22.235; // Primary H₂O line
const H2O_RESONANCE_183: f32 = 183.31; // Strong H₂O line
const H2O_RESONANCE_325: f32 = 325.15; // Additional H₂O line


// ============================================================================
// SIMD Atmospheric Models - Enhanced ITU-R P.676
// ============================================================================

/// Compute clamped Gaussian SIMD (avoids exp overflow for distant peaks)
#[cfg(feature = "simd")]
#[inline]
fn simd_gaussian_clamped(diff: f32x8, amplitude: f32x8) -> f32x8 {
    // Clamp exponent to avoid overflow in simd_exp (valid range: [-88, 88])
    let neg_half_diff_sq = f32x8::splat(-0.5) * diff * diff;
    let clamped = neg_half_diff_sq.simd_max(f32x8::splat(-80.0));
    amplitude * simd_exp(clamped)
}

/// Oxygen absorption (ITU-R P.676 approximation)
///
/// Models the O₂ absorption complex centered at 60 GHz and the
/// isolated line at 118.75 GHz. The 60 GHz complex is critical for
/// 5G mmWave (n260/n261) planning.
///
/// # Arguments
/// * `frequency_ghz` - Frequency in GHz
///
/// # Returns
/// O₂ specific attenuation in dB/km
#[cfg(feature = "simd")]
#[inline]
pub fn o2_absorption_db_per_km(frequency_ghz: f32x8) -> f32x8 {
    let f = frequency_ghz;

    // Background O₂ absorption (frequency squared term)
    // Dominant at lower frequencies where resonances are distant
    let f2 = f * f;
    let background = f32x8::splat(0.0019) * f2 / f32x8::splat(1000.0);

    // 60 GHz O₂ complex (overlapping lines form a broad absorption peak)
    // Peak attenuation ~15 dB/km at exactly 60 GHz
    // Width ~10 GHz due to pressure broadening
    let diff_60 = (f - f32x8::splat(O2_RESONANCE_60)) / f32x8::splat(8.0);
    let peak_60 = simd_gaussian_clamped(diff_60, f32x8::splat(15.0));

    // 118.75 GHz isolated O₂ line
    // Peak attenuation ~1.5 dB/km, narrower than 60 GHz complex
    let diff_119 = (f - f32x8::splat(O2_RESONANCE_119)) / f32x8::splat(3.0);
    let peak_119 = simd_gaussian_clamped(diff_119, f32x8::splat(1.5));

    background + peak_60 + peak_119
}

/// Water vapor absorption (ITU-R P.676 approximation)
///
/// Models H₂O absorption lines at 22.2 GHz, 183 GHz, and 325 GHz.
/// Assumes default water vapor density of 7.5 g/m³.
///
/// # Arguments
/// * `frequency_ghz` - Frequency in GHz
///
/// # Returns
/// H₂O specific attenuation in dB/km
#[cfg(feature = "simd")]
#[inline]
pub fn h2o_absorption_db_per_km(frequency_ghz: f32x8) -> f32x8 {
    let f = frequency_ghz;

    // 22.235 GHz water vapor line
    // Peak ~0.2 dB/km at 7.5 g/m³, pressure broadened width ~5 GHz
    let diff_22 = (f - f32x8::splat(H2O_RESONANCE_22)) / f32x8::splat(5.0);
    let peak_22 = simd_gaussian_clamped(diff_22, f32x8::splat(0.2));

    // 183.31 GHz water vapor line (strong)
    // Peak ~30 dB/km at 7.5 g/m³, narrower due to lower pressure at altitude
    // At sea level, pressure broadening makes it ~5 GHz wide
    let diff_183 = (f - f32x8::splat(H2O_RESONANCE_183)) / f32x8::splat(5.0);
    let peak_183 = simd_gaussian_clamped(diff_183, f32x8::splat(30.0));

    // 325.15 GHz water vapor line
    // Peak ~15 dB/km at 7.5 g/m³
    let diff_325 = (f - f32x8::splat(H2O_RESONANCE_325)) / f32x8::splat(4.0);
    let peak_325 = simd_gaussian_clamped(diff_325, f32x8::splat(15.0));

    peak_22 + peak_183 + peak_325
}

/// Atmospheric gaseous absorption (ITU-R P.676)
///
/// Models absorption due to oxygen and water vapor at sea level.
/// Includes accurate modeling of:
/// - O₂ 60 GHz complex (critical for 5G mmWave)
/// - O₂ 118.75 GHz line
/// - H₂O 22.2 GHz, 183 GHz, 325 GHz lines
///
/// # Arguments
/// * `frequency_ghz` - Frequency in GHz
///
/// # Returns
/// Specific attenuation in dB/km
#[cfg(feature = "simd")]
#[inline]
pub fn atmospheric_attenuation_db_per_km(frequency_ghz: f32x8) -> f32x8 {
    o2_absorption_db_per_km(frequency_ghz) + h2o_absorption_db_per_km(frequency_ghz)
}

/// Apply atmospheric absorption to signal power
///
/// # Arguments
/// * `power` - Input power (linear scale)
/// * `distance_km` - Distance in kilometers
/// * `frequency_ghz` - Frequency in GHz
///
/// # Returns
/// Attenuated power (linear scale)
#[cfg(feature = "simd")]
#[inline]
pub fn apply_atmospheric_simd(power: f32x8, distance_km: f32x8, frequency_ghz: f32x8) -> f32x8 {
    let atten_db_per_km = atmospheric_attenuation_db_per_km(frequency_ghz);
    let total_atten_db = atten_db_per_km * distance_km;

    // Convert to linear and apply
    power * simd_db_to_linear(-total_atten_db)
}

/// Rain attenuation (simplified ITU-R P.838)
///
/// Models signal attenuation due to rain, which becomes significant above 10 GHz.
///
/// # Arguments
/// * `frequency_ghz` - Frequency in GHz
/// * `rain_rate_mm_per_hour` - Rain rate in mm/h
///
/// # Returns
/// Specific attenuation in dB/km
#[cfg(feature = "simd")]
#[inline]
pub fn rain_attenuation_db_per_km(frequency_ghz: f32x8, rain_rate_mm_per_hour: f32x8) -> f32x8 {
    // Power-law model: γ_R = k × R^α
    // k and α are frequency-dependent
    // Simplified approximations for horizontal polarization

    let f = frequency_ghz;

    // Approximate k(f) for 1-100 GHz (horizontal polarization)
    // log10(k) ≈ -3.5 + 0.1 × f for f < 30 GHz (simplified)
    let log_k = f32x8::splat(-3.5) + f32x8::splat(0.05) * f;
    let k = simd_pow(f32x8::splat(10.0), log_k);

    // Approximate α(f) for 1-100 GHz
    // α ≈ 0.8 + 0.01 × f (simplified)
    let alpha = f32x8::splat(0.8) + f32x8::splat(0.01) * f;

    // γ_R = k × R^α
    k * simd_pow(rain_rate_mm_per_hour, alpha)
}

/// Apply rain attenuation to signal power
///
/// # Arguments
/// * `power` - Input power (linear scale)
/// * `distance_km` - Distance in kilometers
/// * `frequency_ghz` - Frequency in GHz
/// * `rain_rate_mm_per_hour` - Rain rate in mm/h
///
/// # Returns
/// Attenuated power (linear scale)
#[cfg(feature = "simd")]
#[inline]
pub fn apply_rain_attenuation_simd(
    power: f32x8,
    distance_km: f32x8,
    frequency_ghz: f32x8,
    rain_rate_mm_per_hour: f32x8,
) -> f32x8 {
    let atten_db_per_km = rain_attenuation_db_per_km(frequency_ghz, rain_rate_mm_per_hour);
    let total_atten_db = atten_db_per_km * distance_km;

    power * simd_db_to_linear(-total_atten_db)
}

/// Combined atmospheric attenuation (gaseous + rain)
///
/// # Arguments
/// * `power` - Input power (linear scale)
/// * `distance_km` - Distance in kilometers
/// * `frequency_ghz` - Frequency in GHz
/// * `rain_rate_mm_per_hour` - Rain rate in mm/h (0 for clear sky)
///
/// # Returns
/// Attenuated power (linear scale)
#[cfg(feature = "simd")]
#[inline]
pub fn apply_total_atmospheric_simd(
    power: f32x8,
    distance_km: f32x8,
    frequency_ghz: f32x8,
    rain_rate_mm_per_hour: f32x8,
) -> f32x8 {
    let gaseous_db = atmospheric_attenuation_db_per_km(frequency_ghz) * distance_km;
    let rain_db = rain_attenuation_db_per_km(frequency_ghz, rain_rate_mm_per_hour) * distance_km;
    let total_db = gaseous_db + rain_db;

    power * simd_db_to_linear(-total_db)
}

/// Fog/cloud attenuation (simplified model)
///
/// Significant for frequencies above 30 GHz.
///
/// # Arguments
/// * `frequency_ghz` - Frequency in GHz
/// * `liquid_water_content` - Liquid water content in g/m³ (typically 0.05-0.5)
///
/// # Returns
/// Specific attenuation in dB/km
#[cfg(feature = "simd")]
#[inline]
pub fn fog_attenuation_db_per_km(frequency_ghz: f32x8, liquid_water_content: f32x8) -> f32x8 {
    // Simplified model: γ_c = K_l × M
    // K_l ≈ 0.4 × f² at frequencies below 100 GHz (very simplified)
    let k_l = f32x8::splat(0.4) * frequency_ghz * frequency_ghz / f32x8::splat(1000.0);
    k_l * liquid_water_content
}

// ============================================================================
// Scalar Atmospheric Models
// ============================================================================

/// Scalar O₂ absorption (ITU-R P.676 approximation)
///
/// Models the O₂ 60 GHz complex and 118.75 GHz line.
#[inline]
pub fn o2_absorption_db_per_km_scalar(frequency_ghz: f32) -> f32 {
    let f = frequency_ghz;

    // Background O₂ absorption
    let f2 = f * f;
    let background = 0.0019 * f2 / 1000.0;

    // 60 GHz O₂ complex (peak ~15 dB/km)
    let diff_60 = (f - O2_RESONANCE_60) / 8.0;
    let peak_60 = 15.0 * (-0.5 * diff_60 * diff_60).exp();

    // 118.75 GHz O₂ line (peak ~1.5 dB/km)
    let diff_119 = (f - O2_RESONANCE_119) / 3.0;
    let peak_119 = 1.5 * (-0.5 * diff_119 * diff_119).exp();

    background + peak_60 + peak_119
}

/// Scalar H₂O absorption (ITU-R P.676 approximation)
///
/// Models H₂O lines at 22.2 GHz, 183 GHz, and 325 GHz.
#[inline]
pub fn h2o_absorption_db_per_km_scalar(frequency_ghz: f32) -> f32 {
    let f = frequency_ghz;

    // 22.235 GHz line (peak ~0.2 dB/km)
    let diff_22 = (f - H2O_RESONANCE_22) / 5.0;
    let peak_22 = 0.2 * (-0.5 * diff_22 * diff_22).exp();

    // 183.31 GHz line (peak ~30 dB/km)
    let diff_183 = (f - H2O_RESONANCE_183) / 5.0;
    let peak_183 = 30.0 * (-0.5 * diff_183 * diff_183).exp();

    // 325.15 GHz line (peak ~15 dB/km)
    let diff_325 = (f - H2O_RESONANCE_325) / 4.0;
    let peak_325 = 15.0 * (-0.5 * diff_325 * diff_325).exp();

    peak_22 + peak_183 + peak_325
}

/// Scalar atmospheric attenuation (O₂ + H₂O combined)
#[inline]
pub fn atmospheric_attenuation_db_per_km_scalar(frequency_ghz: f32) -> f32 {
    o2_absorption_db_per_km_scalar(frequency_ghz) + h2o_absorption_db_per_km_scalar(frequency_ghz)
}

/// Scalar rain attenuation (ITU-R P.838)
#[inline]
pub fn rain_attenuation_db_per_km_scalar(frequency_ghz: f32, rain_rate_mm_per_hour: f32) -> f32 {
    let f = frequency_ghz;
    let log_k = -3.5 + 0.05 * f;
    let k = 10.0_f32.powf(log_k);
    let alpha = 0.8 + 0.01 * f;

    k * rain_rate_mm_per_hour.powf(alpha)
}

/// Apply scalar atmospheric absorption
#[inline]
pub fn apply_atmospheric_scalar(power: f32, distance_km: f32, frequency_ghz: f32) -> f32 {
    let atten_db = atmospheric_attenuation_db_per_km_scalar(frequency_ghz) * distance_km;
    power * 10.0_f32.powf(-atten_db / 10.0)
}

// ============================================================================
// Helper Functions
// ============================================================================

#[cfg(feature = "simd")]
#[inline]
fn simd_db_to_linear(db: f32x8) -> f32x8 {
    crate::simd_rf::math::simd_db_to_linear(db)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 0.1;

    #[test]
    fn test_atmospheric_positive() {
        // Atmospheric attenuation should always be positive
        for f in [1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 183.0] {
            let atten = atmospheric_attenuation_db_per_km_scalar(f);
            assert!(
                atten > 0.0,
                "Attenuation at {} GHz should be positive: {}",
                f,
                atten
            );
        }
    }

    #[test]
    fn test_o2_60ghz_peak() {
        // O₂ absorption should peak near 60 GHz
        let atten_40 = o2_absorption_db_per_km_scalar(40.0);
        let atten_60 = o2_absorption_db_per_km_scalar(60.0);
        let atten_80 = o2_absorption_db_per_km_scalar(80.0);

        // 60 GHz should be the peak
        assert!(
            atten_60 > atten_40,
            "60 GHz O₂ should exceed 40 GHz: {} vs {}",
            atten_60,
            atten_40
        );
        assert!(
            atten_60 > atten_80,
            "60 GHz O₂ should exceed 80 GHz: {} vs {}",
            atten_60,
            atten_80
        );

        // Peak should be around 15 dB/km
        assert!(
            atten_60 > 10.0 && atten_60 < 20.0,
            "60 GHz O₂ peak should be ~15 dB/km: {}",
            atten_60
        );
    }

    #[test]
    fn test_o2_119ghz_peak() {
        // Secondary O₂ peak at 118.75 GHz
        let atten_100 = o2_absorption_db_per_km_scalar(100.0);
        let atten_119 = o2_absorption_db_per_km_scalar(118.75);
        let atten_140 = o2_absorption_db_per_km_scalar(140.0);

        // Local peak around 119 GHz
        assert!(
            atten_119 > atten_100,
            "119 GHz O₂ should exceed 100 GHz"
        );
        assert!(
            atten_119 > atten_140,
            "119 GHz O₂ should exceed 140 GHz"
        );
    }

    #[test]
    fn test_h2o_22ghz_peak() {
        // H₂O absorption peak at 22.235 GHz
        let atten_15 = h2o_absorption_db_per_km_scalar(15.0);
        let atten_22 = h2o_absorption_db_per_km_scalar(22.235);
        let atten_30 = h2o_absorption_db_per_km_scalar(30.0);

        assert!(
            atten_22 > atten_15,
            "22 GHz H₂O should exceed 15 GHz"
        );
        assert!(
            atten_22 > atten_30,
            "22 GHz H₂O should exceed 30 GHz"
        );

        // Peak should be around 0.2 dB/km at 7.5 g/m³
        assert!(
            atten_22 > 0.1 && atten_22 < 0.3,
            "22 GHz H₂O peak should be ~0.2 dB/km: {}",
            atten_22
        );
    }

    #[test]
    fn test_h2o_183ghz_peak() {
        // Strong H₂O absorption at 183 GHz
        let atten_160 = h2o_absorption_db_per_km_scalar(160.0);
        let atten_183 = h2o_absorption_db_per_km_scalar(183.31);
        let atten_200 = h2o_absorption_db_per_km_scalar(200.0);

        assert!(
            atten_183 > atten_160,
            "183 GHz H₂O should exceed 160 GHz"
        );
        assert!(
            atten_183 > atten_200,
            "183 GHz H₂O should exceed 200 GHz"
        );

        // Peak should be around 30 dB/km
        assert!(
            atten_183 > 20.0 && atten_183 < 40.0,
            "183 GHz H₂O peak should be ~30 dB/km: {}",
            atten_183
        );
    }

    #[test]
    fn test_h2o_325ghz_peak() {
        // H₂O absorption at 325 GHz
        let atten_300 = h2o_absorption_db_per_km_scalar(300.0);
        let atten_325 = h2o_absorption_db_per_km_scalar(325.15);
        let atten_350 = h2o_absorption_db_per_km_scalar(350.0);

        assert!(
            atten_325 > atten_300,
            "325 GHz H₂O should exceed 300 GHz"
        );
        assert!(
            atten_325 > atten_350,
            "325 GHz H₂O should exceed 350 GHz"
        );
    }

    #[test]
    fn test_atmospheric_window_35ghz() {
        // 35 GHz is in an atmospheric window (between O₂ and H₂O peaks)
        let atten_35 = atmospheric_attenuation_db_per_km_scalar(35.0);
        let atten_60 = atmospheric_attenuation_db_per_km_scalar(60.0);

        assert!(
            atten_35 < atten_60,
            "35 GHz window should have less attenuation than 60 GHz peak"
        );
    }

    #[test]
    fn test_rain_positive() {
        for f in [10.0, 20.0, 40.0] {
            let atten = rain_attenuation_db_per_km_scalar(f, 10.0); // 10 mm/h moderate rain
            assert!(
                atten > 0.0,
                "Rain attenuation at {} GHz should be positive: {}",
                f,
                atten
            );
        }
    }

    #[test]
    fn test_rain_increases_with_rate() {
        let f = 30.0; // 30 GHz
        let atten_5 = rain_attenuation_db_per_km_scalar(f, 5.0);
        let atten_20 = rain_attenuation_db_per_km_scalar(f, 20.0);
        let atten_50 = rain_attenuation_db_per_km_scalar(f, 50.0);

        assert!(
            atten_20 > atten_5,
            "Higher rain rate should have more attenuation"
        );
        assert!(
            atten_50 > atten_20,
            "Higher rain rate should have more attenuation"
        );
    }

    #[test]
    fn test_rain_increases_with_frequency() {
        let rain = 10.0; // 10 mm/h
        let atten_10 = rain_attenuation_db_per_km_scalar(10.0, rain);
        let atten_30 = rain_attenuation_db_per_km_scalar(30.0, rain);
        let atten_60 = rain_attenuation_db_per_km_scalar(60.0, rain);

        assert!(
            atten_30 > atten_10,
            "Higher frequency should have more rain attenuation"
        );
        assert!(
            atten_60 > atten_30,
            "Higher frequency should have more rain attenuation"
        );
    }

    #[test]
    fn test_apply_atmospheric_reduces_power() {
        let power = 1.0;
        let distance = 10.0; // 10 km
        let freq = 30.0; // 30 GHz

        let attenuated = apply_atmospheric_scalar(power, distance, freq);

        assert!(
            attenuated < power,
            "Atmospheric absorption should reduce power: {} -> {}",
            power,
            attenuated
        );
        assert!(
            attenuated > 0.0,
            "Power should remain positive: {}",
            attenuated
        );
    }

    #[test]
    fn test_no_attenuation_at_low_freq() {
        // At VHF/UHF, atmospheric effects should be negligible
        let atten = atmospheric_attenuation_db_per_km_scalar(0.5); // 500 MHz

        assert!(
            atten < 0.01,
            "Attenuation at 500 MHz should be negligible: {} dB/km",
            atten
        );
    }

    #[test]
    fn test_5g_mmwave_bands() {
        // Test specific 5G mmWave bands
        let atten_n260 = atmospheric_attenuation_db_per_km_scalar(39.0); // n260: 37-40 GHz
        let atten_n261 = atmospheric_attenuation_db_per_km_scalar(28.0); // n261: 27.5-28.35 GHz

        // These bands are in atmospheric windows, moderate attenuation
        assert!(
            atten_n260 < 5.0,
            "n260 band (39 GHz) should have moderate attenuation: {}",
            atten_n260
        );
        assert!(
            atten_n261 < 1.0,
            "n261 band (28 GHz) should have low attenuation: {}",
            atten_n261
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_atmospheric_simd_matches_scalar() {
        let frequencies = [1.0, 22.0, 35.0, 60.0, 100.0, 119.0, 150.0, 183.0];
        let f_simd = f32x8::from_array(frequencies);

        let simd_result = atmospheric_attenuation_db_per_km(f_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &f) in frequencies.iter().enumerate() {
            let scalar_result = atmospheric_attenuation_db_per_km_scalar(f);
            let rel_error = ((simd_arr[i] - scalar_result) / scalar_result.max(0.001)).abs();
            assert!(
                rel_error < 0.1,
                "SIMD/scalar mismatch at {} GHz: {} vs {}",
                f,
                simd_arr[i],
                scalar_result
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_o2_simd_matches_scalar() {
        let frequencies = [40.0, 50.0, 60.0, 70.0, 100.0, 118.75, 130.0, 150.0];
        let f_simd = f32x8::from_array(frequencies);

        let simd_result = o2_absorption_db_per_km(f_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &f) in frequencies.iter().enumerate() {
            let scalar_result = o2_absorption_db_per_km_scalar(f);
            let rel_error = ((simd_arr[i] - scalar_result) / scalar_result.max(0.001)).abs();
            assert!(
                rel_error < 0.1,
                "O₂ SIMD/scalar mismatch at {} GHz: {} vs {}",
                f,
                simd_arr[i],
                scalar_result
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_h2o_simd_matches_scalar() {
        let frequencies = [15.0, 22.235, 30.0, 150.0, 183.31, 200.0, 325.15, 350.0];
        let f_simd = f32x8::from_array(frequencies);

        let simd_result = h2o_absorption_db_per_km(f_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &f) in frequencies.iter().enumerate() {
            let scalar_result = h2o_absorption_db_per_km_scalar(f);
            let rel_error = ((simd_arr[i] - scalar_result) / scalar_result.max(0.001)).abs();
            assert!(
                rel_error < 0.1,
                "H₂O SIMD/scalar mismatch at {} GHz: {} vs {}",
                f,
                simd_arr[i],
                scalar_result
            );
        }
    }
}
