//! Propagation Model Trait
//!
//! Defines the interface for RF path loss calculations with both
//! scalar and SIMD implementations.

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

use crate::constants::SPEED_OF_LIGHT;

#[cfg(feature = "simd")]
use crate::simd_rf::math::simd_log;

const PI: f32 = std::f32::consts::PI;
const LOG10_E: f32 = std::f32::consts::LOG10_E;

/// Propagation model trait for computing RF path loss.
///
/// Implementations provide both scalar and SIMD variants for computing
/// path loss in dB given distance and frequency parameters.
pub trait PropagationModel: Send + Sync {
    /// Compute path loss in dB (scalar)
    ///
    /// # Arguments
    /// * `distance_m` - Distance between TX and RX in meters
    /// * `frequency_hz` - Carrier frequency in Hz
    /// * `tx_height_m` - Transmitter height in meters
    /// * `rx_height_m` - Receiver height in meters
    fn path_loss_db(&self, distance_m: f32, frequency_hz: f32, tx_height_m: f32, rx_height_m: f32) -> f32;

    /// Compute path loss in dB (SIMD 8-wide)
    #[cfg(feature = "simd")]
    fn path_loss_db_simd(
        &self,
        distance_m: f32x8,
        frequency_hz: f32x8,
        tx_height_m: f32x8,
        rx_height_m: f32x8,
    ) -> f32x8;

    /// Model name for debugging/logging
    fn name(&self) -> &'static str;

    /// Valid frequency range in Hz (min, max)
    fn valid_frequency_range(&self) -> (f32, f32);

    /// Valid distance range in meters (min, max)
    fn valid_distance_range(&self) -> (f32, f32);

    /// Check if parameters are within valid ranges
    fn is_valid(&self, distance_m: f32, frequency_hz: f32) -> bool {
        let (fmin, fmax) = self.valid_frequency_range();
        let (dmin, dmax) = self.valid_distance_range();
        frequency_hz >= fmin && frequency_hz <= fmax && distance_m >= dmin && distance_m <= dmax
    }
}

// ============================================================================
// Free Space Path Loss Model
// ============================================================================

/// Free Space Path Loss (FSPL) model
///
/// The simplest path loss model, ideal for unobstructed LOS propagation.
/// Valid for any frequency and distance.
///
/// Formula: L_fs(dB) = 20*log10(4πdf/c)
#[derive(Debug, Clone, Copy, Default)]
pub struct FSPLModel;

impl FSPLModel {
    pub fn new() -> Self {
        Self
    }
}

impl PropagationModel for FSPLModel {
    fn path_loss_db(&self, distance_m: f32, frequency_hz: f32, _tx_height_m: f32, _rx_height_m: f32) -> f32 {
        let d = distance_m.max(0.1);
        20.0 * (4.0 * PI * d * frequency_hz / SPEED_OF_LIGHT).log10()
    }

    #[cfg(feature = "simd")]
    fn path_loss_db_simd(
        &self,
        distance_m: f32x8,
        frequency_hz: f32x8,
        _tx_height_m: f32x8,
        _rx_height_m: f32x8,
    ) -> f32x8 {
        let d = distance_m.simd_max(f32x8::splat(0.1));
        let constant_db = f32x8::splat(-147.55); // 20*log10(4π/c)
        let log10_d = simd_log(d) * f32x8::splat(LOG10_E);
        let log10_f = simd_log(frequency_hz) * f32x8::splat(LOG10_E);
        f32x8::splat(20.0) * (log10_d + log10_f) + constant_db
    }

    fn name(&self) -> &'static str {
        "Free Space Path Loss"
    }

    fn valid_frequency_range(&self) -> (f32, f32) {
        (1.0, 300e9) // 1 Hz to 300 GHz
    }

    fn valid_distance_range(&self) -> (f32, f32) {
        (0.1, 1e9) // 0.1m to 1000 km
    }
}

// ============================================================================
// Log-Distance Path Loss Model
// ============================================================================

/// Log-distance path loss model
///
/// General model for indoor/outdoor propagation with configurable path loss exponent.
///
/// Formula: PL(d) = PL(d₀) + 10n·log₁₀(d/d₀)
///
/// Typical exponents:
/// - Free space: n = 2
/// - Urban: n = 2.7-3.5
/// - Indoor (LOS): n = 1.6-1.8
/// - Indoor (NLOS): n = 4-6
#[derive(Debug, Clone, Copy)]
pub struct LogDistanceModel {
    /// Path loss exponent (typically 2-6)
    pub path_loss_exponent: f32,
    /// Reference distance in meters (typically 1m)
    pub reference_distance_m: f32,
    /// Reference path loss at d₀ (in dB, or None to compute from FSPL)
    pub reference_loss_db: Option<f32>,
}

impl LogDistanceModel {
    /// Create with default reference distance of 1m
    pub fn new(path_loss_exponent: f32) -> Self {
        Self {
            path_loss_exponent,
            reference_distance_m: 1.0,
            reference_loss_db: None,
        }
    }

    /// Create with custom reference distance
    pub fn with_reference_distance(mut self, distance_m: f32) -> Self {
        self.reference_distance_m = distance_m;
        self
    }

    /// Create with explicit reference loss
    pub fn with_reference_loss_db(mut self, loss_db: f32) -> Self {
        self.reference_loss_db = Some(loss_db);
        self
    }

    /// Urban environment preset (n = 3.5)
    pub fn urban() -> Self {
        Self::new(3.5)
    }

    /// Indoor LOS preset (n = 1.8)
    pub fn indoor_los() -> Self {
        Self::new(1.8)
    }

    /// Indoor NLOS preset (n = 4.0)
    pub fn indoor_nlos() -> Self {
        Self::new(4.0)
    }
}

impl Default for LogDistanceModel {
    fn default() -> Self {
        Self::new(2.0) // Free space exponent
    }
}

impl PropagationModel for LogDistanceModel {
    fn path_loss_db(&self, distance_m: f32, frequency_hz: f32, _tx_height_m: f32, _rx_height_m: f32) -> f32 {
        let d = distance_m.max(0.1);
        let d0 = self.reference_distance_m;

        let ref_loss = self.reference_loss_db.unwrap_or_else(|| {
            // Compute FSPL at reference distance
            20.0 * (4.0 * PI * d0 * frequency_hz / SPEED_OF_LIGHT).log10()
        });

        ref_loss + 10.0 * self.path_loss_exponent * (d / d0).log10()
    }

    #[cfg(feature = "simd")]
    fn path_loss_db_simd(
        &self,
        distance_m: f32x8,
        frequency_hz: f32x8,
        _tx_height_m: f32x8,
        _rx_height_m: f32x8,
    ) -> f32x8 {
        let d = distance_m.simd_max(f32x8::splat(0.1));
        let d0 = f32x8::splat(self.reference_distance_m);

        let ref_loss = if let Some(loss) = self.reference_loss_db {
            f32x8::splat(loss)
        } else {
            // FSPL at reference distance
            let constant_db = f32x8::splat(-147.55);
            let log10_d0 = simd_log(d0) * f32x8::splat(LOG10_E);
            let log10_f = simd_log(frequency_hz) * f32x8::splat(LOG10_E);
            f32x8::splat(20.0) * (log10_d0 + log10_f) + constant_db
        };

        let log_ratio = simd_log(d / d0) * f32x8::splat(LOG10_E);
        ref_loss + f32x8::splat(10.0 * self.path_loss_exponent) * log_ratio
    }

    fn name(&self) -> &'static str {
        "Log-Distance Path Loss"
    }

    fn valid_frequency_range(&self) -> (f32, f32) {
        (1e6, 100e9) // 1 MHz to 100 GHz
    }

    fn valid_distance_range(&self) -> (f32, f32) {
        (0.1, 100_000.0) // 0.1m to 100 km
    }
}

// ============================================================================
// Okumura-Hata Urban Model
// ============================================================================

/// Okumura-Hata model for urban macro cells
///
/// Empirical model based on Okumura measurements. Valid for:
/// - Frequency: 150-1500 MHz
/// - Distance: 1-20 km
/// - TX height: 30-200 m
/// - RX height: 1-10 m
#[derive(Debug, Clone, Copy)]
pub struct HataUrbanModel {
    /// Large city correction (true for cities with tall buildings)
    pub large_city: bool,
}

impl HataUrbanModel {
    pub fn new() -> Self {
        Self { large_city: true }
    }

    pub fn medium_city() -> Self {
        Self { large_city: false }
    }
}

impl Default for HataUrbanModel {
    fn default() -> Self {
        Self::new()
    }
}

impl PropagationModel for HataUrbanModel {
    fn path_loss_db(&self, distance_m: f32, frequency_hz: f32, tx_height_m: f32, rx_height_m: f32) -> f32 {
        let d_km = (distance_m / 1000.0).clamp(1.0, 20.0);
        let f_mhz = (frequency_hz / 1e6).clamp(150.0, 1500.0);
        let h_t = tx_height_m.clamp(30.0, 200.0);
        let h_r = rx_height_m.clamp(1.0, 10.0);

        let log_f = f_mhz.log10();
        let log_ht = h_t.log10();
        let log_d = d_km.log10();

        // Mobile antenna correction factor
        let a_hr = if self.large_city && f_mhz >= 300.0 {
            // Large city, f >= 300 MHz
            3.2 * (11.75 * h_r).log10().powi(2) - 4.97
        } else if self.large_city {
            // Large city, f < 300 MHz
            8.29 * (1.54 * h_r).log10().powi(2) - 1.1
        } else {
            // Small/medium city
            (1.1 * log_f - 0.7) * h_r - (1.56 * log_f - 0.8)
        };

        69.55 + 26.16 * log_f - 13.82 * log_ht - a_hr + (44.9 - 6.55 * log_ht) * log_d
    }

    #[cfg(feature = "simd")]
    fn path_loss_db_simd(
        &self,
        distance_m: f32x8,
        frequency_hz: f32x8,
        tx_height_m: f32x8,
        rx_height_m: f32x8,
    ) -> f32x8 {
        let d = (distance_m / f32x8::splat(1000.0))
            .simd_clamp(f32x8::splat(1.0), f32x8::splat(20.0));
        let f = (frequency_hz / f32x8::splat(1e6))
            .simd_clamp(f32x8::splat(150.0), f32x8::splat(1500.0));
        let h_t = tx_height_m.simd_clamp(f32x8::splat(30.0), f32x8::splat(200.0));
        let h_r = rx_height_m.simd_clamp(f32x8::splat(1.0), f32x8::splat(10.0));

        let log_f = simd_log(f) * f32x8::splat(LOG10_E);
        let log_ht = simd_log(h_t) * f32x8::splat(LOG10_E);
        let log_d = simd_log(d) * f32x8::splat(LOG10_E);

        // Large city correction (f >= 300 MHz assumed for SIMD)
        let log_term = simd_log(f32x8::splat(11.75) * h_r) * f32x8::splat(LOG10_E);
        let a_hr = f32x8::splat(3.2) * log_term * log_term - f32x8::splat(4.97);

        f32x8::splat(69.55) + f32x8::splat(26.16) * log_f
            - f32x8::splat(13.82) * log_ht
            - a_hr
            + (f32x8::splat(44.9) - f32x8::splat(6.55) * log_ht) * log_d
    }

    fn name(&self) -> &'static str {
        "Okumura-Hata Urban"
    }

    fn valid_frequency_range(&self) -> (f32, f32) {
        (150e6, 1500e6) // 150-1500 MHz
    }

    fn valid_distance_range(&self) -> (f32, f32) {
        (1000.0, 20_000.0) // 1-20 km
    }
}

// ============================================================================
// COST-231 Hata Model
// ============================================================================

/// COST-231 Hata model extension
///
/// Extends Okumura-Hata to 1500-2000 MHz. Valid for:
/// - Frequency: 1500-2000 MHz
/// - Distance: 1-20 km
/// - TX height: 30-200 m
/// - RX height: 1-10 m
#[derive(Debug, Clone, Copy)]
pub struct Cost231Model {
    /// Metropolitan center correction (dense urban)
    pub metropolitan: bool,
}

impl Cost231Model {
    pub fn new() -> Self {
        Self { metropolitan: false }
    }

    pub fn metropolitan() -> Self {
        Self { metropolitan: true }
    }
}

impl Default for Cost231Model {
    fn default() -> Self {
        Self::new()
    }
}

impl PropagationModel for Cost231Model {
    fn path_loss_db(&self, distance_m: f32, frequency_hz: f32, tx_height_m: f32, rx_height_m: f32) -> f32 {
        let d_km = (distance_m / 1000.0).clamp(1.0, 20.0);
        let f_mhz = (frequency_hz / 1e6).clamp(1500.0, 2000.0);
        let h_t = tx_height_m.clamp(30.0, 200.0);
        let h_r = rx_height_m.clamp(1.0, 10.0);

        let log_f = f_mhz.log10();
        let log_ht = h_t.log10();
        let log_d = d_km.log10();

        // Mobile antenna correction
        let a_hr = (1.1 * log_f - 0.7) * h_r - (1.56 * log_f - 0.8);

        // Metropolitan correction
        let c_m = if self.metropolitan { 3.0 } else { 0.0 };

        46.3 + 33.9 * log_f - 13.82 * log_ht - a_hr + (44.9 - 6.55 * log_ht) * log_d + c_m
    }

    #[cfg(feature = "simd")]
    fn path_loss_db_simd(
        &self,
        distance_m: f32x8,
        frequency_hz: f32x8,
        tx_height_m: f32x8,
        rx_height_m: f32x8,
    ) -> f32x8 {
        let d = (distance_m / f32x8::splat(1000.0))
            .simd_clamp(f32x8::splat(1.0), f32x8::splat(20.0));
        let f = (frequency_hz / f32x8::splat(1e6))
            .simd_clamp(f32x8::splat(1500.0), f32x8::splat(2000.0));
        let h_t = tx_height_m.simd_clamp(f32x8::splat(30.0), f32x8::splat(200.0));
        let h_r = rx_height_m.simd_clamp(f32x8::splat(1.0), f32x8::splat(10.0));

        let log_f = simd_log(f) * f32x8::splat(LOG10_E);
        let log_ht = simd_log(h_t) * f32x8::splat(LOG10_E);
        let log_d = simd_log(d) * f32x8::splat(LOG10_E);

        let a_hr = (f32x8::splat(1.1) * log_f - f32x8::splat(0.7)) * h_r
            - (f32x8::splat(1.56) * log_f - f32x8::splat(0.8));

        let c_m = if self.metropolitan {
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

    fn name(&self) -> &'static str {
        "COST-231 Hata"
    }

    fn valid_frequency_range(&self) -> (f32, f32) {
        (1500e6, 2000e6) // 1500-2000 MHz
    }

    fn valid_distance_range(&self) -> (f32, f32) {
        (1000.0, 20_000.0) // 1-20 km
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1.0; // 1 dB tolerance

    #[test]
    fn test_fspl_known_value() {
        let model = FSPLModel::new();
        // 1 km at 1 GHz should be approximately 91.5 dB
        let loss = model.path_loss_db(1000.0, 1e9, 10.0, 1.5);
        assert!(
            (loss - 91.5).abs() < TOLERANCE,
            "FSPL at 1km, 1GHz should be ~91.5 dB, got {}",
            loss
        );
    }

    #[test]
    fn test_fspl_distance_relationship() {
        let model = FSPLModel::new();
        let f = 1e9;
        let loss_1km = model.path_loss_db(1000.0, f, 10.0, 1.5);
        let loss_2km = model.path_loss_db(2000.0, f, 10.0, 1.5);

        let delta = loss_2km - loss_1km;
        assert!(
            (delta - 6.0).abs() < 0.1,
            "Doubling distance should add 6 dB, got {}",
            delta
        );
    }

    #[test]
    fn test_log_distance_at_reference() {
        let model = LogDistanceModel::new(2.0).with_reference_loss_db(40.0);
        let loss = model.path_loss_db(1.0, 1e9, 10.0, 1.5);
        assert!(
            (loss - 40.0).abs() < 0.1,
            "At d0, loss should equal ref loss: expected 40, got {}",
            loss
        );
    }

    #[test]
    fn test_log_distance_exponent() {
        let model = LogDistanceModel::new(3.0).with_reference_loss_db(40.0);
        let loss_1m = model.path_loss_db(1.0, 1e9, 10.0, 1.5);
        let loss_10m = model.path_loss_db(10.0, 1e9, 10.0, 1.5);

        // PL(10) = PL(1) + 10*3*log10(10) = 40 + 30 = 70 dB
        assert!(
            (loss_10m - 70.0).abs() < 0.1,
            "At 10m with n=3, loss should be 70 dB, got {}",
            loss_10m
        );
    }

    #[test]
    fn test_hata_valid_range() {
        let model = HataUrbanModel::new();
        let loss = model.path_loss_db(1000.0, 900e6, 30.0, 1.5);

        // Should be between 100-150 dB for typical urban
        assert!(
            loss > 100.0 && loss < 150.0,
            "Hata loss {} out of expected range 100-150 dB",
            loss
        );
    }

    #[test]
    fn test_hata_distance_increases_loss() {
        let model = HataUrbanModel::new();
        let loss_1km = model.path_loss_db(1000.0, 900e6, 30.0, 1.5);
        let loss_5km = model.path_loss_db(5000.0, 900e6, 30.0, 1.5);

        assert!(
            loss_5km > loss_1km,
            "Greater distance should have more loss: {} vs {}",
            loss_1km,
            loss_5km
        );
    }

    #[test]
    fn test_cost231_valid_range() {
        let model = Cost231Model::new();
        let loss = model.path_loss_db(1000.0, 1800e6, 30.0, 1.5);

        // Should be between 100-160 dB for typical urban at higher frequency
        assert!(
            loss > 100.0 && loss < 160.0,
            "COST-231 loss {} out of expected range",
            loss
        );
    }

    #[test]
    fn test_cost231_metropolitan_correction() {
        let medium = Cost231Model::new();
        let metro = Cost231Model::metropolitan();

        let loss_medium = medium.path_loss_db(5000.0, 1800e6, 50.0, 1.5);
        let loss_metro = metro.path_loss_db(5000.0, 1800e6, 50.0, 1.5);

        // Metropolitan adds 3 dB
        assert!(
            (loss_metro - loss_medium - 3.0).abs() < 0.1,
            "Metropolitan should add 3 dB: {} vs {}",
            loss_medium,
            loss_metro
        );
    }

    #[test]
    fn test_model_names() {
        assert_eq!(FSPLModel::new().name(), "Free Space Path Loss");
        assert_eq!(LogDistanceModel::default().name(), "Log-Distance Path Loss");
        assert_eq!(HataUrbanModel::new().name(), "Okumura-Hata Urban");
        assert_eq!(Cost231Model::new().name(), "COST-231 Hata");
    }

    #[test]
    fn test_validity_check() {
        let fspl = FSPLModel::new();
        assert!(fspl.is_valid(100.0, 1e9));
        assert!(fspl.is_valid(0.1, 1e9));
        assert!(!fspl.is_valid(0.01, 1e9)); // Below min distance

        let hata = HataUrbanModel::new();
        assert!(hata.is_valid(1000.0, 900e6));
        assert!(!hata.is_valid(100.0, 900e6)); // Below min distance
        assert!(!hata.is_valid(1000.0, 100e6)); // Below min frequency
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_fspl_simd_matches_scalar() {
        let model = FSPLModel::new();
        let distances = [100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0];
        let frequency = 1e9;

        let d_simd = f32x8::from_array(distances);
        let f_simd = f32x8::splat(frequency);
        let h_t = f32x8::splat(30.0);
        let h_r = f32x8::splat(1.5);

        let simd_result = model.path_loss_db_simd(d_simd, f_simd, h_t, h_r);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &d) in distances.iter().enumerate() {
            let scalar = model.path_loss_db(d, frequency, 30.0, 1.5);
            let diff = (simd_arr[i] - scalar).abs();
            assert!(
                diff < 0.1,
                "FSPL SIMD/scalar mismatch at d={}: {} vs {}, diff={}",
                d, simd_arr[i], scalar, diff
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_hata_simd_matches_scalar() {
        let model = HataUrbanModel::new();
        let distances: [f32; 8] = [1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0];
        let frequency = 900e6;

        let d_simd = f32x8::from_array(distances);
        let f_simd = f32x8::splat(frequency);
        let h_t = f32x8::splat(50.0);
        let h_r = f32x8::splat(1.5);

        let simd_result = model.path_loss_db_simd(d_simd, f_simd, h_t, h_r);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &d) in distances.iter().enumerate() {
            let scalar = model.path_loss_db(d, frequency, 50.0, 1.5);
            let diff = (simd_arr[i] - scalar).abs();
            assert!(
                diff < 0.5, // Slightly looser tolerance for complex model
                "Hata SIMD/scalar mismatch at d={}: {} vs {}, diff={}",
                d, simd_arr[i], scalar, diff
            );
        }
    }
}
