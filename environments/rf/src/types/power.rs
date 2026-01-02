//! Power and Gain Types
//!
//! Types for power in different representations (linear, dBm, dBW)
//! and gain/loss values, with explicit domain tracking.

use super::primitives::{PositiveF32, PositivePower};

// ============================================================================
// PowerDbm: Power in dBm
// ============================================================================

/// Power in dBm (decibels relative to 1 milliwatt).
///
/// This type represents power in logarithmic scale.
/// Unlike linear power, dBm values can be negative (sub-milliwatt).
///
/// # Domain
/// Any finite f32 value. Very negative values represent very small power,
/// very positive values represent large power.
///
/// # Common Reference Points
/// - 0 dBm = 1 mW
/// - 30 dBm = 1 W
/// - -30 dBm = 1 μW
/// - -174 dBm/Hz = thermal noise floor at 290K
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PowerDbm(f32);

impl PowerDbm {
    /// Minimum representable power floor.
    /// Below this, we consider power to be effectively zero.
    pub const FLOOR: Self = Self(-300.0);

    /// Zero dBm (1 mW reference).
    pub const ZERO_DBM: Self = Self(0.0);

    /// 30 dBm (1 W).
    pub const ONE_WATT: Self = Self(30.0);

    /// Thermal noise floor (-174 dBm/Hz at 290K).
    pub const THERMAL_NOISE_DBM_HZ: Self = Self(-174.0);

    /// Try to create from raw dBm value.
    #[inline]
    pub fn try_new(dbm: f32) -> Option<Self> {
        dbm.is_finite().then(|| Self(dbm))
    }

    /// Create from raw dBm value, panicking if non-finite.
    #[inline]
    pub fn new(dbm: f32) -> Self {
        Self::try_new(dbm).expect("PowerDbm requires finite value")
    }

    /// Convert from linear power in Watts.
    #[inline]
    pub fn from_watts(watts: PositivePower) -> Self {
        let w = watts.watts();
        if w > 1e-30 {
            // P(dBm) = 10 * log10(P_watts * 1000)
            Self(10.0 * (w * 1000.0).log10())
        } else {
            Self::FLOOR
        }
    }

    /// Convert from linear power (PositiveF32 in Watts).
    #[inline]
    pub fn from_linear(watts: PositiveF32) -> Self {
        let w = watts.get();
        if w > 1e-30 {
            Self(10.0 * (w * 1000.0).log10())
        } else {
            Self::FLOOR
        }
    }

    /// Get the raw dBm value.
    #[inline]
    pub fn as_dbm(self) -> f32 {
        self.0
    }

    /// Convert to linear power in Watts.
    ///
    /// This always succeeds and returns a valid PositivePower.
    #[inline]
    pub fn to_watts(self) -> PositivePower {
        // P_watts = 10^(dBm/10) / 1000
        let watts = 10.0_f32.powf(self.0 / 10.0) / 1000.0;
        PositivePower::new(watts.max(0.0))
    }

    /// Convert to linear power (PositiveF32).
    #[inline]
    pub fn to_linear(self) -> PositiveF32 {
        let watts = 10.0_f32.powf(self.0 / 10.0) / 1000.0;
        PositiveF32::new(watts.max(0.0))
    }

    /// Convert to milliwatts (linear).
    #[inline]
    pub fn to_milliwatts(self) -> PositiveF32 {
        let mw = 10.0_f32.powf(self.0 / 10.0);
        PositiveF32::new(mw.max(0.0))
    }

    /// Add a gain value (in dB).
    #[inline]
    pub fn add_gain(self, gain: GainDb) -> Self {
        Self(self.0 + gain.0)
    }

    /// Subtract a loss value (path loss, in dB).
    #[inline]
    pub fn subtract_loss(self, loss: PathLoss) -> Self {
        Self(self.0 - loss.as_db())
    }

    /// Compare two power levels and return the difference in dB.
    #[inline]
    pub fn difference(self, other: Self) -> f32 {
        self.0 - other.0
    }

    /// Check if power is below the floor threshold.
    #[inline]
    pub fn is_negligible(self) -> bool {
        self.0 <= Self::FLOOR.0
    }

    /// Clamp to a range.
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }
}

impl Default for PowerDbm {
    fn default() -> Self {
        Self::FLOOR
    }
}

impl From<PositivePower> for PowerDbm {
    fn from(power: PositivePower) -> Self {
        Self::from_watts(power)
    }
}

// ============================================================================
// PathLoss: Always Positive Loss in dB
// ============================================================================

/// Path loss in dB (decibels).
///
/// Path loss represents signal attenuation and is always positive.
/// Higher values mean more signal loss.
///
/// # Invariant
/// `value >= 0.0` (loss cannot be negative - that would be gain)
///
/// # Typical Values
/// - 0 dB = no loss (impossible in practice for non-zero distance)
/// - 50-80 dB = indoor/short range
/// - 80-120 dB = outdoor urban
/// - 120-150 dB = long range
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct PathLoss(f32);

impl PathLoss {
    /// Zero loss.
    pub const ZERO: Self = Self(0.0);

    /// Try to create from raw dB value.
    /// Returns None if value is negative or non-finite.
    #[inline]
    pub fn try_new(db: f32) -> Option<Self> {
        (db >= 0.0 && db.is_finite()).then(|| Self(db))
    }

    /// Create from raw dB value, panicking if invalid.
    #[inline]
    pub fn new(db: f32) -> Self {
        Self::try_new(db).expect("PathLoss requires non-negative finite value")
    }

    /// Get the loss value in dB.
    #[inline]
    pub fn as_db(self) -> f32 {
        self.0
    }

    /// Convert to linear attenuation factor (< 1.0).
    ///
    /// This is the factor by which signal power is multiplied.
    /// A 10 dB loss corresponds to a factor of 0.1.
    #[inline]
    pub fn to_linear_factor(self) -> PositiveF32 {
        // Factor = 10^(-dB/10)
        let factor = 10.0_f32.powf(-self.0 / 10.0);
        PositiveF32::new(factor)
    }

    /// Apply this path loss to a power value.
    #[inline]
    pub fn apply(self, power: PositivePower) -> PositivePower {
        let factor = self.to_linear_factor();
        power.scale(factor)
    }

    /// Apply to power in dBm.
    #[inline]
    pub fn apply_dbm(self, power: PowerDbm) -> PowerDbm {
        power.subtract_loss(self)
    }

    /// Add another path loss (losses add in dB).
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Check if loss exceeds a threshold.
    #[inline]
    pub fn exceeds(self, threshold: Self) -> bool {
        self.0 > threshold.0
    }
}

impl std::ops::Add for PathLoss {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

// ============================================================================
// GainDb: Gain (Can Be Negative)
// ============================================================================

/// Gain or loss in dB.
///
/// Unlike PathLoss, this can be negative (representing loss)
/// or positive (representing amplification).
///
/// # Use Cases
/// - Antenna gain
/// - Amplifier gain
/// - Cable loss (negative gain)
/// - System margin calculations
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct GainDb(f32);

impl GainDb {
    /// Zero gain (unity).
    pub const ZERO: Self = Self(0.0);

    /// Try to create from raw dB value.
    #[inline]
    pub fn try_new(db: f32) -> Option<Self> {
        db.is_finite().then(|| Self(db))
    }

    /// Create from raw dB value.
    #[inline]
    pub fn new(db: f32) -> Self {
        Self::try_new(db).expect("GainDb requires finite value")
    }

    /// Create a positive gain.
    #[inline]
    pub fn gain(db: f32) -> Option<Self> {
        (db >= 0.0 && db.is_finite()).then(|| Self(db))
    }

    /// Create a loss (negative gain).
    #[inline]
    pub fn loss(db: f32) -> Option<Self> {
        (db >= 0.0 && db.is_finite()).then(|| Self(-db))
    }

    /// Get the raw dB value.
    #[inline]
    pub fn as_db(self) -> f32 {
        self.0
    }

    /// Check if this represents a gain (>= 0).
    #[inline]
    pub fn is_gain(self) -> bool {
        self.0 >= 0.0
    }

    /// Check if this represents a loss (< 0).
    #[inline]
    pub fn is_loss(self) -> bool {
        self.0 < 0.0
    }

    /// Convert to linear factor.
    #[inline]
    pub fn to_linear(self) -> f32 {
        10.0_f32.powf(self.0 / 10.0)
    }

    /// Add another gain (cascade).
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Negate (convert gain to loss and vice versa).
    #[inline]
    pub fn negate(self) -> Self {
        Self(-self.0)
    }

    /// Convert to PathLoss if this represents a loss.
    #[inline]
    pub fn to_path_loss(self) -> Option<PathLoss> {
        if self.0 <= 0.0 {
            PathLoss::try_new(-self.0)
        } else {
            None
        }
    }
}

impl std::ops::Add for GainDb {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::Neg for GainDb {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

// ============================================================================
// Conversion Utilities
// ============================================================================

/// Convert dB value to linear power ratio.
///
/// This is a general-purpose function for dB conversions.
#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

/// Convert linear power ratio to dB.
///
/// Returns None if the input is not positive.
#[inline]
pub fn linear_to_db(linear: f32) -> Option<f32> {
    if linear > 0.0 {
        Some(10.0 * linear.log10())
    } else {
        None
    }
}

/// Convert dBm to Watts.
#[inline]
pub fn dbm_to_watts(dbm: f32) -> f32 {
    10.0_f32.powf(dbm / 10.0) / 1000.0
}

/// Convert Watts to dBm.
/// Returns the floor value for zero or negative input.
#[inline]
pub fn watts_to_dbm(watts: f32) -> f32 {
    if watts > 1e-30 {
        10.0 * (watts * 1000.0).log10()
    } else {
        -300.0
    }
}

// ============================================================================
// FSPL Constant Verification
// ============================================================================

/// Verified Free Space Path Loss constant.
///
/// FSPL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4π/c)
///
/// The constant is 20*log10(4π/c):
/// c = 299,792,458 m/s
/// 4π/c ≈ 4.188e-8
/// 20*log10(4.188e-8) ≈ -147.55 dB
///
/// This constant is verified at compile time via tests.
pub const FSPL_CONSTANT_DB: f32 = -147.55;

/// Alternative FSPL constant for MHz and km inputs.
/// FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.45
pub const FSPL_CONSTANT_MHZ_KM: f32 = 32.45;

#[cfg(test)]
mod fspl_verification {
    use super::*;
    use crate::constants::{PI, SPEED_OF_LIGHT};

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
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_dbm_construction() {
        assert!(PowerDbm::try_new(0.0).is_some());
        assert!(PowerDbm::try_new(-100.0).is_some());
        assert!(PowerDbm::try_new(50.0).is_some());

        assert!(PowerDbm::try_new(f32::NAN).is_none());
        assert!(PowerDbm::try_new(f32::INFINITY).is_none());
    }

    #[test]
    fn test_power_dbm_conversion_roundtrip() {
        let original = PositivePower::new(0.001); // 1 mW = 0 dBm
        let dbm = PowerDbm::from_watts(original);
        assert!((dbm.as_dbm() - 0.0).abs() < 0.1);

        let back = dbm.to_watts();
        assert!((back.watts() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_power_dbm_reference_points() {
        // 0 dBm = 1 mW
        let p0 = PowerDbm::new(0.0);
        assert!((p0.to_milliwatts().get() - 1.0).abs() < 0.001);

        // 30 dBm = 1 W
        let p30 = PowerDbm::new(30.0);
        assert!((p30.to_watts().watts() - 1.0).abs() < 0.01);

        // -30 dBm = 1 μW
        let pm30 = PowerDbm::new(-30.0);
        assert!((pm30.to_watts().microwatts() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_path_loss_construction() {
        assert!(PathLoss::try_new(0.0).is_some());
        assert!(PathLoss::try_new(100.0).is_some());

        assert!(PathLoss::try_new(-1.0).is_none());
        assert!(PathLoss::try_new(f32::NAN).is_none());
    }

    #[test]
    fn test_path_loss_linear_factor() {
        let loss_10db = PathLoss::new(10.0);
        let factor = loss_10db.to_linear_factor();
        assert!((factor.get() - 0.1).abs() < 0.001);

        let loss_20db = PathLoss::new(20.0);
        let factor = loss_20db.to_linear_factor();
        assert!((factor.get() - 0.01).abs() < 0.0001);
    }

    #[test]
    fn test_path_loss_apply() {
        let power = PositivePower::new(1.0); // 1 W
        let loss = PathLoss::new(10.0); // 10 dB

        let result = loss.apply(power);
        // 10 dB loss means power is reduced by factor of 10
        assert!((result.watts() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_gain_db_construction() {
        assert!(GainDb::try_new(10.0).is_some()); // positive gain
        assert!(GainDb::try_new(-5.0).is_some()); // loss
        assert!(GainDb::try_new(0.0).is_some()); // unity

        assert!(GainDb::try_new(f32::NAN).is_none());
    }

    #[test]
    fn test_gain_db_operations() {
        let gain1 = GainDb::new(10.0);
        let gain2 = GainDb::new(3.0);

        let total = gain1.add(gain2);
        assert!((total.as_db() - 13.0).abs() < 0.001);

        let negated = gain1.negate();
        assert!((negated.as_db() - (-10.0)).abs() < 0.001);
    }

    #[test]
    fn test_gain_to_path_loss() {
        let loss = GainDb::loss(10.0).unwrap();
        assert!(loss.is_loss());
        assert!((loss.as_db() - (-10.0)).abs() < 0.001);

        let path_loss = loss.to_path_loss().unwrap();
        assert!((path_loss.as_db() - 10.0).abs() < 0.001);

        // Positive gain cannot convert to path loss
        let gain = GainDb::gain(5.0).unwrap();
        assert!(gain.to_path_loss().is_none());
    }

    #[test]
    fn test_conversion_utilities() {
        // 10 dB = 10x
        assert!((db_to_linear(10.0) - 10.0).abs() < 0.001);

        // 100x = 20 dB
        assert!((linear_to_db(100.0).unwrap() - 20.0).abs() < 0.001);

        // 0 dBm = 1 mW = 0.001 W
        assert!((dbm_to_watts(0.0) - 0.001).abs() < 1e-6);

        // 1 W = 30 dBm
        assert!((watts_to_dbm(1.0) - 30.0).abs() < 0.1);
    }
}
