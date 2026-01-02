//! Dimensional Types
//!
//! Types with physical dimensions that enforce unit correctness
//! and prevent invalid operations at compile time.

use super::primitives::{NonZeroPositiveF32, PositiveF32};
use crate::constants::SPEED_OF_LIGHT;

// ============================================================================
// Hertz: Frequency
// ============================================================================

/// Frequency in Hertz, guaranteed positive.
///
/// Frequency must always be positive in RF applications.
/// This type wraps NonZeroPositiveF32 to enforce this invariant.
///
/// # Common Constructors
/// - [`Hertz::from_hz`] - From raw Hz
/// - [`Hertz::from_khz`] - From kHz
/// - [`Hertz::from_mhz`] - From MHz
/// - [`Hertz::from_ghz`] - From GHz
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Hertz(NonZeroPositiveF32);

impl Hertz {
    /// Try to create from raw Hz value.
    #[inline]
    pub fn try_new(hz: f32) -> Option<Self> {
        NonZeroPositiveF32::try_new(hz).map(Self)
    }

    /// Create from Hz, panicking if invalid.
    #[inline]
    pub fn new(hz: f32) -> Self {
        Self::try_new(hz).expect("Hertz requires positive finite value")
    }

    /// Create from Hz.
    #[inline]
    pub fn from_hz(hz: f32) -> Option<Self> {
        Self::try_new(hz)
    }

    /// Create from kHz.
    #[inline]
    pub fn from_khz(khz: f32) -> Option<Self> {
        Self::try_new(khz * 1e3)
    }

    /// Create from MHz.
    #[inline]
    pub fn from_mhz(mhz: f32) -> Option<Self> {
        Self::try_new(mhz * 1e6)
    }

    /// Create from GHz.
    #[inline]
    pub fn from_ghz(ghz: f32) -> Option<Self> {
        Self::try_new(ghz * 1e9)
    }

    /// Get value in Hz.
    #[inline]
    pub fn as_hz(self) -> f32 {
        self.0.get()
    }

    /// Get value in kHz.
    #[inline]
    pub fn as_khz(self) -> f32 {
        self.0.get() / 1e3
    }

    /// Get value in MHz.
    #[inline]
    pub fn as_mhz(self) -> f32 {
        self.0.get() / 1e6
    }

    /// Get value in GHz.
    #[inline]
    pub fn as_ghz(self) -> f32 {
        self.0.get() / 1e9
    }

    /// Compute wavelength in meters.
    #[inline]
    pub fn wavelength(self) -> Meters {
        Meters::new(SPEED_OF_LIGHT / self.0.get())
    }

    /// Log base 10 of frequency (for path loss calculations).
    #[inline]
    pub fn log10(self) -> f32 {
        self.0.log10()
    }

    /// Get the inner NonZeroPositiveF32.
    #[inline]
    pub fn inner(self) -> NonZeroPositiveF32 {
        self.0
    }

    /// Add two frequencies.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(NonZeroPositiveF32::new(self.0.get() + other.0.get()))
    }

    /// Subtract frequencies (returns None if result would be non-positive).
    #[inline]
    pub fn sub(self, other: Self) -> Option<Self> {
        let diff = self.0.get() - other.0.get();
        Self::try_new(diff)
    }

    /// Scale by a positive factor.
    #[inline]
    pub fn scale(self, factor: PositiveF32) -> Option<Self> {
        Self::try_new(self.0.get() * factor.get())
    }
}

// ============================================================================
// Meters: Distance
// ============================================================================

/// Distance in meters, guaranteed non-negative.
///
/// Unlike frequency, distance can be zero (e.g., same location).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Meters(f32);

impl Meters {
    /// Zero distance.
    pub const ZERO: Self = Self(0.0);

    /// One meter.
    pub const ONE: Self = Self(1.0);

    /// Try to create from raw meters.
    #[inline]
    pub fn try_new(m: f32) -> Option<Self> {
        (m >= 0.0 && m.is_finite()).then(|| Self(m))
    }

    /// Create from meters, panicking if invalid.
    #[inline]
    pub fn new(m: f32) -> Self {
        Self::try_new(m).expect("Meters requires non-negative finite value")
    }

    /// Create from kilometers.
    #[inline]
    pub fn from_km(km: f32) -> Option<Self> {
        Self::try_new(km * 1000.0)
    }

    /// Get value in meters.
    #[inline]
    pub fn as_m(self) -> f32 {
        self.0
    }

    /// Get value in kilometers.
    #[inline]
    pub fn as_km(self) -> f32 {
        self.0 / 1000.0
    }

    /// Log base 10 of distance (for path loss calculations).
    /// Returns None if distance is zero.
    #[inline]
    pub fn log10(self) -> Option<f32> {
        if self.0 > 0.0 {
            Some(self.0.log10())
        } else {
            None
        }
    }

    /// Square of distance (useful for inverse square law).
    #[inline]
    pub fn squared(self) -> PositiveF32 {
        PositiveF32::new(self.0 * self.0)
    }

    /// Add two distances.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Subtract distances (returns None if result would be negative).
    #[inline]
    pub fn sub(self, other: Self) -> Option<Self> {
        Self::try_new(self.0 - other.0)
    }

    /// Check if distance is effectively zero.
    #[inline]
    pub fn is_negligible(self, threshold: Self) -> bool {
        self.0 < threshold.0
    }

    /// Convert to f32.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

// ============================================================================
// Seconds: Time Duration
// ============================================================================

/// Time duration in seconds, guaranteed non-negative.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Seconds(f32);

impl Seconds {
    /// Zero duration.
    pub const ZERO: Self = Self(0.0);

    /// One second.
    pub const ONE: Self = Self(1.0);

    /// Try to create from raw seconds.
    #[inline]
    pub fn try_new(s: f32) -> Option<Self> {
        (s >= 0.0 && s.is_finite()).then(|| Self(s))
    }

    /// Create from seconds, panicking if invalid.
    #[inline]
    pub fn new(s: f32) -> Self {
        Self::try_new(s).expect("Seconds requires non-negative finite value")
    }

    /// Create from milliseconds.
    #[inline]
    pub fn from_ms(ms: f32) -> Option<Self> {
        Self::try_new(ms * 0.001)
    }

    /// Create from microseconds.
    #[inline]
    pub fn from_us(us: f32) -> Option<Self> {
        Self::try_new(us * 1e-6)
    }

    /// Get value in seconds.
    #[inline]
    pub fn as_s(self) -> f32 {
        self.0
    }

    /// Get value in milliseconds.
    #[inline]
    pub fn as_ms(self) -> f32 {
        self.0 * 1000.0
    }

    /// Get value in microseconds.
    #[inline]
    pub fn as_us(self) -> f32 {
        self.0 * 1e6
    }

    /// Convert to frequency (inverse).
    /// Returns None if duration is zero.
    #[inline]
    pub fn to_frequency(self) -> Option<Hertz> {
        if self.0 > 0.0 {
            Hertz::try_new(1.0 / self.0)
        } else {
            None
        }
    }

    /// Add two durations.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Convert to f32.
    #[inline]
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

// ============================================================================
// RadialVelocity: Direction-Aware Velocity
// ============================================================================

/// Direction of Doppler shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DopplerDirection {
    /// Frequency increases (source approaching receiver).
    BlueShift,
    /// Frequency decreases (source receding from receiver).
    RedShift,
    /// No shift (stationary).
    None,
}

/// Radial velocity with explicit direction semantics.
///
/// This type makes it impossible to get the Doppler sign wrong
/// by encoding direction in the type itself.
///
/// # Convention
/// - `Approaching`: Source moving toward receiver (positive Doppler shift)
/// - `Receding`: Source moving away from receiver (negative Doppler shift)
/// - `Stationary`: No relative motion (zero Doppler shift)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadialVelocity {
    /// Moving toward receiver (blue shift).
    Approaching(NonZeroPositiveF32),
    /// Moving away from receiver (red shift).
    Receding(NonZeroPositiveF32),
    /// Stationary (no Doppler).
    Stationary,
}

impl RadialVelocity {
    /// Create from signed velocity (positive = approaching).
    #[inline]
    pub fn from_signed(velocity_mps: f32) -> Option<Self> {
        if !velocity_mps.is_finite() {
            return None;
        }

        if velocity_mps > 0.0 {
            NonZeroPositiveF32::try_new(velocity_mps).map(Self::Approaching)
        } else if velocity_mps < 0.0 {
            NonZeroPositiveF32::try_new(-velocity_mps).map(Self::Receding)
        } else {
            Some(Self::Stationary)
        }
    }

    /// Create approaching velocity.
    #[inline]
    pub fn approaching(speed_mps: f32) -> Option<Self> {
        NonZeroPositiveF32::try_new(speed_mps).map(Self::Approaching)
    }

    /// Create receding velocity.
    #[inline]
    pub fn receding(speed_mps: f32) -> Option<Self> {
        NonZeroPositiveF32::try_new(speed_mps).map(Self::Receding)
    }

    /// Create stationary (zero velocity).
    #[inline]
    pub fn stationary() -> Self {
        Self::Stationary
    }

    /// Get signed velocity (positive = approaching).
    #[inline]
    pub fn to_signed(self) -> f32 {
        match self {
            Self::Approaching(v) => v.get(),
            Self::Receding(v) => -v.get(),
            Self::Stationary => 0.0,
        }
    }

    /// Get speed magnitude (always non-negative).
    #[inline]
    pub fn speed(self) -> PositiveF32 {
        match self {
            Self::Approaching(v) | Self::Receding(v) => v.to_positive(),
            Self::Stationary => PositiveF32::ZERO,
        }
    }

    /// Get Doppler direction.
    #[inline]
    pub fn direction(self) -> DopplerDirection {
        match self {
            Self::Approaching(_) => DopplerDirection::BlueShift,
            Self::Receding(_) => DopplerDirection::RedShift,
            Self::Stationary => DopplerDirection::None,
        }
    }

    /// Compute Doppler shift for a given carrier frequency.
    ///
    /// Returns the shift magnitude and direction, not the shifted frequency.
    #[inline]
    pub fn doppler_shift(self, carrier: Hertz) -> (PositiveF32, DopplerDirection) {
        let speed = self.speed().get();
        let shift_hz = (speed / SPEED_OF_LIGHT) * carrier.as_hz();

        (PositiveF32::new(shift_hz), self.direction())
    }

    /// Apply Doppler shift to a frequency.
    ///
    /// Returns the shifted frequency.
    #[inline]
    pub fn apply_to_frequency(self, freq: Hertz) -> Option<Hertz> {
        let (shift, direction) = self.doppler_shift(freq);
        let base = freq.as_hz();

        match direction {
            DopplerDirection::BlueShift => Hertz::try_new(base + shift.get()),
            DopplerDirection::RedShift => Hertz::try_new(base - shift.get()),
            DopplerDirection::None => Some(freq),
        }
    }

    /// Compute the Doppler factor (1 + v/c for approaching, 1 - v/c for receding).
    #[inline]
    pub fn doppler_factor(self) -> PositiveF32 {
        let factor = match self {
            Self::Approaching(v) => 1.0 + v.get() / SPEED_OF_LIGHT,
            Self::Receding(v) => 1.0 - v.get() / SPEED_OF_LIGHT,
            Self::Stationary => 1.0,
        };
        // Factor should always be positive for non-relativistic speeds
        PositiveF32::new(factor.max(0.0))
    }
}

impl Default for RadialVelocity {
    fn default() -> Self {
        Self::Stationary
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hertz_construction() {
        assert!(Hertz::from_hz(1.0).is_some());
        assert!(Hertz::from_mhz(100.0).is_some());
        assert!(Hertz::from_ghz(2.4).is_some());

        assert!(Hertz::from_hz(0.0).is_none());
        assert!(Hertz::from_hz(-1.0).is_none());
    }

    #[test]
    fn test_hertz_conversions() {
        let f = Hertz::from_ghz(2.4).unwrap();
        assert!((f.as_hz() - 2.4e9).abs() < 1.0);
        assert!((f.as_mhz() - 2400.0).abs() < 0.001);
        assert!((f.as_ghz() - 2.4).abs() < 1e-6);
    }

    #[test]
    fn test_hertz_wavelength() {
        let f = Hertz::from_ghz(3.0).unwrap();
        let wavelength = f.wavelength();
        // c = f * λ => λ = c/f = 299792458 / 3e9 ≈ 0.1 m
        assert!((wavelength.as_m() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_meters_construction() {
        assert!(Meters::try_new(0.0).is_some());
        assert!(Meters::try_new(1000.0).is_some());
        assert!(Meters::from_km(1.0).is_some());

        assert!(Meters::try_new(-1.0).is_none());
    }

    #[test]
    fn test_meters_operations() {
        let d1 = Meters::new(100.0);
        let d2 = Meters::new(50.0);

        assert!((d1.add(d2).as_m() - 150.0).abs() < 1e-6);
        assert!((d1.sub(d2).unwrap().as_m() - 50.0).abs() < 1e-6);
        assert!(d2.sub(d1).is_none()); // Would be negative
    }

    #[test]
    fn test_seconds_construction() {
        assert!(Seconds::try_new(0.0).is_some());
        assert!(Seconds::from_ms(100.0).is_some());
        assert!(Seconds::from_us(1000.0).is_some());

        assert!(Seconds::try_new(-1.0).is_none());
    }

    #[test]
    fn test_seconds_to_frequency() {
        let t = Seconds::new(0.001); // 1 ms
        let f = t.to_frequency().unwrap();
        assert!((f.as_hz() - 1000.0).abs() < 0.1);

        assert!(Seconds::ZERO.to_frequency().is_none());
    }

    #[test]
    fn test_radial_velocity_from_signed() {
        let v_approach = RadialVelocity::from_signed(10.0).unwrap();
        assert!(matches!(v_approach, RadialVelocity::Approaching(_)));
        assert!((v_approach.to_signed() - 10.0).abs() < 1e-6);

        let v_recede = RadialVelocity::from_signed(-10.0).unwrap();
        assert!(matches!(v_recede, RadialVelocity::Receding(_)));
        assert!((v_recede.to_signed() - (-10.0)).abs() < 1e-6);

        let v_stat = RadialVelocity::from_signed(0.0).unwrap();
        assert!(matches!(v_stat, RadialVelocity::Stationary));
    }

    #[test]
    fn test_radial_velocity_doppler() {
        let carrier = Hertz::from_ghz(1.0).unwrap(); // 1 GHz

        // Approaching at 30 m/s
        let v = RadialVelocity::approaching(30.0).unwrap();
        let (shift, dir) = v.doppler_shift(carrier);

        // Doppler shift ≈ v/c * f = 30/3e8 * 1e9 = 100 Hz
        assert!((shift.get() - 100.0).abs() < 1.0);
        assert_eq!(dir, DopplerDirection::BlueShift);

        // Apply to frequency
        let shifted = v.apply_to_frequency(carrier).unwrap();
        assert!(shifted.as_hz() > carrier.as_hz()); // Blue shift increases frequency
    }

    #[test]
    fn test_radial_velocity_doppler_factor() {
        let v_approach = RadialVelocity::approaching(30.0).unwrap();
        let factor = v_approach.doppler_factor();
        // Factor should be > 1 for approaching
        assert!(factor.get() > 1.0);
        assert!((factor.get() - (1.0 + 30.0 / SPEED_OF_LIGHT)).abs() < 1e-9);

        let v_recede = RadialVelocity::receding(30.0).unwrap();
        let factor = v_recede.doppler_factor();
        // Factor should be < 1 for receding
        assert!(factor.get() < 1.0);
    }
}
