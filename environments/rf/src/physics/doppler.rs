//! Doppler Effects - Type-Safe Implementation
//!
//! SIMD-optimized implementations of Doppler shift calculations including:
//! - Standard Doppler shift from relative velocity
//! - Micro-Doppler from rotating parts (e.g., drone propellers)
//! - Coherence time estimation
//!
//! # Type Safety
//!
//! This module uses the Curry-Howard compliant type system:
//! - `RadialVelocity` encodes direction (approaching/receding) in the type
//! - `Hertz` for carrier frequencies (guaranteed positive)
//! - `Seconds` for coherence time (guaranteed non-negative)
//!
//! Legacy raw functions are provided with `_raw` suffix for gradual migration.

#[cfg(feature = "simd")]
use std::simd::{f32x8, num::SimdFloat};

#[cfg(feature = "simd")]
use crate::simd_rf::math::{simd_sin, simd_sqrt};

use crate::constants::SPEED_OF_LIGHT;
use crate::types::{
    dimensional::{Hertz, RadialVelocity, Seconds},
    primitives::PositiveF32,
};

/// Two times Pi
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

// ============================================================================
// Type-Safe Doppler Functions
// ============================================================================

/// Compute Doppler shift with type-safe radial velocity.
///
/// The direction is impossible to get wrong since it's encoded in `RadialVelocity`.
///
/// # Arguments
/// * `velocity` - Radial velocity (Approaching/Receding/Stationary)
/// * `carrier` - Carrier frequency in Hz
///
/// # Returns
/// (shift magnitude, direction) tuple
#[inline]
pub fn calc_doppler_shift(velocity: RadialVelocity, carrier: Hertz) -> (PositiveF32, crate::types::dimensional::DopplerDirection) {
    velocity.doppler_shift(carrier)
}

/// Compute Doppler shift and return the signed Hz value.
///
/// Positive = frequency increase (approaching), negative = decrease (receding).
#[inline]
pub fn calc_doppler_shift_hz(velocity: RadialVelocity, carrier: Hertz) -> f32 {
    let (shift, dir) = velocity.doppler_shift(carrier);
    match dir {
        crate::types::dimensional::DopplerDirection::BlueShift => shift.get(),
        crate::types::dimensional::DopplerDirection::RedShift => -shift.get(),
        crate::types::dimensional::DopplerDirection::None => 0.0,
    }
}

/// Apply Doppler shift to a frequency.
///
/// Returns the shifted frequency, or None if result would be non-positive.
#[inline]
pub fn apply_doppler_shift(freq: Hertz, velocity: RadialVelocity) -> Option<Hertz> {
    velocity.apply_to_frequency(freq)
}

/// Compute coherence time from maximum Doppler frequency.
///
/// The coherence time is the duration over which the channel impulse response
/// is essentially invariant. Beyond this time, channel estimates become stale.
///
/// Formula: T_c ≈ 0.423 / f_d (for 50% correlation level)
#[inline]
pub fn calc_coherence_time(max_doppler: Hertz) -> Seconds {
    Seconds::new(0.423 / max_doppler.as_hz())
}

/// Compute Doppler spread from velocity and carrier.
///
/// The Doppler spread represents the range of Doppler shifts in the channel,
/// determined by the maximum velocity and carrier frequency.
///
/// Formula: B_d = 2 × f_d = 2 × (v/c) × f_c
#[inline]
pub fn calc_doppler_spread(velocity: RadialVelocity, carrier: Hertz) -> PositiveF32 {
    let speed = velocity.speed().get();
    PositiveF32::new(2.0 * (speed / SPEED_OF_LIGHT) * carrier.as_hz())
}

/// Compute radial velocity from 3D positions and velocity.
///
/// Returns a type-safe `RadialVelocity` encoding both magnitude and direction.
#[inline]
pub fn calc_radial_velocity_from_positions(
    tx_pos: (f32, f32, f32),
    tx_vel: (f32, f32, f32),
    rx_pos: (f32, f32, f32),
) -> RadialVelocity {
    let signed = calc_radial_velocity_scalar_raw(tx_pos, tx_vel, rx_pos);
    RadialVelocity::from_signed(signed).unwrap_or(RadialVelocity::Stationary)
}

// ============================================================================
// Legacy Raw SIMD Functions (for backward compatibility)
// ============================================================================

/// Calculate Doppler shift from radial velocity (SIMD, raw f32x8)
///
/// Formula: f_d = (v/c) × f_c
///
/// # Arguments
/// * `radial_velocity` - Velocity component toward/away from receiver (m/s)
///                       Positive = approaching (blue shift), Negative = receding (red shift)
/// * `carrier_freq` - Carrier frequency in Hz
///
/// # Returns
/// Doppler shift in Hz (positive = frequency increase)
#[cfg(feature = "simd")]
#[inline]
pub fn calc_doppler_shift_simd(radial_velocity: f32x8, carrier_freq: f32x8) -> f32x8 {
    (radial_velocity / f32x8::splat(SPEED_OF_LIGHT)) * carrier_freq
}

/// Calculate micro-Doppler from rotating parts (SIMD, raw f32x8)
///
/// Models the Doppler modulation caused by rotating components like drone propellers.
/// The tip velocity creates a sinusoidal modulation of the Doppler shift.
///
/// Formula: f_md = (2 × v_tip / c) × f_c × sin(2π × f_rot × t)
/// where v_tip = 2π × r × f_rot
///
/// # Arguments
/// * `rotation_rate_hz` - Rotation rate in Hz (rotations per second)
/// * `blade_length_m` - Blade/propeller radius in meters
/// * `carrier_freq` - Carrier frequency in Hz
/// * `time` - Current time in seconds
///
/// # Returns
/// Instantaneous micro-Doppler shift in Hz
#[cfg(feature = "simd")]
#[inline]
pub fn calc_micro_doppler_simd(
    rotation_rate_hz: f32x8,
    blade_length_m: f32x8,
    carrier_freq: f32x8,
    time: f32x8,
) -> f32x8 {
    // Tip velocity: v_tip = 2π × r × f_rot
    let tip_velocity = f32x8::splat(TWO_PI) * blade_length_m * rotation_rate_hz;

    // Maximum Doppler: (2 × v_tip / c) × f_c
    let max_doppler =
        (f32x8::splat(2.0) * tip_velocity / f32x8::splat(SPEED_OF_LIGHT)) * carrier_freq;

    // Sinusoidal modulation from blade rotation
    let phase = f32x8::splat(TWO_PI) * rotation_rate_hz * time;
    max_doppler * simd_sin(phase)
}

/// Calculate coherence time of the channel (SIMD, raw f32x8)
///
/// Formula: T_c ≈ 0.423 / f_d (for 50% correlation level)
#[cfg(feature = "simd")]
#[inline]
pub fn calc_coherence_time_simd(max_doppler_hz: f32x8) -> f32x8 {
    // Avoid division by zero
    let safe_doppler = max_doppler_hz.abs().simd_max(f32x8::splat(0.1));
    f32x8::splat(0.423) / safe_doppler
}

/// Calculate coherence bandwidth of the channel (SIMD, raw f32x8)
///
/// Formula: B_c ≈ 1 / (5 × τ_rms) (for frequency correlation > 0.5)
#[cfg(feature = "simd")]
#[inline]
pub fn calc_coherence_bandwidth_simd(rms_delay_spread: f32x8) -> f32x8 {
    let safe_delay = rms_delay_spread.simd_max(f32x8::splat(1e-9));
    f32x8::splat(1.0) / (f32x8::splat(5.0) * safe_delay)
}

/// Calculate radial velocity between transmitter and receiver (SIMD, raw f32x8)
///
/// Projects the velocity vector onto the line connecting TX and RX.
///
/// # Returns
/// Radial velocity in m/s (positive = approaching, negative = receding)
#[cfg(feature = "simd")]
#[inline]
pub fn calc_radial_velocity_simd(
    tx_pos: (f32x8, f32x8, f32x8),
    tx_vel: (f32x8, f32x8, f32x8),
    rx_pos: (f32x8, f32x8, f32x8),
) -> f32x8 {
    // Direction vector from TX to RX
    let dx = rx_pos.0 - tx_pos.0;
    let dy = rx_pos.1 - tx_pos.1;
    let dz = rx_pos.2 - tx_pos.2;

    // Distance
    let dist = simd_sqrt(dx * dx + dy * dy + dz * dz);
    let safe_dist = dist.simd_max(f32x8::splat(1e-6));

    // Unit vector from TX to RX
    let ux = dx / safe_dist;
    let uy = dy / safe_dist;
    let uz = dz / safe_dist;

    // Project velocity onto direction vector
    // Positive when velocity is toward RX (approaching)
    tx_vel.0 * ux + tx_vel.1 * uy + tx_vel.2 * uz
}

/// Calculate Doppler spread from mobile velocity (SIMD, raw f32x8)
///
/// Formula: B_d = 2 × f_d = 2 × (v/c) × f_c
#[cfg(feature = "simd")]
#[inline]
pub fn calc_doppler_spread_simd(velocity: f32x8, carrier_freq: f32x8) -> f32x8 {
    f32x8::splat(2.0) * (velocity / f32x8::splat(SPEED_OF_LIGHT)) * carrier_freq
}

// ============================================================================
// Legacy Raw Scalar Functions
// ============================================================================

/// Scalar Doppler shift calculation (raw f32)
#[inline]
pub fn calc_doppler_shift_scalar_raw(radial_velocity: f32, carrier_freq: f32) -> f32 {
    (radial_velocity / SPEED_OF_LIGHT) * carrier_freq
}

/// Scalar micro-Doppler calculation (raw f32)
#[inline]
pub fn calc_micro_doppler_scalar(
    rotation_rate_hz: f32,
    blade_length_m: f32,
    carrier_freq: f32,
    time: f32,
) -> f32 {
    let tip_velocity = TWO_PI * blade_length_m * rotation_rate_hz;
    let max_doppler = (2.0 * tip_velocity / SPEED_OF_LIGHT) * carrier_freq;
    let phase = TWO_PI * rotation_rate_hz * time;
    max_doppler * phase.sin()
}

/// Scalar coherence time calculation (raw f32)
#[inline]
pub fn calc_coherence_time_scalar_raw(max_doppler_hz: f32) -> f32 {
    let safe_doppler = max_doppler_hz.abs().max(0.1);
    0.423 / safe_doppler
}

/// Scalar radial velocity calculation (raw f32)
///
/// Returns positive for approaching, negative for receding.
#[inline]
pub fn calc_radial_velocity_scalar_raw(
    tx_pos: (f32, f32, f32),
    tx_vel: (f32, f32, f32),
    rx_pos: (f32, f32, f32),
) -> f32 {
    let dx = rx_pos.0 - tx_pos.0;
    let dy = rx_pos.1 - tx_pos.1;
    let dz = rx_pos.2 - tx_pos.2;

    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-6);

    let ux = dx / dist;
    let uy = dy / dist;
    let uz = dz / dist;

    // Positive when velocity is toward RX (approaching)
    tx_vel.0 * ux + tx_vel.1 * uy + tx_vel.2 * uz
}

// ============================================================================
// Backward Compatibility Aliases
// ============================================================================

/// Alias for legacy code compatibility
#[inline]
pub fn calc_doppler_shift_scalar(radial_velocity: f32, carrier_freq: f32) -> f32 {
    calc_doppler_shift_scalar_raw(radial_velocity, carrier_freq)
}

/// Alias for legacy code compatibility
#[inline]
pub fn calc_coherence_time_scalar(max_doppler_hz: f32) -> f32 {
    calc_coherence_time_scalar_raw(max_doppler_hz)
}

/// Alias for legacy code compatibility
#[inline]
pub fn calc_radial_velocity_scalar(
    tx_pos: (f32, f32, f32),
    tx_vel: (f32, f32, f32),
    rx_pos: (f32, f32, f32),
) -> f32 {
    calc_radial_velocity_scalar_raw(tx_pos, tx_vel, rx_pos)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1.0; // 1 Hz tolerance

    #[test]
    fn test_doppler_approaching() {
        // Object approaching at 30 m/s at 1 GHz
        let velocity = 30.0; // ~108 km/h
        let freq = 1e9;

        let doppler = calc_doppler_shift_scalar_raw(velocity, freq);

        // Expected: (30 / 3e8) * 1e9 ≈ 100 Hz
        assert!(
            (doppler - 100.0).abs() < TOLERANCE,
            "Doppler for 30 m/s at 1 GHz: expected ~100 Hz, got {}",
            doppler
        );
    }

    #[test]
    fn test_doppler_receding() {
        // Object receding at 30 m/s at 1 GHz
        let velocity = -30.0;
        let freq = 1e9;

        let doppler = calc_doppler_shift_scalar_raw(velocity, freq);

        assert!(
            (doppler + 100.0).abs() < TOLERANCE,
            "Doppler for -30 m/s at 1 GHz: expected ~-100 Hz, got {}",
            doppler
        );
    }

    #[test]
    fn test_doppler_sign() {
        // Approaching should give positive shift
        let doppler_approaching = calc_doppler_shift_scalar_raw(10.0, 1e9);
        let doppler_receding = calc_doppler_shift_scalar_raw(-10.0, 1e9);

        assert!(
            doppler_approaching > 0.0,
            "Approaching should have positive Doppler"
        );
        assert!(
            doppler_receding < 0.0,
            "Receding should have negative Doppler"
        );
    }

    #[test]
    fn test_micro_doppler_bounds() {
        // Drone propeller: 100 Hz rotation, 0.15m blade, 5.8 GHz
        let rotation = 100.0;
        let blade = 0.15;
        let freq = 5.8e9;

        // Maximum micro-Doppler should be achieved at some time
        // Sample at finer intervals to catch the peak
        let mut max_md = 0.0_f32;
        for i in 0..1000 {
            let t = i as f32 * 0.0001; // 0.1ms steps
            let md = calc_micro_doppler_scalar(rotation, blade, freq, t).abs();
            max_md = max_md.max(md);
        }

        // Expected max: (2 × 2π × 0.15 × 100 / 3e8) × 5.8e9 ≈ 3640 Hz
        let expected_max = (2.0 * TWO_PI * blade * rotation / SPEED_OF_LIGHT) * freq;

        // Allow 5% tolerance for sampling resolution
        assert!(
            (max_md - expected_max).abs() < expected_max * 0.05,
            "Max micro-Doppler: expected ~{}, got {}",
            expected_max,
            max_md
        );
    }

    #[test]
    fn test_coherence_time_inverse() {
        // Higher Doppler = shorter coherence time
        let t_low = calc_coherence_time_scalar_raw(10.0);
        let t_high = calc_coherence_time_scalar_raw(100.0);

        assert!(
            t_low > t_high,
            "Higher Doppler should have shorter coherence: {} vs {}",
            t_low,
            t_high
        );

        // 10x Doppler should give 10x shorter coherence
        let ratio = t_low / t_high;
        assert!(
            (ratio - 10.0).abs() < 0.1,
            "10x Doppler ratio should give 10x coherence ratio: {}",
            ratio
        );
    }

    #[test]
    fn test_radial_velocity_approaching() {
        // TX moving toward RX
        let tx_pos = (0.0, 0.0, 0.0);
        let tx_vel = (10.0, 0.0, 0.0); // Moving in +x
        let rx_pos = (100.0, 0.0, 0.0); // RX is in +x direction

        let v_r = calc_radial_velocity_scalar_raw(tx_pos, tx_vel, rx_pos);

        assert!(
            (v_r - 10.0).abs() < 0.01,
            "Radial velocity should be 10 m/s approaching: {}",
            v_r
        );
    }

    #[test]
    fn test_radial_velocity_perpendicular() {
        // TX moving perpendicular to line to RX
        let tx_pos = (0.0, 0.0, 0.0);
        let tx_vel = (0.0, 10.0, 0.0); // Moving in +y
        let rx_pos = (100.0, 0.0, 0.0); // RX is in +x direction

        let v_r = calc_radial_velocity_scalar_raw(tx_pos, tx_vel, rx_pos);

        assert!(
            v_r.abs() < 0.01,
            "Perpendicular motion should have zero radial velocity: {}",
            v_r
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_doppler_simd_matches_scalar() {
        let velocities = [10.0, 20.0, -15.0, 30.0, -5.0, 0.0, 100.0, -100.0];
        let freq = 1e9;

        let v_simd = f32x8::from_array(velocities);
        let f_simd = f32x8::splat(freq);
        let simd_result = calc_doppler_shift_simd(v_simd, f_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &v) in velocities.iter().enumerate() {
            let scalar_result = calc_doppler_shift_scalar_raw(v, freq);
            assert!(
                (simd_arr[i] - scalar_result).abs() < 0.1,
                "Doppler SIMD/scalar mismatch: {} vs {}",
                simd_arr[i],
                scalar_result
            );
        }
    }

    // ========================================================================
    // Type-Safe Tests
    // ========================================================================

    #[test]
    fn test_type_safe_doppler_approaching() {
        let carrier = Hertz::from_ghz(1.0).unwrap();
        let velocity = RadialVelocity::approaching(30.0).unwrap();

        let (shift, dir) = calc_doppler_shift(velocity, carrier);

        // Expected: ~100 Hz
        assert!((shift.get() - 100.0).abs() < TOLERANCE);
        assert_eq!(dir, crate::types::dimensional::DopplerDirection::BlueShift);
    }

    #[test]
    fn test_type_safe_doppler_receding() {
        let carrier = Hertz::from_ghz(1.0).unwrap();
        let velocity = RadialVelocity::receding(30.0).unwrap();

        let (shift, dir) = calc_doppler_shift(velocity, carrier);

        // Expected: ~100 Hz magnitude
        assert!((shift.get() - 100.0).abs() < TOLERANCE);
        assert_eq!(dir, crate::types::dimensional::DopplerDirection::RedShift);
    }

    #[test]
    fn test_type_safe_doppler_hz() {
        let carrier = Hertz::from_ghz(1.0).unwrap();

        let approaching = RadialVelocity::approaching(30.0).unwrap();
        let receding = RadialVelocity::receding(30.0).unwrap();

        let hz_approach = calc_doppler_shift_hz(approaching, carrier);
        let hz_recede = calc_doppler_shift_hz(receding, carrier);

        assert!(hz_approach > 0.0, "Approaching should give positive Hz");
        assert!(hz_recede < 0.0, "Receding should give negative Hz");
        assert!((hz_approach - 100.0).abs() < TOLERANCE);
        assert!((hz_recede + 100.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_type_safe_apply_doppler() {
        // Use 1 MHz carrier for f32 precision (1e6 fits in f32 with room for shift)
        let carrier = Hertz::from_mhz(1.0).unwrap(); // 1 MHz = 1e6 Hz
        let velocity = RadialVelocity::approaching(300.0).unwrap(); // 300 m/s for 1 Hz shift

        let shifted = apply_doppler_shift(carrier, velocity).unwrap();

        // Shifted frequency should be higher (blue shift)
        assert!(shifted.as_hz() > carrier.as_hz());

        // Expected shift at 1 MHz with 300 m/s: (300/c) * 1e6 = 1.0 Hz
        let expected_shift = 300.0 / SPEED_OF_LIGHT * 1e6;
        let actual_shift = shifted.as_hz() - carrier.as_hz();
        assert!(
            (actual_shift - expected_shift).abs() < 0.01,
            "Expected ~{} Hz shift, got {} Hz",
            expected_shift,
            actual_shift
        );
    }

    #[test]
    fn test_type_safe_coherence_time() {
        let max_doppler = Hertz::new(100.0);
        let coherence = calc_coherence_time(max_doppler);

        // T_c ≈ 0.423 / 100 = 0.00423 seconds
        assert!((coherence.as_s() - 0.00423).abs() < 0.0001);
    }

    #[test]
    fn test_type_safe_doppler_spread() {
        let carrier = Hertz::from_ghz(1.0).unwrap();
        let velocity = RadialVelocity::approaching(30.0).unwrap();

        let spread = calc_doppler_spread(velocity, carrier);

        // B_d = 2 × 100 = 200 Hz
        assert!((spread.get() - 200.0).abs() < 2.0 * TOLERANCE);
    }

    #[test]
    fn test_type_safe_radial_velocity_conversion() {
        let tx_pos = (0.0, 0.0, 0.0);
        let tx_vel = (10.0, 0.0, 0.0);
        let rx_pos = (100.0, 0.0, 0.0);

        let v = calc_radial_velocity_from_positions(tx_pos, tx_vel, rx_pos);

        assert!(matches!(v, RadialVelocity::Approaching(_)));
        assert!((v.to_signed() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_type_safe_stationary() {
        let carrier = Hertz::from_ghz(1.0).unwrap();
        let velocity = RadialVelocity::Stationary;

        let (shift, dir) = calc_doppler_shift(velocity, carrier);

        assert!(shift.get() < 1e-6);
        assert_eq!(dir, crate::types::dimensional::DopplerDirection::None);
    }
}
