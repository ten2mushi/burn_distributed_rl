//! Specialized Entity Behaviors
//!
//! Behavior implementations for specialized/low-power entities:
//! - **Scheduled**: Timer-based periodic transmission (Smart Meter)
//! - **Reactive**: Response to external triggers (RFID)
//! - **AC Modulated**: 60 Hz power-line coupled interference (Microwave Oven)
//! - **Burst OOK**: Very short On-Off Keying bursts (Key Fob)
//!
//! All behaviors support both scalar and SIMD-8 variants.

#[cfg(feature = "simd")]
use std::simd::f32x8;

use super::{EntitySoA, ContinuousBehavior, BurstyBehavior, PeriodicBehavior};
use crate::config::RFWorldConfig;
use crate::types::frequency::ValidatedFrequencyGrid;

#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

// ============================================================================
// Scheduled Behavior (Smart Meter)
// ============================================================================

/// Scheduled transmission behavior for Smart Meters.
///
/// Smart meters transmit at fixed intervals (hourly/daily) with very short
/// bursts. This differs from periodic in that the transmission window is
/// scheduled rather than continuous pulsing.
pub struct ScheduledBehavior;

impl ScheduledBehavior {
    /// Update a scheduled entity (scalar).
    ///
    /// Smart meters have very long intervals between transmissions.
    /// The `pulse_timer` field is used to track time since last transmission.
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        config: &RFWorldConfig,
    ) {
        // Use periodic behavior with very long intervals
        // The transmission interval is controlled by PRI in the entity config
        PeriodicBehavior::update_scalar(entities, env, entity, dt, config);
    }

    /// Update scheduled entities for a SIMD batch.
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        config: &RFWorldConfig,
    ) {
        PeriodicBehavior::update_simd(entities, batch, entity, dt, config);
    }

    /// Render a smart meter's FSK signal.
    pub fn render_fsk_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        // Smart meter uses narrow FSK (~100 kHz at 900 MHz)
        // Render as narrow Gaussian similar to FM
        ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
    }
}

// ============================================================================
// Reactive Behavior (RFID)
// ============================================================================

/// Reactive backscatter behavior for RFID tags.
///
/// RFID tags only respond when interrogated by a reader. The tag modulates
/// the incoming carrier via backscatter (load modulation). In simulation,
/// we model this as very short bursty responses.
pub struct ReactiveBehavior;

impl ReactiveBehavior {
    /// Update a reactive entity (scalar).
    ///
    /// RFID uses bursty behavior with very low duty cycle.
    /// The entity only transmits when an interrogator is present.
    #[cfg(feature = "simd")]
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        // Use bursty behavior with very short bursts
        BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
    }

    /// Update reactive entities for a SIMD batch.
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        BurstyBehavior::update_simd(entities, batch, entity, dt, rng, config);
    }

    /// Render an RFID tag's backscatter signal.
    ///
    /// RFID backscatter is very narrow and weak, appearing as OOK.
    pub fn render_ook_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        // RFID backscatter is very narrow OOK, render as narrow Gaussian
        ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
    }
}

// ============================================================================
// AC-Modulated Behavior (Microwave Oven)
// ============================================================================

/// AC power-line modulated interference behavior for Microwave Ovens.
///
/// Microwave ovens produce interference at 2.45 GHz that is amplitude
/// modulated by the 60 Hz (50 Hz in EU) AC power cycle. The magnetron
/// only produces RF during the positive half of the AC cycle.
pub struct ACModulatedBehavior;

impl ACModulatedBehavior {
    /// Calculate the 60 Hz modulation factor.
    ///
    /// Returns a value between 0.0 and 1.0 representing the instantaneous
    /// power output of the magnetron based on AC cycle position.
    #[inline]
    pub fn ac_modulation_factor(time: f32, ac_freq_hz: f32) -> f32 {
        // Magnetron only fires during positive half of AC cycle
        // Use |sin(2*pi*f*t)| for full-wave rectified appearance
        let phase = 2.0 * std::f32::consts::PI * ac_freq_hz * time;
        phase.sin().abs()
    }

    /// Update an AC-modulated entity (scalar).
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        config: &RFWorldConfig,
    ) {
        // Use continuous behavior - power modulation happens during render
        ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
    }

    /// Update AC-modulated entities for a SIMD batch.
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        config: &RFWorldConfig,
    ) {
        ContinuousBehavior::update_simd(entities, batch, entity, dt, config);
    }

    /// Render a microwave oven's messy interference.
    ///
    /// Microwave ovens have a wide, drifting spectral signature centered
    /// around 2.45 GHz with ~20 MHz effective bandwidth.
    pub fn render_messy_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        // Microwave produces wide, messy interference
        // Use TV-style rectangular render for wide bandwidth
        ContinuousBehavior::render_tv_scalar(entities, psd, grid, env, entity, config);
    }
}

// ============================================================================
// Burst OOK Behavior (Key Fob)
// ============================================================================

/// Burst On-Off Keying behavior for Key Fobs.
///
/// Key fobs transmit very short (<100 ms) OOK bursts at 315/433 MHz
/// when activated. The burst contains a rolling code or fixed code.
pub struct BurstOOKBehavior;

impl BurstOOKBehavior {
    /// Update a burst OOK entity (scalar).
    #[cfg(feature = "simd")]
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        // Use bursty behavior with very short burst duration
        BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
    }

    /// Update burst OOK entities for a SIMD batch.
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        BurstyBehavior::update_simd(entities, batch, entity, dt, rng, config);
    }

    /// Render a key fob's OOK burst.
    ///
    /// Key fobs use very narrow OOK modulation at 315/433 MHz.
    pub fn render_ook_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        // OOK burst is very narrow, render as narrow Gaussian
        ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac_modulation_factor() {
        // At t=0, sin(0) = 0, abs = 0
        let factor_0 = ACModulatedBehavior::ac_modulation_factor(0.0, 60.0);
        assert!(factor_0 < 0.01, "At t=0, factor should be near 0");

        // At t = 1/(4*60) = quarter period, sin = 1
        let factor_quarter = ACModulatedBehavior::ac_modulation_factor(1.0 / (4.0 * 60.0), 60.0);
        assert!((factor_quarter - 1.0).abs() < 0.01, "At quarter period, factor should be 1");

        // Factor should always be in [0, 1]
        for i in 0..100 {
            let t = i as f32 * 0.001; // 0 to 100 ms
            let factor = ACModulatedBehavior::ac_modulation_factor(t, 60.0);
            assert!(factor >= 0.0 && factor <= 1.0, "Factor must be in [0, 1]");
        }
    }

    #[test]
    fn test_ac_modulation_50hz() {
        // Test 50 Hz (EU) variant
        let factor_quarter = ACModulatedBehavior::ac_modulation_factor(1.0 / (4.0 * 50.0), 50.0);
        assert!((factor_quarter - 1.0).abs() < 0.01, "At quarter period, factor should be 1");
    }
}
