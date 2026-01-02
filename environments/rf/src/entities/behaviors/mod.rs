//! Entity Behaviors
//!
//! Trait definitions and implementations for entity temporal and spectral behavior.
//!
//! # Behavior Categories
//!
//! - **Continuous**: Always active, phase accumulation only
//! - **Bursty**: Timer-based Poisson activation
//! - **Periodic**: Deterministic pulse trains
//! - **Mobile**: Position updates with Doppler effects
//! - **Spread Spectrum**: FHSS and DSSS logic
//!
//! All behaviors use type-safe `ValidatedFrequencyGrid` for frequency operations.

#[cfg(feature = "simd")]
use std::simd::f32x8;

use super::{EntitySoA, EntityType};
use crate::config::RFWorldConfig;
use crate::types::frequency::ValidatedFrequencyGrid;

#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

// Behavior modules
mod continuous;
mod bursty;
mod periodic;
mod mobile;
mod spread_spectrum;
mod specialized;

// Re-exports
pub use continuous::ContinuousBehavior;
pub use bursty::BurstyBehavior;
pub use periodic::PeriodicBehavior;
pub use mobile::MobileBehavior;
pub use spread_spectrum::SpreadSpectrumBehavior;
pub use specialized::{ScheduledBehavior, ReactiveBehavior, ACModulatedBehavior, BurstOOKBehavior};

// ============================================================================
// Behavior Dispatch
// ============================================================================

/// Update all entities for a single environment.
///
/// This function dispatches to the appropriate behavior based on entity type.
#[cfg(feature = "simd")]
pub fn update_all_entities(
    entities: &mut EntitySoA,
    env: usize,
    dt: f32,
    rng: &mut SimdRng,
    config: &RFWorldConfig,
) {
    let max_entities = entities.max_entities();

    for entity in 0..max_entities {
        if !entities.is_active(env, entity) {
            continue;
        }

        let entity_type = entities.get_type(env, entity);

        match entity_type {
            // Continuous entities
            EntityType::TVStation
            | EntityType::FMRadio
            | EntityType::AMRadio
            | EntityType::GNSS
            | EntityType::GeoSatellite => {
                ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // Bursty entities (basic)
            EntityType::LTETower | EntityType::WiFiAP | EntityType::LTEUplink => {
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // Bluetooth (bursty + FHSS)
            EntityType::Bluetooth => {
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
                SpreadSpectrumBehavior::update_fhss_scalar(entities, env, entity, dt, config);
            }

            // Zigbee (bursty + DSSS)
            EntityType::Zigbee => {
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
                // DSSS doesn't need frequency updates like FHSS
            }

            // LoRaWAN (bursty + chirp spread)
            EntityType::LoRaWAN => {
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // Satellite IoT (slotted ALOHA)
            EntityType::SatelliteIoT => {
                // Bursty with slotted random access
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // Periodic entities (radar + aviation)
            EntityType::SBandRadar
            | EntityType::WeatherRadar
            | EntityType::LBandRadar
            | EntityType::ADSB
            | EntityType::DME => {
                PeriodicBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // Mobile entities
            EntityType::DroneAnalog | EntityType::DroneDigital | EntityType::Vehicle => {
                MobileBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // Voice radios (PTT + mobile behavior)
            EntityType::MaritimeVHF
            | EntityType::WalkieTalkie
            | EntityType::GMRS
            | EntityType::P25Radio
            | EntityType::AmateurRadio => {
                // PTT radios use bursty behavior for transmission timing
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
                // Also update position (mobile)
                MobileBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // ================================================================
            // Specialized / Low-Power entities
            // ================================================================

            // Smart Meter (scheduled periodic transmission)
            EntityType::SmartMeter => {
                // Uses periodic behavior for scheduled transmissions
                PeriodicBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // Microwave Oven (60 Hz cyclic interference)
            EntityType::MicrowaveOven => {
                // Continuous when active, no behavioral updates needed
                ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // RFID Tag (reactive backscatter)
            EntityType::RFIDTag => {
                // Uses bursty behavior for reactive responses
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // Key Fob (OOK burst)
            EntityType::KeyFob => {
                // Very short bursty transmissions
                BurstyBehavior::update_scalar(entities, env, entity, dt, rng, config);
            }

            // Wireless Mic (continuous narrow FM)
            EntityType::WirelessMic => {
                // Continuous transmission during active periods
                ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // Baby Monitor (continuous FM/FHSS)
            EntityType::BabyMonitor => {
                // Continuous transmission
                ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // ================================================================
            // Sprint 7: Exotic entities
            // ================================================================

            // Weather Balloon (altitude-varying GFSK telemetry)
            EntityType::WeatherBalloon => {
                // Continuous GFSK transmission, altitude affects propagation
                ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
            }

            // Deep Space Network (ultra-weak CW)
            EntityType::DeepSpaceNetwork => {
                // Continuous weak CW beacon from deep space
                ContinuousBehavior::update_scalar(entities, env, entity, dt, config);
            }
        }
    }
}

/// Update entities for a SIMD batch of 8 environments.
///
/// More efficient than per-environment updates when processing many entities.
#[cfg(feature = "simd")]
pub fn update_all_entities_simd(
    entities: &mut EntitySoA,
    batch: usize,
    dt: f32,
    rng: &mut SimdRng,
    config: &RFWorldConfig,
) {
    let max_entities = entities.max_entities();
    let dt_simd = f32x8::splat(dt);

    for entity in 0..max_entities {
        // Check if any entity in the batch is active for this slot
        let active = entities.load_active_simd(batch, entity);
        if !active.iter().any(|&a| a) {
            continue;
        }

        // Get entity type from first active environment in batch
        // (assumes all envs have same entity type at same index)
        let base_env = batch * 8;
        let entity_type = entities.get_type(base_env, entity);

        match entity_type {
            // Continuous entities
            EntityType::TVStation
            | EntityType::FMRadio
            | EntityType::AMRadio
            | EntityType::GNSS
            | EntityType::GeoSatellite => {
                ContinuousBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // Bursty entities (basic)
            EntityType::LTETower | EntityType::WiFiAP | EntityType::LTEUplink => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // Bluetooth (bursty + FHSS)
            EntityType::Bluetooth => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
                SpreadSpectrumBehavior::update_fhss_simd(entities, batch, entity, dt_simd, config);
            }

            // Zigbee (bursty + DSSS)
            EntityType::Zigbee => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // LoRaWAN (bursty + chirp spread)
            EntityType::LoRaWAN => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // Satellite IoT (slotted ALOHA)
            EntityType::SatelliteIoT => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // Periodic entities (radar + aviation)
            EntityType::SBandRadar
            | EntityType::WeatherRadar
            | EntityType::LBandRadar
            | EntityType::ADSB
            | EntityType::DME => {
                PeriodicBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // Mobile entities
            EntityType::DroneAnalog | EntityType::DroneDigital | EntityType::Vehicle => {
                MobileBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // Voice radios (PTT + mobile behavior)
            EntityType::MaritimeVHF
            | EntityType::WalkieTalkie
            | EntityType::GMRS
            | EntityType::P25Radio
            | EntityType::AmateurRadio => {
                // PTT radios use bursty behavior for transmission timing
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
                // Also update position (mobile)
                MobileBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // ================================================================
            // Specialized / Low-Power entities
            // ================================================================

            // Smart Meter (scheduled periodic transmission)
            EntityType::SmartMeter => {
                PeriodicBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // Microwave Oven (60 Hz cyclic interference)
            EntityType::MicrowaveOven => {
                ContinuousBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // RFID Tag (reactive backscatter)
            EntityType::RFIDTag => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // Key Fob (OOK burst)
            EntityType::KeyFob => {
                BurstyBehavior::update_simd(entities, batch, entity, dt_simd, rng, config);
            }

            // Wireless Mic (continuous narrow FM)
            EntityType::WirelessMic => {
                ContinuousBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // Baby Monitor (continuous FM/FHSS)
            EntityType::BabyMonitor => {
                ContinuousBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // ================================================================
            // Sprint 7: Exotic entities
            // ================================================================

            // Weather Balloon (altitude-varying GFSK telemetry)
            EntityType::WeatherBalloon => {
                ContinuousBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }

            // Deep Space Network (ultra-weak CW)
            EntityType::DeepSpaceNetwork => {
                ContinuousBehavior::update_simd(entities, batch, entity, dt_simd, config);
            }
        }
    }
}

/// Render all entities to the PSD for a single environment.
pub fn render_all_entities(
    entities: &EntitySoA,
    psd: &mut [f32],
    grid: &ValidatedFrequencyGrid,
    env: usize,
    config: &RFWorldConfig,
) {
    let max_entities = entities.max_entities();

    for entity in 0..max_entities {
        if !entities.is_active(env, entity) {
            continue;
        }

        let entity_type = entities.get_type(env, entity);

        // Render based on entity type
        match entity_type {
            EntityType::TVStation => {
                ContinuousBehavior::render_tv_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::FMRadio => {
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::AMRadio => {
                // AM uses narrower Gaussian similar to FM but at MF frequencies
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::GNSS => {
                // GNSS is spread spectrum (DSSS), renders as broadband below noise
                SpreadSpectrumBehavior::render_dsss_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::GeoSatellite => {
                // GeoSat beacon renders as narrow Gaussian (like FM)
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::LTETower | EntityType::WiFiAP => {
                BurstyBehavior::render_ofdm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::LTEUplink => {
                // SC-FDM similar to OFDM but with sinc envelope
                BurstyBehavior::render_ofdm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::Bluetooth => {
                SpreadSpectrumBehavior::render_fhss_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::Zigbee => {
                // Zigbee uses DSSS
                SpreadSpectrumBehavior::render_dsss_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::LoRaWAN => {
                // LoRa uses chirp spread spectrum
                PeriodicBehavior::render_chirp_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::SatelliteIoT => {
                // Satellite IoT is narrow GFSK, render as narrow Gaussian
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::SBandRadar | EntityType::WeatherRadar | EntityType::LBandRadar => {
                // All radars use chirp
                PeriodicBehavior::render_chirp_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::ADSB | EntityType::DME => {
                // ADS-B and DME are narrow pulses, render as narrow Gaussian
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::DroneAnalog => {
                MobileBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::DroneDigital | EntityType::Vehicle => {
                MobileBehavior::render_ofdm_scalar(entities, psd, grid, env, entity, config);
            }

            // Voice radios - FM-based render as Gaussian (narrow FM)
            EntityType::MaritimeVHF
            | EntityType::WalkieTalkie
            | EntityType::GMRS
            | EntityType::AmateurRadio => {
                // Analog FM voice radios render as narrow Gaussian
                MobileBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            EntityType::P25Radio => {
                // P25 C4FM digital - narrow GFSK, render as narrow Gaussian
                MobileBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // ================================================================
            // Specialized / Low-Power entities
            // ================================================================

            // Smart Meter (900 MHz FSK)
            EntityType::SmartMeter => {
                // FSK renders as narrow Gaussian (100 kHz)
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // Microwave Oven (2.45 GHz messy interference)
            EntityType::MicrowaveOven => {
                // Wide, messy Gaussian (20 MHz bandwidth)
                // Power fluctuates with 60 Hz magnetron cycling
                ContinuousBehavior::render_tv_scalar(entities, psd, grid, env, entity, config);
            }

            // RFID Tag (backscatter OOK)
            EntityType::RFIDTag => {
                // Very narrow OOK burst, render as narrow Gaussian
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // Key Fob (OOK/ASK burst)
            EntityType::KeyFob => {
                // Very narrow OOK burst at 315/433 MHz
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // Wireless Mic (narrow FM in UHF TV gaps)
            EntityType::WirelessMic => {
                // 200 kHz narrow FM
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // Baby Monitor (continuous FM or FHSS)
            EntityType::BabyMonitor => {
                // Narrow FM audio transmission
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // ================================================================
            // Sprint 7: Exotic entities
            // ================================================================

            // Weather Balloon (400 MHz GFSK radiosonde)
            EntityType::WeatherBalloon => {
                // GFSK telemetry, render as narrow Gaussian
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }

            // Deep Space Network (ultra-weak 2 GHz CW)
            EntityType::DeepSpaceNetwork => {
                // Extremely narrow CW beacon, render as very narrow Gaussian
                ContinuousBehavior::render_fm_scalar(entities, psd, grid, env, entity, config);
            }
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::dimensional::Hertz;

    #[test]
    fn test_validated_frequency_grid() {
        let grid = ValidatedFrequencyGrid::from_params(1e9, 2e9, 1024).unwrap();

        // Test frequency to bin
        assert_eq!(grid.freq_to_bin(Hertz::new(1e9)), 0);
        assert_eq!(grid.freq_to_bin(Hertz::new(1.5e9)), 512);
        assert_eq!(grid.freq_to_bin(Hertz::new(2e9)), 1023);

        // Test in range
        assert!(grid.in_range(Hertz::new(1.5e9)));
        assert!(!grid.in_range(Hertz::new(0.5e9)));
        assert!(!grid.in_range(Hertz::new(2.5e9)));
    }
}
