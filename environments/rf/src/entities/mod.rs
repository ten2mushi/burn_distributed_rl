//! Entity System
//!
//! This module implements background RF entities that create a realistic
//! electromagnetic environment for cognitive radio and jammer agent training.
//!
//! # Entity Categories
//!
//! - **Continuous**: TV, FM - always active, 100% duty cycle
//! - **Bursty**: LTE, WiFi, Bluetooth - stochastic Poisson arrivals
//! - **Periodic**: Radar - deterministic pulse trains
//! - **Mobile**: Drones, Vehicles - moving with Doppler effects
//!
//! # Architecture
//!
//! Entities use a Struct-of-Arrays (SoA) layout for cache efficiency
//! and SIMD-friendly memory access patterns.

// Entity type definitions
mod entity_types;
pub use entity_types::{EntityType, ModulationType};

// Entity state storage (SoA layout)
mod entity_soa;
pub use entity_soa::EntitySoA;

// Entity spawning
mod spawner;
pub use spawner::EntitySpawner;

// Entity behaviors
pub mod behaviors;

// Drone protocol variants and micro-Doppler
pub mod drone;
pub use drone::{
    DroneControlProtocol, DroneRFConfig, DroneTelemetryProtocol, DroneVideoProtocol,
};

// ============================================================================
// Entity Index Utilities
// ============================================================================

/// Calculate the flat index for an entity in a SoA array.
///
/// # Arguments
/// * `env` - Environment index (0..num_envs)
/// * `entity` - Entity index within environment (0..max_entities)
/// * `max_entities` - Maximum entities per environment
///
/// # Returns
/// Flat index = env * max_entities + entity
#[inline]
pub const fn entity_idx(env: usize, entity: usize, max_entities: usize) -> usize {
    env * max_entities + entity
}

/// Calculate the base index for a SIMD batch of 8 environments.
///
/// # Arguments
/// * `batch` - SIMD batch index (0..num_envs/8)
/// * `entity` - Entity index within environment (0..max_entities)
/// * `max_entities` - Maximum entities per environment
///
/// # Returns
/// Base index for the first environment in the batch
#[inline]
pub const fn simd_entity_base(batch: usize, entity: usize, max_entities: usize) -> usize {
    batch * 8 * max_entities + entity * 8
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_idx() {
        let max_entities = 64;

        // First environment, first entity
        assert_eq!(entity_idx(0, 0, max_entities), 0);

        // First environment, second entity
        assert_eq!(entity_idx(0, 1, max_entities), 1);

        // Second environment, first entity
        assert_eq!(entity_idx(1, 0, max_entities), 64);

        // Third environment, fifth entity
        assert_eq!(entity_idx(2, 4, max_entities), 132);
    }

    #[test]
    fn test_simd_entity_base() {
        let max_entities = 64;

        // First batch, first entity
        assert_eq!(simd_entity_base(0, 0, max_entities), 0);

        // First batch, second entity (8 environments per batch)
        assert_eq!(simd_entity_base(0, 1, max_entities), 8);

        // Second batch (envs 8-15), first entity
        assert_eq!(simd_entity_base(1, 0, max_entities), 512);
    }
}
