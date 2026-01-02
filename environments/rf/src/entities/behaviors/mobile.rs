//! Mobile Entity Behavior
//!
//! Implements behavior for moving entities with Doppler effects:
//! - Analog video drones
//! - Digital video drones
//! - Vehicles (V2X)
//!
//! Mobile entities use Despawn/Respawn at world boundaries:
//! when an entity exits the world, it deactivates and respawns
//! at a random edge position after a short delay.

#[cfg(feature = "simd")]
use std::simd::{cmp::SimdPartialOrd, f32x8, Mask};

use super::ValidatedFrequencyGrid;
use crate::types::dimensional::Hertz;
use crate::config::RFWorldConfig;
use crate::constants;
use crate::entities::{entity_idx, EntitySoA, EntityType};

#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

/// Mobile behavior implementation for moving entities.
pub struct MobileBehavior;

// ============================================================================
// Constants
// ============================================================================

/// Minimum respawn delay (seconds)
const RESPAWN_DELAY_MIN: f32 = 0.5;
/// Maximum respawn delay (seconds)
const RESPAWN_DELAY_MAX: f32 = 2.0;

/// Drone speed range (m/s)
const DRONE_SPEED_MIN: f32 = 5.0;
const DRONE_SPEED_MAX: f32 = 20.0;

/// Vehicle speed range (m/s)
const VEHICLE_SPEED_MIN: f32 = 10.0;
const VEHICLE_SPEED_MAX: f32 = 30.0;

impl MobileBehavior {
    // ========================================================================
    // Parameter Helpers
    // ========================================================================

    /// Get speed range for an entity type.
    fn get_speed_range(entity_type: EntityType) -> (f32, f32) {
        match entity_type {
            EntityType::DroneAnalog | EntityType::DroneDigital => (DRONE_SPEED_MIN, DRONE_SPEED_MAX),
            EntityType::Vehicle => (VEHICLE_SPEED_MIN, VEHICLE_SPEED_MAX),
            _ => (5.0, 15.0),
        }
    }

    // ========================================================================
    // Scalar Updates
    // ========================================================================

    /// Update a mobile entity (scalar version).
    ///
    /// - Euler integration for position
    /// - Boundary check with despawn
    /// - Respawn timer management
    #[cfg(feature = "simd")]
    pub fn update_scalar(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        dt: f32,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        // Check if entity is waiting to respawn
        if entities.respawn_timer[idx] > 0.0 {
            entities.respawn_timer[idx] -= dt;

            // Check if respawn timer expired
            if entities.respawn_timer[idx] <= 0.0 {
                Self::respawn_entity(entities, env, entity, rng, config);
            }
            return;
        }

        // Euler integration
        entities.x[idx] += entities.vx[idx] * dt;
        entities.y[idx] += entities.vy[idx] * dt;
        entities.z[idx] += entities.vz[idx] * dt;

        // Boundary check
        let (world_x, world_y, world_z) = config.world_size;
        let out_of_bounds = entities.x[idx] < 0.0
            || entities.x[idx] > world_x
            || entities.y[idx] < 0.0
            || entities.y[idx] > world_y
            || entities.z[idx] < 0.0
            || entities.z[idx] > world_z;

        if out_of_bounds {
            // Despawn: set respawn timer
            let u = rng.uniform();
            let u_arr: [f32; 8] = u.into();
            entities.respawn_timer[idx] =
                RESPAWN_DELAY_MIN + u_arr[0] * (RESPAWN_DELAY_MAX - RESPAWN_DELAY_MIN);

            // Mark as temporarily inactive (phase = 0)
            entities.phase[idx] = 0.0;
        } else {
            // Active and in bounds
            entities.phase[idx] = 1.0;
        }
    }

    /// Respawn an entity at a random world edge.
    #[cfg(feature = "simd")]
    fn respawn_entity(
        entities: &mut EntitySoA,
        env: usize,
        entity: usize,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());
        let entity_type = EntityType::from(entities.entity_type[idx]);
        let (world_x, world_y, world_z) = config.world_size;

        // Sample random values
        let u = rng.uniform();
        let u_arr: [f32; 8] = u.into();

        // Choose edge (0=left, 1=right, 2=bottom, 3=top)
        let edge = ((u_arr[0] * 4.0) as usize) % 4;

        // Position along edge
        let edge_pos = u_arr[1];

        // Set position based on edge
        match edge {
            0 => {
                // Left edge
                entities.x[idx] = 0.0;
                entities.y[idx] = edge_pos * world_y;
            }
            1 => {
                // Right edge
                entities.x[idx] = world_x;
                entities.y[idx] = edge_pos * world_y;
            }
            2 => {
                // Bottom edge
                entities.x[idx] = edge_pos * world_x;
                entities.y[idx] = 0.0;
            }
            _ => {
                // Top edge
                entities.x[idx] = edge_pos * world_x;
                entities.y[idx] = world_y;
            }
        }

        // Set altitude (drones can be higher, vehicles on ground)
        entities.z[idx] = if entity_type == EntityType::Vehicle {
            0.0
        } else {
            u_arr[2] * world_z
        };

        // Set velocity toward center (±45° from perpendicular)
        let (speed_min, speed_max) = Self::get_speed_range(entity_type);
        let speed = speed_min + u_arr[3] * (speed_max - speed_min);

        // Angle offset from perpendicular (-45° to +45°)
        let angle_offset = (u_arr[4] - 0.5) * constants::PI / 2.0;

        // Base angle toward center
        let center_x = world_x / 2.0;
        let center_y = world_y / 2.0;
        let dx = center_x - entities.x[idx];
        let dy = center_y - entities.y[idx];
        let base_angle = dy.atan2(dx);

        let final_angle = base_angle + angle_offset;

        entities.vx[idx] = speed * final_angle.cos();
        entities.vy[idx] = speed * final_angle.sin();
        entities.vz[idx] = if entity_type == EntityType::Vehicle {
            0.0
        } else {
            (u_arr[5] - 0.5) * 2.0 // Small vertical component
        };

        // Clear respawn timer and mark as active
        entities.respawn_timer[idx] = 0.0;
        entities.phase[idx] = 1.0;
    }

    // ========================================================================
    // SIMD Updates
    // ========================================================================

    /// Update mobile entities for 8 environments (SIMD version).
    #[cfg(feature = "simd")]
    pub fn update_simd(
        entities: &mut EntitySoA,
        batch: usize,
        entity: usize,
        dt: f32x8,
        rng: &mut SimdRng,
        config: &RFWorldConfig,
    ) {
        let base_env = batch * 8;
        let stride = entities.max_entities();

        // Load positions and velocities
        let (x, y, z) = entities.load_position_simd(batch, entity);
        let (vx, vy, vz) = entities.load_velocity_simd(batch, entity);

        // Load respawn timers
        let respawn_timer = f32x8::from_array([
            entities.respawn_timer[entity_idx(base_env + 0, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 1, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 2, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 3, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 4, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 5, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 6, entity, stride)],
            entities.respawn_timer[entity_idx(base_env + 7, entity, stride)],
        ]);

        // Check if waiting to respawn
        let zero = f32x8::splat(0.0);
        let waiting: Mask<i32, 8> = respawn_timer.simd_gt(zero);

        // Euler integration (only for non-waiting entities)
        let new_x = waiting.select(x, x + vx * dt);
        let new_y = waiting.select(y, y + vy * dt);
        let new_z = waiting.select(z, z + vz * dt);

        // Boundary check
        let (world_x, world_y, world_z) = config.world_size;
        let world_x_simd = f32x8::splat(world_x);
        let world_y_simd = f32x8::splat(world_y);
        let world_z_simd = f32x8::splat(world_z);

        let out_x = new_x.simd_lt(zero) | new_x.simd_gt(world_x_simd);
        let out_y = new_y.simd_lt(zero) | new_y.simd_gt(world_y_simd);
        let out_z = new_z.simd_lt(zero) | new_z.simd_gt(world_z_simd);
        let out_of_bounds = out_x | out_y | out_z;

        // Store updated positions
        entities.store_position_simd(batch, entity, new_x, new_y, new_z);

        // Handle respawn timer and despawn logic per-lane
        // (Complex state transitions require scalar fallback)
        for lane in 0..8 {
            let idx = entity_idx(base_env + lane, entity, stride);

            if entities.respawn_timer[idx] > 0.0 {
                entities.respawn_timer[idx] -= dt.to_array()[lane];
                if entities.respawn_timer[idx] <= 0.0 {
                    Self::respawn_entity(entities, base_env + lane, entity, rng, config);
                }
            } else if out_of_bounds.to_array()[lane] {
                // Despawn
                let u = rng.uniform();
                let u_arr: [f32; 8] = u.into();
                entities.respawn_timer[idx] =
                    RESPAWN_DELAY_MIN + u_arr[lane] * (RESPAWN_DELAY_MAX - RESPAWN_DELAY_MIN);
                entities.phase[idx] = 0.0;
            } else {
                entities.phase[idx] = 1.0;
            }
        }
    }

    // ========================================================================
    // Rendering
    // ========================================================================

    /// Render FM video signal to PSD (scalar version).
    ///
    /// Used for analog drone video with micro-Doppler sidebands.
    pub fn render_fm_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        // Check if active (not waiting to respawn)
        if entities.phase[idx] < 0.5 {
            return;
        }

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx];
        let power_dbm = entities.power_dbm[idx];

        // Get grid bounds using type-safe API
        let freq_min = grid.freq_min().as_hz();
        let freq_max = grid.freq_max().as_hz();
        let resolution = grid.resolution().as_hz();

        // Calculate Doppler shift based on radial velocity
        // Assume receiver at world center
        let (world_x, world_y, world_z) = config.world_size;
        let rx_x = world_x / 2.0;
        let rx_y = world_y / 2.0;
        let rx_z = world_z / 2.0;

        let dx = rx_x - entities.x[idx];
        let dy = rx_y - entities.y[idx];
        let dz = rx_z - entities.z[idx];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt() + 1e-10;

        // Unit vector toward receiver
        let ux = dx / dist;
        let uy = dy / dist;
        let uz = dz / dist;

        // Radial velocity (positive = approaching)
        let v_radial = entities.vx[idx] * ux + entities.vy[idx] * uy + entities.vz[idx] * uz;

        // Doppler shift
        let doppler = (v_radial / constants::SPEED_OF_LIGHT) * center_freq;
        let shifted_freq = center_freq + doppler;

        // Check if signal is within grid range
        let low_freq = shifted_freq - bandwidth / 2.0;
        let high_freq = shifted_freq + bandwidth / 2.0;

        if high_freq < freq_min || low_freq > freq_max {
            return;
        }

        // Convert power from dBm to linear
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;

        // FM video has characteristic shape with carrier + sidebands
        // Simplified as Gaussian
        let sigma = bandwidth / 4.0;
        let sqrt_2pi = (2.0 * constants::PI).sqrt();
        let gaussian_norm = power_linear / (sigma * sqrt_2pi);

        let psd_offset = env * config.num_freq_bins;
        let extend = bandwidth;
        let bin_start = grid.freq_to_bin(Hertz::new((shifted_freq - extend).max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new((shifted_freq + extend).min(freq_max)));

        for bin in bin_start..=bin_end {
            let bin_freq = grid.bin_to_freq(bin).as_hz();
            let diff = bin_freq - shifted_freq;
            let exponent = -0.5 * (diff / sigma).powi(2);
            let gaussian_value = gaussian_norm * exponent.exp() * resolution;

            psd[psd_offset + bin] += gaussian_value;
        }
    }

    /// Render OFDM signal to PSD with Doppler effects (scalar version).
    ///
    /// Used for digital drones and V2X. Doppler causes ICI (Inter-Carrier Interference).
    pub fn render_ofdm_scalar(
        entities: &EntitySoA,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        env: usize,
        entity: usize,
        config: &RFWorldConfig,
    ) {
        let idx = entity_idx(env, entity, entities.max_entities());

        // Check if active
        if entities.phase[idx] < 0.5 {
            return;
        }

        let center_freq = entities.center_freq[idx];
        let bandwidth = entities.bandwidth[idx];
        let power_dbm = entities.power_dbm[idx];

        // Get grid bounds using type-safe API
        let freq_min = grid.freq_min().as_hz();
        let freq_max = grid.freq_max().as_hz();
        let resolution = grid.resolution().as_hz();

        // Calculate Doppler shift
        let (world_x, world_y, world_z) = config.world_size;
        let rx_x = world_x / 2.0;
        let rx_y = world_y / 2.0;
        let rx_z = world_z / 2.0;

        let dx = rx_x - entities.x[idx];
        let dy = rx_y - entities.y[idx];
        let dz = rx_z - entities.z[idx];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt() + 1e-10;

        let ux = dx / dist;
        let uy = dy / dist;
        let uz = dz / dist;

        let v_radial = entities.vx[idx] * ux + entities.vy[idx] * uy + entities.vz[idx] * uz;
        let doppler = (v_radial / constants::SPEED_OF_LIGHT) * center_freq;
        let shifted_freq = center_freq + doppler;

        // Check if signal is within grid range
        let low_freq = shifted_freq - bandwidth / 2.0;
        let high_freq = shifted_freq + bandwidth / 2.0;

        if high_freq < freq_min || low_freq > freq_max {
            return;
        }

        // Convert power from dBm to linear
        let power_linear = 10.0_f32.powf(power_dbm / 10.0) / 1000.0;

        // OFDM with ICI spreading - model as slightly wider than nominal
        // ICI increases with Doppler, causing spectral spreading
        let speed = (entities.vx[idx].powi(2)
            + entities.vy[idx].powi(2)
            + entities.vz[idx].powi(2))
        .sqrt();
        let ici_factor = 1.0 + speed / 100.0; // Slight spreading with speed
        let effective_bw = (bandwidth * 0.9 * ici_factor).min(bandwidth * 1.2);

        let power_per_hz = power_linear / effective_bw;
        let power_per_bin = power_per_hz * resolution;

        let bin_start = grid.freq_to_bin(Hertz::new((shifted_freq - effective_bw / 2.0).max(freq_min)));
        let bin_end = grid.freq_to_bin(Hertz::new((shifted_freq + effective_bw / 2.0).min(freq_max)));

        let psd_offset = env * config.num_freq_bins;
        for bin in bin_start..=bin_end {
            psd[psd_offset + bin] += power_per_bin;
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::ModulationType;

    fn test_config() -> RFWorldConfig {
        RFWorldConfig::new()
            .with_num_envs(8)
            .with_freq_bins(512)
            .with_freq_range(5e9, 6e9)
            .with_world_size(1000.0, 1000.0, 100.0)
            .build()
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_mobile_position_update() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);
        let mut rng = SimdRng::new(42);

        entities.set_entity(
            0, 0,
            EntityType::DroneDigital,
            ModulationType::OFDM,
            500.0, 500.0, 50.0, // Center of world
            5.8e9, 20e6, 25.0,
        );
        entities.set_velocity(0, 0, 10.0, 0.0, 0.0); // Moving +x at 10 m/s

        let idx = entities.idx(0, 0);
        let initial_x = entities.x[idx];

        // Update for 1 second
        for _ in 0..100 {
            MobileBehavior::update_scalar(&mut entities, 0, 0, 0.01, &mut rng, &config);
        }

        // Should have moved 10m in x direction
        let final_x = entities.x[idx];
        assert!((final_x - initial_x - 10.0).abs() < 0.1, "Entity should move with velocity");
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_mobile_boundary_despawn() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);
        let mut rng = SimdRng::new(42);

        // Place near right edge, moving right
        entities.set_entity(
            0, 0,
            EntityType::DroneDigital,
            ModulationType::OFDM,
            999.0, 500.0, 50.0, // Near right edge
            5.8e9, 20e6, 25.0,
        );
        entities.set_velocity(0, 0, 100.0, 0.0, 0.0); // Fast +x

        let idx = entities.idx(0, 0);

        // Update until entity crosses boundary
        for _ in 0..10 {
            MobileBehavior::update_scalar(&mut entities, 0, 0, 0.01, &mut rng, &config);
        }

        // Should have despawned (respawn_timer > 0 or respawned)
        // Either the entity crossed and got respawn timer, or already respawned
        let respawn_timer = entities.respawn_timer[idx];
        let x = entities.x[idx];

        // Either waiting to respawn OR already respawned at an edge
        let waiting_or_respawned = respawn_timer > 0.0 || x <= 0.1 || x >= 999.9
            || entities.y[idx] <= 0.1 || entities.y[idx] >= 999.9;

        assert!(waiting_or_respawned, "Entity should despawn at boundary");
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_mobile_respawn_after_delay() {
        let config = test_config();
        let mut entities = EntitySoA::new(8, 64);
        let mut rng = SimdRng::new(42);

        entities.set_entity(
            0, 0,
            EntityType::DroneDigital,
            ModulationType::OFDM,
            500.0, 500.0, 50.0,
            5.8e9, 20e6, 25.0,
        );

        let idx = entities.idx(0, 0);

        // Manually set respawn timer
        entities.respawn_timer[idx] = 0.5;
        entities.phase[idx] = 0.0;

        // Update for 0.6 seconds (should respawn after 0.5s)
        for _ in 0..60 {
            MobileBehavior::update_scalar(&mut entities, 0, 0, 0.01, &mut rng, &config);
        }

        // Should have respawned (phase = 1, respawn_timer = 0)
        assert!(entities.phase[idx] > 0.5, "Entity should be active after respawn");
        assert_eq!(entities.respawn_timer[idx], 0.0, "Respawn timer should be cleared");
    }

    #[test]
    fn test_doppler_shift() {
        // Test that Doppler calculation is correct
        let speed = 20.0; // m/s
        let freq = 5.8e9; // Hz

        let doppler = (speed / constants::SPEED_OF_LIGHT) * freq;

        // Expected: (20 / 3e8) * 5.8e9 ≈ 387 Hz
        assert!((doppler - 387.0).abs() < 1.0, "Doppler shift calculation");
    }

    #[test]
    fn test_ofdm_render_with_doppler() {
        let config = test_config();
        let entities = {
            let mut e = EntitySoA::new(8, 64);
            e.set_entity(
                0, 0,
                EntityType::DroneDigital,
                ModulationType::OFDM,
                600.0, 500.0, 50.0, // Offset from center
                5.8e9, 20e6, 25.0,
            );
            e.set_velocity(0, 0, 50.0, 0.0, 0.0); // Moving right
            let idx = e.idx(0, 0);
            e.phase[idx] = 1.0; // Active
            e
        };

        let grid = config.validated_grid().expect("Valid grid");
        let mut psd = vec![0.0; config.num_envs * config.num_freq_bins];

        MobileBehavior::render_ofdm_scalar(&entities, &mut psd, &grid, 0, 0, &config);

        // Check that power is added
        let center_bin = grid.freq_to_bin(Hertz::new(5.8e9));
        assert!(psd[center_bin] > 0.0 || psd[center_bin - 1] > 0.0 || psd[center_bin + 1] > 0.0,
            "OFDM should add power near center frequency");
    }
}
