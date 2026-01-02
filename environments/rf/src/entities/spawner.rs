//! Entity Spawning System
//!
//! Configurable entity spawning with builder pattern for entity counts
//! and type-specific parameter initialization.

use super::{EntitySoA, EntityType, ModulationType};
use crate::config::EntityConfig;
use crate::constants;

// ============================================================================
// Entity Spawner
// ============================================================================

/// Spawns entities with type-specific parameters.
///
/// The spawner uses a simple pseudo-random number generator based on
/// a linear congruential generator (LCG) for reproducibility.
pub struct EntitySpawner {
    /// Entity configuration
    config: EntityConfig,
    /// World bounds (x, y, z)
    world_size: (f32, f32, f32),
    /// RNG state for reproducibility
    rng_state: u64,
}

impl EntitySpawner {
    /// Create a new spawner from configuration.
    pub fn new(config: EntityConfig, world_size: (f32, f32, f32), seed: u64) -> Self {
        Self {
            config,
            world_size,
            rng_state: seed,
        }
    }

    /// Get a reference to the entity configuration.
    pub fn config(&self) -> &EntityConfig {
        &self.config
    }

    /// Spawn all entities for a single environment.
    ///
    /// Returns the number of entities spawned.
    pub fn spawn_all(&mut self, entities: &mut EntitySoA, env: usize) -> usize {
        let mut entity_idx = 0;

        // ====================================================================
        // Continuous entities
        // ====================================================================
        for _ in 0..self.config.num_tv_stations {
            self.spawn_tv_station(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_fm_radios {
            self.spawn_fm_radio(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_am_radios {
            self.spawn_am_radio(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_gnss {
            self.spawn_gnss(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_geo_satellites {
            self.spawn_geo_satellite(entities, env, entity_idx);
            entity_idx += 1;
        }

        // ====================================================================
        // Bursty entities
        // ====================================================================
        for _ in 0..self.config.num_lte_towers {
            self.spawn_lte_tower(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_wifi_aps {
            self.spawn_wifi_ap(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_bluetooth {
            self.spawn_bluetooth(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_lte_uplink {
            self.spawn_lte_uplink(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_zigbee {
            self.spawn_zigbee(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_lorawan {
            self.spawn_lorawan(entities, env, entity_idx);
            entity_idx += 1;
        }

        // ====================================================================
        // Periodic entities
        // ====================================================================
        for _ in 0..self.config.num_sband_radars {
            self.spawn_sband_radar(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_weather_radars {
            self.spawn_weather_radar(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_lband_radars {
            self.spawn_lband_radar(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_adsb {
            self.spawn_adsb(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_dme {
            self.spawn_dme(entities, env, entity_idx);
            entity_idx += 1;
        }

        // ====================================================================
        // Mobile entities
        // ====================================================================
        for _ in 0..self.config.num_drone_analog {
            self.spawn_drone_analog(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_drone_digital {
            self.spawn_drone_digital(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_vehicles {
            self.spawn_vehicle(entities, env, entity_idx);
            entity_idx += 1;
        }

        // ====================================================================
        // Voice radios (PTT behavior)
        // ====================================================================
        for _ in 0..self.config.num_maritime_vhf {
            self.spawn_maritime_vhf(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_walkie_talkie {
            self.spawn_walkie_talkie(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_gmrs {
            self.spawn_gmrs(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_p25 {
            self.spawn_p25(entities, env, entity_idx);
            entity_idx += 1;
        }

        for _ in 0..self.config.num_amateur_radio {
            self.spawn_amateur_radio(entities, env, entity_idx);
            entity_idx += 1;
        }

        entity_idx
    }

    // ========================================================================
    // RNG Helpers
    // ========================================================================

    /// Generate next random u64 using LCG.
    fn next_u64(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.rng_state = self.rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state
    }

    /// Generate uniform random f32 in [0, 1).
    fn rand_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate uniform random f32 in [min, max).
    fn rand_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.rand_f32() * (max - min)
    }

    /// Generate random position within world bounds.
    fn rand_position(&mut self) -> (f32, f32, f32) {
        (
            self.rand_range(0.0, self.world_size.0),
            self.rand_range(0.0, self.world_size.1),
            self.rand_range(0.0, self.world_size.2),
        )
    }

    /// Generate random velocity for mobile entities.
    fn rand_velocity(&mut self, speed_min: f32, speed_max: f32) -> (f32, f32, f32) {
        // Random direction
        let theta = self.rand_range(0.0, constants::TWO_PI);
        let phi = self.rand_range(-0.3, 0.3); // Mostly horizontal

        let speed = self.rand_range(speed_min, speed_max);

        let cos_phi = phi.cos();
        (
            speed * theta.cos() * cos_phi,
            speed * theta.sin() * cos_phi,
            speed * phi.sin(),
        )
    }

    // ========================================================================
    // Entity Type Spawners
    // ========================================================================

    /// Spawn a TV station entity.
    fn spawn_tv_station(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // TV channels: 470-608 MHz, 6 MHz each
        // Pick a random channel (approximately 23 channels)
        let channel = (self.next_u64() % 23) as f32;
        let center_freq = constants::TV_UHF_START + channel * 6e6 + 3e6;

        entities.set_entity(
            env,
            entity,
            EntityType::TVStation,
            ModulationType::COFDM,
            x,
            y,
            z,
            center_freq,
            6e6, // 6 MHz bandwidth
            self.rand_range(40.0, 50.0), // 40-50 dBm
        );
    }

    /// Spawn an FM radio station entity.
    fn spawn_fm_radio(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // FM radio: 88-108 MHz, 200 kHz channels
        let channel = (self.next_u64() % 100) as f32;
        let center_freq = constants::FM_RADIO_START + channel * 200e3;

        entities.set_entity(
            env,
            entity,
            EntityType::FMRadio,
            ModulationType::FM,
            x,
            y,
            z,
            center_freq,
            200e3, // 200 kHz bandwidth
            self.rand_range(30.0, 45.0), // 30-45 dBm
        );
    }

    /// Spawn an AM radio station entity.
    fn spawn_am_radio(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // AM radio: 535-1705 kHz (MF band), 10 kHz channels
        let channel = (self.next_u64() % 117) as f32; // ~117 channels
        let center_freq = 535e3 + channel * 10e3;

        entities.set_entity(
            env,
            entity,
            EntityType::AMRadio,
            ModulationType::AM,
            x,
            y,
            z,
            center_freq,
            10e3, // 10 kHz bandwidth
            self.rand_range(60.0, 77.0), // 60-77 dBm (1 kW - 50 kW typical)
        );
    }

    /// Spawn a GNSS entity (aggregate of GPS/Galileo/GLONASS satellites).
    ///
    /// GNSS is conceptually at ~20,200 km altitude, but for simulation purposes
    /// we place it at the top of the world bounds. The signal strength is what
    /// matters (-130 dBm is already the surface-level power after path loss).
    fn spawn_gnss(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        // GNSS is always overhead, position doesn't matter much for signal
        // Use center of world at max altitude (top of world bounds)
        let x = self.world_size.0 / 2.0;
        let y = self.world_size.1 / 2.0;
        let z = self.world_size.2; // Top of simulation volume

        // GPS L1: 1575.42 MHz, C/A code bandwidth ~2 MHz
        let center_freq = 1575.42e6;

        entities.set_entity(
            env,
            entity,
            EntityType::GNSS,
            ModulationType::DSSS,
            x,
            y,
            z,
            center_freq,
            2.046e6, // C/A code bandwidth
            -130.0,  // Very weak at surface (~-130 dBm after atmospheric path loss)
        );
    }

    /// Spawn an LTE tower entity.
    fn spawn_lte_tower(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // LTE bands: 700, 850, 1900, 2100 MHz (simplified)
        let bands = [700e6, 850e6, 1900e6, 2100e6];
        let band_idx = (self.next_u64() % 4) as usize;
        let center_freq = bands[band_idx] + self.rand_range(-5e6, 5e6);

        // Bandwidth: 10 or 20 MHz
        let bandwidth = if self.rand_f32() > 0.5 {
            constants::LTE_BW_20
        } else {
            constants::LTE_BW_10
        };

        entities.set_entity(
            env,
            entity,
            EntityType::LTETower,
            ModulationType::OFDM,
            x,
            y,
            z,
            center_freq,
            bandwidth,
            self.rand_range(43.0, 46.0), // 43-46 dBm (typical macro cell)
        );

        // Initialize with random timer for bursty behavior
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.01); // Random initial phase
    }

    /// Spawn a WiFi access point entity.
    fn spawn_wifi_ap(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // WiFi: 2.4 GHz or 5 GHz
        let is_5ghz = self.rand_f32() > 0.5;
        let center_freq = if is_5ghz {
            // 5 GHz channels (simplified)
            5.2e9 + self.rand_range(0.0, 500e6)
        } else {
            // 2.4 GHz channels
            2.412e9 + (self.next_u64() % 13) as f32 * 5e6
        };

        // Bandwidth: 20, 40, or 80 MHz
        let bw_choice = self.next_u64() % 3;
        let bandwidth = match bw_choice {
            0 => constants::WIFI_BW_20,
            1 => constants::WIFI_BW_40,
            _ => constants::WIFI_BW_80,
        };

        entities.set_entity(
            env,
            entity,
            EntityType::WiFiAP,
            ModulationType::OFDM,
            x,
            y,
            z,
            center_freq,
            bandwidth,
            self.rand_range(20.0, 30.0), // 20-30 dBm (typical AP)
        );

        // Initialize with random timer for bursty behavior
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.005);
    }

    /// Spawn a Bluetooth device entity.
    fn spawn_bluetooth(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Bluetooth: 2.4 GHz ISM band, FHSS across 79 channels
        // Start at a random hop
        let hop_idx = (self.next_u64() % 79) as u32;
        let center_freq = 2.402e9 + hop_idx as f32 * 1e6;

        entities.set_entity(
            env,
            entity,
            EntityType::Bluetooth,
            ModulationType::GFSK,
            x,
            y,
            z,
            center_freq,
            1e6, // 1 MHz per channel
            self.rand_range(0.0, 10.0), // 0-10 dBm (Class 1-3)
        );

        let idx = entities.idx(env, entity);
        entities.hop_idx[idx] = hop_idx;
        entities.timer[idx] = self.rand_range(0.0, 0.000625); // Hop period ~625 μs
    }

    /// Spawn an LTE uplink (user equipment) entity.
    fn spawn_lte_uplink(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // LTE uplink uses SC-FDM, paired with downlink bands
        // Common uplink bands: 700, 850, 1900, 2100 MHz
        let bands = [699e6, 824e6, 1850e6, 1920e6];
        let band_idx = (self.next_u64() % 4) as usize;
        let center_freq = bands[band_idx] + self.rand_range(-5e6, 5e6);

        // Uplink typically uses fewer resource blocks (1-25 RBs = 180 kHz - 4.5 MHz)
        let bandwidth = self.rand_range(180e3, 4.5e6);

        entities.set_entity(
            env,
            entity,
            EntityType::LTEUplink,
            ModulationType::SCFDMA,
            x,
            y,
            z,
            center_freq,
            bandwidth,
            self.rand_range(10.0, 23.0), // 10-23 dBm (typical UE)
        );

        // Initialize with random timer for bursty behavior
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.01);
    }

    /// Spawn a Zigbee IoT device entity.
    fn spawn_zigbee(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Zigbee: 2.4 GHz ISM band, channels 11-26 (2.405-2.480 GHz)
        let channel = (self.next_u64() % 16) as f32 + 11.0;
        let center_freq = 2.405e9 + (channel - 11.0) * 5e6;

        entities.set_entity(
            env,
            entity,
            EntityType::Zigbee,
            ModulationType::DSSS,
            x,
            y,
            z,
            center_freq,
            2e6, // 2 MHz bandwidth (O-QPSK DSSS)
            self.rand_range(0.0, 20.0), // 0-20 dBm
        );

        // Very low duty cycle (~1%), initialize timer
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 1.0); // Long sleep periods
    }

    /// Spawn a LoRaWAN device entity.
    fn spawn_lorawan(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // LoRaWAN US915: 902.3-914.9 MHz uplink (64 channels)
        let channel = (self.next_u64() % 64) as f32;
        let center_freq = 902.3e6 + channel * 200e3;

        // Bandwidth: 125 kHz, 250 kHz, or 500 kHz
        let bw_choice = self.next_u64() % 3;
        let bandwidth = match bw_choice {
            0 => 125e3,
            1 => 250e3,
            _ => 500e3,
        };

        entities.set_entity(
            env,
            entity,
            EntityType::LoRaWAN,
            ModulationType::Chirp,
            x,
            y,
            z,
            center_freq,
            bandwidth,
            self.rand_range(10.0, 20.0), // 10-20 dBm
        );

        // Very low duty cycle (<1%), long sleep periods
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 60.0); // Minutes between transmissions
    }

    /// Spawn an S-Band radar entity.
    fn spawn_sband_radar(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // S-Band: 2-4 GHz
        let center_freq = self.rand_range(constants::RADAR_S_START, constants::RADAR_S_END);

        entities.set_entity(
            env,
            entity,
            EntityType::SBandRadar,
            ModulationType::Chirp,
            x,
            y,
            z,
            center_freq,
            10e6, // 10 MHz chirp bandwidth
            self.rand_range(55.0, 65.0), // 55-65 dBm (high power)
        );

        // PRI timer: ~1 ms
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.001);
    }

    /// Spawn a Weather radar entity.
    fn spawn_weather_radar(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Weather radar: 2.7-2.9 GHz
        let center_freq = self.rand_range(2.7e9, 2.9e9);

        entities.set_entity(
            env,
            entity,
            EntityType::WeatherRadar,
            ModulationType::Chirp,
            x,
            y,
            z,
            center_freq,
            5e6, // 5 MHz chirp bandwidth
            self.rand_range(50.0, 60.0), // 50-60 dBm
        );

        // PRI timer: ~3 ms
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.003);
    }

    /// Spawn an L-Band radar entity (ATC, surveillance).
    fn spawn_lband_radar(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // L-Band radar: 1-2 GHz (typically 1.2-1.4 GHz for ATC)
        let center_freq = self.rand_range(1.2e9, 1.4e9);

        entities.set_entity(
            env,
            entity,
            EntityType::LBandRadar,
            ModulationType::Chirp,
            x,
            y,
            z,
            center_freq,
            20e6, // 20 MHz chirp bandwidth (wider than S-band)
            self.rand_range(60.0, 70.0), // 60-70 dBm (very high power)
        );

        // PRI timer: ~1-2 ms
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.002);
    }

    /// Spawn an ADS-B transponder entity.
    ///
    /// ADS-B transmits at 1090 MHz with PPM modulation.
    /// Aircraft transmit asynchronously, ~2 per second.
    fn spawn_adsb(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        // ADS-B transponders are typically on aircraft
        // Use higher z position to simulate airborne aircraft
        let x = self.rand_range(0.0, self.world_size.0);
        let y = self.rand_range(0.0, self.world_size.1);
        let z = self.rand_range(self.world_size.2 * 0.3, self.world_size.2 * 0.9);

        // Fixed at 1090 MHz
        let center_freq = 1090e6;

        entities.set_entity(
            env,
            entity,
            EntityType::ADSB,
            ModulationType::BPSK, // PPM encoded
            x,
            y,
            z,
            center_freq,
            1e6, // ~1 MHz bandwidth (120 µs pulse = ~8.3 kHz, but spectral spread wider)
            self.rand_range(50.0, 57.0), // 50-57 dBm (100-500W)
        );

        // Transmits ~2 per second (500 ms interval), random initial phase
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.5);
    }

    /// Spawn a DME (Distance Measuring Equipment) entity.
    ///
    /// DME operates at 960-1215 MHz with pulse pairs.
    fn spawn_dme(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // DME X-mode channels: 962-1024 MHz (interrogate), 1025-1150 MHz (reply)
        // Simplified: pick a frequency in the DME band
        let center_freq = self.rand_range(962e6, 1150e6);

        entities.set_entity(
            env,
            entity,
            EntityType::DME,
            ModulationType::CW, // Pulse pairs are essentially CW bursts
            x,
            y,
            z,
            center_freq,
            500e3, // ~500 kHz bandwidth (pulse pair spectral width)
            self.rand_range(40.0, 50.0), // 40-50 dBm
        );

        // Reply rate depends on interrogation, ~30 per second typical
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 0.033);
    }

    /// Spawn a geostationary satellite beacon entity.
    ///
    /// GeoSat beacons at C-band (3.7-4.2 GHz) or Ku-band (10.7-12.75 GHz).
    fn spawn_geo_satellite(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        // Geostationary satellites are at ~35,786 km altitude
        // For simulation, place at top of world bounds
        let x = self.world_size.0 / 2.0;
        let y = self.world_size.1 / 2.0;
        let z = self.world_size.2;

        // Choose C-band or Ku-band
        let is_ku_band = self.rand_f32() > 0.5;
        let center_freq = if is_ku_band {
            self.rand_range(10.7e9, 12.75e9) // Ku-band
        } else {
            self.rand_range(3.7e9, 4.2e9) // C-band
        };

        entities.set_entity(
            env,
            entity,
            EntityType::GeoSatellite,
            ModulationType::QPSK,
            x,
            y,
            z,
            center_freq,
            100e3, // 100 kHz beacon bandwidth
            self.rand_range(-100.0, -80.0), // Very weak at Earth surface
        );
    }

    /// Spawn an analog video drone entity.
    fn spawn_drone_analog(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Analog drone video: 5.8 GHz
        let center_freq = 5.8e9 + self.rand_range(-100e6, 100e6);

        entities.set_entity(
            env,
            entity,
            EntityType::DroneAnalog,
            ModulationType::FM,
            x,
            y,
            z,
            center_freq,
            10e6, // 10 MHz video bandwidth
            self.rand_range(20.0, 27.0), // 20-27 dBm
        );

        // Set random velocity (5-20 m/s)
        let (vx, vy, vz) = self.rand_velocity(5.0, 20.0);
        entities.set_velocity(env, entity, vx, vy, vz);
    }

    /// Spawn a digital video drone entity.
    fn spawn_drone_digital(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Digital drone video: 5.8 GHz
        let center_freq = 5.8e9 + self.rand_range(-100e6, 100e6);

        entities.set_entity(
            env,
            entity,
            EntityType::DroneDigital,
            ModulationType::OFDM,
            x,
            y,
            z,
            center_freq,
            20e6, // 20 MHz OFDM bandwidth
            self.rand_range(20.0, 27.0), // 20-27 dBm
        );

        // Set random velocity (5-20 m/s)
        let (vx, vy, vz) = self.rand_velocity(5.0, 20.0);
        entities.set_velocity(env, entity, vx, vy, vz);
    }

    /// Spawn a vehicle entity.
    fn spawn_vehicle(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        // Vehicles start on the ground
        let x = self.rand_range(0.0, self.world_size.0);
        let y = self.rand_range(0.0, self.world_size.1);
        let z = 0.0; // Ground level

        // V2X: 5.9 GHz
        let center_freq = 5.9e9 + self.rand_range(-25e6, 25e6);

        entities.set_entity(
            env,
            entity,
            EntityType::Vehicle,
            ModulationType::OFDM,
            x,
            y,
            z,
            center_freq,
            10e6, // 10 MHz V2X bandwidth
            self.rand_range(10.0, 23.0), // 10-23 dBm
        );

        // Set random velocity (10-30 m/s, horizontal only)
        let (vx, vy, _) = self.rand_velocity(10.0, 30.0);
        entities.set_velocity(env, entity, vx, vy, 0.0);
    }

    // ========================================================================
    // Voice Radio Spawners (PTT behavior)
    // ========================================================================

    /// Spawn a Maritime VHF radio entity.
    ///
    /// Maritime VHF operates at 156-174 MHz with 25 kHz FM channels.
    /// Common channels: 16 (distress, 156.8 MHz), 9 (bridge-bridge), 13 (intership safety)
    fn spawn_maritime_vhf(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Maritime VHF channels: 156.025 - 162.025 MHz
        // 57 channels with 25 kHz spacing
        let channel = (self.next_u64() % 57) as f32;
        let center_freq = 156.025e6 + channel * 25e3;

        entities.set_entity(
            env,
            entity,
            EntityType::MaritimeVHF,
            ModulationType::FM,
            x,
            y,
            z,
            center_freq,
            25e3, // 25 kHz channel bandwidth
            self.rand_range(37.0, 47.0), // 37-47 dBm (5W handheld to 50W ship)
        );

        // PTT behavior: random initial timer
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 30.0); // Sporadic transmissions

        // Slow movement (boats)
        let (vx, vy, _) = self.rand_velocity(2.0, 10.0);
        entities.set_velocity(env, entity, vx, vy, 0.0);
    }

    /// Spawn a Walkie-Talkie (FRS) entity.
    ///
    /// FRS operates at 462 MHz with 12.5 kHz channels.
    /// Low power (0.5-2W), short range consumer radios.
    fn spawn_walkie_talkie(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // FRS channels 1-22: 462.5625 - 467.7125 MHz
        let channel = (self.next_u64() % 22) as f32;
        let center_freq = 462.5625e6 + channel * 25e3;

        entities.set_entity(
            env,
            entity,
            EntityType::WalkieTalkie,
            ModulationType::FM,
            x,
            y,
            z,
            center_freq,
            12.5e3, // 12.5 kHz narrow FM
            self.rand_range(27.0, 33.0), // 27-33 dBm (0.5-2W)
        );

        // PTT behavior: random initial timer
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 60.0); // Sporadic transmissions

        // Walking speed
        let (vx, vy, _) = self.rand_velocity(0.5, 2.0);
        entities.set_velocity(env, entity, vx, vy, 0.0);
    }

    /// Spawn a GMRS radio entity.
    ///
    /// GMRS operates at 462-467 MHz with 25 kHz channels.
    /// Higher power than FRS (up to 50W), may use repeaters.
    fn spawn_gmrs(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // GMRS channels: 462-467 MHz range, shared with FRS
        // Repeater inputs at 467 MHz, outputs at 462 MHz
        let channel = (self.next_u64() % 30) as f32;
        let center_freq = 462.55e6 + channel * 25e3;

        entities.set_entity(
            env,
            entity,
            EntityType::GMRS,
            ModulationType::FM,
            x,
            y,
            z,
            center_freq,
            25e3, // 25 kHz channel bandwidth
            self.rand_range(33.0, 47.0), // 33-47 dBm (2-50W)
        );

        // PTT behavior: random initial timer
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 45.0);

        // Vehicle or walking speed
        let (vx, vy, _) = self.rand_velocity(1.0, 15.0);
        entities.set_velocity(env, entity, vx, vy, 0.0);
    }

    /// Spawn a P25 public safety radio entity.
    ///
    /// P25 operates at 700/800 MHz with C4FM (12.5 kHz narrow).
    /// Digital trunked radio for police, fire, EMS.
    fn spawn_p25(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // P25 bands: 700 MHz (764-776/794-806) or 800 MHz (806-824/851-869)
        let is_800mhz = self.rand_f32() > 0.5;
        let base_freq = if is_800mhz { 851e6 } else { 764e6 };
        let center_freq = base_freq + self.rand_range(0.0, 12e6);

        entities.set_entity(
            env,
            entity,
            EntityType::P25Radio,
            ModulationType::GFSK, // C4FM is 4-level FSK
            x,
            y,
            z,
            center_freq,
            12.5e3, // 12.5 kHz narrow digital
            self.rand_range(33.0, 43.0), // 33-43 dBm (2-20W)
        );

        // PTT behavior: random initial timer
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 20.0); // More frequent for emergency

        // Vehicle speed (patrol cars, fire trucks)
        let (vx, vy, _) = self.rand_velocity(5.0, 25.0);
        entities.set_velocity(env, entity, vx, vy, 0.0);
    }

    /// Spawn an Amateur (Ham) radio entity.
    ///
    /// Multi-band, multi-mode. Common bands: 2m (144 MHz), 70cm (430 MHz).
    /// Modes: SSB, FM, CW, digital modes.
    fn spawn_amateur_radio(&mut self, entities: &mut EntitySoA, env: usize, entity: usize) {
        let (x, y, z) = self.rand_position();

        // Amateur bands: 2m (144-148 MHz) or 70cm (420-450 MHz)
        let band_choice = self.next_u64() % 3;
        let (center_freq, bandwidth, modulation) = match band_choice {
            0 => {
                // 2m FM simplex/repeater
                let channel = (self.next_u64() % 40) as f32;
                (144.0e6 + channel * 25e3, 25e3, ModulationType::FM)
            }
            1 => {
                // 2m SSB (weak signal)
                (144.2e6 + self.rand_range(0.0, 100e3), 2.7e3, ModulationType::SSB)
            }
            _ => {
                // 70cm FM
                let channel = (self.next_u64() % 80) as f32;
                (430.0e6 + channel * 25e3, 25e3, ModulationType::FM)
            }
        };

        entities.set_entity(
            env,
            entity,
            EntityType::AmateurRadio,
            modulation,
            x,
            y,
            z,
            center_freq,
            bandwidth,
            self.rand_range(30.0, 50.0), // 30-50 dBm (1-100W typical)
        );

        // PTT behavior: random initial timer
        let idx = entities.idx(env, entity);
        entities.timer[idx] = self.rand_range(0.0, 120.0); // Variable activity

        // Mostly stationary or slow movement
        let (vx, vy, _) = self.rand_velocity(0.0, 2.0);
        entities.set_velocity(env, entity, vx, vy, 0.0);
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_config_default() {
        let config = EntityConfig::default();

        assert_eq!(config.num_tv_stations, 4);
        assert_eq!(config.num_fm_radios, 3);
        assert_eq!(config.num_am_radios, 2);
        assert_eq!(config.num_gnss, 1);
        assert_eq!(config.num_geo_satellites, 1);
        assert_eq!(config.num_lte_uplink, 4);
        assert_eq!(config.num_zigbee, 5);
        assert_eq!(config.num_lorawan, 2);
        assert_eq!(config.num_lband_radars, 1);
        assert_eq!(config.num_adsb, 3);
        assert_eq!(config.num_dme, 1);
        // Voice radios
        assert_eq!(config.num_maritime_vhf, 1);
        assert_eq!(config.num_walkie_talkie, 2);
        assert_eq!(config.num_gmrs, 1);
        assert_eq!(config.num_p25, 1);
        assert_eq!(config.num_amateur_radio, 1);
        // Continuous: 4+3+2+1+1 = 11
        // Bursty: 2+8+6+4+5+2 = 27
        // Periodic: 1+1+1+3+1 = 7
        // Mobile: 2+2+4 = 8
        // Voice radios: 1+2+1+1+1 = 6
        // Total: 11+27+7+8+6 = 59
        assert_eq!(config.total_entities(), 59);
    }

    #[test]
    fn test_entity_config_presets() {
        let dense = EntityConfig::dense_urban();
        assert!(dense.total_entities() > EntityConfig::default().total_entities());

        let rural = EntityConfig::rural();
        assert!(rural.total_entities() < EntityConfig::default().total_entities());

        let empty = EntityConfig::empty();
        assert_eq!(empty.total_entities(), 0);
    }

    #[test]
    fn test_entity_config_builder() {
        let config = EntityConfig::new()
            .with_tv_stations(10)
            .with_fm_radios(5)
            .with_lte_towers(3);

        assert_eq!(config.num_tv_stations, 10);
        assert_eq!(config.num_fm_radios, 5);
        assert_eq!(config.num_lte_towers, 3);
    }

    #[test]
    fn test_entity_config_fits() {
        let config = EntityConfig::default();
        assert!(config.fits_in(64));
        assert!(!config.fits_in(10));
    }

    #[test]
    fn test_spawner_creation() {
        let config = EntityConfig::default();
        let spawner = EntitySpawner::new(config.clone(), (1000.0, 1000.0, 100.0), 42);

        assert_eq!(spawner.config().total_entities(), config.total_entities());
    }

    #[test]
    fn test_spawn_all() {
        let config = EntityConfig::minimal();
        let mut spawner = EntitySpawner::new(config.clone(), (1000.0, 1000.0, 100.0), 42);
        let mut entities = EntitySoA::new(8, 64);

        let count = spawner.spawn_all(&mut entities, 0);

        assert_eq!(count, config.total_entities());
        assert_eq!(entities.count_active(0), count);
    }

    #[test]
    fn test_spawn_reproducibility() {
        let config = EntityConfig::minimal();

        // First run
        let mut spawner1 = EntitySpawner::new(config.clone(), (1000.0, 1000.0, 100.0), 42);
        let mut entities1 = EntitySoA::new(8, 64);
        spawner1.spawn_all(&mut entities1, 0);

        // Second run with same seed
        let mut spawner2 = EntitySpawner::new(config, (1000.0, 1000.0, 100.0), 42);
        let mut entities2 = EntitySoA::new(8, 64);
        spawner2.spawn_all(&mut entities2, 0);

        // Should be identical
        for i in 0..entities1.max_entities() {
            let idx1 = entities1.idx(0, i);
            let idx2 = entities2.idx(0, i);
            assert_eq!(entities1.x[idx1], entities2.x[idx2]);
            assert_eq!(entities1.center_freq[idx1], entities2.center_freq[idx2]);
        }
    }

    #[test]
    fn test_spawn_positions_in_bounds() {
        let config = EntityConfig::default();
        let world_size = (500.0, 500.0, 50.0);
        let mut spawner = EntitySpawner::new(config, world_size, 12345);
        let mut entities = EntitySoA::new(8, 64);

        spawner.spawn_all(&mut entities, 0);

        for i in 0..entities.max_entities() {
            if entities.is_active(0, i) {
                let idx = entities.idx(0, i);
                assert!(entities.x[idx] >= 0.0 && entities.x[idx] <= world_size.0);
                assert!(entities.y[idx] >= 0.0 && entities.y[idx] <= world_size.1);
                assert!(entities.z[idx] >= 0.0 && entities.z[idx] <= world_size.2);
            }
        }
    }

    #[test]
    fn test_spawn_tv_frequency_range() {
        // Use empty config with just TV stations to avoid exceeding max_entities
        let config = EntityConfig::none().with_tv_stations(20);
        let mut spawner = EntitySpawner::new(config, (1000.0, 1000.0, 100.0), 42);
        let mut entities = EntitySoA::new(8, 64);

        spawner.spawn_all(&mut entities, 0);

        for i in 0..entities.max_entities() {
            if entities.is_active(0, i) && entities.get_type(0, i) == EntityType::TVStation {
                let idx = entities.idx(0, i);
                let freq = entities.center_freq[idx];
                // Should be within TV UHF band
                assert!(freq >= 470e6 && freq <= 620e6, "TV freq out of range: {}", freq);
            }
        }
    }

    #[test]
    fn test_spawn_mobile_has_velocity() {
        // Use empty config with just mobile entities
        let config = EntityConfig::none()
            .with_drones(5, 5)
            .with_vehicles(5);

        let mut spawner = EntitySpawner::new(config, (1000.0, 1000.0, 100.0), 42);
        let mut entities = EntitySoA::new(8, 64);

        spawner.spawn_all(&mut entities, 0);

        // Check that all active mobile entities have non-zero velocity
        for i in 0..entities.max_entities() {
            if entities.is_active(0, i) {
                let entity_type = entities.get_type(0, i);
                if entity_type.is_mobile() {
                    let idx = entities.idx(0, i);
                    let speed_sq = entities.vx[idx].powi(2)
                        + entities.vy[idx].powi(2)
                        + entities.vz[idx].powi(2);
                    assert!(speed_sq > 0.0, "Mobile entity {} has zero velocity", i);
                }
            }
        }
    }
}
