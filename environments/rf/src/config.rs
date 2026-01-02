//! RF World Configuration
//!
//! Configuration structures with builder pattern for setting up
//! RF environment simulations.
//!
//! ## Type-Safe API
//!
//! This module provides both legacy (raw f32) methods and type-safe methods
//! that use `Hertz`, `ValidatedFrequencyGrid`, and `PositivePower` types.
//!
//! Type-safe accessors are available via methods like `freq_min_typed()`,
//! `freq_max_typed()`, and `validated_grid()`.

use crate::constants;
use crate::spectrum::NoiseEnvironment;
use crate::types::{Hertz, ValidatedFrequencyGrid, PositivePower};

// ============================================================================
// Entity Configuration
// ============================================================================

/// Configuration for entity spawning.
///
/// All entity counts are configurable via builder pattern.
#[derive(Clone, Debug)]
pub struct EntityConfig {
    // ========================================================================
    // Continuous entities (100% duty cycle)
    // ========================================================================
    /// Number of TV stations
    pub num_tv_stations: usize,
    /// Number of FM radio stations
    pub num_fm_radios: usize,
    /// Number of AM radio stations
    pub num_am_radios: usize,
    /// Number of GNSS satellite constellations visible
    pub num_gnss: usize,
    /// Number of geostationary satellite beacons
    pub num_geo_satellites: usize,

    // ========================================================================
    // Bursty entities (Poisson arrivals)
    // ========================================================================
    /// Number of LTE towers (downlink)
    pub num_lte_towers: usize,
    /// Number of WiFi access points
    pub num_wifi_aps: usize,
    /// Number of Bluetooth devices
    pub num_bluetooth: usize,
    /// Number of LTE uplink devices (user equipment)
    pub num_lte_uplink: usize,
    /// Number of Zigbee IoT devices
    pub num_zigbee: usize,
    /// Number of LoRaWAN devices
    pub num_lorawan: usize,

    // ========================================================================
    // Periodic entities (pulsed)
    // ========================================================================
    /// Number of S-Band radars
    pub num_sband_radars: usize,
    /// Number of weather radars
    pub num_weather_radars: usize,
    /// Number of L-Band radars (ATC, surveillance)
    pub num_lband_radars: usize,
    /// Number of ADS-B aircraft transponders
    pub num_adsb: usize,
    /// Number of DME navigation aids
    pub num_dme: usize,

    // ========================================================================
    // Mobile entities (Doppler effects, despawn/respawn)
    // ========================================================================
    /// Number of analog video drones
    pub num_drone_analog: usize,
    /// Number of digital video drones
    pub num_drone_digital: usize,
    /// Number of vehicles (V2X)
    pub num_vehicles: usize,

    // ========================================================================
    // Voice radios (PTT behavior)
    // ========================================================================
    /// Number of Maritime VHF radios
    pub num_maritime_vhf: usize,
    /// Number of Walkie-Talkies (FRS)
    pub num_walkie_talkie: usize,
    /// Number of GMRS radios
    pub num_gmrs: usize,
    /// Number of P25 public safety radios
    pub num_p25: usize,
    /// Number of Amateur (Ham) radios
    pub num_amateur_radio: usize,
}

impl Default for EntityConfig {
    fn default() -> Self {
        Self {
            // Continuous
            num_tv_stations: 4,
            num_fm_radios: 3,
            num_am_radios: 2,
            num_gnss: 1, // Always 1 - represents aggregate of visible satellites
            num_geo_satellites: 1,

            // Bursty
            num_lte_towers: 2,
            num_wifi_aps: 8,
            num_bluetooth: 6,
            num_lte_uplink: 4,
            num_zigbee: 5,
            num_lorawan: 2,

            // Periodic
            num_sband_radars: 1,
            num_weather_radars: 1,
            num_lband_radars: 1,
            num_adsb: 3, // Simulates nearby aircraft
            num_dme: 1,

            // Mobile
            num_drone_analog: 2,
            num_drone_digital: 2,
            num_vehicles: 4,

            // Voice radios (PTT)
            num_maritime_vhf: 1,
            num_walkie_talkie: 2,
            num_gmrs: 1,
            num_p25: 1,
            num_amateur_radio: 1,
        }
    }
}

impl EntityConfig {
    /// Create a new entity configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of TV stations.
    pub fn with_tv_stations(mut self, count: usize) -> Self {
        self.num_tv_stations = count;
        self
    }

    /// Set number of FM radios.
    pub fn with_fm_radios(mut self, count: usize) -> Self {
        self.num_fm_radios = count;
        self
    }

    /// Set number of LTE towers.
    pub fn with_lte_towers(mut self, count: usize) -> Self {
        self.num_lte_towers = count;
        self
    }

    /// Set number of WiFi access points.
    pub fn with_wifi_aps(mut self, count: usize) -> Self {
        self.num_wifi_aps = count;
        self
    }

    /// Set number of Bluetooth devices.
    pub fn with_bluetooth(mut self, count: usize) -> Self {
        self.num_bluetooth = count;
        self
    }

    /// Set number of S-Band radars.
    pub fn with_sband_radars(mut self, count: usize) -> Self {
        self.num_sband_radars = count;
        self
    }

    /// Set number of weather radars.
    pub fn with_weather_radars(mut self, count: usize) -> Self {
        self.num_weather_radars = count;
        self
    }

    /// Set number of analog drones.
    pub fn with_drone_analog(mut self, count: usize) -> Self {
        self.num_drone_analog = count;
        self
    }

    /// Set number of digital drones.
    pub fn with_drone_digital(mut self, count: usize) -> Self {
        self.num_drone_digital = count;
        self
    }

    /// Set number of vehicles.
    pub fn with_vehicles(mut self, count: usize) -> Self {
        self.num_vehicles = count;
        self
    }

    /// Set all radar counts at once.
    pub fn with_radars(mut self, sband: usize, weather: usize) -> Self {
        self.num_sband_radars = sband;
        self.num_weather_radars = weather;
        self
    }

    /// Set all drone counts at once.
    pub fn with_drones(mut self, analog: usize, digital: usize) -> Self {
        self.num_drone_analog = analog;
        self.num_drone_digital = digital;
        self
    }

    /// Set number of AM radios.
    pub fn with_am_radios(mut self, count: usize) -> Self {
        self.num_am_radios = count;
        self
    }

    /// Set number of GNSS entities.
    pub fn with_gnss(mut self, count: usize) -> Self {
        self.num_gnss = count;
        self
    }

    /// Set number of LTE uplink devices.
    pub fn with_lte_uplink(mut self, count: usize) -> Self {
        self.num_lte_uplink = count;
        self
    }

    /// Set number of Zigbee devices.
    pub fn with_zigbee(mut self, count: usize) -> Self {
        self.num_zigbee = count;
        self
    }

    /// Set number of LoRaWAN devices.
    pub fn with_lorawan(mut self, count: usize) -> Self {
        self.num_lorawan = count;
        self
    }

    /// Set all IoT device counts at once.
    pub fn with_iot_devices(mut self, zigbee: usize, lorawan: usize) -> Self {
        self.num_zigbee = zigbee;
        self.num_lorawan = lorawan;
        self
    }

    /// Set number of geostationary satellites.
    pub fn with_geo_satellites(mut self, count: usize) -> Self {
        self.num_geo_satellites = count;
        self
    }

    /// Set number of L-Band radars.
    pub fn with_lband_radars(mut self, count: usize) -> Self {
        self.num_lband_radars = count;
        self
    }

    /// Set number of ADS-B transponders.
    pub fn with_adsb(mut self, count: usize) -> Self {
        self.num_adsb = count;
        self
    }

    /// Set number of DME navigation aids.
    pub fn with_dme(mut self, count: usize) -> Self {
        self.num_dme = count;
        self
    }

    /// Set all aviation entity counts at once.
    pub fn with_aviation(mut self, adsb: usize, dme: usize) -> Self {
        self.num_adsb = adsb;
        self.num_dme = dme;
        self
    }

    /// Set number of Maritime VHF radios.
    pub fn with_maritime_vhf(mut self, count: usize) -> Self {
        self.num_maritime_vhf = count;
        self
    }

    /// Set number of Walkie-Talkies (FRS).
    pub fn with_walkie_talkie(mut self, count: usize) -> Self {
        self.num_walkie_talkie = count;
        self
    }

    /// Set number of GMRS radios.
    pub fn with_gmrs(mut self, count: usize) -> Self {
        self.num_gmrs = count;
        self
    }

    /// Set number of P25 public safety radios.
    pub fn with_p25(mut self, count: usize) -> Self {
        self.num_p25 = count;
        self
    }

    /// Set number of Amateur radios.
    pub fn with_amateur_radio(mut self, count: usize) -> Self {
        self.num_amateur_radio = count;
        self
    }

    /// Set all voice radio counts at once.
    pub fn with_voice_radios(
        mut self,
        maritime_vhf: usize,
        walkie_talkie: usize,
        gmrs: usize,
        p25: usize,
        amateur: usize,
    ) -> Self {
        self.num_maritime_vhf = maritime_vhf;
        self.num_walkie_talkie = walkie_talkie;
        self.num_gmrs = gmrs;
        self.num_p25 = p25;
        self.num_amateur_radio = amateur;
        self
    }

    /// Get total entity count.
    pub fn total_entities(&self) -> usize {
        // Continuous
        self.num_tv_stations
            + self.num_fm_radios
            + self.num_am_radios
            + self.num_gnss
            + self.num_geo_satellites
            // Bursty
            + self.num_lte_towers
            + self.num_wifi_aps
            + self.num_bluetooth
            + self.num_lte_uplink
            + self.num_zigbee
            + self.num_lorawan
            // Periodic
            + self.num_sband_radars
            + self.num_weather_radars
            + self.num_lband_radars
            + self.num_adsb
            + self.num_dme
            // Mobile
            + self.num_drone_analog
            + self.num_drone_digital
            + self.num_vehicles
            // Voice radios (PTT)
            + self.num_maritime_vhf
            + self.num_walkie_talkie
            + self.num_gmrs
            + self.num_p25
            + self.num_amateur_radio
    }

    /// Dense urban preset (many entities).
    pub fn dense_urban() -> Self {
        Self {
            // Continuous
            num_tv_stations: 6,
            num_fm_radios: 5,
            num_am_radios: 3,
            num_gnss: 1,
            num_geo_satellites: 2,
            // Bursty
            num_lte_towers: 4,
            num_wifi_aps: 16,
            num_bluetooth: 12,
            num_lte_uplink: 8,
            num_zigbee: 10,
            num_lorawan: 4,
            // Periodic
            num_sband_radars: 2,
            num_weather_radars: 1,
            num_lband_radars: 1,
            num_adsb: 6, // More aircraft near airports
            num_dme: 2,
            // Mobile
            num_drone_analog: 4,
            num_drone_digital: 4,
            num_vehicles: 8,
            // Voice radios (more activity in urban areas)
            num_maritime_vhf: 0, // No maritime in urban
            num_walkie_talkie: 4,
            num_gmrs: 2,
            num_p25: 3, // Emergency services
            num_amateur_radio: 2,
        }
    }

    /// Rural preset (few entities).
    pub fn rural() -> Self {
        Self {
            // Continuous
            num_tv_stations: 2,
            num_fm_radios: 2,
            num_am_radios: 2, // AM propagates far, still audible
            num_gnss: 1,
            num_geo_satellites: 1,
            // Bursty
            num_lte_towers: 1,
            num_wifi_aps: 2,
            num_bluetooth: 2,
            num_lte_uplink: 1,
            num_zigbee: 2,
            num_lorawan: 3, // LoRa designed for rural
            // Periodic
            num_sband_radars: 0,
            num_weather_radars: 1,
            num_lband_radars: 0,
            num_adsb: 1, // Less aircraft in rural areas
            num_dme: 0,
            // Mobile
            num_drone_analog: 1,
            num_drone_digital: 1,
            num_vehicles: 2,
            // Voice radios (less in rural but ham radio popular)
            num_maritime_vhf: 0,
            num_walkie_talkie: 1,
            num_gmrs: 1,
            num_p25: 0,
            num_amateur_radio: 2, // Ham radio popular in rural areas
        }
    }

    /// Minimal preset for testing.
    pub fn minimal() -> Self {
        Self {
            // Continuous
            num_tv_stations: 1,
            num_fm_radios: 1,
            num_am_radios: 1,
            num_gnss: 1,
            num_geo_satellites: 1,
            // Bursty
            num_lte_towers: 1,
            num_wifi_aps: 2,
            num_bluetooth: 2,
            num_lte_uplink: 1,
            num_zigbee: 1,
            num_lorawan: 1,
            // Periodic
            num_sband_radars: 1,
            num_weather_radars: 0,
            num_lband_radars: 0,
            num_adsb: 1,
            num_dme: 0,
            // Mobile
            num_drone_analog: 1,
            num_drone_digital: 1,
            num_vehicles: 1,
            // Voice radios
            num_maritime_vhf: 0,
            num_walkie_talkie: 1,
            num_gmrs: 0,
            num_p25: 0,
            num_amateur_radio: 1,
        }
    }

    /// No entities preset (for testing noise floor only).
    pub fn none() -> Self {
        Self {
            num_tv_stations: 0,
            num_fm_radios: 0,
            num_am_radios: 0,
            num_gnss: 0,
            num_geo_satellites: 0,
            num_lte_towers: 0,
            num_wifi_aps: 0,
            num_bluetooth: 0,
            num_lte_uplink: 0,
            num_zigbee: 0,
            num_lorawan: 0,
            num_sband_radars: 0,
            num_weather_radars: 0,
            num_lband_radars: 0,
            num_adsb: 0,
            num_dme: 0,
            num_drone_analog: 0,
            num_drone_digital: 0,
            num_vehicles: 0,
            num_maritime_vhf: 0,
            num_walkie_talkie: 0,
            num_gmrs: 0,
            num_p25: 0,
            num_amateur_radio: 0,
        }
    }

    /// Alias for none() - empty configuration.
    pub fn empty() -> Self {
        Self::none()
    }

    /// Check if this configuration fits within the given max entities limit.
    pub fn fits_in(&self, max_entities: usize) -> bool {
        self.total_entities() <= max_entities
    }
}

/// Main configuration for the RF World environment
#[derive(Clone, Debug)]
pub struct RFWorldConfig {
    // ========================================================================
    // Environment Dimensions
    // ========================================================================
    /// Number of parallel environments (must be multiple of 8 for SIMD)
    pub num_envs: usize,

    /// Number of frequency bins in the PSD
    pub num_freq_bins: usize,

    /// Minimum frequency in the simulated spectrum (Hz)
    pub freq_min: f32,

    /// Maximum frequency in the simulated spectrum (Hz)
    pub freq_max: f32,

    // ========================================================================
    // Entity Limits
    // ========================================================================
    /// Maximum number of background entities per environment
    pub max_entities: usize,

    /// World size in meters (x, y, z)
    pub world_size: (f32, f32, f32),

    // ========================================================================
    // Timing
    // ========================================================================
    /// Physics simulation frequency (Hz)
    pub physics_freq: u32,

    /// Control/action frequency (Hz)
    pub ctrl_freq: u32,

    /// Maximum steps per episode
    pub max_steps: u32,

    // ========================================================================
    // Random Seed
    // ========================================================================
    /// Random seed for reproducibility
    pub seed: u64,

    // ========================================================================
    // Noise Configuration
    // ========================================================================
    /// Receiver noise figure (dB)
    pub noise_figure: f32,

    /// Base noise floor (dBm/Hz)
    pub noise_floor_dbm_hz: f32,

    /// Noise environment type (affects man-made noise)
    pub noise_environment: NoiseEnvironment,

    // ========================================================================
    // Entity Configuration
    // ========================================================================
    /// Entity spawning configuration
    pub entity_config: EntityConfig,

    // ========================================================================
    // Multi-Agent Configuration (Phase 4)
    // ========================================================================
    /// Number of jammer agents per environment
    pub num_jammers: usize,

    /// Number of cognitive radio agents per environment
    pub num_crs: usize,

    /// SINR threshold for successful CR communication (dB)
    pub sinr_threshold_db: f32,
}

impl Default for RFWorldConfig {
    fn default() -> Self {
        Self {
            // Environment dimensions
            num_envs: 8, // Single SIMD batch
            num_freq_bins: constants::DEFAULT_NUM_FREQ_BINS,
            freq_min: constants::DEFAULT_FREQ_MIN,
            freq_max: constants::DEFAULT_FREQ_MAX,

            // Entity limits
            max_entities: 64,
            world_size: constants::DEFAULT_WORLD_SIZE,

            // Timing
            physics_freq: constants::DEFAULT_PHYSICS_FREQ,
            ctrl_freq: constants::DEFAULT_CTRL_FREQ,
            max_steps: constants::DEFAULT_MAX_STEPS,

            // Random seed
            seed: 42,

            // Noise
            noise_figure: constants::DEFAULT_NOISE_FIGURE,
            noise_floor_dbm_hz: constants::THERMAL_NOISE_DBM_HZ,
            noise_environment: NoiseEnvironment::default(),

            // Entity configuration
            entity_config: EntityConfig::default(),

            // Multi-agent defaults
            num_jammers: 1,
            num_crs: 1,
            sinr_threshold_db: 10.0,
        }
    }
}

impl RFWorldConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of parallel environments
    ///
    /// # Panics
    /// Panics if `num_envs` is not a multiple of 8 (SIMD lane width)
    pub fn with_num_envs(mut self, num_envs: usize) -> Self {
        assert!(
            num_envs % 8 == 0,
            "num_envs must be a multiple of 8 for SIMD, got {}",
            num_envs
        );
        self.num_envs = num_envs;
        self
    }

    /// Set the number of frequency bins
    ///
    /// # Panics
    /// Panics if `num_bins` is 0 or not a power of 2
    pub fn with_freq_bins(mut self, num_bins: usize) -> Self {
        assert!(num_bins > 0, "num_freq_bins must be positive");
        assert!(
            num_bins.is_power_of_two(),
            "num_freq_bins should be a power of 2 for FFT efficiency, got {}",
            num_bins
        );
        self.num_freq_bins = num_bins;
        self
    }

    /// Set the frequency range
    ///
    /// # Arguments
    /// * `min` - Minimum frequency (Hz)
    /// * `max` - Maximum frequency (Hz)
    ///
    /// # Panics
    /// Panics if `min >= max` or if values are non-positive
    pub fn with_freq_range(mut self, min: f32, max: f32) -> Self {
        assert!(min > 0.0, "freq_min must be positive, got {}", min);
        assert!(max > min, "freq_max must be greater than freq_min");
        self.freq_min = min;
        self.freq_max = max;
        self
    }

    /// Set the maximum number of background entities
    pub fn with_max_entities(mut self, max_entities: usize) -> Self {
        self.max_entities = max_entities;
        self
    }

    /// Set the world size in meters
    pub fn with_world_size(mut self, x: f32, y: f32, z: f32) -> Self {
        assert!(x > 0.0 && y > 0.0 && z > 0.0, "world size must be positive");
        self.world_size = (x, y, z);
        self
    }

    /// Set the physics simulation frequency
    pub fn with_physics_freq(mut self, freq: u32) -> Self {
        assert!(freq > 0, "physics_freq must be positive");
        self.physics_freq = freq;
        self
    }

    /// Set the control/action frequency
    ///
    /// # Panics
    /// Panics if `ctrl_freq` is greater than `physics_freq`
    pub fn with_ctrl_freq(mut self, freq: u32) -> Self {
        assert!(freq > 0, "ctrl_freq must be positive");
        assert!(
            freq <= self.physics_freq,
            "ctrl_freq ({}) must be <= physics_freq ({})",
            freq,
            self.physics_freq
        );
        self.ctrl_freq = freq;
        self
    }

    /// Set the maximum steps per episode
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        assert!(steps > 0, "max_steps must be positive");
        self.max_steps = steps;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the receiver noise figure
    pub fn with_noise_figure(mut self, nf_db: f32) -> Self {
        assert!(nf_db >= 0.0, "noise figure must be non-negative");
        self.noise_figure = nf_db;
        self
    }

    /// Set the noise environment type
    pub fn with_noise_environment(mut self, env: NoiseEnvironment) -> Self {
        self.noise_environment = env;
        self
    }

    /// Set the entity configuration
    pub fn with_entity_config(mut self, config: EntityConfig) -> Self {
        self.entity_config = config;
        self
    }

    /// Set number of TV stations (convenience method)
    pub fn with_tv_stations(mut self, count: usize) -> Self {
        self.entity_config.num_tv_stations = count;
        self
    }

    /// Set number of FM radios (convenience method)
    pub fn with_fm_radios(mut self, count: usize) -> Self {
        self.entity_config.num_fm_radios = count;
        self
    }

    /// Set number of LTE towers (convenience method)
    pub fn with_lte_towers(mut self, count: usize) -> Self {
        self.entity_config.num_lte_towers = count;
        self
    }

    /// Set number of WiFi access points (convenience method)
    pub fn with_wifi_aps(mut self, count: usize) -> Self {
        self.entity_config.num_wifi_aps = count;
        self
    }

    /// Set number of Bluetooth devices (convenience method)
    pub fn with_bluetooth(mut self, count: usize) -> Self {
        self.entity_config.num_bluetooth = count;
        self
    }

    /// Set radar counts (convenience method)
    pub fn with_radars(mut self, sband: usize, weather: usize) -> Self {
        self.entity_config.num_sband_radars = sband;
        self.entity_config.num_weather_radars = weather;
        self
    }

    /// Set drone counts (convenience method)
    pub fn with_drones(mut self, analog: usize, digital: usize) -> Self {
        self.entity_config.num_drone_analog = analog;
        self.entity_config.num_drone_digital = digital;
        self
    }

    /// Set vehicle count (convenience method)
    pub fn with_vehicles(mut self, count: usize) -> Self {
        self.entity_config.num_vehicles = count;
        self
    }

    // ========================================================================
    // Multi-Agent Configuration Methods
    // ========================================================================

    /// Set number of jammer agents per environment
    pub fn with_num_jammers(mut self, num: usize) -> Self {
        self.num_jammers = num;
        self
    }

    /// Set number of cognitive radio agents per environment
    pub fn with_num_crs(mut self, num: usize) -> Self {
        self.num_crs = num;
        self
    }

    /// Set SINR threshold for successful CR communication
    pub fn with_sinr_threshold(mut self, threshold_db: f32) -> Self {
        self.sinr_threshold_db = threshold_db;
        self
    }

    /// Finalize and validate the configuration
    pub fn build(self) -> Self {
        // Validate interdependent settings
        assert!(
            self.num_envs % 8 == 0,
            "num_envs must be a multiple of 8"
        );
        assert!(
            self.ctrl_freq <= self.physics_freq,
            "ctrl_freq must be <= physics_freq"
        );

        self
    }

    // ========================================================================
    // Computed Properties
    // ========================================================================

    /// Get the frequency resolution (Hz per bin)
    #[inline]
    pub fn freq_resolution(&self) -> f32 {
        (self.freq_max - self.freq_min) / self.num_freq_bins as f32
    }

    /// Get the number of physics steps per control step
    #[inline]
    pub fn physics_steps_per_ctrl(&self) -> u32 {
        self.physics_freq / self.ctrl_freq
    }

    /// Get the effective noise floor including noise figure (dBm/Hz)
    #[inline]
    pub fn effective_noise_floor(&self) -> f32 {
        self.noise_floor_dbm_hz + self.noise_figure
    }

    /// Get the total spectrum bandwidth (Hz)
    #[inline]
    pub fn bandwidth(&self) -> f32 {
        self.freq_max - self.freq_min
    }

    /// Get the center frequency (Hz)
    #[inline]
    pub fn center_freq(&self) -> f32 {
        (self.freq_min + self.freq_max) / 2.0
    }

    /// Get the number of SIMD batches needed
    #[inline]
    pub fn num_simd_batches(&self) -> usize {
        self.num_envs / 8
    }

    /// Convert frequency to bin index
    #[inline]
    pub fn freq_to_bin(&self, freq: f32) -> usize {
        let normalized = (freq - self.freq_min) / (self.freq_max - self.freq_min);
        let bin = (normalized * self.num_freq_bins as f32) as usize;
        bin.min(self.num_freq_bins - 1)
    }

    /// Convert bin index to center frequency
    #[inline]
    pub fn bin_to_freq(&self, bin: usize) -> f32 {
        let normalized = (bin as f32 + 0.5) / self.num_freq_bins as f32;
        self.freq_min + normalized * (self.freq_max - self.freq_min)
    }

    // ========================================================================
    // Type-Safe Accessors
    // ========================================================================

    /// Get minimum frequency as type-safe `Hertz`.
    ///
    /// Returns `Some(Hertz)` if the frequency is valid, `None` otherwise.
    #[inline]
    pub fn freq_min_typed(&self) -> Option<Hertz> {
        Hertz::try_new(self.freq_min)
    }

    /// Get maximum frequency as type-safe `Hertz`.
    ///
    /// Returns `Some(Hertz)` if the frequency is valid, `None` otherwise.
    #[inline]
    pub fn freq_max_typed(&self) -> Option<Hertz> {
        Hertz::try_new(self.freq_max)
    }

    /// Get center frequency as type-safe `Hertz`.
    #[inline]
    pub fn center_freq_typed(&self) -> Option<Hertz> {
        Hertz::try_new(self.center_freq())
    }

    /// Get bandwidth as type-safe `Hertz`.
    #[inline]
    pub fn bandwidth_typed(&self) -> Option<Hertz> {
        Hertz::try_new(self.bandwidth())
    }

    /// Get frequency resolution as type-safe `Hertz`.
    #[inline]
    pub fn freq_resolution_typed(&self) -> Option<Hertz> {
        Hertz::try_new(self.freq_resolution())
    }

    /// Create a validated frequency grid from this configuration.
    ///
    /// Returns `Some(ValidatedFrequencyGrid)` if the frequency parameters are valid
    /// (freq_min > 0, freq_max > freq_min, num_bins > 0), `None` otherwise.
    ///
    /// # Example
    /// ```ignore
    /// let config = RFWorldConfig::new()
    ///     .with_freq_range(1e9, 2e9)
    ///     .with_freq_bins(1024)
    ///     .build();
    ///
    /// let grid = config.validated_grid().expect("Invalid frequency config");
    /// assert_eq!(grid.num_bins(), 1024);
    /// ```
    #[inline]
    pub fn validated_grid(&self) -> Option<ValidatedFrequencyGrid> {
        ValidatedFrequencyGrid::from_params(self.freq_min, self.freq_max, self.num_freq_bins)
    }

    /// Get noise floor per bin as type-safe `PositivePower`.
    ///
    /// Converts from dBm/Hz to linear power (Watts) for a single bin.
    #[inline]
    pub fn noise_floor_per_bin_typed(&self) -> PositivePower {
        let dbm_per_bin = self.effective_noise_floor() + 10.0 * self.freq_resolution().log10();
        let linear = 10.0_f32.powf(dbm_per_bin / 10.0) * 0.001; // dBm to Watts
        PositivePower::new(linear.max(0.0))
    }

    /// Convert frequency to bin index using type-safe `Hertz`.
    #[inline]
    pub fn freq_to_bin_typed(&self, freq: Hertz) -> usize {
        self.freq_to_bin(freq.as_hz())
    }

    /// Convert bin index to center frequency as type-safe `Hertz`.
    #[inline]
    pub fn bin_to_freq_typed(&self, bin: usize) -> Option<Hertz> {
        Hertz::try_new(self.bin_to_freq(bin))
    }
}

// ============================================================================
// Preset Configurations
// ============================================================================

impl RFWorldConfig {
    /// Configuration for dense urban scenario
    ///
    /// Higher entity count, more interference sources
    pub fn dense_urban() -> Self {
        Self::new()
            .with_num_envs(64)
            .with_freq_bins(1024)
            .with_max_entities(128)
            .with_world_size(500.0, 500.0, 50.0)
            .with_noise_figure(7.0)
            .build()
    }

    /// Configuration for rural scenario
    ///
    /// Fewer entities, larger world, lower noise
    pub fn rural() -> Self {
        Self::new()
            .with_num_envs(64)
            .with_freq_bins(512)
            .with_max_entities(32)
            .with_world_size(5000.0, 5000.0, 200.0)
            .with_noise_figure(3.0)
            .build()
    }

    /// Configuration for fast training
    ///
    /// Smaller state space for rapid iteration
    pub fn fast_training() -> Self {
        Self::new()
            .with_num_envs(8)
            .with_freq_bins(128)
            .with_max_entities(16)
            .with_max_steps(500)
            .build()
    }

    /// Configuration for 2.4 GHz ISM band focus
    pub fn ism_2_4ghz() -> Self {
        Self::new()
            .with_freq_range(constants::ISM_2_4_START, constants::ISM_2_4_END)
            .with_freq_bins(256)
            .with_max_entities(64)
            .build()
    }

    /// Configuration for drone detection scenario
    pub fn drone_detection() -> Self {
        Self::new()
            .with_freq_range(2.0e9, 6.0e9) // Cover both control and video bands
            .with_freq_bins(512)
            .with_max_entities(48)
            .with_world_size(2000.0, 2000.0, 500.0) // Higher altitude for drones
            .build()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RFWorldConfig::default();
        assert_eq!(config.num_envs, 8);
        assert_eq!(config.num_freq_bins, constants::DEFAULT_NUM_FREQ_BINS);
    }

    #[test]
    fn test_builder_pattern() {
        let config = RFWorldConfig::new()
            .with_num_envs(16)
            .with_freq_bins(256)
            .with_seed(12345)
            .build();

        assert_eq!(config.num_envs, 16);
        assert_eq!(config.num_freq_bins, 256);
        assert_eq!(config.seed, 12345);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of 8")]
    fn test_invalid_num_envs() {
        RFWorldConfig::new().with_num_envs(7);
    }

    #[test]
    #[should_panic(expected = "power of 2")]
    fn test_invalid_freq_bins() {
        RFWorldConfig::new().with_freq_bins(100);
    }

    #[test]
    fn test_freq_to_bin() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .build();

        // 1.5 GHz should be at bin 512 (middle of spectrum)
        let bin = config.freq_to_bin(1.5e9);
        assert_eq!(bin, 512);

        // Edge cases
        assert_eq!(config.freq_to_bin(1e9), 0);
        assert_eq!(config.freq_to_bin(2e9), 1023); // Clamped to max
    }

    #[test]
    fn test_bin_to_freq() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .build();

        // Bin 512 center should be at midpoint
        let freq = config.bin_to_freq(512);
        let expected = 1e9 + (512.5 / 1024.0) * 1e9;
        assert!((freq - expected).abs() < 1e3, "freq {} != expected {}", freq, expected);
    }

    #[test]
    fn test_computed_properties() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .with_physics_freq(100)
            .with_ctrl_freq(10)
            .build();

        let expected_resolution = 1e9 / 1024.0; // ~976562.5 Hz per bin
        assert!((config.freq_resolution() - expected_resolution).abs() < 1.0);
        assert_eq!(config.physics_steps_per_ctrl(), 10);
        assert!((config.bandwidth() - 1e9).abs() < 1.0);
        assert!((config.center_freq() - 1.5e9).abs() < 1.0);
    }

    #[test]
    fn test_preset_configs() {
        // Just ensure presets don't panic
        let _ = RFWorldConfig::dense_urban();
        let _ = RFWorldConfig::rural();
        let _ = RFWorldConfig::fast_training();
        let _ = RFWorldConfig::ism_2_4ghz();
        let _ = RFWorldConfig::drone_detection();
    }

    // ========================================================================
    // Type-Safe Method Tests
    // ========================================================================

    #[test]
    fn test_freq_typed_accessors() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .build();

        let freq_min = config.freq_min_typed().expect("freq_min should be valid");
        let freq_max = config.freq_max_typed().expect("freq_max should be valid");

        assert!((freq_min.as_hz() - 1e9).abs() < 1.0);
        assert!((freq_max.as_hz() - 2e9).abs() < 1.0);
    }

    #[test]
    fn test_validated_grid() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .build();

        let grid = config.validated_grid().expect("Grid should be valid");

        assert_eq!(grid.num_bins(), 1024);
        assert!((grid.freq_min().as_hz() - 1e9).abs() < 1.0);
        assert!((grid.freq_max().as_hz() - 2e9).abs() < 1.0);
    }

    #[test]
    fn test_freq_to_bin_typed() {
        use crate::types::Hertz;

        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .build();

        let center = Hertz::new(1.5e9);
        let bin = config.freq_to_bin_typed(center);

        // Should be middle bin
        assert_eq!(bin, 512);
    }

    #[test]
    fn test_bin_to_freq_typed() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .build();

        let freq = config.bin_to_freq_typed(512).expect("Freq should be valid");

        // Should be around 1.5 GHz
        let expected = 1e9 + (512.5 / 1024.0) * 1e9;
        assert!((freq.as_hz() - expected).abs() < 1e3);
    }

    #[test]
    fn test_noise_floor_typed() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .with_freq_bins(1024)
            .with_noise_figure(6.0)
            .build();

        let noise = config.noise_floor_per_bin_typed();

        // Should be positive
        assert!(noise.watts() > 0.0);
    }

    #[test]
    fn test_bandwidth_typed() {
        let config = RFWorldConfig::new()
            .with_freq_range(1e9, 2e9)
            .build();

        let bw = config.bandwidth_typed().expect("Bandwidth should be valid");
        assert!((bw.as_hz() - 1e9).abs() < 1.0);
    }
}
