//! Snapshot types for capturing environment state for visualization.
//!
//! These types provide zero-copy or efficient capture of RF world state
//! at specific timesteps for rendering.

use std::fmt;

/// Snapshot of a single environment's state at one timestep.
#[derive(Clone)]
pub struct EnvSnapshot {
    /// Simulation step number.
    pub step: u64,
    /// Power Spectral Density [num_freq_bins] in linear power (mW).
    pub psd: Vec<f32>,
    /// Frequency bin centers in Hz.
    pub freq_bins: Vec<f32>,
    /// All entities in this environment.
    pub entities: Vec<EntitySnapshot>,
    /// All jammer agents.
    pub jammers: Vec<AgentSnapshot>,
    /// All cognitive radio agents.
    pub crs: Vec<AgentSnapshot>,
    /// Noise floor in dBm.
    pub noise_floor_dbm: f32,
    /// Cumulative episode return (for tracking learning).
    pub episode_return: f32,
    /// World size (width, height) in meters.
    pub world_size: (f32, f32),
    /// Frequency range (min, max) in Hz.
    pub freq_range: (f32, f32),
}

impl EnvSnapshot {
    /// Create a new snapshot with the given parameters.
    pub fn new(
        step: u64,
        psd: Vec<f32>,
        freq_bins: Vec<f32>,
        noise_floor_dbm: f32,
        world_size: (f32, f32),
        freq_range: (f32, f32),
    ) -> Self {
        Self {
            step,
            psd,
            freq_bins,
            entities: Vec::new(),
            jammers: Vec::new(),
            crs: Vec::new(),
            noise_floor_dbm,
            episode_return: 0.0,
            world_size,
            freq_range,
        }
    }

    /// Get PSD values in dBm.
    ///
    /// Note: PSD is stored in linear watts, not mW.
    /// dBm = 10 * log10(watts * 1000) = 10 * log10(watts) + 30
    pub fn psd_dbm(&self) -> Vec<f32> {
        self.psd
            .iter()
            .map(|&p| if p > 0.0 { 10.0 * p.log10() + 30.0 } else { -120.0 })
            .collect()
    }

    /// Get the number of frequency bins.
    pub fn num_freq_bins(&self) -> usize {
        self.psd.len()
    }

    /// Add an entity to the snapshot.
    pub fn add_entity(&mut self, entity: EntitySnapshot) {
        self.entities.push(entity);
    }

    /// Add a jammer agent to the snapshot.
    pub fn add_jammer(&mut self, jammer: AgentSnapshot) {
        self.jammers.push(jammer);
    }

    /// Add a cognitive radio agent to the snapshot.
    pub fn add_cr(&mut self, cr: AgentSnapshot) {
        self.crs.push(cr);
    }
}

impl fmt::Debug for EnvSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnvSnapshot")
            .field("step", &self.step)
            .field("num_freq_bins", &self.psd.len())
            .field("num_entities", &self.entities.len())
            .field("num_jammers", &self.jammers.len())
            .field("num_crs", &self.crs.len())
            .field("noise_floor_dbm", &self.noise_floor_dbm)
            .field("episode_return", &self.episode_return)
            .finish()
    }
}

/// Entity type enumeration matching the RF environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityType {
    /// Television broadcast station (continuous, high power).
    TVStation,
    /// FM radio station (continuous, moderate power).
    FMRadio,
    /// LTE cellular tower (bursty, moderate power).
    LTETower,
    /// WiFi access point (bursty, low power).
    WiFiAP,
    /// S-band radar (periodic pulses, very high power).
    SBandRadar,
    /// X-band weather radar (periodic, high power).
    XBandRadar,
    /// Amateur radio operator (intermittent).
    AmateurRadio,
    /// Drone with analog video (continuous, mobile).
    DroneAnalog,
    /// Drone with digital link (bursty, mobile).
    DroneDigital,
    /// Frequency-hopping spread spectrum device.
    FHSS,
    /// Direct-sequence spread spectrum device.
    DSSS,
    /// Generic/unknown entity.
    Unknown,
}

impl EntityType {
    /// Get the display color for this entity type [R, G, B].
    pub fn color(&self) -> [u8; 3] {
        match self {
            EntityType::TVStation => [100, 149, 237],   // Cornflower blue
            EntityType::FMRadio => [138, 43, 226],      // Blue violet
            EntityType::LTETower => [255, 165, 0],      // Orange
            EntityType::WiFiAP => [50, 205, 50],        // Lime green
            EntityType::SBandRadar => [255, 0, 0],      // Red
            EntityType::XBandRadar => [220, 20, 60],    // Crimson
            EntityType::AmateurRadio => [255, 215, 0],  // Gold
            EntityType::DroneAnalog => [186, 85, 211],  // Medium orchid
            EntityType::DroneDigital => [147, 112, 219],// Medium purple
            EntityType::FHSS => [0, 191, 255],          // Deep sky blue
            EntityType::DSSS => [64, 224, 208],         // Turquoise
            EntityType::Unknown => [128, 128, 128],     // Gray
        }
    }

    /// Get the icon shape for this entity type.
    pub fn icon(&self) -> EntityIcon {
        match self {
            EntityType::TVStation => EntityIcon::Tower,
            EntityType::FMRadio => EntityIcon::Tower,
            EntityType::LTETower => EntityIcon::Triangle,
            EntityType::WiFiAP => EntityIcon::Diamond,
            EntityType::SBandRadar => EntityIcon::Pentagon,
            EntityType::XBandRadar => EntityIcon::Pentagon,
            EntityType::AmateurRadio => EntityIcon::Circle,
            EntityType::DroneAnalog => EntityIcon::Drone,
            EntityType::DroneDigital => EntityIcon::Drone,
            EntityType::FHSS => EntityIcon::Square,
            EntityType::DSSS => EntityIcon::Square,
            EntityType::Unknown => EntityIcon::Circle,
        }
    }

    /// Get a short label for this entity type.
    pub fn label(&self) -> &'static str {
        match self {
            EntityType::TVStation => "TV",
            EntityType::FMRadio => "FM",
            EntityType::LTETower => "LTE",
            EntityType::WiFiAP => "WiFi",
            EntityType::SBandRadar => "S-Radar",
            EntityType::XBandRadar => "X-Radar",
            EntityType::AmateurRadio => "HAM",
            EntityType::DroneAnalog => "UAV-A",
            EntityType::DroneDigital => "UAV-D",
            EntityType::FHSS => "FHSS",
            EntityType::DSSS => "DSSS",
            EntityType::Unknown => "UNK",
        }
    }
}

/// Icon shapes for entity visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityIcon {
    Circle,
    Square,
    Triangle,
    Diamond,
    Pentagon,
    Tower,
    Drone,
}

/// Snapshot of a single entity's state.
#[derive(Debug, Clone)]
pub struct EntitySnapshot {
    /// Entity type.
    pub entity_type: EntityType,
    /// X position in meters.
    pub x: f32,
    /// Y position in meters.
    pub y: f32,
    /// X velocity in m/s (for mobile entities).
    pub vx: f32,
    /// Y velocity in m/s (for mobile entities).
    pub vy: f32,
    /// Center frequency in Hz.
    pub freq: f32,
    /// Bandwidth in Hz.
    pub bandwidth: f32,
    /// Transmit power in dBm.
    pub power_dbm: f32,
    /// Whether currently transmitting.
    pub active: bool,
    /// Optional label/identifier.
    pub label: Option<String>,
}

impl EntitySnapshot {
    /// Create a new entity snapshot.
    pub fn new(entity_type: EntityType, x: f32, y: f32, freq: f32, bandwidth: f32, power_dbm: f32) -> Self {
        Self {
            entity_type,
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            freq,
            bandwidth,
            power_dbm,
            active: true,
            label: None,
        }
    }

    /// Set velocity.
    pub fn with_velocity(mut self, vx: f32, vy: f32) -> Self {
        self.vx = vx;
        self.vy = vy;
        self
    }

    /// Set active status.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Set label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Check if this entity is mobile (has velocity).
    pub fn is_mobile(&self) -> bool {
        self.vx.abs() > 0.01 || self.vy.abs() > 0.01
    }

    /// Get speed in m/s.
    pub fn speed(&self) -> f32 {
        (self.vx * self.vx + self.vy * self.vy).sqrt()
    }

    /// Get heading in radians (0 = east, PI/2 = north).
    pub fn heading(&self) -> f32 {
        self.vy.atan2(self.vx)
    }
}

/// Agent type (jammer or cognitive radio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentType {
    Jammer,
    CognitiveRadio,
}

impl AgentType {
    /// Get the display color for this agent type.
    pub fn color(&self) -> [u8; 3] {
        match self {
            AgentType::Jammer => [255, 0, 0],       // Red
            AgentType::CognitiveRadio => [0, 200, 0], // Green
        }
    }
}

/// Snapshot of a single agent's state.
#[derive(Debug, Clone)]
pub struct AgentSnapshot {
    /// Agent type.
    pub agent_type: AgentType,
    /// Agent index within its type.
    pub index: usize,
    /// X position in meters.
    pub x: f32,
    /// Y position in meters.
    pub y: f32,
    /// Center frequency in Hz.
    pub freq: f32,
    /// Bandwidth in Hz.
    pub bandwidth: f32,
    /// Transmit power in dBm.
    pub power_dbm: f32,
    /// SINR in dB (CR only, None for jammers).
    pub sinr_db: Option<f32>,
    /// Target entity/agent index (Jammer only, None for CRs).
    pub target_idx: Option<usize>,
    /// Whether currently active/transmitting.
    pub active: bool,
    /// Cumulative reward this episode.
    pub cumulative_reward: f32,
}

impl AgentSnapshot {
    /// Create a new jammer snapshot.
    pub fn new_jammer(
        index: usize,
        x: f32,
        y: f32,
        freq: f32,
        bandwidth: f32,
        power_dbm: f32,
    ) -> Self {
        Self {
            agent_type: AgentType::Jammer,
            index,
            x,
            y,
            freq,
            bandwidth,
            power_dbm,
            sinr_db: None,
            target_idx: None,
            active: true,
            cumulative_reward: 0.0,
        }
    }

    /// Create a new cognitive radio snapshot.
    pub fn new_cr(
        index: usize,
        x: f32,
        y: f32,
        freq: f32,
        bandwidth: f32,
        power_dbm: f32,
        sinr_db: f32,
    ) -> Self {
        Self {
            agent_type: AgentType::CognitiveRadio,
            index,
            x,
            y,
            freq,
            bandwidth,
            power_dbm,
            sinr_db: Some(sinr_db),
            target_idx: None,
            active: true,
            cumulative_reward: 0.0,
        }
    }

    /// Set target index (for jammers).
    pub fn with_target(mut self, target: usize) -> Self {
        self.target_idx = Some(target);
        self
    }

    /// Set active status.
    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    /// Set cumulative reward.
    pub fn with_reward(mut self, reward: f32) -> Self {
        self.cumulative_reward = reward;
        self
    }

    /// Check if this is a jammer.
    pub fn is_jammer(&self) -> bool {
        self.agent_type == AgentType::Jammer
    }

    /// Check if this CR is being jammed (low SINR).
    pub fn is_jammed(&self, threshold_db: f32) -> bool {
        if let Some(sinr) = self.sinr_db {
            sinr < threshold_db
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_snapshot() {
        let entity = EntitySnapshot::new(
            EntityType::LTETower,
            100.0, 200.0,
            1800e6, 10e6, 43.0,
        )
        .with_velocity(5.0, 0.0)
        .with_active(true);

        assert!(entity.is_mobile());
        assert!((entity.speed() - 5.0).abs() < 0.01);
        assert!((entity.heading()).abs() < 0.01); // Heading east
    }

    #[test]
    fn test_agent_snapshot() {
        let jammer = AgentSnapshot::new_jammer(0, 50.0, 50.0, 2400e6, 20e6, 30.0)
            .with_target(2);
        assert!(jammer.is_jammer());
        assert_eq!(jammer.target_idx, Some(2));

        let cr = AgentSnapshot::new_cr(0, 75.0, 75.0, 2450e6, 5e6, 10.0, 5.0);
        assert!(!cr.is_jammer());
        assert!(cr.is_jammed(10.0)); // SINR 5 dB < 10 dB threshold
        assert!(!cr.is_jammed(3.0)); // SINR 5 dB > 3 dB threshold
    }

    #[test]
    fn test_psd_dbm_conversion() {
        // PSD is stored in watts (not mW), so dBm = 10*log10(watts) + 30
        // 1 W = 30 dBm, 0.001 W = 0 dBm, 0.000001 W = -30 dBm
        let snapshot = EnvSnapshot::new(
            0,
            vec![1.0, 0.001, 0.000001], // 30 dBm, 0 dBm, -30 dBm (in watts)
            vec![1e9, 2e9, 3e9],
            -100.0,
            (1000.0, 1000.0),
            (1e9, 3e9),
        );

        let psd_dbm = snapshot.psd_dbm();
        assert!((psd_dbm[0] - 30.0).abs() < 0.01);  // 1 W = 30 dBm
        assert!((psd_dbm[1] - 0.0).abs() < 0.01);   // 1 mW = 0 dBm
        assert!((psd_dbm[2] - (-30.0)).abs() < 0.01); // 1 uW = -30 dBm
    }
}
