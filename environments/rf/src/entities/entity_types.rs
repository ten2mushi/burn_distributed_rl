//! Entity Type Definitions
//!
//! Defines the core entity types and modulation schemes for RF environment simulation.
//! Phase 2 implements 10 core types covering continuous, bursty, periodic, and mobile entities.

// ============================================================================
// Entity Type Enumeration
// ============================================================================

/// Core entity types for RF environment simulation.
///
/// Entities are categorized by their temporal behavior:
/// - **Continuous**: Always active, 100% duty cycle (TV, FM)
/// - **Bursty**: Stochastic activation with Poisson arrivals (LTE, WiFi, Bluetooth)
/// - **Periodic**: Deterministic pulse trains (Radar)
/// - **Mobile**: Moving entities with Doppler effects (Drones, Vehicles)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EntityType {
    // ========================================================================
    // Continuous (100% duty cycle)
    // ========================================================================
    /// ATSC/DVB-T digital television broadcast
    /// Frequency: 470-608 MHz, Bandwidth: 6 MHz rectangular
    TVStation = 0,

    /// FM radio broadcast
    /// Frequency: 88-108 MHz, Bandwidth: 200 kHz Gaussian
    FMRadio = 1,

    /// AM radio broadcast
    /// Frequency: 535-1705 kHz (MF band), Bandwidth: 10 kHz
    AMRadio = 2,

    /// Global Navigation Satellite System (GPS, Galileo, GLONASS, BeiDou)
    /// Frequency: 1575.42 MHz (L1), Bandwidth: ~2 MHz DSSS
    /// Power: Very weak (-130 dBm at surface), always present
    GNSS = 3,

    /// Geostationary Satellite beacon
    /// Frequency: 3.7-4.2 GHz (C-band), 10.7-12.75 GHz (Ku-band)
    /// Bandwidth: Narrow beacon (~100 kHz), continuous
    GeoSatellite = 4,

    // ========================================================================
    // Bursty (Poisson arrivals)
    // ========================================================================
    /// LTE/5G base station downlink
    /// Frequency: 700-2100 MHz, Bandwidth: 10-20 MHz OFDM
    LTETower = 10,

    /// WiFi access point (802.11ac/ax)
    /// Frequency: 2.4/5 GHz, Bandwidth: 20-80 MHz OFDM
    WiFiAP = 12,

    /// Bluetooth Low Energy device
    /// Frequency: 2.4 GHz ISM, Bandwidth: 1 MHz, FHSS 79 channels
    Bluetooth = 13,

    /// LTE Uplink (User Equipment to Tower)
    /// Frequency: Varies by band, Bandwidth: Variable SC-FDM
    /// Bursty based on user data transmission
    LTEUplink = 14,

    /// Zigbee IoT device (IEEE 802.15.4)
    /// Frequency: 2.4 GHz ISM (channels 11-26), Bandwidth: 2 MHz DSSS
    /// Very short periodic bursts (~0.01 duty cycle)
    Zigbee = 15,

    /// LoRaWAN IoT device
    /// Frequency: 915 MHz (US), Bandwidth: 125/250/500 kHz chirp
    /// Very low duty cycle (<0.01), slow upchirps
    LoRaWAN = 16,

    /// Satellite IoT (LEO constellation)
    /// Frequency: 1.6 GHz L-band, Bandwidth: ~100 kHz
    /// Slotted ALOHA random access, very low duty cycle
    SatelliteIoT = 17,

    // ========================================================================
    // Periodic (pulsed)
    // ========================================================================
    /// S-Band surveillance/tracking radar
    /// Frequency: 2-4 GHz, High-power chirp pulses, PRI ~1 ms
    SBandRadar = 20,

    /// Weather radar (NEXRAD-like)
    /// Frequency: 2.7-2.9 GHz, Chirp pulses, PRI ~3 ms
    WeatherRadar = 21,

    /// L-Band Radar (Air Traffic Control, Surveillance)
    /// Frequency: 1-2 GHz, Chirp pulses, PRI ~1-3 ms
    LBandRadar = 22,

    /// ADS-B (Automatic Dependent Surveillance-Broadcast)
    /// Frequency: 1090 MHz, PPM modulation, 120 µs bursts
    /// Aircraft transponder, asynchronous transmissions
    ADSB = 23,

    /// DME (Distance Measuring Equipment)
    /// Frequency: 960-1215 MHz, Pulse pairs (12 µs apart)
    /// Aviation navigation aid
    DME = 24,

    // ========================================================================
    // Mobile (Doppler effects, Despawn/Respawn at boundaries)
    // ========================================================================
    /// Analog video drone transmission
    /// Frequency: 5.8 GHz, FM video + micro-Doppler sidebands
    DroneAnalog = 30,

    /// Digital video drone transmission
    /// Frequency: 5.8 GHz, OFDM video with ICI from Doppler
    DroneDigital = 31,

    /// Vehicle-to-Everything (V2X) communication
    /// Frequency: 5.9 GHz, Bandwidth: 10 MHz OFDM
    Vehicle = 33,

    // ========================================================================
    // Voice Radios (PTT behavior, mobile/handheld)
    // ========================================================================
    /// Maritime VHF radio
    /// Frequency: 156-174 MHz, Bandwidth: 25 kHz FM
    /// Push-to-talk voice communication for ships and coast stations
    MaritimeVHF = 40,

    /// Walkie-Talkie (FRS - Family Radio Service)
    /// Frequency: 462 MHz, Bandwidth: 12.5 kHz FM
    /// Low-power handheld, PTT sporadic usage
    WalkieTalkie = 41,

    /// GMRS (General Mobile Radio Service)
    /// Frequency: 462-467 MHz, Bandwidth: 25 kHz FM
    /// Higher power than FRS, may use repeaters
    GMRS = 42,

    /// P25 Public Safety Radio
    /// Frequency: 700/800 MHz, C4FM modulation (12.5 kHz)
    /// Digital trunked radio for first responders
    P25Radio = 43,

    /// Amateur (Ham) Radio
    /// Frequency: Multi-band (2m: 144 MHz, 70cm: 430 MHz)
    /// Multi-mode: SSB, FM, CW, Digital
    AmateurRadio = 44,

    // ========================================================================
    // Specialized / Low-Power
    // ========================================================================
    /// Smart Meter (AMI/AMR)
    /// Frequency: 900 MHz ISM (US), Bandwidth: ~100 kHz FSK
    /// Scheduled periodic transmission (hourly/daily)
    SmartMeter = 50,

    /// Microwave Oven leakage
    /// Frequency: 2.45 GHz ISM, Bandwidth: ~20 MHz messy peak
    /// 60 Hz cyclic modulation from magnetron
    MicrowaveOven = 51,

    /// RFID Tag (passive/semi-passive)
    /// Frequency: 13.56 MHz (HF) or 900 MHz (UHF)
    /// Reactive backscatter when interrogated
    RFIDTag = 52,

    /// Key Fob (automotive/garage)
    /// Frequency: 315 MHz (US), 433 MHz (EU/Asia)
    /// OOK/ASK burst, <100 ms duration
    KeyFob = 53,

    /// Wireless Microphone
    /// Frequency: UHF TV white space (470-698 MHz)
    /// Continuous narrow FM, ~200 kHz bandwidth
    WirelessMic = 54,

    /// Baby Monitor
    /// Frequency: 900 MHz or 2.4 GHz ISM
    /// Continuous FM (analog) or FHSS (digital)
    BabyMonitor = 55,

    /// Weather Balloon (Radiosonde)
    /// Frequency: 400-406 MHz (meteorological band), GFSK modulation
    /// Altitude-varying propagation (0-35 km), continuous telemetry
    WeatherBalloon = 56,

    /// Deep Space Network (DSN)
    /// Frequency: 2.1 GHz S-band, ultra-weak CW beacon
    /// Extreme distance (AU scale), -160 to -180 dBm at Earth
    DeepSpaceNetwork = 57,

    // ========================================================================
    // Reserved for Future Phases
    // ========================================================================
    // /// Cognitive radio agent (Phase 4)
    // CognitiveRadio = 100,
    // /// Jammer agent (Phase 4)
    // Jammer = 101,
}

impl EntityType {
    /// Returns true if this entity type has 100% duty cycle.
    #[inline]
    pub const fn is_continuous(&self) -> bool {
        matches!(
            self,
            EntityType::TVStation
                | EntityType::FMRadio
                | EntityType::AMRadio
                | EntityType::GNSS
                | EntityType::GeoSatellite
        )
    }

    /// Returns true if this entity type uses stochastic (Poisson) activation.
    #[inline]
    pub const fn is_bursty(&self) -> bool {
        matches!(
            self,
            EntityType::LTETower
                | EntityType::WiFiAP
                | EntityType::Bluetooth
                | EntityType::LTEUplink
                | EntityType::Zigbee
                | EntityType::LoRaWAN
                | EntityType::SatelliteIoT
        )
    }

    /// Returns true if this entity type uses deterministic pulse trains.
    #[inline]
    pub const fn is_periodic(&self) -> bool {
        matches!(
            self,
            EntityType::SBandRadar
                | EntityType::WeatherRadar
                | EntityType::LBandRadar
                | EntityType::ADSB
                | EntityType::DME
        )
    }

    /// Returns true if this entity type has mobility and Doppler effects.
    #[inline]
    pub const fn is_mobile(&self) -> bool {
        matches!(
            self,
            EntityType::DroneAnalog
                | EntityType::DroneDigital
                | EntityType::Vehicle
                | EntityType::MaritimeVHF
                | EntityType::WalkieTalkie
                | EntityType::GMRS
                | EntityType::P25Radio
                | EntityType::AmateurRadio
        )
    }

    /// Returns true if this entity type is specialized/low-power.
    #[inline]
    pub const fn is_specialized(&self) -> bool {
        matches!(
            self,
            EntityType::SmartMeter
                | EntityType::MicrowaveOven
                | EntityType::RFIDTag
                | EntityType::KeyFob
                | EntityType::WirelessMic
                | EntityType::BabyMonitor
                | EntityType::WeatherBalloon
                | EntityType::DeepSpaceNetwork
        )
    }

    /// Returns true if this entity uses scheduled transmission (not random).
    #[inline]
    pub const fn uses_scheduled(&self) -> bool {
        matches!(self, EntityType::SmartMeter)
    }

    /// Returns true if this entity is reactive (responds to interrogation).
    #[inline]
    pub const fn is_reactive(&self) -> bool {
        matches!(self, EntityType::RFIDTag)
    }

    /// Returns true if this entity has 60 Hz cyclic interference.
    #[inline]
    pub const fn has_ac_modulation(&self) -> bool {
        matches!(self, EntityType::MicrowaveOven)
    }

    /// Returns true if this entity uses OOK modulation.
    #[inline]
    pub const fn uses_ook(&self) -> bool {
        matches!(self, EntityType::KeyFob)
    }

    /// Returns true if this entity uses slotted ALOHA random access.
    #[inline]
    pub const fn uses_slotted_aloha(&self) -> bool {
        matches!(self, EntityType::SatelliteIoT)
    }

    /// Returns true if this entity has altitude-varying propagation.
    #[inline]
    pub const fn is_altitude_varying(&self) -> bool {
        matches!(self, EntityType::WeatherBalloon)
    }

    /// Returns true if this entity is a deep space signal (extreme distance).
    #[inline]
    pub const fn is_deep_space(&self) -> bool {
        matches!(self, EntityType::DeepSpaceNetwork)
    }

    /// Returns true if this entity type uses push-to-talk behavior.
    #[inline]
    pub const fn uses_ptt(&self) -> bool {
        matches!(
            self,
            EntityType::MaritimeVHF
                | EntityType::WalkieTalkie
                | EntityType::GMRS
                | EntityType::P25Radio
                | EntityType::AmateurRadio
        )
    }

    /// Returns true if this entity uses C4FM modulation (P25 digital voice).
    #[inline]
    pub const fn uses_c4fm(&self) -> bool {
        matches!(self, EntityType::P25Radio)
    }

    /// Returns true if this entity can use SSB modulation.
    #[inline]
    pub const fn uses_ssb(&self) -> bool {
        matches!(self, EntityType::AmateurRadio)
    }

    /// Returns true if this entity uses frequency hopping spread spectrum.
    #[inline]
    pub const fn uses_fhss(&self) -> bool {
        matches!(self, EntityType::Bluetooth)
    }

    /// Returns true if this entity uses direct sequence spread spectrum.
    #[inline]
    pub const fn uses_dsss(&self) -> bool {
        matches!(self, EntityType::GNSS | EntityType::Zigbee)
    }

    /// Returns true if this entity uses chirp spread spectrum.
    #[inline]
    pub const fn uses_chirp(&self) -> bool {
        matches!(
            self,
            EntityType::SBandRadar
                | EntityType::WeatherRadar
                | EntityType::LBandRadar
                | EntityType::LoRaWAN
        )
    }

    /// Returns true if this entity uses PPM (Pulse Position Modulation).
    #[inline]
    pub const fn uses_ppm(&self) -> bool {
        matches!(self, EntityType::ADSB)
    }

    /// Returns true if this entity uses pulse pairs.
    #[inline]
    pub const fn uses_pulse_pairs(&self) -> bool {
        matches!(self, EntityType::DME)
    }

    /// Returns true if this entity uses SC-FDM modulation.
    #[inline]
    pub const fn uses_scfdm(&self) -> bool {
        matches!(self, EntityType::LTEUplink)
    }

    /// Get the behavior category as a string for debugging.
    pub const fn behavior_category(&self) -> &'static str {
        if self.is_continuous() {
            "continuous"
        } else if self.is_bursty() {
            "bursty"
        } else if self.is_periodic() {
            "periodic"
        } else if self.is_mobile() {
            "mobile"
        } else if self.is_specialized() {
            "specialized"
        } else {
            "unknown"
        }
    }

    /// Get the total number of entity types defined.
    pub const fn count() -> usize {
        33
    }

    /// Iterate over all entity types.
    pub fn all() -> impl Iterator<Item = EntityType> {
        [
            // Continuous
            EntityType::TVStation,
            EntityType::FMRadio,
            EntityType::AMRadio,
            EntityType::GNSS,
            EntityType::GeoSatellite,
            // Bursty
            EntityType::LTETower,
            EntityType::WiFiAP,
            EntityType::Bluetooth,
            EntityType::LTEUplink,
            EntityType::Zigbee,
            EntityType::LoRaWAN,
            EntityType::SatelliteIoT,
            // Periodic
            EntityType::SBandRadar,
            EntityType::WeatherRadar,
            EntityType::LBandRadar,
            EntityType::ADSB,
            EntityType::DME,
            // Mobile
            EntityType::DroneAnalog,
            EntityType::DroneDigital,
            EntityType::Vehicle,
            // Voice Radios (PTT)
            EntityType::MaritimeVHF,
            EntityType::WalkieTalkie,
            EntityType::GMRS,
            EntityType::P25Radio,
            EntityType::AmateurRadio,
            // Specialized / Low-Power
            EntityType::SmartMeter,
            EntityType::MicrowaveOven,
            EntityType::RFIDTag,
            EntityType::KeyFob,
            EntityType::WirelessMic,
            EntityType::BabyMonitor,
            EntityType::WeatherBalloon,
            EntityType::DeepSpaceNetwork,
        ]
        .into_iter()
    }
}

impl From<u8> for EntityType {
    fn from(value: u8) -> Self {
        match value {
            // Continuous
            0 => EntityType::TVStation,
            1 => EntityType::FMRadio,
            2 => EntityType::AMRadio,
            3 => EntityType::GNSS,
            4 => EntityType::GeoSatellite,
            // Bursty
            10 => EntityType::LTETower,
            12 => EntityType::WiFiAP,
            13 => EntityType::Bluetooth,
            14 => EntityType::LTEUplink,
            15 => EntityType::Zigbee,
            16 => EntityType::LoRaWAN,
            17 => EntityType::SatelliteIoT,
            // Periodic
            20 => EntityType::SBandRadar,
            21 => EntityType::WeatherRadar,
            22 => EntityType::LBandRadar,
            23 => EntityType::ADSB,
            24 => EntityType::DME,
            // Mobile
            30 => EntityType::DroneAnalog,
            31 => EntityType::DroneDigital,
            33 => EntityType::Vehicle,
            // Voice Radios (PTT)
            40 => EntityType::MaritimeVHF,
            41 => EntityType::WalkieTalkie,
            42 => EntityType::GMRS,
            43 => EntityType::P25Radio,
            44 => EntityType::AmateurRadio,
            // Specialized / Low-Power
            50 => EntityType::SmartMeter,
            51 => EntityType::MicrowaveOven,
            52 => EntityType::RFIDTag,
            53 => EntityType::KeyFob,
            54 => EntityType::WirelessMic,
            55 => EntityType::BabyMonitor,
            56 => EntityType::WeatherBalloon,
            57 => EntityType::DeepSpaceNetwork,
            _ => EntityType::TVStation, // Default fallback
        }
    }
}

impl From<EntityType> for u8 {
    fn from(value: EntityType) -> Self {
        value as u8
    }
}

// ============================================================================
// Modulation Type Enumeration
// ============================================================================

/// Modulation schemes used by RF entities.
///
/// Covers analog, digital narrowband, digital wideband, spread spectrum,
/// and jamming modulation types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ModulationType {
    // ========================================================================
    // Analog
    // ========================================================================
    /// Continuous Wave (unmodulated carrier)
    CW = 0,
    /// Amplitude Modulation
    AM = 1,
    /// Frequency Modulation
    FM = 2,
    /// Single Sideband
    SSB = 3,

    // ========================================================================
    // Digital Narrowband
    // ========================================================================
    /// Binary Phase Shift Keying
    BPSK = 10,
    /// Quadrature Phase Shift Keying
    QPSK = 11,
    /// 16-Quadrature Amplitude Modulation
    QAM16 = 12,
    /// 64-Quadrature Amplitude Modulation
    QAM64 = 13,
    /// 256-Quadrature Amplitude Modulation
    QAM256 = 14,
    /// Gaussian Frequency Shift Keying
    GFSK = 15,
    /// On-Off Keying (simple ASK)
    OOK = 16,

    // ========================================================================
    // Digital Wideband
    // ========================================================================
    /// Orthogonal Frequency Division Multiplexing
    OFDM = 20,
    /// Coded OFDM (used in DVB-T)
    COFDM = 21,
    /// Single Carrier FDMA (used in LTE uplink)
    SCFDMA = 22,

    // ========================================================================
    // Spread Spectrum
    // ========================================================================
    /// Direct Sequence Spread Spectrum
    DSSS = 30,
    /// Frequency Hopping Spread Spectrum
    FHSS = 31,
    /// Linear Frequency Modulation (Chirp)
    Chirp = 32,

    // ========================================================================
    // Jamming (Phase 4)
    // ========================================================================
    /// Barrage jamming (wideband noise)
    Barrage = 40,
    /// Spot jamming (narrowband)
    Spot = 41,
    /// Sweep jamming (frequency sweep)
    Sweep = 42,
    /// Reactive jamming (detect-and-jam)
    Reactive = 43,
}

impl ModulationType {
    /// Returns true if this is an analog modulation type.
    #[inline]
    pub const fn is_analog(&self) -> bool {
        (*self as u8) < 10
    }

    /// Returns true if this is a digital narrowband modulation type.
    #[inline]
    pub const fn is_digital_narrowband(&self) -> bool {
        let v = *self as u8;
        v >= 10 && v < 20
    }

    /// Returns true if this is a digital wideband modulation type.
    #[inline]
    pub const fn is_digital_wideband(&self) -> bool {
        let v = *self as u8;
        v >= 20 && v < 30
    }

    /// Returns true if this is a spread spectrum modulation type.
    #[inline]
    pub const fn is_spread_spectrum(&self) -> bool {
        let v = *self as u8;
        v >= 30 && v < 40
    }

    /// Returns true if this is a jamming modulation type.
    #[inline]
    pub const fn is_jamming(&self) -> bool {
        (*self as u8) >= 40
    }
}

impl From<u8> for ModulationType {
    fn from(value: u8) -> Self {
        match value {
            0 => ModulationType::CW,
            1 => ModulationType::AM,
            2 => ModulationType::FM,
            3 => ModulationType::SSB,
            10 => ModulationType::BPSK,
            11 => ModulationType::QPSK,
            12 => ModulationType::QAM16,
            13 => ModulationType::QAM64,
            14 => ModulationType::QAM256,
            15 => ModulationType::GFSK,
            16 => ModulationType::OOK,
            20 => ModulationType::OFDM,
            21 => ModulationType::COFDM,
            22 => ModulationType::SCFDMA,
            30 => ModulationType::DSSS,
            31 => ModulationType::FHSS,
            32 => ModulationType::Chirp,
            40 => ModulationType::Barrage,
            41 => ModulationType::Spot,
            42 => ModulationType::Sweep,
            43 => ModulationType::Reactive,
            _ => ModulationType::CW, // Default fallback
        }
    }
}

impl From<ModulationType> for u8 {
    fn from(value: ModulationType) -> Self {
        value as u8
    }
}

// ============================================================================
// Default Modulation for Entity Types
// ============================================================================

impl EntityType {
    /// Get the default modulation type for this entity.
    pub const fn default_modulation(&self) -> ModulationType {
        match self {
            // Continuous
            EntityType::TVStation => ModulationType::COFDM,
            EntityType::FMRadio => ModulationType::FM,
            EntityType::AMRadio => ModulationType::AM,
            EntityType::GNSS => ModulationType::DSSS,
            EntityType::GeoSatellite => ModulationType::QPSK, // Typical for satellite beacons
            // Bursty
            EntityType::LTETower => ModulationType::OFDM,
            EntityType::WiFiAP => ModulationType::OFDM,
            EntityType::Bluetooth => ModulationType::GFSK,
            EntityType::LTEUplink => ModulationType::SCFDMA,
            EntityType::Zigbee => ModulationType::DSSS,
            EntityType::LoRaWAN => ModulationType::Chirp,
            // Periodic
            EntityType::SBandRadar => ModulationType::Chirp,
            EntityType::WeatherRadar => ModulationType::Chirp,
            EntityType::LBandRadar => ModulationType::Chirp,
            EntityType::ADSB => ModulationType::BPSK, // PPM encoded as BPSK
            EntityType::DME => ModulationType::CW,    // Pulse pairs, essentially CW bursts
            // Mobile
            EntityType::DroneAnalog => ModulationType::FM,
            EntityType::DroneDigital => ModulationType::OFDM,
            EntityType::Vehicle => ModulationType::OFDM,
            // Voice Radios (PTT)
            EntityType::MaritimeVHF => ModulationType::FM, // Analog FM voice
            EntityType::WalkieTalkie => ModulationType::FM, // Analog FM voice
            EntityType::GMRS => ModulationType::FM,        // Analog FM voice
            EntityType::P25Radio => ModulationType::GFSK,  // C4FM is a form of 4-level FSK
            EntityType::AmateurRadio => ModulationType::SSB, // Most common for HF/VHF DX
            // Specialized / Low-Power
            EntityType::SmartMeter => ModulationType::GFSK,   // FSK-based AMR protocol
            EntityType::MicrowaveOven => ModulationType::CW,  // Magnetron leakage (unmodulated)
            EntityType::RFIDTag => ModulationType::OOK,       // Backscatter modulation
            EntityType::KeyFob => ModulationType::OOK,        // OOK/ASK burst
            EntityType::WirelessMic => ModulationType::FM,    // Narrowband FM audio
            EntityType::BabyMonitor => ModulationType::FM,         // Analog FM or FHSS
            EntityType::WeatherBalloon => ModulationType::GFSK,      // Radiosonde GFSK telemetry
            EntityType::DeepSpaceNetwork => ModulationType::CW,      // Weak CW beacon
            EntityType::SatelliteIoT => ModulationType::GFSK,        // Narrowband IoT
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_categories() {
        // Continuous
        assert!(EntityType::TVStation.is_continuous());
        assert!(EntityType::FMRadio.is_continuous());
        assert!(EntityType::AMRadio.is_continuous());
        assert!(EntityType::GNSS.is_continuous());
        assert!(EntityType::GeoSatellite.is_continuous());
        assert!(!EntityType::LTETower.is_continuous());

        // Bursty
        assert!(EntityType::LTETower.is_bursty());
        assert!(EntityType::WiFiAP.is_bursty());
        assert!(EntityType::Bluetooth.is_bursty());
        assert!(EntityType::LTEUplink.is_bursty());
        assert!(EntityType::Zigbee.is_bursty());
        assert!(EntityType::LoRaWAN.is_bursty());
        assert!(!EntityType::TVStation.is_bursty());

        // Periodic
        assert!(EntityType::SBandRadar.is_periodic());
        assert!(EntityType::WeatherRadar.is_periodic());
        assert!(EntityType::LBandRadar.is_periodic());
        assert!(EntityType::ADSB.is_periodic());
        assert!(EntityType::DME.is_periodic());
        assert!(!EntityType::WiFiAP.is_periodic());

        // Mobile
        assert!(EntityType::DroneAnalog.is_mobile());
        assert!(EntityType::DroneDigital.is_mobile());
        assert!(EntityType::Vehicle.is_mobile());
        assert!(!EntityType::SBandRadar.is_mobile());
    }

    #[test]
    fn test_entity_type_fhss() {
        assert!(EntityType::Bluetooth.uses_fhss());
        assert!(!EntityType::WiFiAP.uses_fhss());
        assert!(!EntityType::LTETower.uses_fhss());
    }

    #[test]
    fn test_entity_type_dsss() {
        assert!(EntityType::GNSS.uses_dsss());
        assert!(EntityType::Zigbee.uses_dsss());
        assert!(!EntityType::WiFiAP.uses_dsss());
    }

    #[test]
    fn test_entity_type_chirp() {
        assert!(EntityType::SBandRadar.uses_chirp());
        assert!(EntityType::WeatherRadar.uses_chirp());
        assert!(EntityType::LBandRadar.uses_chirp());
        assert!(EntityType::LoRaWAN.uses_chirp());
        assert!(!EntityType::WiFiAP.uses_chirp());
    }

    #[test]
    fn test_entity_type_scfdm() {
        assert!(EntityType::LTEUplink.uses_scfdm());
        assert!(!EntityType::LTETower.uses_scfdm());
    }

    #[test]
    fn test_entity_type_ppm() {
        assert!(EntityType::ADSB.uses_ppm());
        assert!(!EntityType::LTETower.uses_ppm());
    }

    #[test]
    fn test_entity_type_pulse_pairs() {
        assert!(EntityType::DME.uses_pulse_pairs());
        assert!(!EntityType::ADSB.uses_pulse_pairs());
    }

    #[test]
    fn test_entity_type_conversion() {
        for entity_type in EntityType::all() {
            let value: u8 = entity_type.into();
            let recovered: EntityType = value.into();
            assert_eq!(entity_type, recovered);
        }
    }

    #[test]
    fn test_entity_type_count() {
        assert_eq!(EntityType::count(), 33);
        assert_eq!(EntityType::all().count(), 33);
    }

    #[test]
    fn test_entity_type_specialized() {
        assert!(EntityType::SmartMeter.is_specialized());
        assert!(EntityType::MicrowaveOven.is_specialized());
        assert!(EntityType::RFIDTag.is_specialized());
        assert!(EntityType::KeyFob.is_specialized());
        assert!(EntityType::WirelessMic.is_specialized());
        assert!(EntityType::BabyMonitor.is_specialized());
        assert!(EntityType::WeatherBalloon.is_specialized());
        assert!(EntityType::DeepSpaceNetwork.is_specialized());
        assert!(!EntityType::WiFiAP.is_specialized());
    }

    #[test]
    fn test_entity_type_slotted_aloha() {
        assert!(EntityType::SatelliteIoT.uses_slotted_aloha());
        assert!(!EntityType::WiFiAP.uses_slotted_aloha());
    }

    #[test]
    fn test_entity_type_altitude_varying() {
        assert!(EntityType::WeatherBalloon.is_altitude_varying());
        assert!(!EntityType::WiFiAP.is_altitude_varying());
    }

    #[test]
    fn test_entity_type_deep_space() {
        assert!(EntityType::DeepSpaceNetwork.is_deep_space());
        assert!(!EntityType::WiFiAP.is_deep_space());
    }

    #[test]
    fn test_entity_type_bursty_satellite_iot() {
        assert!(EntityType::SatelliteIoT.is_bursty());
    }

    #[test]
    fn test_entity_type_ook() {
        assert!(EntityType::KeyFob.uses_ook());
        assert!(!EntityType::WiFiAP.uses_ook());
    }

    #[test]
    fn test_entity_type_scheduled() {
        assert!(EntityType::SmartMeter.uses_scheduled());
        assert!(!EntityType::WiFiAP.uses_scheduled());
    }

    #[test]
    fn test_entity_type_reactive() {
        assert!(EntityType::RFIDTag.is_reactive());
        assert!(!EntityType::WiFiAP.is_reactive());
    }

    #[test]
    fn test_entity_type_ac_modulation() {
        assert!(EntityType::MicrowaveOven.has_ac_modulation());
        assert!(!EntityType::WiFiAP.has_ac_modulation());
    }

    #[test]
    fn test_modulation_type_categories() {
        // Analog
        assert!(ModulationType::CW.is_analog());
        assert!(ModulationType::AM.is_analog());
        assert!(ModulationType::FM.is_analog());
        assert!(!ModulationType::OFDM.is_analog());

        // Digital narrowband
        assert!(ModulationType::BPSK.is_digital_narrowband());
        assert!(ModulationType::GFSK.is_digital_narrowband());
        assert!(ModulationType::OOK.is_digital_narrowband());
        assert!(!ModulationType::OFDM.is_digital_narrowband());

        // Digital wideband
        assert!(ModulationType::OFDM.is_digital_wideband());
        assert!(ModulationType::COFDM.is_digital_wideband());
        assert!(!ModulationType::FM.is_digital_wideband());

        // Spread spectrum
        assert!(ModulationType::DSSS.is_spread_spectrum());
        assert!(ModulationType::FHSS.is_spread_spectrum());
        assert!(ModulationType::Chirp.is_spread_spectrum());
        assert!(!ModulationType::OFDM.is_spread_spectrum());

        // Jamming
        assert!(ModulationType::Barrage.is_jamming());
        assert!(ModulationType::Spot.is_jamming());
        assert!(!ModulationType::OFDM.is_jamming());
    }

    #[test]
    fn test_default_modulation() {
        // Continuous
        assert_eq!(EntityType::TVStation.default_modulation(), ModulationType::COFDM);
        assert_eq!(EntityType::FMRadio.default_modulation(), ModulationType::FM);
        assert_eq!(EntityType::AMRadio.default_modulation(), ModulationType::AM);
        assert_eq!(EntityType::GNSS.default_modulation(), ModulationType::DSSS);
        assert_eq!(EntityType::GeoSatellite.default_modulation(), ModulationType::QPSK);
        // Bursty
        assert_eq!(EntityType::LTETower.default_modulation(), ModulationType::OFDM);
        assert_eq!(EntityType::Bluetooth.default_modulation(), ModulationType::GFSK);
        assert_eq!(EntityType::LTEUplink.default_modulation(), ModulationType::SCFDMA);
        assert_eq!(EntityType::Zigbee.default_modulation(), ModulationType::DSSS);
        assert_eq!(EntityType::LoRaWAN.default_modulation(), ModulationType::Chirp);
        // Periodic
        assert_eq!(EntityType::SBandRadar.default_modulation(), ModulationType::Chirp);
        assert_eq!(EntityType::LBandRadar.default_modulation(), ModulationType::Chirp);
        assert_eq!(EntityType::ADSB.default_modulation(), ModulationType::BPSK);
        assert_eq!(EntityType::DME.default_modulation(), ModulationType::CW);
        // Mobile
        assert_eq!(EntityType::DroneAnalog.default_modulation(), ModulationType::FM);
        // Specialized
        assert_eq!(EntityType::SmartMeter.default_modulation(), ModulationType::GFSK);
        assert_eq!(EntityType::MicrowaveOven.default_modulation(), ModulationType::CW);
        assert_eq!(EntityType::RFIDTag.default_modulation(), ModulationType::OOK);
        assert_eq!(EntityType::KeyFob.default_modulation(), ModulationType::OOK);
        assert_eq!(EntityType::WirelessMic.default_modulation(), ModulationType::FM);
        assert_eq!(EntityType::BabyMonitor.default_modulation(), ModulationType::FM);
        // Sprint 7: Exotic
        assert_eq!(EntityType::WeatherBalloon.default_modulation(), ModulationType::GFSK);
        assert_eq!(EntityType::DeepSpaceNetwork.default_modulation(), ModulationType::CW);
        assert_eq!(EntityType::SatelliteIoT.default_modulation(), ModulationType::GFSK);
    }

    #[test]
    fn test_behavior_category_strings() {
        assert_eq!(EntityType::TVStation.behavior_category(), "continuous");
        assert_eq!(EntityType::GeoSatellite.behavior_category(), "continuous");
        assert_eq!(EntityType::LTETower.behavior_category(), "bursty");
        assert_eq!(EntityType::SatelliteIoT.behavior_category(), "bursty");
        assert_eq!(EntityType::SBandRadar.behavior_category(), "periodic");
        assert_eq!(EntityType::ADSB.behavior_category(), "periodic");
        assert_eq!(EntityType::DroneAnalog.behavior_category(), "mobile");
        assert_eq!(EntityType::SmartMeter.behavior_category(), "specialized");
        assert_eq!(EntityType::KeyFob.behavior_category(), "specialized");
        assert_eq!(EntityType::WeatherBalloon.behavior_category(), "specialized");
        assert_eq!(EntityType::DeepSpaceNetwork.behavior_category(), "specialized");
    }
}
