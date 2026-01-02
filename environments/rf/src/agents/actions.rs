//! Agent Actions
//!
//! Defines action structures for jammer and cognitive radio agents.
//! Actions are normalized to [0, 1] for neural network outputs, with
//! denormalization methods to convert to physical units.

use std::f32::consts::PI;

// ============================================================================
// Jammer Modulation Types
// ============================================================================

/// Jammer modulation/waveform type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum JammerModulation {
    /// Barrage jamming: wideband noise across bandwidth
    #[default]
    Barrage = 0,
    /// Spot jamming: concentrated power on single frequency
    Spot = 1,
    /// Sweep jamming: frequency sweeps across bandwidth
    Sweep = 2,
    /// Reactive jamming: responds to detected transmissions
    Reactive = 3,
}

impl JammerModulation {
    /// Convert from normalized [0, 1] value
    pub fn from_normalized(value: f32) -> Self {
        match (value * 4.0) as u8 {
            0 => Self::Barrage,
            1 => Self::Spot,
            2 => Self::Sweep,
            _ => Self::Reactive,
        }
    }

    /// Convert to normalized [0, 1] value
    pub fn to_normalized(self) -> f32 {
        match self {
            Self::Barrage => 0.125,
            Self::Spot => 0.375,
            Self::Sweep => 0.625,
            Self::Reactive => 0.875,
        }
    }

    /// Get all variants
    pub fn all() -> [Self; 4] {
        [Self::Barrage, Self::Spot, Self::Sweep, Self::Reactive]
    }
}

impl From<u8> for JammerModulation {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Barrage,
            1 => Self::Spot,
            2 => Self::Sweep,
            _ => Self::Reactive,
        }
    }
}

// ============================================================================
// Action Configuration
// ============================================================================

/// Configuration for action bounds and normalization
#[derive(Clone, Debug)]
pub struct ActionConfig {
    // Frequency bounds (Hz)
    /// Minimum frequency for agent actions
    pub freq_min: f32,
    /// Maximum frequency for agent actions
    pub freq_max: f32,

    // Power bounds (dBm)
    /// Minimum transmit power
    pub power_min_dbm: f32,
    /// Maximum transmit power
    pub power_max_dbm: f32,

    // Bandwidth bounds (Hz)
    /// Minimum signal bandwidth
    pub bandwidth_min: f32,
    /// Maximum signal bandwidth
    pub bandwidth_max: f32,

    // Jammer-specific
    /// Whether jammer can target specific CRs
    pub jammer_can_target: bool,
    /// Maximum number of CRs that can be targeted
    pub max_targets: usize,

    // CR-specific
    /// Whether CR can adjust bandwidth
    pub cr_variable_bandwidth: bool,
    /// Minimum gap between frequency hops (Hz)
    pub min_hop_distance: f32,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            // Default to ISM band
            freq_min: 2.4e9,
            freq_max: 2.5e9,
            // Typical power range
            power_min_dbm: -10.0,
            power_max_dbm: 30.0,
            // Bandwidth range
            bandwidth_min: 1e6,   // 1 MHz minimum
            bandwidth_max: 40e6,  // 40 MHz maximum (WiFi-like)
            // Jammer options
            jammer_can_target: true,
            max_targets: 4,
            // CR options
            cr_variable_bandwidth: true,
            min_hop_distance: 5e6, // 5 MHz minimum hop
        }
    }
}

impl ActionConfig {
    /// Create a new action configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set frequency bounds
    pub fn with_freq_range(mut self, min: f32, max: f32) -> Self {
        assert!(max > min, "freq_max must be greater than freq_min");
        self.freq_min = min;
        self.freq_max = max;
        self
    }

    /// Set power bounds
    pub fn with_power_range(mut self, min_dbm: f32, max_dbm: f32) -> Self {
        assert!(max_dbm > min_dbm, "power_max must be greater than power_min");
        self.power_min_dbm = min_dbm;
        self.power_max_dbm = max_dbm;
        self
    }

    /// Set bandwidth bounds
    pub fn with_bandwidth_range(mut self, min: f32, max: f32) -> Self {
        assert!(max > min, "bandwidth_max must be greater than bandwidth_min");
        self.bandwidth_min = min;
        self.bandwidth_max = max;
        self
    }

    /// Denormalize frequency from [0, 1] to Hz
    #[inline]
    pub fn denorm_freq(&self, normalized: f32) -> f32 {
        let clamped = normalized.clamp(0.0, 1.0);
        self.freq_min + clamped * (self.freq_max - self.freq_min)
    }

    /// Normalize frequency from Hz to [0, 1]
    #[inline]
    pub fn norm_freq(&self, freq_hz: f32) -> f32 {
        ((freq_hz - self.freq_min) / (self.freq_max - self.freq_min)).clamp(0.0, 1.0)
    }

    /// Denormalize power from [0, 1] to dBm
    #[inline]
    pub fn denorm_power(&self, normalized: f32) -> f32 {
        let clamped = normalized.clamp(0.0, 1.0);
        self.power_min_dbm + clamped * (self.power_max_dbm - self.power_min_dbm)
    }

    /// Normalize power from dBm to [0, 1]
    #[inline]
    pub fn norm_power(&self, power_dbm: f32) -> f32 {
        ((power_dbm - self.power_min_dbm) / (self.power_max_dbm - self.power_min_dbm))
            .clamp(0.0, 1.0)
    }

    /// Denormalize bandwidth from [0, 1] to Hz
    #[inline]
    pub fn denorm_bandwidth(&self, normalized: f32) -> f32 {
        let clamped = normalized.clamp(0.0, 1.0);
        self.bandwidth_min + clamped * (self.bandwidth_max - self.bandwidth_min)
    }

    /// Normalize bandwidth from Hz to [0, 1]
    #[inline]
    pub fn norm_bandwidth(&self, bandwidth_hz: f32) -> f32 {
        ((bandwidth_hz - self.bandwidth_min) / (self.bandwidth_max - self.bandwidth_min))
            .clamp(0.0, 1.0)
    }

    /// Get the frequency range
    #[inline]
    pub fn freq_range(&self) -> f32 {
        self.freq_max - self.freq_min
    }

    /// Get the power range in dB
    #[inline]
    pub fn power_range_db(&self) -> f32 {
        self.power_max_dbm - self.power_min_dbm
    }

    /// Get the bandwidth range
    #[inline]
    pub fn bandwidth_range(&self) -> f32 {
        self.bandwidth_max - self.bandwidth_min
    }
}

// ============================================================================
// Jammer Action
// ============================================================================

/// Action taken by a jammer agent
///
/// All values are normalized to [0, 1] for neural network compatibility.
/// Use denormalize methods with ActionConfig to get physical values.
#[derive(Clone, Copy, Debug, Default)]
pub struct JammerAction {
    /// Normalized center frequency [0, 1]
    pub frequency: f32,
    /// Normalized bandwidth [0, 1]
    pub bandwidth: f32,
    /// Normalized transmit power [0, 1]
    pub power: f32,
    /// Normalized modulation type [0, 1]
    pub modulation: f32,
}

impl JammerAction {
    /// Create a new jammer action from normalized values
    pub fn new(frequency: f32, bandwidth: f32, power: f32, modulation: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            bandwidth: bandwidth.clamp(0.0, 1.0),
            power: power.clamp(0.0, 1.0),
            modulation: modulation.clamp(0.0, 1.0),
        }
    }

    /// Get denormalized center frequency (Hz)
    #[inline]
    pub fn actual_frequency(&self, config: &ActionConfig) -> f32 {
        config.denorm_freq(self.frequency)
    }

    /// Get denormalized bandwidth (Hz)
    #[inline]
    pub fn actual_bandwidth(&self, config: &ActionConfig) -> f32 {
        config.denorm_bandwidth(self.bandwidth)
    }

    /// Get denormalized power (dBm)
    #[inline]
    pub fn actual_power_dbm(&self, config: &ActionConfig) -> f32 {
        config.denorm_power(self.power)
    }

    /// Get modulation type
    #[inline]
    pub fn modulation_type(&self) -> JammerModulation {
        JammerModulation::from_normalized(self.modulation)
    }

    /// Convert to linear power (mW)
    #[inline]
    pub fn actual_power_linear(&self, config: &ActionConfig) -> f32 {
        db_to_linear(self.actual_power_dbm(config))
    }

    /// Get frequency range occupied by jamming signal
    pub fn frequency_range(&self, config: &ActionConfig) -> (f32, f32) {
        let center = self.actual_frequency(config);
        let bw = self.actual_bandwidth(config);
        (center - bw / 2.0, center + bw / 2.0)
    }

    /// Create from a flat array of 4 normalized values
    pub fn from_flat(values: &[f32]) -> Self {
        assert!(values.len() >= 4, "JammerAction requires 4 values");
        Self::new(values[0], values[1], values[2], values[3])
    }

    /// Convert to a flat array of 4 normalized values
    pub fn to_flat(&self) -> [f32; 4] {
        [self.frequency, self.bandwidth, self.power, self.modulation]
    }

    /// Number of action dimensions for a jammer
    pub const DIM: usize = 4;
}

// ============================================================================
// Cognitive Radio Action
// ============================================================================

/// Action taken by a cognitive radio agent
///
/// All values are normalized to [0, 1] for neural network compatibility.
#[derive(Clone, Copy, Debug, Default)]
pub struct CognitiveRadioAction {
    /// Normalized center frequency [0, 1]
    pub frequency: f32,
    /// Normalized transmit power [0, 1]
    pub power: f32,
    /// Normalized bandwidth [0, 1]
    pub bandwidth: f32,
}

impl CognitiveRadioAction {
    /// Create a new CR action from normalized values
    pub fn new(frequency: f32, power: f32, bandwidth: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            power: power.clamp(0.0, 1.0),
            bandwidth: bandwidth.clamp(0.0, 1.0),
        }
    }

    /// Get denormalized center frequency (Hz)
    #[inline]
    pub fn actual_frequency(&self, config: &ActionConfig) -> f32 {
        config.denorm_freq(self.frequency)
    }

    /// Get denormalized power (dBm)
    #[inline]
    pub fn actual_power_dbm(&self, config: &ActionConfig) -> f32 {
        config.denorm_power(self.power)
    }

    /// Get denormalized bandwidth (Hz)
    #[inline]
    pub fn actual_bandwidth(&self, config: &ActionConfig) -> f32 {
        config.denorm_bandwidth(self.bandwidth)
    }

    /// Convert to linear power (mW)
    #[inline]
    pub fn actual_power_linear(&self, config: &ActionConfig) -> f32 {
        db_to_linear(self.actual_power_dbm(config))
    }

    /// Get frequency range occupied by CR signal
    pub fn frequency_range(&self, config: &ActionConfig) -> (f32, f32) {
        let center = self.actual_frequency(config);
        let bw = self.actual_bandwidth(config);
        (center - bw / 2.0, center + bw / 2.0)
    }

    /// Create from a flat array of 3 normalized values
    pub fn from_flat(values: &[f32]) -> Self {
        assert!(values.len() >= 3, "CognitiveRadioAction requires 3 values");
        Self::new(values[0], values[1], values[2])
    }

    /// Convert to a flat array of 3 normalized values
    pub fn to_flat(&self) -> [f32; 3] {
        [self.frequency, self.power, self.bandwidth]
    }

    /// Number of action dimensions for a CR
    pub const DIM: usize = 3;
}

// ============================================================================
// Action Space
// ============================================================================

/// Combined action space specification
#[derive(Clone, Debug)]
pub struct ActionSpace {
    /// Number of jammer agents
    pub num_jammers: usize,
    /// Number of cognitive radio agents
    pub num_crs: usize,
    /// Action configuration for bounds
    pub config: ActionConfig,
}

impl ActionSpace {
    /// Create a new action space
    pub fn new(num_jammers: usize, num_crs: usize, config: ActionConfig) -> Self {
        Self {
            num_jammers,
            num_crs,
            config,
        }
    }

    /// Total action dimension for all agents
    #[inline]
    pub fn total_dim(&self) -> usize {
        self.num_jammers * JammerAction::DIM + self.num_crs * CognitiveRadioAction::DIM
    }

    /// Jammer action dimension
    #[inline]
    pub fn jammer_dim(&self) -> usize {
        self.num_jammers * JammerAction::DIM
    }

    /// CR action dimension
    #[inline]
    pub fn cr_dim(&self) -> usize {
        self.num_crs * CognitiveRadioAction::DIM
    }

    /// Parse jammer actions from flat array
    ///
    /// Expected layout: [J0_freq, J0_bw, J0_power, J0_mod, J1_freq, ...]
    pub fn parse_jammer_actions(&self, flat: &[f32]) -> Vec<JammerAction> {
        let mut actions = Vec::with_capacity(self.num_jammers);
        for i in 0..self.num_jammers {
            let start = i * JammerAction::DIM;
            if start + JammerAction::DIM <= flat.len() {
                actions.push(JammerAction::from_flat(&flat[start..]));
            }
        }
        actions
    }

    /// Parse CR actions from flat array
    ///
    /// Expected layout after jammer actions: [CR0_freq, CR0_power, CR0_bw, CR1_freq, ...]
    pub fn parse_cr_actions(&self, flat: &[f32]) -> Vec<CognitiveRadioAction> {
        let mut actions = Vec::with_capacity(self.num_crs);
        let jammer_offset = self.jammer_dim();
        for i in 0..self.num_crs {
            let start = jammer_offset + i * CognitiveRadioAction::DIM;
            if start + CognitiveRadioAction::DIM <= flat.len() {
                actions.push(CognitiveRadioAction::from_flat(&flat[start..]));
            }
        }
        actions
    }

    /// Parse all actions from flat array
    pub fn parse_all(
        &self,
        flat: &[f32],
    ) -> (Vec<JammerAction>, Vec<CognitiveRadioAction>) {
        (self.parse_jammer_actions(flat), self.parse_cr_actions(flat))
    }
}

// ============================================================================
// Sweep Jamming Parameters
// ============================================================================

/// Parameters for sweep jamming
#[derive(Clone, Copy, Debug)]
pub struct SweepParams {
    /// Sweep rate (Hz/s)
    pub sweep_rate: f32,
    /// Sweep direction: true = increasing frequency
    pub direction: bool,
    /// Current phase in sweep cycle [0, 2π)
    pub phase: f32,
}

impl Default for SweepParams {
    fn default() -> Self {
        Self {
            sweep_rate: 1e6,    // 1 MHz/s
            direction: true,
            phase: 0.0,
        }
    }
}

impl SweepParams {
    /// Calculate instantaneous frequency offset
    pub fn frequency_offset(&self, time: f32) -> f32 {
        let sign = if self.direction { 1.0 } else { -1.0 };
        let sawtooth = ((self.phase + time * self.sweep_rate) % (2.0 * PI)) / PI - 1.0;
        sign * sawtooth * self.sweep_rate
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert dBm to linear power (mW)
#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

/// Convert linear power (mW) to dBm
#[inline]
#[allow(dead_code)]
fn linear_to_db(linear: f32) -> f32 {
    10.0 * linear.log10()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jammer_modulation_roundtrip() {
        for mod_type in JammerModulation::all() {
            let normalized = mod_type.to_normalized();
            let recovered = JammerModulation::from_normalized(normalized);
            assert_eq!(mod_type, recovered);
        }
    }

    #[test]
    fn test_action_config_denormalization() {
        let config = ActionConfig::default();

        // Test frequency
        assert!((config.denorm_freq(0.0) - config.freq_min).abs() < 1.0);
        assert!((config.denorm_freq(1.0) - config.freq_max).abs() < 1.0);
        assert!((config.denorm_freq(0.5) - (config.freq_min + config.freq_max) / 2.0).abs() < 1.0);

        // Test power
        assert!((config.denorm_power(0.0) - config.power_min_dbm).abs() < 0.01);
        assert!((config.denorm_power(1.0) - config.power_max_dbm).abs() < 0.01);

        // Test bandwidth
        assert!((config.denorm_bandwidth(0.0) - config.bandwidth_min).abs() < 1.0);
        assert!((config.denorm_bandwidth(1.0) - config.bandwidth_max).abs() < 1.0);
    }

    #[test]
    fn test_action_config_normalization() {
        let config = ActionConfig::default();

        // Roundtrip test
        let freq = 2.45e9;
        let normalized = config.norm_freq(freq);
        let denormalized = config.denorm_freq(normalized);
        assert!((denormalized - freq).abs() < 1.0);

        let power = 20.0;
        let normalized = config.norm_power(power);
        let denormalized = config.denorm_power(normalized);
        assert!((denormalized - power).abs() < 0.01);
    }

    #[test]
    fn test_jammer_action_from_flat() {
        let flat = vec![0.5, 0.3, 0.8, 0.1];
        let action = JammerAction::from_flat(&flat);

        assert!((action.frequency - 0.5).abs() < 1e-6);
        assert!((action.bandwidth - 0.3).abs() < 1e-6);
        assert!((action.power - 0.8).abs() < 1e-6);
        assert!((action.modulation - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_jammer_action_to_flat_roundtrip() {
        let action = JammerAction::new(0.5, 0.3, 0.8, 0.1);
        let flat = action.to_flat();
        let recovered = JammerAction::from_flat(&flat);

        assert!((action.frequency - recovered.frequency).abs() < 1e-6);
        assert!((action.bandwidth - recovered.bandwidth).abs() < 1e-6);
        assert!((action.power - recovered.power).abs() < 1e-6);
        assert!((action.modulation - recovered.modulation).abs() < 1e-6);
    }

    #[test]
    fn test_cr_action_from_flat() {
        let flat = vec![0.7, 0.5, 0.2];
        let action = CognitiveRadioAction::from_flat(&flat);

        assert!((action.frequency - 0.7).abs() < 1e-6);
        assert!((action.power - 0.5).abs() < 1e-6);
        assert!((action.bandwidth - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_action_space_total_dim() {
        let config = ActionConfig::default();
        let space = ActionSpace::new(2, 3, config);

        // 2 jammers × 4 + 3 CRs × 3 = 8 + 9 = 17
        assert_eq!(space.total_dim(), 17);
        assert_eq!(space.jammer_dim(), 8);
        assert_eq!(space.cr_dim(), 9);
    }

    #[test]
    fn test_action_space_parsing() {
        let config = ActionConfig::default();
        let space = ActionSpace::new(1, 1, config);

        // [J0_freq, J0_bw, J0_power, J0_mod, CR0_freq, CR0_power, CR0_bw]
        let flat = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

        let (jammer_actions, cr_actions) = space.parse_all(&flat);

        assert_eq!(jammer_actions.len(), 1);
        assert_eq!(cr_actions.len(), 1);

        assert!((jammer_actions[0].frequency - 0.1).abs() < 1e-6);
        assert!((cr_actions[0].frequency - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jammer_frequency_range() {
        let config = ActionConfig::new()
            .with_freq_range(2.4e9, 2.5e9)
            .with_bandwidth_range(1e6, 20e6);

        // Jammer at center frequency with half bandwidth
        let action = JammerAction::new(0.5, 0.5, 0.5, 0.0);

        let (low, high) = action.frequency_range(&config);
        let center = action.actual_frequency(&config);
        let bw = action.actual_bandwidth(&config);

        assert!((center - 2.45e9).abs() < 1.0);
        assert!((low - (center - bw / 2.0)).abs() < 1.0);
        assert!((high - (center + bw / 2.0)).abs() < 1.0);
    }

    #[test]
    fn test_sweep_params() {
        let params = SweepParams {
            sweep_rate: 1e6,
            direction: true,
            phase: 0.0,
        };

        let offset_0 = params.frequency_offset(0.0);
        let offset_1 = params.frequency_offset(1e-6);

        // Should be increasing for positive direction
        assert!(offset_1 > offset_0 || (offset_1 - offset_0).abs() < 1.0);
    }
}
