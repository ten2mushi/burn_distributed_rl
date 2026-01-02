//! Signal Protocol Trait
//!
//! Defines the interface for rendering signals to a power spectral density buffer.
//! Includes implementations for common signal types: flat spectrum, OFDM, FHSS, chirp, DSSS.

use crate::types::dimensional::Hertz;
use crate::types::frequency::ValidatedFrequencyGrid;

/// Signal protocol trait for rendering signals to PSD buffers.
///
/// Implementations describe how a signal's power is distributed across
/// frequency at any given moment, and how the signal evolves over time.
pub trait SignalProtocol: Send + Sync {
    /// Render signal to a PSD buffer slice
    ///
    /// # Arguments
    /// * `psd` - Mutable slice of power spectral density values (linear power units)
    /// * `grid` - Validated frequency grid describing the PSD bins
    /// * `center_hz` - Signal center frequency in Hz
    /// * `bandwidth_hz` - Signal bandwidth in Hz
    /// * `power_linear` - Total signal power in linear units
    fn render_to_psd(
        &self,
        psd: &mut [f32],
        grid: &ValidatedFrequencyGrid,
        center_hz: f32,
        bandwidth_hz: f32,
        power_linear: f32,
    );

    /// Advance protocol state by delta time
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    /// * `rng` - Random number generator seed for stochastic protocols
    fn step(&mut self, dt: f32, rng_seed: u64);

    /// Check if signal is currently active (transmitting)
    fn is_active(&self) -> bool;

    /// Get current center frequency (may change for hopping protocols)
    fn current_frequency(&self) -> f32;

    /// Get current bandwidth (may change for adaptive protocols)
    fn current_bandwidth(&self) -> f32;

    /// Protocol name for debugging/logging
    fn name(&self) -> &'static str;

    /// Create a boxed clone (for trait objects)
    fn clone_box(&self) -> Box<dyn SignalProtocol>;
}

// ============================================================================
// Flat Spectrum (Rectangular PSD)
// ============================================================================

/// Flat spectrum signal with rectangular PSD shape
///
/// Simplest signal model - constant power density across bandwidth.
/// Typical of some analog signals and simplified digital models.
#[derive(Debug, Clone)]
pub struct FlatSpectrum {
    center_hz: f32,
    bandwidth_hz: f32,
    active: bool,
}

impl FlatSpectrum {
    pub fn new(center_hz: f32, bandwidth_hz: f32) -> Self {
        Self {
            center_hz,
            bandwidth_hz,
            active: true,
        }
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }
}

impl SignalProtocol for FlatSpectrum {
    fn render_to_psd(
        &self,
        psd: &mut [f32],
        freq_grid: &ValidatedFrequencyGrid,
        center_hz: f32,
        bandwidth_hz: f32,
        power_linear: f32,
    ) {
        if !self.active || power_linear <= 0.0 {
            return;
        }

        let half_bw = bandwidth_hz / 2.0;
        let f_min = center_hz - half_bw;
        let f_max = center_hz + half_bw;

        // Find affected bins (freq_to_bin clamps to valid range)
        let start_bin = freq_grid.freq_to_bin(Hertz::new(f_min));
        let end_bin = freq_grid.freq_to_bin(Hertz::new(f_max)).min(psd.len().saturating_sub(1));

        if start_bin > end_bin {
            return;
        }

        // Power per bin (uniform distribution)
        let num_bins = (end_bin - start_bin + 1) as f32;
        let power_per_bin = power_linear / num_bins;

        for i in start_bin..=end_bin {
            psd[i] += power_per_bin;
        }
    }

    fn step(&mut self, _dt: f32, _rng_seed: u64) {
        // Flat spectrum is stateless
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn current_frequency(&self) -> f32 {
        self.center_hz
    }

    fn current_bandwidth(&self) -> f32 {
        self.bandwidth_hz
    }

    fn name(&self) -> &'static str {
        "Flat Spectrum"
    }

    fn clone_box(&self) -> Box<dyn SignalProtocol> {
        Box::new(self.clone())
    }
}

// ============================================================================
// OFDM Protocol
// ============================================================================

/// OFDM (Orthogonal Frequency-Division Multiplexing) signal
///
/// Models signals like LTE, WiFi, 5G NR with multiple subcarriers.
/// Power is distributed across subcarriers with optional guard bands.
#[derive(Debug, Clone)]
pub struct OFDMProtocol {
    center_hz: f32,
    bandwidth_hz: f32,
    num_subcarriers: usize,
    subcarrier_spacing_hz: f32,
    guard_band_ratio: f32, // Fraction of bandwidth used as guard bands
    active: bool,
    /// Per-subcarrier power allocation (normalized)
    subcarrier_power: Vec<f32>,
}

impl OFDMProtocol {
    /// Create OFDM signal with specified parameters
    pub fn new(center_hz: f32, bandwidth_hz: f32, num_subcarriers: usize) -> Self {
        let subcarrier_spacing = bandwidth_hz / num_subcarriers as f32;
        let subcarrier_power = vec![1.0 / num_subcarriers as f32; num_subcarriers];

        Self {
            center_hz,
            bandwidth_hz,
            num_subcarriers,
            subcarrier_spacing_hz: subcarrier_spacing,
            guard_band_ratio: 0.1, // 10% guard bands (5% each side)
            active: true,
            subcarrier_power,
        }
    }

    /// LTE-like OFDM preset (15 kHz subcarrier spacing)
    pub fn lte_like(center_hz: f32, bandwidth_hz: f32) -> Self {
        let subcarrier_spacing = 15e3; // 15 kHz
        let num_subcarriers = (bandwidth_hz / subcarrier_spacing) as usize;
        Self::new(center_hz, bandwidth_hz, num_subcarriers.max(12))
    }

    /// WiFi-like OFDM preset (312.5 kHz subcarrier spacing for 20 MHz)
    pub fn wifi_like(center_hz: f32) -> Self {
        Self::new(center_hz, 20e6, 64) // 64 subcarriers in 20 MHz
    }

    /// Set guard band ratio (0.0 to 0.5)
    pub fn with_guard_band(mut self, ratio: f32) -> Self {
        self.guard_band_ratio = ratio.clamp(0.0, 0.5);
        self
    }

    /// Set custom subcarrier power allocation
    pub fn with_power_allocation(mut self, powers: Vec<f32>) -> Self {
        if powers.len() == self.num_subcarriers {
            // Normalize to sum to 1
            let sum: f32 = powers.iter().sum();
            if sum > 0.0 {
                self.subcarrier_power = powers.iter().map(|p| p / sum).collect();
            }
        }
        self
    }

    /// Null specific subcarriers (e.g., DC subcarrier)
    pub fn null_subcarriers(&mut self, indices: &[usize]) {
        for &i in indices {
            if i < self.subcarrier_power.len() {
                self.subcarrier_power[i] = 0.0;
            }
        }
        // Renormalize
        let sum: f32 = self.subcarrier_power.iter().sum();
        if sum > 0.0 {
            for p in &mut self.subcarrier_power {
                *p /= sum;
            }
        }
    }
}

impl SignalProtocol for OFDMProtocol {
    fn render_to_psd(
        &self,
        psd: &mut [f32],
        freq_grid: &ValidatedFrequencyGrid,
        center_hz: f32,
        bandwidth_hz: f32,
        power_linear: f32,
    ) {
        if !self.active || power_linear <= 0.0 {
            return;
        }

        let active_bw = bandwidth_hz * (1.0 - self.guard_band_ratio);
        let half_active_bw = active_bw / 2.0;

        // Calculate subcarrier positions
        for (i, &sc_power) in self.subcarrier_power.iter().enumerate() {
            if sc_power <= 0.0 {
                continue;
            }

            // Subcarrier center frequency
            let offset = (i as f32 - self.num_subcarriers as f32 / 2.0 + 0.5)
                * self.subcarrier_spacing_hz;
            let sc_freq = center_hz + offset;

            // Only render if within active bandwidth
            if (sc_freq - center_hz).abs() > half_active_bw {
                continue;
            }

            // Map to PSD bin
            let bin = freq_grid.freq_to_bin(Hertz::new(sc_freq));
            if bin < psd.len() {
                psd[bin] += power_linear * sc_power;
            }
        }
    }

    fn step(&mut self, _dt: f32, _rng_seed: u64) {
        // Basic OFDM is stateless (could add time-varying power allocation)
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn current_frequency(&self) -> f32 {
        self.center_hz
    }

    fn current_bandwidth(&self) -> f32 {
        self.bandwidth_hz
    }

    fn name(&self) -> &'static str {
        "OFDM"
    }

    fn clone_box(&self) -> Box<dyn SignalProtocol> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Frequency Hopping Spread Spectrum (FHSS)
// ============================================================================

/// Frequency Hopping Spread Spectrum signal
///
/// Signal hops between frequencies according to a pseudo-random pattern.
/// Used in Bluetooth, military radios, and some WiFi modes.
#[derive(Debug, Clone)]
pub struct FHSSProtocol {
    /// Base center frequency
    center_hz: f32,
    /// Bandwidth per hop
    hop_bandwidth_hz: f32,
    /// Total hopping span
    hop_span_hz: f32,
    /// Number of hop channels
    num_channels: usize,
    /// Current channel index
    current_channel: usize,
    /// Hop period in seconds
    hop_period_s: f32,
    /// Time since last hop
    time_since_hop: f32,
    /// Hopping sequence (channel indices)
    hop_sequence: Vec<usize>,
    /// Position in hop sequence
    sequence_pos: usize,
    /// Is transmitting
    active: bool,
}

impl FHSSProtocol {
    /// Create FHSS signal with specified parameters
    pub fn new(
        center_hz: f32,
        hop_bandwidth_hz: f32,
        hop_span_hz: f32,
        num_channels: usize,
        hop_period_s: f32,
    ) -> Self {
        // Generate default sequential hopping pattern
        let hop_sequence: Vec<usize> = (0..num_channels).collect();

        Self {
            center_hz,
            hop_bandwidth_hz,
            hop_span_hz,
            num_channels,
            current_channel: 0,
            hop_period_s,
            time_since_hop: 0.0,
            hop_sequence,
            sequence_pos: 0,
            active: true,
        }
    }

    /// Bluetooth-like FHSS preset (79 channels, 1 MHz each, 1600 hops/sec)
    pub fn bluetooth_like(center_hz: f32) -> Self {
        Self::new(
            center_hz,
            1e6,        // 1 MHz per channel
            79e6,       // 79 MHz span
            79,         // 79 channels
            1.0 / 1600.0, // 1600 hops/sec
        )
    }

    /// Military-style slow hopping
    pub fn military_slow(center_hz: f32, span_hz: f32, channels: usize) -> Self {
        Self::new(center_hz, span_hz / channels as f32, span_hz, channels, 0.1)
    }

    /// Set custom hopping sequence
    pub fn with_hop_sequence(mut self, sequence: Vec<usize>) -> Self {
        // Validate sequence
        let valid: Vec<usize> = sequence
            .into_iter()
            .filter(|&c| c < self.num_channels)
            .collect();
        if !valid.is_empty() {
            self.hop_sequence = valid;
        }
        self
    }

    /// Generate pseudo-random hopping sequence from seed
    pub fn with_random_sequence(mut self, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        self.hop_sequence = (0..self.num_channels).collect();

        // Fisher-Yates shuffle
        for i in (1..self.hop_sequence.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            self.hop_sequence.swap(i, j);
        }
        self
    }

    /// Get current hop frequency
    fn current_hop_frequency(&self) -> f32 {
        let channel_spacing = self.hop_span_hz / self.num_channels as f32;
        let offset = (self.current_channel as f32 - self.num_channels as f32 / 2.0 + 0.5)
            * channel_spacing;
        self.center_hz + offset
    }
}

impl SignalProtocol for FHSSProtocol {
    fn render_to_psd(
        &self,
        psd: &mut [f32],
        freq_grid: &ValidatedFrequencyGrid,
        _center_hz: f32,
        _bandwidth_hz: f32,
        power_linear: f32,
    ) {
        if !self.active || power_linear <= 0.0 {
            return;
        }

        let freq = self.current_hop_frequency();
        let half_bw = self.hop_bandwidth_hz / 2.0;

        let start_bin = freq_grid.freq_to_bin(Hertz::new(freq - half_bw));
        let end_bin = freq_grid.freq_to_bin(Hertz::new(freq + half_bw)).min(psd.len().saturating_sub(1));

        if start_bin > end_bin {
            return;
        }

        let num_bins = (end_bin - start_bin + 1) as f32;
        let power_per_bin = power_linear / num_bins;

        for i in start_bin..=end_bin {
            psd[i] += power_per_bin;
        }
    }

    fn step(&mut self, dt: f32, _rng_seed: u64) {
        if !self.active {
            return;
        }

        self.time_since_hop += dt;

        if self.time_since_hop >= self.hop_period_s {
            self.time_since_hop = 0.0;
            self.sequence_pos = (self.sequence_pos + 1) % self.hop_sequence.len();
            self.current_channel = self.hop_sequence[self.sequence_pos];
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn current_frequency(&self) -> f32 {
        self.current_hop_frequency()
    }

    fn current_bandwidth(&self) -> f32 {
        self.hop_bandwidth_hz
    }

    fn name(&self) -> &'static str {
        "FHSS"
    }

    fn clone_box(&self) -> Box<dyn SignalProtocol> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Chirp Signal
// ============================================================================

/// Linear frequency sweep (chirp) signal
///
/// Used in radar, LoRa, and some spread spectrum systems.
/// Frequency linearly sweeps from start to end over sweep period.
#[derive(Debug, Clone)]
pub struct ChirpProtocol {
    center_hz: f32,
    bandwidth_hz: f32,
    sweep_period_s: f32,
    /// Current position in sweep (0.0 to 1.0)
    sweep_position: f32,
    /// Up-chirp (true) or down-chirp (false)
    up_chirp: bool,
    active: bool,
}

impl ChirpProtocol {
    pub fn new(center_hz: f32, bandwidth_hz: f32, sweep_period_s: f32) -> Self {
        Self {
            center_hz,
            bandwidth_hz,
            sweep_period_s,
            sweep_position: 0.0,
            up_chirp: true,
            active: true,
        }
    }

    /// LoRa-like chirp preset
    pub fn lora_like(center_hz: f32, spreading_factor: u8) -> Self {
        let bandwidth = 125e3; // 125 kHz typical
        let symbol_time = (1 << spreading_factor) as f32 / bandwidth;
        Self::new(center_hz, bandwidth, symbol_time)
    }

    /// Radar-like chirp
    pub fn radar_like(center_hz: f32, bandwidth_hz: f32, pulse_width_s: f32) -> Self {
        Self::new(center_hz, bandwidth_hz, pulse_width_s)
    }

    pub fn down_chirp(mut self) -> Self {
        self.up_chirp = false;
        self
    }

    /// Get instantaneous frequency at current sweep position
    fn instantaneous_frequency(&self) -> f32 {
        let half_bw = self.bandwidth_hz / 2.0;
        if self.up_chirp {
            self.center_hz - half_bw + self.bandwidth_hz * self.sweep_position
        } else {
            self.center_hz + half_bw - self.bandwidth_hz * self.sweep_position
        }
    }
}

impl SignalProtocol for ChirpProtocol {
    fn render_to_psd(
        &self,
        psd: &mut [f32],
        freq_grid: &ValidatedFrequencyGrid,
        _center_hz: f32,
        _bandwidth_hz: f32,
        power_linear: f32,
    ) {
        if !self.active || power_linear <= 0.0 {
            return;
        }

        // For instantaneous chirp, render narrow band at current frequency
        let freq = self.instantaneous_frequency();
        let inst_bw = self.bandwidth_hz / 20.0; // Approximate instantaneous bandwidth
        let resolution = freq_grid.resolution().as_hz();

        let bin = freq_grid.freq_to_bin(Hertz::new(freq));
        if bin < psd.len() {
            // Spread over a few bins
            let spread = (inst_bw / resolution).max(1.0) as usize;
            let half_spread = spread / 2;
            let start = bin.saturating_sub(half_spread);
            let end = (bin + half_spread).min(psd.len().saturating_sub(1));

            let num_bins = (end - start + 1) as f32;
            let power_per_bin = power_linear / num_bins;

            for i in start..=end {
                psd[i] += power_per_bin;
            }
        }
    }

    fn step(&mut self, dt: f32, _rng_seed: u64) {
        if !self.active {
            return;
        }

        self.sweep_position += dt / self.sweep_period_s;
        if self.sweep_position >= 1.0 {
            self.sweep_position -= 1.0;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn current_frequency(&self) -> f32 {
        self.instantaneous_frequency()
    }

    fn current_bandwidth(&self) -> f32 {
        self.bandwidth_hz
    }

    fn name(&self) -> &'static str {
        "Chirp"
    }

    fn clone_box(&self) -> Box<dyn SignalProtocol> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Direct Sequence Spread Spectrum (DSSS)
// ============================================================================

/// Direct Sequence Spread Spectrum signal
///
/// Signal is spread across bandwidth using a pseudo-noise code.
/// Used in GPS, CDMA, and some WiFi modes.
#[derive(Debug, Clone)]
pub struct DSSSProtocol {
    center_hz: f32,
    chip_rate_hz: f32,
    #[allow(dead_code)] // Reserved for future DSSS processing gain calculation
    spreading_factor: usize,
    active: bool,
}

impl DSSSProtocol {
    pub fn new(center_hz: f32, chip_rate_hz: f32, spreading_factor: usize) -> Self {
        Self {
            center_hz,
            chip_rate_hz,
            spreading_factor,
            active: true,
        }
    }

    /// GPS C/A code-like preset
    pub fn gps_like(center_hz: f32) -> Self {
        Self::new(center_hz, 1.023e6, 1023) // 1.023 Mchip/s, 1023 chips
    }

    /// CDMA-like preset
    pub fn cdma_like(center_hz: f32) -> Self {
        Self::new(center_hz, 1.2288e6, 64) // 1.2288 Mchip/s
    }

    /// Effective bandwidth (approximately chip rate)
    fn effective_bandwidth(&self) -> f32 {
        self.chip_rate_hz * 2.0 // Main lobe bandwidth
    }
}

impl SignalProtocol for DSSSProtocol {
    fn render_to_psd(
        &self,
        psd: &mut [f32],
        freq_grid: &ValidatedFrequencyGrid,
        center_hz: f32,
        _bandwidth_hz: f32,
        power_linear: f32,
    ) {
        if !self.active || power_linear <= 0.0 {
            return;
        }

        let bw = self.effective_bandwidth();
        let half_bw = bw / 2.0;

        // DSSS has sinc^2 spectrum shape, approximate with weighted bins
        let start_bin = freq_grid.freq_to_bin(Hertz::new(center_hz - half_bw));
        let end_bin = freq_grid.freq_to_bin(Hertz::new(center_hz + half_bw)).min(psd.len().saturating_sub(1));

        if start_bin > end_bin {
            return;
        }

        let center_bin = freq_grid.freq_to_bin(Hertz::new(center_hz));
        let total_bins = end_bin - start_bin + 1;

        // Apply sinc^2 weighting
        let mut weights = Vec::with_capacity(total_bins);
        let mut weight_sum = 0.0;

        for i in start_bin..=end_bin {
            // Calculate offset from center
            let offset = if i >= center_bin {
                (i - center_bin) as f32 / (total_bins as f32 / 2.0)
            } else {
                -((center_bin - i) as f32 / (total_bins as f32 / 2.0))
            };
            let weight = if offset.abs() < 0.01 {
                1.0
            } else {
                let x = std::f32::consts::PI * offset;
                (x.sin() / x).powi(2)
            };
            weights.push(weight);
            weight_sum += weight;
        }

        // Distribute power according to weights
        for (idx, i) in (start_bin..=end_bin).enumerate() {
            let normalized_weight = weights[idx] / weight_sum;
            psd[i] += power_linear * normalized_weight;
        }
    }

    fn step(&mut self, _dt: f32, _rng_seed: u64) {
        // DSSS is essentially stateless for PSD purposes
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn current_frequency(&self) -> f32 {
        self.center_hz
    }

    fn current_bandwidth(&self) -> f32 {
        self.effective_bandwidth()
    }

    fn name(&self) -> &'static str {
        "DSSS"
    }

    fn clone_box(&self) -> Box<dyn SignalProtocol> {
        Box::new(self.clone())
    }
}

// ============================================================================
// Helper: Simple RNG for protocol use
// ============================================================================

#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_grid() -> ValidatedFrequencyGrid {
        // 2.4-2.5 GHz, 1000 bins = 100 kHz resolution
        ValidatedFrequencyGrid::from_params(2.4e9, 2.5e9, 1000).expect("Valid grid")
    }

    #[test]
    fn test_flat_spectrum_render() {
        let proto = FlatSpectrum::new(2.45e9, 10e6);
        let grid = make_test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        proto.render_to_psd(&mut psd, &grid, 2.45e9, 10e6, 1.0);

        // Check that some power was added
        let total_power: f32 = psd.iter().sum();
        assert!(
            (total_power - 1.0).abs() < 0.01,
            "Total power should be 1.0, got {}",
            total_power
        );
    }

    #[test]
    fn test_flat_spectrum_inactive() {
        let mut proto = FlatSpectrum::new(2.45e9, 10e6);
        proto.set_active(false);

        let grid = make_test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        proto.render_to_psd(&mut psd, &grid, 2.45e9, 10e6, 1.0);

        let total_power: f32 = psd.iter().sum();
        assert!(
            total_power < 1e-10,
            "Inactive signal should add no power, got {}",
            total_power
        );
    }

    #[test]
    fn test_ofdm_subcarriers() {
        let proto = OFDMProtocol::new(2.45e9, 20e6, 64);
        let grid = make_test_grid();
        let mut psd = vec![0.0; grid.num_bins()];

        proto.render_to_psd(&mut psd, &grid, 2.45e9, 20e6, 1.0);

        // Check power distribution
        let total_power: f32 = psd.iter().sum();
        assert!(
            total_power > 0.9 && total_power < 1.1,
            "Total power should be ~1.0, got {}",
            total_power
        );
    }

    #[test]
    fn test_fhss_hopping() {
        let mut proto = FHSSProtocol::new(2.45e9, 1e6, 20e6, 20, 0.01);

        let freq1 = proto.current_frequency();

        // Step past hop period
        proto.step(0.011, 0);

        let freq2 = proto.current_frequency();
        assert_ne!(
            freq1, freq2,
            "Frequency should change after hop: {} vs {}",
            freq1, freq2
        );
    }

    #[test]
    fn test_chirp_sweep() {
        let mut proto = ChirpProtocol::new(2.45e9, 10e6, 0.001);

        let freq1 = proto.current_frequency();
        proto.step(0.0005, 0); // Half sweep
        let freq2 = proto.current_frequency();

        // For up-chirp, frequency should increase
        assert!(
            freq2 > freq1,
            "Up-chirp frequency should increase: {} vs {}",
            freq1,
            freq2
        );

        // Check it's approximately halfway through the sweep
        let expected_delta = 10e6 / 2.0;
        let actual_delta = freq2 - freq1;
        assert!(
            (actual_delta - expected_delta).abs() < 1e6,
            "Sweep should be ~half complete: delta {} vs expected {}",
            actual_delta,
            expected_delta
        );
    }

    #[test]
    fn test_dsss_bandwidth() {
        let proto = DSSSProtocol::gps_like(1.57542e9);

        // GPS C/A code has ~2 MHz main lobe
        let bw = proto.current_bandwidth();
        assert!(
            bw > 1.5e6 && bw < 2.5e6,
            "GPS DSSS bandwidth should be ~2 MHz, got {}",
            bw
        );
    }

    #[test]
    fn test_protocol_names() {
        assert_eq!(FlatSpectrum::new(1e9, 1e6).name(), "Flat Spectrum");
        assert_eq!(OFDMProtocol::new(1e9, 20e6, 64).name(), "OFDM");
        assert_eq!(FHSSProtocol::new(1e9, 1e6, 10e6, 10, 0.01).name(), "FHSS");
        assert_eq!(ChirpProtocol::new(1e9, 10e6, 0.001).name(), "Chirp");
        assert_eq!(DSSSProtocol::gps_like(1e9).name(), "DSSS");
    }

    #[test]
    fn test_clone_box() {
        let proto: Box<dyn SignalProtocol> = Box::new(FlatSpectrum::new(1e9, 1e6));
        let cloned = proto.clone_box();
        assert_eq!(cloned.name(), "Flat Spectrum");
    }
}
