//! Multi-Tap Multipath Channel Models
//!
//! SIMD-optimized implementations of ITU tapped delay line channel models:
//! - ITU EPA (Extended Pedestrian A): 7 taps, max delay 410 ns
//! - ITU EVA (Extended Vehicular A): 9 taps, max delay 2510 ns
//! - ITU ETU (Extended Typical Urban): 9 taps, max delay 5000 ns
//!
//! These models are used for 3GPP LTE channel simulation and provide
//! realistic urban mobile channel characteristics.

#[cfg(feature = "simd")]
use std::simd::f32x8;

#[cfg(feature = "simd")]
use crate::simd_rf::math::{simd_cos, simd_sin, simd_sqrt};
#[cfg(feature = "simd")]
use crate::simd_rf::random::SimdRng;

/// Two times Pi
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

// ============================================================================
// ITU Channel Model Definitions
// ============================================================================

/// ITU Extended Channel Models
///
/// Standard 3GPP/ITU channel models for LTE simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ITUChannelModel {
    /// Extended Pedestrian A - 7 taps, max delay 410 ns
    /// Used for low-mobility pedestrian scenarios
    EPA,
    /// Extended Vehicular A - 9 taps, max delay 2510 ns
    /// Used for medium-mobility vehicular scenarios
    EVA,
    /// Extended Typical Urban - 9 taps, max delay 5000 ns
    /// Used for high-delay-spread urban scenarios
    ETU,
}

/// A single tap in the tapped delay line model
#[derive(Debug, Clone, Copy)]
pub struct ChannelTap {
    /// Delay in nanoseconds
    pub delay_ns: f32,
    /// Relative power in dB (0 dB = reference)
    pub power_db: f32,
}

/// Tapped Delay Line channel model
///
/// Represents a multipath channel as a set of discrete taps,
/// each with a delay and relative power.
#[derive(Debug, Clone)]
pub struct TappedDelayLine {
    /// The taps defining the channel
    pub taps: Vec<ChannelTap>,
    /// Maximum excess delay in nanoseconds
    pub max_delay_ns: f32,
    /// RMS delay spread in nanoseconds
    pub rms_delay_ns: f32,
}

impl ITUChannelModel {
    /// Get the tapped delay line for this channel model
    ///
    /// Returns the standard ITU tap delays and powers.
    pub fn get_tdl(&self) -> TappedDelayLine {
        match self {
            ITUChannelModel::EPA => Self::epa_tdl(),
            ITUChannelModel::EVA => Self::eva_tdl(),
            ITUChannelModel::ETU => Self::etu_tdl(),
        }
    }

    /// Get the number of taps for this model
    pub fn num_taps(&self) -> usize {
        match self {
            ITUChannelModel::EPA => 7,
            ITUChannelModel::EVA => 9,
            ITUChannelModel::ETU => 9,
        }
    }

    /// Extended Pedestrian A - 7 taps
    fn epa_tdl() -> TappedDelayLine {
        let taps = vec![
            ChannelTap { delay_ns: 0.0, power_db: 0.0 },
            ChannelTap { delay_ns: 30.0, power_db: -1.0 },
            ChannelTap { delay_ns: 70.0, power_db: -2.0 },
            ChannelTap { delay_ns: 90.0, power_db: -3.0 },
            ChannelTap { delay_ns: 110.0, power_db: -8.0 },
            ChannelTap { delay_ns: 190.0, power_db: -17.2 },
            ChannelTap { delay_ns: 410.0, power_db: -20.8 },
        ];
        TappedDelayLine {
            taps,
            max_delay_ns: 410.0,
            rms_delay_ns: 45.0,
        }
    }

    /// Extended Vehicular A - 9 taps
    fn eva_tdl() -> TappedDelayLine {
        let taps = vec![
            ChannelTap { delay_ns: 0.0, power_db: 0.0 },
            ChannelTap { delay_ns: 30.0, power_db: -1.5 },
            ChannelTap { delay_ns: 150.0, power_db: -1.4 },
            ChannelTap { delay_ns: 310.0, power_db: -3.6 },
            ChannelTap { delay_ns: 370.0, power_db: -0.6 },
            ChannelTap { delay_ns: 710.0, power_db: -9.1 },
            ChannelTap { delay_ns: 1090.0, power_db: -7.0 },
            ChannelTap { delay_ns: 1730.0, power_db: -12.0 },
            ChannelTap { delay_ns: 2510.0, power_db: -16.9 },
        ];
        TappedDelayLine {
            taps,
            max_delay_ns: 2510.0,
            rms_delay_ns: 357.0,
        }
    }

    /// Extended Typical Urban - 9 taps
    fn etu_tdl() -> TappedDelayLine {
        let taps = vec![
            ChannelTap { delay_ns: 0.0, power_db: -1.0 },
            ChannelTap { delay_ns: 50.0, power_db: -1.0 },
            ChannelTap { delay_ns: 120.0, power_db: -1.0 },
            ChannelTap { delay_ns: 200.0, power_db: 0.0 },
            ChannelTap { delay_ns: 230.0, power_db: 0.0 },
            ChannelTap { delay_ns: 500.0, power_db: 0.0 },
            ChannelTap { delay_ns: 1600.0, power_db: -3.0 },
            ChannelTap { delay_ns: 2300.0, power_db: -5.0 },
            ChannelTap { delay_ns: 5000.0, power_db: -7.0 },
        ];
        TappedDelayLine {
            taps,
            max_delay_ns: 5000.0,
            rms_delay_ns: 991.0,
        }
    }
}

impl TappedDelayLine {
    /// Normalize tap powers so they sum to 1.0 (linear scale)
    ///
    /// This ensures power conservation in the channel model.
    pub fn normalize(&mut self) {
        // Convert dB to linear and sum
        let total_power: f32 = self
            .taps
            .iter()
            .map(|t| 10.0_f32.powf(t.power_db / 10.0))
            .sum();

        // Normalize each tap
        let normalization_db = 10.0 * total_power.log10();
        for tap in &mut self.taps {
            tap.power_db -= normalization_db;
        }
    }

    /// Get tap powers in linear scale
    pub fn linear_powers(&self) -> Vec<f32> {
        self.taps
            .iter()
            .map(|t| 10.0_f32.powf(t.power_db / 10.0))
            .collect()
    }

    /// Get tap delays in seconds
    pub fn delays_seconds(&self) -> Vec<f32> {
        self.taps.iter().map(|t| t.delay_ns * 1e-9).collect()
    }

    /// Calculate RMS delay spread from taps
    pub fn calc_rms_delay_spread(&self) -> f32 {
        let powers = self.linear_powers();
        let delays = self.delays_seconds();
        let total_power: f32 = powers.iter().sum();

        if total_power <= 0.0 {
            return 0.0;
        }

        // Mean delay
        let mean_delay: f32 = powers
            .iter()
            .zip(delays.iter())
            .map(|(p, d)| p * d)
            .sum::<f32>()
            / total_power;

        // Mean square delay
        let mean_sq_delay: f32 = powers
            .iter()
            .zip(delays.iter())
            .map(|(p, d)| p * d * d)
            .sum::<f32>()
            / total_power;

        // RMS delay spread
        (mean_sq_delay - mean_delay * mean_delay).sqrt()
    }
}

// ============================================================================
// Complex Number Support
// ============================================================================

/// Simple complex number for channel coefficients
#[derive(Debug, Clone, Copy, Default)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }

    pub fn magnitude(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_squared(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(&self) -> f32 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl std::ops::Mul<Complex> for Complex {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Add<Complex> for Complex {
    type Output = Complex;
    fn add(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Mul<f32> for Complex {
    type Output = Complex;
    fn mul(self, rhs: f32) -> Complex {
        Complex {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

// ============================================================================
// SIMD Complex Number Support
// ============================================================================

#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy)]
pub struct SimdComplex {
    pub re: f32x8,
    pub im: f32x8,
}

#[cfg(feature = "simd")]
impl SimdComplex {
    pub fn new(re: f32x8, im: f32x8) -> Self {
        Self { re, im }
    }

    pub fn splat(c: Complex) -> Self {
        Self {
            re: f32x8::splat(c.re),
            im: f32x8::splat(c.im),
        }
    }

    pub fn from_polar(magnitude: f32x8, phase: f32x8) -> Self {
        Self {
            re: magnitude * simd_cos(phase),
            im: magnitude * simd_sin(phase),
        }
    }

    pub fn magnitude(&self) -> f32x8 {
        simd_sqrt(self.re * self.re + self.im * self.im)
    }

    pub fn magnitude_squared(&self) -> f32x8 {
        self.re * self.re + self.im * self.im
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

#[cfg(feature = "simd")]
impl std::ops::Mul<SimdComplex> for SimdComplex {
    type Output = SimdComplex;
    fn mul(self, rhs: SimdComplex) -> SimdComplex {
        SimdComplex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[cfg(feature = "simd")]
impl std::ops::Add<SimdComplex> for SimdComplex {
    type Output = SimdComplex;
    fn add(self, rhs: SimdComplex) -> SimdComplex {
        SimdComplex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

#[cfg(feature = "simd")]
impl std::ops::Mul<f32x8> for SimdComplex {
    type Output = SimdComplex;
    fn mul(self, rhs: f32x8) -> SimdComplex {
        SimdComplex {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

// ============================================================================
// Channel Impulse Response Generation
// ============================================================================

/// Channel impulse response coefficients for a single environment
#[derive(Debug, Clone)]
pub struct ChannelImpulseResponse {
    /// Complex coefficients for each tap
    pub coefficients: Vec<Complex>,
    /// Delays in seconds for each tap
    pub delays: Vec<f32>,
    /// Linear power for each tap
    pub powers: Vec<f32>,
}

/// Generate time-varying channel impulse response
///
/// Creates Rayleigh fading coefficients for each tap with appropriate
/// Doppler spread based on velocity.
///
/// # Arguments
/// * `model` - ITU channel model (EPA, EVA, ETU)
/// * `velocity_mps` - Mobile velocity in m/s
/// * `carrier_freq_hz` - Carrier frequency in Hz
/// * `time` - Current time in seconds
/// * `rng` - Random number generator
///
/// # Returns
/// Channel impulse response with complex coefficients
pub fn generate_cir_scalar(
    model: ITUChannelModel,
    velocity_mps: f32,
    carrier_freq_hz: f32,
    time: f32,
    rng: &mut impl FnMut() -> (f32, f32),
) -> ChannelImpulseResponse {
    let tdl = model.get_tdl();
    let num_taps = tdl.taps.len();

    // Maximum Doppler frequency
    let speed_of_light = 299_792_458.0_f32;
    let max_doppler = velocity_mps * carrier_freq_hz / speed_of_light;

    let mut coefficients = Vec::with_capacity(num_taps);
    let mut delays = Vec::with_capacity(num_taps);
    let mut powers = Vec::with_capacity(num_taps);

    for tap in &tdl.taps {
        // Convert dB to linear amplitude
        let amplitude = 10.0_f32.powf(tap.power_db / 20.0);

        // Generate Rayleigh fading with Jakes Doppler spectrum
        // Using sum-of-sinusoids method
        let (gauss_re, gauss_im) = rng();

        // Add time-varying Doppler effect
        let doppler_phase = TWO_PI * max_doppler * time;

        // Combine Rayleigh fading with Doppler
        let phase = doppler_phase + gauss_im.atan2(gauss_re);
        let magnitude = amplitude * (gauss_re * gauss_re + gauss_im * gauss_im).sqrt();

        coefficients.push(Complex::from_polar(magnitude, phase));
        delays.push(tap.delay_ns * 1e-9);
        powers.push(amplitude * amplitude);
    }

    ChannelImpulseResponse {
        coefficients,
        delays,
        powers,
    }
}

/// SIMD Channel impulse response for 8 parallel environments
#[cfg(feature = "simd")]
#[derive(Debug, Clone)]
pub struct SimdChannelImpulseResponse {
    /// Complex coefficients for each tap (8 environments per tap)
    pub coefficients: Vec<SimdComplex>,
    /// Delays in seconds for each tap (same for all environments)
    pub delays: Vec<f32>,
    /// Linear power for each tap (8 environments per tap)
    pub powers: Vec<f32x8>,
}

/// Generate SIMD channel impulse response for 8 environments
///
/// # Arguments
/// * `model` - ITU channel model
/// * `velocity_mps` - Mobile velocities (8 parallel)
/// * `carrier_freq_hz` - Carrier frequency (can vary per env)
/// * `time` - Current time (8 parallel)
/// * `rng` - SIMD random number generator
#[cfg(feature = "simd")]
pub fn generate_cir_simd(
    model: ITUChannelModel,
    velocity_mps: f32x8,
    carrier_freq_hz: f32x8,
    time: f32x8,
    rng: &mut SimdRng,
) -> SimdChannelImpulseResponse {
    let tdl = model.get_tdl();
    let num_taps = tdl.taps.len();

    let speed_of_light = f32x8::splat(299_792_458.0);
    let max_doppler = velocity_mps * carrier_freq_hz / speed_of_light;

    let mut coefficients = Vec::with_capacity(num_taps);
    let mut delays = Vec::with_capacity(num_taps);
    let mut powers = Vec::with_capacity(num_taps);

    for tap in &tdl.taps {
        // Convert dB to linear amplitude
        let amplitude = f32x8::splat(10.0_f32.powf(tap.power_db / 20.0));

        // Generate Rayleigh fading - two independent Gaussians
        let (gauss_re, gauss_im) = rng.randn_pair();

        // Time-varying Doppler phase
        let doppler_phase = f32x8::splat(TWO_PI) * max_doppler * time;

        // Rayleigh magnitude
        let rayleigh_mag = simd_sqrt(gauss_re * gauss_re + gauss_im * gauss_im);

        // Combined phase
        // Note: We'd need simd_atan2 for exact phase, using approximation
        let phase = doppler_phase + gauss_im; // Simplified phase variation

        // Final coefficient
        let coeff = SimdComplex::from_polar(amplitude * rayleigh_mag, phase);

        coefficients.push(coeff);
        delays.push(tap.delay_ns * 1e-9);
        powers.push(amplitude * amplitude);
    }

    SimdChannelImpulseResponse {
        coefficients,
        delays,
        powers,
    }
}

// ============================================================================
// Multipath Application
// ============================================================================

/// Apply multipath channel to a signal (scalar version)
///
/// Convolves the input signal with the channel impulse response.
/// This is a simplified version that applies the multipath as a
/// weighted sum without explicit delay modeling.
///
/// # Arguments
/// * `signal_power` - Input signal power (linear)
/// * `cir` - Channel impulse response
///
/// # Returns
/// Output signal power after multipath (linear)
pub fn apply_multipath_power_scalar(signal_power: f32, cir: &ChannelImpulseResponse) -> f32 {
    // Total channel gain = sum of |h_i|^2 where h_i are tap coefficients
    let total_gain: f32 = cir
        .coefficients
        .iter()
        .map(|c| c.magnitude_squared())
        .sum();

    signal_power * total_gain
}

/// Apply multipath channel to signal power (SIMD version)
#[cfg(feature = "simd")]
pub fn apply_multipath_power_simd(signal_power: f32x8, cir: &SimdChannelImpulseResponse) -> f32x8 {
    let mut total_gain = f32x8::splat(0.0);
    for coeff in &cir.coefficients {
        total_gain = total_gain + coeff.magnitude_squared();
    }
    signal_power * total_gain
}

/// Calculate coherence bandwidth from channel model
///
/// B_c ≈ 1 / (5 × τ_rms)
///
/// # Arguments
/// * `model` - ITU channel model
///
/// # Returns
/// Coherence bandwidth in Hz
pub fn coherence_bandwidth_hz(model: ITUChannelModel) -> f32 {
    let tdl = model.get_tdl();
    let rms_delay_s = tdl.rms_delay_ns * 1e-9;
    1.0 / (5.0 * rms_delay_s)
}

/// Calculate frequency selectivity factor
///
/// Compares signal bandwidth to coherence bandwidth.
/// - Factor > 1: Frequency selective fading
/// - Factor < 1: Flat fading
///
/// # Arguments
/// * `signal_bandwidth_hz` - Signal bandwidth
/// * `model` - ITU channel model
///
/// # Returns
/// Frequency selectivity factor
pub fn frequency_selectivity(signal_bandwidth_hz: f32, model: ITUChannelModel) -> f32 {
    signal_bandwidth_hz / coherence_bandwidth_hz(model)
}

/// Check if channel is frequency selective for given bandwidth
pub fn is_frequency_selective(signal_bandwidth_hz: f32, model: ITUChannelModel) -> bool {
    frequency_selectivity(signal_bandwidth_hz, model) > 1.0
}

// ============================================================================
// Tap Filtering by Bandwidth
// ============================================================================

/// Get effective taps for a given signal bandwidth
///
/// Filters out taps with delays smaller than the symbol period,
/// as they would be combined in the receiver.
///
/// # Arguments
/// * `model` - ITU channel model
/// * `signal_bandwidth_hz` - Signal bandwidth in Hz
///
/// # Returns
/// Filtered TappedDelayLine with resolvable taps
pub fn get_resolvable_taps(model: ITUChannelModel, signal_bandwidth_hz: f32) -> TappedDelayLine {
    let tdl = model.get_tdl();

    // Minimum resolvable delay ≈ 1/B
    let min_delay_s = 1.0 / signal_bandwidth_hz;
    let min_delay_ns = min_delay_s * 1e9;

    // Group taps that are within one symbol period
    let mut grouped_taps = Vec::new();
    let mut current_delay = 0.0_f32;
    let mut current_power_linear = 0.0_f32;

    for tap in &tdl.taps {
        if tap.delay_ns - current_delay >= min_delay_ns && current_power_linear > 0.0 {
            // Output previous group
            grouped_taps.push(ChannelTap {
                delay_ns: current_delay,
                power_db: 10.0 * current_power_linear.log10(),
            });
            current_delay = tap.delay_ns;
            current_power_linear = 10.0_f32.powf(tap.power_db / 10.0);
        } else {
            // Add to current group
            current_power_linear += 10.0_f32.powf(tap.power_db / 10.0);
            if grouped_taps.is_empty() {
                current_delay = tap.delay_ns;
            }
        }
    }

    // Add last group
    if current_power_linear > 0.0 {
        grouped_taps.push(ChannelTap {
            delay_ns: current_delay,
            power_db: 10.0 * current_power_linear.log10(),
        });
    }

    let max_delay = grouped_taps.last().map(|t| t.delay_ns).unwrap_or(0.0);

    TappedDelayLine {
        taps: grouped_taps,
        max_delay_ns: max_delay,
        rms_delay_ns: tdl.rms_delay_ns, // Approximate
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epa_tap_count() {
        let model = ITUChannelModel::EPA;
        assert_eq!(model.num_taps(), 7);
        assert_eq!(model.get_tdl().taps.len(), 7);
    }

    #[test]
    fn test_eva_tap_count() {
        let model = ITUChannelModel::EVA;
        assert_eq!(model.num_taps(), 9);
        assert_eq!(model.get_tdl().taps.len(), 9);
    }

    #[test]
    fn test_etu_tap_count() {
        let model = ITUChannelModel::ETU;
        assert_eq!(model.num_taps(), 9);
        assert_eq!(model.get_tdl().taps.len(), 9);
    }

    #[test]
    fn test_epa_max_delay() {
        let tdl = ITUChannelModel::EPA.get_tdl();
        assert!((tdl.max_delay_ns - 410.0).abs() < 1.0);
    }

    #[test]
    fn test_eva_max_delay() {
        let tdl = ITUChannelModel::EVA.get_tdl();
        assert!((tdl.max_delay_ns - 2510.0).abs() < 1.0);
    }

    #[test]
    fn test_etu_max_delay() {
        let tdl = ITUChannelModel::ETU.get_tdl();
        assert!((tdl.max_delay_ns - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_tap_powers_are_valid() {
        for model in [ITUChannelModel::EPA, ITUChannelModel::EVA, ITUChannelModel::ETU] {
            let tdl = model.get_tdl();
            for tap in &tdl.taps {
                // All powers should be <= 0 dB (no gain)
                assert!(
                    tap.power_db <= 0.1,
                    "Tap power {} dB exceeds 0 dB for {:?}",
                    tap.power_db,
                    model
                );
            }
        }
    }

    #[test]
    fn test_tap_delays_increasing() {
        for model in [ITUChannelModel::EPA, ITUChannelModel::EVA, ITUChannelModel::ETU] {
            let tdl = model.get_tdl();
            let mut prev_delay = -1.0;
            for tap in &tdl.taps {
                assert!(
                    tap.delay_ns >= prev_delay,
                    "Delays not monotonic for {:?}: {} < {}",
                    model,
                    tap.delay_ns,
                    prev_delay
                );
                prev_delay = tap.delay_ns;
            }
        }
    }

    #[test]
    fn test_complex_magnitude() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.magnitude() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_from_polar() {
        let c = Complex::from_polar(5.0, 0.0);
        assert!((c.re - 5.0).abs() < 1e-6);
        assert!(c.im.abs() < 1e-6);

        let c2 = Complex::from_polar(1.0, std::f32::consts::FRAC_PI_2);
        assert!(c2.re.abs() < 1e-6);
        assert!((c2.im - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_multiply() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert!((c.re - (-5.0)).abs() < 1e-6);
        assert!((c.im - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_tdl_normalize() {
        let mut tdl = ITUChannelModel::EPA.get_tdl();
        tdl.normalize();

        let total_power: f32 = tdl.linear_powers().iter().sum();
        assert!(
            (total_power - 1.0).abs() < 0.01,
            "Normalized power should be 1.0, got {}",
            total_power
        );
    }

    #[test]
    fn test_coherence_bandwidth() {
        // EPA has 45 ns RMS delay spread -> B_c ≈ 4.4 MHz
        let bc_epa = coherence_bandwidth_hz(ITUChannelModel::EPA);
        assert!(
            bc_epa > 3e6 && bc_epa < 6e6,
            "EPA coherence bandwidth {} Hz unexpected",
            bc_epa
        );

        // ETU has larger delay spread -> smaller coherence bandwidth
        let bc_etu = coherence_bandwidth_hz(ITUChannelModel::ETU);
        assert!(
            bc_etu < bc_epa,
            "ETU should have smaller coherence bandwidth than EPA"
        );
    }

    #[test]
    fn test_frequency_selectivity() {
        // Narrowband signal (100 kHz) should be flat fading
        let narrow = frequency_selectivity(100_000.0, ITUChannelModel::EPA);
        assert!(narrow < 1.0, "100 kHz should be flat fading in EPA");

        // Wideband signal (20 MHz) should be frequency selective
        let wide = frequency_selectivity(20_000_000.0, ITUChannelModel::EPA);
        assert!(wide > 1.0, "20 MHz should be frequency selective in EPA");
    }

    #[test]
    fn test_generate_cir_scalar() {
        let mut rng_state = 12345u32;
        let mut rng = || {
            // Simple LCG for testing
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r1 = (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let r2 = (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            (r1, r2)
        };

        let cir = generate_cir_scalar(
            ITUChannelModel::EPA,
            10.0,    // 10 m/s velocity
            2.4e9,   // 2.4 GHz carrier
            0.0,     // t=0
            &mut rng,
        );

        assert_eq!(cir.coefficients.len(), 7, "EPA should have 7 taps");
        assert_eq!(cir.delays.len(), 7);
        assert_eq!(cir.powers.len(), 7);

        // Check delays are in seconds
        assert!(
            cir.delays[0] < 1e-6,
            "First delay should be very small: {}",
            cir.delays[0]
        );
    }

    #[test]
    fn test_apply_multipath_power() {
        // Create a simple CIR with known gain
        let cir = ChannelImpulseResponse {
            coefficients: vec![
                Complex::new(0.5, 0.0),
                Complex::new(0.3, 0.4), // magnitude 0.5
            ],
            delays: vec![0.0, 100e-9],
            powers: vec![0.25, 0.25],
        };

        let input_power = 1.0;
        let output_power = apply_multipath_power_scalar(input_power, &cir);

        // Expected: 0.5² + 0.5² = 0.5
        assert!(
            (output_power - 0.5).abs() < 0.01,
            "Multipath power: expected 0.5, got {}",
            output_power
        );
    }

    #[test]
    fn test_resolvable_taps_wideband() {
        // With 100 MHz bandwidth (10 ns symbol), all EPA taps should be resolvable
        let taps = get_resolvable_taps(ITUChannelModel::EPA, 100e6);
        // Should have close to original number of taps
        assert!(
            taps.taps.len() >= 5,
            "Wideband should resolve most taps: {}",
            taps.taps.len()
        );
    }

    #[test]
    fn test_resolvable_taps_narrowband() {
        // With 1 MHz bandwidth (1 us symbol), many EPA taps should be grouped
        let taps = get_resolvable_taps(ITUChannelModel::EPA, 1e6);
        // Should have fewer taps due to grouping
        assert!(
            taps.taps.len() < 7,
            "Narrowband should group taps: {}",
            taps.taps.len()
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_complex_magnitude() {
        let c = SimdComplex::new(f32x8::splat(3.0), f32x8::splat(4.0));
        let mag = c.magnitude();
        let arr: [f32; 8] = mag.into();
        for m in arr {
            assert!((m - 5.0).abs() < 1e-5);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_cir_tap_count() {
        let mut rng = SimdRng::new(42);
        let cir = generate_cir_simd(
            ITUChannelModel::EVA,
            f32x8::splat(30.0),  // 30 m/s
            f32x8::splat(1.8e9), // 1.8 GHz
            f32x8::splat(0.0),
            &mut rng,
        );

        assert_eq!(cir.coefficients.len(), 9, "EVA should have 9 taps");
        assert_eq!(cir.delays.len(), 9);
        assert_eq!(cir.powers.len(), 9);
    }
}
