//! Observation noise injection for domain randomization.
//!
//! Provides configurable Gaussian noise injection per observation component
//! to improve policy robustness through domain randomization.
//!
//! Uses a fast XorShift64 PRNG with Box-Muller transform for Gaussian samples.

#[cfg(feature = "simd")]
use std::simd::f32x8;

// ============================================================================
// Noise Configuration
// ============================================================================

/// Configuration for observation noise injection.
///
/// Each field specifies the standard deviation (sigma) of Gaussian noise
/// to add to that component. `None` means no noise for that component.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseConfig {
    /// Position noise standard deviation [x, y, z] in meters.
    pub position_std: Option<[f32; 3]>,

    /// Attitude noise standard deviation [roll, pitch, yaw] in radians.
    /// Applied to Euler angles.
    pub attitude_std: Option<[f32; 3]>,

    /// Velocity noise standard deviation [vx, vy, vz] in m/s.
    pub velocity_std: Option<[f32; 3]>,

    /// Angular velocity noise standard deviation [wx, wy, wz] in rad/s.
    pub angular_velocity_std: Option<[f32; 3]>,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl NoiseConfig {
    /// No noise (disabled).
    pub const fn disabled() -> Self {
        Self {
            position_std: None,
            attitude_std: None,
            velocity_std: None,
            angular_velocity_std: None,
        }
    }

    /// Typical sensor noise for a micro quadcopter.
    ///
    /// Based on Crazyflie sensor characteristics:
    /// - Position: ~1cm accuracy (motion capture or visual odometry)
    /// - Attitude: ~1 degree accuracy (IMU fusion)
    /// - Velocity: ~0.05 m/s accuracy (derived from position)
    /// - Angular velocity: ~0.1 rad/s accuracy (gyroscope)
    pub fn sensor_realistic() -> Self {
        Self {
            position_std: Some([0.01, 0.01, 0.01]),      // 1cm
            attitude_std: Some([0.017, 0.017, 0.017]),   // ~1 degree
            velocity_std: Some([0.05, 0.05, 0.05]),      // 5cm/s
            angular_velocity_std: Some([0.1, 0.1, 0.1]), // 0.1 rad/s
        }
    }

    /// High noise for aggressive domain randomization.
    pub fn high() -> Self {
        Self {
            position_std: Some([0.05, 0.05, 0.05]),      // 5cm
            attitude_std: Some([0.05, 0.05, 0.05]),      // ~3 degrees
            velocity_std: Some([0.2, 0.2, 0.2]),         // 20cm/s
            angular_velocity_std: Some([0.3, 0.3, 0.3]), // 0.3 rad/s
        }
    }

    /// Check if any noise is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.position_std.is_some()
            || self.attitude_std.is_some()
            || self.velocity_std.is_some()
            || self.angular_velocity_std.is_some()
    }

    /// Builder: set position noise.
    pub fn with_position_noise(mut self, std: [f32; 3]) -> Self {
        self.position_std = Some(std);
        self
    }

    /// Builder: set attitude noise.
    pub fn with_attitude_noise(mut self, std: [f32; 3]) -> Self {
        self.attitude_std = Some(std);
        self
    }

    /// Builder: set velocity noise.
    pub fn with_velocity_noise(mut self, std: [f32; 3]) -> Self {
        self.velocity_std = Some(std);
        self
    }

    /// Builder: set angular velocity noise.
    pub fn with_angular_velocity_noise(mut self, std: [f32; 3]) -> Self {
        self.angular_velocity_std = Some(std);
        self
    }
}

// ============================================================================
// XorShift Random Number Generator
// ============================================================================

/// Fast XorShift64 PRNG for noise generation.
///
/// This is a simple, fast PRNG suitable for simulation noise.
/// Not cryptographically secure.
#[derive(Clone, Debug)]
pub struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    /// Create with a specific seed.
    #[inline]
    pub fn new(seed: u64) -> Self {
        // Ensure non-zero state
        let state = if seed == 0 { 0xDEADBEEF } else { seed };
        Self { state }
    }

    /// Generate the next random u64.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f32 in [0, 1).
    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        // Use upper 23 bits for mantissa
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Generate a pair of independent Gaussian samples using Box-Muller.
    ///
    /// Returns two independent samples from N(0, 1).
    #[inline]
    pub fn next_gaussian_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-10); // Avoid log(0)
        let u2 = self.next_f32();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        (r * theta.cos(), r * theta.sin())
    }

    /// Generate a single Gaussian sample from N(0, 1).
    ///
    /// Less efficient than `next_gaussian_pair` if you need multiple samples.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        self.next_gaussian_pair().0
    }

    /// Generate three independent Gaussian samples (for 3D vectors).
    #[inline]
    pub fn next_gaussian_3d(&mut self) -> [f32; 3] {
        let (g0, g1) = self.next_gaussian_pair();
        let g2 = self.next_gaussian();
        [g0, g1, g2]
    }

    /// Apply Gaussian noise to a 3D vector.
    ///
    /// # Arguments
    /// * `value` - Original value
    /// * `std` - Standard deviation per component
    #[inline]
    pub fn apply_noise_3d(&mut self, value: &mut [f32; 3], std: [f32; 3]) {
        let noise = self.next_gaussian_3d();
        value[0] += noise[0] * std[0];
        value[1] += noise[1] * std[1];
        value[2] += noise[2] * std[2];
    }
}

impl Default for XorShiftRng {
    fn default() -> Self {
        Self::new(42)
    }
}

// ============================================================================
// SIMD Noise Generation
// ============================================================================

#[cfg(feature = "simd")]
pub mod simd {
    use super::*;

    /// Generate 8 independent Gaussian samples using SIMD-friendly structure.
    ///
    /// Uses 4 Box-Muller pairs to generate 8 samples.
    #[inline]
    pub fn next_gaussian_8(rng: &mut XorShiftRng) -> [f32; 8] {
        let (g0, g1) = rng.next_gaussian_pair();
        let (g2, g3) = rng.next_gaussian_pair();
        let (g4, g5) = rng.next_gaussian_pair();
        let (g6, g7) = rng.next_gaussian_pair();
        [g0, g1, g2, g3, g4, g5, g6, g7]
    }

    /// Generate f32x8 of Gaussian samples.
    #[inline]
    pub fn next_gaussian_simd(rng: &mut XorShiftRng) -> f32x8 {
        f32x8::from_array(next_gaussian_8(rng))
    }

    /// Apply noise to f32x8 values.
    #[inline]
    pub fn apply_noise_simd(value: f32x8, std: f32, rng: &mut XorShiftRng) -> f32x8 {
        let noise = next_gaussian_simd(rng);
        value + noise * f32x8::splat(std)
    }
}

// ============================================================================
// Noise Application Functions
// ============================================================================

/// Apply noise to an observation buffer based on config.
///
/// This modifies the observation buffer in place, adding Gaussian noise
/// to the components specified in the config.
///
/// # Arguments
/// * `obs` - Observation buffer (modified in place)
/// * `config` - Noise configuration
/// * `obs_config` - Observation configuration (to know component offsets)
/// * `rng` - Random number generator
pub fn apply_observation_noise(
    obs: &mut [f32],
    noise_config: &NoiseConfig,
    position_offset: Option<usize>,
    attitude_offset: Option<usize>,
    velocity_offset: Option<usize>,
    angular_velocity_offset: Option<usize>,
    rng: &mut XorShiftRng,
) {
    // Apply position noise
    if let (Some(std), Some(offset)) = (&noise_config.position_std, position_offset) {
        let noise = rng.next_gaussian_3d();
        obs[offset] += noise[0] * std[0];
        obs[offset + 1] += noise[1] * std[1];
        obs[offset + 2] += noise[2] * std[2];
    }

    // Apply attitude noise (to Euler angles)
    if let (Some(std), Some(offset)) = (&noise_config.attitude_std, attitude_offset) {
        let noise = rng.next_gaussian_3d();
        obs[offset] += noise[0] * std[0];
        obs[offset + 1] += noise[1] * std[1];
        obs[offset + 2] += noise[2] * std[2];
    }

    // Apply velocity noise
    if let (Some(std), Some(offset)) = (&noise_config.velocity_std, velocity_offset) {
        let noise = rng.next_gaussian_3d();
        obs[offset] += noise[0] * std[0];
        obs[offset + 1] += noise[1] * std[1];
        obs[offset + 2] += noise[2] * std[2];
    }

    // Apply angular velocity noise
    if let (Some(std), Some(offset)) = (&noise_config.angular_velocity_std, angular_velocity_offset)
    {
        let noise = rng.next_gaussian_3d();
        obs[offset] += noise[0] * std[0];
        obs[offset + 1] += noise[1] * std[1];
        obs[offset + 2] += noise[2] * std[2];
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = XorShiftRng::new(42);
        let mut rng2 = XorShiftRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_different_seeds() {
        let mut rng1 = XorShiftRng::new(1);
        let mut rng2 = XorShiftRng::new(2);

        // Should produce different sequences
        let seq1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_uniform_distribution() {
        let mut rng = XorShiftRng::new(12345);
        let samples: Vec<f32> = (0..10000).map(|_| rng.next_f32()).collect();

        // All should be in [0, 1)
        for &s in &samples {
            assert!(s >= 0.0 && s < 1.0, "Sample out of range: {}", s);
        }

        // Mean should be close to 0.5
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            (mean - 0.5).abs() < 0.02,
            "Uniform mean should be ~0.5, got {}",
            mean
        );
    }

    #[test]
    fn test_gaussian_distribution() {
        let mut rng = XorShiftRng::new(54321);
        let samples: Vec<f32> = (0..10000).map(|_| rng.next_gaussian()).collect();

        // Mean should be close to 0
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            mean.abs() < 0.05,
            "Gaussian mean should be ~0, got {}",
            mean
        );

        // Variance should be close to 1
        let variance: f32 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        assert!(
            (variance - 1.0).abs() < 0.1,
            "Gaussian variance should be ~1, got {}",
            variance
        );
    }

    #[test]
    fn test_noise_config_disabled() {
        let config = NoiseConfig::disabled();
        assert!(!config.is_enabled());
    }

    #[test]
    fn test_noise_config_enabled() {
        let config = NoiseConfig::sensor_realistic();
        assert!(config.is_enabled());
    }

    #[test]
    fn test_noise_config_builder() {
        let config = NoiseConfig::disabled()
            .with_position_noise([0.01, 0.01, 0.01])
            .with_velocity_noise([0.05, 0.05, 0.05]);

        assert!(config.position_std.is_some());
        assert!(config.velocity_std.is_some());
        assert!(config.attitude_std.is_none());
        assert!(config.angular_velocity_std.is_none());
    }

    #[test]
    fn test_apply_noise_3d() {
        let mut rng = XorShiftRng::new(42);
        let mut value = [0.0, 0.0, 0.0];
        let std = [1.0, 1.0, 1.0];

        rng.apply_noise_3d(&mut value, std);

        // Values should have changed
        assert!(value[0] != 0.0 || value[1] != 0.0 || value[2] != 0.0);
    }

    #[test]
    fn test_apply_observation_noise() {
        let mut rng = XorShiftRng::new(42);
        let noise_config = NoiseConfig::disabled().with_position_noise([0.1, 0.1, 0.1]);

        // Simulate observation buffer: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        let mut obs = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let original = obs.clone();

        apply_observation_noise(
            &mut obs,
            &noise_config,
            Some(0), // position at offset 0
            None,
            None,
            None,
            &mut rng,
        );

        // Position should have changed
        assert!(obs[0] != original[0] || obs[1] != original[1] || obs[2] != original[2]);

        // Velocity should be unchanged
        assert_eq!(obs[3], original[3]);
        assert_eq!(obs[4], original[4]);
        assert_eq!(obs[5], original[5]);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_gaussian_generation() {
        let mut rng = XorShiftRng::new(42);
        let samples = simd::next_gaussian_8(&mut rng);

        // Should have 8 different values (with high probability)
        let unique: std::collections::HashSet<u32> =
            samples.iter().map(|x| x.to_bits()).collect();
        assert!(unique.len() >= 6, "SIMD samples should be diverse");
    }
}
