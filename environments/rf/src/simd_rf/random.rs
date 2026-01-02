//! SIMD Random Number Generation
//!
//! High-performance vectorized random number generator using Xoshiro256++
//! algorithm, producing 8 independent random streams in parallel.
//!
//! # Distributions
//!
//! - Uniform [0, 1)
//! - Standard Normal (Box-Muller transform)
//! - Rayleigh (for fading simulation)
//! - Exponential (for inter-arrival times)
//! - Bernoulli (for occupancy sampling)

use std::simd::{
    cmp::SimdPartialOrd,
    f32x8,
    num::{SimdFloat, SimdUint},
    u64x8, Mask,
};

use super::math::{simd_cos, simd_log, simd_sin, simd_sqrt, TWO_PI};

/// SIMD Random Number Generator using Xoshiro256++
///
/// Each of the 8 SIMD lanes has its own independent state, allowing
/// for parallel generation of random numbers for 8 environments.
pub struct SimdRng {
    /// State vectors for Xoshiro256++ (4 × 8 u64 values)
    state: [u64x8; 4],
}

impl SimdRng {
    /// Create a new RNG seeded from a single seed value
    ///
    /// Uses SplitMix64 to expand the seed into per-lane states.
    pub fn new(seed: u64) -> Self {
        let mut state = [u64x8::splat(0); 4];

        // Initialize each lane with a different seed using SplitMix64
        for lane in 0..8 {
            let mut sm_state = seed.wrapping_add((lane as u64).wrapping_mul(0x9E3779B97F4A7C15));

            for s in &mut state {
                // SplitMix64 step
                sm_state = sm_state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = sm_state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z = z ^ (z >> 31);

                // Set this lane's state component
                let mut arr: [u64; 8] = (*s).into();
                arr[lane] = z;
                *s = u64x8::from_array(arr);
            }
        }

        Self { state }
    }

    /// Create a new RNG with per-lane seeds
    ///
    /// Each lane gets its own seed, useful for ensuring reproducibility
    /// in multi-environment simulations.
    pub fn from_seeds(seeds: [u64; 8]) -> Self {
        let mut state = [u64x8::splat(0); 4];

        for (lane, &seed) in seeds.iter().enumerate() {
            let mut sm_state = seed;

            for s in &mut state {
                // SplitMix64 step
                sm_state = sm_state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = sm_state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z = z ^ (z >> 31);

                let mut arr: [u64; 8] = (*s).into();
                arr[lane] = z;
                *s = u64x8::from_array(arr);
            }
        }

        Self { state }
    }

    /// Generate raw u64 values using Xoshiro256++
    #[inline]
    fn next_u64(&mut self) -> u64x8 {
        // Xoshiro256++ scrambler: rotl(s0 + s3, 23) + s0
        let s0 = self.state[0];
        let s3 = self.state[3];
        let sum = s0 + s3;

        // rotl(x, 23) = (x << 23) | (x >> 41)
        let result = ((sum << 23) | (sum >> 41)) + s0;

        // Xoshiro256++ state update
        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;

        // rotl(s3, 45)
        self.state[3] = (self.state[3] << 45) | (self.state[3] >> 19);

        result
    }

    /// Generate uniform random f32 values in [0, 1)
    #[inline]
    pub fn uniform(&mut self) -> f32x8 {
        let bits = self.next_u64();

        // Convert upper 23 bits to mantissa of f32 in [1, 2)
        // Then subtract 1 to get [0, 1)
        let upper_bits = (bits >> 41).cast::<u32>(); // 23 bits
        let one_bits = std::simd::u32x8::splat(0x3F800000); // 1.0 in f32 bits
        let float_bits = one_bits | upper_bits;
        let float_val = f32x8::from_bits(float_bits);

        float_val - f32x8::splat(1.0)
    }

    /// Generate uniform random f32 values in [low, high)
    #[inline]
    pub fn uniform_range(&mut self, low: f32x8, high: f32x8) -> f32x8 {
        low + self.uniform() * (high - low)
    }

    /// Generate standard normal random values using Box-Muller transform
    ///
    /// Returns N(0, 1) distributed values.
    #[inline]
    pub fn randn(&mut self) -> f32x8 {
        // Box-Muller: Z = sqrt(-2*ln(U1)) * cos(2*pi*U2)
        let u1 = self.uniform();
        let u2 = self.uniform();

        // Avoid log(0) by clamping
        let eps = f32x8::splat(1e-10);
        let u1_safe = u1 + eps;

        let r = simd_sqrt(f32x8::splat(-2.0) * simd_log(u1_safe));
        let theta = f32x8::splat(TWO_PI) * u2;

        r * simd_cos(theta)
    }

    /// Generate a pair of standard normal values (more efficient)
    ///
    /// Box-Muller naturally produces two values, so this is more efficient
    /// when you need multiple normal samples.
    #[inline]
    pub fn randn_pair(&mut self) -> (f32x8, f32x8) {
        let u1 = self.uniform();
        let u2 = self.uniform();

        let eps = f32x8::splat(1e-10);
        let u1_safe = u1 + eps;

        let r = simd_sqrt(f32x8::splat(-2.0) * simd_log(u1_safe));
        let theta = f32x8::splat(TWO_PI) * u2;

        (r * simd_cos(theta), r * simd_sin(theta))
    }

    /// Generate normal random values with specified mean and standard deviation
    #[inline]
    pub fn normal(&mut self, mean: f32x8, std: f32x8) -> f32x8 {
        mean + std * self.randn()
    }

    /// Generate Rayleigh distributed random values
    ///
    /// Rayleigh distribution is used for modeling fading in non-line-of-sight
    /// wireless channels. The PDF is p(r) = (r/σ²) * exp(-r²/(2σ²))
    ///
    /// # Arguments
    /// * `sigma` - Scale parameter
    ///
    /// # Returns
    /// Values with mean σ√(π/2) ≈ 1.253σ
    #[inline]
    pub fn rayleigh(&mut self, sigma: f32x8) -> f32x8 {
        // R = σ * sqrt(-2 * ln(U))
        let u = self.uniform();
        let eps = f32x8::splat(1e-10);
        let u_safe = u + eps;

        sigma * simd_sqrt(f32x8::splat(-2.0) * simd_log(u_safe))
    }

    /// Generate exponential distributed random values
    ///
    /// Exponential distribution is used for modeling inter-arrival times
    /// and packet durations.
    ///
    /// # Arguments
    /// * `lambda` - Rate parameter (mean = 1/lambda)
    ///
    /// # Returns
    /// Values with mean 1/λ
    #[inline]
    pub fn exponential(&mut self, lambda: f32x8) -> f32x8 {
        // X = -ln(U) / λ
        let u = self.uniform();
        let eps = f32x8::splat(1e-10);
        let u_safe = u + eps;

        -simd_log(u_safe) / lambda
    }

    /// Generate Bernoulli samples (coin flips)
    ///
    /// Returns a mask where true indicates success with probability p.
    ///
    /// # Arguments
    /// * `p` - Success probability in [0, 1]
    #[inline]
    pub fn bernoulli(&mut self, p: f32x8) -> Mask<i32, 8> {
        self.uniform().simd_lt(p)
    }

    /// Generate Bernoulli samples as f32 (1.0 for success, 0.0 for failure)
    #[inline]
    pub fn bernoulli_f32(&mut self, p: f32x8) -> f32x8 {
        let mask = self.bernoulli(p);
        mask.select(f32x8::splat(1.0), f32x8::splat(0.0))
    }

    /// Generate log-normal distributed random values
    ///
    /// Log-normal is used for modeling shadowing in wireless channels.
    ///
    /// # Arguments
    /// * `mu` - Mean of underlying normal distribution
    /// * `sigma` - Std dev of underlying normal distribution
    #[inline]
    pub fn log_normal(&mut self, mu: f32x8, sigma: f32x8) -> f32x8 {
        super::math::simd_exp(mu + sigma * self.randn())
    }

    /// Generate Rician distributed random values
    ///
    /// Rician distribution is used for line-of-sight fading channels.
    /// When K=0, this reduces to Rayleigh.
    ///
    /// # Arguments
    /// * `nu` - Non-centrality parameter (direct path strength)
    /// * `sigma` - Scale parameter (scattered path strength)
    #[inline]
    pub fn rician(&mut self, nu: f32x8, sigma: f32x8) -> f32x8 {
        // R = sqrt((X + ν)² + Y²) where X, Y ~ N(0, σ)
        let x = sigma * self.randn() + nu;
        let y = sigma * self.randn();

        simd_sqrt(x * x + y * y)
    }

    /// Jump the RNG state forward by 2^128 steps
    ///
    /// Useful for creating independent sub-sequences for different
    /// simulation runs without correlation.
    pub fn jump(&mut self) {
        // Jump constants for Xoshiro256++
        const JUMP: [u64; 4] = [
            0x180EC6D33CFD0ABA,
            0xD5A61266F0C9392C,
            0xA9582618E03FC9AA,
            0x39ABDC4529B1661C,
        ];

        let mut s0 = u64x8::splat(0);
        let mut s1 = u64x8::splat(0);
        let mut s2 = u64x8::splat(0);
        let mut s3 = u64x8::splat(0);

        for jc in JUMP.iter() {
            for b in 0..64 {
                if (jc >> b) & 1 != 0 {
                    s0 ^= self.state[0];
                    s1 ^= self.state[1];
                    s2 ^= self.state[2];
                    s3 ^= self.state[3];
                }
                let _ = self.next_u64();
            }
        }

        self.state[0] = s0;
        self.state[1] = s1;
        self.state[2] = s2;
        self.state[3] = s3;
    }
}

impl Clone for SimdRng {
    fn clone(&self) -> Self {
        Self {
            state: self.state,
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const NUM_SAMPLES: usize = 10000;

    #[test]
    fn test_uniform_range() {
        let mut rng = SimdRng::new(12345);

        for _ in 0..NUM_SAMPLES {
            let u = rng.uniform();
            let arr: [f32; 8] = u.into();

            for (lane, val) in arr.iter().enumerate() {
                assert!(
                    *val >= 0.0 && *val < 1.0,
                    "uniform lane {} out of range: {}",
                    lane,
                    val
                );
            }
        }
    }

    #[test]
    fn test_uniform_mean() {
        let mut rng = SimdRng::new(12345);
        let mut sum = f32x8::splat(0.0);

        for _ in 0..NUM_SAMPLES {
            sum = sum + rng.uniform();
        }

        let mean = sum / f32x8::splat(NUM_SAMPLES as f32);
        let mean_arr: [f32; 8] = mean.into();

        // Expected mean = 0.5, with variance = 1/12
        // Std error = sqrt(1/(12*N)) ≈ 0.003 for N=10000
        // 3σ tolerance ≈ 0.01
        for (lane, m) in mean_arr.iter().enumerate() {
            assert!(
                (*m - 0.5).abs() < 0.02,
                "uniform mean lane {}: expected ~0.5, got {}",
                lane,
                m
            );
        }
    }

    #[test]
    fn test_randn_mean_std() {
        let mut rng = SimdRng::new(12345);
        let mut sum = f32x8::splat(0.0);
        let mut sum_sq = f32x8::splat(0.0);

        for _ in 0..NUM_SAMPLES {
            let z = rng.randn();
            sum = sum + z;
            sum_sq = sum_sq + z * z;
        }

        let n = f32x8::splat(NUM_SAMPLES as f32);
        let mean = sum / n;
        let variance = sum_sq / n - mean * mean;
        let std = simd_sqrt(variance);

        let mean_arr: [f32; 8] = mean.into();
        let std_arr: [f32; 8] = std.into();

        // Check mean ≈ 0 (within 3σ ≈ 0.03)
        for (lane, m) in mean_arr.iter().enumerate() {
            assert!(
                m.abs() < 0.05,
                "randn mean lane {}: expected ~0, got {}",
                lane,
                m
            );
        }

        // Check std ≈ 1 (within reasonable tolerance)
        for (lane, s) in std_arr.iter().enumerate() {
            assert!(
                (*s - 1.0).abs() < 0.1,
                "randn std lane {}: expected ~1, got {}",
                lane,
                s
            );
        }
    }

    #[test]
    fn test_rayleigh_mean() {
        let mut rng = SimdRng::new(12345);
        let sigma = f32x8::splat(1.0);
        let mut sum = f32x8::splat(0.0);

        for _ in 0..NUM_SAMPLES {
            sum = sum + rng.rayleigh(sigma);
        }

        let mean = sum / f32x8::splat(NUM_SAMPLES as f32);
        let mean_arr: [f32; 8] = mean.into();

        // Expected mean = σ * sqrt(π/2) ≈ 1.253
        let expected_mean = (std::f32::consts::PI / 2.0).sqrt();

        for (lane, m) in mean_arr.iter().enumerate() {
            assert!(
                (*m - expected_mean).abs() < 0.05,
                "rayleigh mean lane {}: expected ~{}, got {}",
                lane,
                expected_mean,
                m
            );
        }
    }

    #[test]
    fn test_exponential_mean() {
        let mut rng = SimdRng::new(12345);
        let lambda = f32x8::splat(2.0); // Mean should be 0.5
        let mut sum = f32x8::splat(0.0);

        for _ in 0..NUM_SAMPLES {
            sum = sum + rng.exponential(lambda);
        }

        let mean = sum / f32x8::splat(NUM_SAMPLES as f32);
        let mean_arr: [f32; 8] = mean.into();

        // Expected mean = 1/λ = 0.5
        for (lane, m) in mean_arr.iter().enumerate() {
            assert!(
                (*m - 0.5).abs() < 0.05,
                "exponential mean lane {}: expected ~0.5, got {}",
                lane,
                m
            );
        }
    }

    #[test]
    fn test_bernoulli_probability() {
        let mut rng = SimdRng::new(12345);
        let p = f32x8::splat(0.3);
        let mut count = f32x8::splat(0.0);

        for _ in 0..NUM_SAMPLES {
            count = count + rng.bernoulli_f32(p);
        }

        let proportion = count / f32x8::splat(NUM_SAMPLES as f32);
        let prop_arr: [f32; 8] = proportion.into();

        for (lane, prop) in prop_arr.iter().enumerate() {
            assert!(
                (*prop - 0.3).abs() < 0.03,
                "bernoulli proportion lane {}: expected ~0.3, got {}",
                lane,
                prop
            );
        }
    }

    #[test]
    fn test_lane_independence() {
        // Verify that different lanes produce different sequences
        let mut rng = SimdRng::new(12345);

        let u = rng.uniform();
        let arr: [f32; 8] = u.into();

        // Check that at least some lanes are different
        let all_same = arr.iter().all(|&x| (x - arr[0]).abs() < 1e-6);
        assert!(
            !all_same,
            "All lanes produced the same value - may indicate improper seeding"
        );
    }
}
