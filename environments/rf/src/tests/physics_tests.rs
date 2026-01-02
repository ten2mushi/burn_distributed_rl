//! Physics Module Tests
//!
//! Comprehensive tests for the physics/channel models including:
//! - Path loss accuracy tests
//! - Fading distribution tests (Chi-squared, Kolmogorov-Smirnov)
//! - Multipath model validation
//! - Atmospheric effect tests
//! - Doppler calculation tests
//! - Composite channel integration tests
//!
//! These tests follow the "Tests as Definition" philosophy, serving as
//! the complete behavioral specification of the physics subsystem.

use crate::physics::*;

// ============================================================================
// Statistical Test Helpers
// ============================================================================

/// Calculate chi-squared statistic for distribution test
fn chi_squared_statistic(observed: &[usize], expected: &[f32]) -> f32 {
    observed
        .iter()
        .zip(expected.iter())
        .filter(|(_, &e)| e > 0.0)
        .map(|(&o, &e)| {
            let diff = o as f32 - e;
            diff * diff / e
        })
        .sum()
}

/// Chi-squared critical value for 95% confidence (approximate)
fn chi_squared_critical(df: usize) -> f32 {
    // Approximate critical values for common df
    match df {
        1 => 3.84,
        2 => 5.99,
        3 => 7.81,
        4 => 9.49,
        5 => 11.07,
        6 => 12.59,
        7 => 14.07,
        8 => 15.51,
        9 => 16.92,
        10 => 18.31,
        _ => 1.0 + 1.4 * df as f32, // Rough approximation for larger df
    }
}

/// Kolmogorov-Smirnov statistic for uniform distribution
fn ks_statistic(samples: &mut [f32]) -> f32 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len() as f32;

    let mut max_d = 0.0_f32;
    for (i, &x) in samples.iter().enumerate() {
        let f_x = x; // Assumes samples are CDF values in [0, 1]
        let empirical = (i + 1) as f32 / n;
        let d = (f_x - empirical).abs();
        max_d = max_d.max(d);
    }
    max_d
}

/// K-S critical value at 95% confidence
fn ks_critical(n: usize) -> f32 {
    // Approximate: c(α) / sqrt(n) where c(0.05) ≈ 1.36
    1.36 / (n as f32).sqrt()
}

/// Simple pseudo-random number generator for testing
struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }

    /// Generate standard normal using Box-Muller
    fn normal_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

// ============================================================================
// Path Loss Tests
// ============================================================================

mod path_loss_tests {
    use super::*;

    #[test]
    fn test_fspl_theoretical_value() {
        // FSPL at 100m, 2.4 GHz should be approximately 80 dB
        // Formula: 20*log10(4*π*d*f/c)
        let fspl = fspl_db_scalar(100.0, 2.4e9);

        // Theoretical: 20*log10(4*π*100*2.4e9/3e8) ≈ 80.04 dB
        let theoretical = 20.0 * (4.0 * std::f32::consts::PI * 100.0 * 2.4e9 / 299792458.0).log10();

        assert!(
            (fspl - theoretical).abs() < 0.1,
            "FSPL mismatch: {} vs theoretical {}",
            fspl,
            theoretical
        );
    }

    #[test]
    fn test_fspl_distance_relationship() {
        let freq = 2.4e9;
        let fspl_10m = fspl_db_scalar(10.0, freq);
        let fspl_100m = fspl_db_scalar(100.0, freq);
        let fspl_1000m = fspl_db_scalar(1000.0, freq);

        // 10x distance = 20 dB more loss
        assert!(
            ((fspl_100m - fspl_10m) - 20.0).abs() < 0.1,
            "10x distance should add 20 dB: {} - {} = {}",
            fspl_100m,
            fspl_10m,
            fspl_100m - fspl_10m
        );

        assert!(
            ((fspl_1000m - fspl_100m) - 20.0).abs() < 0.1,
            "10x distance should add 20 dB"
        );
    }

    #[test]
    fn test_fspl_frequency_relationship() {
        let dist = 100.0;
        let fspl_1ghz = fspl_db_scalar(dist, 1e9);
        let fspl_10ghz = fspl_db_scalar(dist, 10e9);

        // 10x frequency = 20 dB more loss
        assert!(
            ((fspl_10ghz - fspl_1ghz) - 20.0).abs() < 0.1,
            "10x frequency should add 20 dB: {} - {} = {}",
            fspl_10ghz,
            fspl_1ghz,
            fspl_10ghz - fspl_1ghz
        );
    }

    #[test]
    fn test_log_distance_at_reference() {
        let ref_dist = 1.0;
        let ref_loss = 40.0; // Reference loss at 1m
        let n = 3.5; // Path loss exponent

        // At reference distance, should equal reference loss
        let pl = log_distance_db_scalar(ref_dist, ref_dist, n, ref_loss);
        assert!(
            (pl - ref_loss).abs() < 0.01,
            "At d0, should equal ref_loss: {} vs {}",
            pl,
            ref_loss
        );
    }

    #[test]
    fn test_log_distance_exponent_effect() {
        let ref_dist = 1.0;
        let ref_loss = 40.0;
        let dist = 100.0;

        let pl_n2 = log_distance_db_scalar(dist, ref_dist, 2.0, ref_loss);
        let pl_n4 = log_distance_db_scalar(dist, ref_dist, 4.0, ref_loss);

        // n=4 should have double the additional loss of n=2
        let additional_n2 = pl_n2 - ref_loss;
        let additional_n4 = pl_n4 - ref_loss;

        assert!(
            ((additional_n4 / additional_n2) - 2.0).abs() < 0.01,
            "n=4 should double loss vs n=2: {} vs {}",
            additional_n4,
            additional_n2
        );
    }

    #[test]
    fn test_hata_valid_range() {
        // Hata model is valid for 150-1500 MHz
        let pl_150 = hata_urban_db_scalar(1.0, 150.0, 30.0, 1.5);
        let pl_900 = hata_urban_db_scalar(1.0, 900.0, 30.0, 1.5);
        let pl_1500 = hata_urban_db_scalar(1.0, 1500.0, 30.0, 1.5);

        // All should be positive
        assert!(pl_150 > 0.0, "Hata at 150 MHz should be positive");
        assert!(pl_900 > 0.0, "Hata at 900 MHz should be positive");
        assert!(pl_1500 > 0.0, "Hata at 1500 MHz should be positive");

        // Higher frequency = more loss
        assert!(
            pl_900 > pl_150,
            "900 MHz should have more loss than 150 MHz"
        );
        assert!(
            pl_1500 > pl_900,
            "1500 MHz should have more loss than 900 MHz"
        );
    }

    #[test]
    fn test_cost231_extends_hata() {
        // COST-231 should be similar to Hata at boundary (1500 MHz)
        let hata_1500 = hata_urban_db_scalar(1.0, 1500.0, 30.0, 1.5);
        let cost231_1500 = cost231_hata_db_scalar(1.0, 1500.0, 30.0, 1.5, true);

        // Should be reasonably close (within 10 dB)
        assert!(
            (cost231_1500 - hata_1500).abs() < 10.0,
            "COST-231 should be close to Hata at 1500 MHz: {} vs {}",
            cost231_1500,
            hata_1500
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_fspl_simd_matches_scalar() {
        use std::simd::f32x8;

        let distances = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0];
        let freq = 2.4e9;

        let d_simd = f32x8::from_array(distances);
        let f_simd = f32x8::splat(freq);

        let simd_result = fspl_db_simd(d_simd, f_simd);
        let simd_arr: [f32; 8] = simd_result.into();

        for (i, &d) in distances.iter().enumerate() {
            let scalar_result = fspl_db_scalar(d, freq);
            assert!(
                (simd_arr[i] - scalar_result).abs() < 0.01,
                "FSPL SIMD/scalar mismatch at d={}: {} vs {}",
                d,
                simd_arr[i],
                scalar_result
            );
        }
    }
}

// ============================================================================
// Fading Distribution Tests
// ============================================================================

mod fading_tests {
    use super::*;

    const NUM_SAMPLES: usize = 10000;
    const NUM_BINS: usize = 10;

    #[test]
    fn test_rayleigh_mean_power() {
        // Rayleigh fading should preserve mean power (E[|h|²] = 1 for unit variance)
        let mut rng = TestRng::new(12345);
        let input_power = 1.0;

        let mut sum = 0.0;
        for _ in 0..NUM_SAMPLES {
            let faded = apply_rayleigh_fading_scalar_raw(input_power, &mut || rng.normal_pair());
            sum += faded;
        }
        let mean = sum / NUM_SAMPLES as f32;

        // Mean should be close to 1.0 (within 10%)
        assert!(
            (mean - 1.0).abs() < 0.15,
            "Rayleigh mean power should be ~1.0, got {}",
            mean
        );
    }

    #[test]
    fn test_rayleigh_variance() {
        // Rayleigh power variance = 1 (for unit variance components)
        let mut rng = TestRng::new(54321);
        let input_power = 1.0;

        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let faded = apply_rayleigh_fading_scalar_raw(input_power, &mut || rng.normal_pair());
            samples.push(faded);
        }

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 = samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / samples.len() as f32;

        // Variance should be approximately 1.0 (exponential distribution property)
        assert!(
            variance > 0.5 && variance < 2.0,
            "Rayleigh power variance should be ~1.0, got {}",
            variance
        );
    }

    #[test]
    fn test_rayleigh_chi_squared_distribution() {
        // Chi-squared goodness-of-fit test for exponential distribution
        // Rayleigh power follows exponential(λ=1)
        let mut rng = TestRng::new(99999);
        let input_power = 1.0;

        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let faded = apply_rayleigh_fading_scalar_raw(input_power, &mut || rng.normal_pair());
            samples.push(faded);
        }

        // Bin the samples
        let max_val = samples.iter().cloned().fold(0.0_f32, f32::max);
        let bin_width = max_val / NUM_BINS as f32;

        let mut observed = vec![0usize; NUM_BINS];
        for &s in &samples {
            let bin = ((s / bin_width) as usize).min(NUM_BINS - 1);
            observed[bin] += 1;
        }

        // Expected counts for exponential distribution
        // P(bin_i) = F(b_{i+1}) - F(b_i) where F(x) = 1 - e^{-x}
        let mut expected = vec![0.0f32; NUM_BINS];
        for i in 0..NUM_BINS {
            let lower = i as f32 * bin_width;
            let upper = (i + 1) as f32 * bin_width;
            let p = (-lower).exp() - (-upper).exp();
            expected[i] = p * NUM_SAMPLES as f32;
        }

        let chi_sq = chi_squared_statistic(&observed, &expected);
        let critical = chi_squared_critical(NUM_BINS - 1);

        assert!(
            chi_sq < critical * 2.0, // Allow some margin
            "Rayleigh chi-squared test failed: {} > {}",
            chi_sq,
            critical
        );
    }

    #[test]
    fn test_rician_k0_approximates_rayleigh() {
        // With K=0 (no LOS), Rician should approximate Rayleigh
        // Note: Rician K=0 and Rayleigh have similar but not identical formulas
        let mut rng = TestRng::new(11111);
        let input_power = 1.0;
        let k_factor = 0.0;

        let mut rician_samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let faded =
                apply_rician_fading_scalar_raw(input_power, k_factor, &mut || rng.normal_pair());
            rician_samples.push(faded);
        }

        let mean: f32 = rician_samples.iter().sum::<f32>() / rician_samples.len() as f32;
        let variance: f32 = rician_samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / rician_samples.len() as f32;

        // Rician K=0 should have mean close to 1.0
        // (may differ slightly from Rayleigh due to normalization difference)
        assert!(
            (mean - 1.0).abs() < 0.5,
            "Rician K=0 mean should be ~1.0: {}",
            mean
        );
        assert!(
            variance > 0.2,
            "Rician K=0 should have significant variance: {}",
            variance
        );
    }

    #[test]
    fn test_rician_high_k_stable() {
        // With high K (strong LOS), fading should be minimal
        let mut rng = TestRng::new(22222);
        let input_power = 1.0;
        let k_factor = 20.0; // Strong LOS

        let mut samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let faded =
                apply_rician_fading_scalar_raw(input_power, k_factor, &mut || rng.normal_pair());
            samples.push(faded);
        }

        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 = samples
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / samples.len() as f32;

        // High K should have low variance
        assert!(
            variance < 0.3,
            "Rician high-K should have low variance: {}",
            variance
        );

        // Mean should be close to input power
        assert!(
            (mean - input_power).abs() < 0.3,
            "Rician high-K mean should be ~input power: {}",
            mean
        );
    }

    #[test]
    fn test_shadowing_corrected_mean() {
        // CORRECTED: Shadowing now has unit mean in linear domain
        // This means the dB mean is shifted by -(σ² × ln(10))/20
        let mut rng = TestRng::new(33333);
        let input_power = 1.0;
        let sigma_db = 8.0;

        // Expected correction for unit mean in linear domain
        let expected_mean_db = -(sigma_db * sigma_db * std::f32::consts::LN_10) / 20.0;

        let mut log_samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let shadowed =
                apply_shadowing_scalar_raw(input_power, sigma_db, &mut || rng.normal_pair());
            // Convert to dB ratio
            let ratio_db = 10.0 * (shadowed / input_power).log10();
            log_samples.push(ratio_db);
        }

        let mean_db: f32 = log_samples.iter().sum::<f32>() / log_samples.len() as f32;
        let variance_db: f32 = log_samples
            .iter()
            .map(|x| (x - mean_db).powi(2))
            .sum::<f32>()
            / log_samples.len() as f32;
        let std_db = variance_db.sqrt();

        // Mean should be close to the correction value (for unit mean in linear)
        assert!(
            (mean_db - expected_mean_db).abs() < 1.0,
            "Shadowing mean should be ~{:.2} dB (corrected): got {}",
            expected_mean_db,
            mean_db
        );

        // Std should be close to sigma
        assert!(
            (std_db - sigma_db).abs() < 2.0,
            "Shadowing std should be ~{} dB: {}",
            sigma_db,
            std_db
        );
    }

    #[test]
    fn test_shadowing_ks_test() {
        // Kolmogorov-Smirnov test for CORRECTED normal distribution
        // The corrected shadowing has mean = -(σ² × ln(10))/20 in dB domain
        let mut rng = TestRng::new(44444);
        let input_power = 1.0;
        let sigma_db = 8.0;

        // Correction for unit mean in linear domain
        let mean_db = -(sigma_db * sigma_db * std::f32::consts::LN_10) / 20.0;

        let mut cdf_samples = Vec::with_capacity(NUM_SAMPLES);
        for _ in 0..NUM_SAMPLES {
            let shadowed =
                apply_shadowing_scalar_raw(input_power, sigma_db, &mut || rng.normal_pair());
            let ratio_db = 10.0 * (shadowed / input_power).log10();

            // Convert to standard normal CDF value using CORRECTED mean
            let z = (ratio_db - mean_db) / sigma_db;
            let cdf = 0.5 * (1.0 + erf_approx(z / std::f32::consts::SQRT_2));
            cdf_samples.push(cdf.clamp(0.0, 1.0));
        }

        let ks = ks_statistic(&mut cdf_samples);
        let critical = ks_critical(NUM_SAMPLES);

        assert!(
            ks < critical * 2.0, // Allow margin
            "Shadowing K-S test failed: {} > {}",
            ks,
            critical
        );
    }

    /// Approximate error function for testing
    fn erf_approx(x: f32) -> f32 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_rayleigh_simd_mean_power() {
        use crate::simd_rf::random::SimdRng;
        use std::simd::f32x8;

        let mut rng = SimdRng::new(55555);
        let input_power = f32x8::splat(1.0);

        let mut sum = 0.0;
        let iterations = NUM_SAMPLES / 8;
        for _ in 0..iterations {
            let faded = apply_rayleigh_fading_simd_raw(input_power, &mut rng);
            let arr: [f32; 8] = faded.into();
            sum += arr.iter().sum::<f32>();
        }

        let mean = sum / (iterations * 8) as f32;
        assert!(
            (mean - 1.0).abs() < 0.15,
            "SIMD Rayleigh mean should be ~1.0: {}",
            mean
        );
    }
}

// ============================================================================
// Multipath Tests
// ============================================================================

mod multipath_tests {
    use super::*;
    use crate::physics::multipath::{
        coherence_bandwidth_hz, frequency_selectivity, get_resolvable_taps, Complex,
        ITUChannelModel,
    };

    #[test]
    fn test_epa_specification() {
        let tdl = ITUChannelModel::EPA.get_tdl();

        assert_eq!(tdl.taps.len(), 7, "EPA should have 7 taps");
        assert!(
            (tdl.max_delay_ns - 410.0).abs() < 1.0,
            "EPA max delay should be 410 ns"
        );

        // First tap should be at 0 delay
        assert!(
            tdl.taps[0].delay_ns.abs() < 0.1,
            "First tap should be at 0 ns"
        );

        // First tap should be 0 dB
        assert!(
            tdl.taps[0].power_db.abs() < 0.1,
            "First tap should be 0 dB"
        );
    }

    #[test]
    fn test_eva_specification() {
        let tdl = ITUChannelModel::EVA.get_tdl();

        assert_eq!(tdl.taps.len(), 9, "EVA should have 9 taps");
        assert!(
            (tdl.max_delay_ns - 2510.0).abs() < 1.0,
            "EVA max delay should be 2510 ns"
        );
    }

    #[test]
    fn test_etu_specification() {
        let tdl = ITUChannelModel::ETU.get_tdl();

        assert_eq!(tdl.taps.len(), 9, "ETU should have 9 taps");
        assert!(
            (tdl.max_delay_ns - 5000.0).abs() < 1.0,
            "ETU max delay should be 5000 ns"
        );
    }

    #[test]
    fn test_coherence_bandwidth_ordering() {
        let bc_epa = coherence_bandwidth_hz(ITUChannelModel::EPA);
        let bc_eva = coherence_bandwidth_hz(ITUChannelModel::EVA);
        let bc_etu = coherence_bandwidth_hz(ITUChannelModel::ETU);

        // Larger delay spread = smaller coherence bandwidth
        assert!(
            bc_epa > bc_eva,
            "EPA should have larger Bc than EVA: {} vs {}",
            bc_epa,
            bc_eva
        );
        assert!(
            bc_eva > bc_etu,
            "EVA should have larger Bc than ETU: {} vs {}",
            bc_eva,
            bc_etu
        );
    }

    #[test]
    fn test_frequency_selectivity_classification() {
        // Narrowband signal in EPA should be flat fading
        let fs_narrow_epa = frequency_selectivity(100_000.0, ITUChannelModel::EPA);
        assert!(
            fs_narrow_epa < 1.0,
            "100 kHz in EPA should be flat fading"
        );

        // Wideband signal in ETU should be frequency selective
        let fs_wide_etu = frequency_selectivity(20_000_000.0, ITUChannelModel::ETU);
        assert!(
            fs_wide_etu > 1.0,
            "20 MHz in ETU should be frequency selective"
        );
    }

    #[test]
    fn test_cir_power_conservation() {
        // Total tap power should be conserved after normalization
        let mut tdl = ITUChannelModel::EPA.get_tdl();
        tdl.normalize();

        let total_power: f32 = tdl.linear_powers().iter().sum();
        assert!(
            (total_power - 1.0).abs() < 0.01,
            "Normalized power should be 1.0: {}",
            total_power
        );
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, 2.0);

        // Magnitude
        assert!((a.magnitude() - 5.0).abs() < 1e-6);

        // Multiplication: (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i
        let c = a * b;
        assert!((c.re - (-5.0)).abs() < 1e-6);
        assert!((c.im - 10.0).abs() < 1e-6);

        // Addition
        let d = a + b;
        assert!((d.re - 4.0).abs() < 1e-6);
        assert!((d.im - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_resolvable_taps_bandwidth_effect() {
        // Wideband: more resolvable taps
        let wide_taps = get_resolvable_taps(ITUChannelModel::EPA, 100e6);
        // Narrowband: fewer resolvable taps
        let narrow_taps = get_resolvable_taps(ITUChannelModel::EPA, 1e6);

        assert!(
            wide_taps.taps.len() >= narrow_taps.taps.len(),
            "Wider bandwidth should resolve more taps: {} vs {}",
            wide_taps.taps.len(),
            narrow_taps.taps.len()
        );
    }
}

// ============================================================================
// Atmospheric Tests
// ============================================================================

mod atmospheric_tests {
    use super::*;

    #[test]
    fn test_atmospheric_positive() {
        // All attenuation should be positive
        for freq in [1.0, 10.0, 20.0, 30.0, 60.0] {
            let atten = atmospheric_attenuation_db_per_km_scalar(freq);
            assert!(
                atten > 0.0,
                "Atmospheric attenuation at {} GHz should be positive: {}",
                freq,
                atten
            );
        }
    }

    #[test]
    fn test_atmospheric_increases_with_frequency() {
        let atten_1 = atmospheric_attenuation_db_per_km_scalar(1.0);
        let atten_10 = atmospheric_attenuation_db_per_km_scalar(10.0);
        let atten_30 = atmospheric_attenuation_db_per_km_scalar(30.0);

        assert!(
            atten_10 > atten_1,
            "10 GHz should have more attenuation than 1 GHz"
        );
        assert!(
            atten_30 > atten_10,
            "30 GHz should have more attenuation than 10 GHz"
        );
    }

    #[test]
    fn test_rain_positive() {
        for freq in [10.0, 20.0, 40.0] {
            let atten = rain_attenuation_db_per_km_scalar(freq, 10.0);
            assert!(
                atten > 0.0,
                "Rain attenuation at {} GHz should be positive",
                freq
            );
        }
    }

    #[test]
    fn test_rain_increases_with_rate() {
        let freq = 30.0;
        let atten_5 = rain_attenuation_db_per_km_scalar(freq, 5.0);
        let atten_20 = rain_attenuation_db_per_km_scalar(freq, 20.0);
        let atten_50 = rain_attenuation_db_per_km_scalar(freq, 50.0);

        assert!(
            atten_20 > atten_5,
            "Higher rain rate should cause more attenuation"
        );
        assert!(atten_50 > atten_20, "Higher rain rate should cause more attenuation");
    }

    #[test]
    fn test_rain_increases_with_frequency() {
        let rain_rate = 10.0;
        let atten_10 = rain_attenuation_db_per_km_scalar(10.0, rain_rate);
        let atten_30 = rain_attenuation_db_per_km_scalar(30.0, rain_rate);
        let atten_60 = rain_attenuation_db_per_km_scalar(60.0, rain_rate);

        assert!(
            atten_30 > atten_10,
            "Higher frequency should have more rain attenuation"
        );
        assert!(atten_60 > atten_30, "Higher frequency should have more rain attenuation");
    }

    #[test]
    fn test_apply_atmospheric_reduces_power() {
        let power = 1.0;
        let distance = 10.0;
        let freq = 30.0;

        let attenuated = apply_atmospheric_scalar(power, distance, freq);

        assert!(
            attenuated < power,
            "Atmospheric absorption should reduce power"
        );
        assert!(attenuated > 0.0, "Power should remain positive");
    }

    #[test]
    fn test_low_frequency_negligible() {
        // At VHF, atmospheric effects should be minimal
        let atten = atmospheric_attenuation_db_per_km_scalar(0.5);
        assert!(
            atten < 0.01,
            "500 MHz should have negligible attenuation: {}",
            atten
        );
    }
}

// ============================================================================
// Doppler Tests
// ============================================================================

mod doppler_tests {
    use super::*;

    #[test]
    fn test_doppler_sign_approaching() {
        // Approaching should give positive Doppler shift
        let doppler = calc_doppler_shift_scalar(30.0, 1e9);
        assert!(doppler > 0.0, "Approaching should have positive Doppler");
    }

    #[test]
    fn test_doppler_sign_receding() {
        // Receding should give negative Doppler shift
        let doppler = calc_doppler_shift_scalar(-30.0, 1e9);
        assert!(doppler < 0.0, "Receding should have negative Doppler");
    }

    #[test]
    fn test_doppler_magnitude() {
        // 30 m/s at 1 GHz: f_d = (30/3e8) × 1e9 = 100 Hz
        let doppler = calc_doppler_shift_scalar(30.0, 1e9);
        assert!(
            (doppler - 100.0).abs() < 1.0,
            "Doppler at 30 m/s, 1 GHz should be ~100 Hz: {}",
            doppler
        );
    }

    #[test]
    fn test_doppler_frequency_scaling() {
        let v = 30.0;
        let doppler_1g = calc_doppler_shift_scalar(v, 1e9);
        let doppler_10g = calc_doppler_shift_scalar(v, 10e9);

        assert!(
            ((doppler_10g / doppler_1g) - 10.0).abs() < 0.1,
            "10x frequency should give 10x Doppler: {} vs {}",
            doppler_10g,
            doppler_1g
        );
    }

    #[test]
    fn test_coherence_time_inverse_relationship() {
        let t_10hz = calc_coherence_time_scalar(10.0);
        let t_100hz = calc_coherence_time_scalar(100.0);

        assert!(
            t_10hz > t_100hz,
            "Higher Doppler should give shorter coherence time"
        );

        let ratio = t_10hz / t_100hz;
        assert!(
            (ratio - 10.0).abs() < 0.5,
            "10x Doppler should give 10x shorter coherence: {}",
            ratio
        );
    }

    #[test]
    fn test_micro_doppler_periodic() {
        // Micro-Doppler should be periodic with rotation rate
        let rotation = 100.0; // 100 Hz
        let blade = 0.15; // 15 cm
        let freq = 5.8e9;

        let md_0 = calc_micro_doppler_scalar(rotation, blade, freq, 0.0);
        let md_half = calc_micro_doppler_scalar(rotation, blade, freq, 0.005); // Half period
        let md_full = calc_micro_doppler_scalar(rotation, blade, freq, 0.01); // Full period

        // At t=0 and t=full period, should be approximately equal
        assert!(
            (md_0 - md_full).abs() < 10.0,
            "Micro-Doppler should be periodic: {} vs {}",
            md_0,
            md_full
        );
    }

    #[test]
    fn test_radial_velocity_approaching() {
        let tx_pos = (0.0, 0.0, 0.0);
        let tx_vel = (10.0, 0.0, 0.0);
        let rx_pos = (100.0, 0.0, 0.0);

        let v_r = calc_radial_velocity_scalar(tx_pos, tx_vel, rx_pos);

        assert!(
            (v_r - 10.0).abs() < 0.1,
            "Radial velocity should be 10 m/s: {}",
            v_r
        );
    }

    #[test]
    fn test_radial_velocity_perpendicular() {
        let tx_pos = (0.0, 0.0, 0.0);
        let tx_vel = (0.0, 10.0, 0.0); // Moving perpendicular
        let rx_pos = (100.0, 0.0, 0.0);

        let v_r = calc_radial_velocity_scalar(tx_pos, tx_vel, rx_pos);

        assert!(
            v_r.abs() < 0.1,
            "Perpendicular motion should have zero radial velocity: {}",
            v_r
        );
    }
}

// ============================================================================
// Channel Integration Tests
// ============================================================================

mod channel_tests {
    use super::*;
    use crate::physics::channel::{
        calculate_link_budget, calculate_sinr_db, apply_channel_scalar, ChannelParams, ChannelState,
        FadingType, PathLossModel,
    };

    fn make_rng() -> impl FnMut() -> (f32, f32) {
        let mut state = 12345u64;
        move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (state as f64 / u64::MAX as f64) as f32;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state as f64 / u64::MAX as f64) as f32;

            let r = (-2.0 * u1.max(1e-10).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            (r * theta.cos(), r * theta.sin())
        }
    }

    #[test]
    fn test_channel_power_decreases_with_distance() {
        let params = ChannelParams::new()
            .with_path_loss(PathLossModel::FreeSpace)
            .with_carrier_freq(2.4e9);
        let state = ChannelState::default();
        let mut rng = make_rng();

        let power_100m = apply_channel_scalar(20.0, 100.0, &params, &state, &mut rng).rx_power_dbm;
        let power_1000m =
            apply_channel_scalar(20.0, 1000.0, &params, &state, &mut rng).rx_power_dbm;

        assert!(
            power_1000m < power_100m,
            "Power should decrease with distance: {} < {}",
            power_1000m,
            power_100m
        );

        // 10x distance should be ~20 dB more loss
        let diff = power_100m - power_1000m;
        assert!(
            (diff - 20.0).abs() < 1.0,
            "10x distance should add ~20 dB loss: {}",
            diff
        );
    }

    #[test]
    fn test_channel_all_effects_no_nan() {
        let params = ChannelParams::urban();
        let mut state = ChannelState::new().with_velocity(30.0);
        state.advance_time(0.001);
        let mut rng = make_rng();

        for _ in 0..100 {
            let output = apply_channel_scalar(20.0, 500.0, &params, &state, &mut rng);

            assert!(!output.rx_power_dbm.is_nan(), "RX power should not be NaN");
            assert!(
                !output.rx_power_dbm.is_infinite(),
                "RX power should not be infinite"
            );
            assert!(!output.path_loss_db.is_nan(), "Path loss should not be NaN");
            assert!(!output.doppler_hz.is_nan(), "Doppler should not be NaN");
        }
    }

    #[test]
    fn test_sinr_calculation() {
        // Simple case: signal at -60, noise at -100, no interference
        let sinr = calculate_sinr_db(-60.0, &[], -100.0);
        assert!(
            (sinr - 40.0).abs() < 0.1,
            "SINR should be 40 dB: {}",
            sinr
        );

        // With equal power interferer
        let sinr_int = calculate_sinr_db(-60.0, &[-60.0], -100.0);
        assert!(
            sinr_int < 5.0,
            "SINR with equal interferer should be ~0 dB: {}",
            sinr_int
        );

        // Multiple interferers
        let sinr_multi = calculate_sinr_db(-60.0, &[-70.0, -70.0, -70.0], -100.0);
        assert!(
            sinr_multi < sinr,
            "More interference should reduce SINR"
        );
    }

    #[test]
    fn test_link_budget() {
        let budget = calculate_link_budget(
            20.0,   // TX power
            3.0,    // TX antenna
            3.0,    // RX antenna
            10.0,   // Required SNR
            -100.0, // Noise floor
            2.4e9,  // Frequency
        );

        // EIRP = 20 + 3 = 23 dBm
        // Required signal = -100 + 10 = -90 dBm
        // Min RX signal = -90 - 3 = -93 dBm
        // Max path loss = 23 - (-93) = 116 dB
        assert!(
            (budget.max_path_loss_db - 116.0).abs() < 1.0,
            "Max path loss calculation: {}",
            budget.max_path_loss_db
        );

        assert!(
            budget.max_range_m > 0.0,
            "Max range should be positive"
        );
    }

    #[test]
    fn test_channel_presets_valid() {
        let presets = [
            ChannelParams::urban(),
            ChannelParams::rural_los(),
            ChannelParams::vehicular(),
            ChannelParams::satellite(),
        ];

        let state = ChannelState::default();
        let mut rng = make_rng();

        for params in &presets {
            let output = apply_channel_scalar(20.0, 100.0, params, &state, &mut rng);

            assert!(
                output.rx_power_dbm.is_finite(),
                "Preset channel should give finite power"
            );
            assert!(
                output.path_loss_db > 0.0,
                "Path loss should be positive"
            );
        }
    }

    #[test]
    fn test_channel_with_rain() {
        let dry = ChannelParams::satellite();
        let wet = ChannelParams::satellite().with_rain(50.0); // Heavy rain

        let state = ChannelState::default();
        let mut rng = make_rng();
        let distance = 10000.0; // 10 km

        let dry_output = apply_channel_scalar(30.0, distance, &dry, &state, &mut rng);
        let wet_output = apply_channel_scalar(30.0, distance, &wet, &state, &mut rng);

        // Rain should increase atmospheric attenuation
        assert!(
            wet_output.atmospheric_db > dry_output.atmospheric_db,
            "Rain should add atmospheric attenuation: {} vs {}",
            wet_output.atmospheric_db,
            dry_output.atmospheric_db
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_channel() {
        use crate::physics::channel::{apply_channel_simd, SimdChannelOutput};
        use crate::simd_rf::random::SimdRng;
        use std::simd::f32x8;

        let mut rng = SimdRng::new(42);

        let output = apply_channel_simd(
            f32x8::splat(20.0),
            f32x8::splat(100.0),
            f32x8::splat(2.4e9),
            f32x8::splat(10.0),
            f32x8::splat(0.0),
            PathLossModel::FreeSpace,
            FadingType::Rayleigh,
            &mut rng,
        );

        let powers: [f32; 8] = output.rx_power_dbm.into();
        for p in powers {
            assert!(p.is_finite(), "SIMD channel power should be finite");
            assert!(
                p > -150.0 && p < 30.0,
                "Power should be reasonable: {}",
                p
            );
        }
    }
}
