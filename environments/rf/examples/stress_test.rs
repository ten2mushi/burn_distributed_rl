//! Stress test / benchmark for RF environment throughput.
//!
//! Measures steps per second (SPS) under high load and detects
//! any performance degradation over time.
//!
//! Run with: cargo run --example stress_test --release -p rf_environment
//!
//! The RF environment requires SIMD, which is enabled by default.

use rf_environment::{
    adapter::{AgentView, RFEnvAdapter, VectorizedEnv},
    config::EntityConfig,
    RFWorldConfig,
};
use std::time::{Duration, Instant};

/// Configuration for the stress test
struct StressTestConfig {
    /// Number of environments to run in parallel
    num_envs: usize,
    /// Total number of steps to run
    total_steps: u64,
    /// Number of steps per measurement window
    window_size: u64,
    /// Number of warmup steps before measuring
    warmup_steps: u64,
    /// Number of frequency bins
    num_freq_bins: usize,
    /// Number of jammers per environment
    num_jammers: usize,
    /// Number of cognitive radios per environment
    num_crs: usize,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            num_envs: 1024,
            total_steps: 500_000,
            window_size: 10_000,
            warmup_steps: 10_000,
            num_freq_bins: 256,
            num_jammers: 1,
            num_crs: 1,
        }
    }
}

/// Statistics for a measurement window
#[derive(Debug, Clone)]
struct WindowStats {
    window_idx: usize,
    steps: u64,
    duration_ms: f64,
    sps: f64,
    env_steps_per_sec: f64, // SPS * num_envs
}

fn separator(c: char, n: usize) {
    println!("{}", c.to_string().repeat(n));
}

/// Create RF environment with given parameters
fn create_env(num_envs: usize, num_freq_bins: usize, num_jammers: usize, num_crs: usize) -> RFEnvAdapter {
    // Minimal entity config for stress testing
    let entity_config = EntityConfig::new()
        .with_tv_stations(2)
        .with_fm_radios(1)
        .with_lte_towers(2)
        .with_wifi_aps(4)
        .with_bluetooth(2)
        .with_sband_radars(0)
        .with_weather_radars(0)
        .with_drones(0, 0)
        .with_vehicles(0);

    let config = RFWorldConfig::new()
        .with_num_envs(num_envs)
        .with_freq_bins(num_freq_bins)
        .with_freq_range(400e6, 6e9)
        .with_world_size(1000.0, 1000.0, 100.0)
        .with_max_entities(16)
        .with_physics_freq(1000)
        .with_ctrl_freq(100)
        .with_max_steps(10_000)
        .with_seed(42)
        .with_entity_config(entity_config)
        .with_num_jammers(num_jammers)
        .with_num_crs(num_crs)
        .build();

    RFEnvAdapter::new(config, AgentView::Jammer(0))
}

/// Run the stress test and collect statistics
fn run_stress_test(config: &StressTestConfig) -> Vec<WindowStats> {
    separator('=', 70);
    println!("RF ENVIRONMENT STRESS TEST");
    separator('=', 70);
    println!();

    println!("Mode: SIMD (f32x8) - required for RF environment");
    println!("Environments: {}", config.num_envs);
    println!("Frequency bins: {}", config.num_freq_bins);
    println!("Jammers: {}, CRs: {}", config.num_jammers, config.num_crs);
    println!("Total steps: {}", config.total_steps);
    println!("Window size: {} steps", config.window_size);
    println!("Warmup: {} steps", config.warmup_steps);
    println!();

    // Create environment
    let mut env = create_env(
        config.num_envs,
        config.num_freq_bins,
        config.num_jammers,
        config.num_crs,
    );
    env.reset_all(42);

    println!(
        "Environment created: {} envs x {} obs_size = {} floats per step",
        env.n_envs(),
        env.obs_size(),
        env.n_envs() * env.obs_size()
    );
    println!("Action space: {} discrete actions", env.n_actions());
    println!();

    // Prepare actions (random discrete actions)
    let mut actions: Vec<f32> = (0..config.num_envs)
        .map(|i| (i % env.n_actions()) as f32)
        .collect();

    // Add deterministic variation
    for (i, action) in actions.iter_mut().enumerate() {
        *action = ((i * 17 + 31) % env.n_actions()) as f32;
    }

    // Warmup phase
    println!("Warming up ({} steps)...", config.warmup_steps);
    let warmup_start = Instant::now();
    for _ in 0..config.warmup_steps {
        let result = env.step(&actions);
        // Auto-reset done environments
        if result.dones().iter().any(|&d| d) {
            let mask = rf_environment::adapter::ResetMask::from_dones(&result.dones());
            env.reset_envs(&mask, 42);
        }
    }
    let warmup_duration = warmup_start.elapsed();
    let warmup_sps = config.warmup_steps as f64 / warmup_duration.as_secs_f64();
    println!(
        "Warmup complete: {:.0} steps/sec ({:.2e} env-steps/sec)",
        warmup_sps,
        warmup_sps * config.num_envs as f64
    );
    println!();

    // Main measurement phase
    println!("Running stress test...");
    separator('-', 70);
    println!(
        "{:>8} {:>12} {:>12} {:>15} {:>12}",
        "Window", "Steps", "Time (ms)", "Steps/sec", "Env-steps/s"
    );
    separator('-', 70);

    let mut stats = Vec::new();
    let mut total_steps_done: u64 = 0;
    let mut window_idx: usize = 0;
    let overall_start = Instant::now();

    while total_steps_done < config.total_steps {
        let steps_this_window = config.window_size.min(config.total_steps - total_steps_done);

        let window_start = Instant::now();
        for _ in 0..steps_this_window {
            let result = env.step(&actions);
            // Auto-reset done environments
            if result.dones().iter().any(|&d| d) {
                let mask = rf_environment::adapter::ResetMask::from_dones(&result.dones());
                env.reset_envs(&mask, 42);
            }
        }
        let window_duration = window_start.elapsed();

        let duration_ms = window_duration.as_secs_f64() * 1000.0;
        let sps = steps_this_window as f64 / window_duration.as_secs_f64();
        let env_sps = sps * config.num_envs as f64;

        let window_stat = WindowStats {
            window_idx,
            steps: steps_this_window,
            duration_ms,
            sps,
            env_steps_per_sec: env_sps,
        };

        println!(
            "{:>8} {:>12} {:>12.1} {:>15.0} {:>12.2e}",
            window_idx,
            steps_this_window,
            duration_ms,
            sps,
            env_sps
        );

        stats.push(window_stat);
        total_steps_done += steps_this_window;
        window_idx += 1;
    }

    let overall_duration = overall_start.elapsed();

    separator('-', 70);
    println!();

    // Print summary statistics
    print_summary(&stats, overall_duration, config);

    // Check for degradation
    check_degradation(&stats);

    stats
}

fn print_summary(stats: &[WindowStats], overall_duration: Duration, config: &StressTestConfig) {
    println!("SUMMARY");
    separator('=', 70);

    let total_steps: u64 = stats.iter().map(|s| s.steps).sum();
    let overall_sps = total_steps as f64 / overall_duration.as_secs_f64();
    let overall_env_sps = overall_sps * config.num_envs as f64;

    let sps_values: Vec<f64> = stats.iter().map(|s| s.sps).collect();
    let min_sps = sps_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_sps = sps_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean_sps: f64 = sps_values.iter().sum::<f64>() / sps_values.len() as f64;

    let variance: f64 = sps_values
        .iter()
        .map(|&sps| (sps - mean_sps).powi(2))
        .sum::<f64>()
        / sps_values.len() as f64;
    let std_sps = variance.sqrt();
    let cv = (std_sps / mean_sps) * 100.0; // Coefficient of variation

    println!("Total steps:        {:>15}", total_steps);
    println!(
        "Total time:         {:>15.2} sec",
        overall_duration.as_secs_f64()
    );
    println!("Environments:       {:>15}", config.num_envs);
    println!("Frequency bins:     {:>15}", config.num_freq_bins);
    println!();
    println!("Overall throughput:");
    println!("  Steps/sec:        {:>15.0}", overall_sps);
    println!("  Env-steps/sec:    {:>15.2e}", overall_env_sps);
    println!();
    println!("Per-window statistics (steps/sec):");
    println!("  Min:              {:>15.0}", min_sps);
    println!("  Max:              {:>15.0}", max_sps);
    println!("  Mean:             {:>15.0}", mean_sps);
    println!("  Std Dev:          {:>15.0}", std_sps);
    println!("  CV (variability): {:>14.1}%", cv);
    println!();
}

fn check_degradation(stats: &[WindowStats]) {
    println!("DEGRADATION ANALYSIS");
    separator('=', 70);

    if stats.len() < 4 {
        println!("Not enough windows for degradation analysis.");
        return;
    }

    // Compare first quarter vs last quarter
    let quarter = stats.len() / 4;
    let first_quarter: Vec<f64> = stats[..quarter].iter().map(|s| s.sps).collect();
    let last_quarter: Vec<f64> = stats[stats.len() - quarter..].iter().map(|s| s.sps).collect();

    let first_mean: f64 = first_quarter.iter().sum::<f64>() / first_quarter.len() as f64;
    let last_mean: f64 = last_quarter.iter().sum::<f64>() / last_quarter.len() as f64;

    let degradation_pct = ((first_mean - last_mean) / first_mean) * 100.0;

    println!("First quarter mean: {:>15.0} steps/sec", first_mean);
    println!("Last quarter mean:  {:>15.0} steps/sec", last_mean);
    println!("Change:             {:>14.1}%", -degradation_pct);
    println!();

    // Linear regression to detect trend
    let n = stats.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f64 = stats.iter().map(|s| s.sps).sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, stat) in stats.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (stat.sps - y_mean);
        denominator += (x - x_mean).powi(2);
    }
    let slope = numerator / denominator;
    let slope_pct_per_window = (slope / y_mean) * 100.0;

    println!("Linear trend:       {:>14.2}% per window", slope_pct_per_window);

    // Verdict
    println!();
    if degradation_pct.abs() < 5.0 && slope_pct_per_window.abs() < 0.5 {
        println!(">>> NO SIGNIFICANT DEGRADATION DETECTED");
        println!("    Performance is stable over time.");
    } else if degradation_pct > 0.0 {
        println!("!!! PERFORMANCE DEGRADATION DETECTED");
        println!(
            "    Throughput decreased by {:.1}% over the test.",
            degradation_pct
        );
        if slope_pct_per_window < -0.5 {
            println!("    There is a consistent downward trend.");
        }
    } else {
        println!(">>> Performance improved or remained stable.");
        println!("    (This may indicate warmup effects in early windows)");
    }
    println!();
}

/// Run scaling test across different environment counts
fn run_scaling_test() {
    println!();
    separator('=', 70);
    println!("SCALING TEST");
    separator('=', 70);
    println!();
    println!("Testing throughput at different environment counts...");
    println!();

    // Must be multiples of 8 for SIMD
    let env_counts = [8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let steps_per_test = 10_000u64;
    let warmup = 2_000u64;

    println!(
        "{:>10} {:>15} {:>15} {:>12}",
        "Envs", "Steps/sec", "Env-steps/sec", "Per-env (us)"
    );
    separator('-', 55);

    for &num_envs in &env_counts {
        let mut env = create_env(num_envs, 256, 1, 1);
        env.reset_all(42);

        let actions: Vec<f32> = (0..num_envs).map(|i| (i % 100) as f32).collect();

        // Warmup
        for _ in 0..warmup {
            let result = env.step(&actions);
            if result.dones().iter().any(|&d| d) {
                let mask = rf_environment::adapter::ResetMask::from_dones(&result.dones());
                env.reset_envs(&mask, 42);
            }
        }

        // Measure
        let start = Instant::now();
        for _ in 0..steps_per_test {
            let result = env.step(&actions);
            if result.dones().iter().any(|&d| d) {
                let mask = rf_environment::adapter::ResetMask::from_dones(&result.dones());
                env.reset_envs(&mask, 42);
            }
        }
        let duration = start.elapsed();

        let sps = steps_per_test as f64 / duration.as_secs_f64();
        let env_sps = sps * num_envs as f64;
        let us_per_env = (duration.as_micros() as f64) / (steps_per_test as f64 * num_envs as f64);

        println!(
            "{:>10} {:>15.0} {:>15.2e} {:>12.3}",
            num_envs, sps, env_sps, us_per_env
        );
    }

    println!();
}

/// Run frequency bin scaling test
fn run_freq_bin_scaling_test() {
    println!();
    separator('=', 70);
    println!("FREQUENCY BIN SCALING TEST");
    separator('=', 70);
    println!();
    println!("Testing throughput at different frequency resolutions...");
    println!();

    let freq_bin_counts = [64, 128, 256, 512, 1024];
    let num_envs = 256;
    let steps_per_test = 5_000u64;
    let warmup = 1_000u64;

    println!(
        "{:>12} {:>15} {:>15} {:>12}",
        "Freq Bins", "Steps/sec", "Env-steps/sec", "Per-bin (ns)"
    );
    separator('-', 57);

    for &num_bins in &freq_bin_counts {
        let mut env = create_env(num_envs, num_bins, 1, 1);
        env.reset_all(42);

        let actions: Vec<f32> = (0..num_envs).map(|i| (i % 100) as f32).collect();

        // Warmup
        for _ in 0..warmup {
            let result = env.step(&actions);
            if result.dones().iter().any(|&d| d) {
                let mask = rf_environment::adapter::ResetMask::from_dones(&result.dones());
                env.reset_envs(&mask, 42);
            }
        }

        // Measure
        let start = Instant::now();
        for _ in 0..steps_per_test {
            let result = env.step(&actions);
            if result.dones().iter().any(|&d| d) {
                let mask = rf_environment::adapter::ResetMask::from_dones(&result.dones());
                env.reset_envs(&mask, 42);
            }
        }
        let duration = start.elapsed();

        let sps = steps_per_test as f64 / duration.as_secs_f64();
        let env_sps = sps * num_envs as f64;
        let ns_per_bin =
            (duration.as_nanos() as f64) / (steps_per_test as f64 * num_envs as f64 * num_bins as f64);

        println!(
            "{:>12} {:>15.0} {:>15.2e} {:>12.1}",
            num_bins, sps, env_sps, ns_per_bin
        );
    }

    println!();
}

fn main() {
    // Run main stress test
    let config = StressTestConfig {
        num_envs: 512,
        total_steps: 100_000,
        window_size: 5_000,
        warmup_steps: 5_000,
        num_freq_bins: 256,
        num_jammers: 1,
        num_crs: 1,
    };

    run_stress_test(&config);

    // Run scaling tests
    run_scaling_test();
    run_freq_bin_scaling_test();

    separator('=', 70);
    println!("STRESS TEST COMPLETE");
    separator('=', 70);
}
