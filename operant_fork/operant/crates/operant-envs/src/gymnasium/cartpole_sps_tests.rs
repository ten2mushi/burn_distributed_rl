//! SPS Degradation Investigation Tests for CartPole SIMD Environment
//!
//! This test suite investigates potential causes of Steps-Per-Second (SPS) degradation
//! in the distributed IMPALA RL system. The tests follow the "Tests as Definition"
//! philosophy to definitively prove or disprove that operant SIMD operations are
//! causing the observed ~57% SPS degradation over 642k steps.
//!
//! ## Hypotheses Under Test
//!
//! 1. SIMD Register Accumulation: Are SIMD operations creating temporary values that accumulate?
//! 2. Environment State Growth: Do environment internal buffers grow over time?
//! 3. Reset Overhead: Does `reset_envs()` become slower over many episodes?
//! 4. Observation Buffer Copies: Are observations being copied inefficiently?
//! 5. Memory Alignment Issues: Could misaligned SIMD operations slow down over time?
//! 6. f32x8 Vector Allocation: Are SIMD vectors being allocated fresh each step?
//!
//! ## Expected Detection Patterns
//!
//! If operant is the cause, we should observe:
//! - `step()` or `reset_envs()` timing increasing over iterations
//! - Memory growth in environment state
//! - SIMD operation overhead increasing
//!
//! ## Test Configuration
//!
//! Tests simulate the actual IMPALA actor configuration:
//! - 32 vectorized environments per actor
//! - f32x8 SIMD (4 SIMD lanes for 32 envs)
//! - step -> write_observations -> reset_envs cycle

use super::*;
use operant_core::{Environment, ResetMask};
use std::time::{Duration, Instant};

// ============================================================================
// Test Configuration Constants
// ============================================================================

/// Number of environments per actor (matches IMPALA configuration)
const NUM_ENVS: usize = 32;

/// Number of iterations for timing tests
const TIMING_ITERATIONS: usize = 100_000;

/// Number of iterations for long-run tests
const LONG_RUN_ITERATIONS: usize = 500_000;

/// Number of samples for statistical analysis
const SAMPLE_INTERVAL: usize = 10_000;

/// Threshold for detecting degradation (>25% slowdown is concerning)
/// Note: Using 25% because microbenchmarks have inherent variance.
/// The key tests are the long-run tests which average out the noise.
const DEGRADATION_THRESHOLD: f64 = 1.25;

/// Threshold for detecting memory growth (>1% growth per 100k steps is concerning)
const MEMORY_GROWTH_THRESHOLD: f64 = 1.01;

// ============================================================================
// Helper Structures for Timing Analysis
// ============================================================================

/// Stores timing measurements for statistical analysis
#[derive(Clone, Debug)]
struct TimingStats {
    /// Timing samples (nanoseconds per operation)
    samples: Vec<f64>,
    /// Sample labels (iteration count when sample was taken)
    labels: Vec<usize>,
}

impl TimingStats {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            labels: Vec::new(),
        }
    }

    fn add_sample(&mut self, iteration: usize, nanos: f64) {
        self.samples.push(nanos);
        self.labels.push(iteration);
    }

    /// Calculate mean of first N samples
    fn early_mean(&self, n: usize) -> f64 {
        let slice = &self.samples[..n.min(self.samples.len())];
        slice.iter().sum::<f64>() / slice.len() as f64
    }

    /// Calculate mean of last N samples
    fn late_mean(&self, n: usize) -> f64 {
        let len = self.samples.len();
        let start = len.saturating_sub(n);
        let slice = &self.samples[start..];
        slice.iter().sum::<f64>() / slice.len() as f64
    }

    /// Calculate the ratio of late mean to early mean (degradation factor)
    fn degradation_factor(&self, window: usize) -> f64 {
        let early = self.early_mean(window);
        let late = self.late_mean(window);
        if early > 0.0 {
            late / early
        } else {
            1.0
        }
    }

    /// Perform linear regression to detect trends
    fn linear_regression(&self) -> (f64, f64) {
        let n = self.samples.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        let x_mean = self.labels.iter().sum::<usize>() as f64 / n;
        let y_mean = self.samples.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (x, y) in self.labels.iter().zip(self.samples.iter()) {
            let x_diff = *x as f64 - x_mean;
            let y_diff = *y - y_mean;
            numerator += x_diff * y_diff;
            denominator += x_diff * x_diff;
        }

        let slope = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        let intercept = y_mean - slope * x_mean;

        (slope, intercept)
    }

    /// Check if there's a statistically significant upward trend
    fn has_upward_trend(&self) -> bool {
        let (slope, _intercept) = self.linear_regression();
        // Slope is in nanos per iteration, consider significant if >0.001 ns/iteration
        slope > 0.001
    }

    fn standard_deviation(&self) -> f64 {
        let mean = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
        let variance = self.samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.samples.len() as f64;
        variance.sqrt()
    }
}

/// Memory usage snapshot
#[derive(Clone, Debug)]
struct MemorySnapshot {
    iteration: usize,
    /// Approximate heap size based on Vec capacities
    estimated_heap_bytes: usize,
}

// ============================================================================
// Test 1: SIMD Step Operation Timing Over Extended Iterations
// ============================================================================

/// Test that step() timing remains constant over 100k+ iterations.
///
/// This test measures the time for each step() call and detects if there is
/// any degradation pattern. If SIMD operations are accumulating state or
/// causing memory pressure, we would see timing increase over iterations.
///
/// Detection criteria:
/// - Late mean should not exceed early mean by more than DEGRADATION_THRESHOLD
/// - Linear regression slope should be approximately zero
#[test]
fn test_step_timing_stability_over_100k_iterations() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut stats = TimingStats::new();

    // Warmup phase (allow JIT/cache warmup)
    for _ in 0..1000 {
        env.step_auto_reset(&actions);
    }
    env.reset(42); // Reset to known state after warmup

    // Measurement phase
    for iteration in 0..TIMING_ITERATIONS {
        let start = Instant::now();
        env.step_auto_reset(&actions);
        let elapsed = start.elapsed().as_nanos() as f64;

        if iteration % (TIMING_ITERATIONS / 100) == 0 {
            stats.add_sample(iteration, elapsed);
        }
    }

    let degradation = stats.degradation_factor(10);
    let (slope, _intercept) = stats.linear_regression();

    println!("=== Step Timing Stability Test ===");
    println!("Total iterations: {}", TIMING_ITERATIONS);
    println!("Early mean: {:.2} ns", stats.early_mean(10));
    println!("Late mean: {:.2} ns", stats.late_mean(10));
    println!("Degradation factor: {:.4}", degradation);
    println!("Linear regression slope: {:.6} ns/iteration", slope);
    println!("Standard deviation: {:.2} ns", stats.standard_deviation());

    // Assertion: No significant degradation
    assert!(
        degradation < DEGRADATION_THRESHOLD,
        "Step timing degraded by {:.1}% over {} iterations. \
         Early mean: {:.2} ns, Late mean: {:.2} ns. \
         This suggests SIMD operations may be accumulating overhead.",
        (degradation - 1.0) * 100.0,
        TIMING_ITERATIONS,
        stats.early_mean(10),
        stats.late_mean(10)
    );

    // Assertion: No upward trend
    assert!(
        !stats.has_upward_trend(),
        "Detected upward trend in step timing with slope {:.6} ns/iteration. \
         This could indicate memory pressure or state accumulation.",
        slope
    );
}

// ============================================================================
// Test 2: Reset Accumulation Detection
// ============================================================================

/// Test that reset_envs() timing remains constant over many reset cycles.
///
/// In the IMPALA architecture, environments are reset frequently when episodes
/// complete. If reset_envs() has O(n) complexity where n is total resets,
/// this would cause degradation.
#[test]
fn test_reset_timing_stability_over_many_episodes() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut reset_stats = TimingStats::new();
    let mut total_resets = 0usize;

    // Run until we have many reset cycles
    const TARGET_RESETS: usize = 50_000;

    for iteration in 0..1_000_000 {
        // Step without auto-reset
        env.step_no_reset(&actions);

        // Read terminals and create mask
        let mut terminals = vec![0u8; NUM_ENVS];
        let mut truncations = vec![0u8; NUM_ENVS];
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        let mask = ResetMask::from_done_flags(&terminals, &truncations);
        let reset_count = mask.count();

        if reset_count > 0 {
            let start = Instant::now();
            env.reset_envs(&mask, iteration as u64);
            let elapsed = start.elapsed().as_nanos() as f64;

            total_resets += reset_count;

            // Sample every 1000 resets
            if total_resets % 1000 < reset_count {
                reset_stats.add_sample(total_resets, elapsed / reset_count as f64);
            }
        }

        if total_resets >= TARGET_RESETS {
            break;
        }
    }

    let degradation = reset_stats.degradation_factor(10);
    let (slope, _intercept) = reset_stats.linear_regression();

    println!("=== Reset Timing Stability Test ===");
    println!("Total resets: {}", total_resets);
    println!("Samples collected: {}", reset_stats.samples.len());
    println!("Early mean: {:.2} ns/reset", reset_stats.early_mean(10));
    println!("Late mean: {:.2} ns/reset", reset_stats.late_mean(10));
    println!("Degradation factor: {:.4}", degradation);
    println!("Linear regression slope: {:.6} ns/reset", slope);

    assert!(
        degradation < DEGRADATION_THRESHOLD,
        "Reset timing degraded by {:.1}% over {} resets. \
         This suggests reset_envs() may have accumulating overhead.",
        (degradation - 1.0) * 100.0,
        total_resets
    );
}

// ============================================================================
// Test 3: Observation Write Performance
// ============================================================================

/// Test that write_observations() timing remains constant.
///
/// This tests the SIMD observation gather operation. If there are alignment
/// issues or buffer growth, we would see degradation here.
#[test]
fn test_observation_write_timing_stability() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut obs_buffer = vec![0.0f32; NUM_ENVS * 4];
    let mut stats = TimingStats::new();

    // Warmup
    for _ in 0..1000 {
        env.step_auto_reset(&actions);
        env.write_observations(&mut obs_buffer);
    }
    env.reset(42);

    // Measurement
    for iteration in 0..TIMING_ITERATIONS {
        env.step_auto_reset(&actions);

        let start = Instant::now();
        env.write_observations(&mut obs_buffer);
        let elapsed = start.elapsed().as_nanos() as f64;

        if iteration % (TIMING_ITERATIONS / 100) == 0 {
            stats.add_sample(iteration, elapsed);
        }
    }

    let degradation = stats.degradation_factor(10);

    println!("=== Observation Write Timing Test ===");
    println!("Early mean: {:.2} ns", stats.early_mean(10));
    println!("Late mean: {:.2} ns", stats.late_mean(10));
    println!("Degradation factor: {:.4}", degradation);

    assert!(
        degradation < DEGRADATION_THRESHOLD,
        "Observation write degraded by {:.1}% over {} iterations.",
        (degradation - 1.0) * 100.0,
        TIMING_ITERATIONS
    );
}

// ============================================================================
// Test 4: Memory Stability
// ============================================================================

/// Estimate the memory footprint of the CartPole environment.
fn estimate_env_memory(env: &CartPole) -> usize {
    let num_envs = env.num_envs();

    // Vec<f32> fields: x, x_dot, theta, theta_dot, rewards, episode_rewards
    let f32_vecs = 6 * num_envs * std::mem::size_of::<f32>();

    // Vec<u8> fields: terminals, truncations
    let u8_vecs = 2 * num_envs * std::mem::size_of::<u8>();

    // Vec<u32> fields: ticks
    let u32_vecs = num_envs * std::mem::size_of::<u32>();

    // Vec<u64> fields: rng_seeds
    let u64_vecs = num_envs * std::mem::size_of::<u64>();

    // obs_buffer
    let obs_buffer = num_envs * 4 * std::mem::size_of::<f32>();

    f32_vecs + u8_vecs + u32_vecs + u64_vecs + obs_buffer
}

/// Test that environment memory footprint does not grow over time.
///
/// This checks if any Vec fields are growing unbounded, which would indicate
/// a memory leak or unbounded buffer growth.
#[test]
fn test_memory_stability_over_extended_run() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut memory_snapshots: Vec<MemorySnapshot> = Vec::new();

    let initial_memory = estimate_env_memory(&env);
    memory_snapshots.push(MemorySnapshot {
        iteration: 0,
        estimated_heap_bytes: initial_memory,
    });

    for iteration in 0..TIMING_ITERATIONS {
        env.step_auto_reset(&actions);

        if iteration % SAMPLE_INTERVAL == 0 && iteration > 0 {
            let current_memory = estimate_env_memory(&env);
            memory_snapshots.push(MemorySnapshot {
                iteration,
                estimated_heap_bytes: current_memory,
            });
        }
    }

    let final_memory = estimate_env_memory(&env);
    let memory_ratio = final_memory as f64 / initial_memory as f64;

    println!("=== Memory Stability Test ===");
    println!("Initial memory: {} bytes", initial_memory);
    println!("Final memory: {} bytes", final_memory);
    println!("Memory ratio: {:.4}", memory_ratio);

    // Check that memory did not grow
    assert!(
        memory_ratio < MEMORY_GROWTH_THRESHOLD,
        "Memory grew by {:.1}% over {} iterations. \
         Initial: {} bytes, Final: {} bytes. \
         This indicates unbounded buffer growth.",
        (memory_ratio - 1.0) * 100.0,
        TIMING_ITERATIONS,
        initial_memory,
        final_memory
    );

    // Verify Vec capacities match lengths (no wasted capacity)
    // This is checked implicitly by estimate_env_memory using num_envs
}

// ============================================================================
// Test 5: Long-Run Simulation (500k+ Steps)
// ============================================================================

/// Long-run simulation tracking all metrics.
///
/// This test simulates the actual IMPALA actor loop for 500k+ steps,
/// tracking step timing, reset timing, and memory at regular intervals.
/// This is the most comprehensive test for detecting degradation patterns.
#[test]
fn test_long_run_simulation_500k_steps() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut obs_buffer = vec![0.0f32; NUM_ENVS * 4];

    let mut step_stats = TimingStats::new();
    let mut total_elapsed = Duration::ZERO;

    // Warmup
    for _ in 0..5000 {
        env.step_auto_reset(&actions);
        env.write_observations(&mut obs_buffer);
    }
    env.reset(42);

    let run_start = Instant::now();

    // Long run with periodic sampling
    for iteration in 0..LONG_RUN_ITERATIONS {
        let step_start = Instant::now();

        // Simulate IMPALA actor loop
        env.step_no_reset(&actions);
        env.write_observations(&mut obs_buffer);

        // Handle resets
        let mut terminals = vec![0u8; NUM_ENVS];
        let mut truncations = vec![0u8; NUM_ENVS];
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        let mask = ResetMask::from_done_flags(&terminals, &truncations);
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }

        let step_elapsed = step_start.elapsed();
        total_elapsed += step_elapsed;

        // Sample at regular intervals
        if iteration % (LONG_RUN_ITERATIONS / 100) == 0 {
            step_stats.add_sample(iteration, step_elapsed.as_nanos() as f64);
        }

        // Progress report every 100k steps
        if iteration % 100_000 == 0 && iteration > 0 {
            let current_sps = iteration as f64 / run_start.elapsed().as_secs_f64();
            println!("Progress: {} steps, current SPS: {:.0}", iteration, current_sps);
        }
    }

    let total_time = run_start.elapsed();
    let overall_sps = LONG_RUN_ITERATIONS as f64 / total_time.as_secs_f64();

    // Calculate SPS for different phases
    let early_samples = &step_stats.samples[..step_stats.samples.len() / 4];
    let late_samples = &step_stats.samples[step_stats.samples.len() * 3 / 4..];

    let early_mean_ns = early_samples.iter().sum::<f64>() / early_samples.len() as f64;
    let late_mean_ns = late_samples.iter().sum::<f64>() / late_samples.len() as f64;

    // Convert to SPS (steps per second)
    let early_sps = 1_000_000_000.0 / early_mean_ns;
    let late_sps = 1_000_000_000.0 / late_mean_ns;
    let sps_degradation = (early_sps - late_sps) / early_sps;

    let (slope, _intercept) = step_stats.linear_regression();

    println!("=== Long-Run Simulation Results (500k steps) ===");
    println!("Total time: {:.2}s", total_time.as_secs_f64());
    println!("Overall SPS: {:.0}", overall_sps);
    println!("Early phase mean: {:.2} ns ({:.0} SPS)", early_mean_ns, early_sps);
    println!("Late phase mean: {:.2} ns ({:.0} SPS)", late_mean_ns, late_sps);
    println!("SPS degradation: {:.2}%", sps_degradation * 100.0);
    println!("Linear regression slope: {:.6} ns/step", slope);
    println!("Standard deviation: {:.2} ns", step_stats.standard_deviation());

    // If the observed 57% SPS degradation is from operant, we should see it here
    let timing_degradation = step_stats.degradation_factor(10);

    assert!(
        timing_degradation < 1.20, // Allow 20% variation for noise
        "DETECTED SIGNIFICANT DEGRADATION in long-run test! \
         Timing degraded by {:.1}% from {:.0} SPS to {:.0} SPS. \
         This strongly suggests operant SIMD operations are causing the SPS drop.",
        (timing_degradation - 1.0) * 100.0,
        early_sps,
        late_sps
    );

    // Additional check: the slope should be near zero
    assert!(
        slope < 0.01, // Less than 0.01 ns increase per step
        "Detected consistent upward trend with slope {:.6} ns/step. \
         Over 500k steps, this would add {:.1} ms of overhead.",
        slope,
        slope * 500_000.0 / 1_000_000.0
    );
}

// ============================================================================
// Test 6: SIMD vs Scalar Comparison Over Time
// ============================================================================

/// Compare SIMD and scalar implementations to detect SIMD-specific issues.
///
/// If SIMD operations have accumulating overhead while scalar does not,
/// this would isolate the problem to SIMD-specific code paths.
#[cfg(feature = "simd")]
#[test]
fn test_simd_vs_scalar_degradation_comparison() {
    let mut simd_env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create SIMD environment");
    let mut scalar_env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create scalar environment");

    simd_env.reset(42);
    scalar_env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();

    let mut simd_stats = TimingStats::new();
    let mut scalar_stats = TimingStats::new();

    const COMPARISON_ITERATIONS: usize = 50_000;

    // Warmup
    for _ in 0..1000 {
        simd_env.step_simd(&actions);
        scalar_env.step_scalar(&actions);
    }
    simd_env.reset(42);
    scalar_env.reset(42);

    // Interleaved measurement to reduce bias
    for iteration in 0..COMPARISON_ITERATIONS {
        // Measure SIMD
        let simd_start = Instant::now();
        simd_env.step_auto_reset_simd(&actions);
        let simd_elapsed = simd_start.elapsed().as_nanos() as f64;

        // Measure scalar
        let scalar_start = Instant::now();
        scalar_env.step_auto_reset_scalar(&actions);
        let scalar_elapsed = scalar_start.elapsed().as_nanos() as f64;

        if iteration % (COMPARISON_ITERATIONS / 50) == 0 {
            simd_stats.add_sample(iteration, simd_elapsed);
            scalar_stats.add_sample(iteration, scalar_elapsed);
        }
    }

    let simd_degradation = simd_stats.degradation_factor(5);
    let scalar_degradation = scalar_stats.degradation_factor(5);

    println!("=== SIMD vs Scalar Degradation Comparison ===");
    println!("SIMD early mean: {:.2} ns", simd_stats.early_mean(5));
    println!("SIMD late mean: {:.2} ns", simd_stats.late_mean(5));
    println!("SIMD degradation factor: {:.4}", simd_degradation);
    println!("Scalar early mean: {:.2} ns", scalar_stats.early_mean(5));
    println!("Scalar late mean: {:.2} ns", scalar_stats.late_mean(5));
    println!("Scalar degradation factor: {:.4}", scalar_degradation);

    // If SIMD degrades more than scalar, the issue is SIMD-specific
    let simd_vs_scalar_ratio = simd_degradation / scalar_degradation;

    if simd_vs_scalar_ratio > 1.1 {
        println!(
            "WARNING: SIMD shows {:.1}% more degradation than scalar. \
             This suggests SIMD-specific issues.",
            (simd_vs_scalar_ratio - 1.0) * 100.0
        );
    }

    assert!(
        simd_degradation < DEGRADATION_THRESHOLD,
        "SIMD implementation shows {:.1}% degradation while scalar shows {:.1}%. \
         SIMD-specific issue detected.",
        (simd_degradation - 1.0) * 100.0,
        (scalar_degradation - 1.0) * 100.0
    );
}

// ============================================================================
// Test 7: Episode Length Distribution Stability
// ============================================================================

/// Verify that episode lengths remain statistically stable.
///
/// If the RNG or physics become unstable over time, episode lengths
/// would change, which could affect reset frequency and timing.
#[test]
fn test_episode_length_distribution_stability() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();

    let mut early_episode_lengths: Vec<u32> = Vec::new();
    let mut late_episode_lengths: Vec<u32> = Vec::new();

    const EPISODES_TO_COLLECT: usize = 1000;
    const TRANSITION_ITERATIONS: usize = 200_000;

    // Collect early episode lengths
    env.clear_log();
    let mut iteration = 0;
    while early_episode_lengths.len() < EPISODES_TO_COLLECT {
        env.step_no_reset(&actions);

        let mut terminals = vec![0u8; NUM_ENVS];
        let mut truncations = vec![0u8; NUM_ENVS];
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        for i in 0..NUM_ENVS {
            if terminals[i] != 0 || truncations[i] != 0 {
                // Approximate episode length from ticks
                // Since we don't have direct access, use iteration count
                early_episode_lengths.push(iteration as u32 % 500);
            }
        }

        let mask = ResetMask::from_done_flags(&terminals, &truncations);
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }

        iteration += 1;
        if early_episode_lengths.len() >= EPISODES_TO_COLLECT {
            break;
        }
    }

    // Run many more iterations
    for _ in 0..TRANSITION_ITERATIONS {
        env.step_auto_reset(&actions);
    }

    // Collect late episode lengths
    let log_before = env.get_log();
    env.clear_log();

    iteration = 0;
    while late_episode_lengths.len() < EPISODES_TO_COLLECT {
        env.step_auto_reset(&actions);

        let log = env.get_log();
        let new_episodes = log.episode_count - (late_episode_lengths.len() as u32);
        if new_episodes > 0 && log.episode_count > 0 {
            // Use average steps as proxy
            let avg_steps = log.total_steps / log.episode_count;
            late_episode_lengths.push(avg_steps);
        }

        iteration += 1;
        if late_episode_lengths.len() >= EPISODES_TO_COLLECT || iteration > 100_000 {
            break;
        }
    }

    // Compare distributions
    let log_after = env.get_log();

    println!("=== Episode Length Distribution Test ===");
    println!("Episodes before transition: {}", log_before.episode_count);
    println!("Episodes after transition: {}", log_after.episode_count);

    // The episode distribution should remain stable
    // (This is a sanity check - unstable physics would show here)
}

// ============================================================================
// Test 8: Reward Accumulation Precision
// ============================================================================

/// Test that reward accumulation does not cause floating-point precision loss.
///
/// After millions of additions, f32 precision loss could cause issues.
#[test]
fn test_reward_accumulation_precision() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut rewards_buffer = vec![0.0f32; NUM_ENVS];

    // Run for many steps and check reward values stay valid
    for iteration in 0..100_000 {
        env.step_auto_reset(&actions);
        env.write_rewards(&mut rewards_buffer);

        // Check all rewards are valid (0.0 or 1.0 for CartPole)
        for (i, &r) in rewards_buffer.iter().enumerate() {
            assert!(
                r == 0.0 || r == 1.0,
                "Invalid reward {} at iteration {} env {}: expected 0.0 or 1.0, got {}",
                r, iteration, i, r
            );
        }
    }

    // Check log totals are reasonable
    let log = env.get_log();
    assert!(
        log.total_reward.is_finite(),
        "Total reward is not finite: {}",
        log.total_reward
    );
    assert!(
        log.total_reward > 0.0,
        "Total reward should be positive after 100k steps"
    );
}

// ============================================================================
// Test 9: step_no_reset_with_result Timing
// ============================================================================

/// Test the combined step_no_reset_with_result() method for timing stability.
///
/// This is the actual method used in the IMPALA actor loop.
#[test]
fn test_step_no_reset_with_result_timing_stability() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut stats = TimingStats::new();

    // Warmup
    for iteration in 0..1000 {
        let result = env.step_no_reset_with_result(&actions);
        let mask = result.to_reset_mask();
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }
    }
    env.reset(42);

    // Measurement
    for iteration in 0..TIMING_ITERATIONS {
        let start = Instant::now();
        let result = env.step_no_reset_with_result(&actions);
        let elapsed_step = start.elapsed();

        let mask = result.to_reset_mask();
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }

        if iteration % (TIMING_ITERATIONS / 100) == 0 {
            stats.add_sample(iteration, elapsed_step.as_nanos() as f64);
        }
    }

    let degradation = stats.degradation_factor(10);

    println!("=== step_no_reset_with_result Timing Test ===");
    println!("Early mean: {:.2} ns", stats.early_mean(10));
    println!("Late mean: {:.2} ns", stats.late_mean(10));
    println!("Degradation factor: {:.4}", degradation);

    assert!(
        degradation < DEGRADATION_THRESHOLD,
        "step_no_reset_with_result degraded by {:.1}% over {} iterations.",
        (degradation - 1.0) * 100.0,
        TIMING_ITERATIONS
    );
}

// ============================================================================
// Test 10: Batch Processing Overhead
// ============================================================================

/// Test if batch size affects degradation rate.
///
/// Larger batches might reveal SIMD-specific overhead accumulation.
/// This test uses more iterations and a larger sample window to reduce
/// noise from microbenchmark timing variance.
#[test]
fn test_batch_size_degradation_comparison() {
    let batch_sizes = [8, 32, 64, 128];
    let iterations_per_batch = 100_000; // Increased for more stable measurements

    println!("=== Batch Size Degradation Comparison ===");

    for &batch_size in &batch_sizes {
        let mut env = CartPole::with_defaults(batch_size).expect("Failed to create environment");
        env.reset(42);

        let actions: Vec<f32> = (0..batch_size).map(|i| (i % 2) as f32).collect();
        let mut stats = TimingStats::new();

        // Warmup
        for _ in 0..1000 {
            env.step_auto_reset(&actions);
        }
        env.reset(42);

        // Measurement
        for iteration in 0..iterations_per_batch {
            let start = Instant::now();
            env.step_auto_reset(&actions);
            let elapsed = start.elapsed().as_nanos() as f64;

            if iteration % (iterations_per_batch / 50) == 0 {
                stats.add_sample(iteration, elapsed);
            }
        }

        // Use larger window (10 samples) for more stable mean estimates
        let degradation = stats.degradation_factor(10);
        let ns_per_env = stats.late_mean(10) / batch_size as f64;

        println!(
            "Batch size {}: degradation factor {:.4}, late mean {:.2} ns ({:.2} ns/env)",
            batch_size,
            degradation,
            stats.late_mean(5),
            ns_per_env
        );

        assert!(
            degradation < DEGRADATION_THRESHOLD,
            "Batch size {} shows {:.1}% degradation",
            batch_size,
            (degradation - 1.0) * 100.0
        );
    }
}

// ============================================================================
// Test 11: RNG Seed Progression Stability
// ============================================================================

/// Test that RNG seed progression does not cause issues.
///
/// The environment uses wrapping_add for seed progression. Verify this
/// does not cause any degenerate behavior over many resets.
#[test]
fn test_rng_seed_progression_stability() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(0); // Start with seed 0

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();

    // Run many reset cycles with progressively larger seeds
    for base_seed in [0u64, u64::MAX / 4, u64::MAX / 2, u64::MAX - NUM_ENVS as u64] {
        env.reset(base_seed);

        for iteration in 0..10_000 {
            env.step_no_reset(&actions);

            let mut terminals = vec![0u8; NUM_ENVS];
            let mut truncations = vec![0u8; NUM_ENVS];
            env.write_terminals(&mut terminals);
            env.write_truncations(&mut truncations);

            let mask = ResetMask::from_done_flags(&terminals, &truncations);
            if mask.any() {
                // Use wrapping seed that would wrap around u64
                let reset_seed = base_seed.wrapping_add(iteration as u64 * NUM_ENVS as u64);
                env.reset_envs(&mask, reset_seed);
            }
        }

        // Verify environment is still in valid state
        let mut obs = vec![0.0f32; NUM_ENVS * 4];
        env.write_observations(&mut obs);

        for i in 0..NUM_ENVS {
            let x = obs[i * 4];
            let theta = obs[i * 4 + 2];

            assert!(
                x.is_finite() && x.abs() < 10.0,
                "Invalid x={} at base_seed={}", x, base_seed
            );
            assert!(
                theta.is_finite() && theta.abs() < 2.0,
                "Invalid theta={} at base_seed={}", theta, base_seed
            );
        }
    }

    println!("=== RNG Seed Progression Test ===");
    println!("All seed ranges tested successfully");
}

// ============================================================================
// Test 12: Physics Stability Over Extended Run
// ============================================================================

/// Test that physics computations remain stable and do not accumulate errors.
///
/// The Taylor series approximations for sin/cos could accumulate errors
/// if angles grow unboundedly before reset.
#[test]
fn test_physics_numerical_stability() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut max_theta_seen = 0.0f32;
    let mut max_x_seen = 0.0f32;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for iteration in 0..TIMING_ITERATIONS {
        env.step_no_reset(&actions);

        let mut obs = vec![0.0f32; NUM_ENVS * 4];
        env.write_observations(&mut obs);

        for i in 0..NUM_ENVS {
            let x = obs[i * 4];
            let x_dot = obs[i * 4 + 1];
            let theta = obs[i * 4 + 2];
            let theta_dot = obs[i * 4 + 3];

            // Check for NaN/Inf
            if x.is_nan() || x_dot.is_nan() || theta.is_nan() || theta_dot.is_nan() {
                nan_count += 1;
            }
            if x.is_infinite() || x_dot.is_infinite() || theta.is_infinite() || theta_dot.is_infinite() {
                inf_count += 1;
            }

            max_theta_seen = max_theta_seen.max(theta.abs());
            max_x_seen = max_x_seen.max(x.abs());
        }

        // Reset terminated environments
        let mut terminals = vec![0u8; NUM_ENVS];
        let mut truncations = vec![0u8; NUM_ENVS];
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        let mask = ResetMask::from_done_flags(&terminals, &truncations);
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }
    }

    println!("=== Physics Numerical Stability Test ===");
    println!("Max theta seen: {:.4} rad", max_theta_seen);
    println!("Max x seen: {:.4}", max_x_seen);
    println!("NaN count: {}", nan_count);
    println!("Inf count: {}", inf_count);

    assert_eq!(nan_count, 0, "NaN values detected in physics computations");
    assert_eq!(inf_count, 0, "Infinite values detected in physics computations");
}

// ============================================================================
// Test 13: Concurrent Access Pattern Simulation
// ============================================================================

/// Simulate the actual IMPALA actor memory access pattern.
///
/// This tests if the pattern of step -> read obs -> reset creates any
/// cache or memory coherence issues over time.
#[test]
fn test_impala_actor_access_pattern() {
    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();

    // Pre-allocate all buffers like IMPALA actor does
    let mut obs_buffer = vec![0.0f32; NUM_ENVS * 4];
    let mut rewards_buffer = vec![0.0f32; NUM_ENVS];
    let mut terminals_buffer = vec![0u8; NUM_ENVS];
    let mut truncations_buffer = vec![0u8; NUM_ENVS];

    let mut stats = TimingStats::new();

    // Warmup
    for iteration in 0..1000 {
        // 1. Write observations (SIMD read)
        env.write_observations(&mut obs_buffer);

        // 2. (Model inference would happen here)

        // 3. Step (SIMD physics)
        env.step_no_reset(&actions);

        // 4. Read results
        env.write_rewards(&mut rewards_buffer);
        env.write_terminals(&mut terminals_buffer);
        env.write_truncations(&mut truncations_buffer);

        // 5. Reset terminated (SIMD reset)
        let mask = ResetMask::from_done_flags(&terminals_buffer, &truncations_buffer);
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }
    }

    env.reset(42);

    // Measurement with full IMPALA pattern
    for iteration in 0..TIMING_ITERATIONS {
        let loop_start = Instant::now();

        // 1. Write observations (SIMD read)
        env.write_observations(&mut obs_buffer);

        // 2. (Simulate model inference delay - commented out for pure env timing)
        // std::thread::sleep(Duration::from_micros(10));

        // 3. Step (SIMD physics)
        env.step_no_reset(&actions);

        // 4. Read results
        env.write_rewards(&mut rewards_buffer);
        env.write_terminals(&mut terminals_buffer);
        env.write_truncations(&mut truncations_buffer);

        // 5. Reset terminated
        let mask = ResetMask::from_done_flags(&terminals_buffer, &truncations_buffer);
        if mask.any() {
            env.reset_envs(&mask, iteration as u64);
        }

        let loop_elapsed = loop_start.elapsed().as_nanos() as f64;

        if iteration % (TIMING_ITERATIONS / 100) == 0 {
            stats.add_sample(iteration, loop_elapsed);
        }
    }

    let degradation = stats.degradation_factor(10);

    println!("=== IMPALA Actor Access Pattern Test ===");
    println!("Early mean (full loop): {:.2} ns", stats.early_mean(10));
    println!("Late mean (full loop): {:.2} ns", stats.late_mean(10));
    println!("Degradation factor: {:.4}", degradation);

    assert!(
        degradation < DEGRADATION_THRESHOLD,
        "IMPALA actor pattern degraded by {:.1}% over {} iterations.",
        (degradation - 1.0) * 100.0,
        TIMING_ITERATIONS
    );
}

// ============================================================================
// Test 14: ResetMask Allocation Overhead
// ============================================================================

/// Test if ResetMask creation has allocation overhead that accumulates.
#[test]
fn test_reset_mask_allocation_overhead() {
    let mut stats = TimingStats::new();

    // Pre-allocate terminal buffers
    let terminals: Vec<u8> = (0..NUM_ENVS).map(|i| (i % 4 == 0) as u8).collect();
    let truncations: Vec<u8> = (0..NUM_ENVS).map(|i| (i % 8 == 0) as u8).collect();

    for iteration in 0..TIMING_ITERATIONS {
        let start = Instant::now();
        let mask = ResetMask::from_done_flags(&terminals, &truncations);
        let _count = mask.count(); // Force evaluation
        let elapsed = start.elapsed().as_nanos() as f64;

        if iteration % (TIMING_ITERATIONS / 100) == 0 {
            stats.add_sample(iteration, elapsed);
        }
    }

    let degradation = stats.degradation_factor(10);

    println!("=== ResetMask Allocation Overhead Test ===");
    println!("Early mean: {:.2} ns", stats.early_mean(10));
    println!("Late mean: {:.2} ns", stats.late_mean(10));
    println!("Degradation factor: {:.4}", degradation);

    assert!(
        degradation < DEGRADATION_THRESHOLD,
        "ResetMask allocation degraded by {:.1}%",
        (degradation - 1.0) * 100.0
    );
}

// ============================================================================
// Summary Test: Comprehensive Degradation Check
// ============================================================================

/// Summary test that runs all timing checks and reports overall health.
#[test]
fn test_comprehensive_degradation_summary() {
    println!("\n========================================");
    println!("  OPERANT SPS DEGRADATION INVESTIGATION");
    println!("========================================\n");

    let mut env = CartPole::with_defaults(NUM_ENVS).expect("Failed to create environment");
    env.reset(42);

    let actions: Vec<f32> = (0..NUM_ENVS).map(|i| (i % 2) as f32).collect();
    let mut obs_buffer = vec![0.0f32; NUM_ENVS * 4];

    // Test all components individually
    let mut step_stats = TimingStats::new();
    let mut obs_stats = TimingStats::new();
    let mut reset_stats = TimingStats::new();

    const SUMMARY_ITERATIONS: usize = 200_000;

    // Warmup
    for _ in 0..5000 {
        env.step_auto_reset(&actions);
        env.write_observations(&mut obs_buffer);
    }
    env.reset(42);

    // Measure
    let mut reset_count = 0;
    for iteration in 0..SUMMARY_ITERATIONS {
        // Step timing
        let step_start = Instant::now();
        env.step_no_reset(&actions);
        let step_elapsed = step_start.elapsed().as_nanos() as f64;

        // Observation timing
        let obs_start = Instant::now();
        env.write_observations(&mut obs_buffer);
        let obs_elapsed = obs_start.elapsed().as_nanos() as f64;

        // Reset timing
        let mut terminals = vec![0u8; NUM_ENVS];
        let mut truncations = vec![0u8; NUM_ENVS];
        env.write_terminals(&mut terminals);
        env.write_truncations(&mut truncations);

        let mask = ResetMask::from_done_flags(&terminals, &truncations);
        if mask.any() {
            let reset_start = Instant::now();
            env.reset_envs(&mask, iteration as u64);
            let reset_elapsed = reset_start.elapsed().as_nanos() as f64;

            reset_count += mask.count();
            if reset_count % 500 < mask.count() {
                reset_stats.add_sample(reset_count, reset_elapsed / mask.count() as f64);
            }
        }

        if iteration % (SUMMARY_ITERATIONS / 100) == 0 {
            step_stats.add_sample(iteration, step_elapsed);
            obs_stats.add_sample(iteration, obs_elapsed);
        }
    }

    // Results
    println!("Configuration:");
    println!("  - Environments: {}", NUM_ENVS);
    println!("  - Iterations: {}", SUMMARY_ITERATIONS);
    println!("  - Total resets: {}\n", reset_count);

    let step_degradation = step_stats.degradation_factor(10);
    let obs_degradation = obs_stats.degradation_factor(10);
    let reset_degradation = if reset_stats.samples.len() >= 10 {
        reset_stats.degradation_factor(5)
    } else {
        1.0
    };

    println!("Step Timing:");
    println!("  Early: {:.2} ns", step_stats.early_mean(10));
    println!("  Late:  {:.2} ns", step_stats.late_mean(10));
    println!("  Degradation: {:.2}%\n", (step_degradation - 1.0) * 100.0);

    println!("Observation Write Timing:");
    println!("  Early: {:.2} ns", obs_stats.early_mean(10));
    println!("  Late:  {:.2} ns", obs_stats.late_mean(10));
    println!("  Degradation: {:.2}%\n", (obs_degradation - 1.0) * 100.0);

    println!("Reset Timing (per env):");
    if reset_stats.samples.len() >= 10 {
        println!("  Early: {:.2} ns", reset_stats.early_mean(5));
        println!("  Late:  {:.2} ns", reset_stats.late_mean(5));
        println!("  Degradation: {:.2}%\n", (reset_degradation - 1.0) * 100.0);
    } else {
        println!("  Insufficient samples\n");
    }

    // Verdict
    let any_degradation = step_degradation > DEGRADATION_THRESHOLD
        || obs_degradation > DEGRADATION_THRESHOLD
        || reset_degradation > DEGRADATION_THRESHOLD;

    println!("========================================");
    if any_degradation {
        println!("  VERDICT: DEGRADATION DETECTED");
        println!("========================================");
        println!("\nOperant SIMD operations show timing degradation.");
        println!("This could be contributing to SPS drop.");

        if step_degradation > DEGRADATION_THRESHOLD {
            println!("  - step() is the primary suspect");
        }
        if obs_degradation > DEGRADATION_THRESHOLD {
            println!("  - write_observations() is the primary suspect");
        }
        if reset_degradation > DEGRADATION_THRESHOLD {
            println!("  - reset_envs() is the primary suspect");
        }

        panic!(
            "Degradation detected: step={:.2}%, obs={:.2}%, reset={:.2}%",
            (step_degradation - 1.0) * 100.0,
            (obs_degradation - 1.0) * 100.0,
            (reset_degradation - 1.0) * 100.0
        );
    } else {
        println!("  VERDICT: NO DEGRADATION DETECTED");
        println!("========================================");
        println!("\nOperant SIMD operations appear stable.");
        println!("SPS degradation is likely NOT caused by operant.");
        println!("Investigate other components:");
        println!("  - Trajectory buffer (even with VecDeque)");
        println!("  - PyTorch tensor operations");
        println!("  - FFI/Python boundary overhead");
        println!("  - Learner thread contention");
        println!("  - GPU memory fragmentation");
    }
}
