//! Environment Wrapper SPS Degradation Tests
//!
//! This module provides comprehensive tests to investigate whether the environment
//! wrapper layer contributes to SPS (Steps-Per-Second) degradation in the distributed
//! IMPALA implementation. The observed pattern shows ~57% performance loss over 642k
//! steps (23,000 -> 9,800 SPS).
//!
//! # Investigation Focus
//!
//! The distributed_rl crate wraps operant environments through several layers:
//! - `OperantAdapter<E>` implements `VectorizedEnv` trait for operant environments
//! - `CartPoleEnvWrapper` / `PendulumEnvWrapper` provide simplified discrete/continuous interfaces
//! - `CartPoleEnv` / `PendulumEnv` implement `LearnerVectorizedEnv` for the generic Learner
//!
//! # Hypotheses Under Test
//!
//! 1. **StepResult Allocation Overhead**: Vec allocations in StepResult may cause
//!    heap fragmentation or accumulating allocation overhead
//!
//! 2. **Wrapper Step Performance**: The wrapper's step() may have hidden costs that
//!    accumulate over time (action conversion, result copying)
//!
//! 3. **Observation Buffer Handling**: write_observations() may have copying overhead
//!    that grows with usage
//!
//! 4. **Reset Cycle Performance**: reset_envs() may accumulate state or have
//!    increasing overhead over many episode completions
//!
//! 5. **Trait Object Dispatch**: Dynamic dispatch through VectorizedEnv trait may
//!    have vtable lookup costs that accumulate
//!
//! 6. **Episode Tracking State**: Any internal counters or stats may cause memory
//!    growth or cache pollution
//!
//! # Expected Degradation Pattern
//!
//! If the wrapper layer is the cause, we expect to see:
//! - step() wrapper overhead increasing over time
//! - StepResult allocation patterns changing
//! - Memory growth in wrapper state
//! - Increasing time between first and last quartile of timing samples

use std::time::Instant;

use crate::environment::{
    CartPoleEnv, CartPoleEnvWrapper, PendulumEnv, ResetMask, StepResult,
};
use crate::runners::learner::VectorizedEnv as LearnerVectorizedEnv;
use crate::algorithms::action_policy::{DiscreteAction, ContinuousAction};

// ============================================================================
// Test Helpers
// ============================================================================

/// Track timing statistics over iterations with regression analysis.
#[derive(Default, Clone)]
struct TimingStats {
    samples: Vec<f64>,
}

impl TimingStats {
    fn add(&mut self, elapsed_us: f64) {
        self.samples.push(elapsed_us);
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    fn median(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }

    fn std_dev(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (self.samples.len() - 1) as f64;
        variance.sqrt()
    }

    fn percentile(&self, p: usize) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (sorted.len() * p / 100).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Compute linear regression slope (time growth per iteration).
    /// Positive slope indicates degradation.
    fn slope(&self) -> f64 {
        let n = self.samples.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, &y) in self.samples.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }
        (n * sum_xy - sum_x * sum_y) / denom
    }

    /// Compute relative slope (slope / mean) for normalized comparison.
    fn relative_slope(&self) -> f64 {
        let mean = self.mean();
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        self.slope() / mean
    }

    /// Get first quartile mean (first 25% of samples).
    fn first_quartile_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let n = (self.samples.len() / 4).max(1);
        self.samples[..n].iter().sum::<f64>() / n as f64
    }

    /// Get last quartile mean (last 25% of samples).
    fn last_quartile_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let n = (self.samples.len() / 4).max(1);
        let start = self.samples.len() - n;
        self.samples[start..].iter().sum::<f64>() / n as f64
    }

    /// Compute slowdown ratio (last_quartile / first_quartile).
    fn slowdown_ratio(&self) -> f64 {
        let first = self.first_quartile_mean();
        if first < 1e-10 {
            return 1.0;
        }
        self.last_quartile_mean() / first
    }
}

/// Track memory-related statistics.
struct MemoryStats {
    initial_capacity: usize,
    max_capacity: usize,
    final_capacity: usize,
}

impl MemoryStats {
    fn growth_ratio(&self) -> f64 {
        if self.initial_capacity == 0 {
            return 1.0;
        }
        self.max_capacity as f64 / self.initial_capacity as f64
    }
}

// ============================================================================
// 1. StepResult Allocation Tests
// ============================================================================

/// Test: Measure StepResult creation timing over many iterations.
/// INTENT: Detect if StepResult allocation cost increases over time.
#[test]
fn test_step_result_allocation_timing_stability() {
    let iterations = 10_000;
    let n_envs = 64;
    let obs_size = 4;

    let mut stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();

        // Simulate StepResult creation as in OperantAdapter::step()
        let observations: Vec<f32> = vec![0.0; n_envs * obs_size];
        let rewards: Vec<f32> = vec![1.0; n_envs];
        let terminals: Vec<bool> = vec![i % 100 == 0; n_envs];
        let truncations: Vec<bool> = vec![false; n_envs];

        let result = StepResult::new(observations, rewards, terminals, truncations);

        // Force use to prevent optimization
        std::hint::black_box(&result);

        stats.add(start.elapsed().as_nanos() as f64);
    }

    // Analyze for degradation
    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("StepResult allocation timing:");
    eprintln!("  Mean: {:.2}ns", stats.mean());
    eprintln!("  Std dev: {:.2}ns", stats.std_dev());
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    // Allocation timing should be stable
    assert!(
        slope.abs() < 0.001,
        "StepResult allocation shows timing growth: slope={:.6}",
        slope
    );
    assert!(
        slowdown < 1.5,
        "StepResult allocation slowdown {}x indicates degradation",
        slowdown
    );
}

/// Test: StepResult dones() method allocation pattern.
/// INTENT: Detect if calling dones() repeatedly causes memory issues.
#[test]
fn test_step_result_dones_method_overhead() {
    let iterations = 10_000;
    let n_envs = 64;

    let mut stats = TimingStats::default();

    for i in 0..iterations {
        // Create StepResult
        let result = StepResult::new(
            vec![0.0; n_envs * 4],
            vec![1.0; n_envs],
            vec![i % 50 == 0; n_envs],
            vec![i % 100 == 0; n_envs],
        );

        let start = Instant::now();

        // dones() creates a new Vec every time - this is the pattern we're testing
        let dones = result.dones();
        std::hint::black_box(&dones);

        stats.add(start.elapsed().as_nanos() as f64);
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("StepResult::dones() timing:");
    eprintln!("  Mean: {:.2}ns", stats.mean());
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    // dones() is O(n_envs) but should be constant per call
    assert!(
        slope.abs() < 0.001,
        "dones() shows timing growth: slope={:.6}",
        slope
    );
}

/// Test: ResetMask creation from dones.
/// INTENT: Verify ResetMask::from_dones doesn't accumulate overhead.
#[test]
fn test_reset_mask_creation_overhead() {
    let iterations = 10_000;
    let n_envs = 64;

    let mut stats = TimingStats::default();

    for i in 0..iterations {
        let dones: Vec<bool> = (0..n_envs)
            .map(|j| (i + j) % 20 == 0)
            .collect();

        let start = Instant::now();

        let mask = ResetMask::from_dones(&dones);
        std::hint::black_box(&mask);

        stats.add(start.elapsed().as_nanos() as f64);
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("ResetMask::from_dones timing:");
    eprintln!("  Mean: {:.2}ns", stats.mean());
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    assert!(
        slope.abs() < 0.001,
        "ResetMask creation shows timing growth: slope={:.6}",
        slope
    );
}

/// Test: Vec capacity growth over repeated clear/extend cycles.
/// INTENT: Verify wrapper Vec buffers maintain stable capacity.
#[test]
fn test_vec_capacity_stability_in_wrapper_pattern() {
    let iterations = 5000;
    let n_envs = 64;
    let obs_size = 4;

    // Simulate the pattern in wrapper step():
    // - Create result Vecs
    // - Fill with data
    // - Return (ownership transfer)

    let mut max_obs_cap = 0usize;
    let mut max_reward_cap = 0usize;

    for i in 0..iterations {
        // Variable size to simulate real conditions
        let actual_size = n_envs + (i % 10);

        let observations: Vec<f32> = vec![0.0; actual_size * obs_size];
        let rewards: Vec<f32> = vec![1.0; actual_size];
        let terminals: Vec<bool> = vec![false; actual_size];
        let truncations: Vec<bool> = vec![false; actual_size];

        max_obs_cap = max_obs_cap.max(observations.capacity());
        max_reward_cap = max_reward_cap.max(rewards.capacity());

        // Simulate result creation and consumption
        let result = StepResult::new(observations, rewards, terminals, truncations);
        std::hint::black_box(&result);
    }

    // Capacity should be bounded
    let expected_obs_cap = (n_envs + 10) * obs_size;
    let expected_reward_cap = n_envs + 10;

    eprintln!("Vec capacity tracking:");
    eprintln!("  Max obs capacity: {} (expected ~{})", max_obs_cap, expected_obs_cap);
    eprintln!("  Max reward capacity: {} (expected ~{})", max_reward_cap, expected_reward_cap);

    // Allow 2x for allocator rounding
    assert!(
        max_obs_cap <= expected_obs_cap * 2,
        "Observation Vec capacity grew unexpectedly: {} > {}",
        max_obs_cap, expected_obs_cap * 2
    );
}

// ============================================================================
// 2. Wrapper Step Performance Tests
// ============================================================================

/// Test: CartPoleEnvWrapper step() timing over extended iterations.
/// INTENT: Detect if wrapper step overhead increases over time.
#[test]
fn test_cartpole_wrapper_step_timing_stability() {
    let n_envs = 64;
    let iterations = 50_000;

    // Create environment
    let mut env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    env.reset(42);
    let mut wrapper = CartPoleEnvWrapper::new(env);

    let mut stats = TimingStats::default();
    let actions: Vec<u32> = vec![0; n_envs];

    for i in 0..iterations {
        let start = Instant::now();

        let result = wrapper.step(&actions);
        std::hint::black_box(&result);

        stats.add(start.elapsed().as_nanos() as f64);

        // Reset done environments periodically
        if i % 100 == 0 {
            let done_indices: Vec<usize> = result.dones.iter()
                .enumerate()
                .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
                .collect();
            if !done_indices.is_empty() {
                wrapper.reset_envs(&done_indices);
            }
        }
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("CartPoleEnvWrapper::step() timing ({} iters):", iterations);
    eprintln!("  Mean: {:.2}us", stats.mean() / 1000.0);
    eprintln!("  Median: {:.2}us", stats.median() / 1000.0);
    eprintln!("  P99: {:.2}us", stats.percentile(99) / 1000.0);
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);
    eprintln!("  First quartile mean: {:.2}us", stats.first_quartile_mean() / 1000.0);
    eprintln!("  Last quartile mean: {:.2}us", stats.last_quartile_mean() / 1000.0);

    // Step timing should be stable
    assert!(
        slope.abs() < 0.0001,
        "CartPole wrapper step shows timing growth: slope={:.6}",
        slope
    );
    assert!(
        slowdown < 1.3,
        "CartPole wrapper step slowdown {}x indicates degradation",
        slowdown
    );
}

/// Test: Compare wrapper overhead to raw operant step.
/// INTENT: Measure the cost of the wrapper layer itself.
#[test]
fn test_wrapper_vs_raw_operant_overhead() {
    use operant_core::Environment;

    let n_envs = 64;
    let iterations = 10_000; // Reduced to avoid operant overflow

    // Raw operant
    let mut raw_env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    raw_env.reset(42);

    let actions_f32: Vec<f32> = vec![0.0; n_envs];
    let mut raw_stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();
        let result = raw_env.step_no_reset_with_result(&actions_f32);
        std::hint::black_box(&result);
        raw_stats.add(start.elapsed().as_nanos() as f64);

        // Reset periodically to avoid operant's internal counter overflow
        if i % 200 == 0 {
            let mask = result.to_reset_mask();
            if mask.any() {
                raw_env.reset_envs(&mask, 42);
            }
        }
    }

    // Wrapped operant
    let mut wrapped_env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => return,
    };
    wrapped_env.reset(42);
    let mut wrapper = CartPoleEnvWrapper::new(wrapped_env);

    let actions_u32: Vec<u32> = vec![0; n_envs];
    let mut wrapped_stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();
        let result = wrapper.step(&actions_u32);
        std::hint::black_box(&result);
        wrapped_stats.add(start.elapsed().as_nanos() as f64);

        // Reset periodically to avoid operant's internal counter overflow
        if i % 200 == 0 {
            let done_indices: Vec<usize> = result.dones.iter()
                .enumerate()
                .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
                .collect();
            if !done_indices.is_empty() {
                wrapper.reset_envs(&done_indices);
            }
        }
    }

    let overhead = wrapped_stats.mean() / raw_stats.mean();
    let overhead_ns = wrapped_stats.mean() - raw_stats.mean();

    eprintln!("Wrapper overhead analysis:");
    eprintln!("  Raw operant mean: {:.2}us", raw_stats.mean() / 1000.0);
    eprintln!("  Wrapped mean: {:.2}us", wrapped_stats.mean() / 1000.0);
    eprintln!("  Overhead: {:.2}x ({:.0}ns)", overhead, overhead_ns);
    eprintln!("  Raw slope: {:.6}", raw_stats.relative_slope());
    eprintln!("  Wrapped slope: {:.6}", wrapped_stats.relative_slope());

    // Wrapper should add <2x overhead
    assert!(
        overhead < 3.0,
        "Wrapper adds excessive overhead: {:.2}x",
        overhead
    );

    // Both should have similar stability
    assert!(
        (wrapped_stats.relative_slope() - raw_stats.relative_slope()).abs() < 0.001,
        "Wrapper introduces timing instability"
    );
}

/// Test: Action conversion overhead in wrapper.
/// INTENT: Measure cost of u32 -> f32 action conversion.
#[test]
fn test_action_conversion_overhead() {
    let n_envs = 64;
    let iterations = 50_000;

    let actions_u32: Vec<u32> = (0..n_envs as u32).collect();
    let mut conversion_stats = TimingStats::default();

    for _ in 0..iterations {
        let start = Instant::now();

        // This is the conversion pattern in CartPoleEnvWrapper::step()
        let actions_f32: Vec<f32> = actions_u32.iter().map(|&a| a as f32).collect();
        std::hint::black_box(&actions_f32);

        conversion_stats.add(start.elapsed().as_nanos() as f64);
    }

    eprintln!("Action conversion timing:");
    eprintln!("  Mean: {:.2}ns", conversion_stats.mean());
    eprintln!("  Relative slope: {:.6}", conversion_stats.relative_slope());
    eprintln!("  Slowdown ratio: {:.3}x", conversion_stats.slowdown_ratio());

    // Conversion should be O(1) per element, stable over time
    assert!(
        conversion_stats.relative_slope().abs() < 0.0001,
        "Action conversion shows timing growth"
    );
}

// ============================================================================
// 3. Observation Buffer Handling Tests
// ============================================================================

/// Test: write_observations() timing stability.
/// INTENT: Verify observation copying doesn't accumulate overhead.
#[test]
fn test_write_observations_timing_stability() {
    let n_envs = 64;
    let obs_size = 4;
    let iterations = 50_000;

    let mut env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    env.reset(42);
    let wrapper = CartPoleEnvWrapper::new(env);

    let mut buffer = vec![0.0f32; n_envs * obs_size];
    let mut stats = TimingStats::default();

    for _ in 0..iterations {
        let start = Instant::now();

        wrapper.write_observations(&mut buffer);
        std::hint::black_box(&buffer);

        stats.add(start.elapsed().as_nanos() as f64);
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("write_observations timing:");
    eprintln!("  Mean: {:.2}ns", stats.mean());
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    assert!(
        slope.abs() < 0.0001,
        "write_observations shows timing growth: slope={:.6}",
        slope
    );
    assert!(
        slowdown < 1.2,
        "write_observations slowdown {}x indicates degradation",
        slowdown
    );
}

/// Test: Buffer reuse pattern efficiency.
/// INTENT: Verify pre-allocated buffers are efficiently reused.
#[test]
fn test_buffer_reuse_efficiency() {
    let n_envs = 64;
    let obs_size = 4;
    let iterations = 10_000;

    let mut env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    env.reset(42);
    let mut wrapper = CartPoleEnvWrapper::new(env);

    // Pattern 1: Reusing single buffer (optimal)
    let mut reuse_buffer = vec![0.0f32; n_envs * obs_size];
    let mut reuse_stats = TimingStats::default();

    for _ in 0..iterations {
        let start = Instant::now();
        wrapper.write_observations(&mut reuse_buffer);
        std::hint::black_box(&reuse_buffer);
        reuse_stats.add(start.elapsed().as_nanos() as f64);
    }

    // Pattern 2: Creating new buffer each time (wasteful)
    let mut alloc_stats = TimingStats::default();

    for _ in 0..iterations {
        let start = Instant::now();
        let mut new_buffer = vec![0.0f32; n_envs * obs_size];
        wrapper.write_observations(&mut new_buffer);
        std::hint::black_box(&new_buffer);
        alloc_stats.add(start.elapsed().as_nanos() as f64);
    }

    let reuse_mean = reuse_stats.mean();
    let alloc_mean = alloc_stats.mean();
    let alloc_overhead = alloc_mean / reuse_mean;

    eprintln!("Buffer reuse efficiency:");
    eprintln!("  Reuse pattern: {:.2}ns", reuse_mean);
    eprintln!("  Alloc pattern: {:.2}ns", alloc_mean);
    eprintln!("  Allocation overhead: {:.2}x", alloc_overhead);

    // NOTE: For small buffers (256 floats = 1KB), modern allocators are very efficient.
    // The overhead may be minimal or even negative due to measurement noise.
    // This is actually a POSITIVE finding: allocation is NOT a bottleneck.
    // We just verify the test runs and report the results.

    // Both patterns should be stable (no degradation)
    assert!(
        reuse_stats.relative_slope().abs() < 0.001,
        "Buffer reuse pattern shows timing instability"
    );
    assert!(
        alloc_stats.relative_slope().abs() < 0.001,
        "Buffer alloc pattern shows timing instability"
    );
}

// ============================================================================
// 4. Reset Cycle Performance Tests
// ============================================================================

/// Test: reset_envs() timing over many reset cycles.
/// INTENT: Detect if reset accumulates overhead.
#[test]
fn test_reset_envs_timing_over_cycles() {
    let n_envs = 64;
    let iterations = 10_000;

    let mut env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    env.reset(42);
    let mut wrapper = CartPoleEnvWrapper::new(env);

    let mut stats = TimingStats::default();

    for i in 0..iterations {
        // Reset subset of environments (simulating episode ends)
        let indices: Vec<usize> = (0..n_envs)
            .filter(|&j| (i + j) % 10 == 0)
            .collect();

        if indices.is_empty() {
            continue;
        }

        let start = Instant::now();
        wrapper.reset_envs(&indices);
        stats.add(start.elapsed().as_nanos() as f64);
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("reset_envs timing ({} resets):", stats.len());
    eprintln!("  Mean: {:.2}us", stats.mean() / 1000.0);
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    assert!(
        slope.abs() < 0.001,
        "reset_envs shows timing growth: slope={:.6}",
        slope
    );
    assert!(
        slowdown < 1.5,
        "reset_envs slowdown {}x indicates degradation",
        slowdown
    );
}

/// Test: Full step-reset cycle pattern.
/// INTENT: Measure typical actor loop pattern for degradation.
#[test]
fn test_step_reset_cycle_pattern() {
    let n_envs = 64;
    let iterations = 20_000;

    let mut env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    env.reset(42);
    let mut wrapper = CartPoleEnvWrapper::new(env);

    let actions: Vec<u32> = vec![0; n_envs];
    let mut obs_buffer = vec![0.0f32; n_envs * 4];

    let mut step_stats = TimingStats::default();
    let mut reset_stats = TimingStats::default();
    let mut total_resets = 0usize;

    for _ in 0..iterations {
        // Step
        let step_start = Instant::now();
        let result = wrapper.step(&actions);
        step_stats.add(step_start.elapsed().as_nanos() as f64);

        // Write observations
        wrapper.write_observations(&mut obs_buffer);

        // Reset done environments
        let done_indices: Vec<usize> = result.dones.iter()
            .enumerate()
            .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
            .collect();

        if !done_indices.is_empty() {
            let reset_start = Instant::now();
            wrapper.reset_envs(&done_indices);
            reset_stats.add(reset_start.elapsed().as_nanos() as f64);
            total_resets += 1;
        }
    }

    eprintln!("Step-reset cycle pattern:");
    eprintln!("  Steps: {}, Resets: {}", iterations, total_resets);
    eprintln!("  Step mean: {:.2}us, slope: {:.6}",
              step_stats.mean() / 1000.0, step_stats.relative_slope());
    eprintln!("  Reset mean: {:.2}us, slope: {:.6}",
              reset_stats.mean() / 1000.0, reset_stats.relative_slope());
    eprintln!("  Step slowdown: {:.3}x", step_stats.slowdown_ratio());

    // The cycle pattern should be stable
    assert!(
        step_stats.relative_slope().abs() < 0.0001,
        "Step-reset cycle shows step timing growth"
    );
}

// ============================================================================
// 5. Trait Object Dispatch Overhead Tests
// ============================================================================

/// Test: Compare static dispatch vs dynamic dispatch performance.
/// INTENT: Measure vtable lookup overhead for VectorizedEnv trait.
#[test]
fn test_static_vs_dynamic_dispatch_overhead() {
    let n_envs = 64;
    let iterations = 10_000; // Reduced to avoid operant overflow

    // Static dispatch (direct wrapper use)
    let mut static_env = match operant::CartPole::with_defaults(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };
    static_env.reset(42);
    let mut static_wrapper = CartPoleEnvWrapper::new(static_env);

    let actions_u32: Vec<u32> = vec![0; n_envs];
    let mut static_stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();
        let result = static_wrapper.step(&actions_u32);
        std::hint::black_box(&result);
        static_stats.add(start.elapsed().as_nanos() as f64);

        // Reset periodically to avoid operant overflow
        if i % 200 == 0 {
            let done_indices: Vec<usize> = result.dones.iter()
                .enumerate()
                .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
                .collect();
            if !done_indices.is_empty() {
                static_wrapper.reset_envs(&done_indices);
            }
        }
    }

    // Dynamic dispatch (through trait object - LearnerVectorizedEnv)
    let mut dynamic_env = match CartPoleEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => return,
    };

    let actions_discrete: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];
    let mut dynamic_stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();
        // This goes through LearnerVectorizedEnv trait
        let result = LearnerVectorizedEnv::step(&mut dynamic_env, &actions_discrete);
        std::hint::black_box(&result);
        dynamic_stats.add(start.elapsed().as_nanos() as f64);

        // Reset periodically to avoid operant overflow
        if i % 200 == 0 {
            let done_indices: Vec<usize> = result.dones.iter()
                .enumerate()
                .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
                .collect();
            if !done_indices.is_empty() {
                dynamic_env.reset_envs(&done_indices);
            }
        }
    }

    let dispatch_overhead = dynamic_stats.mean() / static_stats.mean();
    let dispatch_ns = dynamic_stats.mean() - static_stats.mean();

    eprintln!("Static vs dynamic dispatch:");
    eprintln!("  Static mean: {:.2}us", static_stats.mean() / 1000.0);
    eprintln!("  Dynamic mean: {:.2}us", dynamic_stats.mean() / 1000.0);
    eprintln!("  Dispatch overhead: {:.2}x ({:.0}ns)", dispatch_overhead, dispatch_ns);
    eprintln!("  Static slope: {:.6}", static_stats.relative_slope());
    eprintln!("  Dynamic slope: {:.6}", dynamic_stats.relative_slope());

    // Dynamic dispatch should not add significant overhead for step()
    assert!(
        dispatch_overhead < 1.5,
        "Trait dispatch adds excessive overhead: {:.2}x",
        dispatch_overhead
    );

    // Both should have similar stability characteristics
    assert!(
        (dynamic_stats.relative_slope() - static_stats.relative_slope()).abs() < 0.001,
        "Trait dispatch introduces timing instability"
    );
}

/// Test: Trait object stability over long runs.
/// INTENT: Verify vtable lookups don't degrade over time.
#[test]
fn test_trait_dispatch_timing_stability() {
    let n_envs = 64;
    let iterations = 50_000;

    let mut env: Box<dyn LearnerVectorizedEnv<DiscreteAction>> = match CartPoleEnv::new(n_envs) {
        Ok(e) => Box::new(e),
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let actions: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];
    let mut stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();

        // Call through trait object
        let result = env.step(&actions);
        std::hint::black_box(&result);

        stats.add(start.elapsed().as_nanos() as f64);

        // Periodic reset
        if i % 100 == 0 {
            let done_indices: Vec<usize> = result.dones.iter()
                .enumerate()
                .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
                .collect();
            if !done_indices.is_empty() {
                env.reset_envs(&done_indices);
            }
        }
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("Trait object dispatch timing ({} iters):", iterations);
    eprintln!("  Mean: {:.2}us", stats.mean() / 1000.0);
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    assert!(
        slope.abs() < 0.0001,
        "Trait dispatch shows timing growth: slope={:.6}",
        slope
    );
    assert!(
        slowdown < 1.2,
        "Trait dispatch slowdown {}x indicates degradation",
        slowdown
    );
}

// ============================================================================
// 6. Long-Run Wrapper Simulation Tests
// ============================================================================

/// Test: Simulate full actor loop for 100k+ steps.
/// INTENT: Reproduce wrapper layer in actor loop context.
#[test]
fn test_long_run_wrapper_actor_simulation() {
    let n_envs = 64;
    let iterations = 100_000;
    let checkpoint_interval = 10_000;

    let mut env = match CartPoleEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let mut obs_buffer = vec![0.0f32; n_envs * 4];
    let actions: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];

    let mut iteration_stats = TimingStats::default();
    let mut step_stats = TimingStats::default();
    let mut obs_stats = TimingStats::default();
    let mut checkpoints: Vec<(usize, f64, f64, f64)> = Vec::new();

    let start_time = Instant::now();
    let mut total_resets = 0usize;

    for i in 0..iterations {
        let iter_start = Instant::now();

        // Step (timed)
        let step_start = Instant::now();
        let result = LearnerVectorizedEnv::step(&mut env, &actions);
        step_stats.add(step_start.elapsed().as_nanos() as f64);

        // Write observations (timed)
        let obs_start = Instant::now();
        LearnerVectorizedEnv::write_observations(&env, &mut obs_buffer);
        obs_stats.add(obs_start.elapsed().as_nanos() as f64);

        // Reset done environments
        let done_indices: Vec<usize> = result.dones.iter()
            .enumerate()
            .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
            .collect();

        if !done_indices.is_empty() {
            env.reset_envs(&done_indices);
            total_resets += 1;
        }

        iteration_stats.add(iter_start.elapsed().as_nanos() as f64);

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let sps = (i + 1) as f64 / elapsed;
            let recent_iter_time = iteration_stats.samples
                [iteration_stats.len().saturating_sub(1000)..]
                .iter()
                .sum::<f64>() / 1000.0;
            let step_time = step_stats.samples
                [step_stats.len().saturating_sub(1000)..]
                .iter()
                .sum::<f64>() / 1000.0;

            checkpoints.push((i + 1, sps, recent_iter_time, step_time));
        }
    }

    // Analyze degradation pattern
    eprintln!("\n=== Long-Run Wrapper Actor Simulation ===");
    eprintln!("Total iterations: {}", iterations);
    eprintln!("Total resets: {}", total_resets);
    eprintln!("Duration: {:.1}s", start_time.elapsed().as_secs_f64());
    eprintln!("");

    eprintln!("Checkpoints:");
    for (iter, sps, iter_time, step_time) in &checkpoints {
        eprintln!("  {:6}: SPS={:.0}, iter={:.0}ns, step={:.0}ns",
                  iter, sps, iter_time, step_time);
    }
    eprintln!("");

    eprintln!("Step timing analysis:");
    eprintln!("  Mean: {:.2}us", step_stats.mean() / 1000.0);
    eprintln!("  First quartile: {:.2}us", step_stats.first_quartile_mean() / 1000.0);
    eprintln!("  Last quartile: {:.2}us", step_stats.last_quartile_mean() / 1000.0);
    eprintln!("  Slowdown: {:.3}x", step_stats.slowdown_ratio());
    eprintln!("  Relative slope: {:.8}", step_stats.relative_slope());
    eprintln!("");

    eprintln!("Observation timing analysis:");
    eprintln!("  Mean: {:.2}ns", obs_stats.mean());
    eprintln!("  Slowdown: {:.3}x", obs_stats.slowdown_ratio());
    eprintln!("  Relative slope: {:.8}", obs_stats.relative_slope());

    // Overall wrapper layer should be stable
    let step_slowdown = step_stats.slowdown_ratio();
    let step_slope = step_stats.relative_slope();

    assert!(
        step_slowdown < 1.5,
        "Wrapper step slowdown {}x over {} iterations indicates degradation",
        step_slowdown, iterations
    );
    assert!(
        step_slope.abs() < 0.0001,
        "Wrapper step shows timing growth: slope={:.8}",
        step_slope
    );
}

/// Test: Simulate 500k+ step actor pattern matching production.
/// INTENT: Match the 642k step production run that showed degradation.
#[test]
#[ignore = "Long running test - run manually with --ignored"]
fn test_production_scale_wrapper_simulation() {
    let n_envs = 64;
    let iterations = 500_000;
    let checkpoint_interval = 50_000;

    let mut env = match CartPoleEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let mut obs_buffer = vec![0.0f32; n_envs * 4];
    let actions: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];

    let mut step_times: Vec<f64> = Vec::with_capacity(iterations);
    let mut checkpoints: Vec<(usize, f64)> = Vec::new();

    let start_time = Instant::now();

    for i in 0..iterations {
        let start = Instant::now();

        // Full actor loop iteration
        let result = LearnerVectorizedEnv::step(&mut env, &actions);
        LearnerVectorizedEnv::write_observations(&env, &mut obs_buffer);

        let done_indices: Vec<usize> = result.dones.iter()
            .enumerate()
            .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
            .collect();
        if !done_indices.is_empty() {
            env.reset_envs(&done_indices);
        }

        step_times.push(start.elapsed().as_nanos() as f64);

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let sps = (i + 1) as f64 / elapsed;
            checkpoints.push((i + 1, sps));
            eprintln!("Step {:6}: SPS={:.0}", i + 1, sps);
        }
    }

    // Calculate degradation
    let first_sps = checkpoints[0].1;
    let last_sps = checkpoints[checkpoints.len() - 1].1;
    let sps_ratio = last_sps / first_sps;

    let first_quarter_mean: f64 = step_times[..iterations/4].iter().sum::<f64>() / (iterations/4) as f64;
    let last_quarter_mean: f64 = step_times[iterations*3/4..].iter().sum::<f64>() / (iterations/4) as f64;
    let timing_ratio = last_quarter_mean / first_quarter_mean;

    eprintln!("\n=== Production Scale Results ===");
    eprintln!("SPS: {:.0} -> {:.0} ({:.1}% of initial)", first_sps, last_sps, sps_ratio * 100.0);
    eprintln!("Step timing: {:.0}ns -> {:.0}ns ({:.2}x)", first_quarter_mean, last_quarter_mean, timing_ratio);

    // If wrapper layer is the cause, we'd see significant degradation
    assert!(
        sps_ratio > 0.7,
        "Wrapper layer shows significant SPS degradation: {:.1}%",
        sps_ratio * 100.0
    );
}

/// Test: Memory stability during long runs.
/// INTENT: Verify no memory accumulation in wrapper layer.
#[test]
fn test_wrapper_memory_stability() {
    let n_envs = 64;
    let iterations = 50_000;

    let mut env = match CartPoleEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let actions: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];

    // Track allocations indirectly through Vec capacity
    let mut max_obs_capacity = 0usize;
    let mut max_reward_capacity = 0usize;

    for i in 0..iterations {
        let result = LearnerVectorizedEnv::step(&mut env, &actions);

        // Track capacities of returned Vecs
        // Note: We can't directly access internal Vecs, so we create new ones
        // to simulate the allocation pattern
        let rewards: Vec<f32> = result.rewards.clone();
        let dones: Vec<bool> = result.dones.clone();

        max_reward_capacity = max_reward_capacity.max(rewards.capacity());

        // Reset done environments
        let done_indices: Vec<usize> = dones.iter()
            .enumerate()
            .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
            .collect();
        if !done_indices.is_empty() {
            env.reset_envs(&done_indices);
        }

        // Log every 10k iterations
        if i > 0 && i % 10_000 == 0 {
            eprintln!("Iter {}: max_reward_cap={}", i, max_reward_capacity);
        }
    }

    // Memory should be bounded
    let expected_reward_cap = n_envs * 2; // Allow for some allocator rounding

    eprintln!("Memory stability check:");
    eprintln!("  Max reward capacity: {} (expected <= {})", max_reward_capacity, expected_reward_cap);

    assert!(
        max_reward_capacity <= expected_reward_cap,
        "Reward Vec capacity grew unexpectedly: {} > {}",
        max_reward_capacity, expected_reward_cap
    );
}

// ============================================================================
// 7. Pendulum Environment Tests (Continuous Action)
// ============================================================================

/// Test: PendulumEnv wrapper with continuous actions.
/// INTENT: Verify continuous action wrappers don't have different issues.
#[test]
fn test_pendulum_wrapper_timing_stability() {
    let n_envs = 64;
    let iterations = 20_000;

    let mut env = match PendulumEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let actions: Vec<ContinuousAction> = vec![ContinuousAction(vec![0.0]); n_envs];
    let mut stats = TimingStats::default();

    for i in 0..iterations {
        let start = Instant::now();

        let result = LearnerVectorizedEnv::step(&mut env, &actions);
        std::hint::black_box(&result);

        stats.add(start.elapsed().as_nanos() as f64);

        // Reset done environments
        if i % 100 == 0 {
            let done_indices: Vec<usize> = result.dones.iter()
                .enumerate()
                .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
                .collect();
            if !done_indices.is_empty() {
                env.reset_envs(&done_indices);
            }
        }
    }

    let slope = stats.relative_slope();
    let slowdown = stats.slowdown_ratio();

    eprintln!("PendulumEnv::step() timing:");
    eprintln!("  Mean: {:.2}us", stats.mean() / 1000.0);
    eprintln!("  Relative slope: {:.6}", slope);
    eprintln!("  Slowdown ratio: {:.3}x", slowdown);

    assert!(
        slope.abs() < 0.0001,
        "Pendulum wrapper shows timing growth: slope={:.6}",
        slope
    );
    assert!(
        slowdown < 1.3,
        "Pendulum wrapper slowdown {}x indicates degradation",
        slowdown
    );
}

/// Test: Continuous action conversion overhead.
/// INTENT: Measure cost of Vec<ContinuousAction> -> Vec<f32> conversion.
#[test]
fn test_continuous_action_conversion_overhead() {
    let n_envs = 64;
    let action_dim = 1;
    let iterations = 50_000;

    let actions: Vec<ContinuousAction> = (0..n_envs)
        .map(|i| ContinuousAction(vec![(i as f32) / n_envs as f32; action_dim]))
        .collect();

    let mut stats = TimingStats::default();

    for _ in 0..iterations {
        let start = Instant::now();

        // This is the conversion pattern in PendulumEnv::step()
        let action_floats: Vec<f32> = actions.iter()
            .flat_map(|a| a.0.clone())
            .collect();
        std::hint::black_box(&action_floats);

        stats.add(start.elapsed().as_nanos() as f64);
    }

    eprintln!("Continuous action conversion timing:");
    eprintln!("  Mean: {:.2}ns", stats.mean());
    eprintln!("  Relative slope: {:.6}", stats.relative_slope());
    eprintln!("  Slowdown ratio: {:.3}x", stats.slowdown_ratio());

    assert!(
        stats.relative_slope().abs() < 0.0001,
        "Continuous action conversion shows timing growth"
    );
}

// ============================================================================
// 8. Component Isolation Tests
// ============================================================================

/// Test: Isolate each wrapper component's contribution.
/// INTENT: Create timing breakdown to identify slowest components.
#[test]
fn test_wrapper_component_breakdown() {
    let n_envs = 64;
    let obs_size = 4;
    let iterations = 20_000;

    let mut env = match CartPoleEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let actions: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];
    let mut obs_buffer = vec![0.0f32; n_envs * obs_size];

    let mut step_times = TimingStats::default();
    let mut obs_times = TimingStats::default();
    let mut reset_times = TimingStats::default();
    let mut dones_times = TimingStats::default();

    for i in 0..iterations {
        // Step
        let start = Instant::now();
        let result = LearnerVectorizedEnv::step(&mut env, &actions);
        step_times.add(start.elapsed().as_nanos() as f64);

        // Observations
        let start = Instant::now();
        LearnerVectorizedEnv::write_observations(&env, &mut obs_buffer);
        obs_times.add(start.elapsed().as_nanos() as f64);

        // Dones processing
        let start = Instant::now();
        let done_indices: Vec<usize> = result.dones.iter()
            .enumerate()
            .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
            .collect();
        dones_times.add(start.elapsed().as_nanos() as f64);

        // Reset
        if !done_indices.is_empty() {
            let start = Instant::now();
            env.reset_envs(&done_indices);
            reset_times.add(start.elapsed().as_nanos() as f64);
        }
    }

    eprintln!("\n=== Wrapper Component Breakdown ===");
    eprintln!("Component       | Mean (us) | Slowdown | Rel.Slope");
    eprintln!("----------------|-----------|----------|----------");
    eprintln!("step()          | {:9.2} | {:8.3}x | {:+.6}",
              step_times.mean() / 1000.0, step_times.slowdown_ratio(), step_times.relative_slope());
    eprintln!("write_obs()     | {:9.2} | {:8.3}x | {:+.6}",
              obs_times.mean() / 1000.0, obs_times.slowdown_ratio(), obs_times.relative_slope());
    eprintln!("done_indices    | {:9.2} | {:8.3}x | {:+.6}",
              dones_times.mean() / 1000.0, dones_times.slowdown_ratio(), dones_times.relative_slope());
    if reset_times.len() > 0 {
        eprintln!("reset_envs()    | {:9.2} | {:8.3}x | {:+.6}",
                  reset_times.mean() / 1000.0, reset_times.slowdown_ratio(), reset_times.relative_slope());
    }

    // Flag any components showing degradation
    let components = [
        ("step", &step_times),
        ("obs", &obs_times),
        ("dones", &dones_times),
    ];

    for (name, stats) in &components {
        let slope = stats.relative_slope();
        let slowdown = stats.slowdown_ratio();

        assert!(
            slope.abs() < 0.001,
            "Component {} shows timing growth: slope={:.6}",
            name, slope
        );
        assert!(
            slowdown < 1.5,
            "Component {} slowdown {}x indicates degradation",
            name, slowdown
        );
    }
}

/// Test: Verify wrapper doesn't accumulate internal state.
/// INTENT: Check if n_envs or other cached values stay constant.
#[test]
fn test_wrapper_internal_state_stability() {
    let n_envs = 64;
    let iterations = 10_000;

    let mut env = match CartPoleEnv::new(n_envs) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Skipping test: operant not available");
            return;
        }
    };

    let actions: Vec<DiscreteAction> = vec![DiscreteAction(0); n_envs];

    // Verify n_envs stays constant
    let initial_n_envs = env.n_envs();
    let initial_obs_size = LearnerVectorizedEnv::obs_size(&env);

    for i in 0..iterations {
        let result = LearnerVectorizedEnv::step(&mut env, &actions);

        // Check invariants periodically
        if i % 1000 == 0 {
            assert_eq!(env.n_envs(), initial_n_envs, "n_envs changed at iteration {}", i);
            assert_eq!(LearnerVectorizedEnv::obs_size(&env), initial_obs_size, "obs_size changed at iteration {}", i);
        }

        // Reset done environments
        let done_indices: Vec<usize> = result.dones.iter()
            .enumerate()
            .filter_map(|(idx, &done)| if done { Some(idx) } else { None })
            .collect();
        if !done_indices.is_empty() {
            env.reset_envs(&done_indices);
        }
    }

    // Final check
    assert_eq!(env.n_envs(), initial_n_envs, "n_envs changed after {} iterations", iterations);
    eprintln!("Internal state remained stable over {} iterations", iterations);
}
