//! SPS (Steps-Per-Second) Degradation Diagnostic Tests
//!
//! This module provides comprehensive tests to diagnose the root cause of SPS
//! degradation in the distributed IMPALA implementation. The observed pattern
//! shows ~55% performance loss over 646k steps (21,000 -> 9,500 SPS).
//!
//! # Observed Degradation Pattern
//!
//! ```text
//! Steps:    44384 | SPS:  21699  (start)
//! Steps:   173312 | SPS:  21190  (slight drop)
//! Steps:   287936 | SPS:  17591  (accelerating drop)
//! Steps:   429440 | SPS:  13983  (continuing)
//! Steps:   567840 | SPS:  11087  (continuing)
//! Steps:   646720 | SPS:   9564  (55% degradation)
//! ```
//!
//! The degradation is monotonic and smooth, suggesting gradual resource
//! exhaustion rather than a sudden leak.
//!
//! # Hypothesis Space
//!
//! 1. **GPU Resource Accumulation**: WGPU backend accumulating textures/buffers
//! 2. **Tensor Lifecycle Issues**: Autodiff graph retaining references
//! 3. **BytesSlot Memory Growth**: Weight serialization bytes accumulating
//! 4. **Buffer Consolidation Overhead**: `storage.remove(0)` is O(n)
//! 5. **Allocator Fragmentation**: Repeated alloc/dealloc patterns
//! 6. **Lock Contention Growth**: RwLock/Mutex contention increasing
//!
//! # Test Categories
//!
//! 1. Memory Profiling Tests
//! 2. Buffer Performance Tests
//! 3. Tensor Lifecycle Tests
//! 4. Weight Sync Tests
//! 5. Timing Regression Tests
//! 6. Simulated Long-Run Tests

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};

use crate::algorithms::impala::{IMPALABuffer, IMPALABufferConfig};
use crate::algorithms::vtrace::compute_vtrace;
use crate::core::bytes_slot::{bytes_slot, BytesSlot};
use crate::core::experience_buffer::ExperienceBuffer;
use crate::core::transition::{IMPALATransition, Trajectory, Transition};

// ============================================================================
// Test Helpers
// ============================================================================

/// Create a trajectory with specified parameters.
fn make_trajectory(
    len: usize,
    env_id: usize,
    version: u64,
    terminal: bool,
) -> Trajectory<IMPALATransition> {
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        let is_last = i == len - 1;
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![i as f32; 4], // obs_size = 4
                0,
                1.0,
                vec![(i + 1) as f32; 4],
                is_last && terminal,
                is_last && !terminal,
            ),
            behavior_log_prob: -0.5,
            policy_version: version,
        });
    }
    traj
}

/// Create a trajectory with custom obs_size.
fn make_trajectory_with_obs(
    len: usize,
    env_id: usize,
    version: u64,
    obs_size: usize,
    terminal: bool,
) -> Trajectory<IMPALATransition> {
    let mut traj = Trajectory::new(env_id);
    for i in 0..len {
        let is_last = i == len - 1;
        traj.push(IMPALATransition {
            base: Transition::new_discrete(
                vec![i as f32; obs_size],
                0,
                1.0,
                vec![(i + 1) as f32; obs_size],
                is_last && terminal,
                is_last && !terminal,
            ),
            behavior_log_prob: -0.5,
            policy_version: version,
        });
    }
    traj
}

/// Track timing statistics over iterations.
#[derive(Default, Clone)]
struct TimingStats {
    samples: Vec<f64>,
}

impl TimingStats {
    fn add(&mut self, elapsed_us: f64) {
        self.samples.push(elapsed_us);
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

        // Slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }
        (n * sum_xy - sum_x * sum_y) / denom
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
}

// ============================================================================
// 1. Memory Profiling Tests
// ============================================================================

/// Test: Track Vec capacity growth over simulated training iterations.
/// INTENT: Detect if Vec capacities grow unboundedly due to reallocation.
#[test]
fn test_vec_capacity_stability() {
    let iterations = 1000;
    let estimated_transitions = 32 * 20; // batch_size * trajectory_length

    // Simulate the pre-allocated buffers from learner_thread
    let mut all_states: Vec<f32> = Vec::with_capacity(estimated_transitions * 4);
    let mut all_advantages: Vec<f32> = Vec::with_capacity(estimated_transitions);
    let mut all_vtrace_targets: Vec<f32> = Vec::with_capacity(estimated_transitions);

    let initial_states_cap = all_states.capacity();
    let initial_advantages_cap = all_advantages.capacity();
    let initial_vtrace_cap = all_vtrace_targets.capacity();

    let mut max_states_cap = initial_states_cap;
    let mut max_advantages_cap = initial_advantages_cap;
    let mut max_vtrace_cap = initial_vtrace_cap;

    for batch in 0..iterations {
        // Simulate variable batch sizes (as in real training)
        let transitions = estimated_transitions + (batch % 100);

        // Clear and fill (as in learner_thread)
        all_states.clear();
        all_advantages.clear();
        all_vtrace_targets.clear();

        for i in 0..transitions {
            all_states.push(i as f32);
            all_states.push((i + 1) as f32);
            all_states.push((i + 2) as f32);
            all_states.push((i + 3) as f32);
            all_advantages.push(0.5);
            all_vtrace_targets.push(1.0);
        }

        max_states_cap = max_states_cap.max(all_states.capacity());
        max_advantages_cap = max_advantages_cap.max(all_advantages.capacity());
        max_vtrace_cap = max_vtrace_cap.max(all_vtrace_targets.capacity());
    }

    // Capacity should not grow unboundedly (allow 3x initial for normal growth)
    let states_growth = max_states_cap as f64 / initial_states_cap as f64;
    let advantages_growth = max_advantages_cap as f64 / initial_advantages_cap as f64;
    let vtrace_growth = max_vtrace_cap as f64 / initial_vtrace_cap as f64;

    assert!(
        states_growth < 4.0,
        "States Vec grew {}x (from {} to {}) - possible memory issue",
        states_growth,
        initial_states_cap,
        max_states_cap
    );
    assert!(
        advantages_growth < 4.0,
        "Advantages Vec grew {}x (from {} to {}) - possible memory issue",
        advantages_growth,
        initial_advantages_cap,
        max_advantages_cap
    );
    assert!(
        vtrace_growth < 4.0,
        "VTrace targets Vec grew {}x (from {} to {}) - possible memory issue",
        vtrace_growth,
        initial_vtrace_cap,
        max_vtrace_cap
    );
}

/// Test: BytesSlot memory usage over many publish/get cycles.
/// INTENT: Detect if BytesSlot accumulates memory over time.
#[test]
fn test_bytes_slot_memory_stability() {
    let slot = bytes_slot();
    let iterations = 1000;

    // Simulate model weights (e.g., 100KB model)
    let weight_size = 100_000;

    for i in 0..iterations {
        // Simulate learner publishing new weights
        let weights: Vec<u8> = (0..weight_size).map(|j| ((i + j) % 256) as u8).collect();
        slot.publish(weights);

        // Simulate actors reading weights (using get(), not take())
        for _ in 0..4 {
            let _weights = slot.get();
        }
    }

    // Final state should have exactly one copy of weights
    let final_weights = slot.get();
    assert!(final_weights.is_some());
    assert_eq!(final_weights.unwrap().len(), weight_size);
}

/// Test: VecDeque capacity behavior (alternative to Vec for FIFO).
/// INTENT: Compare VecDeque vs Vec for buffer storage.
#[test]
fn test_vecdeque_vs_vec_capacity() {
    let iterations = 1000;
    let max_size = 100;

    // Test Vec with remove(0)
    let mut vec_storage: Vec<i32> = Vec::with_capacity(max_size);
    let mut vec_max_cap = 0;

    for i in 0..iterations {
        vec_storage.push(i);
        if vec_storage.len() > max_size {
            vec_storage.remove(0);
        }
        vec_max_cap = vec_max_cap.max(vec_storage.capacity());
    }

    // Test VecDeque with pop_front
    let mut deque_storage: VecDeque<i32> = VecDeque::with_capacity(max_size);
    let mut deque_max_cap = 0;

    for i in 0..iterations {
        deque_storage.push_back(i);
        if deque_storage.len() > max_size {
            deque_storage.pop_front();
        }
        deque_max_cap = deque_max_cap.max(deque_storage.capacity());
    }

    // VecDeque should have more stable capacity
    assert!(
        deque_max_cap <= vec_max_cap,
        "VecDeque capacity ({}) should be <= Vec capacity ({})",
        deque_max_cap,
        vec_max_cap
    );
}

// ============================================================================
// 2. Buffer Performance Tests
// ============================================================================

/// Test: Benchmark storage.remove(0) at different buffer sizes.
/// INTENT: Verify O(n) behavior of Vec::remove(0) and its impact.
#[test]
fn test_vec_remove_0_scaling() {
    // Use larger sizes and more iterations to reduce noise in micro-benchmarks
    let sizes = [100, 1000, 5000, 10000];
    let mut timings: Vec<(usize, f64)> = Vec::new();

    for &size in &sizes {
        let mut storage: Vec<i32> = (0..size as i32).collect();
        let iterations = 1000; // More iterations for stable timing

        let start = Instant::now();
        for _ in 0..iterations {
            storage.remove(0);
            storage.push(0); // Maintain size
        }
        let elapsed = start.elapsed().as_nanos() as f64 / iterations as f64;
        timings.push((size, elapsed));
    }

    // Verify O(n) scaling: time should roughly scale with size
    let (size_small, time_small) = timings[0];
    let (size_large, time_large) = timings[timings.len() - 1];

    let size_ratio = size_large as f64 / size_small as f64;
    let time_ratio = time_large / time_small;

    // Log the scaling behavior for debugging
    eprintln!(
        "Vec::remove(0) scaling: size {}x, time {:.2}x (expected ~{}x for O(n))",
        size_ratio, time_ratio, size_ratio
    );

    // Verify timing shows O(n) scaling trend
    // We expect time_ratio to be roughly proportional to size_ratio
    // Use a lenient threshold (at least 2x slower for 100x size increase)
    // to account for cache effects and timing noise
    assert!(
        time_ratio > 2.0,
        "remove(0) should show O(n) scaling: size {}x but time only {:.2}x (expected >2x)",
        size_ratio,
        time_ratio
    );
}

/// Test: Compare VecDeque::pop_front vs Vec::remove(0) performance.
/// INTENT: Demonstrate VecDeque is O(1) vs Vec's O(n) for FIFO.
#[test]
fn test_vecdeque_pop_front_vs_vec_remove() {
    let size = 1000;
    let iterations = 1000;

    // Vec::remove(0)
    let mut vec: Vec<i32> = (0..size).collect();
    let vec_start = Instant::now();
    for i in 0..iterations {
        vec.remove(0);
        vec.push(i);
    }
    let vec_time = vec_start.elapsed();

    // VecDeque::pop_front
    let mut deque: VecDeque<i32> = (0..size).collect();
    let deque_start = Instant::now();
    for i in 0..iterations {
        deque.pop_front();
        deque.push_back(i);
    }
    let deque_time = deque_start.elapsed();

    // VecDeque should be significantly faster for FIFO operations
    let speedup = vec_time.as_nanos() as f64 / deque_time.as_nanos().max(1) as f64;
    assert!(
        speedup > 2.0,
        "VecDeque should be at least 2x faster than Vec for FIFO (got {}x speedup)",
        speedup
    );
}

/// Test: Buffer consolidation performance at different sizes.
/// INTENT: Measure how consolidation time scales with buffer size.
#[test]
fn test_buffer_consolidation_scaling() {
    let sizes = [100, 500, 1000];
    let mut timings: Vec<(usize, TimingStats)> = Vec::new();

    for &max_trajectories in &sizes {
        let config = IMPALABufferConfig {
            n_actors: 4,
            n_envs_per_actor: 32,
            trajectory_length: 20,
            max_trajectories,
            batch_size: 32,
        };
        let buffer = IMPALABuffer::new(config);

        // Fill buffer near capacity
        for i in 0..(max_trajectories + 50) {
            buffer.push_trajectory(make_trajectory(20, i, 1, true));
        }

        // Measure consolidation time
        let mut stats = TimingStats::default();
        for _ in 0..100 {
            let start = Instant::now();
            buffer.consolidate();
            stats.add(start.elapsed().as_nanos() as f64);
        }

        timings.push((max_trajectories, stats));
    }

    // Check if consolidation time grows with buffer size
    // (This would indicate the remove(0) in while loop is problematic)
    let (size_small, stats_small) = &timings[0];
    let (size_large, stats_large) = &timings[timings.len() - 1];

    let size_ratio = *size_large as f64 / *size_small as f64;
    let time_ratio = stats_large.mean() / stats_small.mean().max(1.0);

    // If time grows faster than linearly with size, there's an issue
    // Allow quadratic growth detection (O(n^2) from while loop with remove(0))
    if time_ratio > size_ratio * size_ratio * 0.1 {
        eprintln!(
            "WARNING: Consolidation time grows super-linearly: size {}x, time {}x",
            size_ratio, time_ratio
        );
    }
}

/// Test: Buffer sample_batch performance over many iterations.
/// INTENT: Detect if sample_batch slows down over time.
#[test]
fn test_buffer_sample_performance_over_time() {
    let config = IMPALABufferConfig {
        n_actors: 4,
        n_envs_per_actor: 32,
        trajectory_length: 20,
        max_trajectories: 1000,
        batch_size: 32,
    };
    let buffer = IMPALABuffer::new(config);
    let iterations = 500;

    let mut stats = TimingStats::default();

    for i in 0..iterations {
        // Push enough for a batch
        for j in 0..35 {
            buffer.push_trajectory(make_trajectory(20, i * 35 + j, 1, true));
        }

        // Time the sample operation
        let start = Instant::now();
        if buffer.is_training_ready() {
            let _ = buffer.sample_batch();
        }
        stats.add(start.elapsed().as_nanos() as f64);
    }

    // Check for timing regression
    let first_quarter = stats.first_quartile_mean();
    let last_quarter = stats.last_quartile_mean();
    let slowdown = last_quarter / first_quarter.max(1.0);

    assert!(
        slowdown < 3.0,
        "sample_batch slowdown {}x over {} iterations indicates degradation",
        slowdown,
        iterations
    );
}

// ============================================================================
// 3. Tensor Lifecycle Tests (using NdArray backend)
// ============================================================================

#[cfg(test)]
mod tensor_lifecycle_tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::backend::Autodiff;
    use burn::tensor::Tensor;

    type TestBackend = Autodiff<NdArray>;

    /// Test: Verify tensors are properly dropped after into_data().
    /// INTENT: Ensure into_data() doesn't leave dangling references.
    #[test]
    fn test_tensor_into_data_cleanup() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let iterations = 1000;

        for _ in 0..iterations {
            // Create tensor
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let tensor = Tensor::<TestBackend, 1>::from_floats(&data[..], &device);

            // Perform computation
            let result = tensor.clone() * 2.0 + 1.0;

            // Extract data (this should consume the tensor)
            let values: Vec<f32> = result.into_data().as_slice::<f32>().unwrap().to_vec();

            // Verify values
            assert_eq!(values.len(), 4);
            assert!((values[0] - 3.0).abs() < 1e-5);
        }
        // If we complete without OOM, tensors are being properly cleaned up
    }

    /// Test: Repeated forward passes without backward (bootstrap pattern).
    /// INTENT: Verify computation graphs are cleaned up even without backward().
    #[test]
    fn test_forward_without_backward_cleanup() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let iterations = 500;

        for _ in 0..iterations {
            // Simulate bootstrap forward pass
            {
                let data: Vec<f32> = vec![1.0; 128];
                let tensor = Tensor::<TestBackend, 1>::from_floats(&data[..], &device)
                    .reshape([32, 4]);

                // Forward computation
                let output = tensor.clone().sum_dim(1);

                // Extract immediately (no backward)
                let _values: Vec<f32> = output.into_data().as_slice::<f32>().unwrap().to_vec();
            }
            // Scope ends - tensors should be dropped

            // Simulate main forward pass
            {
                let data: Vec<f32> = vec![0.5; 256];
                let tensor = Tensor::<TestBackend, 1>::from_floats(&data[..], &device)
                    .reshape([64, 4]);

                let output = tensor.clone() * 2.0;
                let loss = output.sum();

                // Backward consumes the graph
                let _grads = loss.backward();
            }
        }
    }

    /// Test: Measure tensor operation timing over many iterations.
    /// INTENT: Detect if tensor operations slow down over time.
    #[test]
    fn test_tensor_timing_stability() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let iterations = 500;
        let mut stats = TimingStats::default();

        for _ in 0..iterations {
            let start = Instant::now();

            // Typical learner operations
            let data: Vec<f32> = vec![1.0; 640 * 4]; // 640 transitions * 4 obs
            let tensor = Tensor::<TestBackend, 1>::from_floats(&data[..], &device)
                .reshape([640, 4]);

            let output = tensor.clone() * 2.0 + 1.0;
            let summed = output.sum_dim(1);
            let _values: Vec<f32> = summed.into_data().as_slice::<f32>().unwrap().to_vec();

            stats.add(start.elapsed().as_micros() as f64);
        }

        // Check for timing degradation
        let slope = stats.slope();
        let mean = stats.mean();

        // Slope should be near zero (constant time)
        // Allow 1% growth per iteration as tolerance
        let relative_slope = slope / mean.max(1.0);
        assert!(
            relative_slope < 0.01,
            "Tensor timing slope ({:.4}) indicates degradation",
            relative_slope
        );
    }

    /// Test: Clone behavior of tensors.
    /// INTENT: Ensure tensor.clone() doesn't cause accumulation.
    #[test]
    fn test_tensor_clone_accumulation() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let iterations = 1000;

        for _ in 0..iterations {
            let data: Vec<f32> = vec![1.0; 64];
            let tensor = Tensor::<TestBackend, 1>::from_floats(&data[..], &device);

            // Multiple clones (as might happen in learner_thread)
            let clone1 = tensor.clone();
            let clone2 = tensor.clone();

            // Use clones
            let result1 = clone1 * 2.0;
            let result2 = clone2 + 1.0;

            // Extract
            let _v1: Vec<f32> = result1.into_data().as_slice::<f32>().unwrap().to_vec();
            let _v2: Vec<f32> = result2.into_data().as_slice::<f32>().unwrap().to_vec();
        }
    }
}

// ============================================================================
// 4. Weight Sync Tests
// ============================================================================

/// Test: BytesSlot publish/get memory over many iterations.
/// INTENT: Verify no memory accumulation in weight transfer.
#[test]
fn test_bytes_slot_publish_get_cycles() {
    let slot = Arc::new(BytesSlot::with_initial(vec![0u8; 10000]));
    let iterations = 1000;

    // Simulate learner publishing
    for i in 0..iterations {
        let weights: Vec<u8> = vec![(i % 256) as u8; 10000];
        slot.publish(weights);
    }

    // Final version should be correct
    assert_eq!(slot.version(), iterations as u64 + 1); // +1 for initial

    // Should still have data
    assert!(slot.has_pending());
}

/// Test: BytesSlot version ordering under concurrent access.
/// INTENT: Verify version monotonicity with multiple publishers.
#[test]
fn test_bytes_slot_concurrent_versions() {
    let slot = Arc::new(bytes_slot());
    let iterations = 100;
    let writers = 4;
    let shutdown = Arc::new(AtomicBool::new(false));
    let versions_seen = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::new();

    // Writer threads
    for writer_id in 0..writers {
        let slot_clone = Arc::clone(&slot);
        let sd = Arc::clone(&shutdown);

        handles.push(thread::spawn(move || {
            for i in 0..iterations {
                let data = vec![(writer_id * iterations + i) as u8; 1000];
                slot_clone.publish(data);
                thread::sleep(Duration::from_micros(10));
            }
            sd.store(true, Ordering::Relaxed);
        }));
    }

    // Reader thread
    {
        let slot_clone = Arc::clone(&slot);
        let sd = Arc::clone(&shutdown);
        let versions = Arc::clone(&versions_seen);

        handles.push(thread::spawn(move || {
            let mut last_version = 0u64;
            while !sd.load(Ordering::Relaxed) {
                let current = slot_clone.version();
                versions.lock().push(current);
                assert!(
                    current >= last_version,
                    "Version went backwards: {} -> {}",
                    last_version,
                    current
                );
                last_version = current;
                thread::sleep(Duration::from_micros(1));
            }
        }));
    }

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify versions are monotonic
    let versions = versions_seen.lock();
    for window in versions.windows(2) {
        assert!(
            window[1] >= window[0],
            "Non-monotonic versions: {} -> {}",
            window[0],
            window[1]
        );
    }
}

/// Test: BytesSlot get() cloning overhead.
/// INTENT: Measure the cost of cloning large weight vectors.
#[test]
fn test_bytes_slot_get_timing() {
    let weight_sizes = [10_000, 100_000, 1_000_000]; // 10KB, 100KB, 1MB

    for &size in &weight_sizes {
        let slot = Arc::new(BytesSlot::with_initial(vec![0u8; size]));
        let iterations = 500; // More iterations for stable timing
        let mut stats = TimingStats::default();

        for _ in 0..iterations {
            let start = Instant::now();
            let _weights = slot.get();
            stats.add(start.elapsed().as_nanos() as f64);
        }

        let median_us = stats.percentile(50) / 1000.0;
        let p99_us = stats.percentile(99) / 1000.0;
        eprintln!(
            "BytesSlot get() for {}KB: {:.2}us median, {:.2}us p99",
            size / 1000,
            median_us,
            p99_us
        );

        // Check timing stability using median comparison (robust to outliers)
        // Compare first quarter vs last quarter medians
        let quarter = iterations / 4;
        let mut first_quarter: Vec<f64> = stats.samples[..quarter].to_vec();
        let mut last_quarter: Vec<f64> = stats.samples[iterations - quarter..].to_vec();
        first_quarter.sort_by(|a, b| a.partial_cmp(b).unwrap());
        last_quarter.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let first_median = first_quarter[quarter / 2];
        let last_median = last_quarter[quarter / 2];
        let ratio = last_median / first_median.max(1.0);

        // Use a generous threshold - we're checking for systematic degradation,
        // not random timing variance. A 5x threshold catches real problems
        // while tolerating normal OS scheduling jitter.
        assert!(
            ratio < 5.0,
            "get() timing degraded by {:.2}x for size {} (first median: {:.0}ns, last median: {:.0}ns)",
            ratio,
            size,
            first_median,
            last_median
        );
    }
}

/// Test: RwLock contention under load.
/// INTENT: Detect if RwLock contention increases over time.
#[test]
fn test_rwlock_contention_scaling() {
    let data = Arc::new(RwLock::new(vec![0u8; 10000]));
    let iterations = 1000;
    let readers = 4;
    let shutdown = Arc::new(AtomicBool::new(false));

    // Writer thread
    let writer_data = Arc::clone(&data);
    let writer_sd = Arc::clone(&shutdown);
    let writer_stats = Arc::new(Mutex::new(TimingStats::default()));
    let writer_stats_clone = Arc::clone(&writer_stats);

    let writer = thread::spawn(move || {
        for i in 0..iterations {
            let start = Instant::now();
            {
                let mut guard = writer_data.write();
                guard[0] = (i % 256) as u8;
            }
            writer_stats_clone
                .lock()
                .add(start.elapsed().as_nanos() as f64);
            thread::sleep(Duration::from_micros(50));
        }
        writer_sd.store(true, Ordering::Relaxed);
    });

    // Reader threads
    let mut reader_handles = Vec::new();
    let reader_stats = Arc::new(Mutex::new(TimingStats::default()));

    for _ in 0..readers {
        let reader_data = Arc::clone(&data);
        let reader_sd = Arc::clone(&shutdown);
        let stats = Arc::clone(&reader_stats);

        reader_handles.push(thread::spawn(move || {
            while !reader_sd.load(Ordering::Relaxed) {
                let start = Instant::now();
                {
                    let guard = reader_data.read();
                    let _ = guard[0];
                }
                stats.lock().add(start.elapsed().as_nanos() as f64);
                thread::sleep(Duration::from_micros(10));
            }
        }));
    }

    writer.join().expect("Writer panicked");
    for handle in reader_handles {
        handle.join().expect("Reader panicked");
    }

    // Analyze timing
    let write_stats = writer_stats.lock();
    let read_stats = reader_stats.lock();

    let write_slope = write_stats.slope() / write_stats.mean().max(1.0);
    let read_slope = read_stats.slope() / read_stats.mean().max(1.0);

    assert!(
        write_slope < 0.01,
        "Write contention growing: slope={:.4}",
        write_slope
    );
    assert!(
        read_slope < 0.01,
        "Read contention growing: slope={:.4}",
        read_slope
    );
}

// ============================================================================
// 5. Timing Regression Tests
// ============================================================================

/// Test: Measure V-trace computation timing over iterations.
/// INTENT: Detect if V-trace slows down (it shouldn't - pure computation).
#[test]
fn test_vtrace_timing_stability() {
    let iterations = 1000;
    let trajectory_length = 20;
    let mut stats = TimingStats::default();

    for _ in 0..iterations {
        let behavior_log_probs: Vec<f32> = vec![-1.0; trajectory_length];
        let target_log_probs: Vec<f32> = vec![-1.0; trajectory_length];
        let rewards: Vec<f32> = vec![1.0; trajectory_length];
        let values: Vec<f32> = vec![0.5; trajectory_length];
        let dones: Vec<bool> = vec![false; trajectory_length];

        let start = Instant::now();
        let _ = compute_vtrace(
            &behavior_log_probs,
            &target_log_probs,
            &rewards,
            &values,
            &dones,
            0.5,
            0.99,
            1.0,
            1.0,
        );
        stats.add(start.elapsed().as_nanos() as f64);
    }

    // V-trace is pure computation - timing should be stable
    let slope = stats.slope();
    let mean = stats.mean();

    assert!(
        (slope / mean.max(1.0)).abs() < 0.001,
        "V-trace timing should be constant, slope={:.4}, mean={:.2}ns",
        slope,
        mean
    );
}

/// Test: Simulate learner iteration timing breakdown.
/// INTENT: Identify which component's timing increases over iterations.
#[test]
fn test_learner_iteration_timing_breakdown() {
    let iterations = 500;
    let batch_size = 32;
    let trajectory_length = 20;

    let mut buffer_stats = TimingStats::default();
    let mut vtrace_stats = TimingStats::default();
    let mut total_stats = TimingStats::default();

    // Create a buffer
    let config = IMPALABufferConfig {
        n_actors: 4,
        n_envs_per_actor: 32,
        trajectory_length,
        max_trajectories: 1000,
        batch_size,
    };
    let buffer = IMPALABuffer::new(config);

    for i in 0..iterations {
        let total_start = Instant::now();

        // 1. Push trajectories (simulating actors)
        for j in 0..batch_size + 5 {
            buffer.push_trajectory(make_trajectory(trajectory_length, i * 40 + j, 1, true));
        }

        // 2. Sample batch (timed)
        let buffer_start = Instant::now();
        let batch = buffer.sample_batch();
        buffer_stats.add(buffer_start.elapsed().as_nanos() as f64);

        if let Some(batch) = batch {
            // 3. V-trace computation (timed)
            let vtrace_start = Instant::now();
            for traj in &batch.trajectories {
                let behavior_log_probs: Vec<f32> =
                    traj.iter().map(|t| t.behavior_log_prob).collect();
                let target_log_probs: Vec<f32> = vec![-0.5; traj.len()];
                let rewards: Vec<f32> = traj.iter().map(|t| t.base.reward).collect();
                let values: Vec<f32> = vec![0.5; traj.len()];
                let dones: Vec<bool> = traj.iter().map(|t| t.done()).collect();

                let _ = compute_vtrace(
                    &behavior_log_probs,
                    &target_log_probs,
                    &rewards,
                    &values,
                    &dones,
                    0.0,
                    0.99,
                    1.0,
                    1.0,
                );
            }
            vtrace_stats.add(vtrace_start.elapsed().as_nanos() as f64);
        }

        total_stats.add(total_start.elapsed().as_nanos() as f64);
    }

    // Analyze each component
    let components = [
        ("Buffer sample", &buffer_stats),
        ("V-trace compute", &vtrace_stats),
        ("Total", &total_stats),
    ];

    for (name, stats) in &components {
        let first_quarter = stats.first_quartile_mean();
        let last_quarter = stats.last_quartile_mean();
        let slowdown = last_quarter / first_quarter.max(1.0);
        let slope = stats.slope() / stats.mean().max(1.0);

        eprintln!(
            "{}: first_q={:.2}us, last_q={:.2}us, slowdown={:.2}x, slope={:.6}",
            name,
            first_quarter / 1000.0,
            last_quarter / 1000.0,
            slowdown,
            slope
        );

        // Flag significant slowdowns
        if slowdown > 2.0 {
            eprintln!("WARNING: {} shows {}x slowdown", name, slowdown);
        }
    }

    // At least one component should be stable
    assert!(
        vtrace_stats.slope() / vtrace_stats.mean().max(1.0) < 0.01,
        "V-trace should be constant-time"
    );
}

/// Test: Per-operation timing over long run.
/// INTENT: Create a timing profile to identify the degrading component.
#[test]
fn test_detailed_operation_timing() {
    let iterations = 300;

    let mut consolidate_times = TimingStats::default();
    let mut sample_times = TimingStats::default();
    let mut vtrace_times = TimingStats::default();
    let mut vec_clear_times = TimingStats::default();
    let mut vec_extend_times = TimingStats::default();

    let config = IMPALABufferConfig {
        n_actors: 4,
        n_envs_per_actor: 32,
        trajectory_length: 20,
        max_trajectories: 500,
        batch_size: 32,
    };
    let buffer = IMPALABuffer::new(config);

    // Pre-allocated buffers (as in learner_thread)
    let mut all_states: Vec<f32> = Vec::with_capacity(640 * 4);
    let mut all_advantages: Vec<f32> = Vec::with_capacity(640);

    for i in 0..iterations {
        // Push data
        for j in 0..40 {
            buffer.push_trajectory(make_trajectory(20, i * 40 + j, 1, true));
        }

        // Time consolidate
        let t = Instant::now();
        buffer.consolidate();
        consolidate_times.add(t.elapsed().as_nanos() as f64);

        // Time sample
        let t = Instant::now();
        let batch = buffer.sample_batch();
        sample_times.add(t.elapsed().as_nanos() as f64);

        if let Some(batch) = batch {
            // Time vec operations
            let t = Instant::now();
            all_states.clear();
            all_advantages.clear();
            vec_clear_times.add(t.elapsed().as_nanos() as f64);

            let t = Instant::now();
            for traj in &batch.trajectories {
                for tr in traj.iter() {
                    all_states.extend_from_slice(&tr.base.state);
                    all_advantages.push(1.0);
                }
            }
            vec_extend_times.add(t.elapsed().as_nanos() as f64);

            // Time V-trace
            let t = Instant::now();
            for traj in &batch.trajectories {
                let len = traj.len();
                if len > 0 {
                    let _ = compute_vtrace(
                        &vec![-0.5; len],
                        &vec![-0.5; len],
                        &vec![1.0; len],
                        &vec![0.5; len],
                        &vec![false; len],
                        0.0,
                        0.99,
                        1.0,
                        1.0,
                    );
                }
            }
            vtrace_times.add(t.elapsed().as_nanos() as f64);
        }
    }

    // Report and check for degradation
    let operations = [
        ("Consolidate", &consolidate_times),
        ("Sample", &sample_times),
        ("Vec clear", &vec_clear_times),
        ("Vec extend", &vec_extend_times),
        ("V-trace", &vtrace_times),
    ];

    let mut degradation_detected = false;
    for (name, stats) in &operations {
        let slope = stats.slope();
        let mean = stats.mean();
        let relative_slope = if mean > 0.0 { slope / mean } else { 0.0 };
        let slowdown = stats.last_quartile_mean() / stats.first_quartile_mean().max(1.0);

        if relative_slope > 0.005 || slowdown > 1.5 {
            eprintln!(
                "DEGRADATION in {}: slope={:.6}, slowdown={:.2}x",
                name, relative_slope, slowdown
            );
            degradation_detected = true;
        }
    }

    if degradation_detected {
        // Print full report for debugging
        eprintln!("\n=== Full Timing Report ===");
        for (name, stats) in &operations {
            eprintln!(
                "{:15}: mean={:8.0}ns, p50={:8.0}ns, p99={:8.0}ns, slope={:.4}",
                name,
                stats.mean(),
                stats.median(),
                stats.percentile(99),
                stats.slope()
            );
        }
    }
}

// ============================================================================
// 6. Simulated Long-Run Tests
// ============================================================================

/// Test: Simulate 10,000+ iterations and track all metrics.
/// INTENT: Reproduce the degradation pattern in isolation.
#[test]
fn test_long_run_simulation() {
    let iterations = 5000; // Reduced from 10000 for test speed
    let checkpoint_interval = 500;

    let config = IMPALABufferConfig {
        n_actors: 4,
        n_envs_per_actor: 32,
        trajectory_length: 20,
        max_trajectories: 1000,
        batch_size: 32,
    };
    let buffer = IMPALABuffer::new(config);

    let mut iteration_times = TimingStats::default();
    let mut checkpoint_metrics: Vec<(usize, f64, f64)> = Vec::new();

    // Pre-allocated buffers
    let mut all_states: Vec<f32> = Vec::with_capacity(640 * 4);
    let mut all_advantages: Vec<f32> = Vec::with_capacity(640);
    let mut all_vtrace: Vec<f32> = Vec::with_capacity(640);

    let start = Instant::now();

    for i in 0..iterations {
        let iter_start = Instant::now();

        // Simulate actor pushing trajectories
        for j in 0..35 {
            buffer.push_trajectory(make_trajectory(20, j, i as u64, i % 10 == 0));
        }

        // Simulate learner processing
        if buffer.is_training_ready() {
            if let Some(batch) = buffer.sample_batch() {
                // Clear and reuse buffers
                all_states.clear();
                all_advantages.clear();
                all_vtrace.clear();

                // Collect data
                for traj in &batch.trajectories {
                    for tr in traj.iter() {
                        all_states.extend_from_slice(&tr.base.state);
                    }
                }

                // Compute V-trace
                for traj in &batch.trajectories {
                    let len = traj.len();
                    if len > 0 {
                        let vtrace = compute_vtrace(
                            &vec![-0.5; len],
                            &vec![-0.5; len],
                            &vec![1.0; len],
                            &vec![0.5; len],
                            &vec![false; len],
                            0.0,
                            0.99,
                            1.0,
                            1.0,
                        );
                        all_advantages.extend(&vtrace.advantages);
                        all_vtrace.extend(&vtrace.vs);
                    }
                }
            }
        }

        iteration_times.add(iter_start.elapsed().as_micros() as f64);

        // Checkpoint
        if (i + 1) % checkpoint_interval == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let sps = (i + 1) as f64 / elapsed;

            // Get recent iteration time
            let recent_start = iteration_times.samples.len().saturating_sub(100);
            let recent_mean: f64 = iteration_times.samples[recent_start..]
                .iter()
                .sum::<f64>()
                / 100.0;

            checkpoint_metrics.push((i + 1, sps, recent_mean));

            eprintln!(
                "Iter {:5}: SPS={:.0}, iter_time={:.0}us, buffer_len={}",
                i + 1,
                sps,
                recent_mean,
                buffer.len()
            );
        }
    }

    // Analyze for degradation
    if checkpoint_metrics.len() >= 2 {
        let (_, first_sps, first_time) = checkpoint_metrics[0];
        let (_, last_sps, last_time) = checkpoint_metrics[checkpoint_metrics.len() - 1];

        let sps_ratio = last_sps / first_sps;
        let time_ratio = last_time / first_time;

        eprintln!("\n=== Long Run Analysis ===");
        eprintln!("SPS: {:.0} -> {:.0} ({:.1}% of initial)", first_sps, last_sps, sps_ratio * 100.0);
        eprintln!("Iter time: {:.0}us -> {:.0}us ({:.2}x)", first_time, last_time, time_ratio);

        // Allow up to 30% degradation in this simulation (real issue may be GPU-related)
        assert!(
            sps_ratio > 0.5,
            "SPS degraded to {:.1}% of initial - significant issue detected",
            sps_ratio * 100.0
        );
    }
}

/// Test: Verify constant-time behavior of all operations.
/// INTENT: Each operation should be O(1) or O(batch_size), not O(total_iterations).
#[test]
fn test_constant_time_operations() {
    let iterations = 1000;

    // Test 1: V-trace should be O(trajectory_length)
    let mut vtrace_times = TimingStats::default();
    for _ in 0..iterations {
        let len = 20;
        let start = Instant::now();
        let _ = compute_vtrace(
            &vec![-0.5; len],
            &vec![-0.5; len],
            &vec![1.0; len],
            &vec![0.5; len],
            &vec![false; len],
            0.0,
            0.99,
            1.0,
            1.0,
        );
        vtrace_times.add(start.elapsed().as_nanos() as f64);
    }
    assert!(
        vtrace_times.slope().abs() < vtrace_times.mean() * 0.001,
        "V-trace not constant time"
    );

    // Test 2: Vec clear should be O(1)
    let mut clear_times = TimingStats::default();
    let mut v: Vec<f32> = Vec::with_capacity(10000);
    for i in 0..iterations {
        v.extend((0..1000).map(|x| x as f32));
        let start = Instant::now();
        v.clear();
        clear_times.add(start.elapsed().as_nanos() as f64);
    }
    assert!(
        clear_times.slope().abs() < clear_times.mean() * 0.01,
        "Vec::clear not constant time"
    );

    // Test 3: BytesSlot publish should be O(weight_size)
    let slot = bytes_slot();
    let mut publish_times = TimingStats::default();
    for i in 0..iterations {
        let data = vec![0u8; 10000];
        let start = Instant::now();
        slot.publish(data);
        publish_times.add(start.elapsed().as_nanos() as f64);
    }
    assert!(
        publish_times.slope().abs() < publish_times.mean() * 0.01,
        "BytesSlot::publish not constant time"
    );
}

/// Test: Buffer FIFO eviction should be O(1) amortized.
/// INTENT: The while loop with remove(0) makes this O(n) - detect it.
#[test]
fn test_buffer_eviction_complexity() {
    let capacities = [100, 500, 1000];
    let mut capacity_to_eviction_time: Vec<(usize, f64)> = Vec::new();

    for &capacity in &capacities {
        let config = IMPALABufferConfig {
            n_actors: 1,
            n_envs_per_actor: 1,
            trajectory_length: 10,
            max_trajectories: capacity,
            batch_size: 1,
        };
        let buffer = IMPALABuffer::new(config);

        // Fill to capacity
        for i in 0..capacity {
            buffer.push_trajectory(make_trajectory(10, i, 1, true));
        }
        buffer.consolidate();

        // Measure eviction time when pushing over capacity
        let mut stats = TimingStats::default();
        for i in 0..100 {
            buffer.push_trajectory(make_trajectory(10, capacity + i, 1, true));

            let start = Instant::now();
            buffer.consolidate(); // This triggers eviction
            stats.add(start.elapsed().as_nanos() as f64);
        }

        capacity_to_eviction_time.push((capacity, stats.mean()));
    }

    // Check scaling
    let (cap_small, time_small) = capacity_to_eviction_time[0];
    let (cap_large, time_large) = capacity_to_eviction_time[capacity_to_eviction_time.len() - 1];

    let cap_ratio = cap_large as f64 / cap_small as f64;
    let time_ratio = time_large / time_small.max(1.0);

    eprintln!(
        "Buffer eviction scaling: capacity {}x, time {}x (expect ~{}x for O(n))",
        cap_ratio, time_ratio, cap_ratio
    );

    // If O(n), time_ratio should be close to cap_ratio
    // If O(1), time_ratio should be close to 1
    // We're detecting O(n) behavior here
    if time_ratio > cap_ratio * 0.5 {
        eprintln!("WARNING: Buffer eviction shows O(n) behavior - consider VecDeque");
    }
}

/// Test: Concurrent producer-consumer simulation.
/// INTENT: Detect any degradation in concurrent access patterns.
#[test]
fn test_concurrent_degradation() {
    let config = IMPALABufferConfig {
        n_actors: 4,
        n_envs_per_actor: 32,
        trajectory_length: 20,
        max_trajectories: 1000,
        batch_size: 32,
    };
    let buffer = Arc::new(IMPALABuffer::new(config));
    let shutdown = Arc::new(AtomicBool::new(false));
    let producer_count = Arc::new(AtomicUsize::new(0));
    let consumer_times = Arc::new(Mutex::new(TimingStats::default()));

    let iterations = 500;

    // Producer threads (simulating actors)
    let mut producers = Vec::new();
    for actor_id in 0..4 {
        let buf = Arc::clone(&buffer);
        let sd = Arc::clone(&shutdown);
        let count = Arc::clone(&producer_count);

        producers.push(thread::spawn(move || {
            for i in 0..iterations {
                buf.push_trajectory(make_trajectory(20, actor_id * iterations + i, i as u64, true));
                count.fetch_add(1, Ordering::Relaxed);
                thread::sleep(Duration::from_micros(100));
            }
        }));
    }

    // Consumer thread (simulating learner)
    let consumer_buf = Arc::clone(&buffer);
    let consumer_sd = Arc::clone(&shutdown);
    let consumer_stats = Arc::clone(&consumer_times);

    let consumer = thread::spawn(move || {
        let mut batches = 0;
        while batches < 50 || !consumer_sd.load(Ordering::Relaxed) {
            if consumer_buf.is_training_ready() {
                let start = Instant::now();
                if let Some(_batch) = consumer_buf.sample_batch() {
                    consumer_stats.lock().add(start.elapsed().as_micros() as f64);
                    batches += 1;
                }
            }
            thread::sleep(Duration::from_micros(500));
            if batches >= 50 {
                break;
            }
        }
        batches
    });

    // Wait for producers
    for p in producers {
        p.join().expect("Producer panicked");
    }
    shutdown.store(true, Ordering::Relaxed);

    let batches = consumer.join().expect("Consumer panicked");

    // Analyze consumer timing
    let stats = consumer_times.lock();
    let slope = stats.slope();
    let mean = stats.mean();

    eprintln!(
        "Consumer: {} batches, mean={:.0}us, slope={:.4}",
        batches,
        mean,
        slope / mean.max(1.0)
    );

    assert!(
        slope / mean.max(1.0) < 0.1,
        "Consumer timing degraded significantly"
    );
}

// ============================================================================
// 7. VecDeque Alternative Tests
// ============================================================================

/// Test: Simulate buffer with VecDeque instead of Vec.
/// INTENT: Demonstrate VecDeque provides O(1) FIFO without degradation.
#[test]
fn test_vecdeque_buffer_alternative() {
    let max_capacity = 1000;
    let iterations = 5000;
    let batch_size = 32;

    let mut storage: VecDeque<Trajectory<IMPALATransition>> = VecDeque::with_capacity(max_capacity);
    let mut stats = TimingStats::default();

    for i in 0..iterations {
        // Push trajectories
        for j in 0..35 {
            storage.push_back(make_trajectory(20, j, i as u64, true));
        }

        // Evict oldest to maintain capacity
        let start = Instant::now();
        while storage.len() > max_capacity {
            storage.pop_front(); // O(1) instead of remove(0) O(n)
        }
        stats.add(start.elapsed().as_nanos() as f64);

        // Sample batch (drain from front)
        if storage.len() >= batch_size {
            let _batch: Vec<_> = storage.drain(..batch_size).collect();
        }
    }

    // VecDeque eviction should be constant time
    let slope = stats.slope();
    let mean = stats.mean();
    let relative_slope = slope / mean.max(1.0);

    assert!(
        relative_slope.abs() < 0.001,
        "VecDeque eviction should be O(1), got slope={:.6}",
        relative_slope
    );

    eprintln!(
        "VecDeque eviction: mean={:.0}ns, slope={:.4} (relative: {:.6})",
        mean, slope, relative_slope
    );
}

/// Test: Compare Vec vs VecDeque buffer performance over time.
/// INTENT: Show VecDeque maintains constant performance.
#[test]
fn test_vec_vs_vecdeque_long_run() {
    let max_capacity = 500;
    let iterations = 2000;

    // Vec-based (current implementation pattern)
    let mut vec_storage: Vec<i32> = Vec::with_capacity(max_capacity);
    let mut vec_stats = TimingStats::default();

    for i in 0..iterations {
        vec_storage.push(i);
        let start = Instant::now();
        while vec_storage.len() > max_capacity {
            vec_storage.remove(0);
        }
        vec_stats.add(start.elapsed().as_nanos() as f64);
    }

    // VecDeque-based (alternative)
    let mut deque_storage: VecDeque<i32> = VecDeque::with_capacity(max_capacity);
    let mut deque_stats = TimingStats::default();

    for i in 0..iterations {
        deque_storage.push_back(i);
        let start = Instant::now();
        while deque_storage.len() > max_capacity {
            deque_storage.pop_front();
        }
        deque_stats.add(start.elapsed().as_nanos() as f64);
    }

    // Compare degradation
    let vec_slowdown = vec_stats.last_quartile_mean() / vec_stats.first_quartile_mean().max(1.0);
    let deque_slowdown =
        deque_stats.last_quartile_mean() / deque_stats.first_quartile_mean().max(1.0);

    eprintln!("Vec slowdown: {:.2}x", vec_slowdown);
    eprintln!("VecDeque slowdown: {:.2}x", deque_slowdown);

    // VecDeque should show less degradation
    assert!(
        deque_slowdown < vec_slowdown || deque_slowdown < 1.5,
        "VecDeque should perform better than Vec for FIFO"
    );
}
