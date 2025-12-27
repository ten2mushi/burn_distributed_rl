//! WGPU Pipeline SPS Degradation Tests
//!
//! This module provides comprehensive tests to diagnose SPS (Steps-Per-Second)
//! degradation in the distributed IMPALA implementation using the WGPU backend.
//!
//! # Observed Degradation Pattern
//!
//! ```text
//! Steps:    44,384 | SPS: 23,000 (start)
//! Steps:   173,312 | SPS: 21,190 (slight drop)
//! Steps:   287,936 | SPS: 17,591 (accelerating)
//! Steps:   429,440 | SPS: 13,983 (continuing)
//! Steps:   567,840 | SPS: 11,087 (continuing)
//! Steps:   642,720 | SPS:  9,800 (57% degradation)
//! ```
//!
//! # WGPU-Specific Hypotheses
//!
//! 1. **Command Buffer Accumulation**: WGPU batches GPU commands that may not
//!    be flushed promptly, causing accumulation.
//!
//! 2. **Texture/Buffer Leaks**: GPU resources not being freed properly.
//!
//! 3. **Staging Buffer Growth**: CPU<->GPU transfer buffers accumulating.
//!
//! 4. **Computation Graph Retention**: Even after backward(), graph nodes may
//!    be retained in Autodiff<Wgpu>.
//!
//! 5. **Model Weight Serialization**: `recorder.record()` may create
//!    accumulating GPU resources.
//!
//! 6. **Tensor::from_floats() Overhead**: Tensor creation from slices may
//!    degrade over iterations.
//!
//! 7. **Device Sync Missing**: WGPU needs explicit synchronization to free
//!    resources (device.poll).
//!
//! # Test Strategy
//!
//! These tests use the actual WGPU backend (not NdArray) to detect GPU-specific
//! issues. Each test measures timing over many iterations and detects degradation
//! using linear regression slope analysis.
//!
//! **IMPORTANT**: These tests require WGPU to be available. They are marked with
//! `#[ignore]` for CI environments without GPU access. Run with:
//! `cargo test wgpu_pipeline_sps -- --ignored --test-threads=1`

use std::time::Instant;

// ============================================================================
// NOTE ON WGPU + AUTODIFF SERIALIZATION
// ============================================================================
//
// When using `Autodiff<Wgpu>`, model serialization requires special handling:
// 1. Call `.valid()` to get the inner (non-autodiff) model
// 2. Then call `.into_record()` on the inner model
//
// This is because the Autodiff wrapper adds additional type complexity that
// exceeds Rust's recursion limit for trait resolution.
//
// In production code (distributed_impala_runner.rs), this is handled by:
// `model.clone().valid().into_record()`
// ============================================================================

// ============================================================================
// Timing Statistics Helper
// ============================================================================

/// Timing statistics tracker for detecting degradation patterns.
#[derive(Default, Clone)]
pub struct TimingStats {
    samples: Vec<f64>,
}

impl TimingStats {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { samples: Vec::with_capacity(cap) }
    }

    pub fn add(&mut self, elapsed_us: f64) {
        self.samples.push(elapsed_us);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    pub fn median(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }

    pub fn percentile(&self, p: usize) -> f64 {
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
    pub fn slope(&self) -> f64 {
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

    /// Get first quartile mean (first 25% of samples).
    pub fn first_quartile_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let n = (self.samples.len() / 4).max(1);
        self.samples[..n].iter().sum::<f64>() / n as f64
    }

    /// Get last quartile mean (last 25% of samples).
    pub fn last_quartile_mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let n = (self.samples.len() / 4).max(1);
        let start = self.samples.len() - n;
        self.samples[start..].iter().sum::<f64>() / n as f64
    }

    /// Compute slowdown ratio (last_quarter_mean / first_quarter_mean).
    pub fn slowdown_ratio(&self) -> f64 {
        let first = self.first_quartile_mean();
        let last = self.last_quartile_mean();
        if first < 1e-10 {
            return 1.0;
        }
        last / first
    }

    /// Compute relative slope (slope / mean).
    /// Values > 0.001 indicate noticeable degradation.
    pub fn relative_slope(&self) -> f64 {
        let mean = self.mean();
        if mean < 1e-10 {
            return 0.0;
        }
        self.slope() / mean
    }

    /// Get samples at specific window for trend analysis.
    pub fn window_mean(&self, start_pct: usize, end_pct: usize) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let start_idx = (self.samples.len() * start_pct / 100).min(self.samples.len() - 1);
        let end_idx = (self.samples.len() * end_pct / 100).min(self.samples.len());
        if start_idx >= end_idx {
            return 0.0;
        }
        let slice = &self.samples[start_idx..end_idx];
        slice.iter().sum::<f64>() / slice.len() as f64
    }
}

// ============================================================================
// WGPU Backend Tests
// ============================================================================

mod wgpu_tests {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;
    use burn::module::{AutodiffModule, Module};
    use burn::nn::{Linear, LinearConfig};
    use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::activation::relu;
    use burn::tensor::Tensor;

    // Type alias for inner backend (used for serialization)
    type InnerB = Wgpu;

    type B = Autodiff<Wgpu>;

    // ========================================================================
    // Test Model (Minimal Actor-Critic for CartPole)
    // ========================================================================

    /// Minimal actor-critic network for testing WGPU operations.
    #[derive(Module, Debug)]
    pub struct TestNet<B: burn::tensor::backend::Backend> {
        shared: Linear<B>,
        policy_head: Linear<B>,
        value_head: Linear<B>,
    }

    impl<B: burn::tensor::backend::Backend> TestNet<B> {
        pub fn new(device: &B::Device, obs_size: usize, hidden_size: usize, n_actions: usize) -> Self {
            Self {
                shared: LinearConfig::new(obs_size, hidden_size).init(device),
                policy_head: LinearConfig::new(hidden_size, n_actions).init(device),
                value_head: LinearConfig::new(hidden_size, 1).init(device),
            }
        }

        pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
            let hidden = relu(self.shared.forward(x));
            let logits = self.policy_head.forward(hidden.clone());
            let values = self.value_head.forward(hidden);
            (logits, values)
        }
    }

    // ========================================================================
    // 1. Tensor Creation Timing Tests
    // ========================================================================

    /// Test: Measure Tensor::from_floats() timing over 10,000+ iterations.
    /// INTENT: Detect if CPU->GPU tensor creation slows down over time.
    ///
    /// WGPU creates staging buffers for CPU->GPU transfers. If these are not
    /// properly recycled or freed, the operation will slow down.
    #[test]
    #[ignore] // Requires WGPU hardware
    fn test_tensor_creation_timing_over_iterations() {
        let device = WgpuDevice::default();
        let iterations = 10_000;
        let batch_size = 640; // Typical learner batch
        let obs_size = 4; // CartPole

        let mut stats = TimingStats::with_capacity(iterations);

        // Pre-allocate data buffer (reused each iteration)
        let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];

        for _ in 0..iterations {
            let start = Instant::now();

            // This is the exact pattern from learner_thread
            let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            // Force synchronization to measure actual GPU operation
            let _ = tensor.into_data();

            stats.add(start.elapsed().as_micros() as f64);
        }

        // Analysis
        let relative_slope = stats.relative_slope();
        let slowdown = stats.slowdown_ratio();

        eprintln!("\n=== Tensor Creation Timing ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("First quartile mean: {:.2} us", stats.first_quartile_mean());
        eprintln!("Last quartile mean:  {:.2} us", stats.last_quartile_mean());
        eprintln!("Slowdown ratio: {:.3}x", slowdown);
        eprintln!("Relative slope: {:.6}", relative_slope);

        // PASS criteria: slowdown < 1.5x and relative_slope < 0.001
        assert!(
            slowdown < 1.5,
            "Tensor creation slowed down by {:.2}x over {} iterations - WGPU staging buffer issue?",
            slowdown,
            iterations
        );
        assert!(
            relative_slope < 0.001,
            "Tensor creation has positive timing trend (slope={:.6}) - resource accumulation?",
            relative_slope
        );
    }

    /// Test: Tensor creation with varying sizes.
    /// INTENT: Detect if larger tensors cause disproportionate slowdown.
    #[test]
    #[ignore]
    fn test_tensor_creation_size_scaling() {
        let device = WgpuDevice::default();
        let iterations = 1000;

        let sizes = [
            (64, 4),      // Small: 256 floats
            (640, 4),     // Medium: 2560 floats (typical batch)
            (640, 128),   // Large: 81920 floats
        ];

        for (batch_size, obs_size) in sizes {
            let mut stats = TimingStats::with_capacity(iterations);
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];

            for _ in 0..iterations {
                let start = Instant::now();
                let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                    .reshape([batch_size, obs_size]);
                let _ = tensor.into_data();
                stats.add(start.elapsed().as_micros() as f64);
            }

            eprintln!(
                "Size [{} x {}]: mean={:.2}us, slowdown={:.3}x, slope={:.6}",
                batch_size, obs_size,
                stats.mean(), stats.slowdown_ratio(), stats.relative_slope()
            );

            assert!(
                stats.slowdown_ratio() < 1.5,
                "Size [{} x {}] tensor creation degraded by {:.2}x",
                batch_size, obs_size, stats.slowdown_ratio()
            );
        }
    }

    // ========================================================================
    // 2. Forward Pass Timing Tests
    // ========================================================================

    /// Test: Model forward pass timing over 5,000+ iterations.
    /// INTENT: Detect GPU command buffer accumulation in forward passes.
    #[test]
    #[ignore]
    fn test_forward_pass_timing_over_iterations() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);
        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            // Create input tensor
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let start = Instant::now();
            let (logits, values) = model.forward(input);

            // Force sync to measure actual GPU execution
            let _ = logits.into_data();
            let _ = values.into_data();

            stats.add(start.elapsed().as_micros() as f64);
        }

        let relative_slope = stats.relative_slope();
        let slowdown = stats.slowdown_ratio();

        eprintln!("\n=== Forward Pass Timing ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("First quartile mean: {:.2} us", stats.first_quartile_mean());
        eprintln!("Last quartile mean:  {:.2} us", stats.last_quartile_mean());
        eprintln!("Slowdown ratio: {:.3}x", slowdown);
        eprintln!("Relative slope: {:.6}", relative_slope);

        assert!(
            slowdown < 1.5,
            "Forward pass slowed down by {:.2}x - GPU command accumulation?",
            slowdown
        );
        assert!(
            relative_slope < 0.001,
            "Forward pass has timing trend (slope={:.6}) - resource leak?",
            relative_slope
        );
    }

    /// Test: Multiple forward passes per iteration (bootstrap + main pattern).
    /// INTENT: Detect if the two-forward-pass pattern causes accumulation.
    #[test]
    #[ignore]
    fn test_dual_forward_pass_pattern() {
        let device = WgpuDevice::default();
        let iterations = 2_000;
        let main_batch_size = 640;
        let bootstrap_batch_size = 20; // Subset for bootstrap
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);
        let mut total_stats = TimingStats::with_capacity(iterations);
        let mut bootstrap_stats = TimingStats::with_capacity(iterations);
        let mut main_stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let iter_start = Instant::now();

            // Bootstrap forward pass (isolated scope)
            {
                let bootstrap_data: Vec<f32> = vec![0.5f32; bootstrap_batch_size * obs_size];
                let bootstrap_input = Tensor::<B, 1>::from_floats(bootstrap_data.as_slice(), &device)
                    .reshape([bootstrap_batch_size, obs_size]);

                let start = Instant::now();
                let (_, values) = model.forward(bootstrap_input);
                let _ = values.into_data(); // Extract immediately
                bootstrap_stats.add(start.elapsed().as_micros() as f64);
            }

            // Main forward pass
            {
                let main_data: Vec<f32> = vec![0.5f32; main_batch_size * obs_size];
                let main_input = Tensor::<B, 1>::from_floats(main_data.as_slice(), &device)
                    .reshape([main_batch_size, obs_size]);

                let start = Instant::now();
                let (logits, values) = model.forward(main_input);
                let _ = logits.into_data();
                let _ = values.into_data();
                main_stats.add(start.elapsed().as_micros() as f64);
            }

            total_stats.add(iter_start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Dual Forward Pass Pattern ===");
        eprintln!("Bootstrap: mean={:.2}us, slowdown={:.3}x",
            bootstrap_stats.mean(), bootstrap_stats.slowdown_ratio());
        eprintln!("Main:      mean={:.2}us, slowdown={:.3}x",
            main_stats.mean(), main_stats.slowdown_ratio());
        eprintln!("Total:     mean={:.2}us, slowdown={:.3}x",
            total_stats.mean(), total_stats.slowdown_ratio());

        assert!(
            bootstrap_stats.slowdown_ratio() < 1.5,
            "Bootstrap forward pass degraded by {:.2}x",
            bootstrap_stats.slowdown_ratio()
        );
        assert!(
            main_stats.slowdown_ratio() < 1.5,
            "Main forward pass degraded by {:.2}x",
            main_stats.slowdown_ratio()
        );
    }

    // ========================================================================
    // 3. Backward Pass Timing Tests
    // ========================================================================

    /// Test: Backward pass timing over 5,000+ iterations.
    /// INTENT: Verify computation graph cleanup after backward().
    #[test]
    #[ignore]
    fn test_backward_pass_timing_over_iterations() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);
        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let (logits, values) = model.forward(input);

            // Compute a loss (sum of all outputs)
            let loss = logits.sum() + values.sum();

            let start = Instant::now();
            let _grads = loss.backward();
            stats.add(start.elapsed().as_micros() as f64);
        }

        let relative_slope = stats.relative_slope();
        let slowdown = stats.slowdown_ratio();

        eprintln!("\n=== Backward Pass Timing ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("First quartile mean: {:.2} us", stats.first_quartile_mean());
        eprintln!("Last quartile mean:  {:.2} us", stats.last_quartile_mean());
        eprintln!("Slowdown ratio: {:.3}x", slowdown);
        eprintln!("Relative slope: {:.6}", relative_slope);

        assert!(
            slowdown < 1.5,
            "Backward pass slowed down by {:.2}x - computation graph retention?",
            slowdown
        );
        assert!(
            relative_slope < 0.001,
            "Backward pass has timing trend (slope={:.6}) - graph not freed?",
            relative_slope
        );
    }

    /// Test: Forward without backward (orphaned computation graphs).
    /// INTENT: Detect if unconsumed computation graphs cause memory growth.
    ///
    /// This simulates the bootstrap forward pass pattern where no backward()
    /// is called.
    #[test]
    #[ignore]
    fn test_forward_without_backward_accumulation() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let batch_size = 32;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);
        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let start = Instant::now();
            let (_, values) = model.forward(input);

            // Extract values immediately WITHOUT backward
            // This pattern is used for bootstrap value computation
            let _ = values.into_data();

            stats.add(start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Forward Without Backward ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("Slowdown ratio: {:.3}x", stats.slowdown_ratio());
        eprintln!("Relative slope: {:.6}", stats.relative_slope());

        // This is the critical test - orphaned graphs should NOT accumulate
        assert!(
            stats.slowdown_ratio() < 1.5,
            "Forward without backward caused {:.2}x slowdown - orphaned graph accumulation!",
            stats.slowdown_ratio()
        );
    }

    // ========================================================================
    // 4. into_data() Extraction Timing Tests
    // ========================================================================

    /// Test: Tensor::into_data() timing for GPU->CPU transfer.
    /// INTENT: Detect staging buffer issues in data extraction.
    #[test]
    #[ignore]
    fn test_into_data_extraction_timing() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let batch_size = 640;
        let obs_size = 4;

        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            // Create tensor on GPU
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            // Perform some GPU computation
            let result = tensor * 2.0 + 1.0;

            // Time the GPU->CPU extraction
            let start = Instant::now();
            let _extracted: Vec<f32> = result.into_data().as_slice::<f32>().unwrap().to_vec();
            stats.add(start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== into_data() Extraction Timing ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("First quartile mean: {:.2} us", stats.first_quartile_mean());
        eprintln!("Last quartile mean:  {:.2} us", stats.last_quartile_mean());
        eprintln!("Slowdown ratio: {:.3}x", stats.slowdown_ratio());
        eprintln!("Relative slope: {:.6}", stats.relative_slope());

        assert!(
            stats.slowdown_ratio() < 1.5,
            "into_data() slowed down by {:.2}x - staging buffer issue?",
            stats.slowdown_ratio()
        );
    }

    /// Test: Repeated clone().into_data() pattern.
    /// INTENT: Test the exact pattern used in learner_thread for value extraction.
    #[test]
    #[ignore]
    fn test_clone_into_data_pattern() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let batch_size = 640;

        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size];
            let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device);

            // This is the exact pattern from learner_thread:
            // values_flat.clone().into_data().as_slice()
            let start = Instant::now();
            let cloned = tensor.clone();
            let _values: Vec<f32> = cloned.into_data().as_slice::<f32>().unwrap().to_vec();
            stats.add(start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Clone + into_data Pattern ===");
        eprintln!("Slowdown ratio: {:.3}x", stats.slowdown_ratio());

        assert!(
            stats.slowdown_ratio() < 1.5,
            "clone().into_data() pattern degraded by {:.2}x",
            stats.slowdown_ratio()
        );
    }

    // ========================================================================
    // 5. Weight Serialization Timing Tests
    // ========================================================================
    //
    // NOTE: These tests use the raw Wgpu backend (not Autodiff<Wgpu>) for
    // serialization because Burn's type system has trait bound complexity
    // issues when serializing Autodiff-wrapped models. The raw backend tests
    // still capture GPU resource behavior for serialization operations.

    /// Test: Model weight serialization timing (recorder.record()).
    /// INTENT: Detect if serialization creates accumulating GPU resources.
    #[test]
    #[ignore]
    fn test_weight_serialization_timing() {
        let device = WgpuDevice::default();
        let iterations = 2_000;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        // Use raw Wgpu backend for serialization tests
        let model = TestNet::<InnerB>::new(&device, obs_size, hidden_size, n_actions);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();

            // Serialize model weights to bytes
            let bytes = recorder
                .record(model.clone().into_record(), ())
                .expect("Serialization failed");

            stats.add(start.elapsed().as_micros() as f64);

            // Verify bytes are reasonable size
            assert!(bytes.len() > 1000, "Serialized weights should be substantial");
        }

        eprintln!("\n=== Weight Serialization Timing ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("First quartile mean: {:.2} us", stats.first_quartile_mean());
        eprintln!("Last quartile mean:  {:.2} us", stats.last_quartile_mean());
        eprintln!("Slowdown ratio: {:.3}x", stats.slowdown_ratio());
        eprintln!("Relative slope: {:.6}", stats.relative_slope());

        assert!(
            stats.slowdown_ratio() < 1.5,
            "Weight serialization slowed down by {:.2}x",
            stats.slowdown_ratio()
        );
    }

    /// Test: Weight serialization timing with forward pass (simulated training).
    /// INTENT: Test if serialization after forward pass causes issues.
    /// Note: Uses raw Wgpu (no autodiff) to avoid trait bound issues.
    #[test]
    #[ignore]
    fn test_serialization_after_forward_pass() {
        let device = WgpuDevice::default();
        let iterations = 1_000;
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        // Use raw Wgpu for serialization-focused test
        let model = TestNet::<InnerB>::new(&device, obs_size, hidden_size, n_actions);
        let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();

        let mut forward_stats = TimingStats::with_capacity(iterations);
        let mut serialize_stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<InnerB, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            // Forward
            let fwd_start = Instant::now();
            let (logits, values) = model.forward(input);
            // Force sync
            let _ = logits.into_data();
            let _ = values.into_data();
            forward_stats.add(fwd_start.elapsed().as_micros() as f64);

            // Serialize (after forward)
            let ser_start = Instant::now();
            let _bytes = recorder.record(model.clone().into_record(), ());
            serialize_stats.add(ser_start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Forward + Serialize Timing ===");
        eprintln!("Forward:   slowdown={:.3}x", forward_stats.slowdown_ratio());
        eprintln!("Serialize: slowdown={:.3}x", serialize_stats.slowdown_ratio());

        assert!(
            serialize_stats.slowdown_ratio() < 1.5,
            "Post-forward serialization degraded by {:.2}x",
            serialize_stats.slowdown_ratio()
        );
    }

    /// Test: Combined training + serialization with Autodiff backend.
    /// INTENT: Full training loop timing without actual weight update.
    #[test]
    #[ignore]
    fn test_training_loop_timing_components() {
        let device = WgpuDevice::default();
        let iterations = 1_000;
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);

        let mut forward_stats = TimingStats::with_capacity(iterations);
        let mut backward_stats = TimingStats::with_capacity(iterations);
        let mut total_stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let iter_start = Instant::now();

            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            // Forward
            let fwd_start = Instant::now();
            let (logits, values) = model.forward(input);
            let loss = logits.sum() + values.sum();
            forward_stats.add(fwd_start.elapsed().as_micros() as f64);

            // Backward
            let bwd_start = Instant::now();
            let _grads = loss.backward();
            backward_stats.add(bwd_start.elapsed().as_micros() as f64);

            total_stats.add(iter_start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Training Loop Component Timing ===");
        eprintln!("Forward:   mean={:.2}us, slowdown={:.3}x",
            forward_stats.mean(), forward_stats.slowdown_ratio());
        eprintln!("Backward:  mean={:.2}us, slowdown={:.3}x",
            backward_stats.mean(), backward_stats.slowdown_ratio());
        eprintln!("Total:     mean={:.2}us, slowdown={:.3}x",
            total_stats.mean(), total_stats.slowdown_ratio());

        assert!(
            total_stats.slowdown_ratio() < 1.5,
            "Training loop degraded by {:.2}x",
            total_stats.slowdown_ratio()
        );
    }

    // ========================================================================
    // 6. Device Synchronization Tests
    // ========================================================================

    /// Test: Compare timing with and without explicit device sync.
    /// INTENT: Determine if explicit sync points help with resource cleanup.
    ///
    /// Note: WGPU in Burn may not expose direct device.poll(), but we can
    /// force synchronization through data extraction.
    #[test]
    #[ignore]
    fn test_explicit_sync_impact() {
        let device = WgpuDevice::default();
        let iterations = 2_000;
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);

        // Test WITHOUT sync points (just drop tensors)
        let mut no_sync_stats = TimingStats::with_capacity(iterations);
        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let start = Instant::now();
            let (logits, values) = model.forward(input);
            // Just drop without extracting
            drop(logits);
            drop(values);
            no_sync_stats.add(start.elapsed().as_micros() as f64);
        }

        // Test WITH forced sync (extract data to force GPU completion)
        let mut sync_stats = TimingStats::with_capacity(iterations);
        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let start = Instant::now();
            let (logits, values) = model.forward(input);
            // Force sync by extracting data
            let _ = logits.into_data();
            let _ = values.into_data();
            sync_stats.add(start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Sync Point Impact ===");
        eprintln!("Without sync: mean={:.2}us, slowdown={:.3}x",
            no_sync_stats.mean(), no_sync_stats.slowdown_ratio());
        eprintln!("With sync:    mean={:.2}us, slowdown={:.3}x",
            sync_stats.mean(), sync_stats.slowdown_ratio());

        // Report if sync helps with degradation
        if sync_stats.slowdown_ratio() < no_sync_stats.slowdown_ratio() * 0.8 {
            eprintln!("FINDING: Explicit sync reduces degradation!");
        }
    }

    /// Test: Periodic sync points in long-running loop.
    /// INTENT: Test if periodic device sync prevents accumulation.
    #[test]
    #[ignore]
    fn test_periodic_sync_strategy() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let sync_interval = 100; // Force sync every N iterations
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);
        let mut stats = TimingStats::with_capacity(iterations);

        for i in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let start = Instant::now();
            let (logits, values) = model.forward(input);

            // Periodic sync: force data extraction every sync_interval
            if i % sync_interval == 0 {
                let _ = logits.into_data();
                let _ = values.into_data();
            } else {
                drop(logits);
                drop(values);
            }

            stats.add(start.elapsed().as_micros() as f64);
        }

        eprintln!("\n=== Periodic Sync Strategy ===");
        eprintln!("Sync interval: every {} iterations", sync_interval);
        eprintln!("Slowdown ratio: {:.3}x", stats.slowdown_ratio());
        eprintln!("Relative slope: {:.6}", stats.relative_slope());

        assert!(
            stats.slowdown_ratio() < 1.5,
            "Even with periodic sync, degradation of {:.2}x detected",
            stats.slowdown_ratio()
        );
    }

    // ========================================================================
    // 7. Full Learner Iteration Simulation
    // ========================================================================
    //
    // NOTE: Serialization is tested separately (test_weight_serialization_timing)
    // using the raw Wgpu backend due to trait bound complexity with Autodiff.

    /// Test: Simulate learner loop for 5000+ iterations (without serialization).
    /// INTENT: Identify which WGPU operation causes SPS degradation.
    ///
    /// This test simulates the learner hot loop:
    /// 1. Create tensors from CPU data (Tensor::from_floats)
    /// 2. Forward pass (model.forward)
    /// 3. Extract values (into_data)
    /// 4. Backward pass (loss.backward)
    ///
    /// Serialization is tested separately with the raw Wgpu backend.
    #[test]
    #[ignore]
    fn test_full_learner_iteration_simulation() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let checkpoint_interval = 500;
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);

        // Per-component timing
        let mut tensor_creation_stats = TimingStats::with_capacity(iterations);
        let mut forward_stats = TimingStats::with_capacity(iterations);
        let mut extract_stats = TimingStats::with_capacity(iterations);
        let mut backward_stats = TimingStats::with_capacity(iterations);
        let mut total_stats = TimingStats::with_capacity(iterations);

        // Checkpoint tracking
        let mut checkpoints: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();

        let overall_start = Instant::now();

        for i in 0..iterations {
            let iter_start = Instant::now();

            // 1. Tensor creation (CPU -> GPU)
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let t1 = Instant::now();
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);
            tensor_creation_stats.add(t1.elapsed().as_micros() as f64);

            // 2. Forward pass
            let t2 = Instant::now();
            let (logits, values) = model.forward(input);
            forward_stats.add(t2.elapsed().as_micros() as f64);

            // 3. Extract values (GPU -> CPU)
            let t3 = Instant::now();
            let _log_vals: Vec<f32> = logits.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            let _val_vals: Vec<f32> = values.clone().into_data().as_slice::<f32>().unwrap().to_vec();
            extract_stats.add(t3.elapsed().as_micros() as f64);

            // 4. Compute loss and backward
            let loss = logits.sum() + values.sum();
            let t4 = Instant::now();
            let _grads = loss.backward();
            backward_stats.add(t4.elapsed().as_micros() as f64);

            total_stats.add(iter_start.elapsed().as_micros() as f64);

            // Checkpoint
            if (i + 1) % checkpoint_interval == 0 {
                let recent_start = tensor_creation_stats.len().saturating_sub(checkpoint_interval);

                let tensor_recent = tensor_creation_stats.samples[recent_start..].iter().sum::<f64>()
                    / checkpoint_interval as f64;
                let forward_recent = forward_stats.samples[recent_start..].iter().sum::<f64>()
                    / checkpoint_interval as f64;
                let extract_recent = extract_stats.samples[recent_start..].iter().sum::<f64>()
                    / checkpoint_interval as f64;
                let backward_recent = backward_stats.samples[recent_start..].iter().sum::<f64>()
                    / checkpoint_interval as f64;
                let total_recent = total_stats.samples[recent_start..].iter().sum::<f64>()
                    / checkpoint_interval as f64;

                checkpoints.push((
                    i + 1,
                    tensor_recent,
                    forward_recent,
                    extract_recent,
                    backward_recent,
                    total_recent,
                ));

                let elapsed = overall_start.elapsed().as_secs_f64();
                let sps = (i + 1) as f64 / elapsed;

                eprintln!(
                    "Iter {:5}: SPS={:.0} | tensor={:.0}us fwd={:.0}us ext={:.0}us bwd={:.0}us tot={:.0}us",
                    i + 1, sps, tensor_recent, forward_recent, extract_recent,
                    backward_recent, total_recent
                );
            }
        }

        // Final analysis
        eprintln!("\n=== Full Learner Simulation Results ===");
        eprintln!("Total iterations: {}", iterations);
        eprintln!("\nComponent Slowdown Analysis:");
        eprintln!("  Tensor Creation: {:.3}x (slope={:.6})",
            tensor_creation_stats.slowdown_ratio(), tensor_creation_stats.relative_slope());
        eprintln!("  Forward Pass:    {:.3}x (slope={:.6})",
            forward_stats.slowdown_ratio(), forward_stats.relative_slope());
        eprintln!("  Data Extraction: {:.3}x (slope={:.6})",
            extract_stats.slowdown_ratio(), extract_stats.relative_slope());
        eprintln!("  Backward Pass:   {:.3}x (slope={:.6})",
            backward_stats.slowdown_ratio(), backward_stats.relative_slope());
        eprintln!("  Total:           {:.3}x (slope={:.6})",
            total_stats.slowdown_ratio(), total_stats.relative_slope());

        // Identify the worst degrading component
        let components = [
            ("Tensor Creation", tensor_creation_stats.slowdown_ratio(), tensor_creation_stats.relative_slope()),
            ("Forward Pass", forward_stats.slowdown_ratio(), forward_stats.relative_slope()),
            ("Data Extraction", extract_stats.slowdown_ratio(), extract_stats.relative_slope()),
            ("Backward Pass", backward_stats.slowdown_ratio(), backward_stats.relative_slope()),
        ];

        let (worst_name, worst_slowdown, worst_slope) = components
            .iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        if *worst_slope > 0.001 {
            eprintln!("\n*** DEGRADATION DETECTED: {} ***", worst_name);
            eprintln!("    Slowdown: {:.3}x, Relative Slope: {:.6}", worst_slowdown, worst_slope);
        }

        // Overall assertion
        assert!(
            total_stats.slowdown_ratio() < 2.0,
            "Overall learner loop degraded by {:.2}x - WGPU resource issue detected!",
            total_stats.slowdown_ratio()
        );
    }

    /// Test: Long-running simulation to reproduce production degradation pattern.
    /// INTENT: Run enough iterations to potentially trigger the 57% SPS drop.
    ///
    /// Note: Serialization is excluded due to trait bound complexity with Autodiff.
    /// Run test_weight_serialization_timing separately for serialization timing.
    #[test]
    #[ignore]
    fn test_extended_learner_simulation() {
        let device = WgpuDevice::default();
        let iterations = 10_000; // Extended run
        let batch_size = 640;
        let obs_size = 4;
        let hidden_size = 128;
        let n_actions = 2;

        let model = TestNet::<B>::new(&device, obs_size, hidden_size, n_actions);

        let mut total_stats = TimingStats::with_capacity(iterations);
        let start = Instant::now();

        for i in 0..iterations {
            let iter_start = Instant::now();

            // Full learner iteration (without serialization)
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let input = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);

            let (logits, values) = model.forward(input);
            let _ = logits.clone().into_data();
            let _ = values.clone().into_data();

            let loss = logits.sum() + values.sum();
            let _grads = loss.backward();

            total_stats.add(iter_start.elapsed().as_micros() as f64);

            // Progress report
            if (i + 1) % 1000 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let ips = (i + 1) as f64 / elapsed;
                let recent = total_stats.window_mean(
                    ((i + 1) - 1000) * 100 / iterations,
                    (i + 1) * 100 / iterations,
                );
                eprintln!("Iter {:5}: IPS={:.0}, recent_mean={:.0}us", i + 1, ips, recent);
            }
        }

        let final_slowdown = total_stats.slowdown_ratio();
        let sps_degradation = 1.0 - (1.0 / final_slowdown);

        eprintln!("\n=== Extended Simulation Complete ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("Slowdown ratio: {:.3}x", final_slowdown);
        eprintln!("Equivalent SPS degradation: {:.1}%", sps_degradation * 100.0);

        // Compare to observed 57% degradation
        if sps_degradation > 0.3 {
            eprintln!("\n!!! SIGNIFICANT DEGRADATION REPRODUCED !!!");
            eprintln!("This matches the observed production pattern.");
        }
    }

    // ========================================================================
    // 8. WGPU Resource Tracking Tests
    // ========================================================================

    /// Test: Measure memory-like behavior through timing analysis.
    /// INTENT: Detect resource accumulation indirectly through timing patterns.
    ///
    /// If resources accumulate, operations that interact with the resource pool
    /// (allocation, deallocation) will show increasing timing variance.
    #[test]
    #[ignore]
    fn test_timing_variance_growth() {
        let device = WgpuDevice::default();
        let iterations = 5_000;
        let window_size = 500;
        let batch_size = 640;
        let obs_size = 4;

        let mut stats = TimingStats::with_capacity(iterations);

        for _ in 0..iterations {
            let data: Vec<f32> = vec![0.5f32; batch_size * obs_size];
            let start = Instant::now();
            let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device)
                .reshape([batch_size, obs_size]);
            let _ = tensor.into_data();
            stats.add(start.elapsed().as_micros() as f64);
        }

        // Compute variance in windows
        let mut window_variances: Vec<f64> = Vec::new();
        for chunk in stats.samples.chunks(window_size) {
            if chunk.len() < window_size / 2 {
                continue;
            }
            let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let variance = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / chunk.len() as f64;
            window_variances.push(variance);
        }

        if window_variances.len() >= 2 {
            let first_var = window_variances[0];
            let last_var = *window_variances.last().unwrap();
            let var_growth = last_var / first_var.max(1.0);

            eprintln!("\n=== Timing Variance Analysis ===");
            eprintln!("First window variance: {:.2}", first_var);
            eprintln!("Last window variance:  {:.2}", last_var);
            eprintln!("Variance growth: {:.3}x", var_growth);

            if var_growth > 2.0 {
                eprintln!("WARNING: Timing variance increased significantly - resource fragmentation?");
            }
        }
    }

    /// Test: Rapid allocation/deallocation cycles.
    /// INTENT: Stress test WGPU buffer pool recycling.
    #[test]
    #[ignore]
    fn test_rapid_alloc_dealloc_cycles() {
        let device = WgpuDevice::default();
        let iterations = 10_000;
        let sizes = [64, 256, 1024, 4096]; // Various tensor sizes

        let mut stats = TimingStats::with_capacity(iterations);

        for i in 0..iterations {
            let size = sizes[i % sizes.len()];
            let data: Vec<f32> = vec![0.5f32; size];

            let start = Instant::now();
            // Rapid create and destroy
            let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), &device);
            drop(tensor);
            stats.add(start.elapsed().as_nanos() as f64);
        }

        eprintln!("\n=== Rapid Alloc/Dealloc Cycles ===");
        eprintln!("Iterations: {}", iterations);
        eprintln!("Mean time: {:.2} ns", stats.mean());
        eprintln!("Slowdown ratio: {:.3}x", stats.slowdown_ratio());

        assert!(
            stats.slowdown_ratio() < 2.0,
            "Rapid alloc/dealloc cycles caused {:.2}x slowdown - buffer pool issue?",
            stats.slowdown_ratio()
        );
    }
}

// ============================================================================
// Summary Tests (Run All WGPU Tests)
// ============================================================================

mod summary {
    use super::*;

    /// Helper test to print system info.
    #[test]
    #[ignore]
    fn test_print_wgpu_info() {
        eprintln!("\n=== WGPU Pipeline SPS Test Suite ===");
        eprintln!("These tests investigate SPS degradation in WGPU tensor operations.");
        eprintln!("Run with: cargo test wgpu_pipeline_sps -- --ignored --test-threads=1");
        eprintln!();
        eprintln!("Tests to run:");
        eprintln!("  1. test_tensor_creation_timing_over_iterations");
        eprintln!("  2. test_forward_pass_timing_over_iterations");
        eprintln!("  3. test_backward_pass_timing_over_iterations");
        eprintln!("  4. test_into_data_extraction_timing");
        eprintln!("  5. test_weight_serialization_timing");
        eprintln!("  6. test_explicit_sync_impact");
        eprintln!("  7. test_full_learner_iteration_simulation");
        eprintln!("  8. test_extended_learner_simulation");
        eprintln!();
        eprintln!("If degradation is detected, the test output will identify");
        eprintln!("which WGPU operation is causing the issue.");
    }
}
