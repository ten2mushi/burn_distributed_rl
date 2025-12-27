//! Learning rate schedulers for distributed training.
//!
//! Provides common learning rate scheduling strategies:
//! - `ConstantLR`: Fixed learning rate
//! - `LinearDecay`: Linear interpolation from start to end LR
//! - `CosineAnnealing`: Cosine decay with optional warm restarts
//! - `Warmup`: Wrapper for linear warmup phase
//! - `PolynomialDecay`: Polynomial decay with configurable power
//!
//! # Data Integrity
//!
//! All schedulers validate inputs in debug builds and handle edge cases gracefully
//! in release builds to prevent NaN/Inf propagation in distributed training:
//!
//! - **Division by zero**: `total_steps=0` or `period=0` triggers debug panic,
//!   returns sensible defaults in release
//! - **Non-finite inputs**: NaN/Inf LR values trigger debug panic
//! - **Negative LR**: Negative learning rates trigger debug panic
//! - **Output clamping**: Results are clamped to prevent Inf from edge cases

use std::sync::atomic::{AtomicUsize, Ordering};

/// Learning rate scheduler trait.
///
/// Implementations provide step-dependent learning rates for training loops.
pub trait LRScheduler: Send + Sync {
    /// Get the learning rate for a given step.
    fn get_lr(&self, step: usize) -> f64;

    /// Convenience method to get current LR and increment step atomically.
    fn step(&self) -> f64 {
        self.get_lr(0) // Default: no-op, subclasses override
    }
}

/// Constant learning rate (no scheduling).
///
/// # Data Validation
///
/// In debug builds, panics if LR is:
/// - Non-finite (NaN or Inf)
/// - Negative
///
/// In release builds, non-finite values are replaced with 0.0.
#[derive(Debug)]
pub struct ConstantLR {
    lr: f64,
}

impl ConstantLR {
    /// Create a new constant LR scheduler.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `lr` is NaN, Inf, or negative.
    pub fn new(lr: f64) -> Self {
        debug_assert!(
            lr.is_finite(),
            "ConstantLR: lr must be finite, got {}",
            lr
        );
        debug_assert!(
            lr >= 0.0,
            "ConstantLR: lr must be non-negative, got {}",
            lr
        );

        // In release, sanitize non-finite to 0.0
        let lr = if lr.is_finite() && lr >= 0.0 { lr } else { 0.0 };

        Self { lr }
    }

    /// Get the configured learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

/// Linear decay from start LR to end LR over total_steps.
///
/// After total_steps, returns end_lr (doesn't go below).
///
/// # Data Validation
///
/// In debug builds, panics if:
/// - `total_steps` is 0 (division by zero)
/// - `start_lr` or `end_lr` is NaN/Inf
/// - `start_lr` or `end_lr` is negative
///
/// In release builds:
/// - `total_steps=0` returns `start_lr` for all steps
/// - Non-finite LRs are replaced with 0.0
#[derive(Debug)]
pub struct LinearDecay {
    start_lr: f64,
    end_lr: f64,
    total_steps: usize,
    current_step: AtomicUsize,
}

impl LinearDecay {
    /// Create a new linear decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `start_lr` - Initial learning rate (must be finite and non-negative)
    /// * `end_lr` - Final learning rate (must be finite and non-negative)
    /// * `total_steps` - Number of steps over which to decay (must be > 0)
    ///
    /// # Panics (debug only)
    ///
    /// Panics if any argument is invalid.
    pub fn new(start_lr: f64, end_lr: f64, total_steps: usize) -> Self {
        debug_assert!(
            total_steps > 0,
            "LinearDecay: total_steps must be > 0, got {}",
            total_steps
        );
        debug_assert!(
            start_lr.is_finite(),
            "LinearDecay: start_lr must be finite, got {}",
            start_lr
        );
        debug_assert!(
            end_lr.is_finite(),
            "LinearDecay: end_lr must be finite, got {}",
            end_lr
        );
        debug_assert!(
            start_lr >= 0.0,
            "LinearDecay: start_lr must be non-negative, got {}",
            start_lr
        );
        debug_assert!(
            end_lr >= 0.0,
            "LinearDecay: end_lr must be non-negative, got {}",
            end_lr
        );

        // Sanitize in release builds
        let start_lr = if start_lr.is_finite() && start_lr >= 0.0 {
            start_lr
        } else {
            0.0
        };
        let end_lr = if end_lr.is_finite() && end_lr >= 0.0 {
            end_lr
        } else {
            0.0
        };

        Self {
            start_lr,
            end_lr,
            total_steps,
            current_step: AtomicUsize::new(0),
        }
    }

    /// Reset the scheduler to initial state.
    pub fn reset(&self) {
        self.current_step.store(0, Ordering::SeqCst);
    }

    /// Get the start learning rate.
    pub fn start_lr(&self) -> f64 {
        self.start_lr
    }

    /// Get the end learning rate.
    pub fn end_lr(&self) -> f64 {
        self.end_lr
    }

    /// Get the total steps for decay.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}

impl LRScheduler for LinearDecay {
    fn get_lr(&self, step: usize) -> f64 {
        // Handle division by zero gracefully in release
        if self.total_steps == 0 {
            return self.start_lr;
        }

        let progress = (step as f64) / (self.total_steps as f64);
        let progress = progress.min(1.0);
        let lr = self.start_lr + (self.end_lr - self.start_lr) * progress;

        // Ensure output is always finite (handles edge cases from float arithmetic)
        if lr.is_finite() {
            lr
        } else {
            self.end_lr
        }
    }

    fn step(&self) -> f64 {
        let step = self.current_step.fetch_add(1, Ordering::SeqCst);
        self.get_lr(step)
    }
}

/// Cosine annealing learning rate scheduler.
///
/// Follows a cosine curve from base_lr to min_lr over the period.
/// Optionally supports warm restarts (resets after each period).
///
/// # Data Validation
///
/// In debug builds, panics if:
/// - `period` is 0 (division by zero / modulo by zero)
/// - `base_lr` or `min_lr` is NaN/Inf
/// - `base_lr` or `min_lr` is negative
///
/// In release builds:
/// - `period=0` returns `base_lr` for all steps
/// - Non-finite LRs are replaced with 0.0
#[derive(Debug)]
pub struct CosineAnnealing {
    base_lr: f64,
    min_lr: f64,
    period: usize,
    warm_restarts: bool,
    current_step: AtomicUsize,
}

impl CosineAnnealing {
    /// Create a new cosine annealing scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Maximum learning rate (at start of each cycle)
    /// * `min_lr` - Minimum learning rate (at end of each cycle)
    /// * `period` - Steps per cosine cycle (must be > 0)
    /// * `warm_restarts` - If true, reset to base_lr after each period
    ///
    /// # Panics (debug only)
    ///
    /// Panics if any argument is invalid.
    pub fn new(base_lr: f64, min_lr: f64, period: usize, warm_restarts: bool) -> Self {
        debug_assert!(
            period > 0,
            "CosineAnnealing: period must be > 0, got {}",
            period
        );
        debug_assert!(
            base_lr.is_finite(),
            "CosineAnnealing: base_lr must be finite, got {}",
            base_lr
        );
        debug_assert!(
            min_lr.is_finite(),
            "CosineAnnealing: min_lr must be finite, got {}",
            min_lr
        );
        debug_assert!(
            base_lr >= 0.0,
            "CosineAnnealing: base_lr must be non-negative, got {}",
            base_lr
        );
        debug_assert!(
            min_lr >= 0.0,
            "CosineAnnealing: min_lr must be non-negative, got {}",
            min_lr
        );

        // Sanitize in release builds
        let base_lr = if base_lr.is_finite() && base_lr >= 0.0 {
            base_lr
        } else {
            0.0
        };
        let min_lr = if min_lr.is_finite() && min_lr >= 0.0 {
            min_lr
        } else {
            0.0
        };

        Self {
            base_lr,
            min_lr,
            period,
            warm_restarts,
            current_step: AtomicUsize::new(0),
        }
    }

    /// Create without warm restarts (decays once).
    pub fn without_restarts(base_lr: f64, min_lr: f64, total_steps: usize) -> Self {
        Self::new(base_lr, min_lr, total_steps, false)
    }

    /// Reset the scheduler to initial state.
    pub fn reset(&self) {
        self.current_step.store(0, Ordering::SeqCst);
    }

    /// Get the base (maximum) learning rate.
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// Get the minimum learning rate.
    pub fn min_lr(&self) -> f64 {
        self.min_lr
    }

    /// Get the period (steps per cycle).
    pub fn period(&self) -> usize {
        self.period
    }
}

impl LRScheduler for CosineAnnealing {
    fn get_lr(&self, step: usize) -> f64 {
        // Handle period=0 gracefully in release (avoids panic on % 0)
        if self.period == 0 {
            return self.base_lr;
        }

        let step_in_period = if self.warm_restarts {
            step % self.period
        } else {
            step.min(self.period)
        };

        let progress = (step_in_period as f64) / (self.period as f64);
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        let lr = self.min_lr + (self.base_lr - self.min_lr) * cosine;

        // Ensure output is always finite
        if lr.is_finite() {
            lr
        } else {
            self.base_lr
        }
    }

    fn step(&self) -> f64 {
        let step = self.current_step.fetch_add(1, Ordering::SeqCst);
        self.get_lr(step)
    }
}

/// Warmup wrapper for any scheduler.
///
/// Provides linear warmup from a start LR to the inner scheduler's LR
/// over warmup_steps, then delegates to the inner scheduler.
///
/// # Data Validation
///
/// In debug builds, panics if:
/// - `warmup_start_lr` is NaN/Inf
/// - `warmup_start_lr` is negative
///
/// Note: `warmup_steps=0` is valid and means no warmup phase
/// (immediately delegates to inner scheduler).
///
/// In release builds:
/// - Non-finite `warmup_start_lr` is replaced with 0.0
#[derive(Debug)]
pub struct Warmup<S: LRScheduler> {
    inner: S,
    warmup_steps: usize,
    warmup_start_lr: f64,
    current_step: AtomicUsize,
}

impl<S: LRScheduler> Warmup<S> {
    /// Create a new warmup wrapper.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying scheduler to use after warmup
    /// * `warmup_steps` - Number of steps for linear warmup (0 = no warmup)
    /// * `warmup_start_lr` - Learning rate at step 0 (typically very small or 0)
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `warmup_start_lr` is NaN, Inf, or negative.
    pub fn new(inner: S, warmup_steps: usize, warmup_start_lr: f64) -> Self {
        debug_assert!(
            warmup_start_lr.is_finite(),
            "Warmup: warmup_start_lr must be finite, got {}",
            warmup_start_lr
        );
        debug_assert!(
            warmup_start_lr >= 0.0,
            "Warmup: warmup_start_lr must be non-negative, got {}",
            warmup_start_lr
        );

        // Sanitize in release builds
        let warmup_start_lr = if warmup_start_lr.is_finite() && warmup_start_lr >= 0.0 {
            warmup_start_lr
        } else {
            0.0
        };

        Self {
            inner,
            warmup_steps,
            warmup_start_lr,
            current_step: AtomicUsize::new(0),
        }
    }

    /// Reset the warmup scheduler.
    pub fn reset(&self) {
        self.current_step.store(0, Ordering::SeqCst);
    }

    /// Get the inner scheduler.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get the warmup steps.
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Get the warmup start learning rate.
    pub fn warmup_start_lr(&self) -> f64 {
        self.warmup_start_lr
    }
}

impl<S: LRScheduler> LRScheduler for Warmup<S> {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup to the target LR at warmup_steps
            let target_lr = self.inner.get_lr(0);

            // Handle warmup_steps=0 (shouldn't reach here due to step < 0 being false)
            // but be defensive anyway
            if self.warmup_steps == 0 {
                return target_lr;
            }

            let progress = (step as f64) / (self.warmup_steps as f64);
            let lr = self.warmup_start_lr + (target_lr - self.warmup_start_lr) * progress;

            // Ensure output is finite
            if lr.is_finite() {
                lr
            } else {
                self.warmup_start_lr
            }
        } else {
            // After warmup, delegate to inner scheduler (adjusted step)
            self.inner.get_lr(step - self.warmup_steps)
        }
    }

    fn step(&self) -> f64 {
        let step = self.current_step.fetch_add(1, Ordering::SeqCst);
        self.get_lr(step)
    }
}

/// Polynomial decay scheduler.
///
/// Formula: `lr = end_lr + (base_lr - end_lr) * (1 - step/total_steps)^power`
///
/// # Data Validation
///
/// In debug builds, panics if:
/// - `total_steps` is 0 (division by zero)
/// - `base_lr` or `end_lr` is NaN/Inf
/// - `base_lr` or `end_lr` is negative
/// - `power` is NaN/Inf
/// - `power` is <= 0 (causes no decay or explosion)
///
/// In release builds:
/// - `total_steps=0` returns `base_lr` for all steps
/// - Non-finite LRs are replaced with 0.0
/// - Non-positive power is clamped to 1.0 (linear decay)
/// - Output is clamped to prevent Inf from edge cases
#[derive(Debug)]
pub struct PolynomialDecay {
    base_lr: f64,
    end_lr: f64,
    total_steps: usize,
    power: f64,
    current_step: AtomicUsize,
}

impl PolynomialDecay {
    /// Create a new polynomial decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate (must be finite and non-negative)
    /// * `end_lr` - Final learning rate (must be finite and non-negative)
    /// * `total_steps` - Number of steps for decay (must be > 0)
    /// * `power` - Polynomial power (must be > 0; 1.0 = linear, 2.0 = quadratic)
    ///
    /// # Panics (debug only)
    ///
    /// Panics if any argument is invalid.
    pub fn new(base_lr: f64, end_lr: f64, total_steps: usize, power: f64) -> Self {
        debug_assert!(
            total_steps > 0,
            "PolynomialDecay: total_steps must be > 0, got {}",
            total_steps
        );
        debug_assert!(
            base_lr.is_finite(),
            "PolynomialDecay: base_lr must be finite, got {}",
            base_lr
        );
        debug_assert!(
            end_lr.is_finite(),
            "PolynomialDecay: end_lr must be finite, got {}",
            end_lr
        );
        debug_assert!(
            base_lr >= 0.0,
            "PolynomialDecay: base_lr must be non-negative, got {}",
            base_lr
        );
        debug_assert!(
            end_lr >= 0.0,
            "PolynomialDecay: end_lr must be non-negative, got {}",
            end_lr
        );
        debug_assert!(
            power.is_finite(),
            "PolynomialDecay: power must be finite, got {}",
            power
        );
        debug_assert!(
            power > 0.0,
            "PolynomialDecay: power must be > 0 (got {}). Use power=1.0 for linear decay.",
            power
        );

        // Sanitize in release builds
        let base_lr = if base_lr.is_finite() && base_lr >= 0.0 {
            base_lr
        } else {
            0.0
        };
        let end_lr = if end_lr.is_finite() && end_lr >= 0.0 {
            end_lr
        } else {
            0.0
        };
        // Clamp power to positive value (default to linear if invalid)
        let power = if power.is_finite() && power > 0.0 {
            power
        } else {
            1.0
        };

        Self {
            base_lr,
            end_lr,
            total_steps,
            power,
            current_step: AtomicUsize::new(0),
        }
    }

    /// Reset the scheduler to initial state.
    pub fn reset(&self) {
        self.current_step.store(0, Ordering::SeqCst);
    }

    /// Get the base (initial) learning rate.
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// Get the end (final) learning rate.
    pub fn end_lr(&self) -> f64 {
        self.end_lr
    }

    /// Get the total steps for decay.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Get the polynomial power.
    pub fn power(&self) -> f64 {
        self.power
    }
}

impl LRScheduler for PolynomialDecay {
    fn get_lr(&self, step: usize) -> f64 {
        // Handle division by zero gracefully in release
        if self.total_steps == 0 {
            return self.base_lr;
        }

        let step = step.min(self.total_steps);
        let progress = (step as f64) / (self.total_steps as f64);
        let decay = (1.0 - progress).powf(self.power);
        let lr = self.end_lr + (self.base_lr - self.end_lr) * decay;

        // Ensure output is finite and within reasonable bounds
        if lr.is_finite() {
            // Clamp to [0, max(base_lr, end_lr)] to prevent runaway values
            let max_lr = self.base_lr.max(self.end_lr);
            lr.max(0.0).min(max_lr)
        } else {
            self.end_lr
        }
    }

    fn step(&self) -> f64 {
        let step = self.current_step.fetch_add(1, Ordering::SeqCst);
        self.get_lr(step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLR::new(0.001);
        assert!((sched.get_lr(0) - 0.001).abs() < 1e-10);
        assert!((sched.get_lr(1000) - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_linear_decay() {
        let sched = LinearDecay::new(1.0, 0.0, 100);

        // Start
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-10);
        // Middle
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-10);
        // End
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-10);
        // After end (clamps)
        assert!((sched.get_lr(200) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 100);

        // Start: should be at base_lr
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-10);
        // End: should be at min_lr
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-10);
        // Middle: should be around 0.5
        assert!((sched.get_lr(50) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cosine_warm_restarts() {
        let sched = CosineAnnealing::new(1.0, 0.0, 50, true);

        // Start of first period
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-10);
        // End of first period (before restart)
        assert!((sched.get_lr(49) - 0.0).abs() < 0.1);
        // Start of second period (restart)
        assert!((sched.get_lr(50) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_warmup() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, 0.0);

        // Start of warmup
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-10);
        // Middle of warmup
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-10);
        // End of warmup
        assert!((sched.get_lr(100) - 1.0).abs() < 1e-10);
        // After warmup (constant)
        assert!((sched.get_lr(150) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_decay() {
        // Linear decay (power=1.0) should match LinearDecay
        let sched = PolynomialDecay::new(1.0, 0.0, 100, 1.0);

        assert!((sched.get_lr(0) - 1.0).abs() < 1e-10);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-10);
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_increments() {
        let sched = LinearDecay::new(1.0, 0.0, 10);

        // step() should return LR and auto-increment
        assert!((sched.step() - 1.0).abs() < 1e-10); // step 0
        assert!((sched.step() - 0.9).abs() < 1e-10); // step 1
        assert!((sched.step() - 0.8).abs() < 1e-10); // step 2
    }
}
