//! Comprehensive test suite for learning rate schedulers.
//!
//! These tests follow the "Tests as Definition: The Yoneda Way" philosophy,
//! providing a complete behavioral specification of all schedulers.
//!
//! # Test Categories
//!
//! 1. **Boundary Tests**: Step = 0, total_steps-1, total_steps, total_steps+1, usize::MAX
//! 2. **Input Validation Tests**: NaN, +Inf, -Inf, negative values are rejected
//! 3. **Division by Zero Tests**: All schedulers handle zero divisors gracefully
//! 4. **Mathematical Correctness Tests**: Formula verification at key points
//! 5. **Concurrency Tests**: Thread safety and atomic operations
//! 6. **Overflow Tests**: Step counter near usize::MAX
//! 7. **Composition Tests**: Warmup wrapping various schedulers
//! 8. **Property-Based Tests**: Monotonicity, boundedness, continuity
//!
//! # Defensive Coding
//!
//! All schedulers now implement defensive coding:
//! - **Debug builds**: Invalid inputs (NaN, Inf, negative LR, zero steps) trigger panics
//! - **Release builds**: Invalid inputs are sanitized to safe defaults (0.0 for LR, 1.0 for power)
//! - **Output clamping**: Results are always finite and within expected bounds

use super::*;
use std::thread;

// ============================================================================
// CONSTANT LR TESTS
// ============================================================================

mod constant_lr_tests {
    use super::*;

    // --- Basic Behavior ---

    #[test]
    fn should_return_same_lr_for_any_step() {
        let sched = ConstantLR::new(0.001);
        assert_eq!(sched.get_lr(0), 0.001);
        assert_eq!(sched.get_lr(1), 0.001);
        assert_eq!(sched.get_lr(1_000_000), 0.001);
        assert_eq!(sched.get_lr(usize::MAX), 0.001);
    }

    #[test]
    fn should_return_zero_lr_when_initialized_with_zero() {
        let sched = ConstantLR::new(0.0);
        assert_eq!(sched.get_lr(0), 0.0);
        assert_eq!(sched.get_lr(100), 0.0);
    }

    // --- Edge Cases: Unusual but Valid LR Values ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "non-negative"))]
    fn should_reject_negative_lr_in_debug() {
        // DEFENSIVE: Negative LR is rejected in debug builds.
        // In release builds, it's sanitized to 0.0.
        let sched = ConstantLR::new(-0.001);
        // In release, should be sanitized to 0.0
        assert_eq!(sched.get_lr(0), 0.0);
    }

    #[test]
    fn should_accept_very_small_lr() {
        let sched = ConstantLR::new(1e-15);
        assert_eq!(sched.get_lr(0), 1e-15);
    }

    #[test]
    fn should_accept_very_large_lr() {
        let sched = ConstantLR::new(1e15);
        assert_eq!(sched.get_lr(0), 1e15);
    }

    // --- Non-Finite Value Tests (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "must be finite"))]
    fn should_reject_nan_lr_in_debug() {
        // DEFENSIVE: NaN input triggers panic in debug, sanitized to 0.0 in release.
        let sched = ConstantLR::new(f64::NAN);
        // In release, should be sanitized to 0.0
        assert_eq!(sched.get_lr(0), 0.0);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "must be finite"))]
    fn should_reject_positive_infinity_lr_in_debug() {
        // DEFENSIVE: +Inf input triggers panic in debug, sanitized to 0.0 in release.
        let sched = ConstantLR::new(f64::INFINITY);
        assert_eq!(sched.get_lr(0), 0.0);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "must be finite"))]
    fn should_reject_negative_infinity_lr_in_debug() {
        // DEFENSIVE: -Inf input triggers panic in debug, sanitized to 0.0 in release.
        let sched = ConstantLR::new(f64::NEG_INFINITY);
        assert_eq!(sched.get_lr(0), 0.0);
    }

    // --- Trait Implementation ---

    #[test]
    fn step_should_always_return_same_value() {
        // ConstantLR uses default step() which just calls get_lr(0)
        let sched = ConstantLR::new(0.001);
        assert_eq!(sched.step(), 0.001);
        assert_eq!(sched.step(), 0.001);
        assert_eq!(sched.step(), 0.001);
    }

    // --- Thread Safety ---

    #[test]
    fn should_be_safely_readable_from_multiple_threads() {
        let sched = ConstantLR::new(0.001);
        thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    for step in 0..1000 {
                        assert_eq!(sched.get_lr(step), 0.001);
                    }
                });
            }
        });
    }
}

// ============================================================================
// LINEAR DECAY TESTS
// ============================================================================

mod linear_decay_tests {
    use super::*;

    // --- Basic Behavior: Mathematical Correctness ---

    #[test]
    fn should_return_start_lr_at_step_zero() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_return_end_lr_at_total_steps() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn should_return_midpoint_lr_at_half_steps() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn should_follow_linear_interpolation_formula() {
        // Formula: lr = start_lr + (end_lr - start_lr) * (step / total_steps)
        let start = 0.1;
        let end = 0.01;
        let total = 1000;
        let sched = LinearDecay::new(start, end, total);

        for step in [0, 100, 250, 500, 750, 999, 1000] {
            let progress = (step as f64) / (total as f64);
            let expected = start + (end - start) * progress.min(1.0);
            let actual = sched.get_lr(step);
            assert!(
                (actual - expected).abs() < 1e-15,
                "Step {}: expected {}, got {}",
                step,
                expected,
                actual
            );
        }
    }

    // --- Boundary Tests ---

    #[test]
    fn should_clamp_to_end_lr_after_total_steps() {
        let sched = LinearDecay::new(1.0, 0.1, 100);
        assert!((sched.get_lr(101) - 0.1).abs() < 1e-15);
        assert!((sched.get_lr(1000) - 0.1).abs() < 1e-15);
        assert!((sched.get_lr(usize::MAX) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn should_handle_step_one_before_end() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        let expected = 1.0 + (0.0 - 1.0) * (99.0 / 100.0);
        assert!((sched.get_lr(99) - expected).abs() < 1e-15);
    }

    #[test]
    fn should_handle_step_one_after_end() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        // Should clamp to end_lr
        assert!((sched.get_lr(101) - 0.0).abs() < 1e-15);
    }

    // --- Division by Zero: Defensive Behavior ---
    //
    // DEFENSIVE: total_steps=0 now triggers debug_assert panic in debug builds.
    // In release builds, it returns start_lr for all steps (graceful degradation).

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "total_steps must be > 0"))]
    fn total_steps_zero_panics_in_debug() {
        // DEFENSIVE: Division by zero is caught in debug builds.
        // In release builds, returns start_lr for all steps.
        let sched = LinearDecay::new(1.0, 0.0, 0);

        // In release: graceful degradation returns start_lr
        let lr_0 = sched.get_lr(0);
        assert!(
            (lr_0 - 1.0).abs() < 1e-15,
            "Expected start_lr (1.0) in release mode, got {}",
            lr_0
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "total_steps must be > 0"))]
    fn total_steps_zero_returns_start_lr_in_release() {
        // Verify release behavior with different start/end values
        let sched = LinearDecay::new(0.5, 0.1, 0);

        // In release: all steps should return start_lr = 0.5
        for step in [0, 1, 10, 100, 1000] {
            let lr = sched.get_lr(step);
            assert!(
                (lr - 0.5).abs() < 1e-15,
                "Step {}: expected 0.5 (start_lr), got {}",
                step,
                lr
            );
        }
    }

    // --- Inverted Range Tests ---

    #[test]
    fn should_handle_increasing_lr_when_end_greater_than_start() {
        // This is "inverted" but mathematically valid - warmup-like behavior
        let sched = LinearDecay::new(0.0, 1.0, 100);
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(100) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_handle_equal_start_and_end_lr() {
        let sched = LinearDecay::new(0.5, 0.5, 100);
        assert!((sched.get_lr(0) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(100) - 0.5).abs() < 1e-15);
    }

    // --- Non-Finite Value Tests (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "start_lr must be finite"))]
    fn should_reject_nan_start_lr_in_debug() {
        // DEFENSIVE: NaN is rejected in debug, sanitized to 0.0 in release.
        let sched = LinearDecay::new(f64::NAN, 0.0, 100);
        // In release: sanitized to 0.0, returns valid LR
        assert!(!sched.get_lr(0).is_nan());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "end_lr must be finite"))]
    fn should_reject_nan_end_lr_in_debug() {
        // DEFENSIVE: NaN is rejected in debug, sanitized to 0.0 in release.
        let sched = LinearDecay::new(1.0, f64::NAN, 100);
        // In release: sanitized to 0.0, returns valid LR
        assert!(!sched.get_lr(0).is_nan());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "start_lr must be finite"))]
    fn should_reject_infinity_start_lr_in_debug() {
        // DEFENSIVE: Inf is rejected in debug, sanitized to 0.0 in release.
        let sched = LinearDecay::new(f64::INFINITY, 0.0, 100);
        // In release: sanitized to 0.0, returns valid LR
        let lr = sched.get_lr(0);
        assert!(lr.is_finite(), "Expected finite LR, got {}", lr);
    }

    // --- Negative LR Tests (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "start_lr must be non-negative"))]
    fn should_reject_negative_start_lr_in_debug() {
        // DEFENSIVE: Negative LR is rejected in debug, sanitized to 0.0 in release.
        let sched = LinearDecay::new(-1.0, 0.0, 100);
        // In release: sanitized to 0.0
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "end_lr must be non-negative"))]
    fn should_reject_negative_end_lr_in_debug() {
        // DEFENSIVE: Negative LR is rejected in debug, sanitized to 0.0 in release.
        let sched = LinearDecay::new(0.0, -1.0, 100);
        // In release: sanitized to 0.0
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-15);
    }

    // --- Step Counter and Atomics ---

    #[test]
    fn step_should_increment_internal_counter() {
        let sched = LinearDecay::new(1.0, 0.0, 10);
        assert!((sched.step() - 1.0).abs() < 1e-15); // step 0 -> returns 1.0
        assert!((sched.step() - 0.9).abs() < 1e-15); // step 1 -> returns 0.9
        assert!((sched.step() - 0.8).abs() < 1e-15); // step 2 -> returns 0.8
    }

    #[test]
    fn reset_should_restore_step_counter_to_zero() {
        let sched = LinearDecay::new(1.0, 0.0, 10);
        sched.step();
        sched.step();
        sched.step();
        sched.reset();
        assert!((sched.step() - 1.0).abs() < 1e-15); // Back to step 0
    }

    // --- Precision Tests ---

    #[test]
    fn should_maintain_precision_with_small_lr_values() {
        let sched = LinearDecay::new(1e-12, 1e-15, 1000);
        let lr_0 = sched.get_lr(0);
        let lr_1000 = sched.get_lr(1000);
        assert!((lr_0 - 1e-12).abs() < 1e-20);
        assert!((lr_1000 - 1e-15).abs() < 1e-20);
    }

    #[test]
    fn should_handle_very_large_total_steps() {
        let sched = LinearDecay::new(1.0, 0.0, 1_000_000_000);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
        assert!((sched.get_lr(500_000_000) - 0.5).abs() < 1e-9);
        assert!((sched.get_lr(1_000_000_000) - 0.0).abs() < 1e-15);
    }

    // --- Concurrent Access Tests ---

    #[test]
    fn should_handle_concurrent_step_calls() {
        let sched = LinearDecay::new(1.0, 0.0, 1000);

        thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    for _ in 0..100 {
                        let lr = sched.step();
                        assert!(lr.is_finite());
                    }
                });
            }
        });

        // After 10 threads x 100 steps = 1000 total steps
        // The internal counter should be at 1000
        let final_lr = sched.step();
        // step 1000 is at end_lr (clamped to 0.0)
        assert!((final_lr - 0.0).abs() < 1e-15);
    }

    #[test]
    fn should_handle_concurrent_reset_during_stepping() {
        let sched = LinearDecay::new(1.0, 0.0, 100);

        thread::scope(|s| {
            // Thread 1: Continuously steps
            s.spawn(|| {
                for _ in 0..1000 {
                    let lr = sched.step();
                    // LR should always be in valid range (no NaN or panic)
                    assert!(lr.is_finite(), "Got non-finite LR: {}", lr);
                    assert!(lr >= 0.0 && lr <= 1.0, "LR out of range: {}", lr);
                }
            });

            // Thread 2: Periodically resets
            s.spawn(|| {
                for _ in 0..100 {
                    sched.reset();
                    std::thread::yield_now();
                }
            });
        });
    }

    // --- Overflow Tests ---

    #[test]
    fn should_handle_step_near_usize_max() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        // When step > total_steps, progress is clamped to 1.0
        // Even at usize::MAX, it should clamp correctly
        let lr = sched.get_lr(usize::MAX);
        assert!((lr - 0.0).abs() < 1e-15);
    }

    #[test]
    fn should_handle_total_steps_at_usize_max() {
        // Edge case: total_steps = usize::MAX
        let sched = LinearDecay::new(1.0, 0.0, usize::MAX);
        let lr = sched.get_lr(0);
        assert!((lr - 1.0).abs() < 1e-15);

        // At half of usize::MAX
        let half_max = usize::MAX / 2;
        let lr_mid = sched.get_lr(half_max);
        // Due to floating point, this should be approximately 0.5
        assert!(lr_mid > 0.4 && lr_mid < 0.6);
    }
}

// ============================================================================
// COSINE ANNEALING TESTS
// ============================================================================

mod cosine_annealing_tests {
    use super::*;
    use std::f64::consts::PI;

    // --- Basic Behavior: Mathematical Correctness ---

    #[test]
    fn should_return_base_lr_at_step_zero() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 100);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_return_min_lr_at_period_end() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 100);
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn should_return_midpoint_lr_at_half_period() {
        // Cosine formula: min_lr + (base_lr - min_lr) * (1 + cos(pi * progress)) / 2
        // At progress=0.5: (1 + cos(pi/2)) / 2 = (1 + 0) / 2 = 0.5
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 100);
        let lr = sched.get_lr(50);
        assert!((lr - 0.5).abs() < 1e-10);
    }

    #[test]
    fn should_follow_cosine_formula() {
        let base = 1.0;
        let min = 0.1;
        let period = 100;
        let sched = CosineAnnealing::without_restarts(base, min, period);

        for step in [0, 25, 50, 75, 100] {
            let progress = (step as f64) / (period as f64);
            let cosine = (1.0 + (PI * progress).cos()) / 2.0;
            let expected = min + (base - min) * cosine;
            let actual = sched.get_lr(step);
            assert!(
                (actual - expected).abs() < 1e-10,
                "Step {}: expected {}, got {}",
                step,
                expected,
                actual
            );
        }
    }

    // --- Warm Restarts ---

    #[test]
    fn should_restart_at_period_boundary_with_warm_restarts() {
        let sched = CosineAnnealing::new(1.0, 0.0, 50, true);

        // End of first period (step 49) should be near min_lr
        assert!(sched.get_lr(49) < 0.1);

        // Start of second period (step 50) should be at base_lr
        assert!((sched.get_lr(50) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_cycle_multiple_times_with_warm_restarts() {
        let sched = CosineAnnealing::new(1.0, 0.0, 10, true);

        // Test multiple cycles
        for cycle in 0..5 {
            let cycle_start = cycle * 10;
            assert!(
                (sched.get_lr(cycle_start) - 1.0).abs() < 1e-10,
                "Cycle {} start should be 1.0",
                cycle
            );
        }
    }

    #[test]
    fn should_stay_at_min_after_period_without_restarts() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 100);

        // After period ends, should stay at min_lr
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(150) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(1000) - 0.0).abs() < 1e-15);
    }

    // --- Division by Zero: Defensive Behavior ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "period must be > 0"))]
    fn should_reject_period_zero_in_debug() {
        // DEFENSIVE: period=0 triggers panic in debug, returns base_lr in release.
        let sched = CosineAnnealing::new(1.0, 0.0, 0, false);
        // In release: returns base_lr
        let lr = sched.get_lr(0);
        assert!((lr - 1.0).abs() < 1e-15, "Expected base_lr (1.0), got {}", lr);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "period must be > 0"))]
    fn should_reject_period_zero_with_warm_restarts_in_debug() {
        // DEFENSIVE: period=0 triggers panic in debug, returns base_lr in release.
        let sched = CosineAnnealing::new(1.0, 0.0, 0, true);
        // In release: returns base_lr (no panic on % 0)
        let lr = sched.get_lr(1);
        assert!((lr - 1.0).abs() < 1e-15, "Expected base_lr (1.0), got {}", lr);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "period must be > 0"))]
    fn should_handle_period_zero_gracefully_in_release() {
        // DEFENSIVE: period=0 returns base_lr for all steps in release.
        let sched = CosineAnnealing::new(1.0, 0.0, 0, true);
        // In release: all steps return base_lr
        let lr_0 = sched.get_lr(0);
        assert!((lr_0 - 1.0).abs() < 1e-15, "Expected base_lr (1.0), got {}", lr_0);
    }

    // --- Inverted Range Tests ---

    #[test]
    fn should_handle_base_lr_less_than_min_lr() {
        // Mathematically valid but semantically odd - "inverted cosine"
        let sched = CosineAnnealing::without_restarts(0.0, 1.0, 100);
        // At step 0: 1.0 + (0.0 - 1.0) * 1.0 = 0.0
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
        // At step 100: 1.0 + (0.0 - 1.0) * 0.0 = 1.0
        assert!((sched.get_lr(100) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_handle_equal_base_and_min_lr() {
        let sched = CosineAnnealing::without_restarts(0.5, 0.5, 100);
        assert!((sched.get_lr(0) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(100) - 0.5).abs() < 1e-15);
    }

    // --- Non-Finite Value Tests (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "base_lr must be finite"))]
    fn should_reject_nan_base_lr_in_debug() {
        // DEFENSIVE: NaN is rejected in debug, sanitized to 0.0 in release.
        let sched = CosineAnnealing::without_restarts(f64::NAN, 0.0, 100);
        // In release: sanitized to 0.0
        assert!(!sched.get_lr(0).is_nan());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "min_lr must be finite"))]
    fn should_reject_nan_min_lr_in_debug() {
        // DEFENSIVE: NaN is rejected in debug, sanitized to 0.0 in release.
        let sched = CosineAnnealing::without_restarts(1.0, f64::NAN, 100);
        // In release: sanitized to 0.0
        assert!(!sched.get_lr(0).is_nan());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "base_lr must be finite"))]
    fn should_reject_infinity_base_lr_in_debug() {
        // DEFENSIVE: Inf is rejected in debug, sanitized to 0.0 in release.
        let sched = CosineAnnealing::without_restarts(f64::INFINITY, 0.0, 100);
        // In release: sanitized to 0.0
        assert!(sched.get_lr(0).is_finite());
    }

    // --- Negative LR Tests (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "base_lr must be non-negative"))]
    fn should_reject_negative_base_lr_in_debug() {
        // DEFENSIVE: Negative LR is rejected in debug, sanitized to 0.0 in release.
        let sched = CosineAnnealing::without_restarts(-1.0, 0.0, 100);
        // In release: sanitized to 0.0
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
    }

    // --- Step Counter ---

    #[test]
    fn step_should_increment_and_return_correct_lr() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 10);
        let lr0 = sched.step();
        let lr1 = sched.step();
        assert!((lr0 - 1.0).abs() < 1e-10);
        assert!(lr1 < lr0); // Should be decreasing
    }

    #[test]
    fn reset_should_restore_step_to_zero() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 10);
        sched.step();
        sched.step();
        sched.reset();
        assert!((sched.step() - 1.0).abs() < 1e-10);
    }

    // --- Precision Tests ---

    #[test]
    fn should_handle_very_large_period() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 1_000_000_000);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
        assert!((sched.get_lr(500_000_000) - 0.5).abs() < 1e-6);
    }

    // --- Concurrent Access ---

    #[test]
    fn should_handle_concurrent_step_calls() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 1000);

        thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    for _ in 0..100 {
                        let lr = sched.step();
                        assert!(lr.is_finite());
                        assert!(lr >= 0.0 && lr <= 1.0);
                    }
                });
            }
        });
    }
}

// ============================================================================
// POLYNOMIAL DECAY TESTS
// ============================================================================

mod polynomial_decay_tests {
    use super::*;

    // --- Basic Behavior: Mathematical Correctness ---

    #[test]
    fn should_return_base_lr_at_step_zero() {
        let sched = PolynomialDecay::new(1.0, 0.0, 100, 2.0);
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_return_end_lr_at_total_steps() {
        let sched = PolynomialDecay::new(1.0, 0.0, 100, 2.0);
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn should_follow_polynomial_formula() {
        // Formula: end_lr + (base_lr - end_lr) * (1 - progress)^power
        let base = 1.0;
        let end = 0.1;
        let total = 100;
        let power = 2.0;
        let sched = PolynomialDecay::new(base, end, total, power);

        for step in [0, 25, 50, 75, 100] {
            let progress = (step as f64) / (total as f64);
            let decay = (1.0 - progress).powf(power);
            let expected = end + (base - end) * decay;
            let actual = sched.get_lr(step);
            assert!(
                (actual - expected).abs() < 1e-10,
                "Step {}: expected {}, got {}",
                step,
                expected,
                actual
            );
        }
    }

    #[test]
    fn should_match_linear_decay_when_power_is_one() {
        let poly = PolynomialDecay::new(1.0, 0.0, 100, 1.0);
        let linear = LinearDecay::new(1.0, 0.0, 100);

        for step in 0..=100 {
            let poly_lr = poly.get_lr(step);
            let linear_lr = linear.get_lr(step);
            assert!(
                (poly_lr - linear_lr).abs() < 1e-10,
                "Step {}: poly={}, linear={}",
                step,
                poly_lr,
                linear_lr
            );
        }
    }

    // --- Power Edge Cases (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "power must be > 0"))]
    fn should_reject_power_zero_in_debug() {
        // DEFENSIVE: power=0 triggers panic in debug, clamped to 1.0 in release.
        let sched = PolynomialDecay::new(1.0, 0.0, 100, 0.0);
        // In release: power clamped to 1.0 (linear decay)
        // Linear decay: at step 50, lr = 0.5
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "power must be > 0"))]
    fn should_reject_negative_power_in_debug() {
        // DEFENSIVE: negative power triggers panic in debug, clamped to 1.0 in release.
        let sched = PolynomialDecay::new(1.0, 0.0, 100, -2.0);
        // In release: power clamped to 1.0 (linear decay)
        let lr_50 = sched.get_lr(50);
        // Linear decay: at step 50, lr = 0.5
        assert!((lr_50 - 0.5).abs() < 1e-10);
        // Output is clamped, never infinite
        let lr_100 = sched.get_lr(100);
        assert!(lr_100.is_finite());
    }

    #[test]
    fn should_handle_very_large_power() {
        // Large power causes rapid decay
        let sched = PolynomialDecay::new(1.0, 0.0, 100, 100.0);
        // At step 1, progress=0.01, (0.99)^100 is very small
        let lr_1 = sched.get_lr(1);
        assert!(lr_1 < 0.5); // Should have decayed significantly
    }

    #[test]
    fn should_handle_fractional_power() {
        // sqrt decay
        let sched = PolynomialDecay::new(1.0, 0.0, 100, 0.5);
        let lr_50 = sched.get_lr(50);
        // (1 - 0.5)^0.5 = sqrt(0.5) ~ 0.707
        let expected = 0.0 + (1.0 - 0.0) * (0.5_f64).sqrt();
        assert!((lr_50 - expected).abs() < 1e-10);
    }

    // --- Division by Zero (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "total_steps must be > 0"))]
    fn should_reject_total_steps_zero_in_debug() {
        // DEFENSIVE: total_steps=0 triggers panic in debug, returns base_lr in release.
        let sched = PolynomialDecay::new(1.0, 0.0, 0, 2.0);
        let lr = sched.get_lr(0);
        // In release: returns base_lr
        assert!((lr - 1.0).abs() < 1e-15);
    }

    // --- Boundary Tests ---

    #[test]
    fn should_clamp_to_end_lr_after_total_steps() {
        let sched = PolynomialDecay::new(1.0, 0.1, 100, 2.0);
        assert!((sched.get_lr(101) - 0.1).abs() < 1e-15);
        assert!((sched.get_lr(1000) - 0.1).abs() < 1e-15);
    }

    // --- Non-Finite Values (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "base_lr must be finite"))]
    fn should_reject_nan_base_lr_poly_in_debug() {
        // DEFENSIVE: NaN is rejected in debug, sanitized to 0.0 in release.
        let sched = PolynomialDecay::new(f64::NAN, 0.0, 100, 2.0);
        // In release: sanitized to 0.0
        assert!(!sched.get_lr(0).is_nan());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "power must be finite"))]
    fn should_reject_nan_power_in_debug() {
        // DEFENSIVE: NaN power is rejected in debug, clamped to 1.0 in release.
        let sched = PolynomialDecay::new(1.0, 0.0, 100, f64::NAN);
        // In release: power clamped to 1.0 (linear decay)
        let lr_50 = sched.get_lr(50);
        // Linear decay: at step 50, lr = 0.5
        assert!((lr_50 - 0.5).abs() < 1e-10);
    }

    // --- Step Counter ---

    #[test]
    fn step_should_increment_internal_counter() {
        let sched = PolynomialDecay::new(1.0, 0.0, 10, 1.0);
        let lr0 = sched.step();
        let lr1 = sched.step();
        assert!(lr0 > lr1); // Should be decreasing for power > 0
    }
}

// ============================================================================
// WARMUP WRAPPER TESTS
// ============================================================================

mod warmup_tests {
    use super::*;

    // --- Basic Behavior ---

    #[test]
    fn should_start_at_warmup_start_lr() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, 0.0);
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn should_reach_inner_lr_at_warmup_end() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, 0.0);
        assert!((sched.get_lr(100) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_interpolate_linearly_during_warmup() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, 0.0);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(25) - 0.25).abs() < 1e-15);
        assert!((sched.get_lr(75) - 0.75).abs() < 1e-15);
    }

    #[test]
    fn should_delegate_to_inner_after_warmup() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, 0.0);
        assert!((sched.get_lr(100) - 1.0).abs() < 1e-15);
        assert!((sched.get_lr(150) - 1.0).abs() < 1e-15);
        assert!((sched.get_lr(1000) - 1.0).abs() < 1e-15);
    }

    // --- Step Offset Correctness ---

    #[test]
    fn should_pass_adjusted_step_to_inner_scheduler() {
        // After warmup_steps, inner should receive step - warmup_steps
        let inner = LinearDecay::new(1.0, 0.0, 100);
        let sched = Warmup::new(inner, 50, 0.0);

        // At step 50 (warmup ends), inner receives step 0
        assert!((sched.get_lr(50) - 1.0).abs() < 1e-15);

        // At step 100, inner receives step 50 -> mid-decay
        assert!((sched.get_lr(100) - 0.5).abs() < 1e-15);

        // At step 150, inner receives step 100 -> end_lr
        assert!((sched.get_lr(150) - 0.0).abs() < 1e-15);
    }

    // --- Division by Zero ---

    #[test]
    fn should_immediately_delegate_when_warmup_steps_is_zero() {
        // FINDING: warmup_steps=0 never triggers warmup phase
        // The condition `step < warmup_steps` is always false for usize when warmup_steps=0
        // So it immediately delegates to inner scheduler
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 0, 0.0);

        // At step 0: 0 < 0 is false, delegates to inner.get_lr(0 - 0) = inner.get_lr(0)
        let lr = sched.get_lr(0);
        assert!((lr - 1.0).abs() < 1e-15);

        // At step 1: 1 < 0 is false, delegates to inner
        let lr_1 = sched.get_lr(1);
        assert!((lr_1 - 1.0).abs() < 1e-15);
    }

    #[test]
    fn should_never_enter_warmup_when_warmup_steps_is_zero() {
        // When warmup_steps=0, the condition `step < 0` is always false for usize
        let inner = LinearDecay::new(1.0, 0.0, 100);
        let sched = Warmup::new(inner, 0, 0.5); // warmup_start_lr is ignored

        // All steps delegate directly to inner
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
    }

    // --- Inverted Warmup ---

    #[test]
    fn should_handle_cooldown_when_start_greater_than_target() {
        // "Cooldown" instead of warmup: start high, decrease to target
        let inner = ConstantLR::new(0.5);
        let sched = Warmup::new(inner, 100, 1.0);

        assert!((sched.get_lr(0) - 1.0).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.75).abs() < 1e-15);
        assert!((sched.get_lr(100) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn should_handle_equal_start_and_target() {
        let inner = ConstantLR::new(0.5);
        let sched = Warmup::new(inner, 100, 0.5);

        assert!((sched.get_lr(0) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(100) - 0.5).abs() < 1e-15);
    }

    // --- Non-Finite Values (Defensive Behavior) ---

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "warmup_start_lr must be finite"))]
    fn should_reject_nan_warmup_start_lr_in_debug() {
        // DEFENSIVE: NaN warmup_start_lr is rejected in debug, sanitized to 0.0 in release.
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, f64::NAN);
        // In release: sanitized to 0.0
        assert!(!sched.get_lr(0).is_nan());
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "must be finite"))]
    fn should_reject_nan_from_inner_scheduler_in_debug() {
        // DEFENSIVE: Inner scheduler with NaN is rejected in debug.
        let inner = ConstantLR::new(f64::NAN);
        let sched = Warmup::new(inner, 100, 0.0);
        // In release: inner sanitized to 0.0, so warmup works fine
        assert!(!sched.get_lr(50).is_nan());
    }

    // --- Composition Tests ---

    #[test]
    fn should_compose_with_linear_decay() {
        let inner = LinearDecay::new(1.0, 0.1, 100);
        let sched = Warmup::new(inner, 50, 0.0);

        // Warmup phase
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(25) - 0.5).abs() < 1e-15);
        assert!((sched.get_lr(50) - 1.0).abs() < 1e-15);

        // Decay phase
        assert!((sched.get_lr(100) - 0.55).abs() < 1e-10);
        assert!((sched.get_lr(150) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn should_compose_with_cosine_annealing() {
        let inner = CosineAnnealing::without_restarts(1.0, 0.0, 100);
        let sched = Warmup::new(inner, 50, 0.0);

        // Warmup
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(50) - 1.0).abs() < 1e-10);

        // Cosine decay
        assert!((sched.get_lr(100) - 0.5).abs() < 0.01);
        assert!((sched.get_lr(150) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn should_compose_with_polynomial_decay() {
        let inner = PolynomialDecay::new(1.0, 0.0, 100, 2.0);
        let sched = Warmup::new(inner, 50, 0.0);

        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(50) - 1.0).abs() < 1e-10);
        // After warmup, polynomial decay begins
    }

    #[test]
    fn should_handle_nested_warmup() {
        // Warmup<Warmup<ConstantLR>>
        let inner = ConstantLR::new(1.0);
        let warmup1 = Warmup::new(inner, 50, 0.5);
        let warmup2 = Warmup::new(warmup1, 50, 0.0);

        // Outer warmup: 0 -> warmup1.get_lr(0) = 0.5
        assert!((warmup2.get_lr(0) - 0.0).abs() < 1e-15);
        assert!((warmup2.get_lr(25) - 0.25).abs() < 1e-15);
        assert!((warmup2.get_lr(50) - 0.5).abs() < 1e-15);

        // After outer warmup, delegates to warmup1 with adjusted step
        // warmup1.get_lr(0) = 0.5, warmup1.get_lr(50) = 1.0
        assert!((warmup2.get_lr(100) - 1.0).abs() < 1e-15);
    }

    // --- Step Counter ---

    #[test]
    fn step_should_increment_internal_counter() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 10, 0.0);

        let lr0 = sched.step(); // step 0
        let lr5 = sched.step(); // step 1
        assert!(lr5 > lr0); // Should be increasing during warmup
    }

    #[test]
    fn reset_should_restore_step_to_zero() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 10, 0.0);

        for _ in 0..5 {
            sched.step();
        }
        sched.reset();
        assert!((sched.step() - 0.0).abs() < 1e-15); // Back to step 0
    }

    // --- Inner Scheduler Access ---

    #[test]
    fn should_expose_inner_scheduler() {
        let inner = ConstantLR::new(0.123);
        let sched = Warmup::new(inner, 100, 0.0);

        assert!((sched.inner().get_lr(0) - 0.123).abs() < 1e-15);
    }

    // --- Concurrent Access ---

    #[test]
    fn should_handle_concurrent_step_calls() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 1000, 0.0);

        thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| {
                    for _ in 0..100 {
                        let lr = sched.step();
                        assert!(lr.is_finite());
                        assert!(lr >= 0.0 && lr <= 1.0);
                    }
                });
            }
        });
    }
}

// ============================================================================
// LR SCHEDULER TRAIT TESTS
// ============================================================================

mod trait_tests {
    use super::*;

    #[test]
    fn all_schedulers_should_implement_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ConstantLR>();
        assert_send_sync::<LinearDecay>();
        assert_send_sync::<CosineAnnealing>();
        assert_send_sync::<PolynomialDecay>();
        assert_send_sync::<Warmup<ConstantLR>>();
        assert_send_sync::<Warmup<LinearDecay>>();
    }

    #[test]
    fn constant_lr_default_step_returns_get_lr_zero() {
        // ConstantLR uses the default trait implementation of step()
        let sched = ConstantLR::new(0.001);
        // Default step() just calls get_lr(0)
        assert_eq!(sched.step(), sched.get_lr(0));
    }
}

// ============================================================================
// PROPERTY-BASED TESTS
// ============================================================================

mod property_tests {
    use super::*;

    // --- Monotonicity ---

    #[test]
    fn linear_decay_should_be_monotonically_decreasing_when_start_gt_end() {
        let sched = LinearDecay::new(1.0, 0.0, 100);
        let mut prev_lr = sched.get_lr(0);

        for step in 1..=100 {
            let lr = sched.get_lr(step);
            assert!(
                lr <= prev_lr,
                "Step {}: lr {} should be <= prev {}",
                step,
                lr,
                prev_lr
            );
            prev_lr = lr;
        }
    }

    #[test]
    fn linear_decay_should_be_monotonically_increasing_when_start_lt_end() {
        let sched = LinearDecay::new(0.0, 1.0, 100);
        let mut prev_lr = sched.get_lr(0);

        for step in 1..=100 {
            let lr = sched.get_lr(step);
            assert!(
                lr >= prev_lr,
                "Step {}: lr {} should be >= prev {}",
                step,
                lr,
                prev_lr
            );
            prev_lr = lr;
        }
    }

    #[test]
    fn cosine_annealing_should_be_monotonically_decreasing_without_restarts() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 100);
        let mut prev_lr = sched.get_lr(0);

        for step in 1..=100 {
            let lr = sched.get_lr(step);
            assert!(
                lr <= prev_lr + 1e-10, // Small tolerance for floating point
                "Step {}: lr {} should be <= prev {}",
                step,
                lr,
                prev_lr
            );
            prev_lr = lr;
        }
    }

    #[test]
    fn warmup_should_be_monotonically_increasing_during_warmup_phase() {
        let inner = ConstantLR::new(1.0);
        let sched = Warmup::new(inner, 100, 0.0);
        let mut prev_lr = sched.get_lr(0);

        for step in 1..100 {
            let lr = sched.get_lr(step);
            assert!(
                lr >= prev_lr,
                "Step {}: lr {} should be >= prev {}",
                step,
                lr,
                prev_lr
            );
            prev_lr = lr;
        }
    }

    // --- Boundedness ---

    #[test]
    fn linear_decay_should_stay_within_bounds() {
        let start = 1.0;
        let end = 0.1;
        let sched = LinearDecay::new(start, end, 100);

        for step in 0..200 {
            let lr = sched.get_lr(step);
            // Use small epsilon for floating point comparison
            assert!(
                lr >= end - 1e-10 && lr <= start + 1e-10,
                "Step {}: lr {} out of bounds [{}, {}]",
                step,
                lr,
                end,
                start
            );
        }
    }

    #[test]
    fn cosine_annealing_should_stay_within_bounds() {
        let base = 1.0;
        let min = 0.1;
        let sched = CosineAnnealing::without_restarts(base, min, 100);

        for step in 0..200 {
            let lr = sched.get_lr(step);
            assert!(
                lr >= min && lr <= base,
                "Step {}: lr {} out of bounds [{}, {}]",
                step,
                lr,
                min,
                base
            );
        }
    }

    #[test]
    fn polynomial_decay_should_stay_within_bounds_for_positive_power() {
        let base = 1.0;
        let end = 0.1;
        let sched = PolynomialDecay::new(base, end, 100, 2.0);

        for step in 0..200 {
            let lr = sched.get_lr(step);
            assert!(
                lr >= end && lr <= base,
                "Step {}: lr {} out of bounds [{}, {}]",
                step,
                lr,
                end,
                base
            );
        }
    }

    // --- Continuity ---

    #[test]
    fn linear_decay_should_have_small_changes_for_small_step_changes() {
        let sched = LinearDecay::new(1.0, 0.0, 1000);

        for step in 0..999 {
            let lr1 = sched.get_lr(step);
            let lr2 = sched.get_lr(step + 1);
            let delta = (lr1 - lr2).abs();
            assert!(
                delta < 0.01, // Max change should be 1/1000 = 0.001
                "Step {}: delta {} too large",
                step,
                delta
            );
        }
    }

    #[test]
    fn cosine_annealing_should_have_bounded_changes() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 1000);

        for step in 0..999 {
            let lr1 = sched.get_lr(step);
            let lr2 = sched.get_lr(step + 1);
            let delta = (lr1 - lr2).abs();
            // Cosine derivative is bounded by pi/period
            assert!(
                delta < 0.01,
                "Step {}: delta {} too large",
                step,
                delta
            );
        }
    }

    // --- Idempotency ---

    #[test]
    fn get_lr_should_be_idempotent() {
        let sched = LinearDecay::new(1.0, 0.0, 100);

        for step in 0..110 {
            let lr1 = sched.get_lr(step);
            let lr2 = sched.get_lr(step);
            assert_eq!(lr1, lr2, "get_lr should be idempotent at step {}", step);
        }
    }

    // --- Consistency Between get_lr and step ---

    #[test]
    fn step_should_return_same_value_as_get_lr_for_current_step() {
        let sched = LinearDecay::new(1.0, 0.0, 100);

        // First call to step() returns get_lr(0) and increments to 1
        let step_result = sched.step();
        let get_lr_result = sched.get_lr(0);
        assert_eq!(step_result, get_lr_result);

        // Second call returns get_lr(1) and increments to 2
        let step_result = sched.step();
        let get_lr_result = sched.get_lr(1);
        assert_eq!(step_result, get_lr_result);
    }
}

// ============================================================================
// STRESS TESTS
// ============================================================================

mod stress_tests {
    use super::*;

    #[test]
    fn should_handle_rapid_concurrent_stepping() {
        let sched = LinearDecay::new(1.0, 0.0, 100_000);

        thread::scope(|s| {
            for _ in 0..100 {
                s.spawn(|| {
                    for _ in 0..1000 {
                        let lr = sched.step();
                        assert!(lr.is_finite());
                    }
                });
            }
        });

        // After 100 * 1000 = 100,000 steps, should be at end_lr
        let final_lr = sched.step();
        assert!((final_lr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn should_handle_interleaved_step_and_get_lr() {
        let sched = CosineAnnealing::without_restarts(1.0, 0.0, 1000);

        thread::scope(|s| {
            // Thread 1: Calls step()
            s.spawn(|| {
                for _ in 0..500 {
                    let lr = sched.step();
                    assert!(lr.is_finite());
                    assert!(lr >= 0.0 && lr <= 1.0);
                }
            });

            // Thread 2: Calls get_lr() with various steps
            s.spawn(|| {
                for step in 0..500 {
                    let lr = sched.get_lr(step);
                    assert!(lr.is_finite());
                    assert!(lr >= 0.0 && lr <= 1.0);
                }
            });
        });
    }

    #[test]
    fn should_handle_reset_during_concurrent_access() {
        let sched = LinearDecay::new(1.0, 0.0, 100);

        thread::scope(|s| {
            // Multiple steppers
            for _ in 0..5 {
                s.spawn(|| {
                    for _ in 0..200 {
                        let _ = sched.step();
                    }
                });
            }

            // Periodic resetter
            s.spawn(|| {
                for _ in 0..50 {
                    sched.reset();
                    std::thread::yield_now();
                }
            });
        });

        // Should still be functional
        let lr = sched.get_lr(0);
        assert!((lr - 1.0).abs() < 1e-15);
    }
}

// ============================================================================
// NUMERICAL EDGE CASE TESTS
// ============================================================================

mod numerical_edge_cases {
    use super::*;

    #[test]
    fn should_handle_subnormal_lr_values() {
        // Subnormal (denormalized) floating point numbers
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let sched = ConstantLR::new(subnormal);
        assert_eq!(sched.get_lr(0), subnormal);
    }

    #[test]
    fn should_handle_very_close_start_and_end_lr() {
        // Very small difference between start and end
        let sched = LinearDecay::new(1.0, 1.0 - 1e-15, 100);
        let lr_50 = sched.get_lr(50);
        // Should not lose precision
        assert!(lr_50 > 1.0 - 1e-14 && lr_50 < 1.0);
    }

    #[test]
    fn should_handle_max_f64_values() {
        // Note: f64::MAX may overflow in calculations - scheduler should clamp
        let sched = LinearDecay::new(1e15, 1e14, 100);
        let lr = sched.get_lr(0);
        assert!((lr - 1e15).abs() < 1e10);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "end_lr must be non-negative"))]
    fn should_reject_extreme_f64_range_in_debug() {
        // DEFENSIVE: Negative LRs like -f64::MAX are rejected in debug.
        let sched = LinearDecay::new(f64::MAX, -f64::MAX, 100);
        // In release: sanitized to 0.0
        let lr_0 = sched.get_lr(0);
        assert!(lr_0.is_finite(), "Expected finite LR in release, got {}", lr_0);
    }

    #[test]
    fn should_handle_negative_zero() {
        let sched = ConstantLR::new(-0.0);
        let lr = sched.get_lr(0);
        assert_eq!(lr, 0.0); // -0.0 == 0.0 in IEEE 754
    }

    #[test]
    fn should_preserve_precision_at_large_step_counts() {
        // usize::MAX cast to f64 loses precision
        let sched = LinearDecay::new(1.0, 0.0, usize::MAX);

        // At large steps, floating point precision is limited
        // usize::MAX as f64 has ~15 significant digits
        let lr = sched.get_lr(usize::MAX / 2);
        // Should still be approximately 0.5
        assert!(lr > 0.4 && lr < 0.6);
    }
}

// ============================================================================
// DOCUMENTATION TESTS (Behavior Specification)
// ============================================================================

mod behavioral_specification {
    use super::*;

    /// Demonstrates the complete lifecycle of a LinearDecay scheduler.
    #[test]
    fn linear_decay_lifecycle() {
        // 1. Creation
        let sched = LinearDecay::new(0.001, 0.0001, 1000);

        // 2. Initial state: step counter at 0
        assert!((sched.get_lr(0) - 0.001).abs() < 1e-10);

        // 3. Using step() increments internal counter
        let lr1 = sched.step(); // Returns LR for step 0, increments to 1
        assert!((lr1 - 0.001).abs() < 1e-10);

        let lr2 = sched.step(); // Returns LR for step 1, increments to 2
        assert!(lr2 < lr1);

        // 4. get_lr() does NOT affect internal counter
        let lr_100 = sched.get_lr(100);
        let lr_100_again = sched.get_lr(100);
        assert_eq!(lr_100, lr_100_again);

        // 5. reset() restores counter to 0
        sched.reset();
        let lr_after_reset = sched.step();
        assert!((lr_after_reset - 0.001).abs() < 1e-10);
    }

    /// Demonstrates warmup + decay composition.
    #[test]
    fn warmup_with_decay_lifecycle() {
        // Create a scheduler that warms up over 100 steps, then decays over 1000
        let inner = LinearDecay::new(0.001, 0.0001, 1000);
        let sched = Warmup::new(inner, 100, 0.0);

        // Phase 1: Warmup (steps 0-99)
        // LR increases from 0.0 to inner.get_lr(0) = 0.001
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-15);
        assert!((sched.get_lr(50) - 0.0005).abs() < 1e-10);
        assert!((sched.get_lr(99) - 0.00099).abs() < 1e-8);

        // Phase 2: Transition (step 100)
        // Reaches inner.get_lr(0) = 0.001
        assert!((sched.get_lr(100) - 0.001).abs() < 1e-10);

        // Phase 3: Decay (steps 100+)
        // Inner scheduler receives step - 100
        // At step 1100, inner sees step 1000, returns 0.0001
        assert!((sched.get_lr(1100) - 0.0001).abs() < 1e-10);
    }

    /// Demonstrates cosine annealing with warm restarts.
    #[test]
    fn cosine_warm_restarts_lifecycle() {
        let sched = CosineAnnealing::new(1.0, 0.1, 100, true);

        // Cycle 1: steps 0-99
        assert!((sched.get_lr(0) - 1.0).abs() < 1e-10);
        assert!(sched.get_lr(50) < 0.6 && sched.get_lr(50) > 0.4);
        assert!(sched.get_lr(99) < 0.2);

        // Cycle 2: steps 100-199 (restart)
        assert!((sched.get_lr(100) - 1.0).abs() < 1e-10);
        assert!(sched.get_lr(150) < 0.6 && sched.get_lr(150) > 0.4);

        // Cycle 3: steps 200-299 (another restart)
        assert!((sched.get_lr(200) - 1.0).abs() < 1e-10);
    }
}
