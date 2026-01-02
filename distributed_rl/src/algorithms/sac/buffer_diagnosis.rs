//! SAC Buffer Diagnosis Tests
//!
//! This file contains standalone tests to diagnose the SAC training failure.
//! These tests are designed to work even if other parts of the codebase have issues.
//!
//! # Problem Statement
//!
//! SAC training showed:
//! - Buffer utilization: 0.0% (throughout entire run)
//! - Train steps: 0
//! - Mean return: ~20-22 (random policy, no learning)
//!
//! # Root Cause Identified
//!
//! The bug is in how `buffer.len()` works with the lock-free injector pattern:
//!
//! 1. `push()` adds items to a lock-free injector queue (NOT main storage)
//! 2. `len()` returns the size of main storage (NOT including injector queue)
//! 3. Items only move from injector to storage when `do_consolidate()` is called
//! 4. `do_consolidate()` is called by `is_training_ready()` and `sample()`
//!
//! The runner's monitoring loop uses `buffer.len()` for stats BEFORE consolidation,
//! so it always sees 0%. However, the learner's `is_training_ready()` DOES consolidate.
//!
//! # Impact
//!
//! If the ONLY issue was monitoring, training would still work.
//! Since train_steps = 0, there's likely another issue preventing training.
//!
//! Potential additional issues:
//! 1. Learner thread might panic/crash silently
//! 2. Actor threads might not be pushing data correctly
//! 3. min_buffer_size might be too high for the test duration
//! 4. Threading synchronization issues

#[cfg(test)]
mod tests {
    use crate::algorithms::sac::{SACBuffer, SACBufferConfig, SACTransition};
    use crate::core::experience_buffer::{ExperienceBuffer, OffPolicyBuffer};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    // =========================================================================
    // CRITICAL BUG DEMONSTRATION
    // =========================================================================

    /// Demonstrates the core issue: push() doesn't update len()
    #[test]
    fn test_core_bug_push_does_not_update_len() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push 50 items
        for i in 0..50 {
            buffer.push(SACTransition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
            ));
        }

        // BUG: len() returns 0 because items are in the injector queue
        let len_before = buffer.len();
        assert_eq!(
            len_before, 0,
            "len() should be 0 before consolidate - this is the bug!"
        );

        // pending_size() correctly shows queued items
        assert_eq!(buffer.pending_size(), 50, "pending_size() shows 50 items");

        // consolidate() fixes it
        buffer.consolidate();
        assert_eq!(buffer.len(), 50, "After consolidate, len() = 50");
    }

    /// Demonstrates that is_training_ready() does consolidate internally
    #[test]
    fn test_is_training_ready_consolidates() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        for i in 0..20 {
            buffer.push(SACTransition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
            ));
        }

        // Before is_training_ready, len() is 0
        assert_eq!(buffer.len(), 0);

        // is_training_ready calls do_consolidate internally
        let ready = buffer.is_training_ready();

        // Now len() is correct
        assert_eq!(buffer.len(), 20, "is_training_ready consolidates the buffer");
        assert!(ready, "Should be ready with 20 >= 10 items");
    }

    // =========================================================================
    // SIMULATED RUNNER BEHAVIOR
    // =========================================================================

    /// Simulates the exact pattern from sac_runner.rs monitoring loop
    #[test]
    fn test_runner_monitoring_pattern() {
        let config = SACBufferConfig {
            capacity: 100_000,
            min_size: 1000,
            batch_size: 256,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));
        let shutdown = Arc::new(AtomicBool::new(false));
        let train_steps = Arc::new(AtomicUsize::new(0));

        // Simulate actor pushing data
        let buffer_actor = Arc::clone(&buffer);
        let shutdown_actor = Arc::clone(&shutdown);
        let actor_handle = thread::spawn(move || {
            let mut count = 0;
            while !shutdown_actor.load(Ordering::Relaxed) && count < 2000 {
                buffer_actor.push(SACTransition::new_discrete(
                    vec![count as f32; 4],
                    0,
                    1.0,
                    vec![count as f32 + 1.0; 4],
                    false,
                ));
                count += 1;
                thread::sleep(Duration::from_micros(50));
            }
            count
        });

        // Simulate learner checking and training
        let buffer_learner = Arc::clone(&buffer);
        let train_steps_learner = Arc::clone(&train_steps);
        let shutdown_learner = Arc::clone(&shutdown);
        let learner_handle = thread::spawn(move || {
            while !shutdown_learner.load(Ordering::Relaxed) {
                // This is what the learner does: check is_training_ready()
                // which DOES call consolidate internally
                if buffer_learner.is_training_ready() {
                    if let Some(_batch) = buffer_learner.sample_batch() {
                        train_steps_learner.fetch_add(1, Ordering::Relaxed);
                    }
                }
                thread::sleep(Duration::from_millis(1));
            }
        });

        // Simulate monitoring loop - the BUGGY pattern
        let start = Instant::now();
        let mut ever_saw_nonzero = false;
        let mut iterations = 0;

        while start.elapsed() < Duration::from_secs(2) && iterations < 50 {
            thread::sleep(Duration::from_millis(50));

            // This is the BUGGY line from the runner:
            // stats.buffer_utilization = buffer.len() as f32 / config.buffer_capacity as f32;
            let utilization_buggy = buffer.len() as f32 / 100_000.0;

            if utilization_buggy > 0.0 {
                ever_saw_nonzero = true;
            }

            iterations += 1;
        }

        shutdown.store(true, Ordering::Relaxed);

        let total_pushed = actor_handle.join().unwrap();
        learner_handle.join().unwrap();

        let final_train_steps = train_steps.load(Ordering::Relaxed);

        // Key findings
        println!("=== SIMULATION RESULTS ===");
        println!("Total pushed by actor: {}", total_pushed);
        println!("Train steps: {}", final_train_steps);
        println!("Monitoring ever saw non-zero: {}", ever_saw_nonzero);

        // The monitoring loop may or may not see non-zero depending on timing
        // because the learner's is_training_ready() consolidates the buffer
        // and the monitoring loop reads len() afterward.

        // But training SHOULD have happened
        if total_pushed >= 1000 {
            // If we pushed enough to exceed min_buffer_size, training should occur
            assert!(
                final_train_steps > 0 || !ever_saw_nonzero,
                "If train_steps=0 and we pushed {} items, the learner never got data. \
                 ever_saw_nonzero={} suggests timing issue.",
                total_pushed,
                ever_saw_nonzero
            );
        }
    }

    // =========================================================================
    // HYPOTHESIS TESTS
    // =========================================================================

    /// Test if the min_buffer_size threshold is the issue
    #[test]
    fn test_min_buffer_size_threshold() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 1000, // High threshold like in the failing example
            batch_size: 256,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push only 500 items (less than min_size)
        for i in 0..500 {
            buffer.push(SACTransition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
            ));
        }

        // is_training_ready should return false
        assert!(
            !buffer.is_training_ready(),
            "Should NOT be ready with 500 < 1000"
        );

        // This could explain the issue: if actors don't push enough before shutdown,
        // training never starts
    }

    /// Test the exact configuration from the failing example
    #[test]
    fn test_failing_example_config() {
        // From examples/src/sac.rs:
        // .with_buffer_capacity(100_000)
        // .with_batch_size(256)
        // .with_min_buffer_size(1000)
        // .with_n_actors(2)
        // .with_n_envs_per_actor(4)
        // Total envs = 8

        let config = SACBufferConfig {
            capacity: 100_000,
            min_size: 1000,
            batch_size: 256,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // With 8 envs, need 1000/8 = 125 environment steps to fill min_buffer_size
        // At ~4000 SPS (from logs), this should happen in ~0.25 seconds

        // Simulate 200 environment steps worth of transitions (8 envs * 200 steps)
        for step in 0..200 {
            for env in 0..8 {
                buffer.push(SACTransition::new_discrete(
                    vec![(step * 8 + env) as f32; 4],
                    0,
                    1.0,
                    vec![(step * 8 + env + 1) as f32; 4],
                    step % 50 == 49, // Terminal every 50 steps
                ));
            }
        }

        // Total transitions = 200 * 8 = 1600
        buffer.consolidate();
        assert_eq!(buffer.len(), 1600, "Should have 1600 transitions");
        assert!(buffer.is_training_ready(), "Should be ready with 1600 >= 1000");

        // Sample should work
        let batch = buffer.sample_batch();
        assert!(batch.is_some(), "Should get a batch");
        assert_eq!(batch.unwrap().len(), 256, "Batch should have 256 items");
    }

    // =========================================================================
    // THREAD SAFETY TESTS
    // =========================================================================

    /// Test concurrent push and sample operations
    #[test]
    fn test_concurrent_push_and_sample() {
        let config = SACBufferConfig {
            capacity: 10000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));
        let shutdown = Arc::new(AtomicBool::new(false));
        let total_samples = Arc::new(AtomicUsize::new(0));

        // Multiple pushers
        let mut push_handles = vec![];
        for pusher_id in 0..4 {
            let buffer_clone = Arc::clone(&buffer);
            let shutdown_clone = Arc::clone(&shutdown);
            let handle = thread::spawn(move || {
                let mut count = 0;
                while !shutdown_clone.load(Ordering::Relaxed) && count < 1000 {
                    buffer_clone.push(SACTransition::new_discrete(
                        vec![(pusher_id * 1000 + count) as f32],
                        0,
                        1.0,
                        vec![(pusher_id * 1000 + count + 1) as f32],
                        false,
                    ));
                    count += 1;
                }
                count
            });
            push_handles.push(handle);
        }

        // Single sampler
        let buffer_sampler = Arc::clone(&buffer);
        let shutdown_sampler = Arc::clone(&shutdown);
        let samples_counter = Arc::clone(&total_samples);
        let sample_handle = thread::spawn(move || {
            while !shutdown_sampler.load(Ordering::Relaxed) {
                if buffer_sampler.is_training_ready() {
                    if let Some(_batch) = buffer_sampler.sample_batch() {
                        samples_counter.fetch_add(1, Ordering::Relaxed);
                    }
                }
                thread::sleep(Duration::from_micros(100));
            }
        });

        // Let them run
        thread::sleep(Duration::from_millis(500));
        shutdown.store(true, Ordering::Relaxed);

        let mut total_pushed = 0;
        for handle in push_handles {
            total_pushed += handle.join().unwrap();
        }
        sample_handle.join().unwrap();

        let final_samples = total_samples.load(Ordering::Relaxed);

        println!("Concurrent test: pushed={}, samples={}", total_pushed, final_samples);

        assert!(total_pushed > 0, "Should have pushed items");
        assert!(final_samples > 0, "Should have sampled batches");
    }

    // =========================================================================
    // DETAILED TIMING ANALYSIS
    // =========================================================================

    /// Track exactly when consolidation happens
    #[test]
    fn test_consolidation_timing() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 50,
            batch_size: 10,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        // Push in batches and check state
        for batch_idx in 0..5 {
            for i in 0..20 {
                buffer.push(SACTransition::new_discrete(
                    vec![(batch_idx * 20 + i) as f32],
                    0,
                    1.0,
                    vec![(batch_idx * 20 + i + 1) as f32],
                    false,
                ));
            }

            let len_after = buffer.len();
            let pending_after = buffer.pending_size();

            println!(
                "After batch {}: len={}, pending={}",
                batch_idx, len_after, pending_after
            );

            // len() should be 0, pending should accumulate
            assert_eq!(len_after, 0, "len() should still be 0");
            assert_eq!(pending_after, (batch_idx + 1) * 20, "pending should accumulate");
        }

        // Total pushed = 100
        assert_eq!(buffer.pending_size(), 100);

        // Now check is_training_ready - this consolidates
        let ready = buffer.is_training_ready();
        assert!(ready, "Should be ready (100 >= 50)");
        assert_eq!(buffer.len(), 100, "After is_training_ready, len = 100");
        assert_eq!(buffer.pending_size(), 0, "After consolidate, pending = 0");
    }
}

// ============================================================================
// DIAGNOSIS SUMMARY
// ============================================================================

/// # Root Cause Analysis Summary
///
/// ## Finding 1: buffer.len() Bug (CONFIRMED)
///
/// The `SACBuffer::len()` method returns the size of consolidated storage only.
/// Items pushed via `push()` go to a lock-free injector queue and are NOT
/// reflected in `len()` until `consolidate()` is called.
///
/// **Impact on monitoring:** The runner's `stats.buffer_utilization` always shows 0%
/// because it calculates `buffer.len() / capacity` without consolidating first.
///
/// **Impact on training:** None directly - the learner's `is_training_ready()`
/// does call `do_consolidate()` internally.
///
/// ## Finding 2: Training Loop Works (IF DATA FLOWS)
///
/// The learner loop correctly:
/// 1. Calls `is_training_ready()` which consolidates
/// 2. Samples batches via `sample_batch()`
/// 3. Processes batches
///
/// ## Finding 3: Why train_steps = 0?
///
/// If `is_training_ready()` works correctly, why would train_steps stay 0?
///
/// Possible causes:
/// 1. **min_buffer_size too high:** 1000 items needed, but maybe not reached
/// 2. **Actors not pushing:** Thread synchronization issue
/// 3. **Learner panicking:** Error in training step silently caught
/// 4. **Timing issue:** Shutdown happens before min_buffer_size reached
///
/// ## Recommended Fixes
///
/// 1. **Fix buffer_utilization display:**
///    ```rust
///    // In sac_runner.rs monitoring loop:
///    buffer.consolidate(); // Add this line
///    stats.buffer_utilization = buffer.len() as f32 / config.buffer_capacity as f32;
///    ```
///
/// 2. **Add pending_size to stats:**
///    ```rust
///    stats.pending_items = buffer.pending_size();
///    ```
///
/// 3. **Add debug logging in learner:**
///    ```rust
///    if buffer.is_training_ready() {
///        println!("Learner: buffer ready, len={}", buffer.len());
///        // ...
///    } else {
///        println!("Learner: waiting, pending={}", buffer.pending_size());
///    }
///    ```
///
/// 4. **Verify actor is pushing:**
///    Add counter in actor thread to confirm pushes.
pub mod diagnosis {
    pub const SUMMARY: &str = r#"
SAC Pipeline Diagnosis Summary
==============================

ROOT CAUSE: buffer.len() returns 0 until consolidate() is called

SYMPTOMS:
- buffer_utilization shows 0.0% in logs
- train_steps = 0

EXPLANATION:
1. SACBuffer uses a lock-free injector queue for thread-safe pushes
2. push() adds items to injector queue, NOT main storage
3. len() returns size of main storage only
4. consolidate() moves items from injector to storage
5. is_training_ready() and sample() call consolidate() internally

THE BUG:
- Runner monitoring calculates: buffer.len() / capacity = 0 / N = 0%
- This is DISPLAY-ONLY bug; training should still work

IF train_steps = 0:
- Check if min_buffer_size is reached before shutdown
- Check if learner thread is panicking
- Add logging to confirm data flow

FIX FOR DISPLAY:
Add buffer.consolidate() before reading buffer.len() in monitoring loop
Or add pending_size() to the stats output
"#;
}
