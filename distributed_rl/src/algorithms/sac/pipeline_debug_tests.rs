//! Comprehensive SAC Pipeline Debug Tests
//!
//! These tests are designed to diagnose the SAC training failure where:
//! - Buffer utilization stays at 0.0%
//! - Train steps: 0
//! - No learning occurs
//!
//! Following the "Tests as Definition: the Yoneda Way" philosophy, these tests
//! exhaustively explore all interactions to pinpoint the exact failure point.

use super::*;
use crate::algorithms::sac::{SACBuffer, SACBufferConfig, SACTransition, SACTransitionTrait};
use crate::core::experience_buffer::{ExperienceBuffer, OffPolicyBuffer};
use crate::core::transition::Transition;
use burn::backend::{Autodiff, NdArray};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

type B = Autodiff<NdArray<f32>>;

// ============================================================================
// PART 1: SACBuffer Core Behavior Tests
// ============================================================================

mod buffer_core_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test: len() returns 0 for newly created buffer
    // -------------------------------------------------------------------------
    #[test]
    fn test_new_buffer_has_zero_len() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        assert_eq!(buffer.len(), 0, "New buffer should have len() == 0");
        assert_eq!(
            buffer.current_size(),
            0,
            "New buffer should have current_size() == 0"
        );
        assert_eq!(
            buffer.pending_size(),
            0,
            "New buffer should have pending_size() == 0"
        );
    }

    // -------------------------------------------------------------------------
    // BUG HYPOTHESIS TEST: push() does NOT immediately update len()
    // -------------------------------------------------------------------------
    #[test]
    fn test_push_does_not_immediately_update_len() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push one transition
        let transition = SACTransition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], false);
        buffer.push(transition);

        // BUG DETECTED: len() returns 0 even though we just pushed an item!
        // This is because push() uses the lock-free injector queue,
        // and len() returns self.size which is only updated on consolidate().
        let len_after_push = buffer.len();

        // Document the behavior: len() does NOT reflect pending items
        assert_eq!(
            len_after_push, 0,
            "BUG DETECTED: len() returns 0 immediately after push() because \
             push() only adds to the injector queue, not the main storage. \
             This is the ROOT CAUSE of buffer_utilization showing 0%!"
        );

        // However, pending_size() DOES track pushed items
        assert_eq!(
            buffer.pending_size(),
            1,
            "pending_size() should track items in injector queue"
        );

        // After consolidate, len() should be correct
        buffer.consolidate();
        assert_eq!(
            buffer.len(),
            1,
            "After consolidate(), len() should reflect pushed items"
        );
    }

    // -------------------------------------------------------------------------
    // Test: Multiple pushes accumulate in pending, not in len()
    // -------------------------------------------------------------------------
    #[test]
    fn test_multiple_pushes_accumulate_in_pending_not_len() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        for i in 0..50 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // BUG: len() is still 0!
        assert_eq!(
            buffer.len(),
            0,
            "BUG: After 50 pushes, len() is still 0 because no consolidation happened"
        );

        // But pending_size correctly tracks all 50 items
        assert_eq!(buffer.pending_size(), 50, "pending_size() should be 50");

        // consolidate() fixes this
        buffer.consolidate();
        assert_eq!(buffer.len(), 50, "After consolidate(), len() should be 50");
        assert_eq!(
            buffer.pending_size(),
            0,
            "After consolidate(), pending_size() should be 0"
        );
    }

    // -------------------------------------------------------------------------
    // Test: is_training_ready() DOES call consolidate (implicit)
    // -------------------------------------------------------------------------
    #[test]
    fn test_is_training_ready_calls_consolidate() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push 15 transitions (more than min_size of 10)
        for i in 0..15 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // Before is_training_ready(), len() is 0
        assert_eq!(buffer.len(), 0, "len() should be 0 before is_training_ready()");

        // is_training_ready() calls do_consolidate() internally
        let ready = buffer.is_training_ready();

        // After is_training_ready(), len() is updated
        assert_eq!(
            buffer.len(),
            15,
            "After is_training_ready(), len() should reflect consolidated items"
        );
        assert!(ready, "Buffer should be ready (15 >= 10)");
    }

    // -------------------------------------------------------------------------
    // CRITICAL TEST: Sequence of events in training loop
    // -------------------------------------------------------------------------
    #[test]
    fn test_training_loop_sequence_exposes_bug() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        // Simulate what the runner's monitoring loop does:
        // It reads buffer.len() WITHOUT calling consolidate first!

        let buffer_clone = Arc::clone(&buffer);

        // Spawn a thread that pushes data (like an actor thread would)
        let actor_handle = thread::spawn(move || {
            for i in 0..20 {
                let transition =
                    SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
                buffer_clone.push(transition);
                thread::sleep(Duration::from_millis(10));
            }
        });

        // Give actor thread time to push some data
        thread::sleep(Duration::from_millis(100));

        // This is what the monitoring loop does:
        // stats.buffer_utilization = buffer.len() as f32 / config.buffer_capacity as f32;
        let utilization = buffer.len() as f32 / 100.0;

        // BUG: utilization is 0% even though actor is pushing data!
        // This is because the monitoring loop reads len() which doesn't include pending items
        assert_eq!(
            utilization, 0.0,
            "BUG CONFIRMED: Monitoring loop sees 0% utilization even while data is being pushed"
        );

        actor_handle.join().unwrap();

        // After all pushes, pending_size is 20
        assert_eq!(buffer.pending_size(), 20, "pending_size should be 20");

        // But len() is still 0!
        assert_eq!(
            buffer.len(),
            0,
            "BUG: len() is still 0 even after all actor pushes complete"
        );
    }

    // -------------------------------------------------------------------------
    // Test: sample_batch() does consolidate
    // -------------------------------------------------------------------------
    #[test]
    fn test_sample_batch_consolidates() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        for i in 0..15 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // len() is 0 before sample
        assert_eq!(buffer.len(), 0);

        // sample_batch calls sample which calls do_consolidate
        let batch = buffer.sample_batch();

        assert!(batch.is_some(), "Should be able to sample after consolidation");
        assert_eq!(batch.unwrap().len(), 5, "Batch size should be 5");

        // len() is now updated
        assert_eq!(buffer.len(), 15, "len() should be 15 after sample");
    }
}

// ============================================================================
// PART 2: Actor -> Buffer Data Flow Tests
// ============================================================================

mod actor_to_buffer_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test: Transitions are correctly created
    // -------------------------------------------------------------------------
    #[test]
    fn test_transition_creation_from_base() {
        let base = Transition::new_discrete(
            vec![1.0, 2.0, 3.0, 4.0], // CartPole obs
            1,                         // action
            1.0,                       // reward
            vec![1.1, 2.1, 3.1, 4.1], // next_obs
            false,                     // terminal
            false,                     // truncated
        );

        let sac_transition = SACTransition::new(base);

        assert_eq!(sac_transition.state().len(), 4, "State should have 4 dims (CartPole)");
        assert_eq!(sac_transition.next_state().len(), 4, "Next state should have 4 dims");
        assert_eq!(sac_transition.reward(), 1.0, "Reward should be 1.0");
        assert!(!sac_transition.terminal(), "Should not be terminal");
    }

    // -------------------------------------------------------------------------
    // Test: push_transition works via ExperienceBuffer trait
    // -------------------------------------------------------------------------
    #[test]
    fn test_push_via_trait() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        let transition = SACTransition::new_discrete(vec![1.0], 0, 1.0, vec![2.0], false);

        // Using ExperienceBuffer trait method
        <SACBuffer<SACTransition> as ExperienceBuffer>::push(&buffer, transition);

        assert_eq!(buffer.pending_size(), 1, "Item should be in pending queue");
        buffer.consolidate();
        assert_eq!(buffer.len(), 1, "Item should be in main storage after consolidate");
    }

    // -------------------------------------------------------------------------
    // Test: Concurrent pushes from multiple actors
    // -------------------------------------------------------------------------
    #[test]
    fn test_concurrent_pushes_from_multiple_actors() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        let n_actors = 4;
        let pushes_per_actor = 50;
        let mut handles = vec![];

        for actor_id in 0..n_actors {
            let buffer_clone = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                for i in 0..pushes_per_actor {
                    let state_val = (actor_id * 100 + i) as f32;
                    let transition = SACTransition::new_discrete(
                        vec![state_val],
                        0,
                        1.0,
                        vec![state_val + 1.0],
                        false,
                    );
                    buffer_clone.push(transition);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let expected_total = n_actors * pushes_per_actor;

        // Pending size should match total pushes
        assert_eq!(
            buffer.pending_size(),
            expected_total,
            "All transitions should be in pending queue"
        );

        // len() is still 0!
        assert_eq!(buffer.len(), 0, "len() is 0 before consolidate");

        // After consolidate
        buffer.consolidate();
        assert_eq!(buffer.len(), expected_total, "All transitions should be consolidated");
    }
}

// ============================================================================
// PART 3: Training Loop Condition Tests
// ============================================================================

mod training_loop_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Test: is_training_ready gate behavior
    // -------------------------------------------------------------------------
    #[test]
    fn test_is_training_ready_gate_with_insufficient_data() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100, // Need 100 items to start training
            batch_size: 32,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push only 50 items (less than min_size)
        for i in 0..50 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // is_training_ready should return false
        assert!(
            !buffer.is_training_ready(),
            "Should not be ready with only 50 items (min_size = 100)"
        );

        // But items are now consolidated
        assert_eq!(buffer.len(), 50, "Items should be consolidated by is_training_ready check");
    }

    #[test]
    fn test_is_training_ready_gate_with_sufficient_data() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push exactly 100 items (equal to min_size)
        for i in 0..100 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        assert!(
            buffer.is_training_ready(),
            "Should be ready with exactly 100 items (min_size = 100)"
        );
    }

    // -------------------------------------------------------------------------
    // CRITICAL TEST: Simulate the learner loop
    // -------------------------------------------------------------------------
    #[test]
    fn test_simulated_learner_loop_behavior() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        let buffer_clone = Arc::clone(&buffer);
        let mut train_steps = 0;

        // Simulate learner loop checking and training
        // This is what happens in learner_thread

        // First, no data - loop should not train
        if buffer_clone.is_training_ready() {
            if let Some(_batch) = buffer_clone.sample_batch() {
                train_steps += 1;
            }
        }
        assert_eq!(train_steps, 0, "Should not train without data");

        // Push data via another "actor" thread
        let buffer_push = Arc::clone(&buffer);
        thread::spawn(move || {
            for i in 0..150 {
                let transition =
                    SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
                buffer_push.push(transition);
            }
        })
        .join()
        .unwrap();

        // Now check again - should be ready and train
        if buffer_clone.is_training_ready() {
            if let Some(_batch) = buffer_clone.sample_batch() {
                train_steps += 1;
            }
        }

        assert_eq!(train_steps, 1, "Should train once with sufficient data");
        assert_eq!(buffer.len(), 150, "Buffer should have all 150 items consolidated");
    }

    // -------------------------------------------------------------------------
    // RACE CONDITION TEST: Learner checking while actor pushing
    // -------------------------------------------------------------------------
    #[test]
    fn test_race_between_actor_push_and_learner_check() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 50, // Lower threshold for faster test
            batch_size: 10,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));
        let train_steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Actor thread - continuously pushes
        let buffer_actor = Arc::clone(&buffer);
        let shutdown_actor = Arc::clone(&shutdown);
        let actor_handle = thread::spawn(move || {
            let mut i = 0;
            while !shutdown_actor.load(Ordering::Relaxed) {
                let transition =
                    SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
                buffer_actor.push(transition);
                i += 1;
                thread::sleep(Duration::from_micros(100));
            }
            i // Return total pushes
        });

        // Learner thread - checks and samples
        let buffer_learner = Arc::clone(&buffer);
        let train_steps_learner = Arc::clone(&train_steps);
        let shutdown_learner = Arc::clone(&shutdown);
        let learner_handle = thread::spawn(move || {
            let mut iterations = 0;
            while !shutdown_learner.load(Ordering::Relaxed) && iterations < 1000 {
                if buffer_learner.is_training_ready() {
                    if let Some(_batch) = buffer_learner.sample_batch() {
                        train_steps_learner.fetch_add(1, Ordering::Relaxed);
                    }
                }
                iterations += 1;
                thread::sleep(Duration::from_micros(50));
            }
        });

        // Let them run for a bit
        thread::sleep(Duration::from_millis(100));
        shutdown.store(true, Ordering::Relaxed);

        let total_pushes = actor_handle.join().unwrap();
        learner_handle.join().unwrap();

        let final_train_steps = train_steps.load(Ordering::Relaxed);

        // Verify training happened
        assert!(
            final_train_steps > 0,
            "Training should have occurred at least once. \
             Actor pushed {} items, train_steps = {}",
            total_pushes,
            final_train_steps
        );

        // Verify no data loss
        buffer.consolidate();
        let final_len = buffer.len();
        assert!(
            final_len > 0,
            "Buffer should have items after consolidation"
        );
    }
}

// ============================================================================
// PART 4: Buffer Utilization Calculation Test
// ============================================================================

mod utilization_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // CRITICAL BUG TEST: buffer_utilization calculation in runner
    // -------------------------------------------------------------------------
    #[test]
    fn test_buffer_utilization_shows_zero_during_active_pushing() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        // Push 500 items (50% of capacity)
        for i in 0..500 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // This is how the runner calculates utilization:
        // stats.buffer_utilization = buffer.len() as f32 / config.buffer_capacity as f32;
        let utilization_wrong = buffer.len() as f32 / 1000.0;

        // BUG: Shows 0% even though buffer has 500 items pending!
        assert_eq!(
            utilization_wrong, 0.0,
            "BUG CONFIRMED: buffer_utilization calculation uses len() which is 0 \
             because it doesn't include pending items"
        );

        // The CORRECT way to calculate utilization would be:
        let utilization_correct = (buffer.len() + buffer.pending_size()) as f32 / 1000.0;
        assert!(
            (utilization_correct - 0.5).abs() < 0.01,
            "Correct utilization should be ~50%"
        );

        // Or consolidate first:
        buffer.consolidate();
        let utilization_after_consolidate = buffer.len() as f32 / 1000.0;
        assert!(
            (utilization_after_consolidate - 0.5).abs() < 0.01,
            "After consolidate, utilization should be ~50%"
        );
    }

    // -------------------------------------------------------------------------
    // Test: utilization() method behavior
    // -------------------------------------------------------------------------
    #[test]
    fn test_utilization_method_same_as_len_divided_by_capacity() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        for i in 0..500 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // utilization() also returns 0 because it uses len()
        let util_via_method = buffer.utilization();
        let util_calculated = buffer.len() as f32 / buffer.capacity() as f32;

        assert_eq!(
            util_via_method, util_calculated,
            "utilization() and len()/capacity() should match"
        );
        assert_eq!(util_via_method, 0.0, "Both are 0 because len() is 0");
    }
}

// ============================================================================
// PART 5: Complete Pipeline Simulation
// ============================================================================

mod pipeline_simulation {
    use super::*;

    // -------------------------------------------------------------------------
    // Test: Full pipeline simulation
    // -------------------------------------------------------------------------
    #[test]
    fn test_full_pipeline_simulation() {
        // Configuration matching the failing example
        let config = SACBufferConfig {
            capacity: 100_000,
            min_size: 1000,
            batch_size: 256,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        // Counters
        let env_steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let train_steps = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Spawn 2 actor threads (like the failing example)
        let mut actor_handles = vec![];
        for actor_id in 0..2 {
            let buffer_clone = Arc::clone(&buffer);
            let env_steps_clone = Arc::clone(&env_steps);
            let shutdown_clone = Arc::clone(&shutdown);

            let handle = thread::spawn(move || {
                let n_envs = 4; // Like the failing example
                let mut step = 0;
                while !shutdown_clone.load(Ordering::Relaxed) && step < 10000 {
                    // Simulate stepping n_envs environments
                    for env_id in 0..n_envs {
                        let state_val = ((actor_id * 1000 + step) * n_envs + env_id) as f32;
                        let transition = SACTransition::new_discrete(
                            vec![state_val, state_val + 0.1, state_val + 0.2, state_val + 0.3],
                            if step % 2 == 0 { 0 } else { 1 },
                            1.0,
                            vec![
                                state_val + 1.0,
                                state_val + 1.1,
                                state_val + 1.2,
                                state_val + 1.3,
                            ],
                            step % 50 == 49, // Terminal every 50 steps
                        );
                        buffer_clone.push(transition);
                    }
                    env_steps_clone.fetch_add(n_envs, Ordering::Relaxed);
                    step += 1;
                    thread::sleep(Duration::from_micros(10));
                }
            });
            actor_handles.push(handle);
        }

        // Spawn learner thread
        let buffer_learner = Arc::clone(&buffer);
        let train_steps_clone = Arc::clone(&train_steps);
        let shutdown_learner = Arc::clone(&shutdown);
        let learner_handle = thread::spawn(move || {
            while !shutdown_learner.load(Ordering::Relaxed) {
                if buffer_learner.is_training_ready() {
                    if let Some(_batch) = buffer_learner.sample_batch() {
                        train_steps_clone.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        });

        // Monitoring loop (like the runner does)
        let mut monitoring_iterations = 0;
        let mut ever_saw_nonzero_utilization = false;
        let mut ever_saw_training = false;

        while monitoring_iterations < 500 {
            thread::sleep(Duration::from_millis(10));

            let steps = env_steps.load(Ordering::Relaxed);
            let trains = train_steps.load(Ordering::Relaxed);

            // This is the BUGGY calculation from the runner:
            let utilization_buggy = buffer.len() as f32 / 100_000.0;

            // Correct calculation:
            buffer.consolidate(); // Force consolidation to see true state
            let utilization_correct = buffer.len() as f32 / 100_000.0;

            if utilization_correct > 0.0 {
                ever_saw_nonzero_utilization = true;
            }
            if trains > 0 {
                ever_saw_training = true;
            }

            // Check for early success
            if ever_saw_training && steps > 5000 {
                break;
            }

            monitoring_iterations += 1;
        }

        shutdown.store(true, Ordering::Relaxed);

        for handle in actor_handles {
            handle.join().unwrap();
        }
        learner_handle.join().unwrap();

        // Verify results
        let final_steps = env_steps.load(Ordering::Relaxed);
        let final_trains = train_steps.load(Ordering::Relaxed);

        buffer.consolidate();
        let final_buffer_len = buffer.len();

        assert!(
            final_steps > 0,
            "Actors should have collected env steps. Got: {}",
            final_steps
        );
        assert!(
            final_buffer_len > 0,
            "Buffer should have items. Got: {}",
            final_buffer_len
        );
        assert!(
            ever_saw_nonzero_utilization,
            "Should have seen non-zero utilization at some point"
        );
        assert!(
            ever_saw_training,
            "Training should have occurred. Final train_steps: {}",
            final_trains
        );

        println!("Pipeline simulation results:");
        println!("  Env steps: {}", final_steps);
        println!("  Train steps: {}", final_trains);
        println!("  Buffer len: {}", final_buffer_len);
    }

    // -------------------------------------------------------------------------
    // Test: Why might learner never see data?
    // -------------------------------------------------------------------------
    #[test]
    fn test_learner_sees_data_after_actors_fill_buffer() {
        let config = SACBufferConfig {
            capacity: 1000,
            min_size: 100,
            batch_size: 32,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));

        // Phase 1: Actors push data
        for i in 0..200 {
            let transition =
                SACTransition::new_discrete(vec![i as f32], 0, 1.0, vec![(i + 1) as f32], false);
            buffer.push(transition);
        }

        // Verify data is in pending queue
        assert_eq!(buffer.pending_size(), 200, "200 items should be pending");
        assert_eq!(buffer.len(), 0, "len() should still be 0");

        // Phase 2: Learner checks readiness
        // This is the critical moment - does is_training_ready see the pending data?
        let ready = buffer.is_training_ready();

        // YES! is_training_ready() calls do_consolidate() internally
        assert!(
            ready,
            "is_training_ready() should return true because it consolidates first"
        );

        // Now len() reflects the consolidated data
        assert_eq!(buffer.len(), 200, "len() should be 200 after is_training_ready()");

        // Learner can sample
        let batch = buffer.sample_batch();
        assert!(batch.is_some(), "Should be able to sample a batch");
        assert_eq!(batch.unwrap().len(), 32, "Batch should have 32 items");
    }
}

// ============================================================================
// PART 6: Root Cause Analysis
// ============================================================================

mod root_cause_analysis {
    use super::*;

    /// This test documents the root cause of the SAC training failure.
    ///
    /// The bug is NOT that training doesn't work - it's that the MONITORING
    /// DISPLAY shows 0% utilization even when training IS working.
    ///
    /// However, if there's truly zero training happening, the issue is elsewhere.
    #[test]
    fn test_document_root_cause() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let capacity = config.capacity;
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push data
        for i in 0..50 {
            buffer.push(SACTransition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
            ));
        }

        // SCENARIO 1: What the monitoring loop sees
        let monitoring_len = buffer.len();
        let monitoring_utilization = monitoring_len as f32 / capacity as f32;

        // SCENARIO 2: What the learner sees
        let learner_ready = buffer.is_training_ready();
        let learner_len = buffer.len();

        // The monitoring loop runs BEFORE the learner consolidates,
        // so it sees len() = 0.
        //
        // The learner loop calls is_training_ready() which consolidates,
        // so it sees the actual data.

        println!("ROOT CAUSE ANALYSIS:");
        println!("  Monitoring sees len() = {} (before consolidate)", monitoring_len);
        println!("  Monitoring shows {}% utilization", monitoring_utilization * 100.0);
        println!("  Learner is_training_ready() = {}", learner_ready);
        println!("  After consolidate, len() = {}", learner_len);
        println!();
        println!("HYPOTHESIS 1: Monitoring shows 0% but training works");
        println!("  - buffer.len() returns size of consolidated storage only");
        println!("  - Pushed items are in lock-free injector until consolidated");
        println!("  - Runner's buffer_utilization = buffer.len() / capacity shows 0%");
        println!("  - But learner's is_training_ready() consolidates and sees data");
        println!();
        println!("HYPOTHESIS 2: Training truly not working");
        println!("  - If train_steps stays 0, there's another bug");
        println!("  - Possible: learner thread panics before sampling");
        println!("  - Possible: batch processing fails silently");
        println!("  - Need to check train_step in learner_thread");

        assert_eq!(monitoring_len, 0, "Expected: monitoring sees 0");
        assert!(learner_ready, "Expected: learner can train");
        assert_eq!(learner_len, 50, "Expected: after consolidate, len = 50");
    }

    /// Test to verify if there's a deadlock or panic in the training path
    #[test]
    fn test_training_path_does_not_deadlock() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = Arc::new(SACBuffer::<SACTransition>::new(config));
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Push data
        for i in 0..20 {
            buffer.push(SACTransition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
            ));
        }

        let buffer_clone = Arc::clone(&buffer);
        let completed_clone = Arc::clone(&completed);

        // Try to sample in a separate thread with timeout
        let handle = thread::spawn(move || {
            if buffer_clone.is_training_ready() {
                if let Some(batch) = buffer_clone.sample_batch() {
                    // Simulate processing the batch
                    let total_reward: f32 = batch.iter().map(|t| t.reward()).sum();
                    assert!(total_reward > 0.0);
                    completed_clone.store(true, Ordering::Relaxed);
                }
            }
        });

        // Wait with timeout
        let start = std::time::Instant::now();
        while !completed.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(10));
        }

        handle.join().expect("Thread should not panic");

        assert!(
            completed.load(Ordering::Relaxed),
            "Training path should complete without deadlock"
        );
    }
}

// ============================================================================
// PART 7: Tests for the ACTUAL Training Flow
// ============================================================================

mod actual_training_flow {
    use super::*;

    /// This test examines what happens AFTER is_training_ready() returns true
    /// in the learner_thread function.
    #[test]
    fn test_sample_batch_returns_valid_data() {
        let config = SACBufferConfig {
            capacity: 100,
            min_size: 10,
            batch_size: 5,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push diverse transitions
        for i in 0..20 {
            buffer.push(SACTransition::new_discrete(
                vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32],
                i as u32 % 2,
                (i as f32) / 10.0,
                vec![
                    (i + 1) as f32,
                    ((i + 1) * 2) as f32,
                    ((i + 1) * 3) as f32,
                    ((i + 1) * 4) as f32,
                ],
                i % 5 == 4,
            ));
        }

        assert!(buffer.is_training_ready(), "Should be ready");

        let batch = buffer.sample_batch().expect("Should get batch");

        // Verify batch properties
        assert_eq!(batch.len(), 5, "Batch should have 5 transitions");

        for transition in &batch {
            assert_eq!(transition.state().len(), 4, "State should have 4 dims");
            assert_eq!(transition.next_state().len(), 4, "Next state should have 4 dims");
            assert!(
                transition.reward() >= 0.0 && transition.reward() < 2.0,
                "Reward should be in valid range"
            );
        }
    }

    /// Test that verifies batch tensor creation works correctly
    #[test]
    fn test_batch_to_tensor_conversion() {
        use burn::tensor::Tensor;

        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let config = SACBufferConfig {
            capacity: 100,
            min_size: 3,  // Reduced to match number of transitions we push
            batch_size: 3,
        };
        let buffer = SACBuffer::<SACTransition>::new(config);

        // Push known transitions
        buffer.push_transition(SACTransition::new_discrete(
            vec![1.0, 2.0],
            0,
            1.0,
            vec![2.0, 3.0],
            false,
        ));
        buffer.push_transition(SACTransition::new_discrete(
            vec![3.0, 4.0],
            1,
            2.0,
            vec![4.0, 5.0],
            false,
        ));
        buffer.push_transition(SACTransition::new_discrete(
            vec![5.0, 6.0],
            0,
            3.0,
            vec![6.0, 7.0],
            true,
        ));

        assert!(buffer.is_training_ready());
        let batch = buffer.sample_batch().unwrap();

        // Convert batch to tensors (simulating what train_step does)
        let batch_size = batch.len();
        let obs_size = batch[0].state().len();

        let states: Vec<f32> = batch.iter().flat_map(|t| t.state()).copied().collect();
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward()).collect();
        let terminals: Vec<f32> = batch
            .iter()
            .map(|t| if t.terminal() { 1.0 } else { 0.0 })
            .collect();

        // Create tensors
        let states_tensor = Tensor::<B, 1>::from_floats(states.as_slice(), &device)
            .reshape([batch_size, obs_size]);
        let rewards_tensor = Tensor::<B, 1>::from_floats(rewards.as_slice(), &device);
        let terminals_tensor = Tensor::<B, 1>::from_floats(terminals.as_slice(), &device);

        // Verify shapes
        assert_eq!(states_tensor.dims(), [3, 2], "States should be [3, 2]");
        assert_eq!(rewards_tensor.dims(), [3], "Rewards should be [3]");
        assert_eq!(terminals_tensor.dims(), [3], "Terminals should be [3]");
    }
}
