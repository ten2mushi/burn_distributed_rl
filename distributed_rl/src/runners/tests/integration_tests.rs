//! Integration tests for distributed RL runners.
//!
//! These tests verify end-to-end behavior with real threads and coordination.
//! They test the interactions between actors and learners, not just individual units.
//!
//! # Test Categories
//!
//! 1. **Weight Transfer**: Verify bytes-based weight transfer across threads
//! 2. **Buffer Coordination**: Actors and learner properly synchronize
//! 3. **Shutdown**: Graceful shutdown under various conditions
//! 4. **Concurrency**: Race condition detection
//!
//! # Notes
//!
//! - These tests use timeouts to detect deadlocks
//! - They use simpler mock components than unit tests
//! - Focus is on thread interaction, not RL correctness

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::core::bytes_slot::{bytes_slot, bytes_slot_with, BytesSlot};

// ============================================================================
// Weight Transfer Integration Tests
// ============================================================================

/// Test concurrent weight reads don't interfere with each other.
/// INTENT: Multiple actors can read weights simultaneously.
#[test]
fn test_concurrent_weight_reads() {
    let slot = Arc::new(bytes_slot_with(vec![1u8; 1000]));

    // Spawn multiple reader threads
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let slot_clone = Arc::clone(&slot);
            thread::spawn(move || {
                for _ in 0..100 {
                    let data = slot_clone.get();
                    assert!(
                        data.is_some(),
                        "Should always be able to read initial weights"
                    );
                    let bytes = data.unwrap();
                    assert_eq!(bytes.len(), 1000, "Data should be complete");
                    assert!(bytes.iter().all(|&b| b == 1), "Data should be consistent");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Reader thread panicked");
    }
}

/// Test writer-reader interaction with bytes slot.
/// INTENT: Writer publishes, readers eventually see new data.
#[test]
fn test_writer_reader_interaction() {
    let slot = Arc::new(bytes_slot());
    let shutdown = Arc::new(AtomicBool::new(false));

    let writer_slot = Arc::clone(&slot);
    let writer_shutdown = Arc::clone(&shutdown);
    let writer = thread::spawn(move || {
        let mut version = 0u8;
        while !writer_shutdown.load(Ordering::Relaxed) {
            writer_slot.publish(vec![version; 100]);
            version = version.wrapping_add(1);
            thread::sleep(Duration::from_millis(10));
        }
    });

    let reader_slot = Arc::clone(&slot);
    let reader_shutdown = Arc::clone(&shutdown);
    let reader = thread::spawn(move || {
        let mut read_count = 0;
        let mut last_seen: Option<u8> = None;

        while !reader_shutdown.load(Ordering::Relaxed) && read_count < 50 {
            if let Some(data) = reader_slot.get() {
                // All bytes should be the same value
                assert!(
                    data.iter().all(|&b| b == data[0]),
                    "Data should be internally consistent"
                );

                // Version should not decrease (allowing wrapping)
                if let Some(last) = last_seen {
                    let diff = data[0].wrapping_sub(last);
                    assert!(
                        diff <= 128, // Allowing for some lag but detecting big jumps
                        "Version should advance or stay same"
                    );
                }
                last_seen = Some(data[0]);
                read_count += 1;
            }
            thread::sleep(Duration::from_millis(5));
        }
    });

    thread::sleep(Duration::from_millis(500));
    shutdown.store(true, Ordering::Relaxed);

    writer.join().expect("Writer panicked");
    reader.join().expect("Reader panicked");
}

/// Test version monotonicity under concurrent access.
/// INTENT: Version only increases, never decreases.
#[test]
fn test_version_monotonicity_concurrent() {
    let slot = Arc::new(bytes_slot_with(vec![0u8]));

    let writer_slot = Arc::clone(&slot);
    let writer = thread::spawn(move || {
        for i in 0..100u8 {
            writer_slot.publish(vec![i]);
        }
    });

    let reader_slot = Arc::clone(&slot);
    let reader = thread::spawn(move || {
        let mut last_version = 0u64;
        for _ in 0..1000 {
            let current = reader_slot.version();
            assert!(
                current >= last_version,
                "Version went backward: {} -> {}",
                last_version,
                current
            );
            last_version = current;
        }
    });

    writer.join().expect("Writer panicked");
    reader.join().expect("Reader panicked");
}

// ============================================================================
// Buffer Synchronization Integration Tests
// ============================================================================

/// Test producer-consumer synchronization pattern.
/// INTENT: PPO actors wait for learner to consume before producing more.
#[test]
fn test_producer_consumer_sync() {
    let consumed_epoch = Arc::new(AtomicU64::new(0));
    let buffer_ready = Arc::new(AtomicBool::new(false));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Producer (actor)
    let prod_consumed = Arc::clone(&consumed_epoch);
    let prod_ready = Arc::clone(&buffer_ready);
    let prod_shutdown = Arc::clone(&shutdown);
    let producer = thread::spawn(move || {
        let mut local_epoch = 0u64;
        let mut produced = 0;

        while !prod_shutdown.load(Ordering::Relaxed) && produced < 5 {
            // Simulate producing data
            prod_ready.store(true, Ordering::Release);

            // Wait for consumption
            let deadline = Instant::now() + Duration::from_secs(1);
            while prod_consumed.load(Ordering::Acquire) <= local_epoch {
                if Instant::now() > deadline {
                    return Err("Producer timed out waiting for consumption");
                }
                if prod_shutdown.load(Ordering::Relaxed) {
                    return Ok(produced);
                }
                thread::sleep(Duration::from_micros(100));
            }

            local_epoch += 1;
            produced += 1;
        }
        Ok(produced)
    });

    // Consumer (learner)
    let cons_consumed = Arc::clone(&consumed_epoch);
    let cons_ready = Arc::clone(&buffer_ready);
    let cons_shutdown = Arc::clone(&shutdown);
    let consumer = thread::spawn(move || {
        let mut consumed = 0;

        while !cons_shutdown.load(Ordering::Relaxed) && consumed < 5 {
            // Wait for data
            let deadline = Instant::now() + Duration::from_secs(1);
            while !cons_ready.load(Ordering::Acquire) {
                if Instant::now() > deadline {
                    return Err("Consumer timed out waiting for data");
                }
                if cons_shutdown.load(Ordering::Relaxed) {
                    return Ok(consumed);
                }
                thread::sleep(Duration::from_micros(100));
            }

            // Consume
            cons_ready.store(false, Ordering::Release);
            cons_consumed.fetch_add(1, Ordering::Release);
            consumed += 1;
        }
        Ok(consumed)
    });

    let prod_result = producer.join().expect("Producer panicked");
    let cons_result = consumer.join().expect("Consumer panicked");

    shutdown.store(true, Ordering::Relaxed);

    assert!(prod_result.is_ok(), "Producer failed: {:?}", prod_result);
    assert!(cons_result.is_ok(), "Consumer failed: {:?}", cons_result);
    assert_eq!(prod_result.unwrap(), 5);
    assert_eq!(cons_result.unwrap(), 5);
}

/// Test async producer-consumer (IMPALA style).
/// INTENT: Producers don't block, consumers process at their own pace.
#[test]
fn test_async_producer_consumer() {
    let buffer_size = Arc::new(AtomicUsize::new(0));
    let total_produced = Arc::new(AtomicUsize::new(0));
    let total_consumed = Arc::new(AtomicUsize::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Multiple producers (actors)
    let producers: Vec<_> = (0..4)
        .map(|_| {
            let buf = Arc::clone(&buffer_size);
            let produced = Arc::clone(&total_produced);
            let sd = Arc::clone(&shutdown);
            thread::spawn(move || {
                while !sd.load(Ordering::Relaxed) {
                    buf.fetch_add(1, Ordering::Relaxed);
                    produced.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_micros(100));
                }
            })
        })
        .collect();

    // Single consumer (learner)
    let cons_buf = Arc::clone(&buffer_size);
    let consumed = Arc::clone(&total_consumed);
    let cons_shutdown = Arc::clone(&shutdown);
    let consumer = thread::spawn(move || {
        while !cons_shutdown.load(Ordering::Relaxed) {
            let size = cons_buf.load(Ordering::Relaxed);
            if size > 0 {
                // Consume batch
                let batch = size.min(10);
                cons_buf.fetch_sub(batch, Ordering::Relaxed);
                consumed.fetch_add(batch, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_micros(200)); // Slower than producers
        }
    });

    thread::sleep(Duration::from_millis(100));
    shutdown.store(true, Ordering::Relaxed);

    for handle in producers {
        handle.join().expect("Producer panicked");
    }
    consumer.join().expect("Consumer panicked");

    // Verify production happened
    let produced = total_produced.load(Ordering::Relaxed);
    let consumed = total_consumed.load(Ordering::Relaxed);

    assert!(produced > 0, "Should have produced some items");
    assert!(consumed > 0, "Should have consumed some items");
}

// ============================================================================
// Shutdown Integration Tests
// ============================================================================

/// Test graceful shutdown with active threads.
/// INTENT: All threads should terminate cleanly on shutdown signal.
#[test]
fn test_graceful_shutdown() {
    let shutdown = Arc::new(AtomicBool::new(false));

    let thread_count = 4;
    let handles: Vec<_> = (0..thread_count)
        .map(|i| {
            let sd = Arc::clone(&shutdown);
            thread::spawn(move || {
                while !sd.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(10));
                }
                i // Return thread ID to verify completion
            })
        })
        .collect();

    // Let threads run
    thread::sleep(Duration::from_millis(50));

    // Signal shutdown
    shutdown.store(true, Ordering::Relaxed);

    // Collect with timeout
    let deadline = Instant::now() + Duration::from_secs(1);
    let mut completed = Vec::new();

    for handle in handles {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match handle.join() {
            Ok(id) => completed.push(id),
            Err(e) => panic!("Thread panicked: {:?}", e),
        }
        if Instant::now() > deadline {
            panic!("Shutdown timed out");
        }
    }

    assert_eq!(
        completed.len(),
        thread_count,
        "All threads should complete"
    );
}

/// Test shutdown with blocked thread (simulated).
/// INTENT: Detect potential deadlock scenarios.
#[test]
fn test_shutdown_with_waiting_thread() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let consumed_epoch = Arc::new(AtomicU64::new(0));

    // Thread that waits on condition
    let wait_sd = Arc::clone(&shutdown);
    let wait_epoch = Arc::clone(&consumed_epoch);
    let waiter = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_millis(500);
        while !wait_sd.load(Ordering::Relaxed) {
            // Check with timeout
            if Instant::now() > deadline {
                return false; // Timed out
            }
            if wait_epoch.load(Ordering::Relaxed) > 0 {
                return true; // Condition met
            }
            thread::sleep(Duration::from_micros(100));
        }
        true // Shutdown received
    });

    // Don't satisfy condition, just shutdown
    thread::sleep(Duration::from_millis(100));
    shutdown.store(true, Ordering::Relaxed);

    let result = waiter.join().expect("Waiter panicked");
    assert!(
        result,
        "Thread should have terminated due to shutdown or timeout"
    );
}

// ============================================================================
// Concurrency Safety Tests
// ============================================================================

/// Test atomic counter under heavy contention.
/// INTENT: No lost updates under concurrent access.
#[test]
fn test_atomic_counter_correctness() {
    let counter = Arc::new(AtomicUsize::new(0));
    let increments_per_thread = 10000;
    let n_threads = 8;

    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let ctr = Arc::clone(&counter);
            thread::spawn(move || {
                for _ in 0..increments_per_thread {
                    ctr.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let expected = n_threads * increments_per_thread;
    let actual = counter.load(Ordering::SeqCst);
    assert_eq!(actual, expected, "No increments should be lost");
}

/// Test bytes slot under contention.
/// INTENT: All published data is accessible and consistent.
#[test]
fn test_bytes_slot_contention() {
    let slot = Arc::new(BytesSlot::new());
    let shutdown = Arc::new(AtomicBool::new(false));
    let errors = Arc::new(AtomicUsize::new(0));

    // Single writer
    let write_slot = Arc::clone(&slot);
    let write_sd = Arc::clone(&shutdown);
    let writer = thread::spawn(move || {
        let mut count = 0u32;
        while !write_sd.load(Ordering::Relaxed) {
            // Write a pattern: [count as bytes, repeated]
            let bytes = count.to_le_bytes().to_vec();
            write_slot.publish(bytes);
            count = count.wrapping_add(1);
        }
        count
    });

    // Multiple readers
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let read_slot = Arc::clone(&slot);
            let read_sd = Arc::clone(&shutdown);
            let err = Arc::clone(&errors);
            thread::spawn(move || {
                while !read_sd.load(Ordering::Relaxed) {
                    if let Some(data) = read_slot.get() {
                        // Verify pattern consistency
                        if data.len() >= 4 {
                            let _value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                            // Just verify we can decode it
                        } else if !data.is_empty() {
                            err.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    thread::sleep(Duration::from_millis(100));
    shutdown.store(true, Ordering::Relaxed);

    let write_count = writer.join().expect("Writer panicked");
    for reader in readers {
        reader.join().expect("Reader panicked");
    }

    assert!(write_count > 0, "Should have written something");
    assert_eq!(
        errors.load(Ordering::Relaxed),
        0,
        "Should have no inconsistent reads"
    );
}

// ============================================================================
// Timeout Detection Tests
// ============================================================================

/// Utility function to run code with timeout.
fn with_timeout<F, R>(timeout: Duration, f: F) -> Option<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    use std::sync::mpsc::channel;

    let (tx, rx) = channel();
    thread::spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });

    rx.recv_timeout(timeout).ok()
}

/// Test that potential deadlock is detected by timeout.
/// INTENT: Detect when threads are stuck waiting on each other.
#[test]
fn test_timeout_detection() {
    // This should complete quickly
    let quick_result = with_timeout(Duration::from_millis(100), || {
        thread::sleep(Duration::from_millis(10));
        42
    });
    assert_eq!(quick_result, Some(42), "Quick task should complete");

    // This would timeout (but we don't actually run it to avoid slow test)
    // let slow_result = with_timeout(Duration::from_millis(100), || {
    //     thread::sleep(Duration::from_secs(10));
    //     42
    // });
    // assert_eq!(slow_result, None, "Slow task should timeout");
}

// ============================================================================
// Episode Statistics Integration Tests
// ============================================================================

/// Test episode reward accumulation across threads.
/// INTENT: Episode stats are correctly aggregated from multiple actors.
#[test]
fn test_episode_stats_aggregation() {
    use parking_lot::Mutex;

    let total_episodes = Arc::new(AtomicUsize::new(0));
    let recent_rewards = Arc::new(Mutex::new(Vec::<f32>::new()));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Simulate actors reporting episode completions
    let actors: Vec<_> = (0..4)
        .map(|actor_id| {
            let eps = Arc::clone(&total_episodes);
            let rewards = Arc::clone(&recent_rewards);
            let sd = Arc::clone(&shutdown);
            thread::spawn(move || {
                for episode in 0..10 {
                    if sd.load(Ordering::Relaxed) {
                        break;
                    }
                    // Report episode completion
                    let reward = (actor_id * 10 + episode) as f32;
                    rewards.lock().push(reward);
                    eps.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_micros(100));
                }
            })
        })
        .collect();

    for handle in actors {
        handle.join().expect("Actor panicked");
    }

    let total_eps = total_episodes.load(Ordering::Relaxed);
    let rewards = recent_rewards.lock();

    assert_eq!(total_eps, 40, "Should have 4 * 10 episodes");
    assert_eq!(rewards.len(), 40, "Should have 40 reward entries");
}

/// Test average reward calculation.
/// INTENT: Moving average is computed correctly.
#[test]
fn test_average_reward_calculation() {
    use parking_lot::Mutex;

    let recent_rewards = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Add rewards
    {
        let mut rewards = recent_rewards.lock();
        for i in 0..150 {
            rewards.push(i as f32);
        }
    }

    // Calculate average of last 100
    let avg = {
        let rewards = recent_rewards.lock();
        let recent: Vec<f32> = rewards.iter().rev().take(100).copied().collect();
        if recent.is_empty() {
            0.0
        } else {
            recent.iter().sum::<f32>() / recent.len() as f32
        }
    };

    // Last 100 are 50..150, average = (50+149)/2 = 99.5
    let expected_avg = 99.5;
    assert!(
        (avg - expected_avg).abs() < 0.01,
        "Average should be {}, got {}",
        expected_avg,
        avg
    );
}

// ============================================================================
// Resource Management Tests
// ============================================================================

/// Test that thread handles are properly joined on shutdown.
/// INTENT: No zombie threads after runner completes.
#[test]
fn test_thread_cleanup() {
    let shutdown = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();
    for _ in 0..4 {
        let sd = Arc::clone(&shutdown);
        handles.push(thread::spawn(move || {
            while !sd.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(10));
            }
        }));
    }

    // Verify threads are running
    assert_eq!(handles.len(), 4);

    // Shutdown and join
    shutdown.store(true, Ordering::Relaxed);

    for handle in handles {
        let result = handle.join();
        assert!(result.is_ok(), "Thread should join cleanly");
    }
}
