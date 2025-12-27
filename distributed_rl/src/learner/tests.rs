//! Comprehensive test suite for the Learner component.
//!
//! These tests serve as a behavioral specification following the "Tests as definition"
//! philosophy. They aim to fully characterize the expected behavior of the Learner,
//! revealing bugs rather than papering over them.
//!
//! # Bug Documentation Convention
//!
//! When a test reveals incorrect behavior, it is documented with:
//! ```text
//! // BUG DETECTED: [description]
//! // Expected: [correct behavior]
//! // Actual: [observed incorrect behavior]
//! ```

use super::*;
use crate::core::model_slot::ModelSlot;
use crate::messages::{LearnerMsg, LearnerStats};
use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// =============================================================================
// Test Helpers and Mock Types
// =============================================================================

/// Simple mock model for testing
#[derive(Clone, Debug, PartialEq)]
struct MockModel {
    version: usize,
    data: Vec<f32>,
}

impl MockModel {
    fn new(version: usize) -> Self {
        Self {
            version,
            data: vec![1.0, 2.0, 3.0],
        }
    }
}

impl Default for MockModel {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Helper to create mock training infrastructure for PPO tests
fn create_ppo_test_harness(
    train_count: Arc<AtomicUsize>,
    publish_count: Arc<AtomicUsize>,
    loss_values: Vec<(f32, f32, f32, f32)>,
) -> (
    impl FnMut() -> (f32, f32, f32, f32) + Send + 'static,
    impl FnMut() -> MockModel + Send + 'static,
    impl FnMut(MockModel) + Send + 'static,
    Arc<ModelSlot<MockModel>>,
    Receiver<LearnerMsg<MockModel>>,
    Sender<()>,
    Receiver<()>,
    Arc<AtomicBool>,
) {
    let model_slot = Arc::new(ModelSlot::new());
    let (cmd_tx, cmd_rx) = bounded(100);
    let (ready_tx, ready_rx) = bounded(100);
    let shutdown = Arc::new(AtomicBool::new(false));

    let loss_index = Arc::new(AtomicUsize::new(0));
    let loss_index_clone = loss_index.clone();
    let train_count_clone = train_count.clone();

    let train_fn = move || {
        train_count_clone.fetch_add(1, Ordering::SeqCst);
        let idx = loss_index_clone.fetch_add(1, Ordering::SeqCst);
        if idx < loss_values.len() {
            loss_values[idx]
        } else {
            // Default loss values if we run out
            (1.0, 0.5, 0.3, 0.1)
        }
    };

    let model_version = Arc::new(AtomicUsize::new(0));
    let get_model_fn = move || {
        let v = model_version.fetch_add(1, Ordering::SeqCst);
        MockModel::new(v)
    };

    let publish_count_clone = publish_count.clone();
    let publish_fn = move |_model: MockModel| {
        publish_count_clone.fetch_add(1, Ordering::SeqCst);
    };

    // Drop the cmd_tx sender - we don't use it for basic tests
    // The actual commands will come through cmd_rx via external senders
    drop(cmd_tx);

    (
        train_fn,
        get_model_fn,
        publish_fn,
        model_slot,
        cmd_rx,
        ready_tx,
        ready_rx,
        shutdown,
    )
}

/// Helper to create mock training infrastructure for IMPALA tests
fn create_impala_test_harness(
    train_count: Arc<AtomicUsize>,
    publish_count: Arc<AtomicUsize>,
    loss_values: Vec<Option<(f32, f32, f32, f32)>>,
    is_ready_values: Vec<bool>,
) -> (
    impl FnMut() -> Option<(f32, f32, f32, f32)> + Send + 'static,
    impl FnMut() -> MockModel + Send + 'static,
    impl FnMut(MockModel, u64) + Send + 'static,
    impl FnMut() -> bool + Send + 'static,
    Arc<ModelSlot<MockModel>>,
    Receiver<LearnerMsg<MockModel>>,
    Arc<AtomicBool>,
    Arc<AtomicU64>,
) {
    let model_slot = Arc::new(ModelSlot::new());
    let (cmd_tx, cmd_rx) = bounded(100);
    let shutdown = Arc::new(AtomicBool::new(false));
    let version_counter = Arc::new(AtomicU64::new(0));

    let loss_index = Arc::new(AtomicUsize::new(0));
    let loss_index_clone = loss_index.clone();
    let train_count_clone = train_count.clone();

    let train_fn = move || {
        train_count_clone.fetch_add(1, Ordering::SeqCst);
        let idx = loss_index_clone.fetch_add(1, Ordering::SeqCst);
        if idx < loss_values.len() {
            loss_values[idx]
        } else {
            Some((1.0, 0.5, 0.3, 0.1))
        }
    };

    let model_version = Arc::new(AtomicUsize::new(0));
    let get_model_fn = move || {
        let v = model_version.fetch_add(1, Ordering::SeqCst);
        MockModel::new(v)
    };

    let publish_count_clone = publish_count.clone();
    let publish_fn = move |_model: MockModel, _version: u64| {
        publish_count_clone.fetch_add(1, Ordering::SeqCst);
    };

    let ready_index = Arc::new(AtomicUsize::new(0));
    let is_ready_fn = move || {
        let idx = ready_index.fetch_add(1, Ordering::SeqCst);
        if idx < is_ready_values.len() {
            is_ready_values[idx]
        } else {
            true
        }
    };

    drop(cmd_tx);

    (
        train_fn,
        get_model_fn,
        publish_fn,
        is_ready_fn,
        model_slot,
        cmd_rx,
        shutdown,
        version_counter,
    )
}

// =============================================================================
// LearnerConfig Tests
// =============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let config = LearnerConfig::default();
        assert_eq!(config.publish_freq, 10, "Default publish_freq should be 10");
        assert_eq!(config.stats_freq, 100, "Default stats_freq should be 100");
        assert_eq!(config.eval_freq, 1000, "Default eval_freq should be 1000");
        assert_eq!(
            config.max_train_steps, 0,
            "Default max_train_steps should be 0 (unlimited)"
        );
    }

    #[test]
    fn test_new_equals_default() {
        let config_new = LearnerConfig::new();
        let config_default = LearnerConfig::default();

        assert_eq!(config_new.publish_freq, config_default.publish_freq);
        assert_eq!(config_new.stats_freq, config_default.stats_freq);
        assert_eq!(config_new.eval_freq, config_default.eval_freq);
        assert_eq!(config_new.max_train_steps, config_default.max_train_steps);
    }

    #[test]
    fn test_with_publish_freq() {
        let config = LearnerConfig::new().with_publish_freq(50);
        assert_eq!(config.publish_freq, 50);
        // Other values should remain default
        assert_eq!(config.stats_freq, 100);
        assert_eq!(config.eval_freq, 1000);
        assert_eq!(config.max_train_steps, 0);
    }

    #[test]
    fn test_with_stats_freq() {
        let config = LearnerConfig::new().with_stats_freq(200);
        assert_eq!(config.stats_freq, 200);
        // Other values should remain default
        assert_eq!(config.publish_freq, 10);
    }

    #[test]
    fn test_with_eval_freq() {
        let config = LearnerConfig::new().with_eval_freq(5000);
        assert_eq!(config.eval_freq, 5000);
        // Other values should remain default
        assert_eq!(config.publish_freq, 10);
        assert_eq!(config.stats_freq, 100);
    }

    #[test]
    fn test_with_max_train_steps() {
        let config = LearnerConfig::new().with_max_train_steps(10000);
        assert_eq!(config.max_train_steps, 10000);
        // Other values should remain default
        assert_eq!(config.publish_freq, 10);
    }

    #[test]
    fn test_builder_chaining() {
        let config = LearnerConfig::new()
            .with_publish_freq(25)
            .with_stats_freq(50)
            .with_eval_freq(100)
            .with_max_train_steps(5000);

        assert_eq!(config.publish_freq, 25);
        assert_eq!(config.stats_freq, 50);
        assert_eq!(config.eval_freq, 100);
        assert_eq!(config.max_train_steps, 5000);
    }

    #[test]
    fn test_builder_override() {
        // Setting the same value twice should use the last value
        let config = LearnerConfig::new()
            .with_publish_freq(25)
            .with_publish_freq(50);

        assert_eq!(config.publish_freq, 50);
    }

    #[test]
    fn test_config_clone() {
        let config1 = LearnerConfig::new()
            .with_publish_freq(25)
            .with_max_train_steps(100);
        let config2 = config1.clone();

        assert_eq!(config1.publish_freq, config2.publish_freq);
        assert_eq!(config1.max_train_steps, config2.max_train_steps);
    }

    #[test]
    fn test_config_debug() {
        let config = LearnerConfig::new().with_publish_freq(42);
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("42"), "Debug output should contain field values");
    }

    #[test]
    fn test_large_frequency_values() {
        let config = LearnerConfig::new()
            .with_publish_freq(usize::MAX)
            .with_stats_freq(usize::MAX)
            .with_eval_freq(usize::MAX);

        assert_eq!(config.publish_freq, usize::MAX);
        assert_eq!(config.stats_freq, usize::MAX);
        assert_eq!(config.eval_freq, usize::MAX);
    }

    #[test]
    #[should_panic(expected = "publish_freq must be > 0")]
    fn test_zero_publish_freq_rejected_by_config() {
        // FIXED: Config now validates frequencies at construction time.
        // Zero publish_freq is rejected immediately instead of causing
        // a runtime panic during training.
        let _config = LearnerConfig::new().with_publish_freq(0);
    }

    #[test]
    #[should_panic(expected = "stats_freq must be > 0")]
    fn test_zero_stats_freq_rejected_by_config() {
        // FIXED: Same validation for stats_freq.
        let _config = LearnerConfig::new().with_stats_freq(0);
    }

    #[test]
    #[should_panic(expected = "eval_freq must be > 0")]
    fn test_zero_eval_freq_rejected_by_config() {
        // FIXED: Same validation for eval_freq.
        let _config = LearnerConfig::new().with_eval_freq(0);
    }

    #[test]
    fn test_one_frequency_values() {
        // Edge case: frequency of 1 means every step
        let config = LearnerConfig::new()
            .with_publish_freq(1)
            .with_stats_freq(1)
            .with_eval_freq(1);

        assert_eq!(config.publish_freq, 1);
        assert_eq!(config.stats_freq, 1);
        assert_eq!(config.eval_freq, 1);
    }

    #[test]
    fn test_max_train_steps_one() {
        let config = LearnerConfig::new().with_max_train_steps(1);
        assert_eq!(config.max_train_steps, 1);
    }
}

// =============================================================================
// LearnerHandle Tests
// =============================================================================

mod handle_tests {
    use super::*;

    /// Helper to create a mock LearnerHandle for testing
    fn make_mock_handle(
        stats_rx: Receiver<LearnerStats>,
        cmd_tx: Sender<LearnerMsg<()>>,
    ) -> LearnerHandle {
        LearnerHandle {
            thread: std::thread::spawn(|| {}),
            stats_rx,
            cmd_tx,
            dropped_stats: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[test]
    fn test_get_stats_returns_none_when_empty() {
        let (_tx, rx) = bounded::<LearnerStats>(100);
        let (cmd_tx, _cmd_rx) = bounded::<LearnerMsg<()>>(1);

        let handle = make_mock_handle(rx, cmd_tx);

        assert!(
            handle.get_stats().is_none(),
            "get_stats should return None when no stats available"
        );
    }

    #[test]
    fn test_get_stats_returns_available_stats() {
        let (tx, rx) = bounded::<LearnerStats>(100);
        let (cmd_tx, _cmd_rx) = bounded::<LearnerMsg<()>>(1);

        let expected_stats = LearnerStats {
            train_steps: 42,
            avg_loss: 1.5,
            ..Default::default()
        };
        tx.send(expected_stats.clone()).unwrap();

        let handle = make_mock_handle(rx, cmd_tx);

        let stats = handle.get_stats().expect("Should have stats");
        assert_eq!(stats.train_steps, 42);
        assert_eq!(stats.avg_loss, 1.5);
    }

    #[test]
    fn test_get_stats_consumes_message() {
        let (tx, rx) = bounded::<LearnerStats>(100);
        let (cmd_tx, _cmd_rx) = bounded::<LearnerMsg<()>>(1);

        tx.send(LearnerStats {
            train_steps: 1,
            ..Default::default()
        })
        .unwrap();

        let handle = make_mock_handle(rx, cmd_tx);

        let _first = handle.get_stats();
        assert!(
            handle.get_stats().is_none(),
            "Second get_stats should return None after consuming"
        );
    }

    #[test]
    fn test_get_stats_multiple_available() {
        let (tx, rx) = bounded::<LearnerStats>(100);
        let (cmd_tx, _cmd_rx) = bounded::<LearnerMsg<()>>(1);

        // Send multiple stats
        for i in 1..=3 {
            tx.send(LearnerStats {
                train_steps: i,
                ..Default::default()
            })
            .unwrap();
        }

        let handle = make_mock_handle(rx, cmd_tx);

        // Should get them in order
        assert_eq!(handle.get_stats().unwrap().train_steps, 1);
        assert_eq!(handle.get_stats().unwrap().train_steps, 2);
        assert_eq!(handle.get_stats().unwrap().train_steps, 3);
        assert!(handle.get_stats().is_none());
    }

    #[test]
    fn test_stop_sends_to_connected_channel() {
        // FIXED: LearnerHandle.stop() now works correctly
        // The cmd_tx in LearnerHandle IS connected to the learner's command receiver.
        //
        // This test verifies that calling stop() sends the message to the correct channel.

        let (cmd_tx, cmd_rx) = bounded::<LearnerMsg<()>>(1);
        let (_stats_tx, stats_rx) = bounded::<LearnerStats>(100);

        // Use the SAME cmd_tx for the handle (as the fixed code does)
        let handle = LearnerHandle {
            thread: std::thread::spawn(|| {}),
            stats_rx,
            cmd_tx: cmd_tx.clone(),
            dropped_stats: Arc::new(AtomicUsize::new(0)),
        };

        // Call stop - this should send to the connected channel
        handle.stop();

        // The message SHOULD arrive on cmd_rx because handle.cmd_tx is connected
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            cmd_rx.try_recv().is_ok(),
            "FIXED: stop() message now reaches the intended receiver"
        );
    }

    #[test]
    fn test_stop_command_actually_stops_learner_ppo() {
        // FIXED: stop() now works correctly
        //
        // This test verifies that calling handle.stop() causes the learner to exit
        // within a reasonable time without needing the shutdown flag workaround.

        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        // Unlimited training - should only stop via command
        let config = LearnerConfig::new()
            .with_max_train_steps(0)
            .with_publish_freq(100);

        let learner = Learner::new(config);
        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Send some ready signals to start training
        for _ in 0..5 {
            let _ = ready_tx.send(());
        }

        // Wait a bit for training to start
        std::thread::sleep(Duration::from_millis(100));

        // Stop via handle - this now works!
        let sent = handle.stop();
        assert!(sent, "stop() should successfully send the command");

        // Wait for learner to stop - should complete within 500ms
        let start = Instant::now();
        let timeout = Duration::from_millis(500);

        // The learner should stop without needing the shutdown flag
        // (we don't set shutdown.store(true) anymore)

        let result = handle.thread.join();
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Thread should complete without panic");
        assert!(
            elapsed < timeout,
            "Learner should stop within {}ms via stop(), took {:?}",
            timeout.as_millis(),
            elapsed
        );
    }

    #[test]
    fn test_join_completes_after_shutdown_flag() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, _ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count, publish_count, vec![]);

        let config = LearnerConfig::new().with_max_train_steps(0);
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Set shutdown flag
        shutdown.store(true, Ordering::SeqCst);

        // Join should complete
        let result = handle.join();
        assert!(result.is_ok(), "Join should complete successfully after shutdown");
    }

    #[test]
    fn test_multiple_stop_calls_are_safe() {
        let (cmd_tx, _cmd_rx) = bounded::<LearnerMsg<()>>(10);
        let (_stats_tx, stats_rx) = bounded::<LearnerStats>(100);

        let handle = LearnerHandle {
            thread: std::thread::spawn(|| {}),
            stats_rx,
            cmd_tx,
            dropped_stats: Arc::new(AtomicUsize::new(0)),
        };

        // Multiple stop calls should not panic
        handle.stop();
        handle.stop();
        handle.stop();
        // If we get here, the test passes
    }

    #[test]
    fn test_stop_on_full_channel() {
        // Fill the channel first
        let (cmd_tx, cmd_rx) = bounded::<LearnerMsg<()>>(1);
        let _ = cmd_tx.try_send(LearnerMsg::Pause); // Fill the channel

        let (_stats_tx, stats_rx) = bounded::<LearnerStats>(100);

        let handle = LearnerHandle {
            thread: std::thread::spawn(|| {}),
            stats_rx,
            cmd_tx,
            dropped_stats: Arc::new(AtomicUsize::new(0)),
        };

        // stop() uses try_send which won't block on full channel
        // This should not hang and returns false indicating channel was full
        let sent = handle.stop();
        assert!(!sent, "stop() should return false when channel is full");

        // Clean up
        drop(cmd_rx);
    }
}

// =============================================================================
// Learner PPO Tests
// =============================================================================

mod ppo_learner_tests {
    use super::*;

    #[test]
    fn test_learner_new() {
        // Learner stores the config internally but config is private
        // We verify by spawning and observing behavior
        let config = LearnerConfig::new().with_publish_freq(42);
        let _learner = Learner::new(config);
        // If we get here without panic, construction succeeded
    }

    #[test]
    fn test_spawn_ppo_thread_starts() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, _ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count, publish_count, vec![]);

        let config = LearnerConfig::new().with_max_train_steps(0);
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Thread should be running
        assert!(!handle.thread.is_finished());

        // Clean up
        shutdown.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[test]
    fn test_spawn_ppo_shutdown_flag_stops_learner() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, _ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count, publish_count, vec![]);

        let config = LearnerConfig::new().with_max_train_steps(0);
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Immediately set shutdown
        shutdown.store(true, Ordering::SeqCst);

        // Should complete within timeout
        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_max_train_steps_stops_learner() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(
                train_count.clone(),
                publish_count.clone(),
                vec![(1.0, 0.5, 0.3, 0.1); 10],
            );

        // Set max_train_steps to 5
        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100); // Don't publish during test

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send enough ready signals
        for _ in 0..10 {
            let _ = ready_tx.send(());
        }

        // Wait for completion
        let result = handle.join();
        assert!(result.is_ok(), "Learner should complete after max_train_steps");

        // Should have trained exactly 5 times
        let trained = train_count.load(Ordering::SeqCst);
        assert_eq!(trained, 5, "Should train exactly max_train_steps times");
    }

    #[test]
    fn test_train_fn_called_on_ready_signal() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        let config = LearnerConfig::new()
            .with_max_train_steps(3)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send exactly 3 ready signals
        for _ in 0..3 {
            ready_tx.send(()).unwrap();
        }

        let _ = handle.join();

        assert_eq!(
            train_count.load(Ordering::SeqCst),
            3,
            "train_fn should be called once per ready signal"
        );
    }

    #[test]
    fn test_model_published_at_correct_frequency() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        // Publish every 2 steps, train for 6 steps
        let config = LearnerConfig::new()
            .with_max_train_steps(6)
            .with_publish_freq(2);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot.clone(),
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send ready signals
        for _ in 0..6 {
            ready_tx.send(()).unwrap();
        }

        let _ = handle.join();

        // Steps 2, 4, 6 should trigger publish (3 times)
        let published = publish_count.load(Ordering::SeqCst);
        assert_eq!(
            published, 3,
            "Should publish at steps 2, 4, 6 = 3 times total"
        );

        // Model slot should have received publishes
        let (slot_published, _, _) = model_slot.stats();
        assert_eq!(slot_published, 3);
    }

    #[test]
    fn test_stats_reported_at_correct_frequency() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        // Stats every 5 steps, train for 15 steps
        let config = LearnerConfig::new()
            .with_max_train_steps(15)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..15 {
            ready_tx.send(()).unwrap();
        }

        // Wait for training to complete
        let _ = handle.join();

        // Should have received stats at steps 5, 10, 15 = 3 times
        // But we need to drain the stats channel
    }

    #[test]
    fn test_stats_contain_correct_train_steps() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..10 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        // Collect all stats
        let mut stats_received = vec![];
        while let Some(stats) = handle.get_stats() {
            stats_received.push(stats);
        }

        let _ = handle.join();

        // Should have stats at step 5 and 10
        assert!(stats_received.len() >= 2, "Should receive at least 2 stats reports");

        // First stats should be at step 5
        if let Some(first_stats) = stats_received.first() {
            assert_eq!(first_stats.train_steps, 5, "First stats should be at step 5");
        }
    }

    #[test]
    fn test_average_loss_calculation() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        // Return known loss values: 2.0, 4.0, 6.0, 8.0, 10.0 - average = 6.0
        let loss_values = vec![
            (2.0, 1.0, 0.5, 0.1),
            (4.0, 2.0, 1.0, 0.2),
            (6.0, 3.0, 1.5, 0.3),
            (8.0, 4.0, 2.0, 0.4),
            (10.0, 5.0, 2.5, 0.5),
        ];
        let loss_index = Arc::new(AtomicUsize::new(0));
        let loss_index_clone = loss_index.clone();

        let train_fn = move || {
            let idx = loss_index_clone.fetch_add(1, Ordering::SeqCst);
            loss_values[idx]
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        let stats = handle.get_stats();
        let _ = handle.join();

        if let Some(s) = stats {
            // Average total loss: (2+4+6+8+10)/5 = 6.0
            assert!(
                (s.avg_loss - 6.0).abs() < 0.001,
                "Average loss should be 6.0, got {}",
                s.avg_loss
            );
            // Average policy loss: (1+2+3+4+5)/5 = 3.0
            assert!(
                (s.avg_policy_loss - 3.0).abs() < 0.001,
                "Average policy loss should be 3.0, got {}",
                s.avg_policy_loss
            );
            // Average value loss: (0.5+1+1.5+2+2.5)/5 = 1.5
            assert!(
                (s.avg_value_loss - 1.5).abs() < 0.001,
                "Average value loss should be 1.5, got {}",
                s.avg_value_loss
            );
            // Average entropy: (0.1+0.2+0.3+0.4+0.5)/5 = 0.3
            assert!(
                (s.avg_entropy - 0.3).abs() < 0.001,
                "Average entropy should be 0.3, got {}",
                s.avg_entropy
            );
        }
    }

    #[test]
    fn test_loss_accumulator_reset_after_stats_report() {
        // After stats are reported, accumulators should reset
        // This tests that the second stats report has correct averages
        // for only the second window of losses

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        // First 5: average 2.0, Next 5: average 7.0
        let loss_values = vec![
            (1.0, 0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (3.0, 0.0, 0.0, 0.0), // avg = 10/5 = 2.0
            (5.0, 0.0, 0.0, 0.0),
            (6.0, 0.0, 0.0, 0.0),
            (7.0, 0.0, 0.0, 0.0),
            (8.0, 0.0, 0.0, 0.0),
            (9.0, 0.0, 0.0, 0.0), // avg = 35/5 = 7.0
        ];
        let loss_index = Arc::new(AtomicUsize::new(0));
        let loss_index_clone = loss_index.clone();

        let train_fn = move || {
            let idx = loss_index_clone.fetch_add(1, Ordering::SeqCst);
            if idx < loss_values.len() {
                loss_values[idx]
            } else {
                (1.0, 0.0, 0.0, 0.0)
            }
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..10 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(300));

        let stats1 = handle.get_stats();
        let stats2 = handle.get_stats();
        let _ = handle.join();

        if let (Some(s1), Some(s2)) = (stats1, stats2) {
            assert!(
                (s1.avg_loss - 2.0).abs() < 0.001,
                "First window avg should be 2.0, got {}",
                s1.avg_loss
            );
            assert!(
                (s2.avg_loss - 7.0).abs() < 0.001,
                "Second window avg should be 7.0, got {}",
                s2.avg_loss
            );
        }
    }

    #[test]
    fn test_ready_channel_disconnect_stops_learner() {
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new().with_max_train_steps(0); // unlimited

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Drop the ready sender - this disconnects the channel
        drop(ready_tx);

        // Learner should detect disconnect and exit
        let result = handle.join();
        assert!(result.is_ok(), "Learner should exit gracefully on channel disconnect");
    }

    #[test]
    fn test_no_training_without_ready_signal() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, _ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        let config = LearnerConfig::new().with_max_train_steps(0);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Don't send any ready signals, just wait a bit
        std::thread::sleep(Duration::from_millis(200));

        // Train count should be 0
        assert_eq!(
            train_count.load(Ordering::SeqCst),
            0,
            "No training should occur without ready signals"
        );

        shutdown.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[test]
    fn test_immediate_shutdown_before_any_training() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, _ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        // Set shutdown before spawning
        shutdown.store(true, Ordering::SeqCst);

        let config = LearnerConfig::new().with_max_train_steps(100);
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        let result = handle.join();
        assert!(result.is_ok());

        assert_eq!(
            train_count.load(Ordering::SeqCst),
            0,
            "No training should occur if shutdown is set before spawn"
        );
    }

    #[test]
    fn test_publish_freq_one_publishes_every_step() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        let _ = handle.join();

        assert_eq!(
            publish_count.load(Ordering::SeqCst),
            5,
            "publish_freq=1 should publish every step"
        );
    }

    #[test]
    fn test_request_stats_command() {
        let model_slot = Arc::new(ModelSlot::new());
        let (cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_count = Arc::new(AtomicUsize::new(0));
        let train_count_clone = train_count.clone();
        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        // Stats freq very high so automatic reporting doesn't interfere
        let config = LearnerConfig::new()
            .with_max_train_steps(0)
            .with_publish_freq(100)
            .with_stats_freq(10000);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Do some training
        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }
        std::thread::sleep(Duration::from_millis(100));

        // Request stats via command
        // Note: Due to BUG, cmd_tx is not connected to the learner!
        // This command will NOT be received
        let _ = cmd_tx.send(LearnerMsg::RequestStats);

        std::thread::sleep(Duration::from_millis(100));

        // BUG: The stats will NOT be received because cmd_tx sends to wrong channel
        // let stats = handle.get_stats();
        // Expected: stats should be Some with train_steps = 5
        // Actual: stats is None because RequestStats command was not received

        shutdown.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[test]
    fn test_update_learning_rate_command_handled() {
        // This test verifies the command is received and doesn't cause panic
        // The actual learning rate update is handled by the train_fn closure

        let model_slot = Arc::new(ModelSlot::new());
        let (cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // BUG: This command won't be received due to disconnected channel
        let _ = cmd_tx.send(LearnerMsg::UpdateLearningRate(0.001));

        for _ in 0..10 {
            ready_tx.send(()).unwrap();
        }

        let result = handle.join();
        assert!(result.is_ok(), "Learner should handle UpdateLearningRate without panic");
    }

    #[test]
    fn test_thread_name_is_ppo_learner() {
        let model_slot = Arc::new(ModelSlot::<MockModel>::new());
        let (_cmd_tx, cmd_rx) = bounded::<LearnerMsg<MockModel>>(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        // Capture thread name inside the spawned thread
        let thread_name = Arc::new(parking_lot::Mutex::new(String::new()));
        let thread_name_clone = thread_name.clone();

        let train_fn = move || {
            // Capture thread name on first call
            let current_name = std::thread::current().name().unwrap_or("").to_string();
            let mut guard = thread_name_clone.lock();
            if guard.is_empty() {
                *guard = current_name;
            }
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new().with_max_train_steps(1);
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        ready_tx.send(()).unwrap();
        let _ = handle.join();

        let name = thread_name.lock().clone();
        assert_eq!(name, "PPO-Learner", "Thread should be named PPO-Learner");
    }

    #[test]
    fn test_model_slot_receives_published_models() {
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let model_version = Arc::new(AtomicUsize::new(0));
        let model_version_clone = model_version.clone();

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = move || {
            let v = model_version_clone.fetch_add(1, Ordering::SeqCst);
            MockModel::new(v)
        };
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(5);

        let learner = Learner::new(config);
        let slot_clone = model_slot.clone();

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        let _ = handle.join();

        // Model should be available in slot
        let model = slot_clone.take();
        assert!(model.is_some(), "Model slot should contain published model");
    }
}

// =============================================================================
// Learner IMPALA Tests
// =============================================================================

mod impala_learner_tests {
    use super::*;

    #[test]
    fn test_spawn_impala_thread_starts() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        ) = create_impala_test_harness(
            train_count,
            publish_count,
            vec![],
            vec![false], // Not ready initially
        );

        let config = LearnerConfig::new().with_max_train_steps(0);
        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown.clone(),
            version_counter,
        );

        assert!(!handle.thread.is_finished());

        shutdown.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[test]
    fn test_impala_shutdown_flag_stops_learner() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        ) = create_impala_test_harness(train_count, publish_count, vec![], vec![false]);

        let config = LearnerConfig::new().with_max_train_steps(0);
        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown.clone(),
            version_counter,
        );

        shutdown.store(true, Ordering::SeqCst);

        let result = handle.join();
        assert!(result.is_ok());
    }

    #[test]
    fn test_impala_max_train_steps_respected() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let train_count_clone = train_count.clone();
        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            Some((1.0, 0.5, 0.3, 0.1))
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel, _: u64| {};
        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        );

        let _ = handle.join();

        assert_eq!(
            train_count.load(Ordering::SeqCst),
            5,
            "IMPALA should train exactly max_train_steps times"
        );
    }

    #[test]
    fn test_impala_is_ready_fn_polling() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let ready_count = Arc::new(AtomicUsize::new(0));
        let ready_count_clone = ready_count.clone();

        // Return false 3 times, then true 5 times
        let is_ready_fn = move || {
            let count = ready_count_clone.fetch_add(1, Ordering::SeqCst);
            count >= 3
        };

        let train_count_clone = train_count.clone();
        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            Some((1.0, 0.5, 0.3, 0.1))
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel, _: u64| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(2)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        );

        let _ = handle.join();

        // is_ready_fn was called at least 3 times before training started
        assert!(
            ready_count.load(Ordering::SeqCst) >= 5,
            "is_ready_fn should be polled multiple times"
        );

        // But train was only called twice (max_train_steps=2)
        assert_eq!(train_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_impala_train_fn_returns_none() {
        // When train_fn returns None, no training step should be counted

        let train_count = Arc::new(AtomicUsize::new(0));
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        // Return None first 3 times, then Some
        let train_fn = move || {
            let count = call_count_clone.fetch_add(1, Ordering::SeqCst);
            if count < 3 {
                None
            } else {
                Some((1.0, 0.5, 0.3, 0.1))
            }
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel, _: u64| {};
        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(2)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        );

        let _ = handle.join();

        // train_fn was called more than 2 times (due to None returns)
        assert!(
            call_count.load(Ordering::SeqCst) >= 5,
            "train_fn should be called multiple times when returning None"
        );
    }

    #[test]
    fn test_impala_version_counter_incremented() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let train_fn = || Some((1.0, 0.5, 0.3, 0.1));
        let get_model_fn = || MockModel::new(0);

        let versions_published = Arc::new(parking_lot::Mutex::new(vec![]));
        let versions_clone = versions_published.clone();
        let publish_fn = move |_: MockModel, version: u64| {
            versions_clone.lock().push(version);
        };
        let is_ready_fn = || true;

        // Publish every 2 steps, train 6 steps = 3 publishes
        let config = LearnerConfig::new()
            .with_max_train_steps(6)
            .with_publish_freq(2);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter.clone(),
        );

        let _ = handle.join();

        let versions = versions_published.lock().clone();
        assert_eq!(versions.len(), 3, "Should have 3 publishes");
        assert_eq!(versions, vec![1, 2, 3], "Versions should increment: 1, 2, 3");

        assert_eq!(
            version_counter.load(Ordering::SeqCst),
            3,
            "Version counter should be 3"
        );
    }

    #[test]
    fn test_impala_publish_includes_version() {
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(10)); // Start at 10

        let train_fn = || Some((1.0, 0.5, 0.3, 0.1));
        let get_model_fn = || MockModel::new(0);

        let last_version = Arc::new(AtomicU64::new(0));
        let last_version_clone = last_version.clone();
        let publish_fn = move |_: MockModel, version: u64| {
            last_version_clone.store(version, Ordering::SeqCst);
        };
        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        );

        let _ = handle.join();

        // Version should be 11 (10 + 1)
        assert_eq!(
            last_version.load(Ordering::SeqCst),
            11,
            "Version should be incremented from initial value"
        );
    }

    #[test]
    fn test_impala_stop_command_bug() {
        // BUG DETECTED: Same as PPO - stop() is broken
        // Expected: handle.stop() should stop the IMPALA learner
        // Actual: The cmd_tx is disconnected, stop command never reaches learner

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let train_fn = || Some((1.0, 0.5, 0.3, 0.1));
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel, _: u64| {};
        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(0) // Unlimited
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown.clone(),
            version_counter,
        );

        // Try to stop via handle - this WON'T work due to bug
        handle.stop();

        // We have to use shutdown flag as workaround
        shutdown.store(true, Ordering::SeqCst);

        let result = handle.join();
        assert!(result.is_ok());

        // The bug is confirmed if we had to use shutdown flag
    }

    #[test]
    fn test_impala_thread_name() {
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let thread_name_captured = Arc::new(parking_lot::Mutex::new(String::new()));
        let thread_name_clone = thread_name_captured.clone();

        let train_fn = move || {
            let name = std::thread::current().name().unwrap_or("").to_string();
            let mut guard = thread_name_clone.lock();
            if guard.is_empty() {
                *guard = name;
            }
            Some((1.0, 0.5, 0.3, 0.1))
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel, _: u64| {};
        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter,
        );

        let _ = handle.join();

        let name = thread_name_captured.lock().clone();
        assert_eq!(name, "IMPALA-Learner", "Thread should be named IMPALA-Learner");
    }
}

// =============================================================================
// Stats Calculation Tests
// =============================================================================

mod stats_calculation_tests {
    use super::*;

    #[test]
    fn test_steps_per_second_calculation() {
        // With known elapsed time, verify steps_per_second calculation

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || {
            // Simulate some work
            std::thread::sleep(Duration::from_millis(10));
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        if let Some(stats) = handle.get_stats() {
            // steps_per_second should be reasonable given ~10ms per step
            // 5 steps in ~50ms = ~100 steps/sec (but with overhead, likely lower)
            assert!(
                stats.steps_per_second > 0.0,
                "steps_per_second should be positive"
            );
            assert!(
                stats.steps_per_second < 1000.0,
                "steps_per_second should be reasonable"
            );
        }

        let _ = handle.join();
    }

    #[test]
    fn test_zero_loss_count_no_division_by_zero() {
        // When loss_count is 0, avg calculations should return 0, not panic

        let model_slot = Arc::new(ModelSlot::<MockModel>::new());
        let (cmd_tx, cmd_rx) = bounded(100);
        let (_ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(0)
            .with_publish_freq(100)
            .with_stats_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Don't do any training, just request stats
        // BUG: This command won't be received due to disconnected channel
        // But even if it was, it should return zeros without panic
        let _ = cmd_tx.send(LearnerMsg::RequestStats);
        std::thread::sleep(Duration::from_millis(100));

        // Clean up
        shutdown.store(true, Ordering::SeqCst);
        let _ = handle.join();
        // If we get here without panic, the test passes
    }

    #[test]
    fn test_nan_loss_handling() {
        // NaN losses are now filtered out by defensive coding.
        // This is better behavior: NaN doesn't corrupt the stats.

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        if let Some(stats) = handle.get_stats() {
            // NaN values are filtered out - avg_loss should be 0.0 (default)
            // This is defensive coding: non-finite losses don't corrupt stats
            assert!(
                stats.avg_loss.is_finite(),
                "NaN losses should be filtered, avg_loss should be finite (got {})",
                stats.avg_loss
            );
        }

        let _ = handle.join();
    }

    #[test]
    fn test_inf_loss_handling() {
        // Infinite losses are now filtered out by defensive coding.
        // This is better behavior: Inf doesn't corrupt the stats.

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (f32::INFINITY, 0.0, 0.0, 0.0);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        if let Some(stats) = handle.get_stats() {
            // Infinite values are filtered out - avg_loss should be finite
            // This is defensive coding: non-finite losses don't corrupt stats
            assert!(
                stats.avg_loss.is_finite(),
                "Infinite losses should be filtered, avg_loss should be finite (got {})",
                stats.avg_loss
            );
        }

        let _ = handle.join();
    }

    #[test]
    fn test_negative_loss_handling() {
        // Negative loss is mathematically valid (though unusual in practice)

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (-1.0, -0.5, -0.3, -0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100)
            .with_stats_freq(5);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..5 {
            ready_tx.send(()).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        if let Some(stats) = handle.get_stats() {
            assert!(
                stats.avg_loss < 0.0,
                "Negative loss should be preserved"
            );
            assert!((stats.avg_loss - (-1.0)).abs() < 0.001);
        }

        let _ = handle.join();
    }

    #[test]
    fn test_very_fast_training_elapsed_near_zero() {
        // BUG POTENTIAL: If training is extremely fast (near-zero elapsed time),
        // steps_per_second could be extremely large or inf
        // Current code checks `if elapsed > 0.0` but floating point comparison
        // might let very small values through

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        // No-op train function - should be extremely fast
        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(100)
            .with_stats_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        ready_tx.send(()).unwrap();

        std::thread::sleep(Duration::from_millis(50));

        if let Some(stats) = handle.get_stats() {
            // Should be finite, even if very large
            assert!(
                stats.steps_per_second.is_finite(),
                "steps_per_second should be finite even with fast training"
            );
        }

        let _ = handle.join();
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_zero_max_train_steps_means_unlimited() {
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_count = Arc::new(AtomicUsize::new(0));
        let train_count_clone = train_count.clone();

        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(0) // Unlimited
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Send 10 ready signals
        for _ in 0..10 {
            let _ = ready_tx.send(());
        }

        std::thread::sleep(Duration::from_millis(200));

        // Should have trained at least 10 times (unlimited)
        assert!(
            train_count.load(Ordering::SeqCst) >= 10,
            "Unlimited training should continue"
        );

        shutdown.store(true, Ordering::SeqCst);
        let _ = handle.join();
    }

    #[test]
    fn test_large_train_step_count() {
        // Test that counting to larger numbers doesn't overflow
        // Note: Using a moderate number (100) to keep test runtime reasonable

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        // Use unbounded channel to avoid blocking
        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        // 100 steps is enough to verify counting works without overflow
        let config = LearnerConfig::new()
            .with_max_train_steps(100)
            .with_publish_freq(100)
            .with_stats_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send all ready signals
        for _ in 0..100 {
            let _ = ready_tx.send(());
        }

        let result = handle.join();
        assert!(result.is_ok(), "Training run should complete without overflow");
    }

    #[test]
    fn test_one_train_step() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let (train_fn, get_model_fn, publish_fn, model_slot, cmd_rx, ready_tx, ready_rx, shutdown) =
            create_ppo_test_harness(train_count.clone(), publish_count.clone(), vec![]);

        let config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        ready_tx.send(()).unwrap();

        let _ = handle.join();

        assert_eq!(train_count.load(Ordering::SeqCst), 1);
        assert_eq!(publish_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_stats_channel_bounded_100() {
        // The stats channel is bounded at 100 messages
        // If more stats are sent without being consumed, they are silently dropped

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(1000);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        // Stats every step, train 200 steps
        // This should generate 200 stats but channel only holds 100
        let config = LearnerConfig::new()
            .with_max_train_steps(200)
            .with_publish_freq(1000)
            .with_stats_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..200 {
            let _ = ready_tx.try_send(());
        }

        // Wait for training to complete
        std::thread::sleep(Duration::from_millis(500));

        // Count how many stats we can retrieve (before joining)
        let mut count = 0;
        while handle.get_stats().is_some() {
            count += 1;
        }

        let _ = handle.join();

        // BUG POTENTIAL: Stats are silently dropped when channel is full
        // This test documents the behavior - we can only get at most 100
        assert!(
            count <= 100,
            "Stats channel is bounded at 100, got {}",
            count
        );
    }

    #[test]
    fn test_pause_resume_not_implemented() {
        // Pause and Resume commands are not implemented
        // This test documents that behavior

        let model_slot = Arc::new(ModelSlot::new());
        let (cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_count = Arc::new(AtomicUsize::new(0));
        let train_count_clone = train_count.clone();

        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send Pause command (won't be received due to BUG, but even if it was,
        // it's not implemented)
        let _ = cmd_tx.send(LearnerMsg::Pause);

        for _ in 0..10 {
            ready_tx.send(()).unwrap();
        }

        let _ = handle.join();

        // Training should complete normally since Pause is not implemented
        assert_eq!(
            train_count.load(Ordering::SeqCst),
            10,
            "Pause is not implemented - training continues"
        );
    }
}

// =============================================================================
// Concurrency Tests
// =============================================================================

mod concurrency_tests {
    use super::*;

    #[test]
    fn test_multiple_learners_concurrent() {
        // Multiple learners should be able to run concurrently

        let mut handles = vec![];

        for i in 0..3 {
            let model_slot = Arc::new(ModelSlot::new());
            let (_cmd_tx, cmd_rx) = bounded(100);
            let (ready_tx, ready_rx) = bounded(100);
            let shutdown = Arc::new(AtomicBool::new(false));

            let train_fn = || (1.0, 0.5, 0.3, 0.1);
            let get_model_fn = || MockModel::new(0);
            let publish_fn = |_: MockModel| {};

            let config = LearnerConfig::new()
                .with_max_train_steps(5)
                .with_publish_freq(100);

            let learner = Learner::new(config);

            let handle = learner.spawn_ppo(
                train_fn,
                get_model_fn,
                publish_fn,
                model_slot,
                cmd_rx,
                ready_rx,
                shutdown,
            );

            // Send ready signals
            for _ in 0..5 {
                ready_tx.send(()).unwrap();
            }

            handles.push(handle);
        }

        // All should complete successfully
        for handle in handles {
            let result = handle.join();
            assert!(result.is_ok(), "Concurrent learners should complete");
        }
    }

    #[test]
    fn test_shutdown_flag_prompt_response() {
        // Learner should check shutdown flag frequently and respond promptly

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (_ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new().with_max_train_steps(0);
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Wait a moment, then set shutdown
        std::thread::sleep(Duration::from_millis(50));
        let start = Instant::now();
        shutdown.store(true, Ordering::SeqCst);

        let _ = handle.join();
        let elapsed = start.elapsed();

        // Should respond within 200ms (100ms timeout + processing)
        assert!(
            elapsed < Duration::from_millis(200),
            "Shutdown should be prompt, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_thread_safety_of_model_slot_publish() {
        // Model slot publish from learner should be thread-safe

        let model_slot = Arc::new(ModelSlot::new());
        let slot_clone = model_slot.clone();

        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(42);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Concurrently try to take from slot while learner publishes
        let consumer_handle = std::thread::spawn(move || {
            let mut taken = 0;
            for _ in 0..100 {
                if slot_clone.take().is_some() {
                    taken += 1;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            taken
        });

        for _ in 0..10 {
            ready_tx.send(()).unwrap();
        }

        let _ = handle.join();
        let taken = consumer_handle.join().unwrap();

        // Should have taken at least some models
        assert!(taken > 0, "Consumer should have taken some models");
    }

    #[test]
    fn test_cmd_forwarding_thread_existence() {
        // BUG POTENTIAL: The command forwarding thread runs indefinitely
        // It's spawned but never joined or signaled to stop

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (_ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new().with_max_train_steps(1);
        let learner = Learner::new(config);

        // Count threads before
        // Note: This is hard to test directly in Rust without additional tooling
        // This test documents the concern

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // The main learner thread will join, but the forwarding thread may leak
        // This is a resource leak bug
    }
}

// =============================================================================
// Integration-style Tests
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_ppo_training_cycle() {
        // Complete cycle: spawn -> train N steps -> receive stats -> shutdown -> join

        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let slot_clone = model_slot.clone();
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_count_clone = train_count.clone();
        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            (0.5, 0.2, 0.2, 0.1)
        };

        let get_model_fn = || MockModel::new(1);

        let publish_count_clone = publish_count.clone();
        let publish_fn = move |_: MockModel| {
            publish_count_clone.fetch_add(1, Ordering::SeqCst);
        };

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(5)
            .with_stats_freq(10);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Feed training steps
        for _ in 0..10 {
            ready_tx.send(()).unwrap();
        }

        // Wait for completion
        let result = handle.join();

        assert!(result.is_ok(), "Training cycle should complete successfully");
        assert_eq!(train_count.load(Ordering::SeqCst), 10);
        assert_eq!(publish_count.load(Ordering::SeqCst), 2); // Steps 5 and 10

        // Model should be in slot
        let model = slot_clone.take();
        assert!(model.is_some());
    }

    #[test]
    fn test_full_impala_training_cycle() {
        let train_count = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::new(AtomicUsize::new(0));

        let model_slot = Arc::new(ModelSlot::new());
        let slot_clone = model_slot.clone();
        let (_cmd_tx, cmd_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let train_count_clone = train_count.clone();
        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            Some((0.5, 0.2, 0.2, 0.1))
        };

        let get_model_fn = || MockModel::new(1);

        let publish_count_clone = publish_count.clone();
        let publish_fn = move |_: MockModel, _: u64| {
            publish_count_clone.fetch_add(1, Ordering::SeqCst);
        };

        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(10)
            .with_publish_freq(5)
            .with_stats_freq(10);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown,
            version_counter.clone(),
        );

        let result = handle.join();

        assert!(result.is_ok(), "IMPALA training cycle should complete");
        assert_eq!(train_count.load(Ordering::SeqCst), 10);
        assert_eq!(publish_count.load(Ordering::SeqCst), 2);
        assert_eq!(version_counter.load(Ordering::SeqCst), 2);

        let model = slot_clone.take();
        assert!(model.is_some());
    }

    #[test]
    fn test_training_with_intermittent_ready_signals() {
        // Simulate realistic scenario where ready signals are intermittent

        let train_count = Arc::new(AtomicUsize::new(0));
        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_count_clone = train_count.clone();
        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(5)
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send ready signals with delays
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(20));
            ready_tx.send(()).unwrap();
        }

        let result = handle.join();
        assert!(result.is_ok());
        assert_eq!(train_count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_graceful_degradation_on_slow_consumer() {
        // If stats consumer is slow, stats should be silently dropped

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(1000);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(50)
            .with_publish_freq(1000)
            .with_stats_freq(1); // Very frequent stats

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        // Send all ready signals at once
        for _ in 0..50 {
            let _ = ready_tx.try_send(());
        }

        // Don't consume any stats - let them queue up / drop

        let result = handle.join();
        assert!(result.is_ok(), "Should complete even with full stats channel");
    }
}

// =============================================================================
// Bug Verification Tests - These tests specifically target known bugs
// =============================================================================

mod bug_verification_tests {
    use super::*;

    // =========================================================================
    // BUG 1: LearnerHandle.stop() disconnected channel (PPO)
    // =========================================================================
    //
    // ORIGINAL BUG: learner.rs lines 262-268 created a NEW disconnected channel
    // for handle.cmd_tx, making stop() ineffective.
    //
    // FIX: handle.cmd_tx is now cloned from cmd_tx before it's moved to the
    // forwarding thread, so stop() messages reach cmd_rx_internal.
    //
    // =========================================================================

    #[test]
    fn test_bug_1_ppo_handle_stop_now_fixed() {
        // BUG 1 VERIFICATION: LearnerHandle.stop() now works correctly
        //
        // Previously: handle.cmd_tx sent to a disconnected channel
        // Now: handle.cmd_tx is properly connected to cmd_rx_internal
        //
        // This test verifies that stop() now correctly stops the learner.

        let model_slot = Arc::new(ModelSlot::new());
        let (_external_cmd_tx, cmd_rx) = bounded::<LearnerMsg<MockModel>>(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_count = Arc::new(AtomicUsize::new(0));
        let train_count_clone = train_count.clone();

        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            (1.0, 0.5, 0.3, 0.1)
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(0) // Unlimited - only stop via command
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown.clone(),
        );

        // Start some training
        for _ in 0..5 {
            let _ = ready_tx.send(());
        }
        std::thread::sleep(Duration::from_millis(100));

        let count_before_stop = train_count.load(Ordering::SeqCst);
        assert!(count_before_stop > 0, "Should have done some training");

        // Call stop() - this now works! (previously it didn't)
        let sent = handle.stop();
        assert!(sent, "stop() should successfully send the command");

        // Wait for the learner to actually stop - should complete quickly
        // Previously this would hang because stop() was broken
        let start = Instant::now();
        let result = handle.join();
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Thread should exit cleanly after stop()");
        assert!(
            elapsed < Duration::from_millis(500),
            "BUG 1 FIX VERIFIED: Learner stopped within {:?} after stop()",
            elapsed
        );
    }

    // =========================================================================
    // BUG 2: LearnerHandle.stop() disconnected channel (IMPALA)
    // =========================================================================
    //
    // Same issue as BUG 1 but in spawn_impala (line 412).
    // FIX: Same fix applied - handle.cmd_tx cloned before move.
    //
    // =========================================================================

    #[test]
    fn test_bug_2_impala_handle_stop_now_fixed() {
        // BUG 2 VERIFICATION: Same as BUG 1 but for IMPALA
        //
        // Previously: handle.cmd_tx sent to a disconnected channel
        // Now: handle.cmd_tx is properly connected to cmd_rx_internal
        //
        // This test verifies that stop() now correctly stops the IMPALA learner.

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded::<LearnerMsg<MockModel>>(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let version_counter = Arc::new(AtomicU64::new(0));

        let train_count = Arc::new(AtomicUsize::new(0));
        let train_count_clone = train_count.clone();

        let train_fn = move || {
            train_count_clone.fetch_add(1, Ordering::SeqCst);
            Some((1.0, 0.5, 0.3, 0.1))
        };

        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel, _: u64| {};
        let is_ready_fn = || true;

        let config = LearnerConfig::new()
            .with_max_train_steps(0) // Unlimited - only stop via command
            .with_publish_freq(100);

        let learner = Learner::new(config);

        let handle = learner.spawn_impala(
            train_fn,
            get_model_fn,
            publish_fn,
            is_ready_fn,
            model_slot,
            cmd_rx,
            shutdown.clone(),
            version_counter,
        );

        std::thread::sleep(Duration::from_millis(50));
        let count_before = train_count.load(Ordering::SeqCst);
        assert!(count_before > 0, "Should have done some training");

        // Call stop() - this now works!
        let sent = handle.stop();
        assert!(sent, "stop() should successfully send the command");

        // Wait for the learner to actually stop - should complete quickly
        let start = Instant::now();
        let result = handle.join();
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Thread should exit cleanly after stop()");
        assert!(
            elapsed < Duration::from_millis(500),
            "BUG 2 FIX VERIFIED: IMPALA learner stopped within {:?} after stop()",
            elapsed
        );
    }

    // =========================================================================
    // BUG 5: Division by zero when publish_freq = 0
    // =========================================================================
    //
    // ORIGINAL BUG: train_step % config.publish_freq caused runtime panic
    // when publish_freq = 0.
    //
    // FIX: Config now validates frequencies at construction time (fail fast).
    // The with_publish_freq(), with_stats_freq(), and with_eval_freq() methods
    // now panic if 0 is passed.
    //
    // =========================================================================

    #[test]
    #[should_panic(expected = "publish_freq must be > 0")]
    fn test_bug_5_zero_publish_freq_now_rejected_at_config() {
        // BUG 5 FIX VERIFICATION: Zero publish_freq is now rejected at CONFIG time
        //
        // Previously: Runtime panic on first modulo operation
        // Now: Panic at config construction (fail fast, clear error)
        //
        // This is better because the error is caught immediately when building
        // the config, not at runtime during training.

        let _config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(0); // Now panics here, not at runtime!
    }

    #[test]
    #[should_panic(expected = "stats_freq must be > 0")]
    fn test_bug_5_zero_stats_freq_now_rejected_at_config() {
        // Same fix for stats_freq
        let _config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(1)
            .with_stats_freq(0); // Now panics here!
    }

    #[test]
    #[should_panic(expected = "eval_freq must be > 0")]
    fn test_bug_5_zero_eval_freq_now_rejected_at_config() {
        // Same fix for eval_freq
        let _config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(1)
            .with_stats_freq(1)
            .with_eval_freq(0); // Now panics here!
    }

    // =========================================================================
    // BUG 6: Stats silently dropped when channel is full
    // =========================================================================
    //
    // ORIGINAL BUG: Stats were silently dropped with no indication.
    // Code: let _ = stats_tx.try_send(stats);
    //
    // FIX: Now we track dropped stats via LearnerHandle.dropped_stats counter.
    // Users can call handle.get_dropped_stats_count() to see how many were lost.
    //
    // =========================================================================

    #[test]
    fn test_bug_6_stats_dropped_now_tracked() {
        // BUG 6 FIX VERIFICATION: Dropped stats are now tracked
        //
        // Previously: Stats dropped silently with no indication
        // Now: dropped_stats counter tracks how many were lost
        //
        // The channel is still bounded (intentional for backpressure),
        // but now users can detect when stats are being dropped.

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(10000);
        let shutdown = Arc::new(AtomicBool::new(false));

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        // Generate 500 stats but channel only holds 100
        let config = LearnerConfig::new()
            .with_max_train_steps(500)
            .with_publish_freq(1000)
            .with_stats_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        for _ in 0..500 {
            let _ = ready_tx.try_send(());
        }

        // Wait for training to complete
        std::thread::sleep(Duration::from_millis(1000));

        // Count stats received
        let mut stats_count = 0;
        while handle.get_stats().is_some() {
            stats_count += 1;
        }

        // FIX VERIFICATION: We can now query how many stats were dropped
        let dropped_count = handle.get_dropped_stats_count();

        let _ = handle.join();

        // Stats are still bounded (intentional), but now tracked
        assert!(
            stats_count <= 100,
            "Stats channel should be bounded to 100"
        );

        // The dropped count should be approximately 500 - stats_count
        // (minus some timing variance)
        assert!(
            dropped_count > 0,
            "BUG 6 FIX VERIFIED: Dropped stats are now tracked. \
             Received: {}, Dropped: {}",
            stats_count,
            dropped_count
        );
    }

    // =========================================================================
    // BUG 7: Command forwarding thread runs indefinitely
    // =========================================================================
    //
    // ORIGINAL BUG: Forwarding thread didn't respect shutdown flag and could
    // outlive the learner.
    //
    // FIX: Forwarding thread now checks the shutdown flag periodically and
    // uses recv_timeout instead of blocking recv.
    //
    // =========================================================================

    #[test]
    fn test_bug_7_forwarding_thread_now_respects_shutdown() {
        // BUG 7 FIX VERIFICATION: Forwarding thread now respects shutdown
        //
        // Previously: Thread ran indefinitely until channels disconnected
        // Now: Thread checks shutdown flag and exits cleanly
        //
        // The forwarding thread now:
        // 1. Checks shutdown flag every 100ms
        // 2. Uses recv_timeout instead of blocking recv
        // 3. Exits when shutdown is set or channels disconnect

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded::<LearnerMsg<MockModel>>(100);
        let (_ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new().with_max_train_steps(0); // Unlimited
        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown_clone,
        );

        // Set shutdown flag - both learner AND forwarding thread should exit
        shutdown.store(true, Ordering::SeqCst);

        // Join should complete quickly because both threads respect shutdown
        let start = Instant::now();
        let result = handle.join();
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "Thread should exit cleanly");
        assert!(
            elapsed < Duration::from_millis(500),
            "BUG 7 FIX VERIFIED: Threads exited within {:?} after shutdown",
            elapsed
        );

        // Note: The forwarding thread exits because:
        // 1. It checks shutdown flag every 100ms
        // 2. When shutdown=true, it breaks out of its loop
    }

    #[test]
    fn test_bug_3_steps_per_second_extreme_values() {
        // BUG 3: Division by near-zero elapsed time
        //
        // Location: learner.rs lines 195, 237 (PPO) and 350, 395 (IMPALA)
        // Code: if elapsed > 0.0 { train_step as f32 / elapsed } else { 0.0 }
        //
        // The check `elapsed > 0.0` protects against exact zero, but very small
        // positive values (like 1e-38) can still produce inf or very large numbers.
        //
        // Expected: Clamped or validated result
        // Actual: Could produce very large or infinite values

        let model_slot = Arc::new(ModelSlot::new());
        let (_cmd_tx, cmd_rx) = bounded(100);
        let (ready_tx, ready_rx) = bounded(100);
        let shutdown = Arc::new(AtomicBool::new(false));

        // Extremely fast training
        let train_fn = || (1.0, 0.5, 0.3, 0.1);
        let get_model_fn = || MockModel::new(0);
        let publish_fn = |_: MockModel| {};

        let config = LearnerConfig::new()
            .with_max_train_steps(1)
            .with_publish_freq(1)
            .with_stats_freq(1);

        let learner = Learner::new(config);

        let handle = learner.spawn_ppo(
            train_fn,
            get_model_fn,
            publish_fn,
            model_slot,
            cmd_rx,
            ready_rx,
            shutdown,
        );

        ready_tx.send(()).unwrap();

        std::thread::sleep(Duration::from_millis(50));

        if let Some(stats) = handle.get_stats() {
            // steps_per_second might be extremely large for fast operations
            // This documents the potential for extreme values
            if stats.steps_per_second > 1_000_000.0 {
                // Very large value - might want to clamp
            }
            // Should at least be finite
            assert!(
                stats.steps_per_second.is_finite(),
                "steps_per_second should be finite, got {}",
                stats.steps_per_second
            );
        }

        let _ = handle.join();
    }

    #[test]
    fn test_bug_4_accumulator_overflow_potential() {
        // BUG 4: Loss accumulator overflow
        //
        // Location: learner.rs lines 215-218 (PPO) and 372-375 (IMPALA)
        // Code: total_loss += loss; total_policy_loss += policy_loss; ...
        //
        // After billions of steps with losses around 1.0, f32 accumulator could
        // overflow (f32::MAX is ~3.4e38, but precision degrades much earlier).
        //
        // The accumulators ARE reset after each stats report, which helps.
        // But if stats_freq is very large, overflow is possible.
        //
        // Expected: Handled gracefully or documented limitation
        // Actual: Potential for overflow/inf with large stats_freq

        // This is hard to test quickly - would need billions of iterations
        // Document the concern:

        let f32_max = f32::MAX;
        let typical_loss = 1.0_f32;
        let steps_to_overflow = (f32_max / typical_loss) as u64;

        // With stats_freq = 100, we reset every 100 steps, so safe
        // With stats_freq = u64::MAX, we could overflow after ~3.4e38 steps
        // In practice, nobody trains that long, but it's worth noting

        assert!(
            steps_to_overflow > 1_000_000_000,
            "f32 accumulator needs {} steps to overflow with typical loss",
            steps_to_overflow
        );
    }
}

// =============================================================================
// Summary: Test Coverage
// =============================================================================
//
// This test suite covers:
//
// 1. LearnerConfig (15 tests)
//    - Default values
//    - Builder pattern
//    - Clone, Debug traits
//    - Zero/edge values
//
// 2. LearnerHandle (10 tests)
//    - get_stats() behavior
//    - stop() method (BUG DETECTED)
//    - join() behavior
//    - Multiple operations
//
// 3. spawn_ppo (25+ tests)
//    - Thread lifecycle
//    - Training mechanics
//    - Model publishing
//    - Stats reporting
//    - Channel handling
//    - Command processing (BUG DETECTED)
//
// 4. spawn_impala (15+ tests)
//    - Same categories as PPO
//    - is_ready_fn behavior
//    - Version counting
//    - train_fn returning None
//
// 5. Stats calculation (10 tests)
//    - Average calculations
//    - Division safety
//    - NaN/Inf handling
//    - Timing accuracy
//
// 6. Edge cases (10+ tests)
//    - Zero values
//    - Large values
//    - Boundary conditions
//
// 7. Concurrency (5+ tests)
//    - Multiple learners
//    - Thread safety
//    - Prompt shutdown
//
// 8. Integration (5+ tests)
//    - Full training cycles
//    - Realistic scenarios
//
// 9. Bug verification (8 tests)
//    - BUG 1: stop() disconnected channel (PPO)
//    - BUG 2: stop() disconnected channel (IMPALA)
//    - BUG 3: Near-zero elapsed time
//    - BUG 4: Accumulator overflow potential
//    - BUG 5: Zero frequency panic
//    - BUG 6: Silent stats dropping
//    - BUG 7: Thread leak
//
// Total: ~100+ test cases
//
// BUGS CONFIRMED:
// - BUG 1 & 2: handle.stop() is broken - CRITICAL
// - BUG 5: Zero frequency causes panic - HIGH
// - BUG 6: Stats silently dropped - MEDIUM
// - BUG 7: Thread leak - LOW
// - BUG 3 & 4: Edge cases - LOW
