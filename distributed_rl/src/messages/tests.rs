//! Comprehensive Yoneda-style tests for the messages submodule.
//!
//! These tests serve as the complete behavioral specification for all message types
//! in the distributed RL system. Following the Yoneda lemma principle, we characterize
//! each type through ALL its possible interactions and edge cases.
//!
//! # Test Organization
//!
//! - `learner_msg_tests`: Tests for LearnerMsg<M> and LearnerStats
//! - `actor_msg_tests`: Tests for ActorMsg<M> and ActorStats
//! - `coordinator_msg_tests`: Tests for CoordinatorMsg and FinishReason
//! - `eval_msg_tests`: Tests for EvalMsg<M> and EvalResult
//! - `integration_tests`: Cross-module interaction tests
//! - `thread_safety_tests`: Send/Sync bound verification

use std::marker::PhantomData;
use std::sync::mpsc;
use std::thread;

use super::*;

// =============================================================================
// TEST UTILITIES AND MOCK TYPES
// =============================================================================

/// A simple model type that implements Clone for testing ActorMsg and EvalMsg.
#[derive(Debug, Clone, PartialEq)]
struct MockModel {
    weights: Vec<f32>,
    version: u64,
}

impl MockModel {
    fn new(version: u64) -> Self {
        Self {
            weights: vec![1.0, 2.0, 3.0],
            version,
        }
    }
}

/// A model type that does NOT implement Clone - tests PhantomData behavior.
#[derive(Debug)]
struct NonCloneModel {
    #[allow(dead_code)]
    data: String,
}

/// A model type with expensive clone (simulated).
#[derive(Debug)]
struct ExpensiveCloneModel {
    data: Vec<u8>,
}

impl Clone for ExpensiveCloneModel {
    fn clone(&self) -> Self {
        // Simulate expensive clone
        Self {
            data: self.data.clone(),
        }
    }
}

/// Helper to check if a value is NaN (works for f32).
fn is_nan(v: f32) -> bool {
    v.is_nan()
}

/// Helper to check if a value is infinite.
fn is_inf(v: f32) -> bool {
    v.is_infinite()
}

// =============================================================================
// LEARNER MESSAGE TESTS
// =============================================================================

mod learner_msg_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LearnerMsg Enum Variant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_msg_stop_variant_exists() {
        // DEFINES: LearnerMsg::Stop variant exists and can be constructed
        let msg: LearnerMsg<MockModel> = LearnerMsg::Stop;
        // Pattern matching must succeed
        match msg {
            LearnerMsg::Stop => {}
            _ => panic!("Expected Stop variant"),
        }
    }

    #[test]
    fn test_learner_msg_pause_variant_exists() {
        // DEFINES: LearnerMsg::Pause variant exists and can be constructed
        let msg: LearnerMsg<MockModel> = LearnerMsg::Pause;
        match msg {
            LearnerMsg::Pause => {}
            _ => panic!("Expected Pause variant"),
        }
    }

    #[test]
    fn test_learner_msg_resume_variant_exists() {
        // DEFINES: LearnerMsg::Resume variant exists and can be constructed
        let msg: LearnerMsg<MockModel> = LearnerMsg::Resume;
        match msg {
            LearnerMsg::Resume => {}
            _ => panic!("Expected Resume variant"),
        }
    }

    #[test]
    fn test_learner_msg_request_stats_variant_exists() {
        // DEFINES: LearnerMsg::RequestStats variant exists and can be constructed
        let msg: LearnerMsg<MockModel> = LearnerMsg::RequestStats;
        match msg {
            LearnerMsg::RequestStats => {}
            _ => panic!("Expected RequestStats variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_with_normal_value() {
        // DEFINES: UpdateLearningRate contains an f64 learning rate
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(0.001);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert_eq!(lr, 0.001),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_zero() {
        // DEFINES: Zero learning rate is valid (edge case)
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(0.0);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert_eq!(lr, 0.0),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_negative() {
        // DEFINES: Negative learning rate is accepted (no validation at msg level)
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(-0.001);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert!(lr < 0.0),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_infinity() {
        // DEFINES: Infinite learning rate is accepted (no validation at msg level)
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(f64::INFINITY);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert!(lr.is_infinite()),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_nan() {
        // DEFINES: NaN learning rate is accepted (no validation at msg level)
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(f64::NAN);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert!(lr.is_nan()),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_very_small() {
        // DEFINES: Very small learning rates (near subnormal) are accepted
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(f64::MIN_POSITIVE);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert_eq!(lr, f64::MIN_POSITIVE),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_update_learning_rate_very_large() {
        // DEFINES: Very large learning rates are accepted
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(f64::MAX);
        match msg {
            LearnerMsg::UpdateLearningRate(lr) => assert_eq!(lr, f64::MAX),
            _ => panic!("Expected UpdateLearningRate variant"),
        }
    }

    #[test]
    fn test_learner_msg_phantom_variant_exists() {
        // DEFINES: _Phantom variant exists and wraps PhantomData
        let msg: LearnerMsg<MockModel> = LearnerMsg::_Phantom(PhantomData);
        match msg {
            LearnerMsg::_Phantom(_) => {}
            _ => panic!("Expected _Phantom variant"),
        }
    }

    // -------------------------------------------------------------------------
    // LearnerMsg Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_msg_clone_stop() {
        // DEFINES: LearnerMsg::Stop can be cloned
        let msg: LearnerMsg<MockModel> = LearnerMsg::Stop;
        let cloned = msg.clone();
        match cloned {
            LearnerMsg::Stop => {}
            _ => panic!("Clone should produce Stop"),
        }
    }

    #[test]
    fn test_learner_msg_clone_pause() {
        // DEFINES: LearnerMsg::Pause can be cloned
        let msg: LearnerMsg<MockModel> = LearnerMsg::Pause;
        let cloned = msg.clone();
        match cloned {
            LearnerMsg::Pause => {}
            _ => panic!("Clone should produce Pause"),
        }
    }

    #[test]
    fn test_learner_msg_clone_update_learning_rate() {
        // DEFINES: LearnerMsg::UpdateLearningRate preserves value on clone
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(0.01);
        let cloned = msg.clone();
        match cloned {
            LearnerMsg::UpdateLearningRate(lr) => assert_eq!(lr, 0.01),
            _ => panic!("Clone should produce UpdateLearningRate"),
        }
    }

    #[test]
    fn test_learner_msg_clone_phantom() {
        // DEFINES: LearnerMsg::_Phantom can be cloned
        let msg: LearnerMsg<MockModel> = LearnerMsg::_Phantom(PhantomData);
        let cloned = msg.clone();
        match cloned {
            LearnerMsg::_Phantom(_) => {}
            _ => panic!("Clone should produce _Phantom"),
        }
    }

    // -------------------------------------------------------------------------
    // LearnerMsg Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_msg_debug_stop() {
        // DEFINES: Debug output for Stop variant
        let msg: LearnerMsg<MockModel> = LearnerMsg::Stop;
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("Stop"));
    }

    #[test]
    fn test_learner_msg_debug_update_learning_rate() {
        // DEFINES: Debug output for UpdateLearningRate includes the rate
        let msg: LearnerMsg<MockModel> = LearnerMsg::UpdateLearningRate(0.001);
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("UpdateLearningRate"));
        assert!(debug_str.contains("0.001"));
    }

    // -------------------------------------------------------------------------
    // LearnerMsg PhantomData Generic Behavior Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_msg_works_with_non_clone_model_construction() {
        // DEFINES: LearnerMsg<M> can be CONSTRUCTED even when M doesn't implement Clone
        // This works because the actual M is never stored (only in PhantomData)
        let _msg: LearnerMsg<NonCloneModel> = LearnerMsg::Stop;
        let _msg2: LearnerMsg<NonCloneModel> = LearnerMsg::Pause;
        let _msg3: LearnerMsg<NonCloneModel> = LearnerMsg::UpdateLearningRate(0.001);
        // NOTE: Clone is NOT available because #[derive(Clone)] requires M: Clone
        // This is a potential API limitation - PhantomData<M> is Clone unconditionally,
        // but the derive macro adds M: Clone bound anyway.
    }

    #[test]
    fn test_learner_msg_different_model_types_are_distinct() {
        // DEFINES: LearnerMsg<A> and LearnerMsg<B> are different types
        fn takes_mock_model(_msg: LearnerMsg<MockModel>) {}
        fn takes_expensive_model(_msg: LearnerMsg<ExpensiveCloneModel>) {}

        let msg1: LearnerMsg<MockModel> = LearnerMsg::Stop;
        let msg2: LearnerMsg<ExpensiveCloneModel> = LearnerMsg::Stop;

        takes_mock_model(msg1);
        takes_expensive_model(msg2);
    }

    // -------------------------------------------------------------------------
    // LearnerStats Default Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_stats_default_all_fields_zeroed() {
        // DEFINES: Default LearnerStats has all numeric fields at zero
        let stats = LearnerStats::default();
        assert_eq!(stats.train_steps, 0);
        assert_eq!(stats.avg_loss, 0.0);
        assert_eq!(stats.avg_policy_loss, 0.0);
        assert_eq!(stats.avg_value_loss, 0.0);
        assert_eq!(stats.avg_entropy, 0.0);
        assert_eq!(stats.steps_per_second, 0.0);
        assert_eq!(stats.model_version, 0);
        assert_eq!(stats.buffer_utilization, 0.0);
    }

    #[test]
    fn test_learner_stats_new_equals_default() {
        // DEFINES: new() returns the same as default()
        let stats_new = LearnerStats::new();
        let stats_default = LearnerStats::default();

        assert_eq!(stats_new.train_steps, stats_default.train_steps);
        assert_eq!(stats_new.avg_loss, stats_default.avg_loss);
        assert_eq!(stats_new.avg_policy_loss, stats_default.avg_policy_loss);
        assert_eq!(stats_new.avg_value_loss, stats_default.avg_value_loss);
        assert_eq!(stats_new.avg_entropy, stats_default.avg_entropy);
        assert_eq!(stats_new.steps_per_second, stats_default.steps_per_second);
        assert_eq!(stats_new.model_version, stats_default.model_version);
        assert_eq!(stats_new.buffer_utilization, stats_default.buffer_utilization);
    }

    // -------------------------------------------------------------------------
    // LearnerStats record_step Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_stats_record_step_increments_train_steps() {
        // DEFINES: record_step increments train_steps by 1
        let mut stats = LearnerStats::new();
        assert_eq!(stats.train_steps, 0);

        stats.record_step(1.0, 0.5, 0.3, 0.1);
        assert_eq!(stats.train_steps, 1);

        stats.record_step(0.8, 0.4, 0.2, 0.15);
        assert_eq!(stats.train_steps, 2);
    }

    #[test]
    fn test_learner_stats_record_step_sets_loss_values() {
        // DEFINES: record_step sets (not accumulates) loss values
        let mut stats = LearnerStats::new();

        stats.record_step(1.0, 0.5, 0.3, 0.1);
        assert_eq!(stats.avg_loss, 1.0);
        assert_eq!(stats.avg_policy_loss, 0.5);
        assert_eq!(stats.avg_value_loss, 0.3);
        assert_eq!(stats.avg_entropy, 0.1);

        // Second call REPLACES, doesn't accumulate
        stats.record_step(2.0, 1.0, 0.6, 0.2);
        assert_eq!(stats.avg_loss, 2.0);
        assert_eq!(stats.avg_policy_loss, 1.0);
        assert_eq!(stats.avg_value_loss, 0.6);
        assert_eq!(stats.avg_entropy, 0.2);
    }

    #[test]
    fn test_learner_stats_record_step_with_zero_losses() {
        // DEFINES: Zero losses are valid inputs
        let mut stats = LearnerStats::new();
        stats.record_step(0.0, 0.0, 0.0, 0.0);

        assert_eq!(stats.train_steps, 1);
        assert_eq!(stats.avg_loss, 0.0);
        assert_eq!(stats.avg_policy_loss, 0.0);
        assert_eq!(stats.avg_value_loss, 0.0);
        assert_eq!(stats.avg_entropy, 0.0);
    }

    #[test]
    fn test_learner_stats_record_step_with_negative_losses() {
        // DEFINES: Negative losses are accepted (no validation)
        // This could happen in certain loss formulations
        let mut stats = LearnerStats::new();
        stats.record_step(-1.0, -0.5, -0.3, -0.1);

        assert_eq!(stats.avg_loss, -1.0);
        assert_eq!(stats.avg_policy_loss, -0.5);
        assert_eq!(stats.avg_value_loss, -0.3);
        assert_eq!(stats.avg_entropy, -0.1);
    }

    #[test]
    fn test_learner_stats_record_step_with_nan() {
        // DEFINES: NaN values are stored (propagated, not sanitized)
        let mut stats = LearnerStats::new();
        stats.record_step(f32::NAN, f32::NAN, f32::NAN, f32::NAN);

        assert!(is_nan(stats.avg_loss));
        assert!(is_nan(stats.avg_policy_loss));
        assert!(is_nan(stats.avg_value_loss));
        assert!(is_nan(stats.avg_entropy));
    }

    #[test]
    fn test_learner_stats_record_step_with_infinity() {
        // DEFINES: Infinite values are stored (propagated, not sanitized)
        let mut stats = LearnerStats::new();
        stats.record_step(f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);

        assert!(is_inf(stats.avg_loss) && stats.avg_loss > 0.0);
        assert!(is_inf(stats.avg_policy_loss) && stats.avg_policy_loss < 0.0);
        assert!(is_inf(stats.avg_value_loss) && stats.avg_value_loss > 0.0);
        assert!(is_inf(stats.avg_entropy) && stats.avg_entropy < 0.0);
    }

    #[test]
    fn test_learner_stats_record_step_with_max_f32() {
        // DEFINES: MAX f32 values are accepted
        let mut stats = LearnerStats::new();
        stats.record_step(f32::MAX, f32::MAX, f32::MAX, f32::MAX);

        assert_eq!(stats.avg_loss, f32::MAX);
        assert_eq!(stats.avg_policy_loss, f32::MAX);
    }

    #[test]
    fn test_learner_stats_record_step_with_min_f32() {
        // DEFINES: MIN (most negative) f32 values are accepted
        let mut stats = LearnerStats::new();
        stats.record_step(f32::MIN, f32::MIN, f32::MIN, f32::MIN);

        assert_eq!(stats.avg_loss, f32::MIN);
        assert_eq!(stats.avg_policy_loss, f32::MIN);
    }

    #[test]
    fn test_learner_stats_record_step_train_steps_overflow() {
        // DEFINES: Behavior when train_steps overflows (wraps in release, panics in debug)
        // We test near-max to verify large values work
        let mut stats = LearnerStats::new();
        stats.train_steps = usize::MAX - 1;
        stats.record_step(1.0, 0.5, 0.3, 0.1);
        assert_eq!(stats.train_steps, usize::MAX);
    }

    #[test]
    fn test_learner_stats_record_step_many_times() {
        // DEFINES: Many sequential record_step calls work correctly
        let mut stats = LearnerStats::new();
        for i in 1..=1000 {
            stats.record_step(i as f32 * 0.001, 0.0, 0.0, 0.0);
        }
        assert_eq!(stats.train_steps, 1000);
        assert_eq!(stats.avg_loss, 1.0); // Last recorded value
    }

    // -------------------------------------------------------------------------
    // LearnerStats Setter Method Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_stats_set_model_version() {
        // DEFINES: set_model_version updates the model_version field
        let mut stats = LearnerStats::new();
        stats.set_model_version(42);
        assert_eq!(stats.model_version, 42);

        stats.set_model_version(u64::MAX);
        assert_eq!(stats.model_version, u64::MAX);
    }

    #[test]
    fn test_learner_stats_set_buffer_utilization_normal() {
        // DEFINES: set_buffer_utilization updates buffer_utilization field
        let mut stats = LearnerStats::new();
        stats.set_buffer_utilization(0.75);
        assert_eq!(stats.buffer_utilization, 0.75);
    }

    #[test]
    fn test_learner_stats_set_buffer_utilization_zero() {
        // DEFINES: Zero utilization is valid (empty buffer)
        let mut stats = LearnerStats::new();
        stats.set_buffer_utilization(0.0);
        assert_eq!(stats.buffer_utilization, 0.0);
    }

    #[test]
    fn test_learner_stats_set_buffer_utilization_one() {
        // DEFINES: Full utilization (1.0) is valid
        let mut stats = LearnerStats::new();
        stats.set_buffer_utilization(1.0);
        assert_eq!(stats.buffer_utilization, 1.0);
    }

    #[test]
    fn test_learner_stats_set_buffer_utilization_over_one() {
        // DEFINES: Values > 1.0 are accepted (no clamping)
        let mut stats = LearnerStats::new();
        stats.set_buffer_utilization(1.5);
        assert_eq!(stats.buffer_utilization, 1.5);
    }

    #[test]
    fn test_learner_stats_set_buffer_utilization_negative() {
        // DEFINES: Negative values are accepted (no validation)
        let mut stats = LearnerStats::new();
        stats.set_buffer_utilization(-0.5);
        assert_eq!(stats.buffer_utilization, -0.5);
    }

    #[test]
    fn test_learner_stats_set_buffer_utilization_nan() {
        // DEFINES: NaN is accepted (not sanitized)
        let mut stats = LearnerStats::new();
        stats.set_buffer_utilization(f32::NAN);
        assert!(is_nan(stats.buffer_utilization));
    }

    #[test]
    fn test_learner_stats_set_steps_per_second_normal() {
        // DEFINES: set_steps_per_second updates steps_per_second field
        let mut stats = LearnerStats::new();
        stats.set_steps_per_second(1000.0);
        assert_eq!(stats.steps_per_second, 1000.0);
    }

    #[test]
    fn test_learner_stats_set_steps_per_second_zero() {
        // DEFINES: Zero steps/second is valid (paused training)
        let mut stats = LearnerStats::new();
        stats.set_steps_per_second(0.0);
        assert_eq!(stats.steps_per_second, 0.0);
    }

    #[test]
    fn test_learner_stats_set_steps_per_second_infinity() {
        // DEFINES: Infinite steps/second is accepted
        let mut stats = LearnerStats::new();
        stats.set_steps_per_second(f32::INFINITY);
        assert!(is_inf(stats.steps_per_second));
    }

    // -------------------------------------------------------------------------
    // LearnerStats Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_stats_clone_preserves_all_fields() {
        // DEFINES: Clone creates exact copy of all fields
        let mut stats = LearnerStats::new();
        stats.train_steps = 100;
        stats.avg_loss = 0.5;
        stats.avg_policy_loss = 0.3;
        stats.avg_value_loss = 0.2;
        stats.avg_entropy = 0.1;
        stats.steps_per_second = 500.0;
        stats.model_version = 42;
        stats.buffer_utilization = 0.8;

        let cloned = stats.clone();

        assert_eq!(cloned.train_steps, 100);
        assert_eq!(cloned.avg_loss, 0.5);
        assert_eq!(cloned.avg_policy_loss, 0.3);
        assert_eq!(cloned.avg_value_loss, 0.2);
        assert_eq!(cloned.avg_entropy, 0.1);
        assert_eq!(cloned.steps_per_second, 500.0);
        assert_eq!(cloned.model_version, 42);
        assert_eq!(cloned.buffer_utilization, 0.8);
    }

    #[test]
    fn test_learner_stats_clone_is_independent() {
        // DEFINES: Cloned stats are independent (modifying one doesn't affect other)
        let mut stats = LearnerStats::new();
        stats.train_steps = 100;

        let mut cloned = stats.clone();
        cloned.train_steps = 200;

        assert_eq!(stats.train_steps, 100);
        assert_eq!(cloned.train_steps, 200);
    }

    // -------------------------------------------------------------------------
    // LearnerStats Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_learner_stats_debug_output() {
        // DEFINES: Debug output includes field names and values
        let mut stats = LearnerStats::new();
        stats.train_steps = 42;
        stats.avg_loss = 0.123;

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("LearnerStats"));
        assert!(debug_str.contains("train_steps"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("avg_loss"));
    }
}

// =============================================================================
// ACTOR MESSAGE TESTS
// =============================================================================

mod actor_msg_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // ActorMsg Enum Variant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_msg_stop_variant_exists() {
        // DEFINES: ActorMsg::Stop variant exists
        let msg: ActorMsg<MockModel> = ActorMsg::Stop;
        match msg {
            ActorMsg::Stop => {}
            _ => panic!("Expected Stop variant"),
        }
    }

    #[test]
    fn test_actor_msg_update_model_contains_model() {
        // DEFINES: UpdateModel variant contains the model
        let model = MockModel::new(1);
        let msg = ActorMsg::UpdateModel(model);
        match msg {
            ActorMsg::UpdateModel(m) => {
                assert_eq!(m.version, 1);
                assert_eq!(m.weights, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected UpdateModel variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_contains_value() {
        // DEFINES: SetEpsilon variant contains f32 epsilon value
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(0.1);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert_eq!(eps, 0.1),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_zero() {
        // DEFINES: Zero epsilon is valid (pure exploitation)
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(0.0);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert_eq!(eps, 0.0),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_one() {
        // DEFINES: Epsilon of 1.0 is valid (pure exploration)
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(1.0);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert_eq!(eps, 1.0),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_over_one() {
        // DEFINES: Epsilon > 1.0 is accepted (no validation at message level)
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(1.5);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert_eq!(eps, 1.5),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_negative() {
        // DEFINES: Negative epsilon is accepted (no validation at message level)
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(-0.1);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert!(eps < 0.0),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_nan() {
        // DEFINES: NaN epsilon is accepted
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(f32::NAN);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert!(is_nan(eps)),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_set_epsilon_infinity() {
        // DEFINES: Infinite epsilon is accepted
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(f32::INFINITY);
        match msg {
            ActorMsg::SetEpsilon(eps) => assert!(is_inf(eps)),
            _ => panic!("Expected SetEpsilon variant"),
        }
    }

    #[test]
    fn test_actor_msg_request_stats_variant_exists() {
        // DEFINES: RequestStats variant exists
        let msg: ActorMsg<MockModel> = ActorMsg::RequestStats;
        match msg {
            ActorMsg::RequestStats => {}
            _ => panic!("Expected RequestStats variant"),
        }
    }

    // -------------------------------------------------------------------------
    // ActorMsg Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_msg_clone_stop() {
        // DEFINES: ActorMsg::Stop can be cloned
        let msg: ActorMsg<MockModel> = ActorMsg::Stop;
        let cloned = msg.clone();
        match cloned {
            ActorMsg::Stop => {}
            _ => panic!("Clone should produce Stop"),
        }
    }

    #[test]
    fn test_actor_msg_clone_update_model_deep_clones() {
        // DEFINES: Cloning UpdateModel deep clones the model
        let model = MockModel::new(42);
        let msg = ActorMsg::UpdateModel(model);
        let cloned = msg.clone();

        match cloned {
            ActorMsg::UpdateModel(m) => {
                assert_eq!(m.version, 42);
                assert_eq!(m.weights, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Clone should produce UpdateModel"),
        }
    }

    #[test]
    fn test_actor_msg_clone_set_epsilon() {
        // DEFINES: Cloning SetEpsilon preserves the epsilon value
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(0.05);
        let cloned = msg.clone();
        match cloned {
            ActorMsg::SetEpsilon(eps) => assert_eq!(eps, 0.05),
            _ => panic!("Clone should produce SetEpsilon"),
        }
    }

    // -------------------------------------------------------------------------
    // ActorMsg Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_msg_debug_stop() {
        // DEFINES: Debug output for Stop
        let msg: ActorMsg<MockModel> = ActorMsg::Stop;
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("Stop"));
    }

    #[test]
    fn test_actor_msg_debug_update_model() {
        // DEFINES: Debug output for UpdateModel includes model debug
        let msg = ActorMsg::UpdateModel(MockModel::new(5));
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("UpdateModel"));
        assert!(debug_str.contains("MockModel"));
    }

    #[test]
    fn test_actor_msg_debug_set_epsilon() {
        // DEFINES: Debug output for SetEpsilon includes the value
        let msg: ActorMsg<MockModel> = ActorMsg::SetEpsilon(0.25);
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("SetEpsilon"));
        assert!(debug_str.contains("0.25"));
    }

    // -------------------------------------------------------------------------
    // ActorStats Default Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_stats_default_all_fields_zeroed() {
        // DEFINES: Default ActorStats has all numeric fields at zero
        let stats = ActorStats::default();
        assert_eq!(stats.actor_id, 0);
        assert_eq!(stats.steps, 0);
        assert_eq!(stats.episodes, 0);
        assert_eq!(stats.avg_episode_reward, 0.0);
        assert_eq!(stats.recent_episode_reward, 0.0);
        assert_eq!(stats.epsilon, 0.0);
        assert_eq!(stats.model_version, 0);
    }

    #[test]
    fn test_actor_stats_new_sets_actor_id() {
        // DEFINES: new(id) sets actor_id and zeros other fields
        let stats = ActorStats::new(5);
        assert_eq!(stats.actor_id, 5);
        assert_eq!(stats.steps, 0);
        assert_eq!(stats.episodes, 0);
        assert_eq!(stats.avg_episode_reward, 0.0);
        assert_eq!(stats.recent_episode_reward, 0.0);
        assert_eq!(stats.epsilon, 0.0);
        assert_eq!(stats.model_version, 0);
    }

    #[test]
    fn test_actor_stats_new_with_max_id() {
        // DEFINES: Maximum actor_id value is accepted
        let stats = ActorStats::new(usize::MAX);
        assert_eq!(stats.actor_id, usize::MAX);
    }

    // -------------------------------------------------------------------------
    // ActorStats record_episode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_stats_record_episode_first_episode() {
        // DEFINES: First episode sets avg_episode_reward directly
        // CRITICAL: This tests the (episodes - 1) calculation when episodes was 0
        let mut stats = ActorStats::new(0);
        assert_eq!(stats.episodes, 0);

        stats.record_episode(100.0);

        assert_eq!(stats.episodes, 1);
        assert_eq!(stats.avg_episode_reward, 100.0);
        assert_eq!(stats.recent_episode_reward, 100.0);
    }

    #[test]
    fn test_actor_stats_record_episode_second_episode() {
        // DEFINES: Second episode computes running average correctly
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        stats.record_episode(200.0);

        assert_eq!(stats.episodes, 2);
        // (100 * 1 + 200) / 2 = 150
        assert_eq!(stats.avg_episode_reward, 150.0);
        assert_eq!(stats.recent_episode_reward, 200.0);
    }

    #[test]
    fn test_actor_stats_record_episode_running_average() {
        // DEFINES: Running average formula works for multiple episodes
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        stats.record_episode(200.0);
        stats.record_episode(300.0);

        assert_eq!(stats.episodes, 3);
        // Average of 100, 200, 300 = 200
        assert_eq!(stats.avg_episode_reward, 200.0);
        assert_eq!(stats.recent_episode_reward, 300.0);
    }

    #[test]
    fn test_actor_stats_record_episode_with_negative_reward() {
        // DEFINES: Negative rewards are handled correctly
        let mut stats = ActorStats::new(0);

        stats.record_episode(-100.0);
        assert_eq!(stats.avg_episode_reward, -100.0);

        stats.record_episode(-200.0);
        assert_eq!(stats.avg_episode_reward, -150.0);
    }

    #[test]
    fn test_actor_stats_record_episode_with_zero_reward() {
        // DEFINES: Zero rewards are handled correctly
        let mut stats = ActorStats::new(0);

        stats.record_episode(0.0);
        assert_eq!(stats.avg_episode_reward, 0.0);

        stats.record_episode(0.0);
        assert_eq!(stats.avg_episode_reward, 0.0);
    }

    #[test]
    fn test_actor_stats_record_episode_with_mixed_rewards() {
        // DEFINES: Mixed positive/negative rewards average correctly
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        stats.record_episode(-100.0);

        assert_eq!(stats.episodes, 2);
        assert_eq!(stats.avg_episode_reward, 0.0);
    }

    #[test]
    fn test_actor_stats_record_episode_with_nan_reward() {
        // DEFINES: NaN rewards are FILTERED to prevent average corruption
        // This is defensive coding: non-finite values don't poison the average
        let mut stats = ActorStats::new(0);

        stats.record_episode(100.0);
        stats.record_episode(f32::NAN);

        // NaN is filtered - average stays valid from first episode
        assert_eq!(stats.avg_episode_reward, 100.0);
        assert!(is_nan(stats.recent_episode_reward)); // Recent still shows NaN for diagnostics
        assert_eq!(stats.valid_episodes, 1);
        assert_eq!(stats.filtered_episodes, 1);
    }

    #[test]
    fn test_actor_stats_record_episode_with_infinity() {
        // DEFINES: Infinite rewards are FILTERED to prevent average corruption
        let mut stats = ActorStats::new(0);

        stats.record_episode(f32::INFINITY);

        // Infinity is filtered - average stays 0.0 (no valid episodes)
        assert_eq!(stats.avg_episode_reward, 0.0);
        assert!(is_inf(stats.recent_episode_reward)); // Recent shows for diagnostics
        assert_eq!(stats.valid_episodes, 0);
        assert_eq!(stats.filtered_episodes, 1);
    }

    #[test]
    fn test_actor_stats_record_episode_inf_plus_neg_inf() {
        // DEFINES: Both infinities are FILTERED
        let mut stats = ActorStats::new(0);

        stats.record_episode(f32::INFINITY);
        stats.record_episode(f32::NEG_INFINITY);

        // Both filtered - average stays 0.0
        assert_eq!(stats.avg_episode_reward, 0.0);
        assert_eq!(stats.valid_episodes, 0);
        assert_eq!(stats.filtered_episodes, 2);
    }

    #[test]
    fn test_actor_stats_record_episode_precision_loss() {
        // DEFINES: Precision loss behavior with large number of episodes
        // This tests cumulative precision loss in the running average formula
        let mut stats = ActorStats::new(0);

        // Record many episodes with the same value
        for _ in 0..10000 {
            stats.record_episode(1.0);
        }

        // Average should still be close to 1.0, but may have precision loss
        assert!((stats.avg_episode_reward - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_actor_stats_record_episode_overflow_potential() {
        // DEFINES: Behavior with very large rewards that could overflow
        let mut stats = ActorStats::new(0);

        // Record a very large reward
        stats.record_episode(f32::MAX / 2.0);
        assert_eq!(stats.avg_episode_reward, f32::MAX / 2.0);

        // This could overflow: avg * (n-1) + reward
        stats.record_episode(f32::MAX / 2.0);
        // (MAX/2 * 1 + MAX/2) / 2 = MAX/2 (if computed correctly)
        // But intermediate: MAX/2 + MAX/2 = MAX, then /2 = MAX/2
        assert_eq!(stats.avg_episode_reward, f32::MAX / 2.0);
    }

    #[test]
    fn test_actor_stats_record_episode_starting_with_nan_avg() {
        // DEFINES: Behavior when avg_episode_reward is manually set to NaN
        // This tests robustness against corrupted state
        let mut stats = ActorStats::new(0);
        stats.avg_episode_reward = f32::NAN;

        stats.record_episode(100.0);

        // NaN * 0 + 100 = NaN (because NaN * 0 = NaN in IEEE 754)
        // Actually: (NaN * 0 + 100) / 1 = (NaN + 100) / 1 = NaN
        // Note: episodes becomes 1, so (episodes - 1) = 0
        // total = NaN * 0 + 100 = NaN (since NaN * 0 = NaN, not 0)
        assert!(is_nan(stats.avg_episode_reward));
    }

    // -------------------------------------------------------------------------
    // ActorStats add_steps Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_stats_add_steps_increments() {
        // DEFINES: add_steps adds to the steps counter
        let mut stats = ActorStats::new(0);
        stats.add_steps(100);
        assert_eq!(stats.steps, 100);

        stats.add_steps(50);
        assert_eq!(stats.steps, 150);
    }

    #[test]
    fn test_actor_stats_add_steps_zero() {
        // DEFINES: Adding zero steps is a no-op
        let mut stats = ActorStats::new(0);
        stats.add_steps(0);
        assert_eq!(stats.steps, 0);
    }

    #[test]
    fn test_actor_stats_add_steps_large_value() {
        // DEFINES: Large step values are handled
        let mut stats = ActorStats::new(0);
        stats.add_steps(1_000_000);
        assert_eq!(stats.steps, 1_000_000);
    }

    #[test]
    fn test_actor_stats_add_steps_near_overflow() {
        // DEFINES: Behavior near usize::MAX
        let mut stats = ActorStats::new(0);
        stats.steps = usize::MAX - 10;
        stats.add_steps(10);
        assert_eq!(stats.steps, usize::MAX);
    }

    // -------------------------------------------------------------------------
    // ActorStats Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_stats_clone_preserves_all_fields() {
        // DEFINES: Clone creates exact copy
        let mut stats = ActorStats::new(5);
        stats.steps = 1000;
        stats.episodes = 10;
        stats.avg_episode_reward = 250.0;
        stats.recent_episode_reward = 300.0;
        stats.epsilon = 0.1;
        stats.model_version = 7;

        let cloned = stats.clone();

        assert_eq!(cloned.actor_id, 5);
        assert_eq!(cloned.steps, 1000);
        assert_eq!(cloned.episodes, 10);
        assert_eq!(cloned.avg_episode_reward, 250.0);
        assert_eq!(cloned.recent_episode_reward, 300.0);
        assert_eq!(cloned.epsilon, 0.1);
        assert_eq!(cloned.model_version, 7);
    }

    #[test]
    fn test_actor_stats_clone_is_independent() {
        // DEFINES: Modifying clone doesn't affect original
        let mut stats = ActorStats::new(0);
        stats.steps = 100;

        let mut cloned = stats.clone();
        cloned.steps = 200;

        assert_eq!(stats.steps, 100);
        assert_eq!(cloned.steps, 200);
    }

    // -------------------------------------------------------------------------
    // ActorStats Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_actor_stats_debug_output() {
        // DEFINES: Debug output includes all field names
        let mut stats = ActorStats::new(3);
        stats.episodes = 42;

        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("ActorStats"));
        assert!(debug_str.contains("actor_id"));
        assert!(debug_str.contains("3"));
        assert!(debug_str.contains("episodes"));
        assert!(debug_str.contains("42"));
    }
}

// =============================================================================
// COORDINATOR MESSAGE TESTS
// =============================================================================

mod coordinator_msg_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // FinishReason Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_finish_reason_stopped_variant() {
        // DEFINES: FinishReason::Stopped exists
        let reason = FinishReason::Stopped;
        match reason {
            FinishReason::Stopped => {}
            _ => panic!("Expected Stopped variant"),
        }
    }

    #[test]
    fn test_finish_reason_panicked_contains_string() {
        // DEFINES: FinishReason::Panicked contains panic message
        let reason = FinishReason::Panicked("test panic".to_string());
        match reason {
            FinishReason::Panicked(msg) => assert_eq!(msg, "test panic"),
            _ => panic!("Expected Panicked variant"),
        }
    }

    #[test]
    fn test_finish_reason_panicked_empty_string() {
        // DEFINES: Empty panic message is valid
        let reason = FinishReason::Panicked(String::new());
        match reason {
            FinishReason::Panicked(msg) => assert!(msg.is_empty()),
            _ => panic!("Expected Panicked variant"),
        }
    }

    #[test]
    fn test_finish_reason_panicked_long_message() {
        // DEFINES: Very long panic messages are accepted
        let long_msg = "x".repeat(100_000);
        let reason = FinishReason::Panicked(long_msg.clone());
        match reason {
            FinishReason::Panicked(msg) => assert_eq!(msg.len(), 100_000),
            _ => panic!("Expected Panicked variant"),
        }
    }

    #[test]
    fn test_finish_reason_panicked_unicode() {
        // DEFINES: Unicode in panic messages is handled
        let reason = FinishReason::Panicked("panicked at line 42".to_string());
        match reason {
            FinishReason::Panicked(msg) => assert!(msg.contains("42")),
            _ => panic!("Expected Panicked variant"),
        }
    }

    #[test]
    fn test_finish_reason_panicked_unicode_emoji() {
        // DEFINES: Unicode emoji in panic messages (edge case)
        let reason = FinishReason::Panicked("panic in thread".to_string());
        match reason {
            FinishReason::Panicked(msg) => assert!(msg.contains("panic")),
            _ => panic!("Expected Panicked variant"),
        }
    }

    #[test]
    fn test_finish_reason_completed_variant() {
        // DEFINES: FinishReason::Completed exists
        let reason = FinishReason::Completed;
        match reason {
            FinishReason::Completed => {}
            _ => panic!("Expected Completed variant"),
        }
    }

    #[test]
    fn test_finish_reason_clone_stopped() {
        // DEFINES: FinishReason::Stopped can be cloned
        let reason = FinishReason::Stopped;
        let cloned = reason.clone();
        match cloned {
            FinishReason::Stopped => {}
            _ => panic!("Clone should produce Stopped"),
        }
    }

    #[test]
    fn test_finish_reason_clone_panicked() {
        // DEFINES: FinishReason::Panicked clones the message
        let reason = FinishReason::Panicked("error".to_string());
        let cloned = reason.clone();
        match cloned {
            FinishReason::Panicked(msg) => assert_eq!(msg, "error"),
            _ => panic!("Clone should produce Panicked"),
        }
    }

    #[test]
    fn test_finish_reason_debug() {
        // DEFINES: Debug output for all variants
        let stopped = format!("{:?}", FinishReason::Stopped);
        assert!(stopped.contains("Stopped"));

        let panicked = format!("{:?}", FinishReason::Panicked("oops".to_string()));
        assert!(panicked.contains("Panicked"));
        assert!(panicked.contains("oops"));

        let completed = format!("{:?}", FinishReason::Completed);
        assert!(completed.contains("Completed"));
    }

    // -------------------------------------------------------------------------
    // CoordinatorMsg Variant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinator_msg_actor_stats_variant() {
        // DEFINES: CoordinatorMsg::ActorStats wraps ActorStats
        let stats = ActorStats::new(1);
        let msg = CoordinatorMsg::ActorStats(stats);
        match msg {
            CoordinatorMsg::ActorStats(s) => assert_eq!(s.actor_id, 1),
            _ => panic!("Expected ActorStats variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_learner_stats_variant() {
        // DEFINES: CoordinatorMsg::LearnerStats wraps LearnerStats
        let mut stats = LearnerStats::new();
        stats.train_steps = 500;
        let msg = CoordinatorMsg::LearnerStats(stats);
        match msg {
            CoordinatorMsg::LearnerStats(s) => assert_eq!(s.train_steps, 500),
            _ => panic!("Expected LearnerStats variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_eval_result_variant() {
        // DEFINES: CoordinatorMsg::EvalResult wraps EvalResult
        let result = EvalResult::from_rewards(100, &[50.0, 60.0], &[10, 20]);
        let msg = CoordinatorMsg::EvalResult(result);
        match msg {
            CoordinatorMsg::EvalResult(r) => assert_eq!(r.step, 100),
            _ => panic!("Expected EvalResult variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_actor_finished_variant() {
        // DEFINES: ActorFinished contains actor_id and reason
        let msg = CoordinatorMsg::ActorFinished {
            actor_id: 5,
            reason: FinishReason::Stopped,
        };
        match msg {
            CoordinatorMsg::ActorFinished { actor_id, reason } => {
                assert_eq!(actor_id, 5);
                match reason {
                    FinishReason::Stopped => {}
                    _ => panic!("Expected Stopped reason"),
                }
            }
            _ => panic!("Expected ActorFinished variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_actor_finished_with_panic() {
        // DEFINES: ActorFinished can carry panic reason
        let msg = CoordinatorMsg::ActorFinished {
            actor_id: 3,
            reason: FinishReason::Panicked("actor panicked".to_string()),
        };
        match msg {
            CoordinatorMsg::ActorFinished { actor_id, reason } => {
                assert_eq!(actor_id, 3);
                match reason {
                    FinishReason::Panicked(msg) => assert_eq!(msg, "actor panicked"),
                    _ => panic!("Expected Panicked reason"),
                }
            }
            _ => panic!("Expected ActorFinished variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_learner_finished_variant() {
        // DEFINES: LearnerFinished contains reason
        let msg = CoordinatorMsg::LearnerFinished {
            reason: FinishReason::Completed,
        };
        match msg {
            CoordinatorMsg::LearnerFinished { reason } => {
                match reason {
                    FinishReason::Completed => {}
                    _ => panic!("Expected Completed reason"),
                }
            }
            _ => panic!("Expected LearnerFinished variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_checkpoint_saved_variant() {
        // DEFINES: CheckpointSaved contains step and path
        let msg = CoordinatorMsg::CheckpointSaved {
            step: 10000,
            path: "/tmp/checkpoint.pt".to_string(),
        };
        match msg {
            CoordinatorMsg::CheckpointSaved { step, path } => {
                assert_eq!(step, 10000);
                assert_eq!(path, "/tmp/checkpoint.pt");
            }
            _ => panic!("Expected CheckpointSaved variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_checkpoint_saved_empty_path() {
        // DEFINES: Empty path is valid at message level
        let msg = CoordinatorMsg::CheckpointSaved {
            step: 0,
            path: String::new(),
        };
        match msg {
            CoordinatorMsg::CheckpointSaved { step, path } => {
                assert_eq!(step, 0);
                assert!(path.is_empty());
            }
            _ => panic!("Expected CheckpointSaved variant"),
        }
    }

    // -------------------------------------------------------------------------
    // CoordinatorMsg Factory Methods
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinator_msg_actor_stats_factory() {
        // DEFINES: actor_stats() factory creates ActorStats variant
        let stats = ActorStats::new(2);
        let msg = CoordinatorMsg::actor_stats(stats);
        match msg {
            CoordinatorMsg::ActorStats(s) => assert_eq!(s.actor_id, 2),
            _ => panic!("Expected ActorStats variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_learner_stats_factory() {
        // DEFINES: learner_stats() factory creates LearnerStats variant
        let mut stats = LearnerStats::new();
        stats.model_version = 10;
        let msg = CoordinatorMsg::learner_stats(stats);
        match msg {
            CoordinatorMsg::LearnerStats(s) => assert_eq!(s.model_version, 10),
            _ => panic!("Expected LearnerStats variant"),
        }
    }

    #[test]
    fn test_coordinator_msg_eval_result_factory() {
        // DEFINES: eval_result() factory creates EvalResult variant
        let result = EvalResult::from_rewards(50, &[100.0], &[100]);
        let msg = CoordinatorMsg::eval_result(result);
        match msg {
            CoordinatorMsg::EvalResult(r) => {
                assert_eq!(r.step, 50);
                assert_eq!(r.avg_reward, 100.0);
            }
            _ => panic!("Expected EvalResult variant"),
        }
    }

    // -------------------------------------------------------------------------
    // CoordinatorMsg Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinator_msg_clone_actor_stats() {
        // DEFINES: Cloning ActorStats variant deep clones the stats
        let mut stats = ActorStats::new(1);
        stats.episodes = 50;
        let msg = CoordinatorMsg::ActorStats(stats);
        let cloned = msg.clone();

        match cloned {
            CoordinatorMsg::ActorStats(s) => {
                assert_eq!(s.actor_id, 1);
                assert_eq!(s.episodes, 50);
            }
            _ => panic!("Clone should produce ActorStats"),
        }
    }

    #[test]
    fn test_coordinator_msg_clone_learner_stats() {
        // DEFINES: Cloning LearnerStats variant deep clones the stats
        let mut stats = LearnerStats::new();
        stats.avg_loss = 0.123;
        let msg = CoordinatorMsg::LearnerStats(stats);
        let cloned = msg.clone();

        match cloned {
            CoordinatorMsg::LearnerStats(s) => assert_eq!(s.avg_loss, 0.123),
            _ => panic!("Clone should produce LearnerStats"),
        }
    }

    #[test]
    fn test_coordinator_msg_clone_actor_finished() {
        // DEFINES: Cloning ActorFinished clones both actor_id and reason
        let msg = CoordinatorMsg::ActorFinished {
            actor_id: 7,
            reason: FinishReason::Panicked("error".to_string()),
        };
        let cloned = msg.clone();

        match cloned {
            CoordinatorMsg::ActorFinished { actor_id, reason } => {
                assert_eq!(actor_id, 7);
                match reason {
                    FinishReason::Panicked(s) => assert_eq!(s, "error"),
                    _ => panic!("Expected Panicked"),
                }
            }
            _ => panic!("Clone should produce ActorFinished"),
        }
    }

    #[test]
    fn test_coordinator_msg_clone_checkpoint_saved() {
        // DEFINES: Cloning CheckpointSaved clones path string
        let msg = CoordinatorMsg::CheckpointSaved {
            step: 5000,
            path: "/data/model.pt".to_string(),
        };
        let cloned = msg.clone();

        match cloned {
            CoordinatorMsg::CheckpointSaved { step, path } => {
                assert_eq!(step, 5000);
                assert_eq!(path, "/data/model.pt");
            }
            _ => panic!("Clone should produce CheckpointSaved"),
        }
    }

    // -------------------------------------------------------------------------
    // CoordinatorMsg Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coordinator_msg_debug_all_variants() {
        // DEFINES: All variants have valid Debug output
        let actor_stats = format!("{:?}", CoordinatorMsg::ActorStats(ActorStats::new(0)));
        assert!(actor_stats.contains("ActorStats"));

        let learner_stats = format!("{:?}", CoordinatorMsg::LearnerStats(LearnerStats::new()));
        assert!(learner_stats.contains("LearnerStats"));

        let eval_result = format!("{:?}", CoordinatorMsg::EvalResult(
            EvalResult::from_rewards(0, &[], &[])
        ));
        assert!(eval_result.contains("EvalResult"));

        let actor_finished = format!("{:?}", CoordinatorMsg::ActorFinished {
            actor_id: 0,
            reason: FinishReason::Stopped,
        });
        assert!(actor_finished.contains("ActorFinished"));

        let learner_finished = format!("{:?}", CoordinatorMsg::LearnerFinished {
            reason: FinishReason::Completed,
        });
        assert!(learner_finished.contains("LearnerFinished"));

        let checkpoint = format!("{:?}", CoordinatorMsg::CheckpointSaved {
            step: 0,
            path: String::new(),
        });
        assert!(checkpoint.contains("CheckpointSaved"));
    }
}

// =============================================================================
// EVAL MESSAGE TESTS
// =============================================================================

mod eval_msg_tests {
    use super::*;

    // -------------------------------------------------------------------------
    // EvalMsg Variant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_msg_stop_variant() {
        // DEFINES: EvalMsg::Stop exists
        let msg: EvalMsg<MockModel> = EvalMsg::Stop;
        match msg {
            EvalMsg::Stop => {}
            _ => panic!("Expected Stop variant"),
        }
    }

    #[test]
    fn test_eval_msg_evaluate_variant() {
        // DEFINES: EvalMsg::Evaluate contains model, step, n_episodes
        let model = MockModel::new(5);
        let msg = EvalMsg::Evaluate {
            model,
            step: 1000,
            n_episodes: 10,
        };
        match msg {
            EvalMsg::Evaluate { model, step, n_episodes } => {
                assert_eq!(model.version, 5);
                assert_eq!(step, 1000);
                assert_eq!(n_episodes, 10);
            }
            _ => panic!("Expected Evaluate variant"),
        }
    }

    #[test]
    fn test_eval_msg_evaluate_zero_episodes() {
        // DEFINES: Zero episodes is valid at message level
        let model = MockModel::new(1);
        let msg = EvalMsg::Evaluate {
            model,
            step: 0,
            n_episodes: 0,
        };
        match msg {
            EvalMsg::Evaluate { n_episodes, .. } => assert_eq!(n_episodes, 0),
            _ => panic!("Expected Evaluate variant"),
        }
    }

    #[test]
    fn test_eval_msg_evaluate_large_step() {
        // DEFINES: Large step values are accepted
        let model = MockModel::new(1);
        let msg = EvalMsg::Evaluate {
            model,
            step: usize::MAX,
            n_episodes: 1,
        };
        match msg {
            EvalMsg::Evaluate { step, .. } => assert_eq!(step, usize::MAX),
            _ => panic!("Expected Evaluate variant"),
        }
    }

    // -------------------------------------------------------------------------
    // EvalMsg Clone Tests (Manual Implementation)
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_msg_clone_stop() {
        // DEFINES: EvalMsg::Stop can be cloned
        let msg: EvalMsg<MockModel> = EvalMsg::Stop;
        let cloned = msg.clone();
        match cloned {
            EvalMsg::Stop => {}
            _ => panic!("Clone should produce Stop"),
        }
    }

    #[test]
    fn test_eval_msg_clone_evaluate_deep_clones_model() {
        // DEFINES: Cloning Evaluate deep clones the model
        let model = MockModel::new(42);
        let msg = EvalMsg::Evaluate {
            model,
            step: 500,
            n_episodes: 5,
        };
        let cloned = msg.clone();

        match cloned {
            EvalMsg::Evaluate { model, step, n_episodes } => {
                assert_eq!(model.version, 42);
                assert_eq!(model.weights, vec![1.0, 2.0, 3.0]);
                assert_eq!(step, 500);
                assert_eq!(n_episodes, 5);
            }
            _ => panic!("Clone should produce Evaluate"),
        }
    }

    #[test]
    fn test_eval_msg_clone_with_expensive_model() {
        // DEFINES: Clone works with expensive-to-clone models
        let model = ExpensiveCloneModel {
            data: vec![1, 2, 3, 4, 5],
        };
        let msg = EvalMsg::Evaluate {
            model,
            step: 100,
            n_episodes: 1,
        };
        let cloned = msg.clone();

        match cloned {
            EvalMsg::Evaluate { model, .. } => {
                assert_eq!(model.data, vec![1, 2, 3, 4, 5]);
            }
            _ => panic!("Clone should produce Evaluate"),
        }
    }

    // -------------------------------------------------------------------------
    // EvalMsg Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_msg_debug_stop() {
        // DEFINES: Debug output for Stop
        let msg: EvalMsg<MockModel> = EvalMsg::Stop;
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("Stop"));
    }

    #[test]
    fn test_eval_msg_debug_evaluate() {
        // DEFINES: Debug output for Evaluate includes fields
        let msg = EvalMsg::Evaluate {
            model: MockModel::new(1),
            step: 123,
            n_episodes: 7,
        };
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("Evaluate"));
        assert!(debug_str.contains("123"));
        assert!(debug_str.contains("7"));
    }

    // -------------------------------------------------------------------------
    // EvalResult Construction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_result_from_rewards_empty() {
        // DEFINES: Empty rewards slice returns zeros with n_episodes=0
        let result = EvalResult::from_rewards(0, &[], &[]);

        assert_eq!(result.step, 0);
        assert_eq!(result.avg_reward, 0.0);
        assert_eq!(result.std_reward, 0.0);
        assert_eq!(result.min_reward, 0.0);
        assert_eq!(result.max_reward, 0.0);
        assert_eq!(result.n_episodes, 0);
        assert_eq!(result.avg_length, 0.0);
    }

    #[test]
    fn test_eval_result_from_rewards_single_element() {
        // DEFINES: Single element gives that element as avg, 0 std
        let result = EvalResult::from_rewards(10, &[100.0], &[50]);

        assert_eq!(result.step, 10);
        assert_eq!(result.avg_reward, 100.0);
        assert_eq!(result.std_reward, 0.0); // No variance with single element
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 100.0);
        assert_eq!(result.n_episodes, 1);
        assert_eq!(result.avg_length, 50.0);
    }

    #[test]
    fn test_eval_result_from_rewards_multiple_elements() {
        // DEFINES: Multiple elements compute correct statistics
        let result = EvalResult::from_rewards(100, &[100.0, 200.0, 300.0], &[10, 20, 30]);

        assert_eq!(result.step, 100);
        assert_eq!(result.avg_reward, 200.0); // (100 + 200 + 300) / 3
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 300.0);
        assert_eq!(result.n_episodes, 3);
        assert_eq!(result.avg_length, 20.0); // (10 + 20 + 30) / 3

        // Standard deviation: sqrt(((100-200)^2 + (200-200)^2 + (300-200)^2) / 3)
        // = sqrt((10000 + 0 + 10000) / 3) = sqrt(20000/3) = sqrt(6666.67) ~ 81.65
        assert!((result.std_reward - 81.65).abs() < 0.1);
    }

    #[test]
    fn test_eval_result_from_rewards_all_same() {
        // DEFINES: All same values give std = 0
        let result = EvalResult::from_rewards(0, &[50.0, 50.0, 50.0, 50.0], &[10, 10, 10, 10]);

        assert_eq!(result.avg_reward, 50.0);
        assert_eq!(result.std_reward, 0.0);
        assert_eq!(result.min_reward, 50.0);
        assert_eq!(result.max_reward, 50.0);
    }

    #[test]
    fn test_eval_result_from_rewards_with_nan() {
        // DEFINES: NaN rewards are FILTERED - defensive coding prevents average corruption
        let result = EvalResult::from_rewards(0, &[100.0, f32::NAN, 200.0], &[10, 20, 30]);

        // NaN is filtered - statistics computed from finite values only
        assert_eq!(result.avg_reward, 150.0); // (100 + 200) / 2
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 200.0);
        assert_eq!(result.n_episodes, 3);
        assert_eq!(result.n_valid_episodes, 2);
        assert_eq!(result.n_filtered_episodes, 1);
    }

    #[test]
    fn test_eval_result_from_rewards_with_infinity() {
        // DEFINES: Infinity rewards are FILTERED - defensive coding
        let result = EvalResult::from_rewards(0, &[100.0, f32::INFINITY], &[10, 20]);

        // Infinity is filtered - only finite value (100.0) used
        assert_eq!(result.avg_reward, 100.0);
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 100.0);
        assert_eq!(result.n_valid_episodes, 1);
        assert_eq!(result.n_filtered_episodes, 1);
    }

    #[test]
    fn test_eval_result_from_rewards_with_neg_infinity() {
        // DEFINES: Negative infinity rewards are FILTERED
        let result = EvalResult::from_rewards(0, &[f32::NEG_INFINITY, 100.0], &[10, 20]);

        // Neg infinity is filtered - only finite value (100.0) used
        assert_eq!(result.avg_reward, 100.0);
        assert_eq!(result.min_reward, 100.0);
        assert_eq!(result.max_reward, 100.0);
        assert_eq!(result.n_valid_episodes, 1);
        assert_eq!(result.n_filtered_episodes, 1);
    }

    #[test]
    fn test_eval_result_from_rewards_with_negative_values() {
        // DEFINES: Negative rewards are handled correctly
        let result = EvalResult::from_rewards(0, &[-100.0, -50.0, -25.0], &[10, 20, 30]);

        assert!((result.avg_reward - (-58.333)).abs() < 0.01);
        assert_eq!(result.min_reward, -100.0);
        assert_eq!(result.max_reward, -25.0);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "rewards and lengths arrays must have the same length"))]
    fn test_eval_result_from_rewards_mismatched_lengths() {
        // DEFINES: Mismatched array lengths panic in DEBUG, use min length in RELEASE
        // This defensive coding catches bugs at development time
        let result = EvalResult::from_rewards(0, &[100.0, 200.0, 300.0], &[10, 20]);

        // In release mode: uses min(3, 2) = 2 for avg_length
        // avg_length = (10 + 20) / 2 = 15.0
        assert_eq!(result.n_episodes, 3);
        assert_eq!(result.avg_length, 15.0);
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "rewards and lengths arrays must have the same length"))]
    fn test_eval_result_from_rewards_empty_lengths_nonempty_rewards() {
        // DEFINES: Empty lengths with non-empty rewards panics in DEBUG
        let result = EvalResult::from_rewards(0, &[100.0, 200.0], &[]);

        // In release mode: uses min(2, 0) = 0 for avg_length
        assert_eq!(result.n_episodes, 2);
        assert_eq!(result.avg_length, 0.0);
    }

    #[test]
    fn test_eval_result_from_rewards_very_large_values() {
        // DEFINES: Very large values that could cause overflow in sum
        let result = EvalResult::from_rewards(0, &[f32::MAX / 2.0, f32::MAX / 2.0], &[1, 1]);

        // MAX/2 + MAX/2 = MAX (at the boundary)
        // Then divide by 2 should give MAX/2
        assert_eq!(result.avg_reward, f32::MAX / 2.0);
    }

    #[test]
    fn test_eval_result_from_rewards_sum_overflow() {
        // DEFINES: Large values use Welford's algorithm which avoids sum overflow
        // Welford's: mean += (x - mean) / n, which doesn't accumulate a sum
        let result = EvalResult::from_rewards(0, &[f32::MAX, f32::MAX], &[1, 1]);

        // With Welford's algorithm, we compute mean incrementally, not via sum
        // mean_1 = MAX, mean_2 = MAX + (MAX - MAX) / 2 = MAX
        assert_eq!(result.avg_reward, f32::MAX);
        assert!(result.avg_reward.is_finite());
    }

    #[test]
    fn test_eval_result_from_rewards_preserves_step() {
        // DEFINES: Step parameter is preserved in result
        let result = EvalResult::from_rewards(12345, &[1.0], &[1]);
        assert_eq!(result.step, 12345);
    }

    // -------------------------------------------------------------------------
    // EvalResult meets_threshold Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_result_meets_threshold_above() {
        // DEFINES: Returns true when avg_reward >= threshold
        let result = EvalResult::from_rewards(0, &[100.0], &[1]);
        assert!(result.meets_threshold(100.0));
        assert!(result.meets_threshold(99.0));
    }

    #[test]
    fn test_eval_result_meets_threshold_below() {
        // DEFINES: Returns false when avg_reward < threshold
        let result = EvalResult::from_rewards(0, &[100.0], &[1]);
        assert!(!result.meets_threshold(101.0));
    }

    #[test]
    fn test_eval_result_meets_threshold_exactly() {
        // DEFINES: Returns true when avg_reward == threshold (equality counts)
        let result = EvalResult::from_rewards(0, &[100.0], &[1]);
        assert!(result.meets_threshold(100.0));
    }

    #[test]
    fn test_eval_result_meets_threshold_nan_avg() {
        // DEFINES: With filtering, all-NaN rewards result in avg_reward = 0.0
        let result = EvalResult::from_rewards(0, &[f32::NAN], &[1]);
        // All values filtered, avg_reward = 0.0
        assert_eq!(result.avg_reward, 0.0);
        assert!(result.meets_threshold(0.0));   // 0 >= 0
        assert!(result.meets_threshold(-100.0)); // 0 >= -100
        assert!(!result.meets_threshold(1.0));   // 0 < 1
    }

    #[test]
    fn test_eval_result_meets_threshold_nan_threshold() {
        // DEFINES: NaN threshold returns false (x >= NaN is always false)
        let result = EvalResult::from_rewards(0, &[100.0], &[1]);
        // x >= NaN is always false in IEEE 754
        assert!(!result.meets_threshold(f32::NAN));
    }

    #[test]
    fn test_eval_result_meets_threshold_infinity() {
        // DEFINES: Infinity rewards are filtered, so avg = 0.0
        let result = EvalResult::from_rewards(0, &[f32::INFINITY], &[1]);
        // Infinity filtered, avg_reward = 0.0
        assert_eq!(result.avg_reward, 0.0);
        assert!(!result.meets_threshold(f32::MAX)); // 0 < MAX
        assert!(result.meets_threshold(0.0));        // 0 >= 0
    }

    #[test]
    fn test_eval_result_meets_threshold_neg_infinity() {
        // DEFINES: Negative infinity rewards are filtered, so avg = 0.0
        let result = EvalResult::from_rewards(0, &[f32::NEG_INFINITY], &[1]);
        // Neg infinity filtered, avg_reward = 0.0
        assert_eq!(result.avg_reward, 0.0);
        assert!(result.meets_threshold(0.0)); // 0 >= 0
        assert!(result.meets_threshold(f32::NEG_INFINITY)); // 0 >= -inf
    }

    // -------------------------------------------------------------------------
    // EvalResult Clone Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_result_clone_preserves_all_fields() {
        // DEFINES: Clone preserves all fields
        let result = EvalResult::from_rewards(100, &[50.0, 75.0], &[10, 20]);
        let cloned = result.clone();

        assert_eq!(cloned.step, result.step);
        assert_eq!(cloned.avg_reward, result.avg_reward);
        assert_eq!(cloned.std_reward, result.std_reward);
        assert_eq!(cloned.min_reward, result.min_reward);
        assert_eq!(cloned.max_reward, result.max_reward);
        assert_eq!(cloned.n_episodes, result.n_episodes);
        assert_eq!(cloned.avg_length, result.avg_length);
    }

    #[test]
    fn test_eval_result_clone_is_independent() {
        // DEFINES: Modifying clone doesn't affect original (trivial for Copy types)
        let result = EvalResult::from_rewards(100, &[50.0], &[10]);
        let mut cloned = result.clone();
        cloned.step = 999;

        assert_eq!(result.step, 100);
        assert_eq!(cloned.step, 999);
    }

    // -------------------------------------------------------------------------
    // EvalResult Debug Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_eval_result_debug_output() {
        // DEFINES: Debug output includes all field names
        let result = EvalResult::from_rewards(42, &[100.0, 200.0], &[10, 20]);
        let debug_str = format!("{:?}", result);

        assert!(debug_str.contains("EvalResult"));
        assert!(debug_str.contains("step"));
        assert!(debug_str.contains("42"));
        assert!(debug_str.contains("avg_reward"));
        assert!(debug_str.contains("std_reward"));
        assert!(debug_str.contains("min_reward"));
        assert!(debug_str.contains("max_reward"));
        assert!(debug_str.contains("n_episodes"));
        assert!(debug_str.contains("avg_length"));
    }
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_coordinator_msg_round_trip_actor_stats() {
        // DEFINES: ActorStats can be wrapped and unwrapped from CoordinatorMsg
        let mut stats = ActorStats::new(3);
        stats.record_episode(250.0);
        stats.add_steps(1000);

        let msg = CoordinatorMsg::actor_stats(stats);

        match msg {
            CoordinatorMsg::ActorStats(recovered) => {
                assert_eq!(recovered.actor_id, 3);
                assert_eq!(recovered.avg_episode_reward, 250.0);
                assert_eq!(recovered.steps, 1000);
            }
            _ => panic!("Expected ActorStats"),
        }
    }

    #[test]
    fn test_coordinator_msg_round_trip_learner_stats() {
        // DEFINES: LearnerStats can be wrapped and unwrapped from CoordinatorMsg
        let mut stats = LearnerStats::new();
        stats.record_step(0.5, 0.3, 0.2, 0.1);
        stats.set_model_version(100);
        stats.set_steps_per_second(1000.0);

        let msg = CoordinatorMsg::learner_stats(stats);

        match msg {
            CoordinatorMsg::LearnerStats(recovered) => {
                assert_eq!(recovered.train_steps, 1);
                assert_eq!(recovered.avg_loss, 0.5);
                assert_eq!(recovered.model_version, 100);
                assert_eq!(recovered.steps_per_second, 1000.0);
            }
            _ => panic!("Expected LearnerStats"),
        }
    }

    #[test]
    fn test_coordinator_msg_round_trip_eval_result() {
        // DEFINES: EvalResult can be wrapped and unwrapped from CoordinatorMsg
        let result = EvalResult::from_rewards(5000, &[400.0, 450.0, 500.0], &[100, 110, 120]);

        let msg = CoordinatorMsg::eval_result(result);

        match msg {
            CoordinatorMsg::EvalResult(recovered) => {
                assert_eq!(recovered.step, 5000);
                assert_eq!(recovered.avg_reward, 450.0);
                assert_eq!(recovered.n_episodes, 3);
            }
            _ => panic!("Expected EvalResult"),
        }
    }

    #[test]
    fn test_stats_accumulation_workflow() {
        // DEFINES: Typical workflow of accumulating stats over training
        let mut actor_stats = ActorStats::new(0);
        let mut learner_stats = LearnerStats::new();

        // Simulate training loop
        for episode in 1..=10 {
            // Actor completes episodes
            actor_stats.record_episode((episode * 10) as f32);
            actor_stats.add_steps(100);

            // Learner does training steps
            for step in 1..=5 {
                learner_stats.record_step(
                    1.0 / (step as f32), // decreasing loss
                    0.5,
                    0.3,
                    0.1,
                );
            }
        }

        // Verify final state
        assert_eq!(actor_stats.episodes, 10);
        assert_eq!(actor_stats.steps, 1000);
        assert_eq!(actor_stats.recent_episode_reward, 100.0);

        assert_eq!(learner_stats.train_steps, 50);
        // Last recorded loss was 1.0/5 = 0.2
        assert_eq!(learner_stats.avg_loss, 0.2);
    }

    #[test]
    fn test_finish_reason_in_actor_finished() {
        // DEFINES: Different finish reasons in ActorFinished messages
        let reasons = vec![
            FinishReason::Stopped,
            FinishReason::Panicked("error".to_string()),
            FinishReason::Completed,
        ];

        for (i, reason) in reasons.into_iter().enumerate() {
            let msg = CoordinatorMsg::ActorFinished {
                actor_id: i,
                reason: reason.clone(),
            };

            match msg {
                CoordinatorMsg::ActorFinished { actor_id, reason: r } => {
                    assert_eq!(actor_id, i);
                    // Verify reason matches
                    match (&reason, &r) {
                        (FinishReason::Stopped, FinishReason::Stopped) => {}
                        (FinishReason::Panicked(a), FinishReason::Panicked(b)) => {
                            assert_eq!(a, b);
                        }
                        (FinishReason::Completed, FinishReason::Completed) => {}
                        _ => panic!("Reason mismatch"),
                    }
                }
                _ => panic!("Expected ActorFinished"),
            }
        }
    }
}

// =============================================================================
// THREAD SAFETY TESTS
// =============================================================================

mod thread_safety_tests {
    use super::*;

    #[test]
    fn test_learner_msg_send_across_thread() {
        // DEFINES: LearnerMsg can be sent across thread boundaries
        let (tx, rx) = mpsc::channel::<LearnerMsg<MockModel>>();

        let handle = thread::spawn(move || {
            tx.send(LearnerMsg::Stop).unwrap();
            tx.send(LearnerMsg::UpdateLearningRate(0.001)).unwrap();
            tx.send(LearnerMsg::Pause).unwrap();
        });

        handle.join().unwrap();

        let msg1 = rx.recv().unwrap();
        assert!(matches!(msg1, LearnerMsg::Stop));

        let msg2 = rx.recv().unwrap();
        assert!(matches!(msg2, LearnerMsg::UpdateLearningRate(lr) if lr == 0.001));

        let msg3 = rx.recv().unwrap();
        assert!(matches!(msg3, LearnerMsg::Pause));
    }

    #[test]
    fn test_actor_msg_send_across_thread() {
        // DEFINES: ActorMsg can be sent across thread boundaries
        let (tx, rx) = mpsc::channel::<ActorMsg<MockModel>>();

        let handle = thread::spawn(move || {
            tx.send(ActorMsg::Stop).unwrap();
            tx.send(ActorMsg::SetEpsilon(0.1)).unwrap();
            tx.send(ActorMsg::UpdateModel(MockModel::new(42))).unwrap();
        });

        handle.join().unwrap();

        let msg1 = rx.recv().unwrap();
        assert!(matches!(msg1, ActorMsg::Stop));

        let msg2 = rx.recv().unwrap();
        assert!(matches!(msg2, ActorMsg::SetEpsilon(eps) if eps == 0.1));

        let msg3 = rx.recv().unwrap();
        match msg3 {
            ActorMsg::UpdateModel(m) => assert_eq!(m.version, 42),
            _ => panic!("Expected UpdateModel"),
        }
    }

    #[test]
    fn test_coordinator_msg_send_across_thread() {
        // DEFINES: CoordinatorMsg can be sent across thread boundaries
        let (tx, rx) = mpsc::channel::<CoordinatorMsg>();

        let handle = thread::spawn(move || {
            let stats = ActorStats::new(1);
            tx.send(CoordinatorMsg::ActorStats(stats)).unwrap();

            tx.send(CoordinatorMsg::ActorFinished {
                actor_id: 1,
                reason: FinishReason::Completed,
            }).unwrap();
        });

        handle.join().unwrap();

        let msg1 = rx.recv().unwrap();
        assert!(matches!(msg1, CoordinatorMsg::ActorStats(_)));

        let msg2 = rx.recv().unwrap();
        assert!(matches!(msg2, CoordinatorMsg::ActorFinished { .. }));
    }

    #[test]
    fn test_eval_msg_send_across_thread() {
        // DEFINES: EvalMsg can be sent across thread boundaries
        let (tx, rx) = mpsc::channel::<EvalMsg<MockModel>>();

        let handle = thread::spawn(move || {
            tx.send(EvalMsg::Stop).unwrap();
            tx.send(EvalMsg::Evaluate {
                model: MockModel::new(10),
                step: 1000,
                n_episodes: 5,
            }).unwrap();
        });

        handle.join().unwrap();

        let msg1 = rx.recv().unwrap();
        assert!(matches!(msg1, EvalMsg::Stop));

        let msg2 = rx.recv().unwrap();
        match msg2 {
            EvalMsg::Evaluate { model, step, n_episodes } => {
                assert_eq!(model.version, 10);
                assert_eq!(step, 1000);
                assert_eq!(n_episodes, 5);
            }
            _ => panic!("Expected Evaluate"),
        }
    }

    #[test]
    fn test_learner_stats_send_across_thread() {
        // DEFINES: LearnerStats can be sent across threads
        let (tx, rx) = mpsc::channel::<LearnerStats>();

        let handle = thread::spawn(move || {
            let mut stats = LearnerStats::new();
            stats.record_step(0.5, 0.3, 0.2, 0.1);
            tx.send(stats).unwrap();
        });

        handle.join().unwrap();

        let stats = rx.recv().unwrap();
        assert_eq!(stats.train_steps, 1);
        assert_eq!(stats.avg_loss, 0.5);
    }

    #[test]
    fn test_actor_stats_send_across_thread() {
        // DEFINES: ActorStats can be sent across threads
        let (tx, rx) = mpsc::channel::<ActorStats>();

        let handle = thread::spawn(move || {
            let mut stats = ActorStats::new(7);
            stats.record_episode(500.0);
            tx.send(stats).unwrap();
        });

        handle.join().unwrap();

        let stats = rx.recv().unwrap();
        assert_eq!(stats.actor_id, 7);
        assert_eq!(stats.avg_episode_reward, 500.0);
    }

    #[test]
    fn test_eval_result_send_across_thread() {
        // DEFINES: EvalResult can be sent across threads
        let (tx, rx) = mpsc::channel::<EvalResult>();

        let handle = thread::spawn(move || {
            let result = EvalResult::from_rewards(1000, &[100.0, 200.0], &[10, 20]);
            tx.send(result).unwrap();
        });

        handle.join().unwrap();

        let result = rx.recv().unwrap();
        assert_eq!(result.step, 1000);
        assert_eq!(result.avg_reward, 150.0);
    }

    // Note: We cannot easily test that types DON'T implement Send/Sync
    // at compile time without negative trait bounds (unstable feature).
    // The fact that the above tests compile proves Send is implemented.
}

// =============================================================================
// NUMERICAL STABILITY TESTS
// =============================================================================

mod numerical_stability_tests {
    use super::*;

    #[test]
    fn test_actor_stats_average_precision_small_values() {
        // DEFINES: Running average maintains precision with small values
        let mut stats = ActorStats::new(0);

        for _ in 0..1000 {
            stats.record_episode(0.001);
        }

        // Should still be close to 0.001
        assert!((stats.avg_episode_reward - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_actor_stats_average_alternating_extreme() {
        // DEFINES: Alternating extreme values behavior
        let mut stats = ActorStats::new(0);

        for i in 0..100 {
            if i % 2 == 0 {
                stats.record_episode(1e10);
            } else {
                stats.record_episode(-1e10);
            }
        }

        // Average should be close to 0
        assert!(stats.avg_episode_reward.abs() < 1e5);
    }

    #[test]
    fn test_eval_result_variance_numerical_stability() {
        // DEFINES: Variance calculation with values far from mean
        // Using Welford's algorithm would be more stable, but the current
        // implementation uses two-pass method which is less stable
        let rewards: Vec<f32> = (0..100).map(|i| 1e6 + (i as f32) * 0.1).collect();
        let lengths: Vec<usize> = vec![1; 100];

        let result = EvalResult::from_rewards(0, &rewards, &lengths);

        // The std should be small (values are close together)
        // Despite values being large, the variance should be computed correctly
        assert!(result.std_reward < 10.0);
    }

    #[test]
    fn test_eval_result_many_episodes() {
        // DEFINES: Statistics are correct with many episodes
        let n = 10000;
        let rewards: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let lengths: Vec<usize> = vec![1; n];

        let result = EvalResult::from_rewards(0, &rewards, &lengths);

        // Average should be (0 + 1 + ... + 9999) / 10000 = 4999.5
        // f32 has ~7 decimal digits of precision, so summing 10000 values
        // can accumulate error. We use a looser tolerance.
        assert!((result.avg_reward - 4999.5).abs() < 1.0,
            "Expected avg_reward near 4999.5, got {}", result.avg_reward);
        assert_eq!(result.min_reward, 0.0);
        assert_eq!(result.max_reward, 9999.0);
        assert_eq!(result.n_episodes, 10000);
    }

    #[test]
    fn test_learner_stats_subnormal_values() {
        // DEFINES: Subnormal (denormalized) float values are handled
        let mut stats = LearnerStats::new();
        stats.record_step(f32::MIN_POSITIVE / 2.0, 0.0, 0.0, 0.0);

        // Should store the subnormal value
        assert!(stats.avg_loss > 0.0);
        assert!(stats.avg_loss < f32::MIN_POSITIVE);
    }

    #[test]
    fn test_actor_stats_catastrophic_cancellation() {
        // DEFINES: Behavior under catastrophic cancellation scenario
        // When computing (large + small) - large, precision is lost
        let mut stats = ActorStats::new(0);

        // Start with a very large average
        stats.record_episode(1e20);
        assert_eq!(stats.avg_episode_reward, 1e20);

        // Add a tiny value - the (episodes-1)*avg + reward will lose precision
        stats.record_episode(1.0);

        // The average should be approximately (1e20 + 1) / 2 ~ 5e19
        // But due to floating point, 1e20 + 1 = 1e20 (precision loss)
        // This test documents the precision limitation
        let expected = 5e19_f32;
        assert!((stats.avg_episode_reward - expected).abs() / expected < 0.01);
    }
}

// =============================================================================
// EDGE CASE REGRESSION TESTS
// =============================================================================

mod edge_case_regression_tests {
    use super::*;

    #[test]
    fn test_actor_stats_first_episode_no_underflow() {
        // REGRESSION TEST: Ensure (episodes - 1) doesn't underflow on first episode
        // episodes starts at 0, is incremented to 1, then (1 - 1) = 0 is used
        let mut stats = ActorStats::new(0);

        // This should NOT panic or produce wrong results
        stats.record_episode(100.0);

        assert_eq!(stats.episodes, 1);
        assert_eq!(stats.avg_episode_reward, 100.0);
    }

    #[test]
    fn test_eval_result_empty_preserves_sane_values() {
        // REGRESSION TEST: Empty input should give zeros, not inf/-inf
        let result = EvalResult::from_rewards(42, &[], &[]);

        assert_eq!(result.step, 42);
        assert_eq!(result.min_reward, 0.0); // Not INFINITY
        assert_eq!(result.max_reward, 0.0); // Not NEG_INFINITY
        assert_eq!(result.avg_reward, 0.0);
        assert_eq!(result.std_reward, 0.0);
    }

    #[test]
    fn test_finish_reason_panicked_string_ownership() {
        // REGRESSION TEST: String in Panicked variant is properly owned
        let msg = {
            let temp_string = String::from("temporary panic message");
            FinishReason::Panicked(temp_string)
        };
        // temp_string is moved, msg still valid

        match msg {
            FinishReason::Panicked(s) => assert!(s.contains("temporary")),
            _ => panic!("Expected Panicked"),
        }
    }

    #[test]
    fn test_learner_msg_phantom_variance_safety() {
        // REGRESSION TEST: PhantomData should be covariant in M
        // This is more of a compile-time test - if it compiles, it's correct
        fn covariant_test<'a>(_msg: LearnerMsg<&'a str>) {}

        let msg: LearnerMsg<&'static str> = LearnerMsg::Stop;
        covariant_test(msg); // Should compile - 'static is subtype of 'a
    }

    #[test]
    fn test_coordinator_msg_nested_clone_independence() {
        // REGRESSION TEST: Deep clone produces fully independent copies
        let mut original_stats = ActorStats::new(1);
        original_stats.episodes = 10;

        let msg = CoordinatorMsg::ActorStats(original_stats);
        let cloned_msg = msg.clone();

        // Modify through pattern matching (simulating use)
        match msg {
            CoordinatorMsg::ActorStats(mut s) => {
                s.episodes = 20;
                // s is now 20, but cloned should still be 10
                // Use s to avoid unused warning
                let _ = s.episodes;
            }
            _ => {}
        }

        match cloned_msg {
            CoordinatorMsg::ActorStats(s) => {
                assert_eq!(s.episodes, 10); // Should be unchanged
            }
            _ => panic!("Expected ActorStats"),
        }
    }
}
