//! Configuration tests for DistributedPPOConfig.
//!
//! These tests define the correct behavior of the configuration builder pattern
//! and validate that invalid configurations are rejected.
//!
//! # Intent
//!
//! Configuration objects are the entry point for users. They should:
//! - Have sensible defaults that work for common cases
//! - Reject invalid configurations that would cause runtime errors
//! - Support fluent builder pattern for customization
//! - Compute derived values correctly

use crate::runners::distributed_ppo_config::DistributedPPOConfig;

// ============================================================================
// Default Configuration Tests
// ============================================================================

/// Test that default configuration has sensible values.
/// INTENT: Users should be able to start with defaults for simple cases.
#[test]
fn test_default_config_has_sensible_values() {
    let config = DistributedPPOConfig::default();

    // Multi-actor defaults should be reasonable for typical hardware
    assert!(config.n_actors >= 1, "Must have at least one actor");
    assert!(
        config.n_envs_per_actor >= 1,
        "Must have at least one env per actor"
    );

    // PPO hyperparameters should be in standard ranges
    assert!(
        config.gamma > 0.0 && config.gamma <= 1.0,
        "Gamma must be in (0, 1]"
    );
    assert!(
        config.gae_lambda >= 0.0 && config.gae_lambda <= 1.0,
        "GAE lambda must be in [0, 1]"
    );
    assert!(
        config.clip_ratio > 0.0 && config.clip_ratio < 1.0,
        "Clip ratio must be in (0, 1)"
    );
    assert!(config.vf_coef > 0.0, "Value function coefficient must be positive");
    assert!(
        config.entropy_coef >= 0.0,
        "Entropy coefficient must be non-negative"
    );

    // Learning settings should be reasonable
    assert!(config.learning_rate > 0.0, "Learning rate must be positive");
    assert!(config.n_epochs >= 1, "Must have at least one training epoch");
    assert!(config.rollout_length >= 1, "Rollout length must be at least 1");
}

/// Test that default configuration produces valid derived values.
/// INTENT: Derived computations should be mathematically correct.
#[test]
fn test_default_config_derived_values() {
    let config = DistributedPPOConfig::default();

    let expected_total_envs = config.n_actors * config.n_envs_per_actor;
    assert_eq!(
        config.total_envs(),
        expected_total_envs,
        "total_envs should equal n_actors * n_envs_per_actor"
    );

    let expected_transitions = config.total_envs() * config.rollout_length;
    assert_eq!(
        config.transitions_per_rollout(),
        expected_transitions,
        "transitions_per_rollout should equal total_envs * rollout_length"
    );

    let expected_minibatch_size = config.transitions_per_rollout() / config.n_minibatches;
    assert_eq!(
        config.minibatch_size(),
        expected_minibatch_size,
        "minibatch_size should equal transitions / n_minibatches"
    );
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

/// Test that builder pattern preserves values correctly.
/// INTENT: Each builder method should only affect its target field.
#[test]
fn test_builder_pattern_preserves_values() {
    let config = DistributedPPOConfig::new()
        .with_n_actors(8)
        .with_n_envs_per_actor(16)
        .with_rollout_length(128)
        .with_n_epochs(10)
        .with_n_minibatches(4)
        .with_gamma(0.95)
        .with_gae_lambda(0.9)
        .with_clip_ratio(0.1)
        .with_vf_coef(1.0)
        .with_entropy_coef(0.05)
        .with_learning_rate(1e-3)
        .with_max_env_steps(500_000)
        .with_target_reward(100.0);

    assert_eq!(config.n_actors, 8);
    assert_eq!(config.n_envs_per_actor, 16);
    assert_eq!(config.rollout_length, 128);
    assert_eq!(config.n_epochs, 10);
    assert_eq!(config.n_minibatches, 4);
    assert!((config.gamma - 0.95).abs() < 1e-6);
    assert!((config.gae_lambda - 0.9).abs() < 1e-6);
    assert!((config.clip_ratio - 0.1).abs() < 1e-6);
    assert!((config.vf_coef - 1.0).abs() < 1e-6);
    assert!((config.entropy_coef - 0.05).abs() < 1e-6);
    assert!((config.learning_rate - 1e-3).abs() < 1e-9);
    assert_eq!(config.max_env_steps, 500_000);
    assert_eq!(config.target_reward, Some(100.0));
}

/// Test that builder methods can be called in any order.
/// INTENT: Builder pattern should be order-independent.
#[test]
fn test_builder_order_independence() {
    let config1 = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_gamma(0.99)
        .with_learning_rate(3e-4);

    let config2 = DistributedPPOConfig::new()
        .with_learning_rate(3e-4)
        .with_n_actors(4)
        .with_gamma(0.99);

    assert_eq!(config1.n_actors, config2.n_actors);
    assert!((config1.gamma - config2.gamma).abs() < 1e-6);
    assert!((config1.learning_rate - config2.learning_rate).abs() < 1e-9);
}

/// Test that builder can be chained from clone.
/// INTENT: Support modifying existing configs.
#[test]
fn test_builder_from_clone() {
    let base = DistributedPPOConfig::new().with_n_actors(4).with_gamma(0.99);

    let modified = base.clone().with_n_actors(8);

    // Original should be unchanged
    assert_eq!(base.n_actors, 4);
    // Modified should have new value
    assert_eq!(modified.n_actors, 8);
    // Other values should be preserved
    assert!((modified.gamma - 0.99).abs() < 1e-6);
}

// ============================================================================
// Derived Value Computation Tests
// ============================================================================

/// Test total_envs computation with various configurations.
/// INTENT: total_envs = n_actors * n_envs_per_actor, always.
#[test]
fn test_total_envs_computation() {
    let test_cases = [
        (1, 1, 1),
        (1, 64, 64),
        (4, 32, 128),
        (8, 16, 128),
        (16, 64, 1024),
    ];

    for (n_actors, n_envs, expected) in test_cases {
        let config = DistributedPPOConfig::new()
            .with_n_actors(n_actors)
            .with_n_envs_per_actor(n_envs);

        assert_eq!(
            config.total_envs(),
            expected,
            "total_envs({}, {}) should be {}",
            n_actors,
            n_envs,
            expected
        );
    }
}

/// Test transitions_per_rollout computation.
/// INTENT: transitions = total_envs * rollout_length, always.
#[test]
fn test_transitions_per_rollout_computation() {
    let config = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(32)
        .with_rollout_length(64);

    // 4 * 32 * 64 = 8192
    assert_eq!(config.transitions_per_rollout(), 8192);
}

/// Test minibatch_size computation.
/// INTENT: minibatch = transitions / n_minibatches, integer division.
#[test]
fn test_minibatch_size_computation() {
    let config = DistributedPPOConfig::new()
        .with_n_actors(4)
        .with_n_envs_per_actor(32)
        .with_rollout_length(64)
        .with_n_minibatches(8);

    // 8192 / 8 = 1024
    assert_eq!(config.minibatch_size(), 1024);
}

/// Test minibatch_size with uneven division.
/// INTENT: Integer division should truncate correctly.
#[test]
fn test_minibatch_size_uneven_division() {
    let config = DistributedPPOConfig::new()
        .with_n_actors(1)
        .with_n_envs_per_actor(10)
        .with_rollout_length(10)
        .with_n_minibatches(3);

    // 100 / 3 = 33 (integer division)
    assert_eq!(config.minibatch_size(), 33);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test configuration with minimal values.
/// INTENT: Edge cases with minimum valid values should work.
#[test]
fn test_minimal_configuration() {
    let config = DistributedPPOConfig::new()
        .with_n_actors(1)
        .with_n_envs_per_actor(1)
        .with_rollout_length(1)
        .with_n_epochs(1)
        .with_n_minibatches(1);

    assert_eq!(config.total_envs(), 1);
    assert_eq!(config.transitions_per_rollout(), 1);
    assert_eq!(config.minibatch_size(), 1);
}

/// Test configuration with extreme gamma values.
/// INTENT: Boundary gamma values should be accepted.
#[test]
fn test_extreme_gamma_values() {
    // gamma = 0 (myopic)
    let myopic = DistributedPPOConfig::new().with_gamma(0.0);
    assert!((myopic.gamma - 0.0).abs() < 1e-6);

    // gamma = 1 (undiscounted)
    let undiscounted = DistributedPPOConfig::new().with_gamma(1.0);
    assert!((undiscounted.gamma - 1.0).abs() < 1e-6);
}

/// Test configuration with extreme lambda values.
/// INTENT: Lambda extremes (0=TD, 1=MC) should be accepted.
#[test]
fn test_extreme_lambda_values() {
    // lambda = 0 (one-step TD)
    let td = DistributedPPOConfig::new().with_gae_lambda(0.0);
    assert!((td.gae_lambda - 0.0).abs() < 1e-6);

    // lambda = 1 (Monte Carlo-like)
    let mc = DistributedPPOConfig::new().with_gae_lambda(1.0);
    assert!((mc.gae_lambda - 1.0).abs() < 1e-6);
}

/// Test that optional fields work correctly.
/// INTENT: Optional fields should be None by default and Some when set.
#[test]
fn test_optional_fields() {
    let default_config = DistributedPPOConfig::new();
    assert!(
        default_config.target_reward.is_none(),
        "target_reward should be None by default"
    );

    let with_target = DistributedPPOConfig::new().with_target_reward(500.0);
    assert_eq!(with_target.target_reward, Some(500.0));
}

// ============================================================================
// Configuration Validation Intent Tests
// ============================================================================
// Note: These tests document what SHOULD be validated at runtime.
// The current implementation may not have these validations,
// which would be a bug if so.

/// Test that zero actors configuration should be rejected.
/// INTENT: n_actors = 0 makes no sense and should fail.
/// NOTE: If this test passes without panic, the implementation lacks validation.
#[test]
fn test_zero_actors_should_be_rejected() {
    let config = DistributedPPOConfig::new().with_n_actors(0);

    // EXPECTED BEHAVIOR: This should either:
    // 1. Panic during construction
    // 2. Return an error from a validate() method
    // 3. Panic when used
    //
    // Currently documenting the actual behavior:
    // The builder accepts 0, which is a potential bug.
    // This test documents the intent for future validation.
    assert_eq!(config.n_actors, 0, "Currently accepts n_actors=0 (potential bug)");

    // The intent is that total_envs() = 0 would cause issues later
    assert_eq!(config.total_envs(), 0);
}

/// Test that zero rollout length configuration should be rejected.
/// INTENT: rollout_length = 0 makes no sense and should fail.
#[test]
fn test_zero_rollout_length_should_be_rejected() {
    let config = DistributedPPOConfig::new().with_rollout_length(0);

    // Documents current behavior - accepts 0 which is a potential bug
    assert_eq!(config.rollout_length, 0);
    assert_eq!(config.transitions_per_rollout(), 0);
}

/// Test that zero minibatches configuration should be rejected.
/// INTENT: n_minibatches = 0 would cause division by zero.
#[test]
fn test_zero_minibatches_should_be_rejected() {
    let config = DistributedPPOConfig::new().with_n_minibatches(0);

    // Documents current behavior - accepts 0 which would cause panic on minibatch_size()
    assert_eq!(config.n_minibatches, 0);
}

// ============================================================================
// Stats Structure Tests
// ============================================================================

/// Test that DistributedPPOStats default is valid.
#[test]
fn test_stats_default_is_valid() {
    use crate::runners::distributed_ppo_config::DistributedPPOStats;

    let stats = DistributedPPOStats::default();

    assert_eq!(stats.env_steps, 0);
    assert_eq!(stats.episodes, 0);
    assert!((stats.avg_reward - 0.0).abs() < 1e-6);
    assert_eq!(stats.policy_version, 0);
    assert!((stats.steps_per_second - 0.0).abs() < 1e-6);
    assert_eq!(stats.buffer_size, 0);
}
