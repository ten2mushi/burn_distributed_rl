//! Tests for SAC algorithm components.
//!
//! These tests verify the core SAC functionality:
//! - Configuration
//! - Buffer operations
//! - Loss computations
//! - Entropy tuning

use super::*;
use burn::backend::{Autodiff, NdArray};
use burn::tensor::Tensor;

type B = Autodiff<NdArray<f32>>;

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_sac_config_continuous() {
    let config = SACConfig::continuous();

    assert_eq!(config.tau, 0.005);
    assert!(!config.hard_target_update);
    assert_eq!(config.policy_update_freq, 2);
    assert_eq!(config.target_update_freq, 1);
    assert!(config.auto_entropy_tuning);
}

#[test]
fn test_sac_config_discrete() {
    let config = SACConfig::discrete();

    assert_eq!(config.tau, 1.0);
    assert!(config.hard_target_update);
    assert_eq!(config.policy_update_freq, 1);
    assert_eq!(config.target_update_freq, 8000);
    assert_eq!(config.target_entropy_scale, 0.89);
}

#[test]
fn test_target_entropy_computation() {
    let continuous_config = SACConfig::continuous();
    let discrete_config = SACConfig::discrete();

    // Continuous: -dim(A)
    let target = continuous_config.compute_target_entropy(4, false);
    assert_eq!(target, -4.0);

    // Discrete: 0.89 * log(|A|)
    let target = discrete_config.compute_target_entropy(4, true);
    assert!((target - 0.89 * (4.0_f32).ln()).abs() < 0.01);
}

#[test]
fn test_config_builder() {
    let config = SACConfig::continuous()
        .with_n_actors(8)
        .with_batch_size(512)
        .with_gamma(0.95)
        .with_tau(0.01);

    assert_eq!(config.n_actors, 8);
    assert_eq!(config.batch_size, 512);
    assert_eq!(config.gamma, 0.95);
    assert_eq!(config.tau, 0.01);
}

// ============================================================================
// Buffer Tests
// ============================================================================

#[test]
fn test_sac_buffer_creation() {
    let config = SACBufferConfig {
        capacity: 1000,
        min_size: 100,
        batch_size: 32,
    };

    let buffer = SACBuffer::<SACTransition<()>>::new(config);

    assert_eq!(buffer.current_size(), 0);
    assert_eq!(buffer.config().capacity, 1000);
    assert!(!buffer.is_training_ready());
}

#[test]
fn test_sac_buffer_push_and_sample() {
    let config = SACBufferConfig {
        capacity: 100,
        min_size: 10,
        batch_size: 5,
    };

    let buffer = SACBuffer::<SACTransition<()>>::new(config);

    // Push transitions
    for i in 0..20 {
        let trans = SACTransition::new(
            crate::core::transition::Transition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
                false,
            ),
        );
        buffer.push_transition(trans);
    }

    // Consolidation happens automatically on sample/is_training_ready
    assert_eq!(buffer.current_size() + buffer.pending_size(), 20);
    assert!(buffer.is_training_ready());

    // Sample batch
    let batch = buffer.sample_batch();
    assert!(batch.is_some());
    assert_eq!(batch.unwrap().len(), 5);
}

#[test]
fn test_sac_buffer_ring_overflow() {
    let config = SACBufferConfig {
        capacity: 10,
        min_size: 5,
        batch_size: 3,
    };

    let buffer = SACBuffer::<SACTransition<()>>::new(config);

    // Push more than capacity
    for i in 0..25 {
        let trans = SACTransition::new(
            crate::core::transition::Transition::new_discrete(
                vec![i as f32],
                0,
                1.0,
                vec![(i + 1) as f32],
                false,
                false,
            ),
        );
        buffer.push_transition(trans);
    }

    // Trigger consolidation and check size
    assert!(buffer.is_training_ready());

    // Should only have capacity items
    assert_eq!(buffer.current_size(), 10);
}

// ============================================================================
// Transition Tests
// ============================================================================

#[test]
fn test_sac_transition_creation() {
    let base = crate::core::transition::Transition::new_discrete(
        vec![1.0, 2.0],
        1,
        0.5,
        vec![2.0, 3.0],
        false,
        false,
    );

    let trans = SACTransition::new(base);

    assert_eq!(trans.state(), &[1.0, 2.0]);
    assert_eq!(trans.reward(), 0.5);
    assert!(!trans.terminal());
}

#[test]
fn test_sac_transition_terminal() {
    let base = crate::core::transition::Transition::new_discrete(
        vec![1.0],
        0,
        1.0,
        vec![0.0],
        true,
        false,
    );

    let trans = SACTransition::new(base);

    assert!(trans.terminal());
}

// ============================================================================
// Entropy Tuner Tests
// ============================================================================

#[test]
fn test_entropy_tuner_creation() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let tuner: EntropyTuner<B> = EntropyTuner::new(0.2, -3.0, &device);

    assert!((tuner.alpha() - 0.2).abs() < 0.01);
    assert!((tuner.cached_alpha() - 0.2).abs() < 0.01);
    assert_eq!(tuner.target_entropy(), -3.0);
}

#[test]
fn test_entropy_tuner_loss() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let tuner: EntropyTuner<B> = EntropyTuner::new(0.2, -3.0, &device);

    let log_probs: Tensor<B, 1> = Tensor::from_floats([-2.0, -3.0, -4.0], &device);
    let loss = tuner.loss(log_probs);
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // L = -α * (mean_log_prob + H_target)
    // mean_log_prob = -3.0
    // L = -0.2 * (-3.0 + (-3.0)) = -0.2 * (-6.0) = 1.2
    assert!((loss_val - 1.2).abs() < 0.1);
}

#[test]
fn test_entropy_tuner_cache_update() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let mut tuner: EntropyTuner<B> = EntropyTuner::new(0.2, -3.0, &device);

    // Modify log_alpha
    let new_log_alpha = Tensor::from_floats([0.0], &device); // exp(0) = 1.0
    tuner.set_log_alpha(new_log_alpha);

    // Cached value still old
    assert!((tuner.cached_alpha() - 0.2).abs() < 0.01);

    // Update cache
    tuner.update_cache();

    // Now cached value is updated
    assert!((tuner.cached_alpha() - 1.0).abs() < 0.01);
}

// ============================================================================
// Loss Function Tests
// ============================================================================

#[test]
fn test_sac_critic_loss() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    let q1: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    let q2: Tensor<B, 1> = Tensor::from_floats([1.1, 2.1, 3.1], &device);
    let targets: Tensor<B, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);

    let loss = sac_critic_loss(q1, q2, targets);
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // Q1 is exact match (loss = 0)
    // Q2 is off by 0.1: MSE = (0.01 + 0.01 + 0.01) / 3 = 0.01
    // Total = 0.01
    assert!(loss_val > 0.0);
    assert!(loss_val < 0.05);
}

#[test]
fn test_sac_actor_loss() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    let min_q: Tensor<B, 1> = Tensor::from_floats([10.0, 10.0], &device);
    let log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);
    let alpha = 0.2;

    let loss = sac_actor_loss(min_q, log_probs, alpha);
    let loss_val = loss.into_data().as_slice::<f32>().unwrap()[0];

    // L = mean(α*log_π - Q) = mean(0.2*(-1) - 10) = mean(-10.2) = -10.2
    assert!((loss_val - (-10.2)).abs() < 0.01);
}

#[test]
fn test_sac_td_targets() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    let rewards: Tensor<B, 1> = Tensor::from_floats([1.0, 1.0], &device);
    let terminals: Tensor<B, 1> = Tensor::from_floats([0.0, 1.0], &device);
    let min_q_next: Tensor<B, 1> = Tensor::from_floats([10.0, 10.0], &device);
    let next_log_probs: Tensor<B, 1> = Tensor::from_floats([-1.0, -1.0], &device);

    let targets = sac_td_targets(rewards, terminals, min_q_next, next_log_probs, 0.99, 0.2);
    let data = targets.into_data();
    let slice = data.as_slice::<f32>().unwrap();

    // Non-terminal: y = 1 + 0.99 * (10 - 0.2*(-1)) = 1 + 0.99 * 10.2 = 11.098
    assert!((slice[0] - 11.098).abs() < 0.01);

    // Terminal: y = 1 (no future reward)
    assert!((slice[1] - 1.0).abs() < 0.01);
}

// ============================================================================
// Entropy Helpers Tests
// ============================================================================

#[test]
fn test_target_entropy_continuous() {
    let target = target_entropy_continuous(3);
    assert_eq!(target, -3.0);
}

#[test]
fn test_target_entropy_discrete() {
    let target = target_entropy_discrete(4, 0.89);
    // 0.89 * ln(4) ≈ 0.89 * 1.386 ≈ 1.234
    assert!((target - 1.234).abs() < 0.01);
}

#[test]
fn test_gaussian_entropy() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    // log_std = 0 means std = 1
    let log_std: Tensor<B, 2> = Tensor::from_floats([[0.0, 0.0]], &device);
    let entropy = gaussian_entropy(log_std);
    let val = entropy.into_data().as_slice::<f32>().unwrap()[0];

    // H = 2 * 0.9189 = 1.8378 for 2D Gaussian with std=1
    assert!((val - 1.8378).abs() < 0.01);
}

#[test]
fn test_categorical_entropy() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    // Uniform distribution over 4 actions
    let log_probs: Tensor<B, 2> = Tensor::from_floats(
        [[(0.25_f32).ln(), (0.25_f32).ln(), (0.25_f32).ln(), (0.25_f32).ln()]],
        &device,
    );
    let entropy = categorical_entropy(log_probs);
    let val = entropy.into_data().as_slice::<f32>().unwrap()[0];

    // H of uniform distribution = log(4) ≈ 1.386
    assert!((val - (4.0_f32).ln()).abs() < 0.01);
}

// ============================================================================
// SAC Stats Tests
// ============================================================================

#[test]
fn test_sac_stats() {
    let mut stats = SACStats::new();

    stats.add_episode_return(100.0, 10);
    stats.add_episode_return(200.0, 10);
    stats.add_episode_return(150.0, 10);

    assert_eq!(stats.recent_returns.len(), 3);
    assert!((stats.mean_return - 150.0).abs() < 0.01);
}

#[test]
fn test_sac_stats_max_recent() {
    let mut stats = SACStats::new();

    for i in 0..15 {
        stats.add_episode_return(i as f32, 10);
    }

    // Should only keep last 10
    assert_eq!(stats.recent_returns.len(), 10);
    assert_eq!(stats.recent_returns[0], 5.0);
}

// ============================================================================
// Bug Regression Tests
// ============================================================================

/// Test that target entropy for discrete actions uses action_space_size(), not action_dim().
///
/// Bug: action_dim() returns 1 for discrete (the number of dimensions to encode an action),
/// but target entropy should use action_space_size() (the number of discrete actions).
///
/// For CartPole with 2 actions:
/// - Wrong: 0.89 * ln(1) = 0
/// - Correct: 0.89 * ln(2) ≈ 0.617
#[test]
fn test_discrete_target_entropy_uses_action_space_size() {
    use crate::algorithms::action_policy::{ActionPolicy, DiscretePolicy};

    let policy = DiscretePolicy::new(2); // CartPole: 2 actions

    // action_dim() returns 1 for discrete (dimensionality of action encoding)
    let action_dim: usize = <DiscretePolicy as ActionPolicy<B>>::action_dim(&policy);
    assert_eq!(action_dim, 1, "action_dim() should return 1 for discrete");

    // action_space_size() returns the actual number of actions
    let action_space_size: usize = <DiscretePolicy as ActionPolicy<B>>::action_space_size(&policy);
    assert_eq!(action_space_size, 2, "action_space_size() should return 2 for CartPole");

    // Target entropy calculation - the CORRECT value
    let config = SACConfig::discrete();
    let correct_target_entropy = config.compute_target_entropy(action_space_size, true);

    // 0.89 * ln(2) ≈ 0.617
    assert!(
        (correct_target_entropy - 0.617).abs() < 0.01,
        "Target entropy for 2 actions should be ~0.617, got {}",
        correct_target_entropy
    );

    // Using action_dim (1) would give WRONG result
    let wrong_target_entropy = config.compute_target_entropy(action_dim, true);
    assert_eq!(
        wrong_target_entropy, 0.0,
        "Using action_dim(1) gives target_entropy=0 which is wrong"
    );
}

/// Test that discrete target entropy scales correctly with number of actions.
#[test]
fn test_discrete_target_entropy_various_action_counts() {
    let config = SACConfig::discrete();

    // Test various action space sizes
    let test_cases = [
        (2, 0.89 * (2.0_f32).ln()),  // CartPole
        (4, 0.89 * (4.0_f32).ln()),  // Atari (4 directions)
        (18, 0.89 * (18.0_f32).ln()), // Atari full
    ];

    for (n_actions, expected) in test_cases {
        let target = config.compute_target_entropy(n_actions, true);
        assert!(
            (target - expected).abs() < 0.01,
            "For {} actions, expected target entropy {}, got {}",
            n_actions, expected, target
        );
    }
}
