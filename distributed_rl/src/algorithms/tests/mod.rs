//! Comprehensive tests for the algorithms module.
//!
//! This module contains exhaustive tests following the "Tests as Definition" philosophy,
//! implementing the Yoneda approach where tests serve as the complete specification
//! of behavior. Each test defines one aspect of correct behavior, and together
//! they characterize the implementation completely.
//!
//! # Test Organization
//!
//! - `gae_tests`: Tests for Generalized Advantage Estimation
//! - `vtrace_tests`: Tests for V-trace off-policy correction
//! - `continuous_policy_tests`: Tests for squashed Gaussian policies
//! - `policy_loss_tests`: Tests for PPO/IMPALA loss functions
//! - `integration_tests`: End-to-end algorithm tests

pub mod gae_tests;
pub mod vtrace_tests;
pub mod continuous_policy_tests;
pub mod policy_loss_tests;
pub mod integration_tests;
