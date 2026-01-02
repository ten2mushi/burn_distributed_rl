//! Comprehensive tests for the Quadcopter RL Environment.
//!
//! These tests follow the "Tests as Definition" philosophy, providing
//! complete behavioral specifications through exhaustive test coverage.
//!
//! ## Organization
//!
//! - `physics_tests`: Core physics simulation correctness
//! - `state_tests`: State management and initialization
//! - `environment_tests`: Environment API and trait implementation
//! - `termination_tests`: Episode termination conditions
//! - `reward_tests`: Reward computation and components
//! - `edge_case_tests`: Boundary conditions and edge cases
//! - `integration_tests`: Full environment integration scenarios

pub mod physics_tests;
pub mod state_tests;
pub mod environment_tests;
pub mod termination_tests;
pub mod reward_tests;
pub mod edge_case_tests;
pub mod integration_tests;
