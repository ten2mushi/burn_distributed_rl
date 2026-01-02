//! SIMD-Optimized Quadcopter RL Environment
//!
//! A high-performance quadcopter simulation for reinforcement learning,
//! implementing the operant Environment trait with full SIMD vectorization.
//!
//! # Features
//!
//! - **SIMD Vectorization**: Processes 8 environments per instruction using f32x8
//! - **Quaternion Physics**: Accurate rigid body dynamics with gimbal-lock-free rotation
//! - **Configurable Rewards**: Builder pattern for customizable reward components
//! - **Non-Auto-Reset API**: Supports value-based RL algorithms (DQN, SAC, C51)
//! - **Crazyflie 2.0 Model**: Accurate physical constants from real hardware
//!
//! # Architecture
//!
//! The environment uses a Struct-of-Arrays (SoA) memory layout for cache-efficient
//! SIMD operations. All state variables are stored in separate contiguous arrays,
//! enabling 8-wide vector processing.
//!
//! # Example
//!
//! ```rust,ignore
//! use quadcopter_env::{QuadcopterConfig, ObsConfig, Quadcopter, presets};
//! use operant_core::Environment;
//!
//! // Create environment with default hover reward
//! let config = QuadcopterConfig::new(64)
//!     .with_physics_freq(240)
//!     .with_ctrl_freq(30)
//!     .with_observation(ObsConfig::kinematic())
//!     .with_ground_effect(true)
//!     .with_drag(true);
//!
//! let mut env = Quadcopter::from_config(config).unwrap();
//!
//! // Or with custom reward composition
//! let config = QuadcopterConfig::new(64)
//!     .with_reward_fn(presets::tracking());
//!
//! // Reset and run
//! env.reset(42);
//!
//! // Step with normalized motor commands [-1, 1] → [0, MAX_RPM]
//! let actions = vec![0.0f32; 64 * 4]; // 64 envs × 4 motors
//! env.step(&actions);
//!
//! // Read results
//! let mut obs = vec![0.0f32; 64 * 12]; // 12 = kinematic observation size
//! let mut rewards = vec![0.0f32; 64];
//! env.write_observations(&mut obs);
//! env.write_rewards(&mut rewards);
//! ```
//!
//! # Integration with distributed_rl
//!
//! For use with the distributed_rl training system, enable the `distributed_rl` feature
//! and use the adapter module:
//!
//! ```rust,ignore
//! use quadcopter_env::adapter::distributed::QuadcopterEnv;
//! use quadcopter_env::QuadcopterConfig;
//!
//! let config = QuadcopterConfig::new(64);
//! let env = QuadcopterEnv::from_config(config).unwrap();
//!
//! // Use with PPOContinuous learner
//! let learner = PPOContinuous::new(learner_config, algorithm);
//! learner.run(model, env, optimizer, callback);
//! ```

#![cfg_attr(feature = "simd", feature(portable_simd))]

// Core modules
pub mod config;
pub mod constants;
pub mod state;
pub mod types;

// Physics simulation
pub mod physics;

// Environment components
pub mod env;
pub mod noise;
pub mod normalization;
pub mod observation;
pub mod reward;
pub mod termination;

// Integration adapters
pub mod adapter;

// Visualization (optional)
#[cfg(feature = "render")]
pub mod renderer;

// Comprehensive test suite
#[cfg(test)]
pub mod tests;

// Re-exports for convenience
pub use config::{DistributionType, InitConfig, ObsConfig, QuadcopterConfig, TaskMode, TerminationConfig};
pub use constants::{
    action_to_rpm, rpm_to_action, G, GRAVITY_FORCE, HOVER_RPM, IXX, IYY, IZZ, KF, KM, L, M,
    MAX_RPM, MIN_RPM, MOTOR_TIME_CONSTANT,
};
pub use physics::motor::MotorDynamicsConfig;
pub use env::Quadcopter;
pub use noise::{NoiseConfig, XorShiftRng};
pub use normalization::{NormalizationConfig, RewardProcessor, RunningMeanStd};
pub use state::QuadcopterState;
pub use types::{
    BodyFrame, BoundedRPM, NormalizedAction, PositiveScalar, Rotation, UnitQuaternion, Vec3,
    WorldFrame,
};

// Re-export reward types for easy access
pub use reward::{presets, RewardComponent};
pub use reward::components::{
    ActionMagnitudePenalty, ActionRatePenalty, AliveBonus, AngularVelocityPenalty,
    AttitudePenalty, PositionError, VelocityError,
};

// Re-export adapter types
pub use adapter::{QuadcopterEnvWrapper, QuadcopterStepResult};

#[cfg(feature = "distributed_rl")]
pub use adapter::distributed::QuadcopterEnv;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Basic usage tests are in the tests module (src/tests/)
