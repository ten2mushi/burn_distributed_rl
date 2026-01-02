//! Multi-Agent Support Module
//!
//! This module provides the multi-agent architecture for the RF environment,
//! supporting jammer vs cognitive radio (CR) adversarial scenarios.
//!
//! # Agent Types
//!
//! - **Jammer**: Attempts to disrupt CR communications by transmitting
//!   interference signals on target frequencies
//! - **Cognitive Radio (CR)**: Attempts to maintain communication while
//!   avoiding or mitigating jamming interference
//!
//! # Architecture
//!
//! Uses Struct-of-Arrays (SoA) layout for cache efficiency and SIMD
//! compatibility, matching the existing entity and state patterns.

pub mod actions;
pub mod interference;
pub mod multi_agent;
pub mod presets;
pub mod rewards;

pub use actions::*;
pub use interference::*;
pub use multi_agent::*;
pub use presets::*;
pub use rewards::*;
