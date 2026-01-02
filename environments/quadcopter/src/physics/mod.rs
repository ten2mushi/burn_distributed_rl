//! Physics simulation modules for the quadcopter.
//!
//! Contains:
//! - Quaternion operations (scalar and SIMD)
//! - Core rigid body dynamics
//! - Motor dynamics (first-order response)
//! - Aerodynamic effects (ground effect, drag)
//! - SIMD helper functions

pub mod quaternion;
pub mod dynamics;
pub mod motor;
pub mod aerodynamics;
pub mod simd_helpers;

pub use quaternion::*;
pub use dynamics::*;
pub use motor::*;
pub use aerodynamics::*;
