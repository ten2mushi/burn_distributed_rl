//! Actor system for Distributed.
//!
//! - `Actor`: Generic actor that collects experience from environments
//! - `ActorPool`: Multi-actor coordinator for distributed training
//!
//! # WGPU Thread Safety
//!
//! With WGPU backend, actors can perform model inference directly
//! without synchronization primitives. CubeCL streams provide
//! automatic cross-thread tensor synchronization.

pub mod actor;
pub mod actor_pool;

pub use actor::{Actor, ActorConfig, ActorHandle};
pub use actor_pool::{ActorPool, ActorPoolConfig, ActorPoolBuilder};

// Re-export from messages for convenience
pub use crate::messages::{ActorMsg, ActorStats};
