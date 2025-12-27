//! Learning rate scheduling module.
//!
//! Provides schedulers for dynamic learning rate adjustment during training.
//!
//! ## Available Schedulers
//!
//! - [`ConstantLR`]: No scheduling (constant rate)
//! - [`LinearDecay`]: Linear interpolation from start to end LR
//! - [`CosineAnnealing`]: Cosine decay with optional warm restarts
//! - [`PolynomialDecay`]: Polynomial decay with configurable power
//! - [`Warmup`]: Wrapper for linear warmup phase before any scheduler
//!
//! ## Example
//!
//! ```rust,ignore
//! use burn_rl::distributed::scheduling::{LinearDecay, Warmup, LRScheduler};
//!
//! // Linear decay from 3e-4 to 0 over 1M steps
//! let scheduler = LinearDecay::new(3e-4, 0.0, 1_000_000);
//!
//! // With warmup
//! let with_warmup = Warmup::new(scheduler, 10_000, 0.0);
//!
//! // In training loop:
//! let lr = with_warmup.get_lr(step);
//! model = optimizer.step(lr, model, grads);
//! ```

pub mod lr_scheduler;

#[cfg(test)]
mod tests;

pub use lr_scheduler::{
    LRScheduler,
    ConstantLR,
    LinearDecay,
    CosineAnnealing,
    PolynomialDecay,
    Warmup,
};
