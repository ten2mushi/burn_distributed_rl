//! Type-Safe Refinement Types for RF Simulation
//!
//! This module provides Curry-Howard compliant types that enforce physical
//! invariants at compile time. Types serve as propositions about physical
//! correctness, and valid programs constitute proofs.
//!
//! # Architecture
//!
//! ## Layer 1: Primitive Refinement Types
//! - [`PositivePower`] - Non-negative power values (Watts)
//! - [`PositiveF32`] - Non-negative f32 values
//! - [`NonZeroPositiveF32`] - Strictly positive f32 values
//!
//! ## Layer 2: Dimensional Types
//! - [`Hertz`] - Frequency values (always positive)
//! - [`Meters`] - Distance values (non-negative)
//! - [`Seconds`] - Time duration (non-negative)
//! - [`RadialVelocity`] - Velocity with enforced direction semantics
//!
//! ## Layer 3: Frequency Domain Types
//! - [`FrequencyRange`] - Validated min < max frequency range
//! - [`ValidatedFrequencyGrid`] - Frequency grid with invariant preservation
//!
//! ## Layer 4: Physics Constraint Types
//! - [`PowerDbm`] - Power in dBm with explicit domain
//! - [`PathLoss`] - Path loss values (always positive dB)
//! - [`UnitMeanFading`] - Fading coefficients with proven unit mean
//!
//! ## Layer 5: SIMD Types
//! - [`PositivePower8`] - SIMD vector of positive power values
//! - [`UnitMeanFading8`] - SIMD vector of unit-mean fading coefficients

pub mod primitives;
pub mod dimensional;
pub mod frequency;
pub mod power;
pub mod fading;

#[cfg(feature = "simd")]
pub mod simd;

// Re-export all types at module level
pub use primitives::{PositivePower, PositiveF32, NonZeroPositiveF32};
pub use dimensional::{Hertz, Meters, Seconds, RadialVelocity, DopplerDirection};
pub use frequency::{
    FrequencyRange, ValidatedFrequencyGrid,
    // Frequency bands
    PropagationType, FrequencyBand,
    VLFBand, LFBand, MFBand, HFBand, VHFBand, UHFBand, SHFBand, EHFBand,
    frequency_to_band, frequency_to_propagation,
};
pub use power::{PowerDbm, PathLoss, GainDb};
pub use fading::{UnitMeanFading, UnitMeanShadowing, RayleighFading, RicianFading};

#[cfg(feature = "simd")]
pub use simd::{PositivePower8, PositiveF32x8, UnitMeanFading8, UnitMeanShadowing8};
