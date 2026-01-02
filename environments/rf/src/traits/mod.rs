//! Trait Abstractions for RF Environment
//!
//! This module defines extensible traits for key RF simulation components:
//!
//! - [`PropagationModel`] - Path loss calculations (FSPL, Hata, COST-231, etc.)
//! - [`NoiseModel`] - Noise floor computations (thermal, ITU man-made)
//! - [`SignalProtocol`] - Signal generation patterns (flat, OFDM, FHSS, chirp)
//! - [`FrequencyBand`] - Band configurations combining models with defaults
//!
//! These traits enable:
//! - Custom model implementations for specific use cases
//! - Runtime model selection without code changes
//! - Future band/protocol additions without modifying core simulation

pub mod propagation;
pub mod noise;
pub mod protocol;
pub mod band;

pub use propagation::{
    PropagationModel, FSPLModel, LogDistanceModel, HataUrbanModel, Cost231Model,
};
pub use noise::{NoiseModel, ThermalNoiseModel, ITUNoiseModel};
pub use protocol::{
    SignalProtocol, FlatSpectrum, OFDMProtocol, FHSSProtocol, ChirpProtocol, DSSSProtocol,
};
pub use band::FrequencyBand;
