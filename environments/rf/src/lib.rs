//! RF Environment - High-fidelity SIMD-optimized radio frequency spectrum simulation
//!
//! This crate provides a vectorized RF world simulation for reinforcement learning,
//! featuring multi-agent support (jammers vs cognitive radios) with realistic
//! physics including path loss, fading, and Doppler effects.
//!
//! # Architecture
//!
//! The environment uses Struct-of-Arrays (SoA) layout for cache efficiency
//! and `std::simd` for 8-wide f32 vector operations.
//!
//! # Features
//!
//! - `simd` (default, **required**): Enables SIMD-optimized operations using `std::simd`
//!
//! # Requirements
//!
//! This crate requires the `simd` feature to be enabled (it is enabled by default).
//! SIMD is a baseline requirement for RF environment simulation performance.

#![cfg_attr(feature = "simd", feature(portable_simd))]

// SIMD is a baseline requirement - fail compilation if not enabled
#[cfg(not(feature = "simd"))]
compile_error!(
    "rf_environment requires the 'simd' feature to be enabled. \
     This is the default, but if you explicitly disabled it, please re-enable it. \
     Add to your Cargo.toml: rf_environment = {{ features = [\"simd\"] }}"
);

// SIMD module - core vectorized primitives
#[cfg(feature = "simd")]
pub mod simd_rf;

// Type-safe refinement types (Curry-Howard compliant)
pub mod types;

// Core modules
pub mod config;
pub mod constants;
pub mod env;
pub mod state;

// Entity system (Phase 2)
pub mod entities;

// Spectrum synthesis (Phase 2)
pub mod spectrum;

// Physics/Channel models (Phase 3)
pub mod physics;

// Multi-Agent Support (Phase 4)
pub mod agents;
pub mod observation;

// Trait Abstractions (Phase 5)
pub mod traits;
pub mod bands;

// Renderer module (feature-gated)
#[cfg(any(
    feature = "render",
    feature = "render-terminal",
    feature = "render-html",
    feature = "render-realtime",
    feature = "render-gif"
))]
pub mod renderer;

// VectorizedEnv Adapter (Phase 5)
pub mod adapter;

// Tests
#[cfg(test)]
mod tests;

// Re-exports for convenience
pub use config::{RFWorldConfig, EntityConfig};
pub use env::RFWorld;
pub use state::RFWorldState;

// Entity re-exports
pub use entities::{EntityType, ModulationType, EntitySoA, EntitySpawner};
// EntityConfig is re-exported from config

// Spectrum re-exports
pub use spectrum::{ValidatedPsd, NoiseEnvironment};

#[cfg(feature = "simd")]
pub use simd_rf::{
    complex::SimdComplex,
    math::{
        simd_atan2, simd_cos, simd_db_to_linear, simd_exp, simd_linear_to_db, simd_log, simd_pow,
        simd_rsqrt, simd_sin, simd_sincos, simd_sqrt, simd_wrap_phase,
    },
    random::SimdRng,
};

// Agent re-exports (Phase 4)
pub use agents::{
    ActionConfig, ActionSpace, AgentType, CognitiveRadioAction, JammerAction, JammerModulation,
    MultiAgentActions, MultiAgentConfig, MultiAgentState,
    InterferenceMatrix,
    MultiAgentRewards, RewardConfig, JammerRewardConfig, CRRewardConfig,
    ScenarioConfig, SelfPlayConfig, TeamRewardConfig, Curriculum,
};
pub use observation::{ObservationConfig, MultiAgentObservations, PsdCompression};

// Trait re-exports (Phase 5)
pub use traits::{
    PropagationModel, FSPLModel, LogDistanceModel, HataUrbanModel, Cost231Model,
    NoiseModel, ThermalNoiseModel, ITUNoiseModel,
    SignalProtocol, FlatSpectrum, OFDMProtocol, FHSSProtocol, ChirpProtocol, DSSSProtocol,
    FrequencyBand,
};
pub use bands::{UHFBand, SHFBand};

// Adapter re-exports (Phase 5)
pub use adapter::{
    VectorizedEnv, RFEnvAdapter, RFContinuousEnvAdapter,
    AgentView, StepResult, ResetMask,
};

// Type-safe refinement types re-exports
pub use types::{
    // Primitives
    PositivePower, PositiveF32, NonZeroPositiveF32,
    // Dimensional
    Hertz, Meters, Seconds, RadialVelocity, DopplerDirection,
    // Frequency domain
    FrequencyRange, ValidatedFrequencyGrid,
    // Power types
    PowerDbm, PathLoss, GainDb,
    // Fading types
    UnitMeanFading, UnitMeanShadowing, RayleighFading, RicianFading,
};

#[cfg(feature = "simd")]
pub use types::{PositivePower8, PositiveF32x8, UnitMeanFading8};
