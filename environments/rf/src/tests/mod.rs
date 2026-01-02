//! RF Environment Tests
//!
//! Following the "Tests as Definition: the Yoneda Way" philosophy,
//! these tests serve as the complete behavioral specification of
//! the RF environment.
//!
//! # Test Categories
//!
//! - **simd_tests**: Accuracy and correctness of SIMD primitives
//! - **physics_tests**: Path loss, fading, multipath, atmospheric, Doppler, and channel models
//! - **agent_tests**: Multi-agent actions, state, interference, observations, and rewards
//! - **comprehensive_tests**: Complete behavioral specification covering:
//!   - Config module: Builder pattern, validation, edge cases
//!   - State module: SoA layout, indexing, PSD operations
//!   - Entity module: Spawner, behaviors, EntitySoA operations
//!   - Spectrum module: PSD buffer, patterns, noise modeling
//!   - Observation module: Compression, normalization, multi-agent
//!   - Adapter module: VectorizedEnv interface, agent views
//!   - Integration tests: Full environment workflows

#[cfg(feature = "simd")]
mod simd_tests;

mod physics_tests;
mod agent_tests;
mod comprehensive_tests;
