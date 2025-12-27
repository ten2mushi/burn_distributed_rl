//! Comprehensive test suite for distributed RL runners.
//!
//! These tests serve as the behavioral specification for the runners module,
//! following the "Tests as Definition: the Yoneda Way" philosophy.
//!
//! # Test Organization
//!
//! - `config_tests`: Configuration builder pattern and validation
//! - `storage_tests`: Rollout storage and minibatch generation
//! - `ppo_runner_tests`: Feed-forward distributed PPO runner
//! - `recurrent_ppo_runner_tests`: Recurrent PPO with hidden state management
//! - `impala_runner_tests`: Off-policy IMPALA with V-trace correction
//! - `impala_freeze_bug_tests`: Freeze bug diagnosis and prevention
//! - `integration_tests`: End-to-end multi-thread tests
//! - `sps_degradation_tests`: CPU-side SPS degradation diagnostics
//! - `wgpu_pipeline_sps_tests`: GPU/WGPU-specific SPS degradation tests
//! - `env_wrapper_sps_tests`: Environment wrapper layer SPS degradation tests
//!
//! # Critical Invariants Tested
//!
//! 1. **Terminal vs Truncated Distinction**
//!    - Terminal: episode truly ended (goal/failure), bootstrap = 0
//!    - Truncated: episode cut short (time limit), bootstrap = V(s')
//!
//! 2. **Hidden State Lifecycle (Recurrent)**
//!    - Persists within episode
//!    - Resets ONLY on terminal, NOT on truncated
//!    - Independent across environments
//!
//! 3. **Concurrency Safety**
//!    - BytesSlot version monotonicity
//!    - No torn reads during weight transfer
//!    - Buffer synchronization between actors/learner
//!
//! 4. **Numerical Correctness**
//!    - GAE edge cases (lambda=0, lambda=1, gamma=0, gamma=1)
//!    - Advantage normalization safety (zero variance)
//!    - V-trace importance weight clipping
//!
//! 5. **WGPU/GPU Resource Management** (wgpu_pipeline_sps_tests)
//!    - Tensor creation timing stability
//!    - Forward/backward pass timing consistency
//!    - GPU->CPU data extraction overhead
//!    - Weight serialization resource cleanup
//!    - Device synchronization impact
//!
//! 6. **Environment Wrapper Layer** (env_wrapper_sps_tests)
//!    - StepResult allocation overhead
//!    - Wrapper step performance over time
//!    - Observation buffer handling
//!    - Reset cycle performance
//!    - Trait object dispatch overhead
//!    - Long-run actor loop simulation

pub mod config_tests;
pub mod storage_tests;
pub mod ppo_runner_tests;
pub mod recurrent_ppo_runner_tests;
pub mod impala_runner_tests;
pub mod impala_freeze_bug_tests;
pub mod integration_tests;
pub mod sps_degradation_tests;
pub mod wgpu_pipeline_sps_tests;
pub mod env_wrapper_sps_tests;
