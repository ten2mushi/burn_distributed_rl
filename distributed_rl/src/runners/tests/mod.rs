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
//! - `integration_tests`: End-to-end multi-thread tests
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

pub mod config_tests;
pub mod storage_tests;
pub mod ppo_runner_tests;
pub mod recurrent_ppo_runner_tests;
pub mod impala_runner_tests;
pub mod integration_tests;
