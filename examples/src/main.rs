//! Distributed RL Examples
//!
//! Recursion limit needed for WGPU's deeply nested types.
#![recursion_limit = "256"]
//!
//! # Distributed Training Examples (Multi-Actor, WGPU)
//!
//! All distributed examples use the high-level API with Model Factory pattern:
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - BytesSlot for WGPU-safe weight synchronization
//!
//! ## PPO (On-Policy)
//!
//! ```bash
//! # Discrete actions (CartPole)
//! cargo run --release -- ppo
//! ```
//!
//! ## IMPALA (Off-Policy with V-trace)
//!
//! ```bash
//! # Discrete actions (CartPole)
//! cargo run --features simd --release -- impala
//! ```

mod impala;
mod ppo;
mod ppo_continuous;
mod ppo_continuous_sde_showcase;
mod ppo_continuous_state_independent;
mod recurrent_ppo;
mod recurrent_ppo_continuous;
mod sac;
mod sac_continuous;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Check command line args for algorithm selection
    if args.len() > 1 {
        match args[1].as_str() {
            // ============================================================
            // PPO (Strategy Pattern Runner)
            // ============================================================

            // Feed-forward, discrete actions (CartPole)
            "ppo" => ppo::run(),

            // Feed-forward, continuous actions (Pendulum) - SDE
            "ppo-continuous" => ppo_continuous::run(),

            // Feed-forward, continuous actions (Pendulum) - State-Independent
            "ppo-continuous-si" => ppo_continuous_state_independent::run(),

            // ============================================================
            // SDE Showcase (All Exploration Strategies)
            // ============================================================

            // State-Independent baseline (same as ppo-continuous-si)
            "sde-si" => ppo_continuous_sde_showcase::run_state_independent(),

            // Shared Features SDE (often unstable - for comparison)
            "sde-shared" => ppo_continuous_sde_showcase::run_shared_features(),

            // Separate Network SDE (clean architecture)
            "sde-separate" => ppo_continuous_sde_showcase::run_separate_network(),

            // Residual SDE (SI + small adjustment)
            "sde-residual" => ppo_continuous_sde_showcase::run_residual(),

            // Generalized SDE (gSDE - SB3 style)
            "sde-gsde" => ppo_continuous_sde_showcase::run_generalized(),

            // Recurrent (LSTM), discrete actions (CartPole)
            "recurrent-ppo" => recurrent_ppo::run(),

            // Recurrent (LSTM), continuous actions (Pendulum)
            "recurrent-ppo-continuous" => recurrent_ppo_continuous::run(),

            // ============================================================
            // IMPALA (Off-Policy with V-trace)
            // ============================================================

            // Feed-forward, discrete actions (CartPole)
            "impala" | "distributed-impala" => impala::run(),

            // ============================================================
            // SAC (Off-Policy, Maximum Entropy)
            // ============================================================

            // Feed-forward, discrete actions (CartPole)
            "sac" => sac::run(),

            // Feed-forward, continuous actions (Pendulum)
            "sac-continuous" => sac_continuous::run(),

            _ => {
                println!("Unknown algorithm: {}", args[1]);
                println!();
                print_usage();
            }
        }
    } else {
        print_usage();
    }
}

fn print_usage() {
    println!("Usage: cargo run --features simd --release -- <algorithm>");
    println!();
    println!("=============================================================================");
    println!("                              PPO RUNNER");
    println!("=============================================================================");
    println!();
    println!("  ppo                               Feed-forward, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    Uses PPORunner<A, T, B, S> with");
    println!("                                    FeedForwardStrategy (zero-cost)");
    println!();
    println!("  ppo-continuous                    Feed-forward, continuous actions (SDE)");
    println!("                                    Environment: Pendulum");
    println!("                                    State-Dependent Exploration");
    println!("                                    ~100-500k steps to solve");
    println!();
    println!("  ppo-continuous-si                 Feed-forward, continuous actions");
    println!("                                    Environment: Pendulum");
    println!("                                    State-Independent Exploration");
    println!();
    println!("=============================================================================");
    println!("                    SDE SHOWCASE (Exploration Strategies)");
    println!("=============================================================================");
    println!();
    println!("  sde-si                            State-Independent baseline");
    println!("                                    log_std = Param (single parameter)");
    println!();
    println!("  sde-shared                        Shared Features SDE (UNSTABLE)");
    println!("                                    log_std = Linear(actor_features)");
    println!("                                    Causes gradient interference!");
    println!();
    println!("  sde-separate                      Separate Network SDE (CLEAN)");
    println!("                                    log_std = MLP_separate(obs)");
    println!("                                    No gradient interference");
    println!();
    println!("  sde-residual                      Residual SDE (STABLE + ADAPTIVE)");
    println!("                                    log_std = base + scale*adjustment");
    println!("                                    Best of SI + state-dependent");
    println!();
    println!("  sde-gsde                          Generalized SDE (SB3 style)");
    println!("                                    Separate exploration matrix");
    println!("                                    State-correlated noise");
    println!();
    println!("  recurrent-ppo                     LSTM-based, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    Uses RecurrentStrategy with TBPTT");
    println!("                                    Hidden state reset on episode boundaries");
    println!();
    println!("  recurrent-ppo-continuous          LSTM-based, continuous actions");
    println!("                                    Environment: Pendulum");
    println!("                                    Squashed Gaussian + LSTM policy");
    println!("                                    Per-environment hidden state tracking");
    println!();
    println!("=============================================================================");
    println!("                      IMPALA (Off-Policy, V-trace)");
    println!("=============================================================================");
    println!();
    println!("  impala                            Feed-forward, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    Async collection with V-trace correction");
    println!();
    println!("=============================================================================");
    println!("                      SAC (Off-Policy, Maximum Entropy)");
    println!("=============================================================================");
    println!();
    println!("  sac                               Feed-forward, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    Separate Actor + Twin Q Critics");
    println!("                                    Auto entropy tuning (alpha)");
    println!();
    println!("  sac-continuous                    Feed-forward, continuous actions");
    println!("                                    Environment: Pendulum");
    println!("                                    Squashed Gaussian policy");
    println!("                                    Soft target updates (tau=0.005)");
    println!();
    println!("=============================================================================");
    println!("                              ARCHITECTURE");
    println!("=============================================================================");
    println!();
    println!("  All distributed examples use:");
    println!("    - Model Factory Pattern: Fresh model per actor device");
    println!("    - BytesSlot: WGPU-safe weight transfer (Vec<u8> is Send+Sync)");
    println!("    - BinBytesRecorder: Efficient serialization");
    println!();
    println!("  PPO Type Aliases:");
    println!("    PPODiscrete<B>                  - PPO + discrete + feed-forward");
    println!("    PPOContinuous<B>                - PPO + continuous + feed-forward");
    println!("    RecurrentPPODiscrete<B>         - PPO + discrete + LSTM");
    println!("    RecurrentPPOContinuous<B>       - PPO + continuous + LSTM");
    println!();
    println!("  IMPALA Type Aliases:");
    println!("    IMPALADiscrete<B>               - IMPALA + discrete + feed-forward");
    println!("    IMPALAContinuous<B>             - IMPALA + continuous + feed-forward");
    println!();
    println!("  SAC Type Aliases:");
    println!("    SACDiscrete<B>                  - SAC + discrete + feed-forward");
    println!("    SACContinuous<B>                - SAC + continuous + feed-forward");
    println!("    RecurrentSACDiscrete<B>         - SAC + discrete + LSTM");
    println!("    RecurrentSACContinuous<B>       - SAC + continuous + LSTM");
    println!();
    println!("=============================================================================");
    println!("                              QUICK START");
    println!("=============================================================================");
    println!();
    println!("  # Run PPO on CartPole (recommended first example)");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- ppo");
    println!();
    println!("  # Run PPO Continuous on Pendulum");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- ppo-continuous");
    println!();
    println!("  # Run Recurrent PPO on CartPole (LSTM-based)");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- recurrent-ppo");
    println!();
    println!("  # Run Recurrent PPO Continuous on Pendulum (LSTM + squashed Gaussian)");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- recurrent-ppo-continuous");
    println!();
    println!("  # Run SAC on CartPole (discrete, off-policy)");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- sac");
    println!();
    println!("  # Run SAC Continuous on Pendulum");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- sac-continuous");
    println!();
}
