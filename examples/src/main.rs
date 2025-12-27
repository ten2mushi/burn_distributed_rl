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
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release -- distributed-ppo
//!
//! # Continuous actions (Pendulum)
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release -- distributed-ppo-continuous
//!
//! # Recurrent/LSTM discrete (CartPole)
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release -- distributed-recurrent-ppo
//!
//! # Recurrent/LSTM continuous (Pendulum)
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --release -- distributed-recurrent-ppo-continuous
//! ```
//!
//! ## IMPALA (Off-Policy with V-trace)
//!
//! ```bash
//! # Discrete actions (CartPole)
//! LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- distributed-impala
//! ```

mod distributed_impala;
mod distributed_ppo;
mod distributed_ppo_continuous;
mod distributed_recurrent_ppo;
mod distributed_recurrent_ppo_continuous;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Check command line args for algorithm selection
    if args.len() > 1 {
        match args[1].as_str() {
            // ============================================================
            // Distributed PPO (On-Policy)
            // ============================================================

            // Feed-forward, discrete actions (CartPole)
            "distributed-ppo" => distributed_ppo::run(),

            // Feed-forward, continuous actions (Pendulum)
            "distributed-ppo-continuous" => distributed_ppo_continuous::run(),

            // Recurrent/LSTM, discrete actions (CartPole)
            "distributed-recurrent-ppo" => distributed_recurrent_ppo::run(),

            // Recurrent/LSTM, continuous actions (Pendulum)
            "distributed-recurrent-ppo-continuous" => distributed_recurrent_ppo_continuous::run(),

            // ============================================================
            // Distributed IMPALA (Off-Policy with V-trace)
            // ============================================================

            // Feed-forward, discrete actions (CartPole)
            "distributed-impala" => distributed_impala::run(),

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
    println!("                    DISTRIBUTED PPO (On-Policy, Multi-Actor)");
    println!("=============================================================================");
    println!();
    println!("  distributed-ppo                   Feed-forward, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    4 actors x 32 envs = 128 parallel");
    println!();
    println!("  distributed-ppo-continuous        Feed-forward, continuous actions");
    println!("                                    Environment: Pendulum (swing-up)");
    println!("                                    Squashed Gaussian policy");
    println!();
    println!("  distributed-recurrent-ppo         LSTM, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    Memory for partial observability");
    println!();
    println!("  distributed-recurrent-ppo-continuous");
    println!("                                    LSTM, continuous actions");
    println!("                                    Environment: Pendulum");
    println!("                                    LSTM + Squashed Gaussian");
    println!();
    println!("=============================================================================");
    println!("                   DISTRIBUTED IMPALA (Off-Policy, V-trace)");
    println!("=============================================================================");
    println!();
    println!("  distributed-impala                Feed-forward, discrete actions");
    println!("                                    Environment: CartPole");
    println!("                                    Async collection with V-trace correction");
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
    println!("  Type Aliases:");
    println!("    DistributedPPODiscrete<B>              - PPO + discrete + feed-forward");
    println!("    DistributedPPOContinuous<B>            - PPO + continuous + feed-forward");
    println!("    DistributedRecurrentPPODiscrete<B>     - PPO + discrete + LSTM");
    println!("    DistributedRecurrentPPOContinuous<B>   - PPO + continuous + LSTM");
    println!("    DistributedIMPALADiscrete<B>           - IMPALA + discrete + feed-forward");
    println!();
    println!("=============================================================================");
    println!("                              QUICK START");
    println!("=============================================================================");
    println!();
    println!("  # Run distributed PPO on CartPole (recommended first example)");
    println!("  LIBRARY_PATH=/opt/homebrew/lib cargo run --features simd --release -- distributed-ppo");
    println!();
}
