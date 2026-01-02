//! Recursion limit needed for WGPU's deeply nested types.
#![recursion_limit = "256"]
//!
//! # Distributed Training Examples (Multi-Actor, WGPU)
//!
//! All distributed examples use the high-level API with Model Factory pattern:
//! - N actor threads (each with M vectorized environments)
//! - 1 learner thread (GPU training)
//! - BytesSlot for WGPU-safe weight synchronization

mod impala;
mod ppo;
mod ppo_continuous;
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
    println!("Usage: cargo run --release -- <algorithm>");
    println!();
    println!("  # Run PPO on CartPole (recommended first example)");
    println!("  cargo run --release -- ppo");
    println!();
    println!("  # Run PPO Continuous on Pendulum");
    println!("  cargo run --release -- ppo-continuous");
    println!();
    println!("  # Run Recurrent PPO on CartPole (LSTM-based)");
    println!("  cargo run --release -- recurrent-ppo");
    println!();
    println!("  # Run Recurrent PPO Continuous on Pendulum (LSTM + squashed Gaussian)");
    println!("  cargo run --release -- recurrent-ppo-continuous");
    println!();
    println!("  # Run SAC on CartPole (discrete, off-policy)");
    println!("  cargo run --release -- sac");
    println!();
    println!("  # Run SAC Continuous on Pendulum");
    println!("  cargo run --release -- sac-continuous");
    println!();
}
