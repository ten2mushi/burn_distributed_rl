//! Quadcopter SIMD Visualization Showcase
//!
//! Demonstrates real-time 3D visualization of 8 parallel drone simulations.
//! All drones are rendered in a unified 3D scene with color-coded identities,
//! trajectory trails, and orientation axes.
//!
//! # Running
//!
//! ```bash
//! cargo run --example visualization_showcase --release --features render-realtime
//! ```
//!
//! # Controls
//!
//! | Key | Action |
//! |-----|--------|
//! | 1-8 | Highlight specific drone |
//! | 0 | Clear highlight |
//! | T | Toggle trajectory trails |
//! | A | Toggle orientation axes |
//! | O | Toggle camera orbit |
//! | V | Toggle velocity vectors |
//! | G | Toggle ground grid |
//! | R | Reset camera |
//! | +/- | Zoom in/out |
//! | Arrows | Pan camera |
//! | PgUp/PgDn | Adjust elevation |
//! | ESC | Quit |

use quadcopter_env::{
    Quadcopter, QuadcopterConfig, ObsConfig, TerminationConfig,
    constants::{rpm_to_action, HOVER_RPM},
    presets,
};

#[cfg(feature = "render-realtime")]
use quadcopter_env::renderer::{
    CameraConfig, MultiEnvSnapshot, RollingHistory, RealtimeWindow,
    TrajectoryConfig, VisualizationConfig,
};

use operant_core::Environment;

#[cfg(feature = "render-realtime")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Quadcopter SIMD Visualization Showcase                 ║");
    println!("║       8 Parallel Drone Simulations in Real-Time              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    const NUM_ENVS: usize = 8;
    const TRAJECTORY_LENGTH: usize = 200;
    const SNAPSHOT_CAPACITY: usize = 100;

    // Create environment with 8 parallel drones
    let config = QuadcopterConfig::new(NUM_ENVS)
        .with_physics_freq(240)
        .with_ctrl_freq(30)
        .with_observation(ObsConfig::kinematic())
        .with_reward_fn(presets::hover())
        .with_termination(
            TerminationConfig::new()
                .with_max_steps(300)
                .with_position_bounds([-3.0, 3.0, -3.0, 3.0, 0.0, 4.0])
        );

    let mut env = Quadcopter::from_config(config)?;
    env.reset(42);

    println!("Environment Configuration:");
    println!("  Parallel environments: {}", NUM_ENVS);
    println!("  Physics frequency: 240 Hz");
    println!("  Control frequency: 30 Hz");
    println!("  Max episode steps: 300");
    println!();

    // Create visualization configuration
    let vis_config = VisualizationConfig::new()
        .with_size(1280, 720)
        .with_camera(
            CameraConfig::new()
                .with_position(6.0, 6.0, 5.0)
                .with_target(0.0, 0.0, 1.0)
                .with_auto_orbit(0.15)
                .with_orbit_radius(8.0)
        )
        .with_trajectory(
            TrajectoryConfig::new()
                .with_length(TRAJECTORY_LENGTH)
                .with_enabled(true)
        )
        .with_fps(30);

    // Create visualization window
    println!("Opening visualization window...");
    let mut window = RealtimeWindow::new(
        "Quadcopter SIMD Training - 8 Parallel Environments",
        vis_config,
    )?;

    // Create history buffer
    let mut history = RollingHistory::new(NUM_ENVS, TRAJECTORY_LENGTH, SNAPSHOT_CAPACITY);

    println!();
    println!("Controls:");
    println!("  1-8: Highlight specific drone");
    println!("  0: Clear highlight");
    println!("  T: Toggle trails");
    println!("  A: Toggle axes");
    println!("  O: Toggle orbit");
    println!("  R: Reset camera");
    println!("  +/-: Zoom");
    println!("  Arrows: Pan");
    println!("  ESC: Quit");
    println!();

    // Action generation with diversity
    let hover_action = rpm_to_action(HOVER_RPM);
    let mut actions = vec![hover_action; NUM_ENVS * 4];
    let mut rng = fastrand::Rng::with_seed(12345);

    // Simulation loop
    let mut step: u64 = 0;
    let start_time = std::time::Instant::now();
    let mut last_report = start_time;
    let mut frames_since_report = 0u64;

    println!("Starting simulation loop...");
    println!();

    while window.is_open() {
        // Generate diverse actions for interesting behavior
        // Each environment gets a slightly different perturbation pattern
        for i in 0..NUM_ENVS {
            let env_offset = i * 4;
            let t = step as f32 * 0.05 + i as f32 * 0.7;

            for j in 0..4 {
                // Sinusoidal perturbation + random noise
                let phase = t + j as f32 * 1.5;
                let sinusoidal = phase.sin() * 0.12;
                let random = (rng.f32() - 0.5) * 0.08;

                // Different environments get different behavior patterns
                let pattern = match i % 4 {
                    0 => sinusoidal,                           // Smooth oscillation
                    1 => sinusoidal * 1.5,                     // Larger amplitude
                    2 => if step % 50 < 25 { 0.1 } else { -0.1 }, // Step changes
                    _ => random * 2.0,                          // More random
                };

                actions[env_offset + j] = (hover_action + pattern + random).clamp(-1.0, 1.0);
            }
        }

        // Step environment
        env.step(&actions);
        step += 1;
        frames_since_report += 1;

        // Capture state snapshot
        let snapshot = MultiEnvSnapshot::capture(&env, step);
        history.push(snapshot);

        // Update visualization
        if !window.update(&history)? {
            break;
        }

        // Print stats every second
        let now = std::time::Instant::now();
        if now.duration_since(last_report).as_secs_f32() >= 1.0 {
            let elapsed = now.duration_since(last_report).as_secs_f32();
            let fps = frames_since_report as f32 / elapsed;
            let total_elapsed = now.duration_since(start_time).as_secs_f32();
            let sps = step as f32 / total_elapsed;

            // Get latest stats
            let stats = history.stats();

            println!(
                "Step {:6} | FPS: {:5.1} | SPS: {:6.0} | Reward: {:6.1} [{:6.1}, {:6.1}]",
                step, fps, sps, stats.mean_reward, stats.min_reward, stats.max_reward
            );

            last_report = now;
            frames_since_report = 0;
        }

        // Environment handles auto-reset internally via step()
    }

    // Final statistics
    let total_time = start_time.elapsed().as_secs_f64();
    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                     Showcase Complete                          ");
    println!("════════════════════════════════════════════════════════════════");
    println!("  Total simulation steps: {}", step);
    println!("  Total time: {:.2} seconds", total_time);
    println!("  Average steps/sec: {:.0}", step as f64 / total_time);
    println!("  Environment-steps/sec: {:.2e}", (step * NUM_ENVS as u64) as f64 / total_time);
    println!("════════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "render-realtime"))]
fn main() {
    eprintln!("This example requires the render-realtime feature.");
    eprintln!();
    eprintln!("Run with:");
    eprintln!("  cargo run --example visualization_showcase --release --features render-realtime");
}
