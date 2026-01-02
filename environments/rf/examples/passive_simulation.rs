//! Passive RF Environment Simulation with Live Visualization
//!
//! This example demonstrates:
//! - Creating an RF environment with passive entities only (no jammers/CRs)
//! - Running the simulation loop
//! - Real-time window visualization using minifb
//! - GIF export of the waterfall spectrogram
//!
//! # Running
//!
//! ```bash
//! cargo run --example passive_simulation --features render,render-realtime,render-gif
//! ```
//!
//! # Controls
//!
//! - Press 1: Waterfall spectrogram
//! - Press 2: Spectrum snapshot
//! - Press 3: Entity position map
//! - Press 4: Frequency timeline
//! - Press ESC: Close window

use rf_environment::{EntityConfig, EntityType as EnvEntityType, RFWorld, RFWorldConfig};

#[cfg(feature = "render")]
use rf_environment::renderer::{
    EnvSnapshot, EntitySnapshot, EntityType, PlotType, RollingHistory, WaterfallConfig,
};

#[cfg(feature = "render-realtime")]
use rf_environment::renderer::RealtimeWindow;

#[cfg(feature = "render-gif")]
use rf_environment::renderer::GifRecorder;

/// Convert environment EntityType to renderer EntityType.
#[cfg(feature = "render")]
fn map_entity_type(env_type: EnvEntityType) -> EntityType {
    match env_type {
        EnvEntityType::TVStation => EntityType::TVStation,
        EnvEntityType::FMRadio => EntityType::FMRadio,
        EnvEntityType::LTETower => EntityType::LTETower,
        EnvEntityType::WiFiAP => EntityType::WiFiAP,
        EnvEntityType::SBandRadar => EntityType::SBandRadar,
        EnvEntityType::WeatherRadar => EntityType::XBandRadar,
        EnvEntityType::DroneAnalog => EntityType::DroneAnalog,
        EnvEntityType::DroneDigital => EntityType::DroneDigital,
        EnvEntityType::Bluetooth => EntityType::FHSS,
        EnvEntityType::Vehicle => EntityType::Unknown,
    }
}

/// Create a snapshot from the current environment state.
#[cfg(feature = "render")]
fn capture_snapshot(env: &RFWorld, env_idx: usize, step: u64) -> EnvSnapshot {
    let config = env.config();
    let entities = env.entities();

    // Get PSD for this environment (already in linear watts)
    let psd_linear: Vec<f32> = env.get_observation(env_idx).to_vec();

    // Build frequency bin centers
    let freq_bins: Vec<f32> = (0..config.num_freq_bins)
        .map(|i| config.bin_to_freq(i))
        .collect();

    // Create snapshot
    let mut snapshot = EnvSnapshot::new(
        step,
        psd_linear,
        freq_bins,
        config.effective_noise_floor(),
        (config.world_size.0, config.world_size.1),
        (config.freq_min, config.freq_max),
    );

    // Add entities from this environment
    let max_entities = config.max_entities;
    for entity_idx in 0..max_entities {
        let flat_idx = env_idx * max_entities + entity_idx;

        // Check bounds
        if flat_idx >= entities.active.len() {
            break;
        }

        // Skip inactive entities
        let active = entities.active[flat_idx] != 0;
        if !active {
            continue;
        }

        let entity_type = EnvEntityType::from(entities.entity_type[flat_idx]);

        let entity = EntitySnapshot::new(
            map_entity_type(entity_type),
            entities.x[flat_idx],
            entities.y[flat_idx],
            entities.center_freq[flat_idx],
            entities.bandwidth[flat_idx],
            entities.power_dbm[flat_idx],
        )
        .with_velocity(entities.vx[flat_idx], entities.vy[flat_idx])
        .with_active(active);

        snapshot.add_entity(entity);
    }

    snapshot
}

#[cfg(all(feature = "render", feature = "render-realtime", feature = "render-gif"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RF Environment Passive Simulation ===\n");

    // Configure passive entities (no jammers or cognitive radios)
    let entity_config = EntityConfig::new()
        .with_tv_stations(2)
        .with_fm_radios(2)
        .with_lte_towers(3)
        .with_wifi_aps(5)
        .with_bluetooth(3)
        .with_sband_radars(1)
        .with_weather_radars(1)
        .with_drones(1, 1) // 1 analog, 1 digital drone
        .with_vehicles(2);

    // Configure the environment
    let config = RFWorldConfig::new()
        .with_num_envs(8) // 8 parallel environments (SIMD)
        .with_freq_bins(256)
        .with_freq_range(400e6, 6e9) // 400 MHz to 6 GHz
        .with_world_size(1000.0, 1000.0, 100.0)
        .with_max_entities(32)
        .with_physics_freq(1000)
        .with_ctrl_freq(100)
        .with_max_steps(2000)
        .with_seed(42)
        .with_entity_config(entity_config)
        .with_num_jammers(0) // No jammers
        .with_num_crs(0) // No cognitive radios
        .build();

    println!("Environment Configuration:");
    println!(
        "  Frequency range: {:.0} MHz - {:.0} MHz",
        config.freq_min / 1e6,
        config.freq_max / 1e6
    );
    println!("  Frequency bins: {}", config.num_freq_bins);
    println!("  Resolution: {:.2} MHz/bin", config.freq_resolution() / 1e6);
    println!(
        "  World size: {}m x {}m",
        config.world_size.0, config.world_size.1
    );
    println!("  Max steps: {}", config.max_steps);
    println!("  Parallel envs: {}", config.num_envs);
    println!();

    // Create environment
    let mut env = RFWorld::new(config.clone());
    env.reset();

    println!(
        "Environment created with {} entities per env",
        env.entities().count_active(0)
    );

    // Create rolling history for renderer
    let mut history = RollingHistory::new(200);

    // Create real-time window
    println!("\nOpening visualization window...");
    println!("Controls: 1=Waterfall, 2=Spectrum, 3=EntityMap, 4=Timeline, ESC=Quit\n");

    let mut window = RealtimeWindow::new(
        "RF Environment - Passive Simulation",
        1024,
        768,
        30, // 30 FPS target
    )?;

    // Create GIF recorder for waterfall
    let mut gif_recorder = GifRecorder::new(640, 480, 10, PlotType::Waterfall);
    gif_recorder.set_max_frames(Some(200)); // ~20 seconds at 10 fps
    gif_recorder.set_waterfall_config(WaterfallConfig {
        psd_range_db: (-110.0, -30.0),
        ..WaterfallConfig::default()
    });

    // Also record entity map
    let mut entity_gif = GifRecorder::new(640, 480, 10, PlotType::EntityMap);
    entity_gif.set_max_frames(Some(200));

    // Simulation loop
    let mut step: u64 = 0;
    let env_to_visualize = 0; // Visualize first environment

    println!("Starting simulation loop...");

    while window.is_open() {
        // Step the environment (no actions since no agents)
        env.step();
        step += 1;

        // Capture snapshot
        let snapshot = capture_snapshot(&env, env_to_visualize, step);
        history.push(snapshot);

        // Update window
        if !window.update(&history)? {
            break; // Window was closed
        }

        // Record GIF frames every 10 steps
        if step % 10 == 0 {
            let _ = gif_recorder.add_frame(&history);
            let _ = entity_gif.add_frame(&history);
        }

        // Print progress every 100 steps
        if step % 100 == 0 {
            let stats = history.stats();
            println!(
                "Step {:5} | History: {} frames | PSD range: {:.1} to {:.1} dBm | GIF: {} frames",
                step,
                history.len(),
                stats.min_psd_dbm,
                stats.max_psd_dbm,
                gif_recorder.frame_count()
            );
        }

        // Check if environment is done
        if env.all_done() {
            println!("\nAll environments completed. Resetting...");
            env.reset();
        }

        // Stop after max steps for this demo
        if step >= 2000 {
            break;
        }
    }

    println!("\nSimulation complete after {} steps.", step);

    // Save GIF files
    let output_dir = std::path::Path::new("./renders");
    std::fs::create_dir_all(output_dir)?;

    let waterfall_path = output_dir.join("waterfall.gif");
    let entity_path = output_dir.join("entity_map.gif");

    if gif_recorder.frame_count() > 0 {
        println!(
            "Saving waterfall GIF ({} frames)...",
            gif_recorder.frame_count()
        );
        gif_recorder.save(&waterfall_path)?;
        println!("  Saved: {}", waterfall_path.display());
    }

    if entity_gif.frame_count() > 0 {
        println!(
            "Saving entity map GIF ({} frames)...",
            entity_gif.frame_count()
        );
        entity_gif.save(&entity_path)?;
        println!("  Saved: {}", entity_path.display());
    }

    println!("\nDone! Check the ./renders directory for outputs.");

    Ok(())
}

#[cfg(not(all(feature = "render", feature = "render-realtime", feature = "render-gif")))]
fn main() {
    eprintln!("This example requires the render, render-realtime, and render-gif features.");
    eprintln!("Run with: cargo run --example passive_simulation --features render,render-realtime,render-gif");
}
