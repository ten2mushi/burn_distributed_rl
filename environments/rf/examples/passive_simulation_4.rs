//! Real-Time High-Resolution RF Spectrum Visualization
//!
//! This example extends passive_simulation_3 with two major improvements:
//!
//! 1. **Real-Time Scrolling**: The waterfall Y-axis scrolls with wall-clock time,
//!    not simulation steps. Each row represents a fixed time interval (10ms).
//!
//! 2. **Band-Focused High-Resolution Views**: Press 5-0 to zoom into specific
//!    frequency bands with resolution fine enough to see signal characteristics.
//!
//! # Resolution Comparison
//!
//! | Band     | Span    | Bins | Resolution    | What You Can See              |
//! |----------|---------|------|---------------|-------------------------------|
//! | Overview | 6.5 GHz | 2048 | 3.16 MHz/bin  | Gross spectrum allocation     |
//! | WiFi 2.4 | 100 MHz | 512  | 195 kHz/bin   | Individual channel structure! |
//! | WiFi 5   | 700 MHz | 1024 | 683 kHz/bin   | Channel separation            |
//! | Cellular | 1.4 GHz | 1024 | 1.37 MHz/bin  | LTE carrier blocks            |
//! | VHF      | 270 MHz | 512  | 527 kHz/bin   | FM stations clearly           |
//! | Radar    | 2.0 GHz | 1024 | 1.95 MHz/bin  | Radar sweep patterns          |
//!
//! # Running
//!
//! ```bash
//! cargo run --example passive_simulation_4 --release -p rf_environment --features render,render-realtime
//! ```
//!
//! # Controls
//!
//! **Display Modes:**
//! - Press 1: Waterfall spectrogram (time-frequency heatmap)
//! - Press 2: Spectrum snapshot (instantaneous PSD)
//! - Press 3: Entity map (2D positions with velocity arrows)
//! - Press 4: Frequency timeline
//!
//! **Band Selection (recreates environment with optimal resolution):**
//! - Press 5: Overview (30 MHz - 6.5 GHz, full spectrum)
//! - Press 6: WiFi 2.4 GHz (2.4-2.5 GHz, HIGH RESOLUTION)
//! - Press 7: WiFi 5 GHz (5.15-5.85 GHz)
//! - Press 8: Cellular (700 MHz - 2.1 GHz)
//! - Press 9: VHF Broadcast (30-300 MHz, FM/TV)
//! - Press 0: Radar S-Band (2.0-4.0 GHz)
//!
//! - Press ESC: Close window

use std::time::{Duration, Instant};

use rf_environment::{EntityConfig, EntityType as EnvEntityType, RFWorld, RFWorldConfig};

#[cfg(feature = "render")]
use rf_environment::renderer::{
    ColormapType, EntityMapConfig, EntitySnapshot, EntityType, EnvSnapshot, FreqUnit,
    RollingHistory, SpectrumConfig, TimeUnit, WaterfallConfig,
};

#[cfg(feature = "render-realtime")]
use rf_environment::renderer::{Key, RealtimeWindow};

// ============================================================================
// Real-Time Configuration
// ============================================================================

/// Interval between captured snapshots (100 Hz = 10ms)
const SNAPSHOT_INTERVAL: Duration = Duration::from_millis(10);

/// Duration of visible history in seconds (4 seconds of waterfall)
const HISTORY_DURATION_SECONDS: f32 = 4.0;

/// Number of history frames to keep (duration / interval)
const HISTORY_CAPACITY: usize = (HISTORY_DURATION_SECONDS * 100.0) as usize; // 400 frames

/// Number of simulation steps to integrate for each snapshot (peak-hold).
///
/// # Why Peak-Hold?
///
/// Radar pulses are 1-2 μs long with 1-3 ms PRI (0.1% duty cycle).
/// A single simulation step (1ms) is unlikely to catch a pulse.
/// By running multiple steps and keeping the MAX power per bin,
/// we implement "peak-hold" detection like real spectrum analyzers.
///
/// With 10 integration steps over 10ms, we guarantee catching at least
/// one radar pulse per snapshot (since PRI is ~1ms).
const PEAK_HOLD_STEPS: usize = 10;

// ============================================================================
// Band Selection Mode
// ============================================================================

/// Frequency band modes for high-resolution viewing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(feature = "render")]
enum BandMode {
    /// Full spectrum overview: 30 MHz - 6.5 GHz
    Overview,
    /// WiFi 2.4 GHz ISM band: 2.4 - 2.5 GHz (HIGH RESOLUTION)
    Wifi24,
    /// WiFi 5 GHz UNII bands: 5.15 - 5.85 GHz
    Wifi5,
    /// Cellular bands: 700 MHz - 2.1 GHz
    Cellular,
    /// VHF Broadcast: 30 - 300 MHz (FM, TV, VHF marine)
    VhfBroadcast,
    /// Radar: 2.5 - 4.0 GHz (S-band, weather)
    Radar,
}

#[cfg(feature = "render")]
impl BandMode {
    /// Get the frequency range for this band mode.
    fn freq_range(&self) -> (f32, f32) {
        match self {
            Self::Overview => (30e6, 6.5e9),
            Self::Wifi24 => (2.4e9, 2.5e9),       // 100 MHz span
            Self::Wifi5 => (5.15e9, 5.85e9),      // 700 MHz span
            Self::Cellular => (700e6, 2.1e9),     // 1.4 GHz span
            Self::VhfBroadcast => (30e6, 300e6),  // 270 MHz span
            Self::Radar => (2.0e9, 4.0e9),        // 2 GHz span (full S-band)
        }
    }

    /// Get the number of frequency bins for this band mode.
    fn freq_bins(&self) -> usize {
        match self {
            Self::Overview => 2048,    // 3.16 MHz/bin
            Self::Wifi24 => 512,       // 195 kHz/bin - can see channel structure!
            Self::Wifi5 => 1024,       // 683 kHz/bin
            Self::Cellular => 1024,    // 1.37 MHz/bin
            Self::VhfBroadcast => 512, // 527 kHz/bin
            Self::Radar => 1024,       // 1.95 MHz/bin (2 GHz / 1024)
        }
    }

    /// Get the display name for this band mode.
    fn name(&self) -> &'static str {
        match self {
            Self::Overview => "Overview (30 MHz - 6.5 GHz)",
            Self::Wifi24 => "WiFi 2.4 GHz (2.4-2.5 GHz)",
            Self::Wifi5 => "WiFi 5 GHz (5.15-5.85 GHz)",
            Self::Cellular => "Cellular (700 MHz - 2.1 GHz)",
            Self::VhfBroadcast => "VHF Broadcast (30-300 MHz)",
            Self::Radar => "Radar S-Band (2.0-4.0 GHz)",
        }
    }

    /// Get the resolution for this band mode.
    fn resolution_khz(&self) -> f32 {
        let (f_min, f_max) = self.freq_range();
        let span = f_max - f_min;
        span / (self.freq_bins() as f32) / 1000.0
    }
}

// ============================================================================
// Entity Type Mapping
// ============================================================================

/// Convert environment EntityType to renderer EntityType.
#[cfg(feature = "render")]
fn map_entity_type(env_type: EnvEntityType) -> EntityType {
    match env_type {
        // Continuous
        EnvEntityType::TVStation => EntityType::TVStation,
        EnvEntityType::FMRadio => EntityType::FMRadio,
        EnvEntityType::AMRadio => EntityType::FMRadio,
        EnvEntityType::GNSS => EntityType::DSSS,
        EnvEntityType::GeoSatellite => EntityType::Unknown,
        // Bursty
        EnvEntityType::LTETower => EntityType::LTETower,
        EnvEntityType::WiFiAP => EntityType::WiFiAP,
        EnvEntityType::Bluetooth => EntityType::FHSS,
        EnvEntityType::LTEUplink => EntityType::LTETower,
        EnvEntityType::Zigbee => EntityType::DSSS,
        EnvEntityType::LoRaWAN => EntityType::DSSS,
        // Periodic
        EnvEntityType::SBandRadar => EntityType::SBandRadar,
        EnvEntityType::WeatherRadar => EntityType::XBandRadar,
        EnvEntityType::LBandRadar => EntityType::SBandRadar,
        EnvEntityType::ADSB => EntityType::Unknown,
        EnvEntityType::DME => EntityType::Unknown,
        // Mobile
        EnvEntityType::DroneAnalog => EntityType::DroneAnalog,
        EnvEntityType::DroneDigital => EntityType::DroneDigital,
        EnvEntityType::Vehicle => EntityType::Unknown,
        // Voice/PTT
        EnvEntityType::MaritimeVHF => EntityType::AmateurRadio,
        EnvEntityType::WalkieTalkie => EntityType::Unknown,
        EnvEntityType::GMRS => EntityType::AmateurRadio,
        EnvEntityType::P25Radio => EntityType::Unknown,
        EnvEntityType::AmateurRadio => EntityType::AmateurRadio,
        // Fallback
        _ => EntityType::Unknown,
    }
}

// ============================================================================
// Environment Creation
// ============================================================================

/// Create the entity configuration (same as passive_simulation_3).
#[cfg(feature = "render")]
fn create_entity_config() -> EntityConfig {
    EntityConfig::new()
        // CONTINUOUS (100% duty cycle) - 5 types
        .with_tv_stations(4)
        .with_fm_radios(5)
        .with_am_radios(3)
        .with_gnss(1)
        .with_geo_satellites(2)
        // BURSTY (Poisson arrivals) - 6 types
        .with_lte_towers(6)
        .with_wifi_aps(10)
        .with_bluetooth(8)
        .with_lte_uplink(4)
        .with_zigbee(5)
        .with_lorawan(3)
        // PERIODIC (deterministic pulses) - 5 types
        .with_sband_radars(2)
        .with_weather_radars(1)
        .with_lband_radars(1)
        .with_adsb(4)
        .with_dme(2)
        // MOBILE (Doppler effects, trajectory) - 3 types
        .with_drones(3, 3)
        .with_vehicles(5)
        // VOICE/PTT (Push-to-talk behavior) - 5 types
        .with_maritime_vhf(2)
        .with_walkie_talkie(4)
        .with_gmrs(2)
        .with_p25(2)
        .with_amateur_radio(3)
}

/// Create an RFWorld environment for the given band mode.
#[cfg(feature = "render")]
fn create_env_for_band(entity_config: &EntityConfig, band: BandMode) -> RFWorld {
    let (freq_min, freq_max) = band.freq_range();
    let freq_bins = band.freq_bins();

    let config = RFWorldConfig::new()
        .with_num_envs(8)
        .with_freq_bins(freq_bins)
        .with_freq_range(freq_min, freq_max)
        .with_world_size(3000.0, 3000.0, 300.0)
        .with_max_entities(128)
        .with_physics_freq(1000)
        .with_ctrl_freq(100)
        .with_max_steps(100000) // Longer for real-time
        .with_seed(42)
        .with_entity_config(entity_config.clone())
        .with_num_jammers(0)
        .with_num_crs(0)
        .build();

    RFWorld::new(config)
}

// ============================================================================
// Snapshot Capture
// ============================================================================

/// Create a snapshot with peak-hold PSD from the environment state.
///
/// The `peak_psd` parameter contains peak-held PSD values accumulated over
/// multiple simulation steps. This implements spectrum analyzer peak-hold
/// functionality to capture brief radar pulses.
///
/// The `elapsed_ms` parameter is used instead of simulation step to enable
/// real-time based scrolling.
#[cfg(feature = "render")]
fn capture_snapshot_with_peak_hold(
    env: &RFWorld,
    env_idx: usize,
    elapsed_ms: u64,
    peak_psd: &[f32],
) -> EnvSnapshot {
    let config = env.config();
    let entities = env.entities();

    // Build frequency bin centers
    let freq_bins: Vec<f32> = (0..config.num_freq_bins)
        .map(|i| config.bin_to_freq(i))
        .collect();

    // Create snapshot with peak-held PSD and elapsed time
    let mut snapshot = EnvSnapshot::new(
        elapsed_ms, // Using step field to store timestamp in ms
        peak_psd.to_vec(),
        freq_bins,
        config.effective_noise_floor(),
        (config.world_size.0, config.world_size.1),
        (config.freq_min, config.freq_max),
    );

    // Add entities from this environment
    let max_entities = config.max_entities;
    for entity_idx in 0..max_entities {
        let flat_idx = env_idx * max_entities + entity_idx;

        if flat_idx >= entities.active.len() {
            break;
        }

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

/// Step simulation and accumulate peak-hold PSD.
///
/// Runs `PEAK_HOLD_STEPS` simulation steps, tracking the maximum PSD value
/// per frequency bin. This ensures brief radar pulses are captured even
/// when the visualization sample rate is lower than the pulse rate.
///
/// Returns `(peak_psd, did_reset)` where `did_reset` indicates if episode reset occurred.
#[cfg(feature = "render")]
fn step_with_peak_hold(
    env: &mut RFWorld,
    env_idx: usize,
    step: &mut u64,
) -> (Vec<f32>, bool) {
    let config = env.config();
    let num_bins = config.num_freq_bins;

    // Initialize peak buffer with zeros
    let mut peak_psd = vec![0.0_f32; num_bins];
    let mut did_reset = false;

    // Run multiple steps, tracking max per bin
    for _ in 0..PEAK_HOLD_STEPS {
        env.step();
        *step += 1;

        // Get current PSD
        let current_psd = env.get_observation(env_idx);

        // Update peak values
        for (peak, &current) in peak_psd.iter_mut().zip(current_psd.iter()) {
            *peak = peak.max(current);
        }

        // Handle episode reset
        if env.all_done() {
            env.reset();
            did_reset = true;
        }
    }

    (peak_psd, did_reset)
}

// ============================================================================
// Display Helpers
// ============================================================================

/// Print the controls help.
fn print_controls() {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      KEYBOARD CONTROLS                        ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ DISPLAY MODES:                                                ║");
    println!("║   1 = Waterfall spectrogram (time-frequency)                  ║");
    println!("║   2 = Spectrum snapshot (instantaneous PSD)                   ║");
    println!("║   3 = Entity map (2D positions)                               ║");
    println!("║   4 = Frequency timeline                                      ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ BAND SELECTION (recreates environment with optimal resolution):║");
    println!("║   5 = Overview (30 MHz - 6.5 GHz, 3.16 MHz/bin)               ║");
    println!("║   6 = WiFi 2.4 GHz (2.4-2.5 GHz, 195 kHz/bin) ★ HIGH RES      ║");
    println!("║   7 = WiFi 5 GHz (5.15-5.85 GHz, 683 kHz/bin)                 ║");
    println!("║   8 = Cellular (700 MHz - 2.1 GHz, 1.37 MHz/bin)              ║");
    println!("║   9 = VHF Broadcast (30-300 MHz, 527 kHz/bin)                 ║");
    println!("║   0 = Radar S-Band (2.0-4.0 GHz, 1.95 MHz/bin)                ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║   ESC = Close window                                          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
}

/// Print current band configuration.
#[cfg(feature = "render")]
fn print_band_info(band: BandMode) {
    let (freq_min, freq_max) = band.freq_range();
    let resolution_khz = band.resolution_khz();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║ Current Band: {:<47} ║", band.name());
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Frequency Range: {:.2} MHz - {:.2} GHz                          ║",
        freq_min / 1e6,
        freq_max / 1e9
    );
    println!("║ Frequency Bins:  {}                                          ║", band.freq_bins());
    println!(
        "║ Resolution:      {:.1} kHz/bin                                 ║",
        resolution_khz
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");
}

// ============================================================================
// Main
// ============================================================================

#[cfg(all(feature = "render", feature = "render-realtime"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║   RF ENVIRONMENT: REAL-TIME HIGH-RESOLUTION VISUALIZATION     ║");
    println!("║                                                               ║");
    println!("║   Features:                                                   ║");
    println!("║     • Real-time based waterfall scrolling                     ║");
    println!("║     • Band-focused high-resolution views                      ║");
    println!("║     • 2560x1440 ultra-wide display                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    print_controls();

    // Create entity configuration
    let entity_config = create_entity_config();
    println!("Entity Configuration: {} total entities", entity_config.total_entities());

    // Initial band mode
    let mut current_band = BandMode::Overview;
    print_band_info(current_band);

    // Create initial environment
    let mut env = create_env_for_band(&entity_config, current_band);
    env.reset();

    // Create rolling history for renderer (4 seconds at 100 Hz)
    let mut history = RollingHistory::new(HISTORY_CAPACITY);

    // Create real-time window (2560x1440 ultra-wide)
    println!("\nOpening visualization window...");
    let mut window = RealtimeWindow::new(
        "RF Environment - Real-Time High-Resolution",
        2560,
        1440,
        60, // 60 FPS for smoother real-time scrolling
    )?;

    // Configure waterfall with time-based display
    window.set_waterfall_config(WaterfallConfig {
        psd_range_db: (-140.0, -20.0),
        colormap: ColormapType::Inferno,
        show_colorbar: true,
        show_entity_markers: true,
        show_agent_markers: false,
        freq_unit: FreqUnit::GHz,
        time_unit: TimeUnit::Milliseconds, // Time-based display
        seconds_per_step: 0.001,           // 1 ms per "step" (our repurposed time field)
        ..Default::default()
    });

    // Configure spectrum snapshot
    window.set_spectrum_config(SpectrumConfig {
        psd_range_db: (-140.0, -20.0),
        show_noise_floor: true,
        show_entity_markers: true,
        show_agent_markers: false,
        label_top_n_peaks: 15,
        fill_under_curve: true,
        freq_unit: FreqUnit::GHz,
        ..Default::default()
    });

    // Configure entity map
    let config = env.config();
    window.set_entity_map_config(EntityMapConfig {
        world_size: Some((config.world_size.0, config.world_size.1)),
        icon_size: 8,
        show_labels: true,
        show_velocity_arrows: true,
        show_coverage_circles: false,
        show_jammer_beams: false,
        show_grid: true,
        grid_spacing: 500.0,
        background_color: [10, 15, 25],
        show_legend: true,
        sinr_threshold_db: 10.0,
    });

    // Real-time clock management
    let mut sim_start = Instant::now();
    let mut last_snapshot_time = Instant::now();
    let mut accumulated_sim_time = Duration::ZERO;
    let sim_step_duration = Duration::from_micros(1000); // 1ms per simulation step

    let mut step: u64 = 0;
    let env_to_visualize = 0;

    println!("\nStarting real-time simulation loop...");
    println!("Waterfall scrolls at real wall-clock time (1 row = 10ms)");
    println!();

    while window.is_open() {
        let now = Instant::now();

        // ====================================================================
        // Check for band selection key presses
        // ====================================================================
        let new_band = if window.is_key_pressed(Key::Key5) {
            Some(BandMode::Overview)
        } else if window.is_key_pressed(Key::Key6) {
            Some(BandMode::Wifi24)
        } else if window.is_key_pressed(Key::Key7) {
            Some(BandMode::Wifi5)
        } else if window.is_key_pressed(Key::Key8) {
            Some(BandMode::Cellular)
        } else if window.is_key_pressed(Key::Key9) {
            Some(BandMode::VhfBroadcast)
        } else if window.is_key_pressed(Key::Key0) {
            Some(BandMode::Radar)
        } else {
            None
        };

        // Handle band change
        if let Some(band) = new_band {
            if band != current_band {
                println!("\nSwitching to {:?}...", band);
                current_band = band;

                // Recreate environment with new frequency range and resolution
                env = create_env_for_band(&entity_config, current_band);
                env.reset();

                // Update entity map config for new world
                let config = env.config();
                window.set_entity_map_config(EntityMapConfig {
                    world_size: Some((config.world_size.0, config.world_size.1)),
                    icon_size: 8,
                    show_labels: true,
                    show_velocity_arrows: true,
                    show_coverage_circles: false,
                    show_jammer_beams: false,
                    show_grid: true,
                    grid_spacing: 500.0,
                    background_color: [10, 15, 25],
                    show_legend: true,
                    sinr_threshold_db: 10.0,
                });

                // Clear history (old data incompatible with new freq resolution)
                history.clear();

                // Reset timing
                sim_start = Instant::now();
                last_snapshot_time = Instant::now();
                accumulated_sim_time = Duration::ZERO;
                step = 0;

                print_band_info(current_band);
            }
        }

        // ====================================================================
        // Step simulation with peak-hold to catch brief radar pulses
        // ====================================================================
        // Check if it's time to capture a snapshot
        if now.duration_since(last_snapshot_time) >= SNAPSHOT_INTERVAL {
            // Run PEAK_HOLD_STEPS simulation steps, tracking maximum PSD per bin
            // This ensures brief radar pulses (1-2 μs) are captured even with
            // visualization rates much lower than pulse rates
            let (peak_psd, did_reset) = step_with_peak_hold(&mut env, env_to_visualize, &mut step);

            // Track accumulated simulation time
            accumulated_sim_time += sim_step_duration * PEAK_HOLD_STEPS as u32;

            if did_reset {
                println!("\nEpisode complete. Resetting...");
            }

            // Capture snapshot using peak-held PSD values
            let elapsed_ms = now.duration_since(sim_start).as_millis() as u64;
            let snapshot = capture_snapshot_with_peak_hold(&env, env_to_visualize, elapsed_ms, &peak_psd);
            history.push(snapshot);
            last_snapshot_time = now;
        }

        // ====================================================================
        // Update window
        // ====================================================================
        if !window.update(&history)? {
            break;
        }

        // Print progress every 5 seconds
        let elapsed_secs = now.duration_since(sim_start).as_secs();
        if elapsed_secs > 0 && elapsed_secs % 5 == 0 && now.duration_since(last_snapshot_time).as_millis() < 100 {
            let stats = history.stats();
            let active_count = env.entities().count_active(env_to_visualize);
            println!(
                "Time {:5}s | Step {:8} | Active: {:3} entities | PSD: {:.1} to {:.1} dBm",
                elapsed_secs,
                step,
                active_count,
                stats.min_psd_dbm,
                stats.max_psd_dbm,
            );
        }
    }

    let elapsed = sim_start.elapsed();
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!(
        "║ Simulation complete: {:6} steps in {:.1} seconds              ║",
        step,
        elapsed.as_secs_f32()
    );
    println!(
        "║ Effective rate: {:.1} steps/second                            ║",
        step as f32 / elapsed.as_secs_f32()
    );
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(all(feature = "render", feature = "render-realtime")))]
fn main() {
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║ ERROR: Missing required features                              ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════╣");
    eprintln!("║ This example requires: render, render-realtime                ║");
    eprintln!("║                                                               ║");
    eprintln!("║ Run with:                                                     ║");
    eprintln!("║   cargo run --example passive_simulation_4 --release \\        ║");
    eprintln!("║     -p rf_environment --features render,render-realtime       ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");
}
