# RF Environment

SIMD-optimized radio frequency spectrum simulation for reinforcement learning.

## Features

- **SIMD Optimization**: 8-wide f32 vector operations via `std::simd`
- **Type-Safe Physics**: Compile-time guarantees on physical correctness
- **Multi-Agent Support**: Jammers vs. Cognitive Radios adversarial scenarios
- **Rich Entity System**: TV, FM, LTE, WiFi, Bluetooth, Radar, Drones, etc.
- **ITU-R Compliance**: Man-made noise (P.372), rain attenuation (P.838)
- **Visualization**: Real-time waterfall, spectrum, entity maps, GIF export

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rf_environment = { path = "../environments/rf" }

# Optional: Enable rendering
rf_environment = { path = "../environments/rf", features = ["render", "render-realtime"] }
```

**Note**: Requires nightly Rust for `portable_simd`:

## Quick Start

### Basic Environment

```rust
use rf_environment::{RFWorld, RFWorldConfig, EntityConfig};

fn main() {
    // Configure entities
    let entity_config = EntityConfig::new()
        .with_tv_stations(2)
        .with_lte_towers(3)
        .with_wifi_aps(5)
        .with_drones(1, 1);

    // Configure environment
    let config = RFWorldConfig::new()
        .with_num_envs(8)              // SIMD batch size (must be multiple of 8)
        .with_freq_bins(256)           // PSD resolution
        .with_freq_range(400e6, 6e9)   // 400 MHz to 6 GHz
        .with_world_size(1000.0, 1000.0, 100.0)
        .with_max_steps(1000)
        .with_entity_config(entity_config)
        .build();

    // Create and run
    let mut env = RFWorld::new(config);
    env.reset();

    for _ in 0..100 {
        let state = env.step();
        let observations = env.get_observations();
        // observations: [num_envs × num_freq_bins] PSD values
    }
}
```


## Physics Models

### Path Loss Models

| Model | Frequency Range | Distance Range | Use Case |
|-------|-----------------|----------------|----------|
| FSPL | Any | Any | Free-space, LOS |
| Log-Distance | Any | Indoor/outdoor | General purpose |
| Hata Urban | 150-1500 MHz | 1-20 km | Urban macro cells |
| COST-231 | 1500-2000 MHz | 1-20 km | Urban micro cells |
| Ground Wave | VLF/LF | Long range | AM broadcast |
| Waveguide | VLF (3-30 kHz) | Worldwide | Submarine comms |

### Fading Models

| Model | Distribution | Use Case | Unit Mean? |
|-------|--------------|----------|------------|
| Rayleigh | Exponential power | NLOS multipath | ✓ Proven |
| Rician | Non-central χ² | LOS + multipath | ✓ Proven |
| Log-Normal | Log-normal | Shadowing | ✓ Corrected |
| Composite | Product | Combined effects | ✓ By construction |

### Noise Models

| Model | Standard | Use Case |
|-------|----------|----------|
| Thermal | Johnson-Nyquist | Baseline noise floor |
| ITU Man-Made | ITU-R P.372 | Urban/suburban noise |

## Entity Types

### Continuous (100% duty cycle)
- TV Stations, FM Radio, AM Radio
- GNSS satellites, Geostationary beacons

### Bursty (Poisson arrivals)
- LTE towers/uplink, WiFi, Bluetooth
- Zigbee, LoRaWAN

### Periodic (Pulsed)
- S-Band radars, Weather radars, L-Band radars
- ADS-B transponders, DME navigation

### Mobile (Doppler effects)
- Drones (analog/digital video)
- Vehicles (V2X)

### Voice (Push-to-talk)
- Maritime VHF, Walkie-talkies, GMRS
- P25, Amateur radio

## Configuration Presets

```rust
use rf_environment::{RFWorldConfig, EntityConfig};

// Dense urban scenario
let config = RFWorldConfig::dense_urban();

// Rural scenario
let config = RFWorldConfig::rural();

// Fast training (smaller state space)
let config = RFWorldConfig::fast_training();

// 2.4 GHz ISM band focus
let config = RFWorldConfig::ism_2_4ghz();

// Drone detection scenario
let config = RFWorldConfig::drone_detection();

// Entity presets
let entities = EntityConfig::dense_urban();
let entities = EntityConfig::rural();
let entities = EntityConfig::minimal();
```

## Rendering & Visualization

Enable with features:

```toml
rf_environment = { features = ["render", "render-realtime", "render-gif"] }
```

```rust
use rf_environment::renderer::{
    RealtimeWindow, RollingHistory, GifRecorder, PlotType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut history = RollingHistory::new(200);
    let mut window = RealtimeWindow::new("RF Sim", 1024, 768, 30)?;
    let mut gif = GifRecorder::new(640, 480, 10, PlotType::Waterfall);

    while window.is_open() {
        // ... simulation step ...
        history.push(snapshot);
        window.update(&history)?;
        gif.add_frame(&history)?;
    }

    gif.save("waterfall.gif")?;
    Ok(())
}
```

**Plot Types**:
- `Waterfall`: Spectrogram over time
- `Spectrum`: Instantaneous PSD
- `EntityMap`: Entity positions in 2D space
- `Timeline`: Frequency usage over time

## Performance

The crate uses several optimizations:

1. **SIMD**: 8-wide f32 vectors for parallel computation
2. **SoA Layout**: Cache-friendly struct-of-arrays for entities
3. **Batch Processing**: All environments step together
4. **Lazy Evaluation**: PSD only computed when needed

## Testing

```bash
# Run all tests
cargo test

# Run with SIMD features
cargo test --features simd

# Run physics verification tests (100k samples)
cargo test physics_tests -- --ignored

# Run specific test
cargo test test_shadowing_corrected_unit_mean
```