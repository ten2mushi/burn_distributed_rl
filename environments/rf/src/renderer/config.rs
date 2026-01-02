//! Renderer configuration types.
//!
//! Provides comprehensive configuration for all visualization aspects
//! including plot types, backends, styling, and output formats.

use std::path::PathBuf;

use super::colormap::ColormapType;

/// Main renderer configuration.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Output directory for rendered images.
    pub output_dir: PathBuf,
    /// Environment index to visualize (for multi-env training).
    pub env_index: usize,
    /// Number of steps to retain in history for waterfall/timeline.
    pub history_capacity: usize,
    /// Render every N steps (0 = manual control).
    pub render_every: u64,
    /// Which plots to include.
    pub plots: Vec<PlotType>,
    /// Image dimensions (width, height).
    pub image_size: (u32, u32),
    /// Waterfall-specific configuration.
    pub waterfall: WaterfallConfig,
    /// Spectrum snapshot configuration.
    pub spectrum: SpectrumConfig,
    /// Entity map configuration.
    pub entity_map: EntityMapConfig,
    /// Timeline configuration.
    pub timeline: TimelineConfig,
    /// Backend to use for rendering.
    pub backend: BackendType,
    /// Enable verbose logging.
    pub verbose: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./renders"),
            env_index: 0,
            history_capacity: 200,
            render_every: 100,
            plots: vec![
                PlotType::Waterfall,
                PlotType::SpectrumSnapshot,
                PlotType::EntityMap,
                PlotType::FrequencyTimeline,
            ],
            image_size: (800, 600),
            waterfall: WaterfallConfig::default(),
            spectrum: SpectrumConfig::default(),
            entity_map: EntityMapConfig::default(),
            timeline: TimelineConfig::default(),
            backend: BackendType::Image,
            verbose: false,
        }
    }
}

impl RenderConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set output directory.
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Set environment index to visualize.
    pub fn env_index(mut self, idx: usize) -> Self {
        self.env_index = idx;
        self
    }

    /// Set history capacity for waterfall/timeline.
    pub fn history_capacity(mut self, capacity: usize) -> Self {
        self.history_capacity = capacity;
        self
    }

    /// Set render frequency (every N steps).
    pub fn render_every(mut self, n: u64) -> Self {
        self.render_every = n;
        self
    }

    /// Set which plots to render.
    pub fn plots(mut self, plots: Vec<PlotType>) -> Self {
        self.plots = plots;
        self
    }

    /// Set image dimensions.
    pub fn image_size(mut self, width: u32, height: u32) -> Self {
        self.image_size = (width, height);
        self
    }

    /// Set backend type.
    pub fn backend(mut self, backend: BackendType) -> Self {
        self.backend = backend;
        self
    }

    /// Enable verbose logging.
    pub fn verbose(mut self, enable: bool) -> Self {
        self.verbose = enable;
        self
    }

    /// Configure waterfall settings.
    pub fn with_waterfall(mut self, config: WaterfallConfig) -> Self {
        self.waterfall = config;
        self
    }

    /// Configure spectrum settings.
    pub fn with_spectrum(mut self, config: SpectrumConfig) -> Self {
        self.spectrum = config;
        self
    }

    /// Configure entity map settings.
    pub fn with_entity_map(mut self, config: EntityMapConfig) -> Self {
        self.entity_map = config;
        self
    }

    /// Configure timeline settings.
    pub fn with_timeline(mut self, config: TimelineConfig) -> Self {
        self.timeline = config;
        self
    }
}

/// Types of plots that can be rendered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlotType {
    /// Waterfall spectrogram (time-frequency heatmap).
    Waterfall,
    /// Instantaneous spectrum snapshot (PSD vs frequency).
    SpectrumSnapshot,
    /// Entity position map (2D bird's-eye view).
    EntityMap,
    /// Agent frequency timeline (frequency vs time).
    FrequencyTimeline,
}

impl PlotType {
    /// Get default filename suffix for this plot type.
    pub fn suffix(&self) -> &'static str {
        match self {
            PlotType::Waterfall => "waterfall",
            PlotType::SpectrumSnapshot => "spectrum",
            PlotType::EntityMap => "entity_map",
            PlotType::FrequencyTimeline => "timeline",
        }
    }

    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            PlotType::Waterfall => "Waterfall Spectrogram",
            PlotType::SpectrumSnapshot => "Spectrum Snapshot",
            PlotType::EntityMap => "Entity Position Map",
            PlotType::FrequencyTimeline => "Frequency Timeline",
        }
    }
}

/// Rendering backend types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendType {
    /// PNG image output via plotters.
    #[default]
    Image,
    /// ASCII terminal output.
    Terminal,
    /// HTML/JS dashboard.
    Html,
}

/// Waterfall plot configuration.
#[derive(Clone, Debug)]
pub struct WaterfallConfig {
    /// PSD display range in dBm (min, max).
    pub psd_range_db: (f32, f32),
    /// Colormap to use.
    pub colormap: ColormapType,
    /// Show colorbar legend.
    pub show_colorbar: bool,
    /// Show entity frequency markers.
    pub show_entity_markers: bool,
    /// Show agent frequency markers.
    pub show_agent_markers: bool,
    /// Frequency unit for axis labels.
    pub freq_unit: FreqUnit,
    /// Time axis shows steps or seconds.
    pub time_unit: TimeUnit,
    /// Seconds per step (for time conversion).
    pub seconds_per_step: f32,
}

impl Default for WaterfallConfig {
    fn default() -> Self {
        Self {
            psd_range_db: (-120.0, -40.0),
            colormap: ColormapType::Viridis,
            show_colorbar: true,
            show_entity_markers: false,
            show_agent_markers: true,
            freq_unit: FreqUnit::MHz,
            time_unit: TimeUnit::Steps,
            seconds_per_step: 0.01, // 100 Hz control
        }
    }
}

/// Spectrum snapshot configuration.
#[derive(Clone, Debug)]
pub struct SpectrumConfig {
    /// PSD display range in dBm (min, max).
    pub psd_range_db: (f32, f32),
    /// Show noise floor line.
    pub show_noise_floor: bool,
    /// Show entity frequency markers.
    pub show_entity_markers: bool,
    /// Show agent markers (jammer = red, CR = green).
    pub show_agent_markers: bool,
    /// Label the N strongest peaks.
    pub label_top_n_peaks: usize,
    /// Line color [R, G, B].
    pub line_color: [u8; 3],
    /// Fill under curve.
    pub fill_under_curve: bool,
    /// Frequency unit for axis labels.
    pub freq_unit: FreqUnit,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            psd_range_db: (-120.0, -40.0),
            show_noise_floor: true,
            show_entity_markers: true,
            show_agent_markers: true,
            label_top_n_peaks: 3,
            line_color: [0, 128, 255],
            fill_under_curve: true,
            freq_unit: FreqUnit::MHz,
        }
    }
}

/// Entity map configuration.
#[derive(Clone, Debug)]
pub struct EntityMapConfig {
    /// World size in meters (width, height) - if None, use from snapshot.
    pub world_size: Option<(f32, f32)>,
    /// Icon size in pixels.
    pub icon_size: u32,
    /// Show entity labels.
    pub show_labels: bool,
    /// Show velocity arrows for mobile entities.
    pub show_velocity_arrows: bool,
    /// Show coverage circles based on power.
    pub show_coverage_circles: bool,
    /// Show jammer beam cones.
    pub show_jammer_beams: bool,
    /// Show grid lines.
    pub show_grid: bool,
    /// Grid spacing in meters (if grid enabled).
    pub grid_spacing: f32,
    /// Background color [R, G, B].
    pub background_color: [u8; 3],
    /// Show legend.
    pub show_legend: bool,
    /// SINR threshold for CR "jammed" indication.
    pub sinr_threshold_db: f32,
}

impl Default for EntityMapConfig {
    fn default() -> Self {
        Self {
            world_size: None,
            icon_size: 12,
            show_labels: true,
            show_velocity_arrows: true,
            show_coverage_circles: false,
            show_jammer_beams: true,
            show_grid: true,
            grid_spacing: 100.0, // 100m grid
            background_color: [20, 20, 30],
            show_legend: true,
            sinr_threshold_db: 10.0,
        }
    }
}

/// Timeline configuration.
#[derive(Clone, Debug)]
pub struct TimelineConfig {
    /// Show jammer frequency traces.
    pub show_jammers: bool,
    /// Show CR frequency traces.
    pub show_crs: bool,
    /// Show entity frequency traces (for passive simulations).
    pub show_entities: bool,
    /// Show collision markers.
    pub show_collisions: bool,
    /// Collision overlap threshold in Hz.
    pub collision_threshold_hz: f32,
    /// Show bandwidth as shaded region.
    pub show_bandwidth_shading: bool,
    /// Jammer line color [R, G, B].
    pub jammer_color: [u8; 3],
    /// CR line color [R, G, B].
    pub cr_color: [u8; 3],
    /// Collision marker color [R, G, B].
    pub collision_color: [u8; 3],
    /// Frequency unit for axis labels.
    pub freq_unit: FreqUnit,
    /// Time axis shows steps or seconds.
    pub time_unit: TimeUnit,
    /// Seconds per step (for time conversion).
    pub seconds_per_step: f32,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            show_jammers: true,
            show_crs: true,
            show_entities: true, // Show entities when no agents present
            show_collisions: true,
            collision_threshold_hz: 1e6, // 1 MHz
            show_bandwidth_shading: true,
            jammer_color: [255, 60, 60],
            cr_color: [60, 200, 60],
            collision_color: [255, 255, 0],
            freq_unit: FreqUnit::MHz,
            time_unit: TimeUnit::Steps,
            seconds_per_step: 0.01,
        }
    }
}

/// Frequency unit for display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FreqUnit {
    Hz,
    KHz,
    #[default]
    MHz,
    GHz,
}

impl FreqUnit {
    /// Convert Hz to this unit.
    pub fn from_hz(&self, hz: f32) -> f32 {
        match self {
            FreqUnit::Hz => hz,
            FreqUnit::KHz => hz / 1e3,
            FreqUnit::MHz => hz / 1e6,
            FreqUnit::GHz => hz / 1e9,
        }
    }

    /// Get unit suffix.
    pub fn suffix(&self) -> &'static str {
        match self {
            FreqUnit::Hz => "Hz",
            FreqUnit::KHz => "kHz",
            FreqUnit::MHz => "MHz",
            FreqUnit::GHz => "GHz",
        }
    }

    /// Format a frequency value with unit.
    pub fn format(&self, hz: f32) -> String {
        format!("{:.1} {}", self.from_hz(hz), self.suffix())
    }
}

/// Time unit for display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimeUnit {
    #[default]
    Steps,
    Seconds,
    Milliseconds,
}

impl TimeUnit {
    /// Get unit suffix.
    pub fn suffix(&self) -> &'static str {
        match self {
            TimeUnit::Steps => "steps",
            TimeUnit::Seconds => "s",
            TimeUnit::Milliseconds => "ms",
        }
    }
}

/// Real-time window configuration (when render-realtime feature enabled).
#[derive(Clone, Debug)]
pub struct RealtimeConfig {
    /// Window title.
    pub title: String,
    /// Window dimensions (width, height).
    pub window_size: (usize, usize),
    /// Target frames per second.
    pub target_fps: u32,
    /// Initial plot to show.
    pub initial_plot: PlotType,
    /// Allow keyboard switching between plots.
    pub allow_plot_switching: bool,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            title: "RF Environment".to_string(),
            window_size: (800, 600),
            target_fps: 30,
            initial_plot: PlotType::Waterfall,
            allow_plot_switching: true,
        }
    }
}

/// GIF export configuration (when render-gif feature enabled).
#[derive(Clone, Debug)]
pub struct GifConfig {
    /// Output GIF dimensions (width, height).
    pub size: (u16, u16),
    /// Frames per second.
    pub fps: u16,
    /// Which plot to record.
    pub plot_type: PlotType,
    /// Maximum number of frames (None = unlimited).
    pub max_frames: Option<usize>,
    /// Repeat count (0 = infinite loop).
    pub repeat: u16,
}

impl Default for GifConfig {
    fn default() -> Self {
        Self {
            size: (640, 480),
            fps: 10,
            plot_type: PlotType::Waterfall,
            max_frames: Some(300), // 30 seconds at 10 fps
            repeat: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freq_unit_conversion() {
        let hz = 2.4e9; // 2.4 GHz

        assert!((FreqUnit::Hz.from_hz(hz) - 2.4e9).abs() < 1.0);
        assert!((FreqUnit::KHz.from_hz(hz) - 2.4e6).abs() < 1.0);
        assert!((FreqUnit::MHz.from_hz(hz) - 2400.0).abs() < 0.1);
        assert!((FreqUnit::GHz.from_hz(hz) - 2.4).abs() < 0.001);
    }

    #[test]
    fn test_config_builder() {
        let config = RenderConfig::new()
            .output_dir("./my_renders")
            .env_index(2)
            .history_capacity(500)
            .render_every(50)
            .image_size(1920, 1080);

        assert_eq!(config.output_dir, PathBuf::from("./my_renders"));
        assert_eq!(config.env_index, 2);
        assert_eq!(config.history_capacity, 500);
        assert_eq!(config.render_every, 50);
        assert_eq!(config.image_size, (1920, 1080));
    }

    #[test]
    fn test_plot_type_metadata() {
        assert_eq!(PlotType::Waterfall.suffix(), "waterfall");
        assert_eq!(PlotType::SpectrumSnapshot.name(), "Spectrum Snapshot");
    }
}
