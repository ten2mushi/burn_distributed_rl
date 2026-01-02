//! Real-time window visualization using minifb.
//!
//! Provides a live display window for monitoring training progress.

use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};

use crate::renderer::{
    backends::ImageBackend, EntityMapConfig, PlotType, RenderError, RenderResult,
    RollingHistory, SpectrumConfig, TimelineConfig, WaterfallConfig,
};

/// Real-time visualization window.
pub struct RealtimeWindow {
    window: Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
    current_plot: PlotType,
    last_update: Instant,
    frame_duration: Duration,
    waterfall_config: WaterfallConfig,
    spectrum_config: SpectrumConfig,
    entity_map_config: EntityMapConfig,
    timeline_config: TimelineConfig,
}

impl RealtimeWindow {
    /// Create a new real-time window.
    ///
    /// # Arguments
    ///
    /// * `title` - Window title
    /// * `width` - Window width in pixels
    /// * `height` - Window height in pixels
    /// * `target_fps` - Target frame rate
    pub fn new(title: &str, width: usize, height: usize, target_fps: u32) -> RenderResult<Self> {
        let window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: true,
                scale_mode: minifb::ScaleMode::AspectRatioStretch,
                ..WindowOptions::default()
            },
        )
        .map_err(|e| RenderError::WindowCreation(e.to_string()))?;

        let buffer = vec![0u32; width * height];
        let frame_duration = Duration::from_secs_f64(1.0 / target_fps as f64);

        Ok(Self {
            window,
            buffer,
            width,
            height,
            current_plot: PlotType::Waterfall,
            last_update: Instant::now(),
            frame_duration,
            waterfall_config: WaterfallConfig::default(),
            spectrum_config: SpectrumConfig::default(),
            entity_map_config: EntityMapConfig::default(),
            timeline_config: TimelineConfig::default(),
        })
    }

    /// Check if the window is still open.
    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    /// Get the current plot type being displayed.
    pub fn current_plot(&self) -> PlotType {
        self.current_plot
    }

    /// Set the plot type to display.
    pub fn set_plot(&mut self, plot: PlotType) {
        self.current_plot = plot;
    }

    /// Update the window with the latest data.
    ///
    /// Returns `Ok(true)` if the window is still open, `Ok(false)` if closed.
    pub fn update(&mut self, history: &RollingHistory) -> RenderResult<bool> {
        if !self.is_open() {
            return Ok(false);
        }

        // Handle keyboard input for plot switching
        self.handle_input();

        // Rate limit updates
        let now = Instant::now();
        if now.duration_since(self.last_update) < self.frame_duration {
            return Ok(true);
        }
        self.last_update = now;

        // Handle window resize
        let (new_width, new_height) = self.window.get_size();
        if new_width != self.width || new_height != self.height {
            self.width = new_width;
            self.height = new_height;
            self.buffer.resize(new_width * new_height, 0);
        }

        // Render the current plot
        self.render_to_buffer(history)?;

        // Update window
        self.window
            .update_with_buffer(&self.buffer, self.width, self.height)
            .map_err(|e| RenderError::WindowCreation(e.to_string()))?;

        Ok(true)
    }

    /// Handle keyboard input for plot switching.
    fn handle_input(&mut self) {
        if self.window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
            self.current_plot = PlotType::Waterfall;
        } else if self.window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
            self.current_plot = PlotType::SpectrumSnapshot;
        } else if self.window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
            self.current_plot = PlotType::EntityMap;
        } else if self.window.is_key_pressed(Key::Key4, minifb::KeyRepeat::No) {
            self.current_plot = PlotType::FrequencyTimeline;
        }
    }

    /// Render the current plot to the internal buffer.
    fn render_to_buffer(&mut self, history: &RollingHistory) -> RenderResult<()> {
        if history.is_empty() {
            // Clear to dark background
            self.buffer.fill(0xFF141420);
            return Ok(());
        }

        let backend = ImageBackend::new(self.width as u32, self.height as u32);

        // Render to RGB bytes
        let rgb_data = match self.current_plot {
            PlotType::Waterfall => {
                let matrix = history.psd_matrix();
                if matrix.num_time_steps == 0 {
                    self.buffer.fill(0xFF141420);
                    return Ok(());
                }
                backend.render_waterfall_to_buffer(&matrix, &self.waterfall_config)?
            }
            PlotType::SpectrumSnapshot => {
                let snapshot = history.latest().ok_or(RenderError::EmptyHistory)?;
                use crate::renderer::backends::RenderBackend;
                backend.render_spectrum(snapshot, &self.spectrum_config)?
            }
            PlotType::EntityMap => {
                let snapshot = history.latest().ok_or(RenderError::EmptyHistory)?;
                use crate::renderer::backends::RenderBackend;
                backend.render_entity_map(snapshot, &self.entity_map_config)?
            }
            PlotType::FrequencyTimeline => {
                let traces = history.agent_freq_traces();
                let collisions = history.find_collisions(self.timeline_config.collision_threshold_hz);
                use crate::renderer::backends::RenderBackend;
                backend.render_timeline(&traces, &collisions, &self.timeline_config)?
            }
        };

        // Convert RGB to ARGB for minifb
        self.rgb_to_argb(&rgb_data);

        Ok(())
    }

    /// Convert RGB buffer to ARGB buffer for minifb.
    fn rgb_to_argb(&mut self, rgb: &[u8]) {
        let pixels = self.width * self.height;
        for i in 0..pixels.min(rgb.len() / 3) {
            let r = rgb[i * 3] as u32;
            let g = rgb[i * 3 + 1] as u32;
            let b = rgb[i * 3 + 2] as u32;
            self.buffer[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }

    /// Set waterfall configuration.
    pub fn set_waterfall_config(&mut self, config: WaterfallConfig) {
        self.waterfall_config = config;
    }

    /// Set spectrum configuration.
    pub fn set_spectrum_config(&mut self, config: SpectrumConfig) {
        self.spectrum_config = config;
    }

    /// Set entity map configuration.
    pub fn set_entity_map_config(&mut self, config: EntityMapConfig) {
        self.entity_map_config = config;
    }

    /// Set timeline configuration.
    pub fn set_timeline_config(&mut self, config: TimelineConfig) {
        self.timeline_config = config;
    }

    /// Get window dimensions.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Check if a key was pressed (single press, no repeat).
    ///
    /// Useful for band selection or other user-triggered actions in examples.
    pub fn is_key_pressed(&self, key: Key) -> bool {
        self.window.is_key_pressed(key, minifb::KeyRepeat::No)
    }

    /// Check if a key is currently held down.
    pub fn is_key_down(&self, key: Key) -> bool {
        self.window.is_key_down(key)
    }
}
