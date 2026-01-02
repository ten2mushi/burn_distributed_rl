//! GIF export for animated training visualization.
//!
//! Records frames during training and exports to animated GIF.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use gif::{Encoder, Frame, Repeat};

use crate::renderer::{
    backends::ImageBackend, EntityMapConfig, PlotType, RenderError, RenderResult, RollingHistory,
    SpectrumConfig, TimelineConfig, WaterfallConfig,
};

/// GIF recorder for animated visualizations.
pub struct GifRecorder {
    frames: Vec<RgbaFrame>,
    width: u16,
    height: u16,
    frame_delay: u16,
    plot_type: PlotType,
    max_frames: Option<usize>,
    waterfall_config: WaterfallConfig,
    spectrum_config: SpectrumConfig,
    entity_map_config: EntityMapConfig,
    timeline_config: TimelineConfig,
}

/// Internal frame storage.
struct RgbaFrame {
    data: Vec<u8>,
}

impl GifRecorder {
    /// Create a new GIF recorder.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `fps` - Frames per second (determines playback speed)
    /// * `plot_type` - Which plot to record
    pub fn new(width: u16, height: u16, fps: u16, plot_type: PlotType) -> Self {
        // GIF frame delay is in centiseconds (1/100th of a second)
        let frame_delay = (100 / fps.max(1)).max(1);

        Self {
            frames: Vec::new(),
            width,
            height,
            frame_delay,
            plot_type,
            max_frames: Some(300), // Default: 30 seconds at 10 fps
            waterfall_config: WaterfallConfig::default(),
            spectrum_config: SpectrumConfig::default(),
            entity_map_config: EntityMapConfig::default(),
            timeline_config: TimelineConfig::default(),
        }
    }

    /// Set maximum number of frames to record.
    ///
    /// Use `None` for unlimited (be careful with memory!).
    pub fn set_max_frames(&mut self, max: Option<usize>) {
        self.max_frames = max;
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

    /// Get the current frame count.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Check if recorder has reached max frames.
    pub fn is_full(&self) -> bool {
        if let Some(max) = self.max_frames {
            self.frames.len() >= max
        } else {
            false
        }
    }

    /// Add a frame from the current history state.
    ///
    /// Returns `Ok(true)` if frame was added, `Ok(false)` if at capacity.
    pub fn add_frame(&mut self, history: &RollingHistory) -> RenderResult<bool> {
        if self.is_full() {
            return Ok(false);
        }

        if history.is_empty() {
            return Ok(false);
        }

        let backend = ImageBackend::new(self.width as u32, self.height as u32);

        // Render to RGB bytes
        let rgb_data = match self.plot_type {
            PlotType::Waterfall => {
                let matrix = history.psd_matrix();
                if matrix.num_time_steps == 0 {
                    return Ok(false);
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

        // Convert RGB to RGBA
        let rgba_data = rgb_to_rgba(&rgb_data);

        self.frames.push(RgbaFrame { data: rgba_data });

        Ok(true)
    }

    /// Save recorded frames to a GIF file.
    pub fn save(&self, path: impl AsRef<Path>) -> RenderResult<()> {
        if self.frames.is_empty() {
            return Err(RenderError::EmptyHistory);
        }

        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);

        let mut encoder = Encoder::new(writer, self.width, self.height, &[])
            .map_err(|e| RenderError::GifEncoding(e.to_string()))?;

        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| RenderError::GifEncoding(e.to_string()))?;

        for frame_data in &self.frames {
            let mut frame = Frame::from_rgba_speed(
                self.width,
                self.height,
                &mut frame_data.data.clone(),
                10, // Speed: 1-30, lower = better quality but slower
            );
            frame.delay = self.frame_delay;

            encoder
                .write_frame(&frame)
                .map_err(|e| RenderError::GifEncoding(e.to_string()))?;
        }

        Ok(())
    }

    /// Clear all recorded frames.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Get estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.frames.len() * (self.width as usize * self.height as usize * 4)
    }
}

/// Convert RGB buffer to RGBA buffer.
fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let pixels = rgb.len() / 3;
    let mut rgba = Vec::with_capacity(pixels * 4);

    for i in 0..pixels {
        rgba.push(rgb[i * 3]);     // R
        rgba.push(rgb[i * 3 + 1]); // G
        rgba.push(rgb[i * 3 + 2]); // B
        rgba.push(255);            // A
    }

    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gif_recorder_creation() {
        let recorder = GifRecorder::new(640, 480, 10, PlotType::Waterfall);
        assert_eq!(recorder.frame_count(), 0);
        assert!(!recorder.is_full());
    }

    #[test]
    fn test_frame_delay_calculation() {
        // 10 fps -> 10 centiseconds per frame
        let recorder = GifRecorder::new(100, 100, 10, PlotType::Waterfall);
        assert_eq!(recorder.frame_delay, 10);

        // 20 fps -> 5 centiseconds per frame
        let recorder = GifRecorder::new(100, 100, 20, PlotType::Waterfall);
        assert_eq!(recorder.frame_delay, 5);
    }

    #[test]
    fn test_rgb_to_rgba() {
        let rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // Red, Green, Blue
        let rgba = rgb_to_rgba(&rgb);

        assert_eq!(rgba.len(), 12);
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]); // Red + Alpha
        assert_eq!(&rgba[4..8], &[0, 255, 0, 255]); // Green + Alpha
        assert_eq!(&rgba[8..12], &[0, 0, 255, 255]); // Blue + Alpha
    }
}
