//! Real-time window visualization using minifb.

use super::backends::{ImageBackend, RenderError, RenderResult};
use super::config::{CameraConfig, VisualizationConfig};
use super::history::RollingHistory;
use super::projection::{orbit_camera_position, PerspectiveProjection};

use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};

/// Real-time visualization window.
pub struct RealtimeWindow {
    /// minifb window handle.
    window: Window,
    /// ARGB pixel buffer for minifb.
    buffer: Vec<u32>,
    /// Image backend for rendering.
    backend: ImageBackend,
    /// Perspective projection.
    projection: PerspectiveProjection,
    /// Visualization configuration.
    config: VisualizationConfig,
    /// Current camera config (may be modified by orbit).
    camera: CameraConfig,
    /// Last update timestamp.
    last_update: Instant,
    /// Target frame duration.
    frame_duration: Duration,
    /// Current orbit angle (radians).
    orbit_angle: f32,
    /// Orbit elevation angle.
    orbit_elevation: f32,
}

impl RealtimeWindow {
    /// Create a new visualization window.
    pub fn new(title: &str, config: VisualizationConfig) -> RenderResult<Self> {
        let window = Window::new(
            title,
            config.width as usize,
            config.height as usize,
            WindowOptions {
                resize: true,
                scale_mode: minifb::ScaleMode::AspectRatioStretch,
                ..WindowOptions::default()
            },
        )
        .map_err(|e| RenderError::InitError(e.to_string()))?;

        let buffer = vec![0u32; (config.width * config.height) as usize];
        let backend = ImageBackend::new(config.width, config.height);
        let projection = PerspectiveProjection::new(&config.camera, config.width, config.height);
        let camera = config.camera.clone();
        let frame_duration = Duration::from_secs_f64(1.0 / config.target_fps as f64);

        Ok(Self {
            window,
            buffer,
            backend,
            projection,
            camera,
            config,
            last_update: Instant::now(),
            frame_duration,
            orbit_angle: 0.0,
            orbit_elevation: 0.5, // ~30 degrees
        })
    }

    /// Check if window is still open.
    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    /// Update the display with latest history.
    ///
    /// Returns `Ok(true)` if update succeeded, `Ok(false)` if window was closed.
    pub fn update(&mut self, history: &RollingHistory) -> RenderResult<bool> {
        if !self.is_open() {
            return Ok(false);
        }

        // Handle input
        self.handle_input();

        // Rate limit updates
        let now = Instant::now();
        if now.duration_since(self.last_update) < self.frame_duration {
            // Just update window without re-rendering
            self.window
                .update_with_buffer(&self.buffer, self.config.width as usize, self.config.height as usize)
                .map_err(|e| RenderError::RenderError(e.to_string()))?;
            return Ok(true);
        }
        self.last_update = now;

        // Handle window resize
        let (new_width, new_height) = self.window.get_size();
        if new_width != self.config.width as usize || new_height != self.config.height as usize {
            self.config.width = new_width as u32;
            self.config.height = new_height as u32;
            self.buffer.resize(new_width * new_height, 0);
            self.backend = ImageBackend::new(self.config.width, self.config.height);
            self.projection
                .update_viewport(self.config.width, self.config.height, &self.camera);
        }

        // Update camera orbit if enabled
        if self.camera.auto_orbit {
            self.orbit_angle += self.camera.orbit_speed * self.frame_duration.as_secs_f32();
            self.camera.position = orbit_camera_position(
                self.camera.target,
                self.camera.orbit_radius,
                self.orbit_elevation,
                self.orbit_angle,
            );
            self.projection.update_camera(&self.camera);
        }

        // Render scene
        self.backend
            .render(history, &self.projection, &self.config)?;

        // Convert RGB to ARGB for minifb
        self.rgb_to_argb();

        // Update window
        self.window
            .update_with_buffer(&self.buffer, self.config.width as usize, self.config.height as usize)
            .map_err(|e| RenderError::RenderError(e.to_string()))?;

        Ok(true)
    }

    /// Handle keyboard input.
    fn handle_input(&mut self) {
        // Number keys 1-8: Highlight specific drone
        for i in 0..8 {
            let key = match i {
                0 => Key::Key1,
                1 => Key::Key2,
                2 => Key::Key3,
                3 => Key::Key4,
                4 => Key::Key5,
                5 => Key::Key6,
                6 => Key::Key7,
                7 => Key::Key8,
                _ => continue,
            };
            if self.window.is_key_pressed(key, minifb::KeyRepeat::No) {
                self.config.drone_style.highlighted_env = Some(i);
            }
        }

        // Key 0: Clear highlight
        if self.window.is_key_pressed(Key::Key0, minifb::KeyRepeat::No) {
            self.config.drone_style.highlighted_env = None;
        }

        // T: Toggle trajectory
        if self.window.is_key_pressed(Key::T, minifb::KeyRepeat::No) {
            self.config.trajectory.enabled = !self.config.trajectory.enabled;
        }

        // A: Toggle axes
        if self.window.is_key_pressed(Key::A, minifb::KeyRepeat::No) {
            self.config.drone_style.show_axes = !self.config.drone_style.show_axes;
        }

        // O: Toggle orbit
        if self.window.is_key_pressed(Key::O, minifb::KeyRepeat::No) {
            self.camera.auto_orbit = !self.camera.auto_orbit;
        }

        // V: Toggle velocity vectors
        if self.window.is_key_pressed(Key::V, minifb::KeyRepeat::No) {
            self.config.drone_style.show_velocity = !self.config.drone_style.show_velocity;
        }

        // G: Toggle grid
        if self.window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
            self.config.scene.show_grid = !self.config.scene.show_grid;
        }

        // R: Reset camera
        if self.window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
            self.camera = self.config.camera.clone();
            self.orbit_angle = 0.0;
            self.orbit_elevation = 0.5;
            self.projection.update_camera(&self.camera);
        }

        // Arrow keys: Pan camera target
        let pan_speed = 0.2;
        if self.window.is_key_down(Key::Left) {
            self.camera.target[0] -= pan_speed;
            self.camera.position[0] -= pan_speed;
            self.projection.update_camera(&self.camera);
        }
        if self.window.is_key_down(Key::Right) {
            self.camera.target[0] += pan_speed;
            self.camera.position[0] += pan_speed;
            self.projection.update_camera(&self.camera);
        }
        if self.window.is_key_down(Key::Up) {
            self.camera.target[1] += pan_speed;
            self.camera.position[1] += pan_speed;
            self.projection.update_camera(&self.camera);
        }
        if self.window.is_key_down(Key::Down) {
            self.camera.target[1] -= pan_speed;
            self.camera.position[1] -= pan_speed;
            self.projection.update_camera(&self.camera);
        }

        // +/-: Zoom
        let zoom_speed = 0.3;
        if self.window.is_key_down(Key::Equal) || self.window.is_key_down(Key::NumPadPlus) {
            self.camera.orbit_radius = (self.camera.orbit_radius - zoom_speed).max(2.0);
            if self.camera.auto_orbit {
                self.camera.position = orbit_camera_position(
                    self.camera.target,
                    self.camera.orbit_radius,
                    self.orbit_elevation,
                    self.orbit_angle,
                );
                self.projection.update_camera(&self.camera);
            }
        }
        if self.window.is_key_down(Key::Minus) || self.window.is_key_down(Key::NumPadMinus) {
            self.camera.orbit_radius = (self.camera.orbit_radius + zoom_speed).min(20.0);
            if self.camera.auto_orbit {
                self.camera.position = orbit_camera_position(
                    self.camera.target,
                    self.camera.orbit_radius,
                    self.orbit_elevation,
                    self.orbit_angle,
                );
                self.projection.update_camera(&self.camera);
            }
        }

        // Page Up/Down: Adjust elevation
        if self.window.is_key_down(Key::PageUp) {
            self.orbit_elevation = (self.orbit_elevation + 0.02).min(1.4); // ~80 degrees max
        }
        if self.window.is_key_down(Key::PageDown) {
            self.orbit_elevation = (self.orbit_elevation - 0.02).max(0.1); // ~6 degrees min
        }
    }

    /// Convert RGB buffer to ARGB for minifb.
    fn rgb_to_argb(&mut self) {
        let rgb = self.backend.buffer();
        let pixels = self.config.width as usize * self.config.height as usize;

        for i in 0..pixels {
            let r = rgb[i * 3] as u32;
            let g = rgb[i * 3 + 1] as u32;
            let b = rgb[i * 3 + 2] as u32;
            self.buffer[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }

    /// Get current configuration (for saving settings).
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut VisualizationConfig {
        &mut self.config
    }
}
