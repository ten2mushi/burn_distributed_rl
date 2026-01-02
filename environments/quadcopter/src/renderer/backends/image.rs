//! Image-based rendering backend using plotters.

use super::RenderResult;
use crate::renderer::config::{DroneStyleConfig, FadeStyle, SceneConfig, TrajectoryConfig, VisualizationConfig};
use crate::renderer::drone_shape::{arm_lines_world, body_axes_world, motor_positions_world};
use crate::renderer::history::RollingHistory;
use crate::renderer::projection::PerspectiveProjection;
use crate::renderer::snapshot::MultiEnvSnapshot;
use crate::renderer::{env_color, ENV_COLORS};

use plotters::prelude::*;
use plotters::backend::BitMapBackend;

/// Image-based rendering backend.
pub struct ImageBackend {
    /// Image width.
    width: u32,
    /// Image height.
    height: u32,
    /// RGB buffer.
    buffer: Vec<u8>,
}

impl ImageBackend {
    /// Create a new image backend.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![0u8; (width * height * 3) as usize],
        }
    }

    /// Render the scene to the internal buffer.
    pub fn render(
        &mut self,
        history: &RollingHistory,
        projection: &PerspectiveProjection,
        config: &VisualizationConfig,
    ) -> RenderResult<()> {
        // Clear buffer with background color
        let bg = config.scene.background_color;
        for i in 0..(self.width * self.height) as usize {
            self.buffer[i * 3] = bg[0];
            self.buffer[i * 3 + 1] = bg[1];
            self.buffer[i * 3 + 2] = bg[2];
        }

        {
            let root = BitMapBackend::with_buffer(&mut self.buffer, (self.width, self.height))
                .into_drawing_area();

            let _ = root.fill(&RGBColor(bg[0], bg[1], bg[2]));

            // Get latest snapshot
            let snapshot = match history.latest() {
                Some(s) => s,
                None => return Ok(()),
            };

            // Render scene elements
            Self::render_ground(&root, projection, &config.scene);
            Self::render_trajectories(&root, history, projection, &config.trajectory);
            Self::render_targets(&root, snapshot, projection, &config.scene);
            Self::render_drones(&root, snapshot, projection, &config.drone_style);
            Self::render_hud(&root, history, config);

            let _ = root.present();
        }

        Ok(())
    }

    /// Get the rendered buffer as RGB bytes.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Get buffer as mutable for direct manipulation.
    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// Render ground plane with grid.
    fn render_ground(
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        projection: &PerspectiveProjection,
        config: &SceneConfig,
    ) {
        if !config.show_ground {
            return;
        }

        let gc = config.ground_color;
        let grid_color = RGBColor(gc[0] + 20, gc[1] + 20, gc[2] + 20);

        let [x_min, x_max, y_min, y_max, _, _] = config.bounds;

        // Draw grid lines
        if config.show_grid {
            let spacing = config.grid_spacing;
            let mut x = x_min;
            while x <= x_max {
                if let (Some(start), Some(end)) = (
                    projection.project([x, y_min, 0.0]),
                    projection.project([x, y_max, 0.0]),
                ) {
                    let _ = root.draw(&PathElement::new(
                        [(start.0 as i32, start.1 as i32), (end.0 as i32, end.1 as i32)],
                        grid_color.stroke_width(1),
                    ));
                }
                x += spacing;
            }

            let mut y = y_min;
            while y <= y_max {
                if let (Some(start), Some(end)) = (
                    projection.project([x_min, y, 0.0]),
                    projection.project([x_max, y, 0.0]),
                ) {
                    let _ = root.draw(&PathElement::new(
                        [(start.0 as i32, start.1 as i32), (end.0 as i32, end.1 as i32)],
                        grid_color.stroke_width(1),
                    ));
                }
                y += spacing;
            }
        }

        // Draw world axes at origin
        if config.show_world_axes {
            let axis_len = 1.0;
            let origin = [0.0, 0.0, 0.0];

            // X axis (red)
            if let Some(((ox, oy), (ex, ey))) =
                projection.project_direction(origin, [1.0, 0.0, 0.0], axis_len)
            {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    RED.stroke_width(2),
                ));
            }

            // Y axis (green)
            if let Some(((ox, oy), (ex, ey))) =
                projection.project_direction(origin, [0.0, 1.0, 0.0], axis_len)
            {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    GREEN.stroke_width(2),
                ));
            }

            // Z axis (blue)
            if let Some(((ox, oy), (ex, ey))) =
                projection.project_direction(origin, [0.0, 0.0, 1.0], axis_len)
            {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    BLUE.stroke_width(2),
                ));
            }
        }
    }

    /// Render trajectory trails.
    fn render_trajectories(
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        history: &RollingHistory,
        projection: &PerspectiveProjection,
        config: &TrajectoryConfig,
    ) {
        if !config.enabled {
            return;
        }

        for (env_idx, trajectory) in history.trajectories().enumerate() {
            let positions: Vec<_> = trajectory.positions().iter().collect();
            if positions.len() < 2 {
                continue;
            }

            let color = env_color(env_idx);
            let num_points = positions.len();

            for i in 0..positions.len() - 1 {
                let p1 = positions[i];
                let p2 = positions[i + 1];

                // Calculate alpha based on position in trail
                let alpha = match config.fade {
                    FadeStyle::Linear => {
                        config.min_alpha + (1.0 - config.min_alpha) * (i as f32 / num_points as f32)
                    }
                    FadeStyle::Exponential => {
                        let t = i as f32 / num_points as f32;
                        config.min_alpha + (1.0 - config.min_alpha) * t * t
                    }
                    FadeStyle::None => 1.0,
                };

                if let (Some(s1), Some(s2)) = (projection.project(*p1), projection.project(*p2)) {
                    let blended = RGBColor(
                        ((color[0] as f32 * alpha) as u8).max(30),
                        ((color[1] as f32 * alpha) as u8).max(30),
                        ((color[2] as f32 * alpha) as u8).max(30),
                    );
                    let _ = root.draw(&PathElement::new(
                        [(s1.0 as i32, s1.1 as i32), (s2.0 as i32, s2.1 as i32)],
                        blended.stroke_width(config.line_width as u32),
                    ));
                }
            }
        }
    }

    /// Render target positions.
    fn render_targets(
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        snapshot: &MultiEnvSnapshot,
        projection: &PerspectiveProjection,
        config: &SceneConfig,
    ) {
        if !config.show_targets {
            return;
        }

        for drone in snapshot.iter() {
            let target = drone.target_position;
            if let Some((x, y)) = projection.project(target) {
                let color = env_color(drone.env_idx);
                let size = (config.target_size * 30.0) as i32;

                // Draw target as cross
                let _ = root.draw(&Cross::new(
                    (x as i32, y as i32),
                    size,
                    RGBColor(color[0], color[1], color[2]).stroke_width(2),
                ));
            }
        }
    }

    /// Render all drones.
    fn render_drones(
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        snapshot: &MultiEnvSnapshot,
        projection: &PerspectiveProjection,
        config: &DroneStyleConfig,
    ) {
        // Sort drones by depth (furthest first)
        let mut drones: Vec<_> = snapshot.drones.iter().collect();
        drones.sort_by(|a, b| {
            let da = projection.distance_to_camera(a.position);
            let db = projection.distance_to_camera(b.position);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        for drone in drones {
            Self::render_drone(root, drone, projection, config);
        }
    }

    /// Render a single drone.
    fn render_drone(
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        drone: &crate::renderer::snapshot::DroneSnapshot,
        projection: &PerspectiveProjection,
        config: &DroneStyleConfig,
    ) {
        let pos = drone.position;
        let quat = drone.quaternion;
        let color = env_color(drone.env_idx);
        let is_highlighted = config.highlighted_env == Some(drone.env_idx);

        // Get screen position of center
        let center_2d = match projection.project(pos) {
            Some(p) => p,
            None => return, // Behind camera
        };

        let base_color = RGBColor(color[0], color[1], color[2]);
        let stroke_width = if is_highlighted { 3 } else { 2 };

        // Draw arms
        let arms = arm_lines_world(pos, quat, config.size);
        for (start, end) in &arms {
            if let (Some(s), Some(e)) = (projection.project(*start), projection.project(*end)) {
                let _ = root.draw(&PathElement::new(
                    [(s.0 as i32, s.1 as i32), (e.0 as i32, e.1 as i32)],
                    base_color.stroke_width(stroke_width),
                ));
            }
        }

        // Draw motors
        if config.show_motors {
            let motors = motor_positions_world(pos, quat, config.size);
            for (i, motor_pos) in motors.iter().enumerate() {
                if let Some((mx, my)) = projection.project(*motor_pos) {
                    let rpm_factor = (drone.motor_rpms[i] / crate::constants::MAX_RPM).clamp(0.0, 1.0);
                    let brightness = (color[0] as f32 * 0.5 + 127.0 * rpm_factor) as u8;
                    let motor_color = RGBColor(brightness, brightness, brightness);

                    let radius = (config.motor_radius * config.size * 50.0) as i32;
                    let _ = root.draw(&Circle::new(
                        (mx as i32, my as i32),
                        radius,
                        motor_color.filled(),
                    ));
                }
            }
        }

        // Draw body center
        let body_radius = (config.size * 15.0) as i32;
        let _ = root.draw(&Circle::new(
            (center_2d.0 as i32, center_2d.1 as i32),
            body_radius,
            base_color.filled(),
        ));

        // Draw orientation axes
        if config.show_axes {
            let (right, forward, up) = body_axes_world(quat);
            let axis_len = config.size * config.axes_scale;

            // X axis (right) - red
            if let Some(((ox, oy), (ex, ey))) = projection.project_direction(pos, right, axis_len) {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    RED.stroke_width(2),
                ));
            }

            // Y axis (forward) - green
            if let Some(((ox, oy), (ex, ey))) = projection.project_direction(pos, forward, axis_len) {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    GREEN.stroke_width(2),
                ));
            }

            // Z axis (up) - blue
            if let Some(((ox, oy), (ex, ey))) = projection.project_direction(pos, up, axis_len) {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    BLUE.stroke_width(2),
                ));
            }
        }

        // Draw heading arrow
        if config.show_heading {
            let (_, forward, _) = body_axes_world(quat);
            let arrow_len = config.size * 1.5;
            if let Some(((ox, oy), (ex, ey))) = projection.project_direction(pos, forward, arrow_len) {
                let _ = root.draw(&PathElement::new(
                    [(ox as i32, oy as i32), (ex as i32, ey as i32)],
                    WHITE.stroke_width(2),
                ));
            }
        }

        // Draw label
        if config.show_labels {
            let label = format!("{}", drone.env_idx);
            let label_y = center_2d.1 as i32 - body_radius - 5;
            let _ = root.draw(&Text::new(
                label,
                (center_2d.0 as i32, label_y),
                ("sans-serif", 14).into_font().color(&WHITE),
            ));
        }
    }

    /// Render HUD overlay with stats.
    fn render_hud(
        root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
        history: &RollingHistory,
        config: &VisualizationConfig,
    ) {
        let stats = history.stats();

        // Top-left: Step counter
        let step_text = format!("Step: {}", stats.step);
        let _ = root.draw(&Text::new(
            step_text,
            (10, 20),
            ("sans-serif", 16).into_font().color(&WHITE),
        ));

        // Top-left: FPS
        let fps_text = format!("FPS: {:.1}", stats.fps);
        let _ = root.draw(&Text::new(
            fps_text,
            (10, 40),
            ("sans-serif", 14).into_font().color(&RGBColor(150, 150, 150)),
        ));

        // Top-right: Reward stats
        let reward_text = format!(
            "Reward: {:.1} [{:.1}, {:.1}]",
            stats.mean_reward, stats.min_reward, stats.max_reward
        );
        let _ = root.draw(&Text::new(
            reward_text,
            (config.width as i32 - 200, 20),
            ("sans-serif", 14).into_font().color(&WHITE),
        ));

        // Bottom: Legend
        let legend_y = config.height as i32 - 30;
        for (i, color) in ENV_COLORS.iter().enumerate() {
            let x = 20 + i as i32 * 50;
            let _ = root.draw(&Rectangle::new(
                [(x, legend_y), (x + 15, legend_y + 15)],
                RGBColor(color[0], color[1], color[2]).filled(),
            ));
            let _ = root.draw(&Text::new(
                format!("{}", i),
                (x + 20, legend_y + 12),
                ("sans-serif", 12).into_font().color(&WHITE),
            ));
        }
    }
}
