//! PNG image rendering backend using plotters.

use std::path::Path;

use plotters::prelude::*;

use crate::renderer::{
    AgentTraces, CollisionEvent, EntityMapConfig, EnvSnapshot, PsdMatrix,
    RenderError, RenderResult, SpectrumConfig, TimelineConfig, WaterfallConfig,
};

/// Image backend for PNG output via plotters.
pub struct ImageBackend {
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
}

impl ImageBackend {
    /// Create a new image backend with the specified dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Render waterfall to a PNG file.
    pub fn render_waterfall_to_file(
        &self,
        psd_matrix: &PsdMatrix,
        config: &WaterfallConfig,
        path: impl AsRef<Path>,
    ) -> RenderResult<()> {
        let root =
            BitMapBackend::new(path.as_ref(), (self.width, self.height)).into_drawing_area();

        root.fill(&RGBColor(20, 20, 30))
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        self.draw_waterfall(&root, psd_matrix, config)?;

        root.present()
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        Ok(())
    }

    /// Render waterfall to RGB buffer.
    pub fn render_waterfall_to_buffer(
        &self,
        psd_matrix: &PsdMatrix,
        config: &WaterfallConfig,
    ) -> RenderResult<Vec<u8>> {
        let mut buffer = vec![0u8; (self.width * self.height * 3) as usize];

        {
            let root = BitMapBackend::with_buffer(&mut buffer, (self.width, self.height))
                .into_drawing_area();

            root.fill(&RGBColor(20, 20, 30))
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

            self.draw_waterfall(&root, psd_matrix, config)?;

            root.present()
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
        }

        Ok(buffer)
    }

    fn draw_waterfall<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        psd_matrix: &PsdMatrix,
        config: &WaterfallConfig,
    ) -> RenderResult<()>
    where
        DB::ErrorType: 'static,
    {
        if psd_matrix.num_time_steps == 0 || psd_matrix.num_freq_bins == 0 {
            return Err(RenderError::EmptyHistory);
        }

        // Calculate plot area (leave room for axes and colorbar)
        let (plot_width, plot_height) = if config.show_colorbar {
            (self.width - 100, self.height - 80)
        } else {
            (self.width - 60, self.height - 60)
        };

        let margin_left = 60i32;
        let margin_top = 30i32;

        // Normalize PSD to [0, 1]
        let (min_db, max_db) = config.psd_range_db;
        let normalized = psd_matrix.normalize(min_db, max_db);

        // Get colormap
        let colormap = config.colormap;

        // Draw heatmap
        let cell_width = plot_width as f32 / psd_matrix.num_freq_bins as f32;
        let cell_height = plot_height as f32 / psd_matrix.num_time_steps as f32;

        for t in 0..psd_matrix.num_time_steps {
            for f in 0..psd_matrix.num_freq_bins {
                let value = normalized[t * psd_matrix.num_freq_bins + f];
                let [r, g, b] = colormap.map(value);

                let x = margin_left + (f as f32 * cell_width) as i32;
                let y = margin_top + (t as f32 * cell_height) as i32;
                let w = (cell_width.ceil()) as i32 + 1;
                let h = (cell_height.ceil()) as i32 + 1;

                root.draw(&Rectangle::new(
                    [(x, y), (x + w, y + h)],
                    RGBColor(r, g, b).filled(),
                ))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }
        }

        // Draw axes
        let style = TextStyle::from(("sans-serif", 12).into_font()).color(&WHITE);

        // X-axis (frequency)
        let freq_min = psd_matrix.freq_bins.first().copied().unwrap_or(0.0);
        let freq_max = psd_matrix.freq_bins.last().copied().unwrap_or(1.0);

        for i in 0..=4 {
            let x = margin_left + (i as f32 * plot_width as f32 / 4.0) as i32;
            let freq = freq_min + (freq_max - freq_min) * i as f32 / 4.0;
            let label = config.freq_unit.format(freq);

            root.draw(&Text::new(
                label,
                (x, margin_top + plot_height as i32 + 15),
                &style,
            ))
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
        }

        // X-axis label
        root.draw(&Text::new(
            format!("Frequency ({})", config.freq_unit.suffix()),
            (
                margin_left + plot_width as i32 / 2 - 40,
                self.height as i32 - 10,
            ),
            &style,
        ))
        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Y-axis (time) - newest at bottom
        let time_min = psd_matrix.time_steps.first().copied().unwrap_or(0);
        let time_max = psd_matrix.time_steps.last().copied().unwrap_or(1);

        for i in 0..=4 {
            let y = margin_top + (i as f32 * plot_height as f32 / 4.0) as i32;
            let step = time_min + ((time_max - time_min) as f32 * i as f32 / 4.0) as u64;
            let label = format!("{}", step);

            root.draw(&Text::new(label, (5, y), &style))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
        }

        // Y-axis label
        root.draw(&Text::new(
            "Time (steps)",
            (5, margin_top - 15),
            &style,
        ))
        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Title
        let title_style = TextStyle::from(("sans-serif", 16).into_font()).color(&WHITE);
        root.draw(&Text::new(
            "Waterfall Spectrogram",
            (margin_left + plot_width as i32 / 2 - 80, 10),
            &title_style,
        ))
        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Colorbar (if enabled)
        if config.show_colorbar {
            let cb_x = margin_left + plot_width as i32 + 20;
            let cb_width = 20i32;
            let cb_height = plot_height as i32;

            // Draw colorbar gradient
            for i in 0..cb_height {
                let value = 1.0 - (i as f32 / cb_height as f32);
                let [r, g, b] = colormap.map(value);
                root.draw(&Rectangle::new(
                    [
                        (cb_x, margin_top + i),
                        (cb_x + cb_width, margin_top + i + 1),
                    ],
                    RGBColor(r, g, b).filled(),
                ))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }

            // Colorbar labels
            for i in 0..=4 {
                let y = margin_top + (i as f32 * cb_height as f32 / 4.0) as i32;
                let db = max_db - (max_db - min_db) * i as f32 / 4.0;
                let label = format!("{:.0}", db);
                root.draw(&Text::new(label, (cb_x + cb_width + 5, y), &style))
                    .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }

            root.draw(&Text::new("dBm", (cb_x + cb_width + 5, margin_top - 15), &style))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
        }

        Ok(())
    }

    /// Render spectrum snapshot to a PNG file.
    pub fn render_spectrum_to_file(
        &self,
        snapshot: &EnvSnapshot,
        config: &SpectrumConfig,
        path: impl AsRef<Path>,
    ) -> RenderResult<()> {
        let root =
            BitMapBackend::new(path.as_ref(), (self.width, self.height)).into_drawing_area();

        root.fill(&RGBColor(20, 20, 30))
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        self.draw_spectrum(&root, snapshot, config)?;

        root.present()
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        Ok(())
    }

    fn draw_spectrum<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        snapshot: &EnvSnapshot,
        config: &SpectrumConfig,
    ) -> RenderResult<()>
    where
        DB::ErrorType: 'static,
    {
        let psd_dbm = snapshot.psd_dbm();
        if psd_dbm.is_empty() {
            return Err(RenderError::EmptyHistory);
        }

        let (min_db, max_db) = config.psd_range_db;
        let freq_min = snapshot.freq_range.0;
        let freq_max = snapshot.freq_range.1;

        // Build chart
        let mut chart = ChartBuilder::on(root)
            .caption(
                format!("Spectrum Snapshot (Step {})", snapshot.step),
                ("sans-serif", 16).into_font().color(&WHITE),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(freq_min..freq_max, min_db..max_db)
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc(format!("Frequency ({})", config.freq_unit.suffix()))
            .y_desc("Power (dBm)")
            .axis_desc_style(("sans-serif", 12).into_font().color(&WHITE))
            .label_style(("sans-serif", 10).into_font().color(&WHITE))
            .light_line_style(RGBColor(50, 50, 60))
            .bold_line_style(RGBColor(80, 80, 90))
            .draw()
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Create frequency-power pairs
        let data: Vec<(f32, f32)> = snapshot
            .freq_bins
            .iter()
            .zip(psd_dbm.iter())
            .map(|(&f, &p)| (f, p.clamp(min_db, max_db)))
            .collect();

        // Draw filled area under curve
        if config.fill_under_curve {
            let [r, g, b] = config.line_color;
            chart
                .draw_series(AreaSeries::new(
                    data.iter().cloned(),
                    min_db,
                    RGBColor(r / 3, g / 3, b / 3).mix(0.5),
                ))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
        }

        // Draw line
        let [r, g, b] = config.line_color;
        chart
            .draw_series(LineSeries::new(data.iter().cloned(), RGBColor(r, g, b).stroke_width(2)))
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Draw noise floor
        if config.show_noise_floor {
            chart
                .draw_series(LineSeries::new(
                    [(freq_min, snapshot.noise_floor_dbm), (freq_max, snapshot.noise_floor_dbm)],
                    RGBColor(100, 100, 100).stroke_width(1),
                ))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
        }

        // Draw entity markers
        if config.show_entity_markers {
            for entity in &snapshot.entities {
                if entity.active {
                    let [r, g, b] = entity.entity_type.color();
                    chart
                        .draw_series(std::iter::once(Rectangle::new(
                            [
                                (entity.freq - entity.bandwidth / 2.0, min_db),
                                (entity.freq + entity.bandwidth / 2.0, min_db + 5.0),
                            ],
                            RGBColor(r, g, b).filled(),
                        )))
                        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
                }
            }
        }

        // Draw agent markers
        if config.show_agent_markers {
            // Jammers (red rectangles at top)
            for jammer in &snapshot.jammers {
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [
                            (jammer.freq - jammer.bandwidth / 2.0, max_db - 5.0),
                            (jammer.freq + jammer.bandwidth / 2.0, max_db),
                        ],
                        RGBColor(255, 60, 60).filled(),
                    )))
                    .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }

            // CRs (green circles)
            for cr in &snapshot.crs {
                let color = if cr.is_jammed(10.0) {
                    RGBColor(255, 100, 100) // Jammed - reddish
                } else {
                    RGBColor(60, 200, 60) // OK - green
                };

                chart
                    .draw_series(std::iter::once(Circle::new(
                        (cr.freq, max_db - 10.0),
                        5,
                        color.filled(),
                    )))
                    .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }
        }

        Ok(())
    }

    /// Render entity map to a PNG file.
    pub fn render_entity_map_to_file(
        &self,
        snapshot: &EnvSnapshot,
        config: &EntityMapConfig,
        path: impl AsRef<Path>,
    ) -> RenderResult<()> {
        let root =
            BitMapBackend::new(path.as_ref(), (self.width, self.height)).into_drawing_area();

        let [br, bg, bb] = config.background_color;
        root.fill(&RGBColor(br, bg, bb))
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        self.draw_entity_map(&root, snapshot, config)?;

        root.present()
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        Ok(())
    }

    fn draw_entity_map<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        snapshot: &EnvSnapshot,
        config: &EntityMapConfig,
    ) -> RenderResult<()>
    where
        DB::ErrorType: 'static,
    {
        let (world_w, world_h) = config.world_size.unwrap_or(snapshot.world_size);

        // Build chart
        let mut chart = ChartBuilder::on(root)
            .caption(
                format!("Entity Map (Step {})", snapshot.step),
                ("sans-serif", 16).into_font().color(&WHITE),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..world_w, 0.0..world_h)
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc("X (m)")
            .y_desc("Y (m)")
            .axis_desc_style(("sans-serif", 12).into_font().color(&WHITE))
            .label_style(("sans-serif", 10).into_font().color(&WHITE))
            .light_line_style(RGBColor(40, 40, 50))
            .draw()
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Draw entities
        for entity in &snapshot.entities {
            let [r, g, b] = entity.entity_type.color();
            let size = config.icon_size;

            // Draw entity icon
            chart
                .draw_series(std::iter::once(Circle::new(
                    (entity.x, entity.y),
                    size / 2,
                    RGBColor(r, g, b).filled(),
                )))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

            // Draw velocity arrow for mobile entities
            if config.show_velocity_arrows && entity.is_mobile() {
                let scale = 10.0; // Arrow length scale
                let end_x = entity.x + entity.vx * scale;
                let end_y = entity.y + entity.vy * scale;

                chart
                    .draw_series(LineSeries::new(
                        [(entity.x, entity.y), (end_x, end_y)],
                        RGBColor(r, g, b).stroke_width(2),
                    ))
                    .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }
        }

        // Draw jammers
        for jammer in &snapshot.jammers {
            let size = config.icon_size + 4;

            // Red filled circle with black border
            chart
                .draw_series(std::iter::once(Circle::new(
                    (jammer.x, jammer.y),
                    size / 2,
                    RGBColor(255, 60, 60).filled(),
                )))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

            chart
                .draw_series(std::iter::once(Circle::new(
                    (jammer.x, jammer.y),
                    size / 2,
                    BLACK.stroke_width(2),
                )))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

            // Draw label
            if config.show_labels {
                chart
                    .draw_series(std::iter::once(Text::new(
                        format!("J{}", jammer.index),
                        (jammer.x + 10.0, jammer.y + 10.0),
                        ("sans-serif", 10).into_font().color(&WHITE),
                    )))
                    .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }
        }

        // Draw CRs
        for cr in &snapshot.crs {
            let size = config.icon_size + 2;
            let color = if cr.is_jammed(config.sinr_threshold_db) {
                RGBColor(255, 100, 100)
            } else {
                RGBColor(60, 200, 60)
            };

            // Green/red filled circle
            chart
                .draw_series(std::iter::once(Circle::new(
                    (cr.x, cr.y),
                    size / 2,
                    color.filled(),
                )))
                .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

            // Draw label
            if config.show_labels {
                chart
                    .draw_series(std::iter::once(Text::new(
                        format!("CR{}", cr.index),
                        (cr.x + 10.0, cr.y + 10.0),
                        ("sans-serif", 10).into_font().color(&WHITE),
                    )))
                    .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            }
        }

        Ok(())
    }

    /// Render frequency timeline to a PNG file.
    pub fn render_timeline_to_file(
        &self,
        traces: &AgentTraces,
        collisions: &[CollisionEvent],
        config: &TimelineConfig,
        path: impl AsRef<Path>,
    ) -> RenderResult<()> {
        let root =
            BitMapBackend::new(path.as_ref(), (self.width, self.height)).into_drawing_area();

        root.fill(&RGBColor(20, 20, 30))
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        self.draw_timeline(&root, traces, collisions, config)?;

        root.present()
            .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

        Ok(())
    }

    fn draw_timeline<DB: DrawingBackend>(
        &self,
        root: &DrawingArea<DB, plotters::coord::Shift>,
        traces: &AgentTraces,
        collisions: &[CollisionEvent],
        config: &TimelineConfig,
    ) -> RenderResult<()>
    where
        DB::ErrorType: 'static,
    {
        // Check if we have any data to display
        if !traces.has_data() {
            // Draw "No data" message
            let style = TextStyle::from(("sans-serif", 20).into_font()).color(&WHITE);
            root.draw(&Text::new(
                "No frequency traces available",
                (self.width as i32 / 2 - 120, self.height as i32 / 2),
                &style,
            ))
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
            return Ok(());
        }

        // Determine ranges
        let (freq_min, freq_max) = traces.freq_range().unwrap_or((0.0, 1e9));

        // Get time range from agents and entities
        let jammer_times = traces.jammer_traces.iter()
            .filter_map(|t| t.first().map(|&(step, _, _)| step));
        let cr_times = traces.cr_traces.iter()
            .filter_map(|t| t.first().map(|&(step, _, _, _)| step));
        let entity_times = traces.entity_traces.iter()
            .filter_map(|t| t.data.first().map(|&(step, _, _, _)| step));
        let time_min: u64 = jammer_times.chain(cr_times).chain(entity_times).min().unwrap_or(0);

        let jammer_times_max = traces.jammer_traces.iter()
            .filter_map(|t| t.last().map(|&(step, _, _)| step));
        let cr_times_max = traces.cr_traces.iter()
            .filter_map(|t| t.last().map(|&(step, _, _, _)| step));
        let entity_times_max = traces.entity_traces.iter()
            .filter_map(|t| t.data.last().map(|&(step, _, _, _)| step));
        let time_max: u64 = jammer_times_max.chain(cr_times_max).chain(entity_times_max).max().unwrap_or(100);

        // Build chart
        let mut chart = ChartBuilder::on(root)
            .caption(
                "Frequency Timeline",
                ("sans-serif", 16).into_font().color(&WHITE),
            )
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(time_min as f64..time_max as f64, freq_min as f64..freq_max as f64)
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Time (steps)")
            .y_desc(format!("Frequency ({})", config.freq_unit.suffix()))
            .axis_desc_style(("sans-serif", 12).into_font().color(&WHITE))
            .label_style(("sans-serif", 10).into_font().color(&WHITE))
            .light_line_style(RGBColor(40, 40, 50))
            .draw()
            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;

        // Draw entity traces (background layer - draw first)
        if config.show_entities {
            for entity_trace in &traces.entity_traces {
                let [r, g, b] = entity_trace.entity_type.color();

                // Only draw active points
                let points: Vec<(f64, f64)> = entity_trace
                    .data
                    .iter()
                    .filter(|&&(_, _, _, active)| active)
                    .map(|&(step, freq, _, _)| (step as f64, freq as f64))
                    .collect();

                if !points.is_empty() {
                    chart
                        .draw_series(LineSeries::new(points, RGBColor(r, g, b).stroke_width(2)))
                        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
                }
            }
        }

        // Draw jammer traces (on top of entities)
        if config.show_jammers {
            let [r, g, b] = config.jammer_color;
            for trace in &traces.jammer_traces {
                let points: Vec<(f64, f64)> = trace
                    .iter()
                    .map(|&(step, freq, _)| (step as f64, freq as f64))
                    .collect();

                if !points.is_empty() {
                    chart
                        .draw_series(LineSeries::new(points, RGBColor(r, g, b).stroke_width(2)))
                        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
                }
            }
        }

        // Draw CR traces
        if config.show_crs {
            let [r, g, b] = config.cr_color;
            for trace in &traces.cr_traces {
                let points: Vec<(f64, f64)> = trace
                    .iter()
                    .map(|&(step, freq, _, _)| (step as f64, freq as f64))
                    .collect();

                if !points.is_empty() {
                    chart
                        .draw_series(LineSeries::new(points, RGBColor(r, g, b).stroke_width(2)))
                        .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
                }
            }
        }

        // Draw collision markers
        if config.show_collisions {
            let [r, g, b] = config.collision_color;
            for collision in collisions {
                // Find CR frequency at collision time
                if let Some(cr_trace) = traces.cr_traces.get(collision.cr_idx) {
                    if let Some(&(_, freq, _, _)) = cr_trace
                        .iter()
                        .find(|&&(step, _, _, _)| step == collision.step)
                    {
                        chart
                            .draw_series(std::iter::once(Circle::new(
                                (collision.step as f64, freq as f64),
                                6,
                                RGBColor(r, g, b).filled(),
                            )))
                            .map_err(|e| RenderError::ImageEncoding(format!("{:?}", e)))?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl super::RenderBackend for ImageBackend {
    type Output = Vec<u8>;

    fn render_waterfall(
        &self,
        psd_matrix: &PsdMatrix,
        config: &WaterfallConfig,
    ) -> RenderResult<Self::Output> {
        self.render_waterfall_to_buffer(psd_matrix, config)
    }

    fn render_spectrum(
        &self,
        snapshot: &EnvSnapshot,
        config: &SpectrumConfig,
    ) -> RenderResult<Self::Output> {
        let mut buffer = vec![0u8; (self.width * self.height * 3) as usize];

        {
            let root = BitMapBackend::with_buffer(&mut buffer, (self.width, self.height))
                .into_drawing_area();

            root.fill(&RGBColor(20, 20, 30))
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

            self.draw_spectrum(&root, snapshot, config)?;

            root.present()
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
        }

        Ok(buffer)
    }

    fn render_entity_map(
        &self,
        snapshot: &EnvSnapshot,
        config: &EntityMapConfig,
    ) -> RenderResult<Self::Output> {
        let mut buffer = vec![0u8; (self.width * self.height * 3) as usize];

        {
            let root = BitMapBackend::with_buffer(&mut buffer, (self.width, self.height))
                .into_drawing_area();

            let [br, bg, bb] = config.background_color;
            root.fill(&RGBColor(br, bg, bb))
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

            self.draw_entity_map(&root, snapshot, config)?;

            root.present()
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
        }

        Ok(buffer)
    }

    fn render_timeline(
        &self,
        traces: &AgentTraces,
        collisions: &[CollisionEvent],
        config: &TimelineConfig,
    ) -> RenderResult<Self::Output> {
        let mut buffer = vec![0u8; (self.width * self.height * 3) as usize];

        {
            let root = BitMapBackend::with_buffer(&mut buffer, (self.width, self.height))
                .into_drawing_area();

            root.fill(&RGBColor(20, 20, 30))
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;

            self.draw_timeline(&root, traces, collisions, config)?;

            root.present()
                .map_err(|e| RenderError::ImageEncoding(e.to_string()))?;
        }

        Ok(buffer)
    }
}
