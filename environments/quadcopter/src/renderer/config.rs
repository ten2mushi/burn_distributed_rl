//! Configuration types for visualization.

/// Main visualization configuration.
#[derive(Clone, Debug)]
pub struct VisualizationConfig {
    /// Window/image dimensions.
    pub width: u32,
    /// Window/image height.
    pub height: u32,
    /// Camera configuration.
    pub camera: CameraConfig,
    /// Drone rendering style.
    pub drone_style: DroneStyleConfig,
    /// Trajectory trail configuration.
    pub trajectory: TrajectoryConfig,
    /// Scene configuration.
    pub scene: SceneConfig,
    /// Target FPS for realtime mode.
    pub target_fps: u32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            camera: CameraConfig::default(),
            drone_style: DroneStyleConfig::default(),
            trajectory: TrajectoryConfig::default(),
            scene: SceneConfig::default(),
            target_fps: 30,
        }
    }
}

impl VisualizationConfig {
    /// Create a new visualization config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set window dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set camera configuration.
    pub fn with_camera(mut self, camera: CameraConfig) -> Self {
        self.camera = camera;
        self
    }

    /// Set drone style configuration.
    pub fn with_drone_style(mut self, style: DroneStyleConfig) -> Self {
        self.drone_style = style;
        self
    }

    /// Set trajectory configuration.
    pub fn with_trajectory(mut self, trajectory: TrajectoryConfig) -> Self {
        self.trajectory = trajectory;
        self
    }

    /// Set target FPS.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.target_fps = fps;
        self
    }
}

/// 3D camera configuration for perspective projection.
#[derive(Clone, Debug)]
pub struct CameraConfig {
    /// Camera position in world frame [x, y, z].
    pub position: [f32; 3],
    /// Look-at target point [x, y, z].
    pub target: [f32; 3],
    /// Up vector [x, y, z].
    pub up: [f32; 3],
    /// Field of view in degrees.
    pub fov_degrees: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// Enable automatic camera orbit animation.
    pub auto_orbit: bool,
    /// Orbit speed in radians per second.
    pub orbit_speed: f32,
    /// Orbit radius (distance from target).
    pub orbit_radius: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: [5.0, 5.0, 4.0],
            target: [0.0, 0.0, 1.0],
            up: [0.0, 0.0, 1.0],
            fov_degrees: 60.0,
            near: 0.1,
            far: 100.0,
            auto_orbit: false,
            orbit_speed: 0.2,
            orbit_radius: 7.0,
        }
    }
}

impl CameraConfig {
    /// Create a new camera config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set camera position.
    pub fn with_position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }

    /// Set look-at target.
    pub fn with_target(mut self, x: f32, y: f32, z: f32) -> Self {
        self.target = [x, y, z];
        self
    }

    /// Set field of view.
    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov_degrees = fov_degrees;
        self
    }

    /// Enable auto-orbit with specified speed.
    pub fn with_auto_orbit(mut self, speed: f32) -> Self {
        self.auto_orbit = true;
        self.orbit_speed = speed;
        self
    }

    /// Set orbit radius.
    pub fn with_orbit_radius(mut self, radius: f32) -> Self {
        self.orbit_radius = radius;
        self
    }
}

/// Drone appearance configuration.
#[derive(Clone, Debug)]
pub struct DroneStyleConfig {
    /// Base size of drone (wingspan in world units).
    pub size: f32,
    /// Show orientation axes (RGB = XYZ in body frame).
    pub show_axes: bool,
    /// Axes length multiplier.
    pub axes_scale: f32,
    /// Show motor circles.
    pub show_motors: bool,
    /// Motor circle radius multiplier.
    pub motor_radius: f32,
    /// Show heading arrow.
    pub show_heading: bool,
    /// Show velocity vector.
    pub show_velocity: bool,
    /// Velocity vector scale.
    pub velocity_scale: f32,
    /// Show environment ID labels.
    pub show_labels: bool,
    /// Highlighted environment index (None = no highlight).
    pub highlighted_env: Option<usize>,
}

impl Default for DroneStyleConfig {
    fn default() -> Self {
        Self {
            size: 0.3,
            show_axes: true,
            axes_scale: 0.5,
            show_motors: true,
            motor_radius: 0.15,
            show_heading: true,
            show_velocity: false,
            velocity_scale: 0.3,
            show_labels: true,
            highlighted_env: None,
        }
    }
}

impl DroneStyleConfig {
    /// Create a new drone style config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set drone size.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Toggle orientation axes.
    pub fn with_axes(mut self, show: bool) -> Self {
        self.show_axes = show;
        self
    }

    /// Toggle velocity vectors.
    pub fn with_velocity(mut self, show: bool) -> Self {
        self.show_velocity = show;
        self
    }

    /// Set highlighted environment.
    pub fn with_highlight(mut self, env_idx: Option<usize>) -> Self {
        self.highlighted_env = env_idx;
        self
    }
}

/// Trajectory trail configuration.
#[derive(Clone, Debug)]
pub struct TrajectoryConfig {
    /// Enable trajectory trails.
    pub enabled: bool,
    /// Number of positions to retain.
    pub length: usize,
    /// Line width in pixels.
    pub line_width: f32,
    /// Fade style for trail.
    pub fade: FadeStyle,
    /// Minimum alpha for oldest point (0.0 - 1.0).
    pub min_alpha: f32,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            length: 200,
            line_width: 1.5,
            fade: FadeStyle::Linear,
            min_alpha: 0.1,
        }
    }
}

impl TrajectoryConfig {
    /// Create a new trajectory config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set trajectory length.
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = length;
        self
    }

    /// Toggle trajectory display.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set fade style.
    pub fn with_fade(mut self, fade: FadeStyle) -> Self {
        self.fade = fade;
        self
    }
}

/// Fade style for trajectory trails.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FadeStyle {
    /// Linear fade from full to min alpha.
    Linear,
    /// Exponential fade (faster decay at end).
    Exponential,
    /// No fade (constant alpha).
    None,
}

/// Scene configuration.
#[derive(Clone, Debug)]
pub struct SceneConfig {
    /// World bounds [x_min, x_max, y_min, y_max, z_min, z_max].
    pub bounds: [f32; 6],
    /// Show ground plane.
    pub show_ground: bool,
    /// Show grid lines on ground.
    pub show_grid: bool,
    /// Grid spacing in world units.
    pub grid_spacing: f32,
    /// Show world axes at origin.
    pub show_world_axes: bool,
    /// Background color [R, G, B].
    pub background_color: [u8; 3],
    /// Ground color [R, G, B].
    pub ground_color: [u8; 3],
    /// Show target positions.
    pub show_targets: bool,
    /// Target marker size.
    pub target_size: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            bounds: [-3.0, 3.0, -3.0, 3.0, 0.0, 4.0],
            show_ground: true,
            show_grid: true,
            grid_spacing: 1.0,
            show_world_axes: true,
            background_color: [30, 30, 40],
            ground_color: [50, 55, 60],
            show_targets: true,
            target_size: 0.15,
        }
    }
}

impl SceneConfig {
    /// Create a new scene config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set world bounds.
    pub fn with_bounds(mut self, bounds: [f32; 6]) -> Self {
        self.bounds = bounds;
        self
    }

    /// Toggle ground plane.
    pub fn with_ground(mut self, show: bool) -> Self {
        self.show_ground = show;
        self
    }

    /// Toggle grid.
    pub fn with_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Toggle target display.
    pub fn with_targets(mut self, show: bool) -> Self {
        self.show_targets = show;
        self
    }
}
