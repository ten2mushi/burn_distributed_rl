//! Configuration types and builders for the Quadcopter environment.
//!
//! Provides flexible configuration through builder patterns for:
//! - Observation space composition
//! - Reward function components (compile-time generic)
//! - Episode termination conditions
//! - Initial state distributions
//! - Physics parameters

use crate::constants::*;
use crate::noise::NoiseConfig;
use crate::normalization::NormalizationConfig;
use crate::physics::motor::MotorDynamicsConfig;
use crate::reward::{presets, RewardComponent};

// ============================================================================
// Observation Configuration
// ============================================================================

/// Configuration for observation space using builder pattern.
///
/// # Example
/// ```ignore
/// let obs_config = ObsConfig::new()
///     .with_position()
///     .with_euler()
///     .with_velocity()
///     .with_angular_velocity();
/// ```
#[derive(Clone, Debug, Default)]
pub struct ObsConfig {
    /// Include position [x, y, z]
    pub position: bool,
    /// Include quaternion [qw, qx, qy, qz]
    pub quaternion: bool,
    /// Include Euler angles [roll, pitch, yaw]
    pub euler: bool,
    /// Include linear velocity [vx, vy, vz]
    pub velocity: bool,
    /// Include angular velocity [wx, wy, wz]
    pub angular_velocity: bool,
    /// Include last applied RPMs (normalized) [rpm0, rpm1, rpm2, rpm3]
    pub last_action: bool,
    /// Include target position [tx, ty, tz]
    pub target_position: bool,
    /// Include target velocity [tvx, tvy, tvz]
    pub target_velocity: bool,
    /// Number of previous actions to include in observation (0 = none)
    pub action_buffer_len: usize,
    /// Observation noise configuration
    pub noise: NoiseConfig,
}

impl ObsConfig {
    /// Create empty observation config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Include position in observation.
    pub fn with_position(mut self) -> Self {
        self.position = true;
        self
    }

    /// Include quaternion in observation.
    pub fn with_quaternion(mut self) -> Self {
        self.quaternion = true;
        self
    }

    /// Include Euler angles in observation.
    pub fn with_euler(mut self) -> Self {
        self.euler = true;
        self
    }

    /// Include linear velocity in observation.
    pub fn with_velocity(mut self) -> Self {
        self.velocity = true;
        self
    }

    /// Include angular velocity in observation.
    pub fn with_angular_velocity(mut self) -> Self {
        self.angular_velocity = true;
        self
    }

    /// Include last action (normalized RPMs) in observation.
    pub fn with_last_action(mut self) -> Self {
        self.last_action = true;
        self
    }

    /// Include target position in observation.
    pub fn with_target_position(mut self) -> Self {
        self.target_position = true;
        self
    }

    /// Include target velocity in observation.
    pub fn with_target_velocity(mut self) -> Self {
        self.target_velocity = true;
        self
    }

    /// Include action history buffer in observation.
    pub fn with_action_buffer(mut self, len: usize) -> Self {
        self.action_buffer_len = len;
        self
    }

    /// Set observation noise configuration.
    ///
    /// Use `NoiseConfig::sensor_realistic()` for typical sensor noise,
    /// or `NoiseConfig::disabled()` (default) for clean observations.
    pub fn with_noise(mut self, noise: NoiseConfig) -> Self {
        self.noise = noise;
        self
    }

    /// Standard kinematic observation: [pos, euler, vel, ang_vel] = 12 dims
    pub fn kinematic() -> Self {
        Self::new()
            .with_position()
            .with_euler()
            .with_velocity()
            .with_angular_velocity()
    }

    /// Full state observation with quaternion: [pos, quat, vel, ang_vel] = 13 dims
    pub fn full_state() -> Self {
        Self::new()
            .with_position()
            .with_quaternion()
            .with_velocity()
            .with_angular_velocity()
    }

    /// Tracking observation: kinematic + target = 15 dims
    pub fn tracking() -> Self {
        Self::kinematic().with_target_position()
    }

    /// Compute total observation size.
    pub fn observation_size(&self) -> usize {
        let mut size = 0;
        if self.position {
            size += 3;
        }
        if self.quaternion {
            size += 4;
        }
        if self.euler {
            size += 3;
        }
        if self.velocity {
            size += 3;
        }
        if self.angular_velocity {
            size += 3;
        }
        if self.last_action {
            size += 4;
        }
        if self.target_position {
            size += 3;
        }
        if self.target_velocity {
            size += 3;
        }
        size + self.action_buffer_len * 4
    }

    /// Compute offsets for noise-injectable components.
    ///
    /// Returns (position_offset, euler_offset, velocity_offset, angular_velocity_offset).
    /// Components that are not enabled return None.
    pub fn noise_offsets(&self) -> (Option<usize>, Option<usize>, Option<usize>, Option<usize>) {
        let mut offset = 0;

        let position_offset = if self.position {
            let o = offset;
            offset += 3;
            Some(o)
        } else {
            None
        };

        if self.quaternion {
            offset += 4;
        }

        let euler_offset = if self.euler {
            let o = offset;
            offset += 3;
            Some(o)
        } else {
            None
        };

        let velocity_offset = if self.velocity {
            let o = offset;
            offset += 3;
            Some(o)
        } else {
            None
        };

        let angular_velocity_offset = if self.angular_velocity {
            let o = offset;
            // offset += 3; // Not needed, last component
            Some(o)
        } else {
            None
        };

        (position_offset, euler_offset, velocity_offset, angular_velocity_offset)
    }
}

// ============================================================================
// Reward Configuration (Type Aliases for Backward Compatibility)
// ============================================================================

/// Type alias for the default hover reward configuration.
///
/// This is a tuple of all standard reward components composed at compile-time
/// for zero-cost abstraction.
pub type RewardConfig = presets::HoverReward;

// ============================================================================
// Termination Configuration
// ============================================================================

/// Configuration for episode termination conditions.
#[derive(Clone, Debug)]
pub struct TerminationConfig {
    /// Maximum steps before truncation
    pub max_steps: u32,
    /// Position bounds [x_min, x_max, y_min, y_max, z_min, z_max]
    pub position_bounds: Option<[f32; 6]>,
    /// Maximum roll/pitch angle before truncation (radians)
    pub attitude_bounds: Option<f32>,
    /// Maximum velocity magnitude before truncation
    pub velocity_bounds: Option<f32>,
    /// Truncate if z < 0 (ground collision)
    pub ground_collision: bool,
}

impl Default for TerminationConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            position_bounds: Some([-2.0, 2.0, -2.0, 2.0, 0.0, 3.0]),
            attitude_bounds: Some(0.8), // ~45 degrees
            velocity_bounds: None,
            ground_collision: true,
        }
    }
}

impl TerminationConfig {
    /// Create a new termination config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum episode steps.
    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps;
        self
    }

    /// Set position bounds.
    pub fn with_position_bounds(mut self, bounds: [f32; 6]) -> Self {
        self.position_bounds = Some(bounds);
        self
    }

    /// Disable position bounds.
    pub fn without_position_bounds(mut self) -> Self {
        self.position_bounds = None;
        self
    }

    /// Set attitude bounds (max roll/pitch in radians).
    pub fn with_attitude_bounds(mut self, max_angle: f32) -> Self {
        self.attitude_bounds = Some(max_angle);
        self
    }

    /// Disable attitude bounds.
    pub fn without_attitude_bounds(mut self) -> Self {
        self.attitude_bounds = None;
        self
    }

    /// Set velocity bounds.
    pub fn with_velocity_bounds(mut self, max_vel: f32) -> Self {
        self.velocity_bounds = Some(max_vel);
        self
    }

    /// Disable velocity bounds.
    pub fn without_velocity_bounds(mut self) -> Self {
        self.velocity_bounds = None;
        self
    }

    /// Enable/disable ground collision truncation.
    pub fn with_ground_collision(mut self, enabled: bool) -> Self {
        self.ground_collision = enabled;
        self
    }
}

// ============================================================================
// Initialization Configuration
// ============================================================================

/// Type of random distribution for initialization.
#[derive(Clone, Debug, Copy, PartialEq)]
pub enum DistributionType {
    /// Uniform distribution in range
    Uniform,
    /// Gaussian distribution with mean at center of range
    Gaussian,
    /// Fixed value (no randomization)
    Fixed,
}

impl Default for DistributionType {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Configuration for initial state distribution.
#[derive(Clone, Debug)]
pub struct InitConfig {
    /// Position distribution type
    pub position_dist: DistributionType,
    /// Position range [x_min, x_max, y_min, y_max, z_min, z_max]
    pub position_range: [f32; 6],

    /// Velocity distribution type
    pub velocity_dist: DistributionType,
    /// Velocity range [vx_min, vx_max, vy_min, vy_max, vz_min, vz_max]
    pub velocity_range: [f32; 6],

    /// Attitude distribution type
    pub attitude_dist: DistributionType,
    /// Attitude range [roll_min, roll_max, pitch_min, pitch_max, yaw_min, yaw_max]
    pub attitude_range: [f32; 6],

    /// Angular velocity distribution type
    pub angular_vel_dist: DistributionType,
    /// Angular velocity range [wx_min, wx_max, wy_min, wy_max, wz_min, wz_max]
    pub angular_vel_range: [f32; 6],

    /// Initialize at hover RPM
    pub hover_init: bool,
}

impl Default for InitConfig {
    fn default() -> Self {
        use std::f32::consts::PI;
        Self {
            position_dist: DistributionType::Uniform,
            position_range: [-0.5, 0.5, -0.5, 0.5, 0.5, 1.5],

            velocity_dist: DistributionType::Fixed,
            velocity_range: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

            attitude_dist: DistributionType::Uniform,
            attitude_range: [-0.1, 0.1, -0.1, 0.1, -PI, PI],

            angular_vel_dist: DistributionType::Fixed,
            angular_vel_range: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

            hover_init: true,
        }
    }
}

impl InitConfig {
    /// Create a new initialization config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set position initialization range.
    pub fn with_position_range(mut self, range: [f32; 6]) -> Self {
        self.position_range = range;
        self
    }

    /// Set position distribution type.
    pub fn with_position_dist(mut self, dist: DistributionType) -> Self {
        self.position_dist = dist;
        self
    }

    /// Set velocity initialization range.
    pub fn with_velocity_range(mut self, range: [f32; 6]) -> Self {
        self.velocity_range = range;
        self
    }

    /// Set velocity distribution type.
    pub fn with_velocity_dist(mut self, dist: DistributionType) -> Self {
        self.velocity_dist = dist;
        self
    }

    /// Set attitude initialization range.
    pub fn with_attitude_range(mut self, range: [f32; 6]) -> Self {
        self.attitude_range = range;
        self
    }

    /// Set attitude distribution type.
    pub fn with_attitude_dist(mut self, dist: DistributionType) -> Self {
        self.attitude_dist = dist;
        self
    }

    /// Set angular velocity initialization range.
    pub fn with_angular_vel_range(mut self, range: [f32; 6]) -> Self {
        self.angular_vel_range = range;
        self
    }

    /// Set angular velocity distribution type.
    pub fn with_angular_vel_dist(mut self, dist: DistributionType) -> Self {
        self.angular_vel_dist = dist;
        self
    }

    /// Enable/disable hover initialization for RPMs.
    pub fn with_hover_init(mut self, enabled: bool) -> Self {
        self.hover_init = enabled;
        self
    }

    /// Fixed start: drone at origin, level, at rest.
    pub fn fixed_start() -> Self {
        Self {
            position_dist: DistributionType::Fixed,
            position_range: [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            velocity_dist: DistributionType::Fixed,
            velocity_range: [0.0; 6],
            attitude_dist: DistributionType::Fixed,
            attitude_range: [0.0; 6],
            angular_vel_dist: DistributionType::Fixed,
            angular_vel_range: [0.0; 6],
            hover_init: true,
        }
    }
}

// ============================================================================
// Task Mode
// ============================================================================

/// Operating mode for the quadcopter environment.
#[derive(Clone, Debug, Copy, PartialEq)]
pub enum TaskMode {
    /// Hover at a fixed target position
    Hover,
    /// Track externally provided targets each step
    Tracking,
}

impl Default for TaskMode {
    fn default() -> Self {
        Self::Hover
    }
}

// ============================================================================
// Main Environment Configuration
// ============================================================================

/// Main configuration for the Quadcopter environment.
///
/// Generic over reward type `R` which must implement [`RewardComponent`].
/// Defaults to [`presets::HoverReward`] for standard hover tasks.
///
/// # Example
///
/// ```ignore
/// use quadcopter_env::{QuadcopterConfig, reward::presets};
///
/// // Default configuration with hover reward
/// let config = QuadcopterConfig::new(64);
///
/// // Custom reward composition
/// let config = QuadcopterConfig::with_reward(64, presets::tracking());
/// ```
#[derive(Clone)]
pub struct QuadcopterConfig<R: RewardComponent = presets::HoverReward> {
    /// Number of parallel environments
    pub num_envs: usize,
    /// Physics simulation frequency (Hz)
    pub physics_freq: u32,
    /// Control frequency (Hz)
    pub ctrl_freq: u32,
    /// Observation configuration
    pub obs: ObsConfig,
    /// Reward function (compile-time generic for zero-cost composition)
    pub reward: R,
    /// Reward normalization and clipping configuration
    pub normalization: NormalizationConfig,
    /// Termination configuration
    pub termination: TerminationConfig,
    /// Initialization configuration
    pub init: InitConfig,
    /// Task mode (hover or tracking)
    pub task_mode: TaskMode,
    /// Default hover target position [x, y, z]
    pub hover_target: [f32; 3],
    /// Enable ground effect aerodynamics
    pub enable_ground_effect: bool,
    /// Enable air drag aerodynamics
    pub enable_drag: bool,
    /// Motor dynamics configuration (first-order lag response)
    pub motor_dynamics: MotorDynamicsConfig,
    /// Number of worker threads (1 = single-threaded)
    pub workers: usize,
}

impl Default for QuadcopterConfig<presets::HoverReward> {
    fn default() -> Self {
        Self {
            num_envs: 1,
            physics_freq: DEFAULT_PHYSICS_FREQ,
            ctrl_freq: DEFAULT_CTRL_FREQ,
            obs: ObsConfig::kinematic(),
            reward: presets::hover(),
            normalization: NormalizationConfig::default(),
            termination: TerminationConfig::default(),
            init: InitConfig::default(),
            task_mode: TaskMode::Hover,
            hover_target: [0.0, 0.0, 1.0],
            enable_ground_effect: true,
            enable_drag: true,
            motor_dynamics: MotorDynamicsConfig::default(),
            workers: 1,
        }
    }
}

impl QuadcopterConfig<presets::HoverReward> {
    /// Create a new configuration with specified number of environments.
    ///
    /// Uses the default hover reward configuration.
    pub fn new(num_envs: usize) -> Self {
        Self {
            num_envs,
            ..Default::default()
        }
    }
}

impl<R: RewardComponent> QuadcopterConfig<R> {
    /// Create a new configuration with a custom reward function.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use quadcopter_env::{QuadcopterConfig, reward::presets};
    ///
    /// let config = QuadcopterConfig::with_reward(64, presets::tracking());
    /// ```
    pub fn with_reward_fn<R2: RewardComponent>(self, reward: R2) -> QuadcopterConfig<R2> {
        QuadcopterConfig {
            num_envs: self.num_envs,
            physics_freq: self.physics_freq,
            ctrl_freq: self.ctrl_freq,
            obs: self.obs,
            reward,
            normalization: self.normalization,
            termination: self.termination,
            init: self.init,
            task_mode: self.task_mode,
            hover_target: self.hover_target,
            enable_ground_effect: self.enable_ground_effect,
            enable_drag: self.enable_drag,
            motor_dynamics: self.motor_dynamics,
            workers: self.workers,
        }
    }

    /// Set physics simulation frequency.
    pub fn with_physics_freq(mut self, freq: u32) -> Self {
        self.physics_freq = freq;
        self
    }

    /// Set control frequency.
    pub fn with_ctrl_freq(mut self, freq: u32) -> Self {
        self.ctrl_freq = freq;
        self
    }

    /// Set observation configuration.
    pub fn with_observation(mut self, obs: ObsConfig) -> Self {
        self.obs = obs;
        self
    }

    /// Set termination configuration.
    pub fn with_termination(mut self, termination: TerminationConfig) -> Self {
        self.termination = termination;
        self
    }

    /// Set initialization configuration.
    pub fn with_init(mut self, init: InitConfig) -> Self {
        self.init = init;
        self
    }

    /// Set task mode.
    pub fn with_task_mode(mut self, mode: TaskMode) -> Self {
        self.task_mode = mode;
        self
    }

    /// Set hover target position.
    pub fn with_hover_target(mut self, target: [f32; 3]) -> Self {
        self.hover_target = target;
        self
    }

    /// Enable/disable ground effect.
    pub fn with_ground_effect(mut self, enabled: bool) -> Self {
        self.enable_ground_effect = enabled;
        self
    }

    /// Enable/disable air drag.
    pub fn with_drag(mut self, enabled: bool) -> Self {
        self.enable_drag = enabled;
        self
    }

    /// Set motor dynamics configuration.
    ///
    /// Use `MotorDynamicsConfig::instantaneous()` for legacy instant response,
    /// or `MotorDynamicsConfig::realistic()` for ~15ms first-order lag.
    pub fn with_motor_dynamics(mut self, config: MotorDynamicsConfig) -> Self {
        self.motor_dynamics = config;
        self
    }

    /// Set reward normalization configuration.
    ///
    /// Use `NormalizationConfig::ppo_default()` for standard PPO-style normalization,
    /// or `NormalizationConfig::disabled()` (default) for raw rewards.
    pub fn with_normalization(mut self, config: NormalizationConfig) -> Self {
        self.normalization = config;
        self
    }

    /// Set number of worker threads.
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers.max(1);
        self
    }

    /// Get total observation size.
    pub fn observation_size(&self) -> usize {
        self.obs.observation_size()
    }

    /// Get physics timestep.
    pub fn dt_physics(&self) -> f32 {
        1.0 / self.physics_freq as f32
    }

    /// Get control timestep.
    pub fn dt_ctrl(&self) -> f32 {
        1.0 / self.ctrl_freq as f32
    }

    /// Get number of physics steps per control step.
    pub fn physics_steps_per_ctrl(&self) -> u32 {
        self.physics_freq / self.ctrl_freq
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_envs == 0 {
            return Err("num_envs must be at least 1".to_string());
        }

        if self.physics_freq == 0 || self.ctrl_freq == 0 {
            return Err("frequencies must be positive".to_string());
        }

        if self.physics_freq % self.ctrl_freq != 0 {
            return Err(format!(
                "physics_freq ({}) must be divisible by ctrl_freq ({})",
                self.physics_freq, self.ctrl_freq
            ));
        }

        if self.obs.observation_size() == 0 {
            return Err("observation size must be at least 1".to_string());
        }

        Ok(())
    }

    /// Build the Quadcopter environment.
    pub fn build(self) -> Result<crate::env::Quadcopter<R>, String> {
        self.validate()?;
        crate::env::Quadcopter::from_config(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obs_config_sizes() {
        assert_eq!(ObsConfig::new().with_position().observation_size(), 3);
        assert_eq!(ObsConfig::kinematic().observation_size(), 12);
        assert_eq!(ObsConfig::full_state().observation_size(), 13);
        assert_eq!(ObsConfig::tracking().observation_size(), 15);

        let custom = ObsConfig::new()
            .with_position()
            .with_euler()
            .with_action_buffer(5);
        assert_eq!(custom.observation_size(), 3 + 3 + 5 * 4);
    }

    #[test]
    fn test_config_validation() {
        assert!(QuadcopterConfig::new(64).validate().is_ok());
        assert!(QuadcopterConfig::new(0).validate().is_err());

        let bad_freq = QuadcopterConfig::new(1)
            .with_physics_freq(240)
            .with_ctrl_freq(70); // 240 not divisible by 70
        assert!(bad_freq.validate().is_err());
    }

    #[test]
    fn test_physics_steps_per_ctrl() {
        let config = QuadcopterConfig::new(1)
            .with_physics_freq(240)
            .with_ctrl_freq(30);
        assert_eq!(config.physics_steps_per_ctrl(), 8);
    }

    #[test]
    fn test_reward_presets() {
        use crate::reward::RewardComponent;
        use crate::state::QuadcopterState;

        let hover = presets::hover();
        let tracking = presets::tracking();

        // Test that presets return valid rewards
        let state = QuadcopterState::new(1, 0);
        let hover_reward = hover.compute(&state, 0);
        let tracking_reward = tracking.compute(&state, 0);

        assert!(hover_reward.is_finite());
        assert!(tracking_reward.is_finite());
    }

    #[test]
    fn test_config_with_custom_reward() {
        use crate::reward::components::*;

        // Create config with custom reward
        let config = QuadcopterConfig::new(64)
            .with_reward_fn((
                PositionError { weight: 2.0 },
                AliveBonus { bonus: 0.5 },
            ));

        assert_eq!(config.num_envs, 64);
        assert_eq!(config.reward.0.weight, 2.0);
        assert_eq!(config.reward.1.bonus, 0.5);
    }
}
