//! Snapshot types for capturing environment state.

use crate::env::Quadcopter;
use crate::reward::RewardComponent;

/// Point-in-time state of a single quadcopter for visualization.
#[derive(Clone, Debug)]
pub struct DroneSnapshot {
    /// Environment index (0-7 for 8 parallel envs).
    pub env_idx: usize,
    /// Position in world frame [x, y, z] meters.
    pub position: [f32; 3],
    /// Orientation as quaternion [w, x, y, z].
    pub quaternion: [f32; 4],
    /// Linear velocity [vx, vy, vz] m/s.
    pub velocity: [f32; 3],
    /// Angular velocity [wx, wy, wz] rad/s.
    pub angular_velocity: [f32; 3],
    /// Motor RPMs [m0, m1, m2, m3].
    pub motor_rpms: [f32; 4],
    /// Target position [tx, ty, tz] meters.
    pub target_position: [f32; 3],
    /// Cumulative episode reward.
    pub episode_reward: f32,
    /// Episode step count.
    pub step_count: u32,
    /// Whether this environment just reset.
    pub just_reset: bool,
}

impl DroneSnapshot {
    /// Create a new drone snapshot.
    pub fn new(env_idx: usize) -> Self {
        Self {
            env_idx,
            position: [0.0, 0.0, 1.0],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            motor_rpms: [0.0; 4],
            target_position: [0.0, 0.0, 1.0],
            episode_reward: 0.0,
            step_count: 0,
            just_reset: false,
        }
    }

    /// Get speed (magnitude of velocity).
    #[inline]
    pub fn speed(&self) -> f32 {
        let [vx, vy, vz] = self.velocity;
        (vx * vx + vy * vy + vz * vz).sqrt()
    }

    /// Get angular speed (magnitude of angular velocity).
    #[inline]
    pub fn angular_speed(&self) -> f32 {
        let [wx, wy, wz] = self.angular_velocity;
        (wx * wx + wy * wy + wz * wz).sqrt()
    }

    /// Get average motor RPM.
    #[inline]
    pub fn avg_rpm(&self) -> f32 {
        self.motor_rpms.iter().sum::<f32>() / 4.0
    }
}

/// Snapshot of all parallel environments at a single timestep.
#[derive(Clone, Debug)]
pub struct MultiEnvSnapshot {
    /// Global simulation step.
    pub step: u64,
    /// Wall-clock timestamp for FPS calculation.
    pub timestamp: std::time::Instant,
    /// Individual drone snapshots for each environment.
    pub drones: Vec<DroneSnapshot>,
}

impl MultiEnvSnapshot {
    /// Create an empty snapshot for the given number of environments.
    pub fn new(num_envs: usize) -> Self {
        Self {
            step: 0,
            timestamp: std::time::Instant::now(),
            drones: (0..num_envs).map(DroneSnapshot::new).collect(),
        }
    }

    /// Capture current state from a Quadcopter environment.
    pub fn capture<R: RewardComponent>(env: &Quadcopter<R>, step: u64) -> Self {
        let state = env.state();
        let num_envs = state.num_envs;

        let drones: Vec<DroneSnapshot> = (0..num_envs)
            .map(|idx| DroneSnapshot {
                env_idx: idx,
                position: state.get_position(idx),
                quaternion: state.get_quaternion(idx),
                velocity: state.get_velocity(idx),
                angular_velocity: state.get_angular_velocity(idx),
                motor_rpms: state.get_actual_rpm(idx),
                target_position: state.get_target_position(idx),
                episode_reward: state.episode_reward[idx],
                step_count: state.step_count[idx],
                just_reset: state.step_count[idx] == 0,
            })
            .collect();

        Self {
            step,
            timestamp: std::time::Instant::now(),
            drones,
        }
    }

    /// Get number of environments in this snapshot.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.drones.len()
    }

    /// Get a specific drone snapshot.
    #[inline]
    pub fn drone(&self, idx: usize) -> &DroneSnapshot {
        &self.drones[idx]
    }

    /// Iterate over all drone snapshots.
    pub fn iter(&self) -> impl Iterator<Item = &DroneSnapshot> {
        self.drones.iter()
    }
}
