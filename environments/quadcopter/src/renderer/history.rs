//! Rolling history buffer for temporal visualization data.

use std::collections::VecDeque;
use super::snapshot::MultiEnvSnapshot;
use super::env_color;

/// Rolling trajectory buffer for one environment.
#[derive(Clone, Debug)]
pub struct DroneTrajectory {
    /// Circular buffer of positions.
    positions: VecDeque<[f32; 3]>,
    /// Maximum capacity.
    capacity: usize,
    /// Environment color [R, G, B].
    pub color: [u8; 3],
}

impl DroneTrajectory {
    /// Create a new trajectory buffer.
    pub fn new(capacity: usize, env_idx: usize) -> Self {
        Self {
            positions: VecDeque::with_capacity(capacity),
            capacity,
            color: env_color(env_idx),
        }
    }

    /// Push a new position to the trajectory.
    pub fn push(&mut self, position: [f32; 3]) {
        if self.positions.len() >= self.capacity {
            self.positions.pop_front();
        }
        self.positions.push_back(position);
    }

    /// Clear the trajectory (e.g., on episode reset).
    pub fn clear(&mut self) {
        self.positions.clear();
    }

    /// Get number of positions in trajectory.
    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Check if trajectory is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Iterate over positions from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &[f32; 3]> {
        self.positions.iter()
    }

    /// Get positions as a slice for rendering.
    pub fn positions(&self) -> &VecDeque<[f32; 3]> {
        &self.positions
    }
}

/// Rolling history for all environments.
#[derive(Clone)]
pub struct RollingHistory {
    /// Trajectory per environment.
    trajectories: Vec<DroneTrajectory>,
    /// Recent snapshots for analysis/replay.
    snapshots: VecDeque<MultiEnvSnapshot>,
    /// Snapshot buffer capacity.
    snapshot_capacity: usize,
    /// Number of environments.
    num_envs: usize,
}

impl RollingHistory {
    /// Create a new rolling history.
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `trajectory_length` - Number of positions to retain per trajectory
    /// * `snapshot_capacity` - Number of recent snapshots to retain
    pub fn new(num_envs: usize, trajectory_length: usize, snapshot_capacity: usize) -> Self {
        let trajectories = (0..num_envs)
            .map(|idx| DroneTrajectory::new(trajectory_length, idx))
            .collect();

        Self {
            trajectories,
            snapshots: VecDeque::with_capacity(snapshot_capacity),
            snapshot_capacity,
            num_envs,
        }
    }

    /// Push a new snapshot and update trajectories.
    pub fn push(&mut self, snapshot: MultiEnvSnapshot) {
        // Update trajectories with new positions
        for drone in &snapshot.drones {
            if drone.env_idx < self.num_envs {
                // Clear trajectory on episode reset
                if drone.just_reset {
                    self.trajectories[drone.env_idx].clear();
                }
                self.trajectories[drone.env_idx].push(drone.position);
            }
        }

        // Add snapshot to buffer
        if self.snapshots.len() >= self.snapshot_capacity {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&MultiEnvSnapshot> {
        self.snapshots.back()
    }

    /// Get trajectory for a specific environment.
    pub fn trajectory(&self, env_idx: usize) -> &DroneTrajectory {
        &self.trajectories[env_idx]
    }

    /// Iterate over all trajectories.
    pub fn trajectories(&self) -> impl Iterator<Item = &DroneTrajectory> {
        self.trajectories.iter()
    }

    /// Get number of snapshots in history.
    #[inline]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Check if history is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Get number of environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Iterate over recent snapshots (oldest to newest).
    pub fn iter(&self) -> impl Iterator<Item = &MultiEnvSnapshot> {
        self.snapshots.iter()
    }

    /// Compute statistics from recent history.
    pub fn stats(&self) -> HistoryStats {
        let mut stats = HistoryStats::default();

        if let Some(latest) = self.latest() {
            stats.step = latest.step;

            let rewards: Vec<f32> = latest.drones.iter().map(|d| d.episode_reward).collect();
            if !rewards.is_empty() {
                stats.mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
                stats.min_reward = rewards.iter().copied().fold(f32::INFINITY, f32::min);
                stats.max_reward = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }

            let steps: Vec<u32> = latest.drones.iter().map(|d| d.step_count).collect();
            if !steps.is_empty() {
                stats.mean_episode_length = steps.iter().sum::<u32>() as f32 / steps.len() as f32;
            }
        }

        // Compute FPS from snapshot timestamps
        if self.snapshots.len() >= 2 {
            let oldest = &self.snapshots[0];
            let newest = self.snapshots.back().unwrap();
            let elapsed = newest.timestamp.duration_since(oldest.timestamp).as_secs_f32();
            if elapsed > 0.0 {
                stats.fps = (self.snapshots.len() - 1) as f32 / elapsed;
            }
        }

        stats
    }
}

/// Statistics computed from history.
#[derive(Clone, Debug, Default)]
pub struct HistoryStats {
    /// Current simulation step.
    pub step: u64,
    /// Mean episode reward across environments.
    pub mean_reward: f32,
    /// Minimum episode reward.
    pub min_reward: f32,
    /// Maximum episode reward.
    pub max_reward: f32,
    /// Mean episode length.
    pub mean_episode_length: f32,
    /// Estimated frames per second.
    pub fps: f32,
}
