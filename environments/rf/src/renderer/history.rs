//! Rolling history buffer for temporal visualization.
//!
//! Maintains a fixed-size buffer of environment snapshots for
//! waterfall plots and frequency timelines.

use std::collections::VecDeque;

use super::snapshot::EnvSnapshot;

/// Rolling history buffer with fixed capacity.
///
/// Uses a `VecDeque` for O(1) push/pop at both ends while
/// maintaining bounded memory usage.
#[derive(Clone)]
pub struct RollingHistory {
    /// Maximum number of snapshots to retain.
    capacity: usize,
    /// Ring buffer of snapshots (newest at back).
    buffer: VecDeque<EnvSnapshot>,
}

impl RollingHistory {
    /// Create a new rolling history with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a new snapshot, dropping the oldest if at capacity.
    pub fn push(&mut self, snapshot: EnvSnapshot) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(snapshot);
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get the number of snapshots in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&EnvSnapshot> {
        self.buffer.back()
    }

    /// Get the oldest snapshot.
    pub fn oldest(&self) -> Option<&EnvSnapshot> {
        self.buffer.front()
    }

    /// Get a snapshot by index (0 = oldest).
    pub fn get(&self, index: usize) -> Option<&EnvSnapshot> {
        self.buffer.get(index)
    }

    /// Iterate over snapshots from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = &EnvSnapshot> {
        self.buffer.iter()
    }

    /// Iterate over snapshots from newest to oldest.
    pub fn iter_rev(&self) -> impl Iterator<Item = &EnvSnapshot> {
        self.buffer.iter().rev()
    }

    /// Get time range (oldest step, newest step).
    pub fn time_range(&self) -> Option<(u64, u64)> {
        if self.buffer.is_empty() {
            None
        } else {
            Some((
                self.buffer.front().unwrap().step,
                self.buffer.back().unwrap().step,
            ))
        }
    }

    /// Extract PSD matrix for waterfall plot.
    ///
    /// Returns a 2D matrix where:
    /// - Rows: Time steps (oldest at row 0)
    /// - Columns: Frequency bins
    /// - Values: Power in linear scale
    pub fn psd_matrix(&self) -> PsdMatrix {
        if self.buffer.is_empty() {
            return PsdMatrix {
                data: Vec::new(),
                num_time_steps: 0,
                num_freq_bins: 0,
                time_steps: Vec::new(),
                freq_bins: Vec::new(),
            };
        }

        let num_freq_bins = self.buffer.front().unwrap().psd.len();
        let num_time_steps = self.buffer.len();

        // Flatten into row-major order
        let mut data = Vec::with_capacity(num_time_steps * num_freq_bins);
        let mut time_steps = Vec::with_capacity(num_time_steps);

        for snapshot in &self.buffer {
            data.extend_from_slice(&snapshot.psd);
            time_steps.push(snapshot.step);
        }

        let freq_bins = self.buffer.front().unwrap().freq_bins.clone();

        PsdMatrix {
            data,
            num_time_steps,
            num_freq_bins,
            time_steps,
            freq_bins,
        }
    }

    /// Extract agent and entity frequency traces for timeline plot.
    ///
    /// Returns (jammer_traces, cr_traces, entity_traces) where each trace is
    /// a vector of (step, freq, bandwidth, ...) per agent/entity.
    pub fn agent_freq_traces(&self) -> AgentTraces {
        if self.buffer.is_empty() {
            return AgentTraces::empty();
        }

        // Determine number of agents from first snapshot
        let first = self.buffer.front().unwrap();
        let num_jammers = first.jammers.len();
        let num_crs = first.crs.len();
        let num_entities = first.entities.len();

        let mut jammer_traces: Vec<Vec<(u64, f32, f32)>> = vec![Vec::new(); num_jammers];
        let mut cr_traces: Vec<Vec<(u64, f32, f32, Option<f32>)>> = vec![Vec::new(); num_crs];

        // Initialize entity traces with their types from first snapshot
        let mut entity_traces: Vec<EntityTrace> = first
            .entities
            .iter()
            .map(|e| EntityTrace {
                entity_type: e.entity_type,
                data: Vec::new(),
            })
            .collect();

        for snapshot in &self.buffer {
            let step = snapshot.step;

            // Extract jammer frequencies
            for (i, jammer) in snapshot.jammers.iter().enumerate() {
                if i < num_jammers {
                    jammer_traces[i].push((step, jammer.freq, jammer.bandwidth));
                }
            }

            // Extract CR frequencies and SINR
            for (i, cr) in snapshot.crs.iter().enumerate() {
                if i < num_crs {
                    cr_traces[i].push((step, cr.freq, cr.bandwidth, cr.sinr_db));
                }
            }

            // Extract entity frequencies
            for (i, entity) in snapshot.entities.iter().enumerate() {
                if i < num_entities {
                    entity_traces[i]
                        .data
                        .push((step, entity.freq, entity.bandwidth, entity.active));
                }
            }
        }

        AgentTraces {
            jammer_traces,
            cr_traces,
            entity_traces,
        }
    }

    /// Find collision events where jammer and CR frequencies overlap.
    ///
    /// Returns list of (step, jammer_idx, cr_idx, overlap_hz).
    pub fn find_collisions(&self, overlap_threshold_hz: f32) -> Vec<CollisionEvent> {
        let mut collisions = Vec::new();

        for snapshot in &self.buffer {
            for (j_idx, jammer) in snapshot.jammers.iter().enumerate() {
                for (c_idx, cr) in snapshot.crs.iter().enumerate() {
                    // Check frequency overlap
                    let j_low = jammer.freq - jammer.bandwidth / 2.0;
                    let j_high = jammer.freq + jammer.bandwidth / 2.0;
                    let c_low = cr.freq - cr.bandwidth / 2.0;
                    let c_high = cr.freq + cr.bandwidth / 2.0;

                    let overlap = (j_high.min(c_high) - j_low.max(c_low)).max(0.0);

                    if overlap >= overlap_threshold_hz {
                        collisions.push(CollisionEvent {
                            step: snapshot.step,
                            jammer_idx: j_idx,
                            cr_idx: c_idx,
                            overlap_hz: overlap,
                            cr_sinr_db: cr.sinr_db,
                        });
                    }
                }
            }
        }

        collisions
    }

    /// Get statistics over the history.
    pub fn stats(&self) -> HistoryStats {
        if self.buffer.is_empty() {
            return HistoryStats::default();
        }

        let mut min_psd = f32::MAX;
        let mut max_psd = f32::MIN;
        let mut total_reward = 0.0;

        for snapshot in &self.buffer {
            for &p in &snapshot.psd {
                if p > 0.0 {
                    min_psd = min_psd.min(p);
                    max_psd = max_psd.max(p);
                }
            }
            total_reward += snapshot.episode_return;
        }

        let avg_reward = total_reward / self.buffer.len() as f32;

        // PSD is stored in watts, convert to dBm: dBm = 10*log10(watts) + 30
        HistoryStats {
            num_snapshots: self.buffer.len(),
            min_psd_linear: min_psd,
            max_psd_linear: max_psd,
            min_psd_dbm: if min_psd > 0.0 { 10.0 * min_psd.log10() + 30.0 } else { -120.0 },
            max_psd_dbm: if max_psd > 0.0 { 10.0 * max_psd.log10() + 30.0 } else { -120.0 },
            avg_episode_return: avg_reward,
        }
    }
}

/// PSD matrix for waterfall visualization.
#[derive(Clone, Debug)]
pub struct PsdMatrix {
    /// Flattened row-major data [time * freq].
    pub data: Vec<f32>,
    /// Number of time steps (rows).
    pub num_time_steps: usize,
    /// Number of frequency bins (columns).
    pub num_freq_bins: usize,
    /// Step numbers for each row.
    pub time_steps: Vec<u64>,
    /// Frequency bin centers in Hz.
    pub freq_bins: Vec<f32>,
}

impl PsdMatrix {
    /// Get value at (time_idx, freq_idx).
    pub fn get(&self, time_idx: usize, freq_idx: usize) -> f32 {
        self.data[time_idx * self.num_freq_bins + freq_idx]
    }

    /// Get a row (all frequency bins at one time step).
    pub fn row(&self, time_idx: usize) -> &[f32] {
        let start = time_idx * self.num_freq_bins;
        &self.data[start..start + self.num_freq_bins]
    }

    /// Convert to dBm.
    ///
    /// Note: PSD is stored in linear watts, not mW.
    /// dBm = 10 * log10(watts) + 30
    pub fn to_dbm(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&p| if p > 0.0 { 10.0 * p.log10() + 30.0 } else { -120.0 })
            .collect()
    }

    /// Normalize to [0, 1] based on dB range.
    ///
    /// Note: PSD is stored in linear watts, not mW.
    /// dBm = 10 * log10(watts) + 30
    pub fn normalize(&self, min_db: f32, max_db: f32) -> Vec<f32> {
        let range = max_db - min_db;
        self.data
            .iter()
            .map(|&p| {
                let db = if p > 0.0 { 10.0 * p.log10() + 30.0 } else { -120.0 };
                ((db - min_db) / range).clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// Agent frequency traces for timeline visualization.
#[derive(Clone, Debug)]
pub struct AgentTraces {
    /// Jammer traces: Vec of (step, freq, bandwidth) per jammer.
    pub jammer_traces: Vec<Vec<(u64, f32, f32)>>,
    /// CR traces: Vec of (step, freq, bandwidth, sinr_db) per CR.
    pub cr_traces: Vec<Vec<(u64, f32, f32, Option<f32>)>>,
    /// Entity traces: Vec of (step, freq, bandwidth, entity_type, active) per entity.
    /// entity_type is stored as u8 for efficiency.
    pub entity_traces: Vec<EntityTrace>,
}

/// A single entity's frequency trace over time.
#[derive(Clone, Debug)]
pub struct EntityTrace {
    /// Entity type (for color selection).
    pub entity_type: super::snapshot::EntityType,
    /// Frequency/bandwidth/active data over time: (step, freq, bandwidth, active).
    pub data: Vec<(u64, f32, f32, bool)>,
}

impl AgentTraces {
    /// Create empty traces.
    pub fn empty() -> Self {
        Self {
            jammer_traces: Vec::new(),
            cr_traces: Vec::new(),
            entity_traces: Vec::new(),
        }
    }

    /// Get number of jammers.
    pub fn num_jammers(&self) -> usize {
        self.jammer_traces.len()
    }

    /// Get number of CRs.
    pub fn num_crs(&self) -> usize {
        self.cr_traces.len()
    }

    /// Get number of entities with traces.
    pub fn num_entities(&self) -> usize {
        self.entity_traces.len()
    }

    /// Check if there are any traces (agents or entities).
    pub fn has_data(&self) -> bool {
        !self.jammer_traces.is_empty()
            || !self.cr_traces.is_empty()
            || !self.entity_traces.is_empty()
    }

    /// Get frequency range across all agents and entities.
    pub fn freq_range(&self) -> Option<(f32, f32)> {
        let mut min_freq = f32::MAX;
        let mut max_freq = f32::MIN;
        let mut found = false;

        for trace in &self.jammer_traces {
            for &(_, freq, bw) in trace {
                min_freq = min_freq.min(freq - bw / 2.0);
                max_freq = max_freq.max(freq + bw / 2.0);
                found = true;
            }
        }

        for trace in &self.cr_traces {
            for &(_, freq, bw, _) in trace {
                min_freq = min_freq.min(freq - bw / 2.0);
                max_freq = max_freq.max(freq + bw / 2.0);
                found = true;
            }
        }

        // Include entity frequency traces
        for entity_trace in &self.entity_traces {
            for &(_, freq, bw, active) in &entity_trace.data {
                if active {
                    min_freq = min_freq.min(freq - bw / 2.0);
                    max_freq = max_freq.max(freq + bw / 2.0);
                    found = true;
                }
            }
        }

        if found {
            Some((min_freq, max_freq))
        } else {
            None
        }
    }
}

/// Collision event between jammer and CR.
#[derive(Clone, Debug)]
pub struct CollisionEvent {
    /// Step when collision occurred.
    pub step: u64,
    /// Jammer index.
    pub jammer_idx: usize,
    /// CR index.
    pub cr_idx: usize,
    /// Overlap bandwidth in Hz.
    pub overlap_hz: f32,
    /// CR's SINR at collision time.
    pub cr_sinr_db: Option<f32>,
}

/// Statistics over history buffer.
#[derive(Clone, Debug, Default)]
pub struct HistoryStats {
    pub num_snapshots: usize,
    pub min_psd_linear: f32,
    pub max_psd_linear: f32,
    pub min_psd_dbm: f32,
    pub max_psd_dbm: f32,
    pub avg_episode_return: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::snapshot::AgentSnapshot;

    fn make_snapshot(step: u64, psd: Vec<f32>) -> EnvSnapshot {
        let num_bins = psd.len();
        EnvSnapshot::new(
            step,
            psd,
            (0..num_bins).map(|i| 1e9 + i as f32 * 1e6).collect(),
            -100.0,
            (1000.0, 1000.0),
            (1e9, 2e9),
        )
    }

    #[test]
    fn test_rolling_buffer_capacity() {
        let mut history = RollingHistory::new(3);

        history.push(make_snapshot(0, vec![1.0]));
        history.push(make_snapshot(1, vec![2.0]));
        history.push(make_snapshot(2, vec![3.0]));
        assert_eq!(history.len(), 3);

        // Push one more, should drop oldest
        history.push(make_snapshot(3, vec![4.0]));
        assert_eq!(history.len(), 3);
        assert_eq!(history.oldest().unwrap().step, 1);
        assert_eq!(history.latest().unwrap().step, 3);
    }

    #[test]
    fn test_psd_matrix() {
        let mut history = RollingHistory::new(10);
        history.push(make_snapshot(0, vec![1.0, 2.0, 3.0]));
        history.push(make_snapshot(1, vec![4.0, 5.0, 6.0]));

        let matrix = history.psd_matrix();
        assert_eq!(matrix.num_time_steps, 2);
        assert_eq!(matrix.num_freq_bins, 3);
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 2), 6.0);
    }

    #[test]
    fn test_agent_traces() {
        let mut history = RollingHistory::new(10);

        let mut snap1 = make_snapshot(0, vec![1.0]);
        snap1.add_jammer(AgentSnapshot::new_jammer(0, 0.0, 0.0, 2.4e9, 20e6, 30.0));
        snap1.add_cr(AgentSnapshot::new_cr(0, 0.0, 0.0, 2.45e9, 5e6, 10.0, 15.0));
        history.push(snap1);

        let mut snap2 = make_snapshot(1, vec![2.0]);
        snap2.add_jammer(AgentSnapshot::new_jammer(0, 0.0, 0.0, 2.42e9, 20e6, 30.0));
        snap2.add_cr(AgentSnapshot::new_cr(0, 0.0, 0.0, 2.5e9, 5e6, 10.0, 20.0));
        history.push(snap2);

        let traces = history.agent_freq_traces();
        assert_eq!(traces.num_jammers(), 1);
        assert_eq!(traces.num_crs(), 1);
        assert_eq!(traces.jammer_traces[0].len(), 2);
        assert_eq!(traces.cr_traces[0].len(), 2);
    }

    #[test]
    fn test_find_collisions() {
        let mut history = RollingHistory::new(10);

        let mut snap = make_snapshot(0, vec![1.0]);
        // Jammer at 2.4 GHz, 20 MHz BW -> 2.39-2.41 GHz
        snap.add_jammer(AgentSnapshot::new_jammer(0, 0.0, 0.0, 2.4e9, 20e6, 30.0));
        // CR at 2.41 GHz, 10 MHz BW -> 2.405-2.415 GHz (overlaps with jammer)
        snap.add_cr(AgentSnapshot::new_cr(0, 0.0, 0.0, 2.41e9, 10e6, 10.0, 5.0));
        history.push(snap);

        let collisions = history.find_collisions(1e6); // 1 MHz threshold
        assert_eq!(collisions.len(), 1);
        assert!(collisions[0].overlap_hz > 4e6); // Should be about 5 MHz overlap
    }
}
