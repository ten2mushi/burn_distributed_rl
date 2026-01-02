//! Entity Struct-of-Arrays (SoA) State Storage
//!
//! SoA layout for cache-efficient SIMD processing of entity data.
//! All arrays are organized as [num_envs × max_entities] for
//! optimal memory access patterns when processing 8 environments in parallel.

#[cfg(feature = "simd")]
use std::simd::f32x8;

use super::{entity_idx, EntityType, ModulationType};

// ============================================================================
// Entity SoA Structure
// ============================================================================

/// Entity state storage in Struct-of-Arrays (SoA) layout.
///
/// Each field is a flat array of size [num_envs × max_entities].
/// This layout enables efficient SIMD processing across 8 environments.
///
/// # Memory Layout
///
/// For env=0..N and entity=0..M, the index is: `env * max_entities + entity`
///
/// # Active Entities
///
/// Entities with `active[idx] == 0` are inactive and should be skipped
/// during update and render operations.
#[derive(Clone)]
pub struct EntitySoA {
    // ========================================================================
    // Position [num_envs × max_entities]
    // ========================================================================
    /// X coordinate in meters
    pub x: Vec<f32>,
    /// Y coordinate in meters
    pub y: Vec<f32>,
    /// Z coordinate in meters (altitude)
    pub z: Vec<f32>,

    // ========================================================================
    // Velocity [num_envs × max_entities]
    // ========================================================================
    /// X velocity in m/s
    pub vx: Vec<f32>,
    /// Y velocity in m/s
    pub vy: Vec<f32>,
    /// Z velocity in m/s
    pub vz: Vec<f32>,

    // ========================================================================
    // RF Parameters [num_envs × max_entities]
    // ========================================================================
    /// Center frequency in Hz
    pub center_freq: Vec<f32>,
    /// Bandwidth in Hz
    pub bandwidth: Vec<f32>,
    /// Transmit power in dBm
    pub power_dbm: Vec<f32>,

    // ========================================================================
    // Type Information [num_envs × max_entities]
    // ========================================================================
    /// Entity type (see EntityType enum)
    pub entity_type: Vec<u8>,
    /// Modulation type (see ModulationType enum)
    pub modulation: Vec<u8>,

    // ========================================================================
    // Temporal State [num_envs × max_entities]
    // ========================================================================
    /// Active flag (1 = active, 0 = inactive)
    pub active: Vec<u8>,
    /// Timer for bursty/periodic behaviors (counts down or wraps)
    pub timer: Vec<f32>,
    /// Phase accumulator for continuous signals
    pub phase: Vec<f32>,
    /// Hop index for FHSS (Bluetooth)
    pub hop_idx: Vec<u32>,
    /// Respawn delay timer for mobile entities
    pub respawn_timer: Vec<f32>,

    // ========================================================================
    // Configuration Cache
    // ========================================================================
    /// Number of environments
    num_envs: usize,
    /// Maximum entities per environment
    max_entities: usize,
}

impl EntitySoA {
    /// Create a new EntitySoA structure for the given configuration.
    ///
    /// All entities are initialized as inactive.
    pub fn new(num_envs: usize, max_entities: usize) -> Self {
        let total = num_envs * max_entities;

        Self {
            // Position
            x: vec![0.0; total],
            y: vec![0.0; total],
            z: vec![0.0; total],

            // Velocity
            vx: vec![0.0; total],
            vy: vec![0.0; total],
            vz: vec![0.0; total],

            // RF Parameters
            center_freq: vec![0.0; total],
            bandwidth: vec![0.0; total],
            power_dbm: vec![0.0; total],

            // Type
            entity_type: vec![0; total],
            modulation: vec![0; total],

            // Temporal state
            active: vec![0; total],
            timer: vec![0.0; total],
            phase: vec![0.0; total],
            hop_idx: vec![0; total],
            respawn_timer: vec![0.0; total],

            // Cache
            num_envs,
            max_entities,
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the number of environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get the maximum entities per environment.
    #[inline]
    pub fn max_entities(&self) -> usize {
        self.max_entities
    }

    /// Get the total capacity (num_envs × max_entities).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.num_envs * self.max_entities
    }

    // ========================================================================
    // Index Functions
    // ========================================================================

    /// Get the flat index for an entity.
    #[inline]
    pub fn idx(&self, env: usize, entity: usize) -> usize {
        debug_assert!(env < self.num_envs);
        debug_assert!(entity < self.max_entities);
        entity_idx(env, entity, self.max_entities)
    }

    /// Get the base index for a SIMD batch of 8 environments.
    ///
    /// Returns the index for (batch * 8, entity).
    #[inline]
    pub fn simd_base(&self, batch: usize, entity: usize) -> usize {
        debug_assert!(batch * 8 < self.num_envs);
        debug_assert!(entity < self.max_entities);
        batch * 8 * self.max_entities + entity
    }

    // ========================================================================
    // Entity Management
    // ========================================================================

    /// Check if an entity is active.
    #[inline]
    pub fn is_active(&self, env: usize, entity: usize) -> bool {
        self.active[self.idx(env, entity)] != 0
    }

    /// Activate an entity.
    #[inline]
    pub fn activate(&mut self, env: usize, entity: usize) {
        let idx = entity_idx(env, entity, self.max_entities);
        self.active[idx] = 1;
    }

    /// Deactivate an entity.
    #[inline]
    pub fn deactivate(&mut self, env: usize, entity: usize) {
        let idx = entity_idx(env, entity, self.max_entities);
        self.active[idx] = 0;
    }

    /// Get the entity type for an entity.
    #[inline]
    pub fn get_type(&self, env: usize, entity: usize) -> EntityType {
        EntityType::from(self.entity_type[self.idx(env, entity)])
    }

    /// Get the modulation type for an entity.
    #[inline]
    pub fn get_modulation(&self, env: usize, entity: usize) -> ModulationType {
        ModulationType::from(self.modulation[self.idx(env, entity)])
    }

    /// Set entity parameters.
    pub fn set_entity(
        &mut self,
        env: usize,
        entity: usize,
        entity_type: EntityType,
        modulation: ModulationType,
        x: f32,
        y: f32,
        z: f32,
        center_freq: f32,
        bandwidth: f32,
        power_dbm: f32,
    ) {
        let idx = self.idx(env, entity);

        self.entity_type[idx] = entity_type.into();
        self.modulation[idx] = modulation.into();
        self.x[idx] = x;
        self.y[idx] = y;
        self.z[idx] = z;
        self.center_freq[idx] = center_freq;
        self.bandwidth[idx] = bandwidth;
        self.power_dbm[idx] = power_dbm;
        self.active[idx] = 1;

        // Reset temporal state
        self.vx[idx] = 0.0;
        self.vy[idx] = 0.0;
        self.vz[idx] = 0.0;
        self.timer[idx] = 0.0;
        self.phase[idx] = 0.0;
        self.hop_idx[idx] = 0;
        self.respawn_timer[idx] = 0.0;
    }

    /// Set entity velocity.
    #[inline]
    pub fn set_velocity(&mut self, env: usize, entity: usize, vx: f32, vy: f32, vz: f32) {
        let idx = self.idx(env, entity);
        self.vx[idx] = vx;
        self.vy[idx] = vy;
        self.vz[idx] = vz;
    }

    /// Reset all entities in an environment to inactive.
    pub fn reset_env(&mut self, env: usize) {
        for entity in 0..self.max_entities {
            let idx = self.idx(env, entity);
            self.active[idx] = 0;
            self.timer[idx] = 0.0;
            self.phase[idx] = 0.0;
            self.hop_idx[idx] = 0;
            self.respawn_timer[idx] = 0.0;
        }
    }

    /// Reset all entities in all environments.
    pub fn reset_all(&mut self) {
        self.active.fill(0);
        self.timer.fill(0.0);
        self.phase.fill(0.0);
        self.hop_idx.fill(0);
        self.respawn_timer.fill(0.0);
    }

    /// Count active entities in an environment.
    pub fn count_active(&self, env: usize) -> usize {
        (0..self.max_entities)
            .filter(|&e| self.is_active(env, e))
            .count()
    }

    /// Count active entities of a specific type in an environment.
    pub fn count_active_of_type(&self, env: usize, entity_type: EntityType) -> usize {
        (0..self.max_entities)
            .filter(|&e| {
                let idx = self.idx(env, e);
                self.active[idx] != 0 && self.entity_type[idx] == entity_type as u8
            })
            .count()
    }

    // ========================================================================
    // SIMD Load/Store Helpers
    // ========================================================================

    /// Load 8 contiguous f32 values starting at base index.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn load_f32x8(&self, arr: &[f32], base: usize) -> f32x8 {
        debug_assert!(base + 8 <= arr.len());
        f32x8::from_slice(&arr[base..base + 8])
    }

    /// Store 8 contiguous f32 values starting at base index.
    #[cfg(feature = "simd")]
    #[inline]
    pub fn store_f32x8(&self, arr: &mut [f32], base: usize, values: f32x8) {
        debug_assert!(base + 8 <= arr.len());
        let vals: [f32; 8] = values.into();
        arr[base..base + 8].copy_from_slice(&vals);
    }

    /// Load x, y, z positions for 8 environments as SIMD vectors.
    ///
    /// # Arguments
    /// * `batch` - SIMD batch index (0..num_envs/8)
    /// * `entity` - Entity index within each environment
    ///
    /// # Returns
    /// Tuple of (x, y, z) as f32x8 vectors.
    #[cfg(feature = "simd")]
    pub fn load_position_simd(&self, batch: usize, entity: usize) -> (f32x8, f32x8, f32x8) {
        // For SIMD, we need strided access since entities are stored per-env
        // Layout is: [env0_entity0, env0_entity1, ..., env1_entity0, ...]
        // We want: [env0_entityN, env1_entityN, ..., env7_entityN]

        let base_env = batch * 8;
        let stride = self.max_entities;

        let x = f32x8::from_array([
            self.x[entity_idx(base_env + 0, entity, stride)],
            self.x[entity_idx(base_env + 1, entity, stride)],
            self.x[entity_idx(base_env + 2, entity, stride)],
            self.x[entity_idx(base_env + 3, entity, stride)],
            self.x[entity_idx(base_env + 4, entity, stride)],
            self.x[entity_idx(base_env + 5, entity, stride)],
            self.x[entity_idx(base_env + 6, entity, stride)],
            self.x[entity_idx(base_env + 7, entity, stride)],
        ]);

        let y = f32x8::from_array([
            self.y[entity_idx(base_env + 0, entity, stride)],
            self.y[entity_idx(base_env + 1, entity, stride)],
            self.y[entity_idx(base_env + 2, entity, stride)],
            self.y[entity_idx(base_env + 3, entity, stride)],
            self.y[entity_idx(base_env + 4, entity, stride)],
            self.y[entity_idx(base_env + 5, entity, stride)],
            self.y[entity_idx(base_env + 6, entity, stride)],
            self.y[entity_idx(base_env + 7, entity, stride)],
        ]);

        let z = f32x8::from_array([
            self.z[entity_idx(base_env + 0, entity, stride)],
            self.z[entity_idx(base_env + 1, entity, stride)],
            self.z[entity_idx(base_env + 2, entity, stride)],
            self.z[entity_idx(base_env + 3, entity, stride)],
            self.z[entity_idx(base_env + 4, entity, stride)],
            self.z[entity_idx(base_env + 5, entity, stride)],
            self.z[entity_idx(base_env + 6, entity, stride)],
            self.z[entity_idx(base_env + 7, entity, stride)],
        ]);

        (x, y, z)
    }

    /// Store x, y, z positions for 8 environments from SIMD vectors.
    #[cfg(feature = "simd")]
    pub fn store_position_simd(
        &mut self,
        batch: usize,
        entity: usize,
        x: f32x8,
        y: f32x8,
        z: f32x8,
    ) {
        let base_env = batch * 8;
        let stride = self.max_entities;

        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let z_arr: [f32; 8] = z.into();

        for lane in 0..8 {
            let idx = entity_idx(base_env + lane, entity, stride);
            self.x[idx] = x_arr[lane];
            self.y[idx] = y_arr[lane];
            self.z[idx] = z_arr[lane];
        }
    }

    /// Load velocity for 8 environments as SIMD vectors.
    #[cfg(feature = "simd")]
    pub fn load_velocity_simd(&self, batch: usize, entity: usize) -> (f32x8, f32x8, f32x8) {
        let base_env = batch * 8;
        let stride = self.max_entities;

        let vx = f32x8::from_array([
            self.vx[entity_idx(base_env + 0, entity, stride)],
            self.vx[entity_idx(base_env + 1, entity, stride)],
            self.vx[entity_idx(base_env + 2, entity, stride)],
            self.vx[entity_idx(base_env + 3, entity, stride)],
            self.vx[entity_idx(base_env + 4, entity, stride)],
            self.vx[entity_idx(base_env + 5, entity, stride)],
            self.vx[entity_idx(base_env + 6, entity, stride)],
            self.vx[entity_idx(base_env + 7, entity, stride)],
        ]);

        let vy = f32x8::from_array([
            self.vy[entity_idx(base_env + 0, entity, stride)],
            self.vy[entity_idx(base_env + 1, entity, stride)],
            self.vy[entity_idx(base_env + 2, entity, stride)],
            self.vy[entity_idx(base_env + 3, entity, stride)],
            self.vy[entity_idx(base_env + 4, entity, stride)],
            self.vy[entity_idx(base_env + 5, entity, stride)],
            self.vy[entity_idx(base_env + 6, entity, stride)],
            self.vy[entity_idx(base_env + 7, entity, stride)],
        ]);

        let vz = f32x8::from_array([
            self.vz[entity_idx(base_env + 0, entity, stride)],
            self.vz[entity_idx(base_env + 1, entity, stride)],
            self.vz[entity_idx(base_env + 2, entity, stride)],
            self.vz[entity_idx(base_env + 3, entity, stride)],
            self.vz[entity_idx(base_env + 4, entity, stride)],
            self.vz[entity_idx(base_env + 5, entity, stride)],
            self.vz[entity_idx(base_env + 6, entity, stride)],
            self.vz[entity_idx(base_env + 7, entity, stride)],
        ]);

        (vx, vy, vz)
    }

    /// Load RF parameters for 8 environments as SIMD vectors.
    ///
    /// # Returns
    /// Tuple of (center_freq, bandwidth, power_dbm) as f32x8 vectors.
    #[cfg(feature = "simd")]
    pub fn load_rf_params_simd(&self, batch: usize, entity: usize) -> (f32x8, f32x8, f32x8) {
        let base_env = batch * 8;
        let stride = self.max_entities;

        let freq = f32x8::from_array([
            self.center_freq[entity_idx(base_env + 0, entity, stride)],
            self.center_freq[entity_idx(base_env + 1, entity, stride)],
            self.center_freq[entity_idx(base_env + 2, entity, stride)],
            self.center_freq[entity_idx(base_env + 3, entity, stride)],
            self.center_freq[entity_idx(base_env + 4, entity, stride)],
            self.center_freq[entity_idx(base_env + 5, entity, stride)],
            self.center_freq[entity_idx(base_env + 6, entity, stride)],
            self.center_freq[entity_idx(base_env + 7, entity, stride)],
        ]);

        let bw = f32x8::from_array([
            self.bandwidth[entity_idx(base_env + 0, entity, stride)],
            self.bandwidth[entity_idx(base_env + 1, entity, stride)],
            self.bandwidth[entity_idx(base_env + 2, entity, stride)],
            self.bandwidth[entity_idx(base_env + 3, entity, stride)],
            self.bandwidth[entity_idx(base_env + 4, entity, stride)],
            self.bandwidth[entity_idx(base_env + 5, entity, stride)],
            self.bandwidth[entity_idx(base_env + 6, entity, stride)],
            self.bandwidth[entity_idx(base_env + 7, entity, stride)],
        ]);

        let power = f32x8::from_array([
            self.power_dbm[entity_idx(base_env + 0, entity, stride)],
            self.power_dbm[entity_idx(base_env + 1, entity, stride)],
            self.power_dbm[entity_idx(base_env + 2, entity, stride)],
            self.power_dbm[entity_idx(base_env + 3, entity, stride)],
            self.power_dbm[entity_idx(base_env + 4, entity, stride)],
            self.power_dbm[entity_idx(base_env + 5, entity, stride)],
            self.power_dbm[entity_idx(base_env + 6, entity, stride)],
            self.power_dbm[entity_idx(base_env + 7, entity, stride)],
        ]);

        (freq, bw, power)
    }

    /// Load active flags for 8 environments as SIMD mask.
    #[cfg(feature = "simd")]
    pub fn load_active_simd(&self, batch: usize, entity: usize) -> [bool; 8] {
        let base_env = batch * 8;
        let stride = self.max_entities;

        [
            self.active[entity_idx(base_env + 0, entity, stride)] != 0,
            self.active[entity_idx(base_env + 1, entity, stride)] != 0,
            self.active[entity_idx(base_env + 2, entity, stride)] != 0,
            self.active[entity_idx(base_env + 3, entity, stride)] != 0,
            self.active[entity_idx(base_env + 4, entity, stride)] != 0,
            self.active[entity_idx(base_env + 5, entity, stride)] != 0,
            self.active[entity_idx(base_env + 6, entity, stride)] != 0,
            self.active[entity_idx(base_env + 7, entity, stride)] != 0,
        ]
    }

    /// Load timer values for 8 environments.
    #[cfg(feature = "simd")]
    pub fn load_timer_simd(&self, batch: usize, entity: usize) -> f32x8 {
        let base_env = batch * 8;
        let stride = self.max_entities;

        f32x8::from_array([
            self.timer[entity_idx(base_env + 0, entity, stride)],
            self.timer[entity_idx(base_env + 1, entity, stride)],
            self.timer[entity_idx(base_env + 2, entity, stride)],
            self.timer[entity_idx(base_env + 3, entity, stride)],
            self.timer[entity_idx(base_env + 4, entity, stride)],
            self.timer[entity_idx(base_env + 5, entity, stride)],
            self.timer[entity_idx(base_env + 6, entity, stride)],
            self.timer[entity_idx(base_env + 7, entity, stride)],
        ])
    }

    /// Store timer values for 8 environments.
    #[cfg(feature = "simd")]
    pub fn store_timer_simd(&mut self, batch: usize, entity: usize, timer: f32x8) {
        let base_env = batch * 8;
        let stride = self.max_entities;
        let arr: [f32; 8] = timer.into();

        for lane in 0..8 {
            self.timer[entity_idx(base_env + lane, entity, stride)] = arr[lane];
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_soa_creation() {
        let soa = EntitySoA::new(8, 64);

        assert_eq!(soa.num_envs(), 8);
        assert_eq!(soa.max_entities(), 64);
        assert_eq!(soa.capacity(), 8 * 64);
        assert_eq!(soa.x.len(), 8 * 64);
        assert_eq!(soa.active.len(), 8 * 64);
    }

    #[test]
    fn test_entity_indexing() {
        let soa = EntitySoA::new(8, 64);

        assert_eq!(soa.idx(0, 0), 0);
        assert_eq!(soa.idx(0, 63), 63);
        assert_eq!(soa.idx(1, 0), 64);
        assert_eq!(soa.idx(7, 63), 7 * 64 + 63);
    }

    #[test]
    fn test_entity_activation() {
        let mut soa = EntitySoA::new(8, 64);

        // Initially inactive
        assert!(!soa.is_active(0, 0));

        // Activate
        soa.activate(0, 0);
        assert!(soa.is_active(0, 0));

        // Deactivate
        soa.deactivate(0, 0);
        assert!(!soa.is_active(0, 0));
    }

    #[test]
    fn test_set_entity() {
        let mut soa = EntitySoA::new(8, 64);

        soa.set_entity(
            0,
            0,
            EntityType::TVStation,
            ModulationType::COFDM,
            100.0,
            200.0,
            50.0,
            550e6,
            6e6,
            45.0,
        );

        assert!(soa.is_active(0, 0));
        assert_eq!(soa.get_type(0, 0), EntityType::TVStation);
        assert_eq!(soa.get_modulation(0, 0), ModulationType::COFDM);

        let idx = soa.idx(0, 0);
        assert!((soa.x[idx] - 100.0).abs() < 1e-6);
        assert!((soa.y[idx] - 200.0).abs() < 1e-6);
        assert!((soa.z[idx] - 50.0).abs() < 1e-6);
        assert!((soa.center_freq[idx] - 550e6).abs() < 1.0);
        assert!((soa.bandwidth[idx] - 6e6).abs() < 1.0);
        assert!((soa.power_dbm[idx] - 45.0).abs() < 1e-6);
    }

    #[test]
    fn test_reset_env() {
        let mut soa = EntitySoA::new(8, 64);

        // Activate some entities
        soa.activate(0, 0);
        soa.activate(0, 1);
        let idx = soa.idx(0, 0);
        soa.timer[idx] = 1.5;

        // Reset environment 0
        soa.reset_env(0);

        assert!(!soa.is_active(0, 0));
        assert!(!soa.is_active(0, 1));
        let idx = soa.idx(0, 0);
        assert_eq!(soa.timer[idx], 0.0);
    }

    #[test]
    fn test_count_active() {
        let mut soa = EntitySoA::new(8, 64);

        assert_eq!(soa.count_active(0), 0);

        soa.activate(0, 0);
        soa.activate(0, 5);
        soa.activate(0, 10);

        assert_eq!(soa.count_active(0), 3);
        assert_eq!(soa.count_active(1), 0); // Different env
    }

    #[test]
    fn test_count_active_of_type() {
        let mut soa = EntitySoA::new(8, 64);

        soa.set_entity(
            0, 0,
            EntityType::TVStation,
            ModulationType::COFDM,
            0.0, 0.0, 0.0, 500e6, 6e6, 30.0,
        );
        soa.set_entity(
            0, 1,
            EntityType::TVStation,
            ModulationType::COFDM,
            0.0, 0.0, 0.0, 506e6, 6e6, 30.0,
        );
        soa.set_entity(
            0, 2,
            EntityType::FMRadio,
            ModulationType::FM,
            0.0, 0.0, 0.0, 100e6, 200e3, 20.0,
        );

        assert_eq!(soa.count_active_of_type(0, EntityType::TVStation), 2);
        assert_eq!(soa.count_active_of_type(0, EntityType::FMRadio), 1);
        assert_eq!(soa.count_active_of_type(0, EntityType::LTETower), 0);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_load_position() {
        let mut soa = EntitySoA::new(8, 64);

        // Set positions for entity 5 in each of 8 environments
        for env in 0..8 {
            let idx = soa.idx(env, 5);
            soa.x[idx] = env as f32 * 10.0;
            soa.y[idx] = env as f32 * 20.0;
            soa.z[idx] = env as f32 * 5.0;
        }

        let (x, y, z) = soa.load_position_simd(0, 5);
        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let z_arr: [f32; 8] = z.into();

        for env in 0..8 {
            assert!((x_arr[env] - env as f32 * 10.0).abs() < 1e-6);
            assert!((y_arr[env] - env as f32 * 20.0).abs() < 1e-6);
            assert!((z_arr[env] - env as f32 * 5.0).abs() < 1e-6);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_store_position() {
        let mut soa = EntitySoA::new(8, 64);

        let x = f32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = f32x8::from_array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        let z = f32x8::from_array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]);

        soa.store_position_simd(0, 3, x, y, z);

        for env in 0..8 {
            let idx = soa.idx(env, 3);
            assert!((soa.x[idx] - (env + 1) as f32).abs() < 1e-6);
            assert!((soa.y[idx] - (env + 1) as f32 * 10.0).abs() < 1e-6);
            assert!((soa.z[idx] - (env + 1) as f32 * 100.0).abs() < 1e-6);
        }
    }
}
