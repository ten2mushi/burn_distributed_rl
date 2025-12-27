# Creating Custom Environments

This specific guide addresses the user request to "create custom environments using pufferlib" by showing how to build **Operant** environments. Operant is the high-performance, native Rust equivalent to the PufferLib vectorization pattern.

## Anatomy of a Custom Environment

To create a high-performance environment, you must implement the `operant_core::Environment` trait.

### 1. Define the State Struct (SoA Layout)

Refrain from creating a `struct Agent { x: f32, v: f32 }`. Instead, create a structure that holds vectors of components.

```rust
use operant_core::{Environment, ResetMask, StepResult, LogData};

pub struct MyCustomEnv {
    // State variables (Vectors of size num_envs)
    positions: Vec<f32>,
    velocities: Vec<f32>,
    
    // Standard bookkeeping buffers
    rewards: Vec<f32>,
    terminals: Vec<u8>,
    truncations: Vec<u8>,
    episode_ticks: Vec<u32>,
    
    // Configuration
    num_envs: usize,
    max_steps: u32,
    
    // RNG seeds per environment for determinism
    rng_seeds: Vec<u64>,
}
```

### 2. Implement Initialization

Initialize all vectors to size `num_envs`.

```rust
impl MyCustomEnv {
    pub fn new(num_envs: usize, max_steps: u32) -> Self {
        Self {
            positions: vec![0.0; num_envs],
            velocities: vec![0.0; num_envs],
            rewards: vec![0.0; num_envs],
            terminals: vec![0; num_envs],
            truncations: vec![0; num_envs],
            episode_ticks: vec![0; num_envs],
            num_envs,
            max_steps,
            rng_seeds: (0..num_envs as u64).collect(),
        }
    }
}
```

### 3. Implement Single Environment Logic (`reset` & `step`)

It is best to first implement "scalar" logic (working on one index) before optimizing for SIMD.

```rust
impl MyCustomEnv {
    #[inline]
    fn reset_single(&mut self, idx: usize, seed: u64) {
        // Use the seed to re-initialize state
        self.positions[idx] = 0.0;
        self.velocities[idx] = 0.0;
        self.episode_ticks[idx] = 0;
        // NOTE: Do not clear rewards/terminals here; they belong to the previous step
    }

    #[inline]
    fn step_single(&mut self, idx: usize, action: f32) {
        // Physics logic
        self.positions[idx] += self.velocities[idx];
        self.velocities[idx] += action; // Simple dynamics
        
        self.episode_ticks[idx] += 1;
        
        // Termination logic
        let terminal = self.positions[idx].abs() > 10.0;
        let truncated = self.episode_ticks[idx] >= self.max_steps;
        
        // Update buffers
        self.terminals[idx] = terminal as u8;
        self.truncations[idx] = truncated as u8;
        self.rewards[idx] = if terminal { 0.0 } else { 1.0 };
    }
}
```

### 4. Implement the `Environment` Trait

Connect your scalar logic to the vectorized API.

```rust
impl Environment for MyCustomEnv {
    fn num_envs(&self) -> usize { self.num_envs }
    
    fn observation_size(&self) -> usize { 2 } // [pos, vel]
    
    fn num_actions(&self) -> Option<usize> { None } // Continuous actions
    
    fn reset(&mut self, seed: u64) {
        for i in 0..self.num_envs {
            self.reset_single(i, seed + i as u64);
            // Clear outputs for clean slate
            self.rewards[i] = 0.0;
            self.terminals[i] = 0;
        }
    }
    
    fn step(&mut self, actions: &[f32]) {
        // 1. Step all environments
        for i in 0..self.num_envs {
            self.step_single(i, actions[i]);
        }
        
        // 2. Auto-reset terminated ones (Standard Gym behavior)
        for i in 0..self.num_envs {
            if self.terminals[i] != 0 || self.truncations[i] != 0 {
                let new_seed = self.rng_seeds[i].wrapping_add(self.num_envs as u64);
                self.reset_single(i, new_seed);
            }
        }
    }
    
    // Implement data writing (Zero-Copy Interface)
    fn write_observations(&self, buffer: &mut [f32]) {
        for i in 0..self.num_envs {
            // Interleaved [obs0_0, obs0_1, obs1_0, obs1_1, ...]
            buffer[i*2] = self.positions[i];
            buffer[i*2 + 1] = self.velocities[i];
        }
    }
    
    fn write_rewards(&self, buffer: &mut [f32]) {
        buffer[..self.num_envs].copy_from_slice(&self.rewards);
    }
    
    fn write_terminals(&self, buffer: &mut [u8]) {
        buffer[..self.num_envs].copy_from_slice(&self.terminals);
    }
    
    fn write_truncations(&self, buffer: &mut [u8]) {
        buffer[..self.num_envs].copy_from_slice(&self.truncations);
    }
}
```

## Moving to SIMD (Optimization)
To achieve "PufferLib Killer" speeds, replace the loop in `step` with SIMD operations using `std::simd` (nightly) or the provided SIMD helpers in `operant-core` if available. Each SIMD instruction will update 8 environments at once.
