# Structural Philosophy & Design Principles

## The Operant Vision: Native Performance

Operant represents a paradigm shift in reinforcement learning environments. While traditional libraries often wrap existing Single-Agent environments (like Gymnasium) to create vectorized environments, **Operant implements environments natively in Rust**.

### Relation to PufferLib
You might be familiar with **PufferLib**, which revolutionized RL by efficiently wrapping standard environments to maximize throughput. Operant shares the same goal—**massive throughput for vectorized environments**—but achieves it through a fundamentally different approach:
- **PufferLib**: Highly optimized wrappers around existing Python/C environments.
- **Operant**: Native Rust implementations designed from the ground up for SIMD and cache efficiency.

Operant is designed to be the "next generation" of high-throughput training, targeting >1,000,000 Steps Per Second (SPS) on a single workstation.

## Core Pillars of Design

### 1. Struct-of-Arrays (SoA) Layout
The "hidden meaning" behind Operant's speed is its memory layout.
- **Traditional (AoS)**: `[Env1(x, v), Env2(x, v), ...]`
  - Poor cache locality for updates.
  - Hard to vectorize.
- **Operant (SoA)**: `x: [Env1, Env2, ...], v: [Env1, Env2, ...]`
  - **Sequential Access**: Updating all `x` values happens in a contiguous memory block.
  - **SIMD Ready**: You can load 8 floats into a single AVX register instantly.

### 2. SIMD-First Architecture
Operant is built to exploit Single Instruction, Multiple Data (SIMD) capabilities.
- **Batch Processing**: Instead of updating one environment at a time, Operant updates 8 (AVX2) or 16 (AVX-512) environments in a single CPU instruction.
- **Branchless Logic**: Conditional logic (like "if user crashed") is replaced by bit-masking operations, preventing pipeline stalls.

### 3. Zero-Copy Semantics
Data movement is the enemy of performance.
- **Internal State**: Kept in Rust `Vec<f32>`.
- **Python Interface**: Exposed strictly via the `PyBuffer` protocol.
- **Observation Writes**: Written directly into a pre-allocated buffer that PyTorch/NumPy can read without copying.

## Intended Usage & "Hidden" Semantics

### The `step_no_reset` Paradigm
A common pitfall in vectorized environments is the "auto-reset" ambiguity. When an environment terminates:
1. Conventional VecEnvs reset immediately and return the *new* observation (Start state of next episode).
2. This breaks Value-Based RL (like DQN/PPO rely on `bootstrap_value`), which needs the *terminal* state of the *finished* episode to calculate correct targets.

**Operant's Solution**:
- **`step()` / `step_auto_reset()`**: For Policy Gradients where strict terminal values might be approximated or handled via truncation.
- **`step_no_reset()`**: Exposes the true terminal state for correct Q-value bootstrapping. You must manually call `reset_envs()` afterwards. This is the "hidden" power user feature for algorithmic correctness.

### The Role of `ResetMask`
Partial resets are expensive if not handled correctly. `ResetMask` uses bit-packing (64 envs per u64) to handle resets efficiently.
- Iterating a list of indices is O(N).
- Checking a bitmask is O(N/64) or O(k) with hardware instruction support (trailing zeros), making sparse resets incredibly fast.
