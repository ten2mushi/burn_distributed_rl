# API Reference

This section provides clear guidance on the key symbols and definitions used in Operant.

## Key Traits

### `operant_core::Environment`
The core trait for all vectorized environments.

| Method | Description |
|--------|-------------|
| `num_envs(&self) -> usize` | Returns the number of parallel instances. |
| `reset(&mut self, seed)` | Resets ALL environments (hard reset). |
| `step(&mut self, actions)` | Steps all envs and **auto-resets** terminated ones. |
| `step_no_reset(&mut self, actions)` | Steps envs **without** auto-reset. Essential for Value-Based RL correctness. |
| `write_observations(...)` | Writes current state to a flat buffer (Zero-Copy). |

### `operant_core::LogData`
Trait for collecting metrics (reward, length) efficiently across threads.
- `merge(&mut self, other)`: Aggregates logs from parallel workers.

## Key Structures

### `operant_core::StepResult<'a>`
A zero-copy view into the environment's state after a step.
- **Fields**:
  - `observations`: Flat slice of all observations.
  - `rewards`: Slice of rewards.
  - `terminals`: Slice of terminal flags (u8).
  - `truncations`: Slice of truncation flags (u8).
- **Usage**: Returned by `step_no_reset_with_result` to avoid copying data.

### `operant_core::ResetMask`
Efficient bitmask for handling "sparse" resets after `step_no_reset`.
- **API**:
  - `ResetMask::from_done_flags(terminals, truncations)`: Creates mask from boolean buffers.
  - `iter_set()`: Returns an iterator over indices that need reset (O(k) complexity).
- **Purpose**: Avoids iterating over all 8092 environments when only 5 need resetting.

## Type Definition Guide

- **`f32x8`**: The workhorse SIMD type (AVX2). Represents 8 float values.
- **`u8` for Booleans**: We use `u8` (0 or 1) for boolean flags to ensure compact storage and easy export to NumPy/PyTorch (which often expect bytes).
- **`step_id` / `ticks`**: Usually a `Vec<u32>` inside the env struct to track episode lengths for truncation.
