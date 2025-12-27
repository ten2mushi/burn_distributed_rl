# Performance Optimizations & "Hidden" Semantics

To unlock the full potential of Operant (and beat PufferLib benchmarks), you must adhere to specific optimization patterns.

## 1. Memory Layout: The SoA Rule

**Insight**: The most common bottleneck is cache misses.
**Rule**: Always use Struct-of-Arrays (SoA).

- **Bad**: `Vec<Struct>` (Array of Structs). Accessing `env[i].x` then `env[i+1].x` skips over `env[i].y`, polluting the cache line.
- **Good**: `Struct<Vec>` (Struct of Arrays). `x[i]` and `x[i+1]` are adjacent.

**Hidden Meaning**: When you define `x: Vec<f32>`, you are implicitly defining a "lane" of data that fits perfectly into a CPU register.

## 2. SIMD Thinking (The 8x Speedup)

Operant targets AVX2 (256-bit), which holds 8 `f32` values.
- **Scalar Loop**: Executes 1 env per cycle (theoretical max).
- **SIMD Loop**: Executes 8 envs per cycle.

**How to implement**:
```rust
// Instead of:
for i in 0..num_envs {
    x[i] += v[i];
}
// Do (pseudo-SIMD):
for chunk in 0..num_envs/8 {
    let x_vec = f32x8::load(x, chunk);
    let v_vec = f32x8::load(v, chunk);
    let new_x = x_vec + v_vec; // CPU adds 8 floats AT ONCE
    new_x.store(x, chunk);
}
```

## 3. Branchless Programming

**Insight**: `if` statements kill instruction pipelines, especially inside loops.
**Rule**: Use bitmasks and `select` instructions.

**Example**: Reset logic.
- **Branching (Slow)**:
```rust
if position > limit {
    terminal = true;
    reward = 0.0;
} else {
    terminal = false;
    reward = 1.0;
}
```
- **Branchless (Fast)**:
```rust
let limit_mask = position.simd_gt(limit_vec); // generic mask
let reward = limit_mask.select(0.0, 1.0);     // hardware blend instruction
```

## 4. Parallelism Scaling

**Insight**: SIMD saturates one core. Modern CPUs have 16+ cores.
**Rule**: Chunk your environments across threads using `rayon`.

Operant's `step_parallel` divides the `num_envs` into chunks (e.g., 1024 envs per thread).
- **Sweet Spot**: Usually 2000-4000 environments per thread depending on complexity.
- **Caution**: Thread synchronization overhead can hurt small batches (< 1024 envs).

## 5. Pre-allocated Buffers

**Insight**: Allocating memory (`Vec::new()`) in the hot loop is a performance killer.
**Rule**: Allocate EVERYTHING in `new()`.
- Observations buffer
- Reset masks
- Temporary computation buffers

**Hidden Semantics of `step_no_reset_with_result`**:
This method exists specifically to return a reference to *internal, pre-allocated* buffers. It avoids creating a new struct on the heap/stack every step, returning a `StepResult` that points directly to the persistent state.
