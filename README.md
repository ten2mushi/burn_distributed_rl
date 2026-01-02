## Locally distributed training of rl agents

several crates:
    - distributed_rl: rl agents training
    - environments: environments for rl agents:
        - quadcopter: quadcopter environment
        - rf: radio frequency spectrum simulation for reinforcement learning

needs rust nightly (may need to use rustup default nightly)

install:
```bash
git clone https://github.com/ten2mushi/burn_distributed_rl.git
cd burn_distributed_rl
cargo build --release
```

run examples:
```bash
cargo run --release -- ppo
cargo run --release -- recurrent-ppo
```

Notes:

- under development.
    - discrete feedforward ppo training cartpole between 500 - 1000 steps. Still some oscillations to investigate.

- uses a fork of operant in order to support non-auto-reset API: added those methods to the Environment trait:
    - step_no_reset: step all environments without auto-reset. Terminal flags are preserved
    - step_no_reset_with_result: combined step + read operation returning a `StepResult` for zero-copy access
    - reset_envs: reset specific environments identified by a bitmask
    - supports_no_reset: true if the environment implements the non-auto-reset methods


changelog:
        - refactor to ppo runner with cleaner abstractions and compositions
        - kl early stop
        - global advantage normalisation
        - value loss clipping
        - lr annealing
        - gradient clipping
        - minibatch sequence shuffling
    - added quadcopter environment (crazyflie 2.0, x-configuration, rpm action space, 12D observation space, single agent, trajectory tracking, velocity control, batch environments, cpu simd, decoupled physics/control frequencies, episode termination, ground effect, kinematic observations, action buffer, supports_no_reset)
    - added rf environment (radio frequency spectrum simulation for reinforcement learning)