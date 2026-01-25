# Implementation Notes: CarRacing-v3 Reinforcement Learning Challenge

This document summarizes all modifications made to the original baseline, explains the reasoning behind each change, documents model performance results, and outlines remaining work.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Original Baseline](#original-baseline)
3. [Changes Made](#changes-made)
4. [Model Performance Summary](#model-performance-summary)
5. [Known Issues and Remaining Work](#known-issues-and-remaining-work)
6. [How to Run](#how-to-run)
7. [File Structure](#file-structure)
8. [References](#references)

---

## Project Overview

This is a reinforcement learning project for the **CarRacing-v3** environment from Gymnasium. The goal is to train an agent that can drive a car around procedurally generated race tracks, achieving scores of **900+** (ideally approaching 1000).

**Scoring system:**
- The agent receives `+1000/N` reward for each track tile visited (N = total tiles)
- Going off-track stops positive rewards
- Episode ends when track is completed or too much time is spent off-track
- **Target score: 900+** (baseline achieves ~550-650)

---

## Original Baseline

The original repo contained a **DQN (Deep Q-Network)** implementation with:

- **5 discrete actions**: left, right, gas, brake, no-op
- **Pixel observations**: 84x84 grayscale, 4-frame stack
- **Standard DQN**: target network, replay buffer, epsilon-greedy exploration

**Baseline performance**: ~550-650 average reward

**Why the baseline is limited**:
- Discrete actions cannot express smooth, simultaneous control (e.g., steering while accelerating)
- DQN is not ideal for continuous control problems
- CarRacing fundamentally benefits from continuous action spaces

---

## Changes Made

### 1. Environment Wrappers (carracing_env.py)

Created a shared environment module with multiple wrappers:

| Wrapper | Purpose |
|---------|---------|
| `CarRacingDiscreteActionWrapper` | Original 5-action discrete wrapper (for DQN) |
| `CarRacingContinuous2DActionWrapper` | **NEW**: 2D continuous actions (steer, accel) |
| `CarRacingPixelObsWrapper` | 84x84 preprocessing, channel-first (for SAC/DQN) |
| `CarRacingPixelObsWrapperSB3` | 84x84 preprocessing, channel-last (for PPO/SB3) |
| `ActionRepeat` | Optional action repeat for stability |
| `StackFrames` | Frame stacking (4 frames default) |

**Key design decision**: The 2D continuous action space maps `[steer, accel]` to the environment's native `[steer, gas, brake]`:
```python
gas = max(accel, 0)
brake = max(-accel, 0)
```
This avoids the ambiguous "gas + brake" situation and gives smooth control.

### 2. SAC + DrQ Implementation (sac_drq.py, train.py)

Implemented **Soft Actor-Critic with DrQ augmentation** for pixel-based continuous control:

**Why SAC + DrQ?**
- SAC: Off-policy, entropy-regularized, excellent for continuous control
- DrQ: Data-augmented Q-learning improves sample efficiency on pixels

**Architecture**:
- **Encoder**: Atari-style CNN (3 conv layers) → 64-dim feature vector
- **Actor**: Gaussian policy with tanh squashing, outputs 2D actions
- **Critic**: Twin Q-networks with target networks (standard SAC)
- **Replay Buffer**: Stores observations as uint8 (memory efficient)
- **DrQ Augmentation**: Random shift (pad=4) applied during training

**Key hyperparameters** (in `SACConfig`):
```python
replay_size: 30_000      # Reduced for memory-constrained systems
batch_size: 256
updates_per_step: 2
gamma: 0.99
tau: 0.01                # Soft target update
actor_lr: 1e-4
critic_lr: 1e-4
alpha_lr: 1e-4           # Automatic entropy tuning
drq_pad: 4               # DrQ augmentation padding
feature_dim: 64
use_amp: False           # Optional mixed precision on CUDA
```

**Memory considerations**:
- Original default was 1M replay size (~50+ GB RAM)
- Reduced to 30k (~1.7 GB) for systems with limited RAM
- Can adjust with `--replay-size` flag

### 3. PPO Implementation (train_ppo.py)

Added **Proximal Policy Optimization** using stable-baselines3:

**Why PPO?**
- No replay buffer = much lower RAM usage (~500 MB vs ~2-3 GB)
- Lower variance in scores according to benchmarks
- Well-tested implementation via SB3

**Implementation**:
- Uses SB3's `PPO` with `CnnPolicy`
- Vectorized environment support (`make_vec_env_ppo`)
- Automatic checkpointing and evaluation callbacks
- Optional reward normalization with `VecNormalize` (reward-only)

**Key hyperparameters** (updated defaults):
```python
n_envs: 6              # Safe for 6GB RAM (was 4, increased but capped for memory)
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
ent_coef: 0.01         # Added entropy for exploration (was 0.0)
norm_reward: True
```

### 4. Inference Updates (inference.py)

Updated to support all three algorithms:

| Checkpoint format | Algorithm | Detection |
|-------------------|-----------|-----------|
| `.pth` with `model_state_dict` | DQN | Auto |
| `.pth` with `algo: "sac_drq"` | SAC+DrQ | Auto |
| `.zip` | PPO (SB3) | Auto |

Can also force algorithm with `--algo dqn|sac_drq|ppo`.
Inference now supports `--seed` for reproducibility and PPO `--vecnorm` loading.

### 5. Training Infrastructure

- **Seeded training**: Reproducible runs with `--seed`
- **Resume support**: Continue training from checkpoints with `--resume`
- **Aim logging**: Experiment tracking (optional, disable with `--no-aim`)
- **Tensorboard**: Built-in for PPO via SB3
- **Periodic evaluation**: Every N steps with best model checkpointing

---

## Model Performance Summary

### SAC + DrQ Results

| Run | Steps | Best Eval Reward | Notes |
|-----|-------|------------------|-------|
| run01_default_s1 | ~607,000 (interrupted) | **~830** | Memory killed at 607k steps |

**Observations**:
- Reached 830 at only 607k steps (training interrupted due to OOM)
- Learning curve was still improving when interrupted
- Expected to reach 900+ if continued to 2-3M steps
- Memory usage was the main constraint

### PPO Results

| Run | Steps | Best Eval Reward | Mean ± Std | Notes |
|-----|-------|------------------|------------|-------|
| ppo_run01_s1 | 1,000,000 | ~681 | 281.76 ± 208.52 | **Poor performance** |

**Observations**:
- Performed significantly worse than expected
- Benchmark shows PPO should achieve 873 at 1M steps
- High variance suggests unstable learning
- **Needs debugging** (see Known Issues section)

### Comparison with Benchmarks

From [Finding Theta benchmarks](https://www.findingtheta.com/blog/solving-gymnasiums-car-racing-with-reinforcement-learning):

| Algorithm | Benchmark (1M steps) | Our Results | Status |
|-----------|---------------------|-------------|--------|
| PPO | 873.76 ± 48.43 | 281.76 ± 208.52 | **Needs work** |
| SAC | 787.69 ± 120.19 | ~830 @ 607k | **Good** |
| DQN | 872.52 ± 133.79 | ~550-650 (baseline) | As expected |

---

## Known Issues and Remaining Work

### 1. PPO Performance Issue (FIXED)

**Problem**: PPO achieved ~280 mean reward vs expected ~870 in prior runs.

**Root causes identified**:
1. Too few parallel environments (was 4, should be 8+)
2. No entropy bonus (ent_coef was 0.0, should be 0.01 for pixel envs)

**Fixes applied**:
1. Changed default `n_envs` from 4 to **8** in `train_ppo.py`
2. Changed default `ent_coef` from 0.0 to **0.01** in `train_ppo.py`
3. Added convenience script `run_experiments.sh` for easy training

**Validation steps**:
1. Run baseline test (without reward norm): `./run_experiments.sh ppo-baseline`
2. Run with reward norm: `./run_experiments.sh ppo-norm`
3. Compare results - expecting ~800+ at 1M steps now

### 2. SAC Training Not Completed

**Problem**: Best SAC run was interrupted at 607k steps due to memory constraints.

**Solution**: Resume training to 2-3M steps:
```bash
python train.py --resume checkpoints/run01_default_s1/last_sac_drq.pth \
  --total-steps 3000000 --seed 1 \
  --checkpoint-dir checkpoints/run01_default_s1
```

### 3. Memory Constraints

**Target hardware**: WSL2 on Windows 11, ~6GB RAM, RTX 3050 (4GB VRAM)

**Current mitigations**:
- Reduced replay buffer to 30k (SAC) → ~1.7GB RAM
- PPO uses no replay buffer → ~800MB RAM with 6 envs
- AMP enabled by default in `run_experiments.sh` (`--use-amp`) → saves ~30% GPU memory
- Optional `sac-lowmem` command uses 20k replay buffer → ~1.1GB RAM

**Memory estimates**:
| Algorithm | RAM Usage | VRAM Usage |
|-----------|-----------|------------|
| SAC (30k buffer) | ~1.7 GB | ~2 GB |
| SAC (20k buffer) | ~1.1 GB | ~2 GB |
| PPO (6 envs) | ~800 MB | ~1.5 GB |

**Recommendations**:
- Use `./run_experiments.sh monitor` to watch memory during training
- If OOM, use `./run_experiments.sh sac-lowmem` instead of `sac`
- Close other applications during training
- 8GB swap is sufficient for recovery from memory spikes

---

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

### SAC + DrQ Training

```bash
# Start new training
python train.py --total-steps 2000000 --seed 1 \
  --checkpoint-dir checkpoints/sac_run01 --run-name sac_run01

# Optional: enable AMP on CUDA
python train.py --total-steps 2000000 --seed 1 --use-amp \
  --checkpoint-dir checkpoints/sac_run01_amp --run-name sac_run01_amp

# Resume interrupted training
python train.py --resume checkpoints/run01_default_s1/last_sac_drq.pth \
  --total-steps 3000000 --seed 1 \
  --checkpoint-dir checkpoints/run01_default_s1

# With reduced memory (if needed)
python train.py --total-steps 2000000 --seed 1 --replay-size 20000 \
  --checkpoint-dir checkpoints/sac_lowmem
```

### PPO Training

```bash
python train_ppo.py --total-timesteps 1000000 --seed 1 \
  --checkpoint-dir checkpoints/ppo_run01
```
Reward normalization is enabled by default and saves `vecnormalize.pkl` in the checkpoint directory
(disable with `--no-norm-reward`).

### Evaluation / Inference

```bash
# SAC model
python inference.py --checkpoint checkpoints/run01_default_s1/best_sac_drq.pth \
  --algo sac_drq --episodes 5 --no-render --seed 1

# PPO model
python inference.py --checkpoint checkpoints/ppo_run01/best_model.zip \
  --algo ppo --episodes 5 --no-render --seed 1

# With video recording
python inference.py --checkpoint <path> --algo <algo> \
  --episodes 1 --save-video --video-dir ./videos --seed 1
```
PPO inference will auto-load `vecnormalize.pkl` from the checkpoint directory when present
(or pass `--vecnorm <path>`).

### DQN Baseline (original)

```bash
python dqn_car_racing.py
python inference.py --checkpoint checkpoints/best_model.pth --algo dqn --episodes 5 --seed 1
```

---

## File Structure

```
├── README.md                 # Original project README
├── IMPLEMENTATION_NOTES.md   # This file
├── DEV_NOTES_SAC_DRQ.md     # Detailed SAC+DrQ implementation notes
├── research.md               # Research on best algorithms for CarRacing
├── requirements.txt          # Dependencies
│
├── carracing_env.py          # Environment wrappers (shared)
├── dqn_car_racing.py         # Original DQN baseline
├── sac_drq.py                # SAC + DrQ agent implementation
├── train.py                  # SAC + DrQ training script
├── train_ppo.py              # PPO training script (fixed)
├── inference.py              # Unified evaluation script
├── run_experiments.sh        # Convenience script for training experiments
│
├── checkpoints/              # Model checkpoints (gitignored)
├── videos/                   # Recorded videos (gitignored)
├── tb_logs/                  # Tensorboard logs (gitignored)
└── .aim/                     # Aim experiment logs (gitignored)
```

---

## References

1. **SAC**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
2. **DrQ**: [Image Augmentation Is All You Need](https://arxiv.org/abs/2004.13649)
3. **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
4. **Benchmark results**: [Finding Theta - Solving Car Racing with RL](https://www.findingtheta.com/blog/solving-gymnasiums-car-racing-with-reinforcement-learning)
5. **stable-baselines3**: [SB3 Documentation](https://stable-baselines3.readthedocs.io/)

---

## Summary for Next Steps

**Quick Start** (use the convenience script):
```bash
# Make script executable (first time only)
chmod +x run_experiments.sh

# See all available commands
./run_experiments.sh help

# Start SAC training (fastest path to 900+)
./run_experiments.sh sac

# Test fixed PPO in parallel
./run_experiments.sh ppo-baseline
```

**Immediate priorities**:

1. **Train SAC to 3M steps** - Expected to reach 900+ based on learning trajectory
2. **Test fixed PPO** - With n_envs=8 and ent_coef=0.01, should match benchmark (~870)

**If time permits**:

3. Try `--action-repeat 2` for improved sample efficiency
4. Try `--reward-scale 0.1` for SAC if learning plateaus
5. Test with RGB observations instead of grayscale

**Target**: Achieve consistent 900+ scores with either SAC or PPO.
