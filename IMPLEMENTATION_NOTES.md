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
- **Encoder**: Atari-style CNN (3 conv layers) → 50-dim feature vector
- **Actor**: Gaussian policy with tanh squashing, outputs 2D actions
- **Critic**: Twin Q-networks with target networks (standard SAC)
- **Replay Buffer**: Stores observations as uint8 (memory efficient)
- **DrQ Augmentation**: Random shift (pad=4) applied during training

**Key hyperparameters** (in `SACConfig`):
```python
replay_size: 30_000      # Reduced for memory-constrained systems
batch_size: 256
gamma: 0.99
tau: 0.01                # Soft target update
actor_lr: 1e-4
critic_lr: 1e-4
alpha_lr: 1e-4           # Automatic entropy tuning
drq_pad: 4               # DrQ augmentation padding
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

**Key hyperparameters**:
```python
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
```

### 4. Inference Updates (inference.py)

Updated to support all three algorithms:

| Checkpoint format | Algorithm | Detection |
|-------------------|-----------|-----------|
| `.pth` with `model_state_dict` | DQN | Auto |
| `.pth` with `algo: "sac_drq"` | SAC+DrQ | Auto |
| `.zip` | PPO (SB3) | Auto |

Can also force algorithm with `--algo dqn|sac_drq|ppo`.

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

### 1. PPO Performance Issue (HIGH PRIORITY)

**Problem**: PPO achieves only ~280 mean reward vs expected ~870.

**Possible causes**:
1. **Environment wrapper mismatch**: The SB3 wrapper might not be correctly configured
2. **Observation format**: SB3 expects channel-last, our wrapper provides this but may have issues
3. **Frame stacking**: VecFrameStack behavior might differ from our manual stacking
4. **Hyperparameters**: Default values may need tuning for our specific setup

**Debugging steps needed**:
1. Verify observation shapes at each stage
2. Compare with working SB3 CarRacing examples
3. Check if the action space is being handled correctly
4. Try using SB3's built-in `make_vec_env` instead of custom wrappers
5. Test with SB3's default preprocessing (no custom wrappers)

**Potential fix**: Use SB3's native environment handling:
```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
# Or use SB3's built-in CarRacing support
```

### 2. SAC Training Not Completed

**Problem**: Best SAC run was interrupted at 607k steps due to memory constraints.

**Solution**: Resume training to 2-3M steps:
```bash
python train.py --resume checkpoints/run01_default_s1/last_sac_drq.pth \
  --total-steps 3000000 --seed 1 \
  --checkpoint-dir checkpoints/run01_default_s1
```

### 3. Memory Constraints

**Problem**: Training on systems with limited RAM (8GB) causes OOM kills.

**Current mitigations**:
- Reduced replay buffer to 30k (SAC)
- PPO uses no replay buffer

**Recommendations**:
- Use WSL2 with increased memory allocation (6GB+ RAM, 8GB+ swap)
- Monitor memory during training: `watch -n 5 free -h`
- Close other applications during training
- Consider cloud training for longer runs

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

# Resume interrupted training
python train.py --resume checkpoints/run01_default_s1/last_sac_drq.pth \
  --total-steps 3000000 --seed 1 \
  --checkpoint-dir checkpoints/run01_default_s1

# With reduced memory (if needed)
python train.py --total-steps 2000000 --seed 1 --replay-size 20000 \
  --checkpoint-dir checkpoints/sac_lowmem
```

### PPO Training (needs debugging)

```bash
python train_ppo.py --total-timesteps 1000000 --seed 1 \
  --checkpoint-dir checkpoints/ppo_run01
```

### Evaluation / Inference

```bash
# SAC model
python inference.py --checkpoint checkpoints/run01_default_s1/best_sac_drq.pth \
  --algo sac_drq --episodes 5 --no-render

# PPO model
python inference.py --checkpoint checkpoints/ppo_run01/best_model.zip \
  --algo ppo --episodes 5 --no-render

# With video recording
python inference.py --checkpoint <path> --algo <algo> \
  --episodes 1 --save-video --video-dir ./videos
```

### DQN Baseline (original)

```bash
python dqn_car_racing.py
python inference.py --checkpoint checkpoints/best_model.pth --algo dqn --episodes 5
```

---

## File Structure

```
├── README.md                 # Original project README
├── IMPLEMENTATION_NOTES.md   # This file
├── DEV_NOTES_SAC_DRQ.md     # Detailed SAC+DrQ implementation notes
├── requirements.txt          # Dependencies
│
├── carracing_env.py          # Environment wrappers (shared)
├── dqn_car_racing.py         # Original DQN baseline
├── sac_drq.py                # SAC + DrQ agent implementation
├── train.py                  # SAC + DrQ training script
├── train_ppo.py              # PPO training script (needs work)
├── inference.py              # Unified evaluation script
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

**Immediate priorities**:

1. **Debug PPO** - Figure out why it's performing so poorly compared to benchmarks
2. **Continue SAC training** - Resume the interrupted run to 2-3M steps to push past 900

**If time permits**:

3. Try hyperparameter sweeps on the better-performing algorithm
4. Test with RGB observations instead of grayscale
5. Experiment with action repeat values

**Target**: Achieve consistent 900+ scores with either SAC or PPO.
