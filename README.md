# EE569 Deep Learning: CarRacing-v3 RL Challenge

**Course:** EE569 Deep Learning  
**Institution:** University of Tripoli, Libya  
**Instructor:** Dr. Nuri Benbarka  
**Team:** Ahmed Mohamed Bakory, Faisal Ali Elhouderi, Muhammed Ali Muhmoud  
**Repository:** https://github.com/G-O-4/EE569-CarRacing-Challenge-G-O-4

---

## Results Summary

| Algorithm | Mean Reward | Std | Status |
|-----------|-------------|-----|--------|
| **SAC+DrQ** | **861.47** | 52.73 | Exceeds 800 threshold |
| PPO (Gaussian) | 306.92 | 42.89 | Needs improvement |
| SAC+DrQ (modified) | 209.51 | 23.00 | Broken, reverted |

Our best model (SAC+DrQ) exceeds both the **700 success** and **800 excellence** thresholds.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/G-O-4/EE569-CarRacing-Challenge-G-O-4.git
cd EE569-CarRacing-Challenge-G-O-4

# Create environment
conda create -n car_racing python=3.10
conda activate car_racing

# Install dependencies
pip install -r requirements.txt
```

### Training

**SAC+DrQ (Recommended):**
```bash
python train.py --total-steps 2000000 --seed 1 --checkpoint-dir checkpoints/sac_run --run-name sac_baseline
```

**PPO with Beta Distribution:**
```bash
python train_ppo.py --total-timesteps 1000000 --seed 42 --n-envs 8 --batch-size 256 --learning-rate 2.5e-4 --clip-range 0.1 --checkpoint-dir checkpoints/ppo_run --run-name ppo_beta
```

**PPO with Gaussian (fallback):**
```bash
python train_ppo.py --total-timesteps 1000000 --seed 42 --use-gaussian --checkpoint-dir checkpoints/ppo_gauss --run-name ppo_gaussian
```

### Evaluation

```bash
# SAC+DrQ
python inference.py --checkpoint checkpoints/sac_run/best_sac_drq.pth --algo sac_drq --episodes 3 --no-render --seed 1

# PPO
python inference.py --checkpoint checkpoints/ppo_run/best_model.zip --algo ppo --episodes 3 --no-render --seed 1

# Save video
python inference.py --checkpoint checkpoints/sac_run/best_sac_drq.pth --algo sac_drq --episodes 1 --save-video --video-dir ./videos
```

### View Logs

```bash
aim up
```

---

## Recent Fixes Applied

The following fixes were applied on January 27, 2026. See [status/CHANGES.md](status/CHANGES.md) for full details.

### SAC+DrQ Reverted to Working Configuration

The modifications in commits `9ceae12` and `c86d66f` broke SAC performance (861 → 209). Changes reverted:

| Parameter | Broken | Fixed |
|-----------|--------|-------|
| `replay_size` | 70,000 | 30,000 |
| `updates_per_step` | 2 | 1 |
| `feature_dim` | 64 | 50 |
| AMP (mixed precision) | enabled | removed |

**Root cause:** AMP refactoring broke gradient flow in the `update()` method by reusing stale encoder features.

### PPO Upgraded to Beta Distribution

Based on research (see [status/research.md](status/research.md)):
- Standard Gaussian PPO: ~897 avg reward
- **Beta Distribution PPO: ~913 avg reward** (state of the art)

New features:
- `BetaActorCriticPolicy` - naturally bounded to [-1, 1] action space
- `AimWriter` - reliable metric logging (fixes missing metrics issue)
- Optimized hyperparameters: `n_envs=8`, `batch_size=256`, `clip_range=0.1`

### Deleted `run_experiments.sh`

The script had `--no-aim` flag by default, causing runs to not be tracked. All commands should now be run directly in the terminal.

---

## Planned Runs

These runs have been designed but **not yet executed**. All will be tracked by Aim.

### Bonus 1: Highest Score

| Run | Command | Expected |
|-----|---------|----------|
| 1 | `python train.py --total-steps 2000000 --seed 1 --checkpoint-dir checkpoints/sac_2M_s1 --run-name sac_2M_baseline` | 850-880 |
| 2 | `python train_ppo.py --total-timesteps 2000000 --seed 42 --n-envs 8 --batch-size 256 --learning-rate 2.5e-4 --clip-range 0.1 --checkpoint-dir checkpoints/ppo_beta_2M --run-name ppo_beta_2M` | 880-920 |

### Bonus 2: Efficiency Champion

| Run | Command | Expected |
|-----|---------|----------|
| 3 | `python train.py --total-steps 1000000 --seed 1 --checkpoint-dir checkpoints/sac_1M_eff --run-name sac_1M_efficiency` | 750-800 |
| 4 | `python train_ppo.py --total-timesteps 1000000 --seed 42 --n-envs 8 --batch-size 256 --learning-rate 2.5e-4 --clip-range 0.1 --checkpoint-dir checkpoints/ppo_beta_1M --run-name ppo_beta_1M_efficiency` | 750-850 |
| 5 | `python train.py --total-steps 500000 --seed 1 --action-repeat 2 --checkpoint-dir checkpoints/sac_500k_ar2 --run-name sac_500k_action_repeat` | 700-800 |

### Recommended Order

1. **Run 3** (SAC 1M) - Quick validation that fixes work
2. **Run 4** (Beta-PPO 1M) - Test PPO improvements
3. **Run 1** (SAC 2M) - Main Bonus 1 attempt
4. **Run 5** (SAC 500k AR2) - Bonus 2 efficiency test
5. **Run 2** (Beta-PPO 2M) - Best shot at highest score

---

## Completed Runs

| Run | Algorithm | Steps | Reward | Duration | Notes |
|-----|-----------|-------|--------|----------|-------|
| 1 | SAC+DrQ (initial) | 1.91M | **861.47** | 25h 15m | Best result, tracked in Aim |
| 2 | PPO (Gaussian) | 1.00M | 306.92 | 3h 35m | Aim tracked (system metrics only) |
| 3 | SAC+DrQ (modified) | ~1.7M | 209.51 | ~38h | Not tracked (--no-aim), reverted |
| 4-6 | SAC+DrQ | 20-40k | N/A | N/A | Killed by OOM |

---

## Project Structure

```
├── train.py           # SAC+DrQ training script
├── train_ppo.py       # PPO training script (Beta/Gaussian)
├── inference.py       # Evaluation and video recording
├── sac_drq.py         # SAC+DrQ agent implementation
├── carracing_env.py   # Environment wrappers
├── dqn_car_racing.py  # DQN baseline (from original repo)
├── requirements.txt   # Dependencies
├── status/
│   ├── CHANGES.md           # Detailed fix documentation
│   ├── performed_runs.txt   # Run history and plans
│   ├── research.md          # PPO/SAC research notes
│   └── IMPLEMENTATION_NOTES.md
├── report/
│   ├── report.tex     # LaTeX report
│   └── images/        # Aim screenshots and figures
└── checkpoints/       # Model checkpoints (gitignored)
```

---

## Hardware Requirements

Tested on:
- **GPU:** NVIDIA RTX 3050 (4GB VRAM)
- **RAM:** 8GB (WSL2: ~5.8GB available)

Memory usage:
| Algorithm | RAM | GPU |
|-----------|-----|-----|
| SAC (30k buffer) | ~1.7 GB | ~1.5 GB |
| Beta-PPO (8 envs) | ~1.2 GB | ~1.5 GB |

---

## References

- [Soft Actor-Critic (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)
- [DrQ: Image Augmentation (Kostrikov et al., 2020)](https://arxiv.org/abs/2004.13649)
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Beta Policy for Continuous Control (Chou et al., 2017)](https://arxiv.org/abs/1707.02152)

---

## Original Assignment

This repository is a fork of [Dr. Nuri's EE569-CarRacing-Challenge](https://github.com/Nuri-benbarka/EE569-CarRacing-Challenge).

### Performance Thresholds
- **Success:** Average score > 700 over 3 evaluation episodes
- **Excellence:** Average score > 800 with minimal environment interactions

### Bonus Marks
- **Bonus 1:** Highest evaluation score among groups
- **Bonus 2:** Lowest total environment interactions while achieving >700
