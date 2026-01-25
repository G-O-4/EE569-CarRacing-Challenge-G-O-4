# EE569 Deep Learning: CarRacing-v3 RL Challenge 🏎️💨

**Course:** EE569 Deep Learning  
**Due Date:** 29 December 2025, 23:59  
**Weight:** 6 marks (out of 100)  
**Group Size:** 2-3 students  

## 📋 Overview

Your mission is to implement and train a Reinforcement Learning agent to master the **CarRacing-v3** environment in Gymnasium. You may use any algorithm covered in class (DQN, TD3, SAC, World Models, or CQL).

**Goal:** Achieve professional-level racing performance while minimizing sample complexity.

### 🎯 Performance Thresholds
- **Success:** Average score > **700** over 3 evaluation episodes.
- **Excellence:** Average score > **800** with minimal environment interactions.

### 🚫 Restrictions
1. **Inputs:** Must work with pixel inputs (84×84).
2. **Environment:** No modifications to the environment logic allowed (beyond standard preprocessing like resizing/stacking).

---

## 🚀 Getting Started (Baseline)

This repository provides a **DQN baseline implementation** to get you started.

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Nuri-benbarka/EE569-CarRacing-Challenge.git
cd car_racing

# Create environment (recommended)
conda create -n car_racing python=3.10
conda activate car_racing

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

Train the agent using `dqn_car_racing.py`. This script includes Aim logging.

```bash
python dqn_car_racing.py
```

To view training logs with Aim:
```bash
aim up
```

### 3. Evaluation

Run the inference script to evaluate your trained model and generate videos.

```bash
# Run 3 evaluation episodes (default uses checkpoints/best_model.pth)
python inference.py --episodes 3

# Save video of the run
python inference.py --episodes 1 --save-video

# Evaluate a specific checkpoint without rendering (faster)
python inference.py --checkpoint checkpoints/final_model.pth --no-render
```

---

## 🏁 High-Score Track (Recommended): SAC + DrQ (Continuous Control)

The baseline DQN uses a **very coarse 5-action discretization**, which can’t smoothly steer while accelerating/braking. To push toward **850–950+**, this repo includes a **pixel-based SAC + DrQ** implementation in `train.py` (continuous 2D actions: steer + accel).

### 1. Training (SAC + DrQ)

```bash
# Train for ~2M env steps (good first budget for 800+; extend to ~3M if close)
python train.py --total-steps 2000000 --seed 1

# Optional: enable mixed precision on CUDA GPUs
python train.py --total-steps 2000000 --seed 1 --use-amp

# Optional: use RGB instead of grayscale (more compute/memory)
python train.py --total-steps 2000000 --seed 1 --rgb --frame-stack 3
```

**Outputs:**
- Checkpoints in `./checkpoints/`: `best_sac_drq.pth`, `last_sac_drq.pth`, `final_sac_drq.pth`
- Aim logs (if enabled): run `aim up`

### 2. Evaluation / Video (SAC + DrQ)

```bash
# Evaluate best checkpoint (3 episodes, no rendering)
python inference.py --checkpoint checkpoints/best_sac_drq.pth --algo sac_drq --episodes 3 --no-render --seed 1

# Save a video of the best checkpoint
python inference.py --checkpoint checkpoints/best_sac_drq.pth --algo sac_drq --episodes 1 --save-video --video-dir ./videos --seed 1
```

### 3. Extend a run (resume training)

```bash
# Continue training from a previous checkpoint up to a higher total step count
python train.py --resume checkpoints/last_sac_drq.pth --total-steps 3000000 --seed 1
```

---

## 🔬 Suggested Experiment Matrix (6+ runs)

Keep runs **step-budgeted** and early-stop stalled configs (balanced score + sample-efficiency).

| Run | Change | Example command | Step budget |
|-----|--------|------------------|------------|
| 1 | SAC (grayscale) + DrQ (default) | `python train.py --total-steps 2000000 --seed 1` | 2.0M |
| 2 | Reward scale sweep | `python train.py --total-steps 2000000 --seed 1 --reward-scale 0.1` | 2.0M |
| 3 | Batch / updates-per-step sweep | `python train.py --total-steps 2000000 --seed 1 --batch-size 256 --updates-per-step 4` | 2.0M |
| 4 | Encoder size sweep | `python train.py --total-steps 2000000 --seed 1 --feature-dim 128 --hidden-dim 1024` | 2.0M |
| 5 | RGB vs grayscale | `python train.py --total-steps 2000000 --seed 1 --rgb --frame-stack 3` | 2.0M |
| 6 | Action repeat | `python train.py --total-steps 2000000 --seed 1 --action-repeat 2` | 2.0M |
| 7 | Seed robustness | `python train.py --total-steps 1000000 --seed 2` | 1.0–2.0M |

---

## 📊 Deliverables

Submit a single ZIP file containing:

### 1. Code 💻
- `train.py` (or your modified `dqn_car_racing.py`) - Main training script.
- `inference.py` - Evaluation & video recording.
- `requirements.txt` - Complete dependency list with versions.
- **Micro-objective:** Code must be clean, modular, and well-commented.

### 2. Report 📄
A **4-6 page PDF** (`report.pdf`) covering:
- **Method Selection:** Why this algorithm? Theoretical advantages?
- **Implementation Details:** Architectures, loss functions, hyperparameters.
- **Experimental Design:** Table of **6+ experiments** (varied hyperparams, architectures, or algorithms).
- **Results:** Learning curves, convergence analysis, final performance.
- **Comparison:** Sample efficiency vs. final performance.
- **Ablation Study:** What components mattered most? (e.g., target networks, epsilon decay).
- **Challenges:** Negative results and what you learned from failures.

### 3. Aim Experiment Logs 📈
Include your `.aim` directory or a comprehensive export of your logs to prove reproducibility.

### 4. Video 🎥
- `best_run.mp4`: A video of your best evaluation episode (score > 700).
- Must demonstrate smooth, stable driving.
- Use the `--save-video` flag in `inference.py`.

### 5. README 📖 
- Clear instructions to reproduce your results.
- Exact training and evaluation commands.
- Brief description of your algorithm and findings.

---

## 🏆 Grading Scheme (6 Marks Total)

### Base Grade
| Component | Marks | Criteria |
|-----------|-------|----------|
| **Method Implementation** | 1.0 | Correctness and completeness of the RL algorithm. |
| **Algorithmic Understanding** | 1.0 | Quality of report and theoretical justification. |
| **Experimental Rigor** | 1.0 | 6+ meaningful experiments & ablation studies. |
| **Performance** | 1.0 | Achieving > 700 average reward. |
| **Reproducibility** | 1.0 | aim logs, clear README, clean code. |
| **Video Quality** | 1.0 | Demonstrates learned behavior (smooth driving). |

### Bonus Marks (Maximum +2)
**🏅 Bonus 1: Highest Score (+1)**
- Awarded to the group with the **highest evaluation score** (avg over 3 episodes).
- Must be > 700 to qualify. Winner takes all (no ties).

**🌱 Bonus 2: Efficiency Champion (+1)**
- Awarded to the group with the **lowest total environment interactions** (sum of steps across all experiments) while still achieving a score > 700.
- Proven via `aim_run['hparams']['total_steps']` or logs.

## توّكّل على الله، واستمتع بتعليم مركبتك القيادة! 💨🚗

