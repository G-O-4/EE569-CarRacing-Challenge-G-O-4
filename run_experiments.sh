#!/bin/bash
# CarRacing Training Experiments
# Optimized for: WSL2 with ~6GB RAM, RTX 3050 GPU
# Run these commands to train agents targeting 900+ score

set -e

echo "========================================"
echo "CarRacing 900+ Training Experiments"
echo "========================================"
echo "Target: WSL2 (~6GB RAM), RTX 3050 GPU"
echo ""

# Create checkpoint directories
mkdir -p checkpoints/sac_run01
mkdir -p checkpoints/ppo_baseline
mkdir -p checkpoints/ppo_with_norm

case "${1:-help}" in
    sac)
        echo "[SAC+DrQ] Starting training to 3M steps..."
        echo "Using: --use-amp, replay_size=70k (~3.9GB RAM) - optimized for 6GB system"
        echo "Expected: ~870-890 at 2M steps, 900+ at 3M steps"
        echo ""
        python train.py \
            --total-steps 3000000 \
            --seed 1 \
            --use-amp \
            --checkpoint-dir checkpoints/sac_run01 \
            --run-name sac_run01 \
            --no-aim
        ;;
    
    sac-lowmem)
        echo "[SAC+DrQ Low Memory] Training with reduced replay buffer..."
        echo "Using: replay_size=50k (~2.8GB RAM), --use-amp - if OOM with default"
        echo ""
        python train.py \
            --total-steps 3000000 \
            --seed 1 \
            --use-amp \
            --replay-size 50000 \
            --checkpoint-dir checkpoints/sac_lowmem \
            --run-name sac_lowmem \
            --no-aim
        ;;
    
    ppo-baseline)
        echo "[PPO Baseline] Testing without reward normalization..."
        echo "Using: n_envs=6 (safe for 6GB RAM), ent_coef=0.01"
        echo ""
        python train_ppo.py \
            --total-timesteps 1000000 \
            --n-envs 6 \
            --ent-coef 0.01 \
            --no-norm-reward \
            --seed 42 \
            --checkpoint-dir checkpoints/ppo_baseline \
            --run-name ppo_baseline \
            --no-aim
        ;;
    
    ppo-norm)
        echo "[PPO with Norm] Training with reward normalization..."
        echo "Using: n_envs=6, ent_coef=0.01, norm-reward"
        echo ""
        python train_ppo.py \
            --total-timesteps 1000000 \
            --n-envs 6 \
            --ent-coef 0.01 \
            --seed 42 \
            --checkpoint-dir checkpoints/ppo_with_norm \
            --run-name ppo_with_norm \
            --no-aim
        ;;
    
    ppo-extended)
        echo "[PPO Extended] Training for 2M steps..."
        echo "Using: n_envs=6, ent_coef=0.01, 2M timesteps"
        echo ""
        python train_ppo.py \
            --total-timesteps 2000000 \
            --n-envs 6 \
            --ent-coef 0.01 \
            --seed 42 \
            --checkpoint-dir checkpoints/ppo_extended \
            --run-name ppo_extended \
            --no-aim
        ;;
    
    sac-ablation)
        echo "[SAC Ablation] Testing with action-repeat=2 and reward-scale=0.1..."
        echo "Using: --use-amp, action-repeat=2, reward-scale=0.1"
        echo ""
        python train.py \
            --total-steps 2000000 \
            --action-repeat 2 \
            --reward-scale 0.1 \
            --use-amp \
            --seed 1 \
            --checkpoint-dir checkpoints/sac_ablation \
            --run-name sac_ablation \
            --no-aim
        ;;
    
    eval-sac)
        echo "[Evaluation] Evaluating SAC model..."
        python inference.py \
            --checkpoint checkpoints/sac_run01/best_sac_drq.pth \
            --algo sac_drq \
            --episodes 10 \
            --no-render \
            --seed 1
        ;;
    
    eval-ppo)
        echo "[Evaluation] Evaluating PPO model..."
        python inference.py \
            --checkpoint checkpoints/ppo_baseline/best_model.zip \
            --algo ppo \
            --episodes 10 \
            --no-render \
            --seed 1
        ;;
    
    monitor)
        echo "[Monitor] Watching memory usage..."
        echo "Press Ctrl+C to stop"
        watch -n 5 'free -h && echo "" && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi not available"'
        ;;
    
    help|*)
        echo "Usage: ./run_experiments.sh <command>"
        echo ""
        echo "Training Commands (optimized for 6GB RAM + RTX 3050):"
        echo "  sac           - Train SAC+DrQ to 3M steps with AMP (best path to 900+)"
        echo "  sac-lowmem    - SAC with smaller replay buffer (if OOM issues)"
        echo "  ppo-baseline  - Test PPO without reward normalization (1M steps)"
        echo "  ppo-norm      - Train PPO with reward normalization (1M steps)"
        echo "  ppo-extended  - Train PPO for 2M steps"
        echo "  sac-ablation  - SAC with action-repeat=2 and reward-scale=0.1"
        echo ""
        echo "Evaluation Commands:"
        echo "  eval-sac      - Evaluate best SAC model (10 episodes)"
        echo "  eval-ppo      - Evaluate best PPO model (10 episodes)"
        echo ""
        echo "Utility Commands:"
        echo "  monitor       - Watch RAM and GPU memory usage"
        echo ""
        echo "Recommended order:"
        echo "  1. ./run_experiments.sh sac          # Start SAC (fastest to 900+)"
        echo "  2. ./run_experiments.sh ppo-baseline # Test PPO fixes (after SAC completes)"
        echo ""
        echo "Memory estimates (for ~6GB available RAM):"
        echo "  SAC (default):  ~3.9GB RAM (70k buffer) + ~2GB VRAM"
        echo "  SAC (lowmem):   ~2.8GB RAM (50k buffer) + ~2GB VRAM"
        echo "  PPO (6 envs):   ~800MB RAM + ~1.5GB VRAM"
        echo ""
        echo "Larger replay buffer = better sample diversity = faster learning!"
        echo ""
        ;;
esac
