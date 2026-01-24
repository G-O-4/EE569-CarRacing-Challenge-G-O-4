---
name: Push to 900+ score
overview: Resume the existing 607k-step run to 2-3M steps (primary path to 900+), then optionally run a tuned variant with reward scaling if time permits.
todos:
  - id: resume-2m
    content: Resume run01 from last checkpoint to 2M total steps
    status: pending
  - id: eval-2m
    content: "Evaluate best checkpoint at 2M steps (target: 900+)"
    status: pending
  - id: extend-3m
    content: (If needed) Extend to 3M steps if score is 870-890
    status: pending
  - id: final-video
    content: Record video of best scoring checkpoint
    status: pending
isProject: false
---

# Push from 830 to 900+ (or 1000)

## Why 830 at 607k is promising

Your agent learned to drive reasonably well in only ~600k steps. SAC+DrQ typically continues improving up to 2-3M steps on CarRacing. The simplest and most reliable path to 900+ is **more training on your current config**.

## Recommended approach

### Step 1: Resume the first run to 2M steps (primary)

Use `--resume` to continue from where you left off:

```bash
python train.py --resume checkpoints/run01_default_s1/last_sac_drq.pth \
  --total-steps 2000000 --seed 1 \
  --checkpoint-dir checkpoints/run01_default_s1 --run-name run01_default_s1
```

This will:

- Load weights + optimizer state from step ~607k
- Continue training up to 2M total steps
- Overwrite `best_sac_drq.pth` when eval improves

**Expected outcome**: 850-920+ by 2M steps based on typical SAC+DrQ curves.

**Time estimate**: ~8-12 hours on your 3050 for the remaining ~1.4M steps.

### Step 2: Evaluate at 2M

```bash
python inference.py --checkpoint checkpoints/run01_default_s1/best_sac_drq.pth \
  --algo sac_drq --episodes 5 --no-render
```

If you hit 900+, you're done. If you're at 870-890, continue to Step 3.

### Step 3 (if needed): Extend to 3M steps

```bash
python train.py --resume checkpoints/run01_default_s1/last_sac_drq.pth \
  --total-steps 3000000 --seed 1 \
  --checkpoint-dir checkpoints/run01_default_s1 --run-name run01_default_s1
```

**Time estimate**: Another ~8-10 hours for the extra 1M steps.

### Step 4 (optional parallel experiment): Reward scaling run

If you want a second data point while the main run continues (or afterward), try reward scaling which often helps at high performance levels:

```bash
python train.py --total-steps 2000000 --seed 1 --reward-scale 0.1 \
  --checkpoint-dir checkpoints/run02_rs0p1_s1 --run-name run02_rs0p1_s1
```

This starts fresh but with scaled rewards (can stabilize Q-value estimates at high scores).

## Memory safety reminder

Your current setup (6GB RAM, 8GB swap, replay_size=30k) should be fine. Just avoid heavy browser/app usage during training.

To monitor:

```bash
watch -n 5 free -h
```

## Recording the final video

Once you have a checkpoint scoring 900+:

```bash
python inference.py --checkpoint checkpoints/run01_default_s1/best_sac_drq.pth \
  --algo sac_drq --episodes 1 --save-video --video-dir ./videos
```

## Summary

| Priority | Action | Expected score | Time |

|----------|--------|----------------|------|

| 1 | Resume run01 to 2M steps | 850-920 | 8-12h |

| 2 | Extend to 3M if needed | 900-950+ | +8-10h |

| 3 | (Optional) Fresh run with reward-scale 0.1 | Comparison | 12-16h |

The single most impactful thing is simply **more training steps on your existing run**.