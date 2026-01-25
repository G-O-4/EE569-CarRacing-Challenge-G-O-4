import argparse
import os
import random
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch

from carracing_env import make_env
from sac_drq import SACAgent, SACConfig, ReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(agent: SACAgent, cfg: SACConfig, device: torch.device, eval_seed_base: Optional[int]) -> Tuple[float, float]:
    env = make_env(
        env_id=cfg.env_id,
        render_mode=None,
        action_mode="continuous2d",
        grayscale=cfg.grayscale,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
    )
    rewards = []
    for i in range(cfg.num_eval_episodes):
        if eval_seed_base is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=int(eval_seed_base) + i)
        done = False
        ep_r = 0.0
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_r += float(reward)
        rewards.append(ep_r)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def save_checkpoint(
    path: str,
    agent: SACAgent,
    cfg: SACConfig,
    step: int,
    episode: int,
    best_eval_reward: float,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "algo": "sac_drq",
        "step": int(step),
        "episode": int(episode),
        "best_eval_reward": float(best_eval_reward),
        "cfg": asdict(cfg),
        "agent": agent.state_dict(),
        "optim": {
            "actor": agent.actor_opt.state_dict(),
            "critic": agent.critic_opt.state_dict(),
            "alpha": agent.alpha_opt.state_dict(),
        },
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="Train SAC+DrQ on CarRacing-v3 (pixels)")
    parser.add_argument("--total-steps", type=int, default=SACConfig.total_steps)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable AMP mixed precision on CUDA (default: off, or use checkpoint value when resuming)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume training from a SAC+DrQ checkpoint path")

    parser.add_argument("--grayscale", action="store_true", help="Use grayscale observations (default)")
    parser.add_argument("--rgb", action="store_true", help="Use RGB observations")
    parser.add_argument("--frame-stack", type=int, default=SACConfig.frame_stack)
    parser.add_argument("--action-repeat", type=int, default=SACConfig.action_repeat)

    parser.add_argument("--eval-every-steps", type=int, default=SACConfig.eval_every_steps)
    parser.add_argument("--num-eval-episodes", type=int, default=SACConfig.num_eval_episodes)
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=10000,
        help="Base seed for eval episodes (set to -1 for random eval). Seeds used: base, base+1, ...",
    )

    parser.add_argument("--batch-size", type=int, default=SACConfig.batch_size)
    parser.add_argument("--start-steps", type=int, default=SACConfig.start_steps)
    parser.add_argument("--update-after", type=int, default=SACConfig.update_after)
    parser.add_argument("--updates-per-step", type=int, default=SACConfig.updates_per_step)
    parser.add_argument("--replay-size", type=int, default=SACConfig.replay_size)
    parser.add_argument("--reward-scale", type=float, default=SACConfig.reward_scale)
    parser.add_argument("--drq-pad", type=int, default=SACConfig.drq_pad)
    parser.add_argument("--feature-dim", type=int, default=SACConfig.feature_dim)
    parser.add_argument("--hidden-dim", type=int, default=SACConfig.hidden_dim)

    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--run-name", type=str, default="car_racing_sac_drq")
    parser.add_argument("--no-aim", action="store_true", help="Disable Aim logging")

    args = parser.parse_args()

    grayscale = True
    if args.rgb:
        grayscale = False
    if args.grayscale:
        grayscale = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    print(f"Using device: {device}")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    use_amp = False if args.use_amp is None else bool(args.use_amp)
    if use_amp and device.type != "cuda":
        print("Warning: AMP requested but CUDA is unavailable. Disabling AMP.")
        use_amp = False

    cfg = SACConfig(
        grayscale=grayscale,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        total_steps=args.total_steps,
        eval_every_steps=args.eval_every_steps,
        num_eval_episodes=args.num_eval_episodes,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        update_after=args.update_after,
        updates_per_step=args.updates_per_step,
        replay_size=args.replay_size,
        reward_scale=args.reward_scale,
        drq_pad=args.drq_pad,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        use_amp=use_amp,
    )

    aim_run = None
    if not args.no_aim:
        try:
            from aim import Run

            aim_run = Run(experiment=args.run_name)
            aim_run["hparams"] = {
                **asdict(cfg),
                "seed": args.seed,
                "device": str(device),
            }
            print("Aim logging enabled. Run 'aim up' to view dashboard.")
        except Exception as e:
            print(f"Warning: Aim logging disabled (import/init failed): {e}")
            aim_run = None

    env = make_env(
        env_id=cfg.env_id,
        render_mode=None,
        action_mode="continuous2d",
        grayscale=cfg.grayscale,
        frame_stack=cfg.frame_stack,
        action_repeat=cfg.action_repeat,
    )

    obs_shape = env.observation_space.shape  # (C,84,84)
    action_dim = env.action_space.shape[0]  # 2

    agent = SACAgent(obs_shape=obs_shape, action_dim=action_dim, cfg=cfg, device=device)
    replay = ReplayBuffer(obs_shape=obs_shape, action_dim=action_dim, capacity=cfg.replay_size)

    best_eval = -float("inf")
    step = 0
    episode = 0

    # Episode seeding: deterministic but diverse tracks across episodes.
    episode_seed = int(args.seed)
    obs, _ = env.reset(seed=episode_seed)
    ep_reward = 0.0
    ep_len = 0

    # Resume (optional): restores weights + optimizer states + step counters.
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload.get("algo") != "sac_drq":
            raise ValueError(f"Can only resume SAC+DrQ checkpoints. Found algo={payload.get('algo')!r}")

        # Structural settings are taken from the checkpoint config to avoid shape mismatch.
        ckpt_cfg = payload.get("cfg", {})
        cfg = SACConfig(**{k: v for k, v in ckpt_cfg.items() if k in SACConfig().__dict__})

        # Allow extending the run horizon / eval cadence.
        cfg.total_steps = int(args.total_steps)
        cfg.eval_every_steps = int(args.eval_every_steps)
        cfg.num_eval_episodes = int(args.num_eval_episodes)
        if args.use_amp is not None:
            cfg.use_amp = bool(args.use_amp)
        if cfg.use_amp and device.type != "cuda":
            print("Warning: AMP enabled in config but CUDA is unavailable. Disabling AMP.")
            cfg.use_amp = False

        # Recreate env/agent with checkpoint cfg
        env.close()
        env = make_env(
            env_id=cfg.env_id,
            render_mode=None,
            action_mode="continuous2d",
            grayscale=cfg.grayscale,
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
        )
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]

        agent = SACAgent(obs_shape=obs_shape, action_dim=action_dim, cfg=cfg, device=device)
        agent.load_state_dict(payload["agent"])

        optim = payload.get("optim", {})
        if "actor" in optim:
            agent.actor_opt.load_state_dict(optim["actor"])
        if "critic" in optim:
            agent.critic_opt.load_state_dict(optim["critic"])
        if "alpha" in optim:
            agent.alpha_opt.load_state_dict(optim["alpha"])

        # Replay is not resumed (would be extremely large); we refill it from fresh interaction.
        replay = ReplayBuffer(obs_shape=obs_shape, action_dim=action_dim, capacity=cfg.replay_size)

        step = int(payload.get("step", 0))
        episode = int(payload.get("episode", 0))
        best_eval = float(payload.get("best_eval_reward", best_eval))

        episode_seed = int(args.seed) + episode
        obs, _ = env.reset(seed=episode_seed)
        ep_reward = 0.0
        ep_len = 0

        print(f"Resumed from {args.resume} at step={step}, episode={episode}, best_eval={best_eval:.1f}")

    print("Starting SAC+DrQ training...")
    while step < cfg.total_steps:
        # Action selection
        if step < cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, eval_mode=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay.add(obs, action, float(reward), next_obs, done)

        obs = next_obs
        ep_reward += float(reward)
        ep_len += 1
        step += 1

        # Updates
        if len(replay) >= max(cfg.update_after, cfg.batch_size) and step % cfg.update_every == 0:
            for _ in range(cfg.updates_per_step):
                metrics = agent.update(replay)
                if aim_run is not None:
                    aim_run.track(metrics, step=step)

        # End episode
        if done:
            episode += 1
            if aim_run is not None:
                aim_run.track(
                    {
                        "episode_reward": ep_reward,
                        "episode_len": ep_len,
                    },
                    step=step,
                )
            if episode % 10 == 0:
                print(f"Episode {episode} | step {step}/{cfg.total_steps} | reward {ep_reward:.1f} | len {ep_len}")

            episode_seed += 1
            obs, _ = env.reset(seed=episode_seed)
            ep_reward = 0.0
            ep_len = 0

        # Periodic evaluation
        if step % cfg.eval_every_steps == 0:
            eval_seed_base = None if args.eval_seed_base is not None and int(args.eval_seed_base) < 0 else int(args.eval_seed_base)
            eval_mean, eval_std = evaluate(agent, cfg, device, eval_seed_base=eval_seed_base)
            print(f"[EVAL] step {step} | mean {eval_mean:.1f} ± {eval_std:.1f} | best {best_eval:.1f}")
            if aim_run is not None:
                aim_run.track({"eval_reward_mean": eval_mean, "eval_reward_std": eval_std}, step=step)

            if eval_mean > best_eval:
                best_eval = eval_mean
                best_path = os.path.join(args.checkpoint_dir, "best_sac_drq.pth")
                save_checkpoint(best_path, agent, cfg, step=step, episode=episode, best_eval_reward=best_eval)
                print(f"Saved new best checkpoint: {best_path}")

            # Always save a last checkpoint on eval intervals
            last_path = os.path.join(args.checkpoint_dir, "last_sac_drq.pth")
            save_checkpoint(last_path, agent, cfg, step=step, episode=episode, best_eval_reward=best_eval)

    env.close()

    final_path = os.path.join(args.checkpoint_dir, "final_sac_drq.pth")
    save_checkpoint(final_path, agent, cfg, step=step, episode=episode, best_eval_reward=best_eval)
    print(f"Training complete. Best eval mean reward: {best_eval:.1f}")
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()


