"""
PPO training script for CarRacing-v3 using stable-baselines3.

Advantages over SAC+DrQ:
- No replay buffer = much lower RAM usage (~500MB vs ~2-3GB)
- Lower variance in scores
- Benefits from parallel environments

Usage:
    python train_ppo.py --total-timesteps 1000000 --seed 1 --checkpoint-dir checkpoints/ppo_run01
"""

import argparse
import os
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from carracing_env import make_vec_env_ppo, make_eval_env_ppo


class AimCallback(BaseCallback):
    """
    Callback for logging to Aim experiment tracker.
    """

    def __init__(self, run, verbose=0):
        super().__init__(verbose)
        self.run = run

    def _on_step(self) -> bool:
        # Log scalar metrics every 1000 steps
        if self.n_calls % 1000 == 0 and self.run is not None:
            # Get info from logger
            if len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                self.run.track(
                    {"episode_reward_mean": ep_rew_mean, "episode_len_mean": ep_len_mean},
                    step=self.num_timesteps,
                )
        return True


class BestModelCallback(BaseCallback):
    """
    Callback to save the best model based on evaluation rewards.
    Works alongside EvalCallback to track the best model.
    """

    def __init__(self, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        return True


def main():
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v3 using stable-baselines3")

    # Training params
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")

    # Environment params
    parser.add_argument("--grayscale", action="store_true", default=True, help="Use grayscale (default)")
    parser.add_argument("--rgb", action="store_true", help="Use RGB observations")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--action-repeat", type=int, default=1, help="Repeat each action N times")

    # PPO hyperparams
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ppo", help="Checkpoint directory")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency")
    parser.add_argument("--tb-log", type=str, default="./tb_logs/", help="Tensorboard log directory")
    parser.add_argument("--run-name", type=str, default="ppo_car_racing", help="Run name for logging")
    parser.add_argument("--no-aim", action="store_true", help="Disable Aim logging")

    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (.zip)")

    args = parser.parse_args()

    # Determine grayscale setting
    grayscale = True
    if args.rgb:
        grayscale = False

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print(f"PPO Training on CarRacing-v3")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Seed: {args.seed}")
    print(f"  Grayscale: {grayscale}")
    print(f"  Frame stack: {args.frame_stack}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    # Create training environment
    env = make_vec_env_ppo(
        env_id="CarRacing-v3",
        n_envs=args.n_envs,
        grayscale=grayscale,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed,
        use_subproc=False,  # DummyVecEnv is safer for most setups
    )

    # Create evaluation environment
    eval_env = make_eval_env_ppo(
        env_id="CarRacing-v3",
        render_mode=None,
        grayscale=grayscale,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=args.seed + 10000,  # Different seed for eval
    )

    # Initialize Aim logging
    aim_run = None
    if not args.no_aim:
        try:
            from aim import Run

            aim_run = Run(experiment=args.run_name)
            aim_run["hparams"] = {
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                "grayscale": grayscale,
                "frame_stack": args.frame_stack,
                "action_repeat": args.action_repeat,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
            }
            print("Aim logging enabled. Run 'aim up' to view dashboard.")
        except Exception as e:
            print(f"Warning: Aim logging disabled ({e})")
            aim_run = None

    # Create or load PPO model
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=args.tb_log)
        # Reset timesteps for continued training
        model._total_timesteps = 0
    else:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            seed=args.seed,
            tensorboard_log=args.tb_log,
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback - save periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    # Eval callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.checkpoint_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # Aim callback
    if aim_run is not None:
        aim_callback = AimCallback(aim_run)
        callbacks.append(aim_callback)

    # Train
    print("\nStarting PPO training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=args.run_name,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.zip")
    print(f"Final model: {final_path}.zip")
    print(f"Tensorboard logs: {args.tb_log}")
    print("\nTo evaluate:")
    print(f"  python inference.py --checkpoint {args.checkpoint_dir}/best_model.zip --algo ppo --episodes 5")
    print("=" * 60)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
