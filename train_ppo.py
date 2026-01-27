"""
PPO training script for CarRacing-v3 using stable-baselines3 with Beta distribution.

Based on research showing Beta-PPO achieves ~913 avg reward vs Gaussian's ~897.
Beta distribution is naturally bounded, ideal for continuous actions in [-1, 1].

Features:
- Custom Beta distribution policy for bounded continuous actions
- AimWriter for reliable metric logging (logs ALL SB3 training metrics)
- Optimized hyperparameters from research

Usage:
    python train_ppo.py --total-timesteps 1000000 --seed 42 --checkpoint-dir checkpoints/ppo_beta
"""

import argparse
import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.logger import KVWriter, Logger, make_output_format
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces

from carracing_env import make_vec_env_ppo, make_eval_env_ppo


# =============================================================================
# AimWriter - Reliable Aim Logging for SB3
# =============================================================================

class AimWriter(KVWriter):
    """
    Custom KVWriter that logs ALL SB3 metrics directly to Aim.
    This ensures reliable tracking of policy_loss, value_loss, entropy, etc.
    """
    
    def __init__(self, aim_run):
        self.aim_run = aim_run
    
    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        for key, value in key_values.items():
            if isinstance(value, (int, float)) and not key.startswith('time/'):
                # Clean up key names for Aim (replace / with _)
                clean_key = key.replace('/', '_')
                self.aim_run.track(value, name=clean_key, step=step)
    
    def close(self) -> None:
        pass


class StdoutWriter(KVWriter):
    """Simple stdout writer to keep console output."""
    
    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Tuple[str, ...]], step: int = 0) -> None:
        # PPO already prints to stdout via verbose=1, so we don't need to duplicate
        pass
    
    def close(self) -> None:
        pass


def setup_aim_logger(model, aim_run):
    """
    Configure SB3 to log directly to Aim.
    This replaces the default logger with one that forwards to Aim.
    """
    output_formats = [AimWriter(aim_run), StdoutWriter()]
    logger = Logger(folder=None, output_formats=output_formats)
    model.set_logger(logger)
    return logger


# =============================================================================
# Beta Distribution for Bounded Continuous Actions
# =============================================================================

class BetaDistributionWrapper(Distribution):
    """
    Beta distribution wrapper for SB3 compatibility.
    Beta distribution is naturally bounded to [0, 1], scaled to [-1, 1] for actions.
    
    Research shows Beta-PPO achieves ~913 avg reward vs Gaussian's ~897 on CarRacing.
    """
    
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None
    
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """Create the layer that outputs alpha and beta parameters."""
        return nn.Linear(latent_dim, 2 * self.action_dim)
    
    def proba_distribution(self, mean_actions: torch.Tensor) -> "BetaDistributionWrapper":
        """
        Create distribution from network output.
        mean_actions contains [alpha_params, beta_params] concatenated.
        """
        # Split into alpha and beta, apply softplus to ensure > 0, add 1 for stability
        alpha = nn.functional.softplus(mean_actions[..., :self.action_dim]) + 1.0
        beta = nn.functional.softplus(mean_actions[..., self.action_dim:]) + 1.0
        self.distribution = Beta(alpha, beta)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions."""
        # Scale actions from [-1, 1] to [0, 1] for Beta distribution
        scaled_actions = (actions + 1.0) / 2.0
        # Clamp to avoid numerical issues at boundaries
        scaled_actions = torch.clamp(scaled_actions, 1e-6, 1.0 - 1e-6)
        log_prob = self.distribution.log_prob(scaled_actions)
        # Adjust for the scaling: log|d(scaled)/d(action)| = log(0.5) per dimension
        log_prob = log_prob - np.log(2.0)
        return log_prob.sum(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """Compute entropy of the distribution."""
        return self.distribution.entropy().sum(dim=-1)
    
    def sample(self) -> torch.Tensor:
        """Sample actions from the distribution."""
        # Beta samples are in [0, 1], scale to [-1, 1]
        samples = self.distribution.rsample()
        return 2.0 * samples - 1.0
    
    def mode(self) -> torch.Tensor:
        """Return the mode of the distribution."""
        alpha = self.distribution.concentration1
        beta = self.distribution.concentration0
        # Mode of Beta(a,b) = (a-1)/(a+b-2) for a,b > 1
        mode = (alpha - 1.0) / (alpha + beta - 2.0 + 1e-8)
        mode = torch.clamp(mode, 0.0, 1.0)
        return 2.0 * mode - 1.0
    
    def actions_from_params(self, mean_actions: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(mean_actions)
        return self.mode() if deterministic else self.sample()
    
    def log_prob_from_params(self, mean_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_params(mean_actions)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BetaActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy using Beta distribution for continuous actions.
    This achieves better performance on bounded action spaces like CarRacing.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Dict[str, list]] = None,
        activation_fn: type = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Beta distribution doesn't use SDE
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=False,  # Force disable SDE for Beta
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=False,  # Beta handles bounds naturally
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
    
    def _build(self, lr_schedule: Schedule) -> None:
        """Build the network with Beta distribution."""
        self._build_mlp_extractor()
        
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        
        # Create Beta distribution
        action_dim = self.action_space.shape[0]
        self.action_dist = BetaDistributionWrapper(action_dim)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)
        
        # Value network
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # Initialize weights
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(lambda m: self.init_weights(m, gain))
        
        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: get actions, values, and log probabilities."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions)
        actions = distribution.mode() if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions for PPO update."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy
    
    def get_distribution(self, obs: torch.Tensor) -> BetaDistributionWrapper:
        """Get the action distribution for given observations."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi = self.mlp_extractor.forward_actor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features[0])
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for given observations."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_vf = self.mlp_extractor.forward_critic(features)
        else:
            latent_vf = self.mlp_extractor.forward_critic(features[1])
        return self.value_net(latent_vf)


# =============================================================================
# Fallback AimCallback (used alongside AimWriter for episode stats)
# =============================================================================

class AimEpisodeCallback(BaseCallback):
    """
    Callback for logging episode statistics to Aim.
    Works alongside AimWriter which handles training metrics.
    """

    def __init__(self, aim_run, verbose=0):
        super().__init__(verbose)
        self.aim_run = aim_run
        self.last_log_step = 0

    def _on_step(self) -> bool:
        # Log episode stats every 1000 steps
        if self.num_timesteps - self.last_log_step >= 1000 and self.aim_run is not None:
            self.last_log_step = self.num_timesteps
            
            if len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                self.aim_run.track(
                    {"episode_reward_mean": ep_rew_mean, "episode_len_mean": ep_len_mean},
                    step=self.num_timesteps,
                )
        return True


# =============================================================================
# Main Training Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Beta-PPO on CarRacing-v3")

    # Training params
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (set to -1 for random)")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (8 recommended)")

    # Environment params
    parser.add_argument("--grayscale", action="store_true", default=True, help="Use grayscale (default)")
    parser.add_argument("--rgb", action="store_true", help="Use RGB observations")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--action-repeat", type=int, default=1, help="Repeat each action N times")
    parser.add_argument(
        "--no-norm-reward",
        dest="norm_reward",
        action="store_false",
        help="Disable reward normalization (VecNormalize)",
    )
    parser.add_argument("--clip-reward", type=float, default=10.0, help="Reward clip for VecNormalize")

    # PPO hyperparams (optimized for CarRacing based on research)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size (larger for images)")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.1, help="PPO clip range (tighter for stability)")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    
    # Policy type
    parser.add_argument("--use-gaussian", action="store_true", help="Use Gaussian instead of Beta distribution")

    # Logging and checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ppo", help="Checkpoint directory")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency")
    parser.add_argument("--tb-log", type=str, default="./tb_logs/", help="Tensorboard log directory")
    parser.add_argument("--run-name", type=str, default="ppo_beta_car_racing", help="Run name for logging")
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

    policy_type = "Gaussian" if args.use_gaussian else "Beta"
    print(f"PPO ({policy_type} distribution) Training on CarRacing-v3")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Seed: {args.seed}")
    print(f"  Grayscale: {grayscale}")
    print(f"  Frame stack: {args.frame_stack}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Clip range: {args.clip_range}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Reward normalization: {args.norm_reward}")

    seed = None if args.seed is None or int(args.seed) < 0 else int(args.seed)

    # Create training environment
    env = make_vec_env_ppo(
        env_id="CarRacing-v3",
        n_envs=args.n_envs,
        grayscale=grayscale,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=seed,
        use_subproc=False,
    )

    # Reward normalization (optional)
    if args.norm_reward:
        vecnorm_path = None
        if args.resume is not None:
            candidate = os.path.join(os.path.dirname(args.resume), "vecnormalize.pkl")
            if os.path.exists(candidate):
                vecnorm_path = candidate

        if vecnorm_path is not None:
            print(f"Loading VecNormalize stats from: {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
        else:
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=args.clip_reward)

    # Create evaluation environment
    eval_seed = None if seed is None else seed + 10000
    eval_env = make_eval_env_ppo(
        env_id="CarRacing-v3",
        render_mode=None,
        grayscale=grayscale,
        frame_stack=args.frame_stack,
        action_repeat=args.action_repeat,
        seed=eval_seed,
    )
    if args.norm_reward:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
        eval_env.training = False
        eval_env.norm_reward = False

    # Initialize Aim logging
    aim_run = None
    if not args.no_aim:
        try:
            from aim import Run

            aim_run = Run(experiment=args.run_name)
            aim_run["hparams"] = {
                "algorithm": "PPO",
                "policy": policy_type,
                "total_timesteps": args.total_timesteps,
                "n_envs": args.n_envs,
                "seed": seed,
                "grayscale": grayscale,
                "frame_stack": args.frame_stack,
                "action_repeat": args.action_repeat,
                "norm_reward": args.norm_reward,
                "clip_reward": args.clip_reward,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
            }
            print("Aim logging enabled. Run 'aim up' to view dashboard.")
        except Exception as e:
            print(f"Warning: Aim logging disabled ({e})")
            aim_run = None

    # Select policy class
    policy_class = "CnnPolicy" if args.use_gaussian else BetaActorCriticPolicy

    # Create or load PPO model
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=args.tb_log)
        model._total_timesteps = 0
    else:
        model = PPO(
            policy_class,
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

    # Setup Aim logger (replaces default logger to capture ALL metrics)
    if aim_run is not None:
        setup_aim_logger(model, aim_run)

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=args.norm_reward,
    )
    callbacks.append(checkpoint_callback)

    # Eval callback
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

    # Episode stats callback (supplements AimWriter)
    if aim_run is not None:
        aim_callback = AimEpisodeCallback(aim_run)
        callbacks.append(aim_callback)

    # Train
    print(f"\nStarting PPO ({policy_type}) training...")
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

    if args.norm_reward and isinstance(env, VecNormalize):
        vecnorm_path = os.path.join(args.checkpoint_dir, "vecnormalize.pkl")
        env.save(vecnorm_path)
        print(f"VecNormalize stats saved to: {vecnorm_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Algorithm: PPO with {policy_type} distribution")
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
