# inference.py
import argparse
import os
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

from carracing_env import make_env, make_eval_env_ppo
from dqn_car_racing import DQN
from sac_drq import SACAgent, SACConfig


def detect_algo(checkpoint_path: str, checkpoint_data=None) -> str:
    """
    Detect algorithm from checkpoint path or data.
    - .zip files are PPO (stable-baselines3 format)
    - .pth files are DQN or SAC+DrQ (PyTorch format)
    """
    # PPO checkpoints are .zip files
    if checkpoint_path.endswith(".zip"):
        return "ppo"
    
    # For .pth files, check the checkpoint content
    if checkpoint_data is not None:
        if isinstance(checkpoint_data, dict) and checkpoint_data.get("algo") == "sac_drq":
            return "sac_drq"
        if isinstance(checkpoint_data, dict) and "agent" in checkpoint_data and "actor" in checkpoint_data.get("agent", {}):
            return "sac_drq"
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            return "dqn"
    
    return "unknown"


def load_dqn(checkpoint_path, input_shape, n_actions, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DQN(input_shape, n_actions).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded DQN checkpoint from episode {checkpoint.get('episode', 'N/A')}")
    if "reward" in checkpoint:
        print(f"   Best reward: {checkpoint['reward']:.2f}")
    return model


def load_sac_drq(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algo = detect_algo(checkpoint_path, checkpoint)
    if algo != "sac_drq":
        raise ValueError(f"Checkpoint does not look like SAC+DrQ: {checkpoint_path}")

    cfg_dict = checkpoint.get("cfg") or checkpoint.get("agent", {}).get("cfg") or {}
    cfg = SACConfig(**{k: v for k, v in cfg_dict.items() if k in SACConfig().__dict__})

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
    env.close()

    agent = SACAgent(obs_shape=obs_shape, action_dim=action_dim, cfg=cfg, device=device)
    agent.load_state_dict(checkpoint["agent"])
    agent.actor.eval()
    agent.critic.eval()

    step = checkpoint.get("step", "N/A")
    best = checkpoint.get("best_eval_reward", None)
    print(f"✅ Loaded SAC+DrQ checkpoint at step {step}")
    if best is not None:
        print(f"   Best eval mean reward: {best:.2f}")

    return agent, cfg

def load_ppo(checkpoint_path, device, env=None):
    """Load a PPO model from stable-baselines3 checkpoint (.zip)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    from stable_baselines3 import PPO
    
    # SB3 handles device automatically, but we can specify
    device_str = "cuda" if device.type == "cuda" else "cpu"
    model = PPO.load(checkpoint_path, env=env, device=device_str)
    
    print(f"✅ Loaded PPO checkpoint: {checkpoint_path}")
    return model


def run_episode(env, policy, device, render=True, seed=None):
    """Run a single episode and return total reward (works for DQN or SAC)."""
    if seed is None:
        state, _ = env.reset()
    else:
        state, _ = env.reset(seed=int(seed))
    total_reward = 0
    done = False
    
    with torch.no_grad():
        while not done:
            if isinstance(policy, DQN):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                action = policy(state_tensor).max(1)[1].item()
            else:
                # SAC agent
                action = policy.act(state, eval_mode=True)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
            
            if render and hasattr(env, 'render'):
                env.render()
    
    return total_reward


def run_episode_ppo(env, model, deterministic=True, seed=None):
    """
    Run a single episode with PPO model (uses VecEnv).
    Returns total reward.
    
    Note: VecEnv doesn't support per-episode seeding in reset().
    The seed parameter is accepted for API consistency but ignored.
    Seeding happens at environment creation time for VecEnv.
    """
    # VecEnv wrappers don't support seed in reset(), just reset normally
    obs = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = env.step(action)
        total_reward += float(reward[0])
        done = done_arr[0]
        
        # Check for episode end in info (VecEnv wraps terminal obs)
        if done and "episode" in info[0]:
            # Use the actual episode reward from info
            total_reward = info[0]["episode"]["r"]
    
    return total_reward

def main():
    parser = argparse.ArgumentParser(
        description='Load and run trained DQN on CarRacing-v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --checkpoint checkpoints/best_model.pth --episodes 3
  python inference.py --checkpoint checkpoints/final_model.pth --no-render --episodes 10
  python inference.py --save-video --video-dir ./test_videos
        """
    )
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint (default: checkpoints/best_model.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering (faster)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video recordings')
    parser.add_argument('--video-dir', type=str, default='./videos',
                       help='Directory to save videos (default: ./videos)')
    parser.add_argument('--seed', type=int, default=1,
                       help='Base seed for evaluation (set to -1 for random)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    parser.add_argument('--algo', type=str, default='auto', choices=['auto', 'dqn', 'sac_drq', 'ppo'],
                       help='Which algorithm checkpoint to load (default: auto-detect)')
    parser.add_argument('--grayscale', action='store_true', default=True,
                       help='Use grayscale for DQN/PPO (default)')
    parser.add_argument('--rgb', action='store_true',
                       help='Use RGB for DQN/PPO')
    parser.add_argument('--frame-stack', type=int, default=4,
                       help='Frame stack size for DQN/PPO (default: 4)')
    parser.add_argument('--action-repeat', type=int, default=1,
                       help='Action repeat for DQN/PPO (default: 1)')
    parser.add_argument('--vecnorm', type=str, default=None,
                       help='Path to VecNormalize stats for PPO (default: auto-detect)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"❌ Checkpoint not found: {args.checkpoint}")
    
    # Create environment with appropriate render mode
    render_mode = 'rgb_array' if args.save_video else ('human' if not args.no_render else None)
    env = None
    
    # Detect algorithm and load checkpoint
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    
    # Determine grayscale setting for DQN/PPO
    grayscale = True
    if args.rgb:
        grayscale = False
    seed = None if args.seed is not None and int(args.seed) < 0 else int(args.seed)
    
    # Detect algorithm from file extension first, then content
    if args.algo != "auto":
        algo = args.algo
    elif args.checkpoint.endswith(".zip"):
        algo = "ppo"
    else:
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        algo = detect_algo(args.checkpoint, raw)

    if algo == "dqn":
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        env = make_env(
            render_mode=render_mode,
            action_mode="discrete",
            grayscale=grayscale,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
        )
        if args.save_video:
            os.makedirs(args.video_dir, exist_ok=True)
            env = RecordVideo(env, args.video_dir, name_prefix="inference_dqn")
            print(f"🎬 Video recording enabled, saving to: {args.video_dir}")
        n_actions = env.action_space.n
        input_shape = env.observation_space.shape
        policy = load_dqn(args.checkpoint, input_shape, n_actions, device)
    elif algo == "sac_drq":
        agent, cfg = load_sac_drq(args.checkpoint, device)
        env = make_env(
            env_id=cfg.env_id,
            render_mode=render_mode,
            action_mode="continuous2d",
            grayscale=cfg.grayscale,
            frame_stack=cfg.frame_stack,
            action_repeat=cfg.action_repeat,
        )
        if args.save_video:
            os.makedirs(args.video_dir, exist_ok=True)
            env = RecordVideo(env, args.video_dir, name_prefix="inference_sac_drq")
            print(f"🎬 Video recording enabled, saving to: {args.video_dir}")
        policy = agent
    elif algo == "ppo":
        # PPO uses VecEnv with VecFrameStack
        from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
        
        env = make_eval_env_ppo(
            env_id="CarRacing-v3",
            render_mode=render_mode,
            grayscale=grayscale,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            seed=seed,
        )

        vecnorm_path = args.vecnorm
        if vecnorm_path is None:
            candidate = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
            if os.path.exists(candidate):
                vecnorm_path = candidate

        if vecnorm_path is not None and os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ Loaded VecNormalize stats: {vecnorm_path}")
        
        if args.save_video:
            os.makedirs(args.video_dir, exist_ok=True)
            env = VecVideoRecorder(
                env,
                args.video_dir,
                record_video_trigger=lambda x: True,  # Record all episodes
                video_length=2000,
                name_prefix="inference_ppo",
            )
            print(f"🎬 Video recording enabled, saving to: {args.video_dir}")

        policy = load_ppo(args.checkpoint, device, env=env)
    else:
        raise ValueError(f"Could not determine algorithm (detected={algo!r}). Pass --algo dqn, sac_drq, or ppo.")
    
    # Run episodes
    print(f"\n🏁 Running {args.episodes} episode(s)...\n")
    rewards = []
    
    try:
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}...", end=" ")
            episode_seed = None if seed is None else seed + episode
            
            if algo == "ppo":
                # PPO uses VecEnv
                episode_reward = run_episode_ppo(env, policy, deterministic=True, seed=episode_seed)
            else:
                # DQN and SAC use regular env
                episode_reward = run_episode(env, policy, device, render=not args.no_render, seed=episode_seed)
            
            rewards.append(episode_reward)
            print(f"Reward: {episode_reward:.2f}")
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user")
    finally:
        if env is not None:
            env.close()
    
    # Print summary
    if rewards:
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Episodes completed: {len(rewards)}")
        print(f"Mean Reward:        {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Max Reward:         {np.max(rewards):.2f}")
        print(f"Min Reward:         {np.min(rewards):.2f}")
        print("="*60)
    
    if args.save_video:
        print(f"\n🎥 Videos saved in: {args.video_dir}")

if __name__ == '__main__':
    main()