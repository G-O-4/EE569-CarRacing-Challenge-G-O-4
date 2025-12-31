import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import random
import math
import collections
from collections import deque
import os
from aim import Run, Image
from carracing_env import (
    CarRacingDiscreteActionWrapper as CarRacingActionWrapper,
    CarRacingPixelObsWrapper as CarRacingImageWrapper,
    StackFrames,
    make_env,
)

# ---------------------------------------------------------------------------- #
#                                Constants                                     #
# ---------------------------------------------------------------------------- #
NUM_EPISODES = 800  # 600-1000 is the sweet spot
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0     # Start at 1.0 for full exploration
EPS_END = 0.05
EPS_DECAY = 10000   # Slower decay (higher value = slower)
TARGET_UPDATE = 10  # Episodes
MEMORY_SIZE = 100000  # Larger replay buffer
LEARNING_RATE = 1e-4
STACK_SIZE = 4
MAX_STEPS_PER_EPISODE = 1000  # Shorter episodes prevent endless loops

# New constants
EVAL_FREQUENCY = 50  # Evaluate every 50 episodes
NUM_EVAL_EPISODES = 3  # Episodes per evaluation
SAVE_BEST_VIDEO = True  # Set to True to save video of best episode
VIDEO_DIR = "./videos"
CHECKPOINT_DIR = "./checkpoints"
USE_AIM = True  # Set to True to enable Aim logging

# Create directories
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------------------------- #
#                                Wrappers                                      #
# ---------------------------------------------------------------------------- #
# Wrappers and `make_env` are imported from `carracing_env.py` so they can be
# shared across algorithms (DQN baseline, SAC+DrQ, etc.).

# ---------------------------------------------------------------------------- #
#                                DQN Model                                     #
# ---------------------------------------------------------------------------- #

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size of flattened feature map
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---------------------------------------------------------------------------- #
#                            Replay Buffer                                     #
# ---------------------------------------------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )

    def __len__(self):
        return len(self.buffer)

# ---------------------------------------------------------------------------- #
#                         Evaluation & Video Recording                         #
# ---------------------------------------------------------------------------- #

def evaluate_policy(policy_net, env_name, num_episodes, video_dir, episode_idx, device):
    """Evaluate policy and optionally record video"""
    # Create fresh environment for evaluation
    eval_env = make_env(
        env_id=env_name,
        render_mode="rgb_array",
        action_mode="discrete",
        grayscale=True,
        frame_stack=STACK_SIZE,
    )
    
    # Setup video recording if enabled
    video_path = None
    if video_dir:
        from gymnasium.wrappers import RecordVideo
        eval_env = RecordVideo(
            eval_env,
            video_dir,
            episode_trigger=lambda x: x == 0,  # Record only first eval episode
            name_prefix=f"eval_episode_{episode_idx}"
        )
    
    policy_net.eval()
    total_rewards = []
    
    with torch.no_grad():
        for i in range(num_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            for t in range(1000):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                action = policy_net(state_tensor).max(1)[1].item()
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(episode_reward)
    
    policy_net.train()
    eval_env.close()
    
    # Find the video file path
    if video_dir:
        video_files = [f for f in os.listdir(video_dir) 
                      if f.startswith(f"eval_episode_{episode_idx}") and f.endswith(".mp4")]
        if video_files:
            video_path = os.path.join(video_dir, video_files[0])
    
    return np.mean(total_rewards), video_path

# ---------------------------------------------------------------------------- #
#                                Training                                      #
# ---------------------------------------------------------------------------- #

def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None
    
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    
    # Convert to tensors
    states = torch.FloatTensor(states).to(device) / 255.0
    next_states = torch.FloatTensor(next_states).to(device) / 255.0
    actions = torch.LongTensor(actions).to(device).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
    dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
    
    # Current Q values
    current_q_values = policy_net(states).gather(1, actions)
    
    # Target Q values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * GAMMA * next_q_values
    
    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_values, expected_q_values)
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.item()

def main():
    # Initialize Aim run
    aim_run = None
    if USE_AIM:
        aim_run = Run(experiment="car_racing_dqn")
        aim_run["hparams"] = {
            "NUM_EPISODES": NUM_EPISODES,
            "BATCH_SIZE": BATCH_SIZE,
            "GAMMA": GAMMA,
            "EPS_START": EPS_START,
            "EPS_END": EPS_END,
            "EPS_DECAY": EPS_DECAY,
            "TARGET_UPDATE": TARGET_UPDATE,
            "MEMORY_SIZE": MEMORY_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "ENV_NAME": "CarRacing-v3",
        }
        print("Aim logging enabled. Run 'aim up' to view results.")
    
    env = make_env()
    
    # Initialize Networks
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape
    
    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    steps_done = 0
    best_reward = -float('inf')
    
    print("Starting Training Loop...")
    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        total_reward = 0
        
        for t in range(1000):
            # Epsilon-greedy action selection
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if random.random() > eps_threshold:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
                    action = policy_net(state_tensor).max(1)[1].item()
            else:
                action = env.action_space.sample()
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Store transition
            memory.push(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Optimize model
            loss = optimize_model(policy_net, target_net, optimizer, memory)
            if USE_AIM and loss is not None:
                aim_run.track({"loss": loss}, step=steps_done)
            
            if done:
                break
        
        # Log episode metrics
        if USE_AIM:
            aim_run.track({
                "episode_reward": total_reward,
                "epsilon": eps_threshold
            }, step=i_episode)
        
        # Update target network
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {i_episode+1}/{NUM_EPISODES} - Reward: {total_reward:.2f} - Epsilon: {eps_threshold:.3f}")
        
        # Periodic evaluation
        if (i_episode + 1) % EVAL_FREQUENCY == 0:
            print(f"\n--- Running Evaluation (Episode {i_episode+1}) ---")
            eval_reward, video_path = evaluate_policy(
                policy_net, "CarRacing-v3", NUM_EVAL_EPISODES,
                VIDEO_DIR if SAVE_BEST_VIDEO else None,
                i_episode + 1, device
            )
            print(f"Eval Reward: {eval_reward:.2f} (Best: {best_reward:.2f})\n")
            
            if USE_AIM:
                aim_run.track({"eval_reward": eval_reward}, step=i_episode+1)
            
            # Save best model and video
            if eval_reward > best_reward:
                best_reward = eval_reward
                print(f"🎉 New best reward! Saving checkpoint and video...")
                
                # Save best model
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
                torch.save({
                    'episode': i_episode + 1,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward': best_reward,
                }, checkpoint_path)
                
                # Log video to Aim
                if SAVE_BEST_VIDEO and video_path and os.path.exists(video_path):
                    try:
                        aim_run.track(Image(video_path), step=i_episode+1, name="best_episode_video")
                        print(f"Video uploaded to Aim: {video_path}")
                    except Exception as e:
                        print(f"Warning: Could not upload video to Aim: {e}")
        
        # Save last model periodically
        if (i_episode + 1) % 100 == 0:
            last_path = os.path.join(CHECKPOINT_DIR, "last_model.pth")
            torch.save({
                'episode': i_episode + 1,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, last_path)
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save({
        'episode': NUM_EPISODES,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    print(f"\nTraining Complete! Best Reward: {best_reward:.2f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Videos saved in: {VIDEO_DIR}")
    if USE_AIM:
        print(f"Aim logs saved. Run 'aim up' to view dashboard.")
    
    env.close()

if __name__ == "__main__":
    main()