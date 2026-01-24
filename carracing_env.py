import gymnasium as gym
import numpy as np
import cv2
from collections import deque


class CarRacingDiscreteActionWrapper(gym.ActionWrapper):
    """
    Maps discrete actions to the continuous action space of CarRacing-v3.

    Discrete Actions:
      0: Do nothing
      1: Turn Left
      2: Turn Right
      3: Gas
      4: Brake
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(5)

    def action(self, action):
        if action == 0:  # Do nothing
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if action == 1:  # Turn Left
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        if action == 2:  # Turn Right
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if action == 3:  # Gas
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if action == 4:  # Brake
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


class CarRacingContinuous2DActionWrapper(gym.ActionWrapper):
    """
    Exposes a simplified 2D continuous action space:
      - steer ∈ [-1, 1]
      - accel ∈ [-1, 1]   (positive => gas, negative => brake)

    Mapped to env's native 3D action:
      [steer, gas=max(accel,0), brake=max(-accel,0)]
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], -1.0, 1.0))
        gas = float(np.clip(max(accel, 0.0), 0.0, 1.0))
        brake = float(np.clip(max(-accel, 0.0), 0.0, 1.0))
        return np.array([steer, gas, brake], dtype=np.float32)


class CarRacingPixelObsWrapper(gym.ObservationWrapper):
    """
    Pixel preprocessing:
      - crop out the bottom HUD/status bar
      - center-crop to 84x84 (no aspect distortion)
      - grayscale (optional) OR RGB
      - return channel-first uint8: (C, 84, 84) where C is 1 or 3
    """

    def __init__(self, env, grayscale: bool = True):
        super().__init__(env)
        self.grayscale = grayscale
        c = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c, 84, 84),
            dtype=np.uint8,
        )

    def observation(self, frame):
        # Original CarRacing frame is (96, 96, 3) RGB.
        # Crop top 84 pixels (remove bottom status bar) and center-crop width to 84.
        frame = frame[:84, 6:90]  # (84, 84, 3)

        if self.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # (84, 84)
            return gray[None, :, :].astype(np.uint8)  # (1, 84, 84)

        rgb = np.transpose(frame, (2, 0, 1))  # (3, 84, 84)
        return rgb.astype(np.uint8)


class ActionRepeat(gym.Wrapper):
    """Repeat the same action for N env steps, accumulating reward."""

    def __init__(self, env, repeat: int = 1):
        super().__init__(env)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class StackFrames(gym.Wrapper):
    """
    Concatenate the last k preprocessed frames along the channel dimension.

    Input:  (C, 84, 84)
    Output: (k*C, 84, 84)
    """

    def __init__(self, env, stack_size: int = 4):
        super().__init__(env)
        self.stack_size = int(stack_size)
        self.frames = deque(maxlen=self.stack_size)

        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.stack_size * c, h, w),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)


def make_env(
    env_id: str = "CarRacing-v3",
    render_mode=None,
    action_mode: str = "discrete",
    grayscale: bool = True,
    frame_stack: int = 4,
    action_repeat: int = 1,
):
    """
    Create and wrap CarRacing environment.

    action_mode:
      - "discrete": 5 discrete actions (baseline DQN-friendly)
      - "continuous2d": 2D continuous actions for SAC/TD3 style agents
    """
    env = gym.make(env_id, continuous=True, render_mode=render_mode)

    if action_mode == "discrete":
        env = CarRacingDiscreteActionWrapper(env)
    elif action_mode == "continuous2d":
        env = CarRacingContinuous2DActionWrapper(env)
    else:
        raise ValueError(f"Unknown action_mode={action_mode!r}")

    env = ActionRepeat(env, repeat=action_repeat)
    env = CarRacingPixelObsWrapper(env, grayscale=grayscale)
    env = StackFrames(env, stack_size=frame_stack)
    return env


# ---------------------------------------------------------------------------
#                      Stable-Baselines3 PPO Helpers
# ---------------------------------------------------------------------------

class CarRacingPixelObsWrapperSB3(gym.ObservationWrapper):
    """
    Pixel preprocessing for SB3 (channel-last format):
      - crop out the bottom HUD/status bar
      - center-crop to 84x84
      - grayscale (optional) OR RGB
      - return channel-last uint8: (84, 84, C) where C is 1 or 3
    
    SB3's CnnPolicy expects channel-last by default.
    """

    def __init__(self, env, grayscale: bool = True):
        super().__init__(env)
        self.grayscale = grayscale
        c = 1 if grayscale else 3
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, c),
            dtype=np.uint8,
        )

    def observation(self, frame):
        # Original CarRacing frame is (96, 96, 3) RGB.
        # Crop top 84 pixels (remove bottom status bar) and center-crop width to 84.
        frame = frame[:84, 6:90]  # (84, 84, 3)

        if self.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # (84, 84)
            return gray[:, :, None].astype(np.uint8)  # (84, 84, 1)

        return frame.astype(np.uint8)  # (84, 84, 3)


def make_env_sb3(
    env_id: str = "CarRacing-v3",
    render_mode=None,
    grayscale: bool = True,
    action_repeat: int = 1,
    seed: int = None,
):
    """
    Create a single CarRacing env for SB3 (no frame stacking here; use VecFrameStack).
    
    Returns a function that creates the env (for use with DummyVecEnv/SubprocVecEnv).
    """
    def _init():
        env = gym.make(env_id, continuous=True, render_mode=render_mode)
        env = CarRacingContinuous2DActionWrapper(env)
        env = ActionRepeat(env, repeat=action_repeat)
        env = CarRacingPixelObsWrapperSB3(env, grayscale=grayscale)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def make_vec_env_ppo(
    env_id: str = "CarRacing-v3",
    n_envs: int = 1,
    grayscale: bool = True,
    frame_stack: int = 4,
    action_repeat: int = 1,
    seed: int = None,
    use_subproc: bool = False,
):
    """
    Create a vectorized environment for SB3 PPO training.
    
    Args:
        env_id: Gymnasium environment ID
        n_envs: Number of parallel environments
        grayscale: Use grayscale observations
        frame_stack: Number of frames to stack
        action_repeat: Repeat each action N times
        seed: Random seed (each env gets seed + i)
        use_subproc: Use SubprocVecEnv instead of DummyVecEnv (faster but more memory)
    
    Returns:
        Vectorized environment with frame stacking applied
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
    
    env_fns = [
        make_env_sb3(
            env_id=env_id,
            render_mode=None,
            grayscale=grayscale,
            action_repeat=action_repeat,
            seed=seed + i if seed is not None else None,
        )
        for i in range(n_envs)
    ]
    
    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    
    # Apply frame stacking
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")
    
    return vec_env


def make_eval_env_ppo(
    env_id: str = "CarRacing-v3",
    render_mode=None,
    grayscale: bool = True,
    frame_stack: int = 4,
    action_repeat: int = 1,
    seed: int = None,
):
    """
    Create a single vectorized env for PPO evaluation (with optional rendering).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    
    env_fn = make_env_sb3(
        env_id=env_id,
        render_mode=render_mode,
        grayscale=grayscale,
        action_repeat=action_repeat,
        seed=seed,
    )
    
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="last")
    
    return vec_env


