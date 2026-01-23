import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class SACConfig:
    env_id: str = "CarRacing-v3"
    grayscale: bool = True
    frame_stack: int = 4
    action_repeat: int = 1

    # Training horizon / evaluation
    total_steps: int = 2_000_000
    eval_every_steps: int = 50_000
    num_eval_episodes: int = 3

    # Replay / updates
    # Memory usage: each transition stores 2 x (stack x 84 x 84) uint8 images.
    # With stack=4: ~56 KB per transition, so 30k ≈ 1.7 GB RAM (pre-allocated).
    # WSL2 users: if OOM, reduce further or increase WSL2 memory limit via .wslconfig
    replay_size: int = 30_000
    batch_size: int = 256
    start_steps: int = 10_000
    update_after: int = 10_000
    update_every: int = 1
    updates_per_step: int = 1

    # SAC hyperparams
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 1e-4
    init_alpha: float = 0.1
    target_entropy: Optional[float] = None  # if None => -action_dim

    # Model sizes
    feature_dim: int = 50
    hidden_dim: int = 1024

    # DrQ augmentation
    drq_pad: int = 4

    # Misc
    reward_scale: float = 1.0
    grad_clip_norm: float = 10.0


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """
    DrQ-style random shift augmentation.
    x: (B, C, H, W) float in [0, 1]
    """
    if pad <= 0:
        return x
    b, c, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")

    # Per-sample random crop back to (H, W).
    max_off = 2 * pad + 1
    top = torch.randint(0, max_off, (b,), device=x.device)
    left = torch.randint(0, max_off, (b,), device=x.device)

    row_idx = top[:, None] + torch.arange(h, device=x.device)[None, :]  # (B, H)
    col_idx = left[:, None] + torch.arange(w, device=x.device)[None, :]  # (B, W)

    b_idx = torch.arange(b, device=x.device)[:, None, None]  # (B, 1, 1)
    row_idx = row_idx[:, :, None]  # (B, H, 1)
    col_idx = col_idx[:, None, :]  # (B, 1, W)

    out = x[b_idx, :, row_idx, col_idx]  # (B, H, W, C)
    return out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)


class ReplayBuffer:
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)

        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self._dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        self._obs[self._idx] = obs
        self._next_obs[self._idx] = next_obs
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = float(done)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self._size, size=int(batch_size))

        obs = torch.as_tensor(self._obs[idxs], device=device).float().div_(255.0)
        next_obs = torch.as_tensor(self._next_obs[idxs], device=device).float().div_(255.0)
        actions = torch.as_tensor(self._actions[idxs], device=device)
        rewards = torch.as_tensor(self._rewards[idxs], device=device)
        dones = torch.as_tensor(self._dones[idxs], device=device)

        return obs, actions, rewards, next_obs, dones


class Encoder(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], feature_dim: int):
        super().__init__()
        c, h, w = obs_shape
        assert h == 84 and w == 84, "This encoder assumes 84x84 inputs."

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # 84 -> 20 -> 9 -> 7 => 7*7*64 = 3136
        self.fc = nn.Linear(3136, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        x = torch.tanh(x)
        return x


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = -10.0
        self.log_std_max = 2.0

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(features)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self(features)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)

        # Tanh-squash correction.
        log_prob = dist.log_prob(x_t)
        log_prob = log_prob - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mu_tanh = torch.tanh(mu)
        return y_t, log_prob, mu_tanh


class Critic(nn.Module):
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = Encoder(obs_shape, feature_dim=feature_dim)

        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        if detach_encoder:
            h = h.detach()
        h_action = torch.cat([h, action], dim=-1)
        return self.q1(h_action), self.q2(h_action)


class SACAgent:
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = int(action_dim)

        self.critic = Critic(obs_shape, action_dim, cfg.feature_dim, cfg.hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_dim, cfg.feature_dim, cfg.hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor = Actor(cfg.feature_dim, action_dim, cfg.hidden_dim).to(device)

        init_log_alpha = math.log(cfg.init_alpha)
        self.log_alpha = torch.tensor([init_log_alpha], device=device, requires_grad=True)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        obs: uint8 numpy (C,H,W)
        returns action numpy (action_dim,) in [-1,1]
        """
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0).float().div_(255.0)
        with torch.no_grad():
            h = self.critic.encoder(obs_t)
            if eval_mode:
                mu, _ = self.actor(h)
                a = torch.tanh(mu)
            else:
                a, _, _ = self.actor.sample(h)
        return a.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay: ReplayBuffer) -> dict:
        cfg = self.cfg
        obs, actions, rewards, next_obs, dones = replay.sample(cfg.batch_size, self.device)

        # DrQ augmentation
        obs = random_shift(obs, pad=cfg.drq_pad)
        next_obs = random_shift(next_obs, pad=cfg.drq_pad)

        rewards = rewards * cfg.reward_scale

        with torch.no_grad():
            next_h = self.critic.encoder(next_obs)
            next_action, next_logp, _ = self.actor.sample(next_h)
            target_q1, target_q2 = self.critic_target(next_obs, next_action, detach_encoder=False)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_logp
            target_q = rewards + (1.0 - dones) * cfg.gamma * target_v

        # Critic update
        cur_q1, cur_q2 = self.critic(obs, actions, detach_encoder=False)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=cfg.grad_clip_norm)
        self.critic_opt.step()

        # Actor + alpha update (encoder detached)
        h = self.critic.encoder(obs).detach()
        pi, logp, _ = self.actor.sample(h)
        q1_pi, q2_pi = self.critic(obs, pi, detach_encoder=True)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - min_q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=cfg.grad_clip_norm)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update target critic
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1.0 - cfg.tau)
                p_targ.data.add_(cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_logp": float(logp.mean().item()),
            "mean_q_pi": float(min_q_pi.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "cfg": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict):
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor.load_state_dict(state["actor"])
        log_alpha = float(np.array(state.get("log_alpha", [math.log(self.cfg.init_alpha)])).reshape(-1)[0])
        self.log_alpha = torch.tensor([log_alpha], device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)


