"""
Deep Q-Network agent trained on the from-scratch Gridworld.

Components:
    - Q-network: two-layer MLP (state → 64 → 64 → n_actions)
    - Replay buffer: simple ring buffer of (s, a, r, s', done)
    - Target network: hard-copy from main net every `target_sync_every` steps
    - ε-greedy exploration with linear decay
    - Loss: smooth-L1 (Huber) of TD error

Trains for ~250 episodes; produces dashboard figures.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridworld import Gridworld, GridworldConfig

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"
COLOR_LIGHT_BLUE = "#9EB7D6"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})


# --------------------------------------------------------------- DQN -------
class QNet(nn.Module):
    def __init__(self, n_states: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    episodes: int = 250
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 200
    target_sync_every: int = 100        # gradient steps
    seed: int = 42


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        return [np.stack(x) for x in zip(*batch)]

    def __len__(self) -> int:
        return len(self.buf)


def run_baseline(env: Gridworld, n_episodes: int = 50, seed: int = 0) -> list[float]:
    """Random-action baseline — what the agent must beat."""
    rng = random.Random(seed)
    rewards = []
    for _ in range(n_episodes):
        env.reset()
        total = 0.0
        done = False
        while not done:
            _, r, done = env.step(rng.randint(0, env.n_actions - 1))
            total += r
        rewards.append(total)
    return rewards


def train(cfg: DQNConfig | None = None) -> dict:
    cfg = cfg or DQNConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = Gridworld()
    qnet = QNet(env.n_states, env.n_actions)
    tgt = QNet(env.n_states, env.n_actions)
    tgt.load_state_dict(qnet.state_dict())

    optim = torch.optim.Adam(qnet.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size)

    rewards_log: list[float] = []
    lengths_log: list[int] = []
    epsilon_log: list[float] = []
    grad_step = 0

    for ep in range(1, cfg.episodes + 1):
        s = env.reset()
        total_r, n_steps, done = 0.0, 0, False
        eps = max(
            cfg.epsilon_end,
            cfg.epsilon_start - (cfg.epsilon_start - cfg.epsilon_end) * (ep / cfg.epsilon_decay_episodes),
        )

        while not done:
            # ε-greedy action selection.
            if random.random() < eps:
                a = random.randint(0, env.n_actions - 1)
            else:
                with torch.no_grad():
                    q = qnet(torch.from_numpy(s).unsqueeze(0))
                    a = int(q.argmax(dim=1).item())

            ns, r, done = env.step(a)
            buffer.push(s, np.int64(a), np.float32(r), ns, np.float32(done))
            s = ns
            total_r += r
            n_steps += 1

            # Learn.
            if len(buffer) >= cfg.batch_size:
                bs, ba, br, bns, bd = buffer.sample(cfg.batch_size)
                bs_t = torch.from_numpy(bs).float()
                ba_t = torch.from_numpy(ba).long()
                br_t = torch.from_numpy(br).float()
                bns_t = torch.from_numpy(bns).float()
                bd_t = torch.from_numpy(bd).float()

                q_pred = qnet(bs_t).gather(1, ba_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = tgt(bns_t).max(dim=1)[0]
                    q_target = br_t + cfg.gamma * q_next * (1.0 - bd_t)
                loss = F.smooth_l1_loss(q_pred, q_target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                grad_step += 1

                if grad_step % cfg.target_sync_every == 0:
                    tgt.load_state_dict(qnet.state_dict())

        rewards_log.append(total_r)
        lengths_log.append(n_steps)
        epsilon_log.append(eps)

        if ep % 25 == 0 or ep == 1:
            avg = np.mean(rewards_log[-25:])
            print(f"episode {ep:3d}  ε={eps:.3f}  avg_reward(last 25)={avg:.3f}")

    # Random baseline for comparison.
    baseline = run_baseline(Gridworld(), n_episodes=50, seed=0)

    return {
        "qnet": qnet,
        "rewards": rewards_log,
        "lengths": lengths_log,
        "epsilons": epsilon_log,
        "baseline_rewards": baseline,
        "env_cfg": env.cfg.__dict__,
    }


# ---------------------------------------------------------------- figures --
def fig_grid_layout(env_cfg: dict, out_path: Path) -> None:
    s = env_cfg["size"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    for r in range(s):
        for c in range(s):
            cell = (r, c)
            color = "#FFFFFF"
            label = ""
            if cell == tuple(env_cfg["start"]):
                color, label = COLOR_LIGHT_BLUE, "S"
            elif cell == tuple(env_cfg["goal"]):
                color, label = "#A8D5BA", "G"
            elif list(cell) in [list(p) for p in env_cfg["pits"]]:
                color, label = "#F2B5B5", "P"
            ax.add_patch(patches.Rectangle((c, s - 1 - r), 1, 1, facecolor=color,
                                           edgecolor=COLOR_LIGHT_GRAY, linewidth=1.0))
            if label:
                ax.text(c + 0.5, s - 1 - r + 0.5, label, ha="center", va="center",
                        fontsize=18, weight="bold", color=COLOR_TEXT)
    ax.set_xlim(0, s); ax.set_ylim(0, s)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Gridworld layout — start / goal / pits")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_reward_curve(rewards: list[float], baseline: list[float], out_path: Path) -> None:
    eps = range(1, len(rewards) + 1)
    rolling = np.convolve(rewards, np.ones(20) / 20, mode="valid")
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ax.plot(eps, rewards, color=COLOR_LIGHT_BLUE, linewidth=0.7, alpha=0.75,
            label="per-episode")
    ax.plot(range(20, len(rewards) + 1), rolling, color=COLOR_BLUE, linewidth=2.0,
            label="rolling mean (window=20)")
    ax.axhline(np.mean(baseline), color=COLOR_RED, linewidth=1.5, linestyle="--",
               label=f"random baseline mean = {np.mean(baseline):.2f}")
    ax.set_xlabel("Episode"); ax.set_ylabel("Episode total reward")
    ax.set_title("Episode reward over training")
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_episode_length(lengths: list[int], out_path: Path) -> None:
    eps = range(1, len(lengths) + 1)
    rolling = np.convolve(lengths, np.ones(20) / 20, mode="valid")
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    ax.plot(eps, lengths, color=COLOR_LIGHT_GRAY, linewidth=0.7, alpha=0.75,
            label="per-episode")
    ax.plot(range(20, len(lengths) + 1), rolling, color=COLOR_GRAY, linewidth=2.0,
            label="rolling mean (window=20)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps until episode end")
    ax.set_title("Episode length over training (lower = more direct path to goal)")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_policy(qnet: QNet, env_cfg: dict, out_path: Path) -> None:
    s = env_cfg["size"]
    arrows = ["↑", "↓", "←", "→"]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    qnet.eval()
    for r in range(s):
        for c in range(s):
            cell = (r, c)
            color = "#FFFFFF"
            label = ""
            if cell == tuple(env_cfg["start"]):
                color, label = COLOR_LIGHT_BLUE, "S"
            elif cell == tuple(env_cfg["goal"]):
                color, label = "#A8D5BA", "G"
            elif list(cell) in [list(p) for p in env_cfg["pits"]]:
                color, label = "#F2B5B5", "P"

            ax.add_patch(patches.Rectangle((c, s - 1 - r), 1, 1, facecolor=color,
                                           edgecolor=COLOR_LIGHT_GRAY, linewidth=1.0))
            if label:
                ax.text(c + 0.5, s - 1 - r + 0.5, label, ha="center", va="center",
                        fontsize=14, weight="bold", color=COLOR_TEXT)
                continue

            # Get best action for this state from the trained Q-net.
            obs = np.zeros(s * s, dtype=np.float32)
            obs[r * s + c] = 1.0
            with torch.no_grad():
                q = qnet(torch.from_numpy(obs).unsqueeze(0))
                best = int(q.argmax(dim=1).item())
            ax.text(c + 0.5, s - 1 - r + 0.5, arrows[best], ha="center", va="center",
                    fontsize=16, color=COLOR_BLUE, weight="bold")

    ax.set_xlim(0, s); ax.set_ylim(0, s)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Learned policy — best action per cell")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_epsilon(epsilons: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
    ax.plot(range(1, len(epsilons) + 1), epsilons, color=COLOR_RED, linewidth=1.6)
    ax.set_xlabel("Episode"); ax.set_ylabel("ε (exploration rate)")
    ax.set_title("ε-greedy exploration schedule")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    cfg = DQNConfig()
    out = train(cfg)

    final_rewards = out["rewards"][-25:]
    baseline_mean = float(np.mean(out["baseline_rewards"]))
    final_mean = float(np.mean(final_rewards))
    final_length = float(np.mean(out["lengths"][-25:]))

    print(f"\nFinal 25 episodes mean reward: {final_mean:.3f}")
    print(f"Random baseline mean reward:   {baseline_mean:.3f}")
    print(f"Final 25 episodes mean length: {final_length:.1f} steps")

    Path("results").mkdir(exist_ok=True)
    summary = {
        "config": cfg.__dict__,
        "env_cfg": out["env_cfg"],
        "final_25_mean_reward": final_mean,
        "final_25_mean_length": final_length,
        "baseline_mean_reward": baseline_mean,
        "rewards": out["rewards"],
        "lengths": out["lengths"],
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_grid_layout(out["env_cfg"], assets / "01_layout.png")
    fig_reward_curve(out["rewards"], out["baseline_rewards"], assets / "02_rewards.png")
    fig_episode_length(out["lengths"], assets / "03_lengths.png")
    fig_policy(out["qnet"], out["env_cfg"], assets / "04_policy.png")
    fig_epsilon(out["epsilons"], assets / "05_epsilon.png")

    print(f"\nFigures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
