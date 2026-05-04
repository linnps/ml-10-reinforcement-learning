"""
A from-scratch 6x6 Gridworld environment for RL.

State:    (row, col)  ∈  {0..5} × {0..5}, encoded as 36-D one-hot for the network.
Actions:  0 = up, 1 = down, 2 = left, 3 = right
Reward:   +1.0 on reaching the goal
          -1.0 on stepping into a pit
          -0.01 per step (encourage shorter paths)
          attempting to leave the grid is a no-op (no penalty beyond step cost)
Episode end: goal, pit, or max_steps (default 60).

No third-party env dependency — this is a complete, self-contained simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridworldConfig:
    size: int = 6
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (5, 5)
    pits: tuple[tuple[int, int], ...] = ((1, 3), (3, 1), (4, 3))
    max_steps: int = 60
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    pit_reward: float = -1.0


class Gridworld:
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, cfg: GridworldConfig | None = None) -> None:
        self.cfg = cfg or GridworldConfig()
        self._step_count = 0
        self.pos = self.cfg.start

    @property
    def n_states(self) -> int:
        return self.cfg.size * self.cfg.size

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> np.ndarray:
        self.pos = self.cfg.start
        self._step_count = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.cfg.size - 1, self.pos[0] + dr))
        nc = max(0, min(self.cfg.size - 1, self.pos[1] + dc))
        self.pos = (nr, nc)
        self._step_count += 1

        reward = self.cfg.step_penalty
        done = False
        if self.pos == self.cfg.goal:
            reward = self.cfg.goal_reward
            done = True
        elif self.pos in self.cfg.pits:
            reward = self.cfg.pit_reward
            done = True
        elif self._step_count >= self.cfg.max_steps:
            done = True

        return self._obs(), float(reward), done

    def _obs(self) -> np.ndarray:
        idx = self.pos[0] * self.cfg.size + self.pos[1]
        v = np.zeros(self.n_states, dtype=np.float32)
        v[idx] = 1.0
        return v

    def render_ascii(self) -> str:
        s = self.cfg.size
        rows = []
        for r in range(s):
            row = []
            for c in range(s):
                if (r, c) == self.cfg.start:
                    row.append("S")
                elif (r, c) == self.cfg.goal:
                    row.append("G")
                elif (r, c) in self.cfg.pits:
                    row.append("P")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)


if __name__ == "__main__":
    env = Gridworld()
    print(env.render_ascii())
