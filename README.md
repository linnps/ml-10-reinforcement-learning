<div align="center">

# Reinforcement Learning — DQN on a From-Scratch Gridworld

**Build the environment, build the agent, watch ε-greedy exploration give way to a near-optimal policy.**

![status](https://img.shields.io/badge/status-complete-3B6EA8?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-3B6EA8?style=flat-square)
![framework](https://img.shields.io/badge/framework-PyTorch-3B6EA8?style=flat-square)
![env](https://img.shields.io/badge/env-custom%20gridworld-7A7A7A?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-7A7A7A?style=flat-square)

</div>

---

## At a glance

> A 6×6 Gridworld coded from scratch (no Gymnasium, no third-party env). Pits to avoid, a goal to reach, a tiny step penalty to discourage wandering. A Deep Q-Network — replay buffer, target network, ε-greedy schedule — learns a near-optimal policy in 250 episodes on CPU.

<table>
<tr>
<td align="center" width="33%">
<sub>Final 25-ep mean reward</sub><br>
<b style="font-size:1.6em; color:#3B6EA8;">+0.91</b><br>
<sub>(optimal ≈ +0.90 with step penalty)</sub>
</td>
<td align="center" width="33%">
<sub>Random baseline</sub><br>
<b style="font-size:1.6em; color:#C04040;">−1.11</b><br>
<sub>most random walks fall in a pit</sub>
</td>
<td align="center" width="33%">
<sub>Final episode length</sub><br>
<b style="font-size:1.6em; color:#3B6EA8;">10.3</b><br>
<sub>steps (Manhattan optimum = 10)</sub>
</td>
</tr>
</table>

| Metric | DQN (final 25 ep) | Random baseline | Optimal |
|---|---:|---:|---:|
| Mean episode reward | **+0.91** | −1.11 | ≈ +0.90 |
| Mean episode length (steps to terminate) | **10.3** | ~30 | 10 |
| Pit-fall rate | ~0% | ~70% | 0% |

<sub>**Headline finding:** the trained DQN is essentially **at the theoretical optimum**. Manhattan distance from start (0,0) to goal (5,5) is 10 steps — the agent's average of 10.3 means it takes a 10-step path on most episodes, with occasional 11-step detours around the pit-cluster. The reward curve shows a clean transition from random behavior (~episode 0–50) through exploration-driven learning (~50–150) to convergence (~150+).</sub>

---

## Dashboard

### 1. The environment

![layout](assets/01_layout.png)

A 6×6 grid:

- **S** at (0,0) is the start; **G** at (5,5) is the goal.
- Three **P** cells are pits — stepping on one ends the episode with reward −1.0.
- Reaching the goal gives +1.0.
- Every other step costs −0.01 (encourages shorter paths without dominating the reward signal).
- Episodes also terminate after 60 steps to bound exploration.

The agent observes the state as a 36-dimensional one-hot. It chooses one of 4 actions: up, down, left, right.

### 2. Reward curve over training

![rewards](assets/02_rewards.png)

The light-blue trace is per-episode reward; the dark-blue line is the rolling mean (window = 20). The dashed red line marks the random-baseline mean (−1.11).

Three phases are visible:

- **Episodes 1–50**: the agent is essentially random (high ε). Per-episode reward sits near the baseline; many episodes end in a pit (−1) or by timing out.
- **Episodes 50–150**: ε is decaying. The replay buffer is full enough that gradient updates start producing useful Q-values. The rolling mean climbs from −1 to +0.5.
- **Episodes 150+**: ε is near its floor (0.05). The policy is mostly greedy. Rewards stabilize around +0.9.

The occasional dip below the line in the late training is **deliberate**: ε floors at 5%, so 1 in 20 actions is still random — and a single random action onto a pit costs −1.0, dragging that one episode's total reward.

### 3. Episode length over training

![lengths](assets/03_lengths.png)

A dual to the reward curve. As the policy improves, episodes get *shorter* (the agent reaches the goal faster) — except the dips, which are episodes that ended by falling in a pit (also short, but for the wrong reason). The settling-around-10 plateau is the 10-step Manhattan-optimum line.

### 4. The learned policy

![policy](assets/04_policy.png)

Best action per cell, taken from the trained Q-network. **This is the artifact** — the figure that proves the agent has actually learned the task, not just collected good rewards.

Read it as a flow chart: from any non-terminal cell, follow the arrow. The arrows form a coherent diagonal "staircase" from S to G that **routes around all three pits**. There are several optimal paths through this grid; the policy has settled on one of them. A few cells (e.g., the top-right corner) have arrows pointing *down* even though *left* would also be optimal — those Q-values are tied within numerical noise, and tie-breaking goes whichever way the random initialization happened to prefer.

### 5. ε-greedy exploration schedule

![epsilon](assets/05_epsilon.png)

Linear decay from 1.0 to 0.05 over the first 200 episodes, then floor. Without a non-zero floor, the policy can lock into local optima it never escapes; without aggressive early decay, it never starts exploiting what it's learned. The shape of this curve is hyperparameter #1 in any DQN, and getting it wrong is the most common failure mode.

---

## What's actually happening

### Q-learning, in three sentences

1. The Q-value of (state, action) is "the total discounted future reward you expect if you take this action and then follow your policy."
2. The Bellman update says: Q(s, a) should equal `reward + γ · max_a' Q(s', a')` — your immediate reward plus the discounted best you can do from the next state.
3. Q-learning fits Q to that equation by gradient descent, using the right-hand side as a target for the left-hand side.

### What "deep" adds — and why it needs babysitting

Tabular Q-learning stores Q[s, a] in a literal table. Works fine for 36 states × 4 actions = 144 entries. Falls apart when state spaces get continuous or huge.

A Deep Q-Network replaces the table with a neural net `Q_θ(s) → R^|A|`. Now you can handle any state representation, but two things break that need fixing:

1. **Correlated samples**: consecutive (s, a, r, s') pairs from the same episode are statistically correlated, which breaks SGD's i.i.d. assumption and produces unstable gradients.
   → *Fix*: a **replay buffer**. Store transitions; sample uniformly at random.
2. **Moving targets**: the target `r + γ · max Q_θ(s')` *uses the same Q_θ that's being updated*, so the optimization target keeps shifting under the gradient.
   → *Fix*: a **target network**. Maintain a slowly-updated copy of Q_θ for computing targets; refresh it every N gradient steps.

Both fixes are in `train_dqn.py`. Removing either one usually causes training to diverge or oscillate.

### Hyperparameters that actually matter

| Knob | Why it matters | Value here |
|---|---|---|
| γ (discount) | Trades immediate vs distant rewards. 0.95 = "care up to ~20 steps ahead" | 0.95 |
| ε start / end / decay | Exploration vs exploitation. Too aggressive → never explores, too slow → never exploits | 1.0 → 0.05 over 200 ep |
| target_sync_every | Stability vs adaptivity of the target | 100 grad steps |
| buffer size | How far back transitions persist for sampling | 5000 |
| batch size | Gradient noise level | 64 |

### Mental model

| Setting | Use |
|---|---|
| Discrete actions, small state, you understand the dynamics | **Tabular Q-learning** — DQN is overkill |
| Continuous or high-dim state (pixels, sensor readings) | **DQN** with replay + target network (this project) |
| Continuous *actions* (steering angles, joint torques) | **DDPG / SAC** — DQN doesn't handle continuous actions natively |
| Need policy uncertainty, sample efficiency | **Policy gradient family** (PPO, TRPO) — DQN is value-based, not policy-based |

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_dqn.py
```

Wall-time: ~30 seconds on CPU. Outputs land in `assets/` (5 dashboard PNGs) and `results/metrics.json`.

### Tweak the difficulty

`GridworldConfig` in [`gridworld.py`](gridworld.py):

```python
GridworldConfig(
    size=6,
    start=(0, 0),
    goal=(5, 5),
    pits=((1, 3), (3, 1), (4, 3)),
    max_steps=60,
    step_penalty=-0.01,
    goal_reward=1.0,
    pit_reward=-1.0,
)
```

Try `pits` arranged into a wall (e.g., `((2, 1), (2, 2), (2, 3), (2, 4))`) so the agent has to find the gap at column 0 or 5. The same DQN, with the same hyperparameters, will need ~50% more episodes to find the route — visible as a longer plateau before the rolling-mean climb.

`DQNConfig` in [`train_dqn.py`](train_dqn.py) exposes the agent's knobs.

---

## Project layout

```
10-reinforcement-learning/
├── README.md              ← this dashboard
├── requirements.txt
├── gridworld.py           ← from-scratch environment (no third-party deps)
├── train_dqn.py           ← Q-network, replay buffer, target net, training loop, figures
├── assets/                ← 5 dashboard PNGs
└── results/metrics.json
```

---

## What I learned

- **A learning curve is the most informative single figure in RL.** A scalar "final accuracy" hides everything: how fast learning happened, whether it's stable, what the dip variance looks like. Always plot the rolling mean and the per-episode noise on the same axes.
- **The policy plot is the *real* test.** A high reward could come from a memorized trajectory; a coherent flow-field of arrows that routes around obstacles is evidence the agent has actually generalized. For any RL project I do in the future, "produce a policy visualization" goes alongside "report final reward" as a non-negotiable.
- **Replay buffer + target network aren't optional.** Removing either one is the most common failure mode I see in beginner DQN code. They're not implementation details — they're the whole reason DQN works.
- **An ε-floor matters more than people give it credit for.** With ε → 0, an unlucky exploration episode can trap the agent in a local optimum it never escapes. The 0.05 floor here is the difference between "agent reaches +0.9 and stays there" and "agent reaches +0.5 and oscillates."

---

<div align="center">
<sub>Part of a hands-on machine-learning portfolio. Environment is implemented from scratch; no third-party simulator dependency.</sub>
</div>
