# Market Search & Purchase Scheduling Using Reinforcement Learning
**P. Tharun Reddy | A00051705 | MSc Computing | University of Roehampton | 2026**

---

## Requirements

This project has ONE external dependency:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy   | ≥ 1.24  | All numerical computation, neural networks, Q-tables |

Everything else (json, os, sys, time, math, collections, typing) is Python standard library.

---

## Installation

```bash
# Option 1 — pip
pip install numpy

# Option 2 — using requirements.txt (recommended)
pip install -r code/requirements.txt

# Option 3 — specific version
pip install "numpy>=1.24.0"
```

**Python version:** 3.8 or later (tested on 3.12)

---

## How to Run

```bash
cd code

# Baseline experiment: all 13 agents, 1,000 test episodes each (~30 min)
python main.py

# 4 parameter sensitivity experiments
python run_experiments.py

# Quick test (5 episodes, confirms everything works)
python -c "
from environment import MarketEnvironment
from baseline_agents import GreedyAgent
from trainer import run_episode
env = MarketEnvironment(seed=42)
agent = GreedyAgent(env, seed=42)
s = run_episode(env, agent, training=False, ep_seed=0)
print(f'Test OK — Reward: {s[\"total_reward\"]:+.1f}  Completion: {s[\"completion_rate\"]:.0%}')
"
```

Open `experiments_dashboard.html` in any browser to view interactive charts.

---

## File Structure

```
code/
  requirements.txt         ← Install dependencies with pip install -r this
  environment.py           MDP: 6 stores, 20 goods, 120-min budget
  agents.py                Action encoding, BaseAgent class
  qlearning_agent.py       Q-Learning — off-policy tabular (200k episodes)
  sarsa_agent.py           SARSA — on-policy tabular (200k episodes)
  expected_sarsa_agent.py  Expected SARSA — variance-reduced tabular
  dqn_agent.py             DQN — experience replay + target network
  double_dqn_agent.py      Double DQN — removes overestimation bias
  dueling_dqn_agent.py     Dueling DQN — V(s) + A(s,a) architecture
  ppo_agent.py             PPO — actor-critic + GAE + clipped surrogate
  baseline_agents.py       Random agent, Greedy agent
  heuristic_agents.py      Best-Deal, TimeAware, Prof-Heuristic, Freshness-1st
  trainer.py               train_agent(), evaluate_agent(), run_episode()
  main.py                  Main runner: all 13 agents, results.json
  run_experiments.py       4 parameter experiments
  env_exp1_duration.py     Exp1: 180-min budget + store revisiting
  env_exp2_goods.py        Exp2: 10 goods per store (restricted coverage)
  env_exp3_availability.py Exp3: 80% brand availability

figures/                   8 PNG figures embedded in the report
experiments_dashboard.html Open in any browser — interactive results
master_results.json        Verified baseline results (all 13 agents, 1000 eps)
all_experiment_results.json All 4 experiment results
FINAL_REPORT_SUBMISSION.docx The dissertation report
README.md                  This file
```

---

## Key Results (baseline, 1,000 test episodes, 95% CI = ±1.96σ/√1000)

| Agent | Reward | 95% CI | Completion | Discount |
|---|---|---|---|---|
| Dueling-DQN | +498.66 | [495.4, 501.9] | 98.3% | −17.1% |
| Double-DQN | +475.40 | [471.8, 479.0] | 97.1% | −15.8% |
| DQN | +472.93 | [469.5, 476.4] | 98.7% | −16.2% |
| SARSA | +375.77 | [369.9, 381.7] | 95.9% | −8.7% |
| Q-Learning | +355.00 | [348.7, 361.3] | 91.5% | −8.9% |
| Greedy | +301.64 | [298.0, 305.3] | 100.0% | −6.9% |
| PPO | +184.17 | [180.1, 188.2] | 99.6% | −2.1% |
| Random | −50.88 | [−61.7, −40.1] | 62.8% | −0.5% |
