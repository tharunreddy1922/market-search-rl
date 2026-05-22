# Tharun Reddy — A00051705
# MSc Data Science — RL Shopping Optimisation
# UPDATED EXPERIMENTS PACKAGE

## What's New (Professor's Requirements)

### New Algorithms Added:
- **REINFORCE** — Monte Carlo Policy Gradient (Williams, 1992)
- **A2C** — Advantage Actor-Critic (Mnih et al., 2016)
- **Value Iteration** — Dynamic Programming / Model-Based

### New Experiments:
- All RL algorithms tested across ALL parameter configurations
- Records: time spent, stores visited, completion rate for EVERY experiment
- 4 experiment types × 3 parameter values × 13 algorithms

## How to Run

### Step 1 — Install requirements
```
pip install numpy matplotlib
```

### Step 2 — Run all experiments
```
cd code/
python run_all_experiments.py
```

### What it produces:
- `full_experiment_results.json` — all raw results
- `../figures/fig_*.png` — 6 updated figures
- `../experiments_dashboard.html` — interactive dashboard

### Estimated runtime: 20-40 minutes on modern PC

## Experiment Grid

| Experiment | Parameter | Values |
|------------|-----------|--------|
| Baseline   | —         | 120min, 20 goods, 60% avail |
| Duration   | Time budget | 60 / 120 / 180 min |
| Availability | Avail prob | 40% / 60% / 80% |
| Goods Count | N goods   | 10 / 20 / 40 |

## Algorithm Summary

| Algorithm    | Type           | Training |
|-------------|----------------|---------|
| Q-Learning  | Tabular RL     | Episodes |
| SARSA       | Tabular RL     | Episodes |
| ExpSARSA    | Tabular RL     | Episodes |
| DQN         | Deep RL        | Episodes |
| DoubleDQN   | Deep RL        | Episodes |
| DuelingDQN  | Deep RL        | Episodes |
| PPO         | Policy Gradient| Rollouts |
| REINFORCE ⭐ | Policy Gradient| Episodes |
| A2C ⭐       | Policy Gradient| Online   |
| ValueIter ⭐ | Model-Based DP | Analytic |
| Greedy      | Heuristic      | None     |
| TimeAware   | Heuristic      | None     |
| Freshness   | Heuristic      | None     |

⭐ = new in this version
