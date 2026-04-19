"""
============================================================
  Extended Experiments: PPO + 3 Parameter Studies
  Market Search and Purchase Scheduling Using RL


============================================================

EXPERIMENTS:
  Exp 0: PPO (policy gradient) vs DQN family vs Q-Learning — baseline env
  Exp 1: Duration 3 hours (180 min), store revisiting allowed
  Exp 2: Each store stocks only 10 goods (specialised coverage with overlap)
  Exp 3: Brand availability probability raised from 60% to 80%

TRAINING BUDGETS:
  Tabular (Q-Learning): 50,000 episodes   [full: 200k — reduce for speed]
  Deep RL (DQN/DDQN):  3,000 episodes     [full: 10k  — convergence confirmed]
  PPO:                 3,000 episodes     [on-policy policy gradient]
  Heuristics:          0 episodes         [no training]
  Eval:                500 test episodes  [statistically robust]

BASELINE RESULTS (from main.py, full training):
  Double-DQN: +475.40 | DQN: +472.93 | Q-Learning: +355.00
  Greedy: +301.64 | TimeAware: +254.37 | Quality-Thresh: +148.69
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time
from typing import Dict, List, Tuple

from environment import MarketEnvironment
from env_exp1_duration import MarketEnvExp1
from env_exp2_goods import MarketEnvExp2
from env_exp3_availability import MarketEnvExp3

from qlearning_agent import QLearningAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from ppo_agent import PPOAgent
from baseline_agents import GreedyAgent
from heuristic_agents import HeuristicAgent, TimeAwareAgent
from trainer import train_agent, evaluate_agent

# ── Configuration ─────────────────────────────────────────────────────────────
SEED           = 42
TRAIN_TABULAR  = 50_000    # increased from 200k baseline but enough for convergence
TRAIN_DEEP     = 3_000     # convergence confirmed at 3k eps
TEST_EPISODES  = 500
OUTPUT_FILE    = "experiment_results.json"

BASELINE = {
    "Double-DQN":    {"reward": 475.40, "completion": 0.9708, "stores": 2.71, "premium": -15.80, "pref": 3.39, "expiry": 3.48},
    "DQN":           {"reward": 472.93, "completion": 0.9866, "stores": 2.47, "premium": -16.20, "pref": 3.41, "expiry": 3.50},
    "Q-Learning":    {"reward": 355.00, "completion": 0.9150, "stores": 4.50, "premium": -8.90,  "pref": 3.35, "expiry": 3.44},
    "PPO":           {"reward": None,   "completion": None,   "stores": None, "premium": None,   "pref": None, "expiry": None},
    "Greedy":        {"reward": 301.64, "completion": 1.0000, "stores": 6.00, "premium": -6.93,  "pref": 3.45, "expiry": 3.50},
    "TimeAware":   {"reward": 254.37, "completion": 0.9998, "stores": 5.80, "premium": -1.77,  "pref": 4.50, "expiry": 2.42},
    "Quality-Thresh":{"reward": 148.69, "completion": 0.9492, "stores": 5.32, "premium": -0.48,  "pref": 4.17, "expiry": 1.51},
}

all_results = {"baseline": BASELINE, "experiments": {}}


# ── Agent factory ─────────────────────────────────────────────────────────────
def make_agents(env, seed=SEED):
    return [
        ("Double-DQN",
         DoubleDQNAgent(env, seed=seed+4, lr=3e-4, gamma=0.95,
                        eps_start=1.0, eps_end=0.05, eps_decay=0.9997,
                        batch_size=64, target_update_freq=200, h1=64, h2=32),
         TRAIN_DEEP),

        ("DQN",
         DQNAgent(env, seed=seed+3, lr=3e-4, gamma=0.95,
                  eps_start=1.0, eps_end=0.05, eps_decay=0.9997,
                  batch_size=64, target_update_freq=200, h1=64, h2=32),
         TRAIN_DEEP),

        ("Q-Learning",
         QLearningAgent(env, seed=seed, lr=0.2, gamma=0.95,
                        eps_start=1.0, eps_end=0.02, eps_decay=0.999975),
         TRAIN_TABULAR),

        ("PPO",
         PPOAgent(env, seed=seed+10, lr_actor=3e-4, lr_critic=1e-3,
                  gamma=0.95, lam=0.95, clip_eps=0.2,
                  n_epochs=4, rollout_steps=256, entropy_coef=0.01,
                  h1=64, h2=32),
         TRAIN_DEEP),

        ("Greedy",
         GreedyAgent(env, seed=seed+5), 0),

        ("TimeAware",
         TimeAwareAgent(env, seed=seed+8), 0),

        ("Quality-Thresh",
         HeuristicAgent(env, seed=seed+6), 0),
    ]


# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(label: str, env, agents_cfg: List[Tuple]) -> Dict:
    results = {}
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for name, agent, n_eps in agents_cfg:
        print(f"\n  [{name}]", end="", flush=True)
        t0 = time.time()

        if n_eps > 0:
            print(f"  training {n_eps:,} eps...", end="", flush=True)
            train_agent(agent, n_episodes=n_eps,
                        log_interval=max(n_eps, n_eps+1))  # silent training
        else:
            print("  no training...", end="", flush=True)

        elapsed = time.time() - t0
        print(f"  evaluating {TEST_EPISODES} eps...", end="", flush=True)

        eval_stats, _ = evaluate_agent(agent, n_episodes=TEST_EPISODES)
        eval_stats["train_episodes"] = n_eps
        eval_stats["train_time_s"] = round(elapsed, 1)
        results[name] = eval_stats

        r = eval_stats
        print(f"\n     reward={r['total_reward_mean']:+.1f}  "
              f"completion={r['completion_rate_mean']:.1%}  "
              f"stores={r['stores_visited_mean']:.2f}  "
              f"premium={r['avg_premium_pct_mean']:+.1f}%  "
              f"pref={r['avg_pref_score_mean']:.2f}  "
              f"expiry={r['avg_expiry_days_mean']:.2f}d  "
              f"[{elapsed:.0f}s train]")

    return results


# ════════════════════════════════════════════════════════
#  EXP 0 — PPO on baseline environment
# ════════════════════════════════════════════════════════
env0 = MarketEnvironment(seed=SEED)
r0 = run_experiment("EXP 0: PPO vs Deep RL vs Tabular — Baseline Environment", env0, make_agents(env0))
all_results["experiments"]["Exp0_PPO_Baseline"] = {
    "description": "PPO (policy gradient) compared against DQN, Double-DQN, Q-Learning, and heuristics on the original environment",
    "environment_change": "None — identical to dissertation baseline",
    "results": r0
}

# ════════════════════════════════════════════════════════
#  EXP 1 — 3-hour duration, store revisiting allowed
# ════════════════════════════════════════════════════════
env1 = MarketEnvExp1(seed=SEED)
r1 = run_experiment("EXP 1: Duration 180 min — Store Revisiting Allowed", env1, make_agents(env1))
all_results["experiments"]["Exp1_Duration_180min"] = {
    "description": "Duration extended from 120 to 180 minutes. Store revisiting allowed: first visit costs 15 min, subsequent visits cost only 5 min travel.",
    "environment_change": "TOTAL_DURATION: 120 → 180 min; revisiting a store costs travel time only (5 min), not full 15-min visit again",
    "results": r1
}

# ════════════════════════════════════════════════════════
#  EXP 2 — 10 goods per store (restricted specialised coverage)
# ════════════════════════════════════════════════════════
env2 = MarketEnvExp2(seed=SEED)
r2 = run_experiment("EXP 2: Restricted Coverage — 10 Goods per Store", env2, make_agents(env2))
all_results["experiments"]["Exp2_10Goods_Per_Store"] = {
    "description": "Each store stocks only 10 of 20 goods (3 brands each). Stores specialise with deliberate partial overlap ensuring every good is reachable.",
    "environment_change": "GOODS_PER_STORE: 20 → 10. Store mapping: S0=goods0-9, S1=goods10-19, S2=goods0-4+15-19, S3=goods5-9+10-14, S4=goods0-4+10-14, S5=goods5-9+15-19",
    "results": r2
}

# ════════════════════════════════════════════════════════
#  EXP 3 — 80% availability probability
# ════════════════════════════════════════════════════════
env3 = MarketEnvExp3(seed=SEED)
r3 = run_experiment("EXP 3: Higher Availability — 80% Probability", env3, make_agents(env3))
all_results["experiments"]["Exp3_Avail_80pct"] = {
    "description": "Brand availability probability raised from 60% to 80%. All other parameters identical to baseline.",
    "environment_change": "AVAIL_PROB: 0.60 → 0.80",
    "results": r3
}

# ── Save results ──────────────────────────────────────────────────────────────
def safe(obj):
    if isinstance(obj, dict): return {k: safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64, float)): return round(float(obj), 4)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj

with open(OUTPUT_FILE, "w") as f:
    json.dump(safe(all_results), f, indent=2)

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print("  COMPLETE RESULTS SUMMARY")
print(f"{'='*72}")

exp_labels = {
    "Exp0_PPO_Baseline":      "Exp0 PPO+Baseline",
    "Exp1_Duration_180min":   "Exp1 180min+Revisit",
    "Exp2_10Goods_Per_Store": "Exp2 10goods/store",
    "Exp3_Avail_80pct":       "Exp3 80% avail",
}

agent_order = ["Double-DQN", "DQN", "PPO", "Q-Learning", "Greedy", "TimeAware", "Quality-Thresh"]

print(f"\n  {'Agent':<16}  {'Reward':>8}  {'Complet':>7}  {'Stores':>6}  {'Premium':>8}  {'Pref':>5}  {'Expiry':>6}")
print(f"  {'-'*66}")

for exp_key, exp_label in exp_labels.items():
    print(f"\n  ── {exp_label} ──")
    res = all_results["experiments"][exp_key]["results"]
    for agent_name in agent_order:
        if agent_name not in res: continue
        s = res[agent_name]
        print(f"  {agent_name:<16}  "
              f"{s['total_reward_mean']:>+8.1f}  "
              f"{s['completion_rate_mean']:>7.1%}  "
              f"{s['stores_visited_mean']:>6.2f}  "
              f"{s['avg_premium_pct_mean']:>+7.1f}%  "
              f"{s['avg_pref_score_mean']:>5.2f}  "
              f"{s['avg_expiry_days_mean']:>6.2f}d")

print(f"\n  Saved to: {OUTPUT_FILE}")
print(f"{'='*72}")
