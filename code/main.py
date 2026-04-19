"""
============================================================
  Market Search and Purchase Scheduling
  Using Reinforcement Learning

  
  P. Tharun Reddy | A00051705
  University of Roehampton | March 2026
============================================================

HOW TO RUN:
  pip install numpy
  python main.py

AGENTS (13 total):

  Deep RL — neural networks, trained from experience:
    Dueling-DQN       (dual-stream V+A architecture, 10,000 episodes)
    DQN               (experience replay + target network, 10,000 episodes)
    Double-DQN        (decoupled action selection/evaluation, 10,000 episodes)
    PPO               (on-policy actor-critic, 10,000 episodes)

  Tabular RL — Q-table, no neural network:
    Q-Learning        (off-policy, 200,000 episodes)
    SARSA             (on-policy, 200,000 episodes)
    Expected-SARSA    (variance-reduced on-policy, 200,000 episodes)

  Heuristic — hand-coded rules, no training:
    Greedy            (coverage-maximising, buy best available)
    Best-Deal         (preference + price scoring)
    TimeAware         (adaptive quality thresholds)
    Prof-Heuristic    (preference ≥ 3 AND expiry < 4)
    Freshness-First   (urgency-driven, lowest expiry first)

  Baseline:
    Random            (uniform random valid action)

ENVIRONMENT:
  6 stores (star graph), 120-min budget, 20 goods, 5 brands each
  3 brands per store, 60% availability, ±30% price premium
  Expiry: Uniform[0–7] days, γ = 0.95

HYPERPARAMETERS (fixed and consistent across all deep RL agents):
  lr          = 3e-4   (canonical DQN learning rate)
  gamma       = 0.95
  eps_start   = 1.0  →  eps_end = 0.05
  eps_decay   = 0.9997  (reaches eps_end after ~333 eps / 3.3% of training)
  batch_size  = 64
  buffer_size = 10,000
  target_update_freq = 200 steps
  h1 = 64, h2 = 32  (two-hidden-layer MLP)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time

from environment import MarketEnvironment
from qlearning_agent import QLearningAgent
from sarsa_agent import SARSAAgent
from expected_sarsa_agent import ExpectedSARSAAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent
from baseline_agents import RandomAgent, GreedyAgent
from heuristic_agents import (HeuristicAgent, BestDealAgent,
                               TimeAwareAgent, FreshnessFirstAgent)
from trainer import train_agent, evaluate_agent

# ── Configuration ────────────────────────────────────────────────────────────
SEED          = 42
TRAIN_TABULAR = 200_000
TRAIN_DEEP    = 10_000
TEST_EPISODES = 1_000
OUTPUT_FILE   = "results.json"

# Shared deep RL hyperparameters (consistent across all agents)
DEEP_HP = dict(
    lr=3e-4, gamma=0.95,
    eps_start=1.0, eps_end=0.05, eps_decay=0.9997,
    batch_size=64, target_update_freq=200, h1=64, h2=32
)

# ── Setup ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Market Search & Purchase Scheduling Using RL")
print(f"  P. Tharun Reddy | A00051705 | MSc Computing")
print("=" * 65)

env = MarketEnvironment(seed=SEED)
results = {}
training_curves = {}

# ── Agent definitions ─────────────────────────────────────────────────────────
agents_config = [
    # ── Deep RL ───────────────────────────────────────────────────────────────
    ("Dueling-DQN",
     DuelingDQNAgent(env, seed=SEED,   **DEEP_HP), TRAIN_DEEP),
    ("DQN",
     DQNAgent(env,        seed=SEED+1, **DEEP_HP), TRAIN_DEEP),
    ("Double-DQN",
     DoubleDQNAgent(env,  seed=SEED+2, **DEEP_HP), TRAIN_DEEP),
    ("PPO",
     PPOAgent(env, seed=SEED+3,
              lr_actor=3e-4, lr_critic=1e-3, gamma=0.95,
              lam=0.95, clip_eps=0.2, n_epochs=4,
              rollout_steps=256, entropy_coef=0.01,
              h1=64, h2=32),             TRAIN_DEEP),

    # ── Tabular RL ────────────────────────────────────────────────────────────
    ("Q-Learning",
     QLearningAgent(env, seed=SEED+4,
                    lr=0.2, gamma=0.95,
                    eps_start=1.0, eps_end=0.02,
                    eps_decay=0.999975),  TRAIN_TABULAR),
    ("SARSA",
     SARSAAgent(env,     seed=SEED+5,
                lr=0.2, gamma=0.95,
                eps_start=1.0, eps_end=0.02,
                eps_decay=0.999975),      TRAIN_TABULAR),
    ("Expected-SARSA",
     ExpectedSARSAAgent(env, seed=SEED+6,
                        lr=0.2, gamma=0.95,
                        eps_start=1.0, eps_end=0.02,
                        eps_decay=0.999975), TRAIN_TABULAR),

    # ── Heuristic agents ──────────────────────────────────────────────────────
    ("Greedy",         GreedyAgent(env,         seed=SEED+7),  0),
    ("Best-Deal",      BestDealAgent(env,        seed=SEED+8),  0),
    ("TimeAware",      TimeAwareAgent(env,       seed=SEED+9),  0),
    ("Prof-Heuristic", HeuristicAgent(env,       seed=SEED+10), 0),
    ("Freshness-1st",  FreshnessFirstAgent(env,  seed=SEED+11), 0),

    # ── Baseline ──────────────────────────────────────────────────────────────
    ("Random",         RandomAgent(env,          seed=SEED+12), 0),
]

# ── Training & evaluation loop ───────────────────────────────────────────────
for name, agent, n_eps in agents_config:
    print(f"\n{'─' * 55}")
    print(f"  Agent: {name}")
    print(f"{'─' * 55}")
    t0 = time.time()

    if n_eps > 0:
        print(f"  Training for {n_eps:,} episodes...")
        history = train_agent(agent, n_episodes=n_eps,
                              log_interval=max(1000, n_eps // 10))
    else:
        print("  No training — evaluating directly")
        history = []

    train_time = time.time() - t0

    if history:
        step = max(1, len(history) // 200)
        training_curves[name] = {
            'rewards':    [float(h['total_reward'])    for h in history[::step]],
            'completion': [float(h['completion_rate']) for h in history[::step]],
        }

    print(f"  Evaluating on {TEST_EPISODES:,} test episodes...")
    eval_stats, all_ep_stats = evaluate_agent(agent, n_episodes=TEST_EPISODES)
    eval_stats['train_time_s']   = round(train_time, 2)
    eval_stats['train_episodes'] = n_eps
    results[name] = eval_stats

    print(f"  ✓  Reward:     {eval_stats['total_reward_mean']:>+9.2f}  ±{eval_stats['total_reward_std']:.2f}")
    print(f"     Completion: {eval_stats['completion_rate_mean']:>9.2%}")
    print(f"     Stores:     {eval_stats['stores_visited_mean']:>9.2f}")
    print(f"     Premium:    {eval_stats['avg_premium_pct_mean']:>+9.2f}%")
    print(f"     Pref Score: {eval_stats['avg_pref_score_mean']:>9.2f} / 5")
    print(f"     Expiry:     {eval_stats['avg_expiry_days_mean']:>9.2f} days")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    'results': results,
    'training_curves': training_curves,
    'config': {
        'seed': SEED,
        'train_episodes_tabular': TRAIN_TABULAR,
        'train_episodes_deep_rl': TRAIN_DEEP,
        'test_episodes': TEST_EPISODES,
        'hyperparameters_deep_rl': DEEP_HP,
        'environment': {
            'n_stores': env.N_STORES,
            'n_goods': env.N_GOODS,
            'n_brands': env.N_BRANDS,
            'brands_per_store': env.BRANDS_PER_STORE,
            'total_duration_min': env.TOTAL_DURATION,
            'travel_time_min': env.TRAVEL_TIME,
            'visit_time_min': env.VISIT_TIME,
            'avail_prob': env.AVAIL_PROB,
            'max_premium': env.MAX_PREMIUM,
        }
    }
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"  FINAL RESULTS — {TEST_EPISODES} Test Episodes Each")
print(f"{'=' * 70}")
print(f"  {'Agent':<18} {'Reward':>9} {'±Std':>7} {'Compl%':>8} "
      f"{'Stores':>7} {'Premium%':>9} {'Pref':>6} {'Expiry':>7}")
print(f"  {'─' * 68}")

ordered = ["Dueling-DQN","DQN","Double-DQN","PPO",
           "Q-Learning","SARSA","Expected-SARSA",
           "Greedy","Best-Deal","TimeAware","Prof-Heuristic",
           "Freshness-1st","Random"]

for name in ordered:
    if name not in results:
        continue
    r = results[name]
    print(f"  {name:<18} "
          f"{r['total_reward_mean']:>+9.2f} "
          f"{r['total_reward_std']:>7.2f} "
          f"{r['completion_rate_mean']:>8.2%} "
          f"{r['stores_visited_mean']:>7.2f} "
          f"{r['avg_premium_pct_mean']:>+8.2f}% "
          f"{r['avg_pref_score_mean']:>6.2f} "
          f"{r['avg_expiry_days_mean']:>7.2f}")

print(f"\n  Results saved → {OUTPUT_FILE}")
print(f"  Open experiments_dashboard.html in browser for interactive charts")
print(f"{'=' * 70}")

# ── Generate all 8 figures ────────────────────────────────────────────────────
try:
    from plot_figures import generate_all_figures
    generate_all_figures(
        results=results,
        training_curves=training_curves,
    )
except Exception as e:
    print(f"\n  [Warning] Figure generation failed: {e}")
    print(f"  Run manually: python plot_figures.py")
