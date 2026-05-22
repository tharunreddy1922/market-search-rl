"""
============================================================
STEP 2: Policy Gradient Training-Size Sweep
Tharun Reddy — A00051705

Run this after the main experiment script finishes.
Sweeps PPO, REINFORCE, A2C across training budgets.

Then combines ALL results into one final dashboard.
============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import time

from environment import MarketEnvironment
from ppo_agent import PPOAgent
from reinforce_agent import REINFORCEAgent
from a2c_agent import A2CAgent
from trainer import run_episode

SEED     = 42
TEST_EP  = 200
PG_SIZES = [500, 1000, 1500, 3000, 5000, 10000]
OUTPUT   = "pg_sweep_results.json"

def make_env(seed=SEED):
    return MarketEnvironment(seed=seed)

def evaluate(agent, env, n=TEST_EP):
    stats = []
    for i in range(n):
        s = run_episode(env, agent, training=False, ep_seed=99999 + i)
        stats.append(s)
    def m(k): return round(float(np.mean([s[k] for s in stats])), 4)
    def sd(k): return round(float(np.std([s[k] for s in stats])), 4)
    return {
        'reward':    m('total_reward'),   'reward_std':  sd('total_reward'),
        'cr':        m('completion_rate'), 'cr_std':     sd('completion_rate'),
        'stores':    m('stores_visited'),  'time_used':  m('time_used'),
        'premium':   m('avg_premium_pct'), 'pref':       m('avg_pref_score'),
        'expiry':    m('avg_expiry_days'),
    }

def make_agent(name, env):
    if name == 'PPO':
        return PPOAgent(env, seed=SEED+6, lr_actor=3e-4, lr_critic=1e-3,
                        gamma=0.95, lam=0.95, clip_eps=0.2,
                        n_epochs=4, rollout_steps=256, entropy_coef=0.01,
                        h1=64, h2=32)
    elif name == 'REINFORCE':
        return REINFORCEAgent(env, seed=SEED+7, lr=1e-3, gamma=0.95, h1=64, h2=32)
    elif name == 'A2C':
        return A2CAgent(env, seed=SEED+8, lr_actor=3e-4, lr_critic=1e-3,
                        gamma=0.95, entropy_coef=0.01, h1=64, h2=32)

results = {}

print(f"\n{'='*65}")
print(f"TRAINING-SIZE SWEEP — Policy Gradient Agents")
print(f"Sizes: {PG_SIZES}")
print(f"{'='*65}")

for name in ['PPO', 'REINFORCE', 'A2C']:
    print(f"\n  {name}")
    env = make_env(seed=SEED + hash(name) % 97)
    agent = make_agent(name, env)
    results[name] = {}
    trained = 0

    for N in PG_SIZES:
        t0 = time.time()
        for ep in range(N - trained):
            run_episode(env, agent, training=True)
            if hasattr(agent, 'post_episode'):
                agent.post_episode()
        trained = N
        m = evaluate(agent, env)
        m['train_episodes'] = N
        results[name][str(N)] = m
        print(f"    {name:<14} N={N:>6,}  "
              f"R={m['reward']:>7.1f}  CR={m['cr']:.1%}  "
              f"Stores={m['stores']:.1f}  Time={m['time_used']:.0f}min  "
              f"[{time.time()-t0:.0f}s]", flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved → {out}")
print("Now run: python build_final_dashboard.py")
