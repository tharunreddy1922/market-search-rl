# ══════════════════════════════════════════════════════
# CHANGE 14 — NEW FILE: code/significance_table.py
# Standalone script to print the full significance table
# WHY: Can be run separately and used in dissertation
# ══════════════════════════════════════════════════════
 
# CREATE a new file: code/significance_table.py
 
"""
Statistical Significance Table
================================
Loads full_experiment_results.json, re-evaluates agents,
runs pairwise Welch's t-tests, and prints/saves a full table.
 
Run: python significance_table.py
"""
 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import numpy as np
import json
from scipy import stats as scipy_stats
 
from environment import MarketEnvironment
from qlearning_agent import QLearningAgent
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent import PPOAgent
from reinforce_agent import REINFORCEAgent
from a2c_agent import A2CAgent
from value_iteration_agent import ValueIterationAgent
from heuristic_agents import HeuristicAgent, TimeAwareAgent, FreshnessFirstAgent, RandomAgent
from trainer import run_episode
 
SEED    = 42
TEST_EP = 300  # more episodes = tighter p-values
TRAIN   = 1500
 
 
def collect_rewards(agent, env, n=TEST_EP):
    return [run_episode(env, agent, training=False, ep_seed=99999+i)['total_reward']
            for i in range(n)]
 
 
def welch_test(a, b):
    t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)
 
 
def run_significance_table():
    env = MarketEnvironment(seed=SEED)
 
    agents = {
        'DQN':       DQNAgent(env, seed=SEED+3, lr=3e-4, gamma=0.99,
                              eps_start=1.0, eps_end=0.05,
                              eps_decay=(0.05/1.0)**(1/TRAIN),
                              batch_size=64, target_update_freq=100, h1=64, h2=32),
        'DoubleDQN': DoubleDQNAgent(env, seed=SEED+4, lr=3e-4, gamma=0.99,
                                    eps_start=1.0, eps_end=0.05,
                                    eps_decay=(0.05/1.0)**(1/TRAIN),
                                    batch_size=64, target_update_freq=100, h1=64, h2=32),
        'PPO':       PPOAgent(env, seed=SEED+6, lr_actor=3e-4, lr_critic=1e-3,
                              gamma=0.99, lam=0.97, clip_eps=0.15, n_epochs=6,
                              rollout_steps=512, entropy_coef=0.05, h1=64, h2=32),
        'REINFORCE': REINFORCEAgent(env, seed=SEED+7, lr=3e-4, gamma=0.99, h1=64, h2=32),
        'A2C':       A2CAgent(env, seed=SEED+8, lr_actor=3e-4, lr_critic=1e-3,
                              gamma=0.99, entropy_coef=0.05, h1=64, h2=32),
        'ValueIter': ValueIterationAgent(env, seed=SEED+9, gamma=0.99, n_iter=200),
        'TimeAware': TimeAwareAgent(env, seed=SEED+11),
        'Random':    RandomAgent(env, seed=SEED+13),
    }
 
    print("Training agents...")
    for name, agent in agents.items():
        if name == 'ValueIter':
            agent.train()
        elif name not in ('TimeAware', 'Random'):
            for ep in range(TRAIN):
                run_episode(env, agent, training=True)
                if hasattr(agent, 'post_episode'): agent.post_episode()
        print(f"  {name} ready")
 
    print("\nCollecting test rewards...")
    rewards = {}
    means   = {}
    for name, agent in agents.items():
        r = collect_rewards(agent, env)
        rewards[name] = r
        means[name]   = np.mean(r)
        print(f"  {name:<14} mean={means[name]:>7.1f}  std={np.std(r):>6.1f}")
 
    # Pairwise t-tests
    agent_names = list(agents.keys())
    print("\n=== WELCH'S T-TEST PAIRWISE SIGNIFICANCE TABLE ===")
    print(f"{'Pair':<30} {'t-stat':>8} {'p-value':>10} {'Sig?':>6} {'Better':>14}")
    print("-" * 72)
 
    table = {}
    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):
            if j <= i: continue
            t, p = welch_test(rewards[a], rewards[b])
            sig  = p < 0.05
            better = a if means[a] > means[b] else b
            key  = f"{a} vs {b}"
            table[key] = {'t': round(t,4), 'p': round(p,6),
                          'significant': sig, 'better': better}
            marker = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {key:<28} {t:>8.3f} {p:>10.4f} {marker:>6} {better:>14}")
 
    with open('significance_table.json', 'w') as f:
        json.dump(table, f, indent=2)
    print("\nSaved significance_table.json")
    return table
 
 
if __name__ == '__main__':
    run_significance_table()