# ══════════════════════════════════════════════════════
# CHANGE 13 — NEW FILE: code/ablation_study.py
# Add reward component ablation experiment
# WHY: Required by the professor's original brief
# ══════════════════════════════════════════════════════
 
# CREATE a new file: code/ablation_study.py
# Copy-paste the content below into it:
 
"""
Reward Component Ablation Study
================================
Tests the effect of removing each reward component one at a time.
 
Original reward:
  R = W_PREMIUM * premium  +  W_PREF * pref  +  W_EXPIRY * expiry
    + W_MISSING * missing   +  W_STORE * store_visit
 
Ablation conditions:
  - Full:         all components active (baseline)
  - No Premium:   W_PREMIUM = 0
  - No Pref:      W_PREF_SCORE = 0
  - No Expiry:    W_EXPIRY = 0
  - No StoreP:    W_STORE_VISIT = 0   (remove store visit penalty)
 
Run: python ablation_study.py
"""
 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
from environment import MarketEnvironment
from dqn_agent import DQNAgent
from heuristic_agents import HeuristicAgent, TimeAwareAgent
from trainer import run_episode
 
SEED       = 42
TRAIN_EP   = 1500
TEST_EP    = 200
 
ABLATIONS = {
    'Full (baseline)': {},
    'No premium':      {'W_PREMIUM': 0.0},
    'No preference':   {'W_PREF_SCORE': 0.0},
    'No expiry':       {'W_EXPIRY': 0.0},
    'No store penalty':{'W_STORE_VISIT': 0.0},
}
 
 
def make_env_ablated(overrides: dict, seed=SEED):
    env = MarketEnvironment(seed=seed)
    for attr, val in overrides.items():
        setattr(env, attr, val)
    env._generate_static_params()
    return env
 
 
def eval_agent(agent, env, n=TEST_EP):
    stats = []
    for i in range(n):
        s = run_episode(env, agent, training=False, ep_seed=99999+i)
        stats.append(s)
    def m(k): return round(float(np.mean([x[k] for x in stats])), 4)
    def ci(k): return round(float(1.96 * np.std([x[k] for x in stats]) / np.sqrt(n)), 4)
    return {'reward': m('total_reward'), 'cr': m('completion_rate'),
            'stores': m('stores_visited'), 'time': m('time_used'),
            'reward_ci': ci('total_reward'), 'cr_ci': ci('completion_rate')}
 
 
def run_ablation():
    results = {}
 
    for condition, overrides in ABLATIONS.items():
        print(f"\n  Condition: {condition}")
        env = make_env_ablated(overrides, seed=SEED)
 
        # Train DQN under ablated reward
        agent = DQNAgent(env, seed=SEED, lr=3e-4, gamma=0.99,
                         eps_start=1.0, eps_end=0.05,
                         eps_decay=(0.05/1.0)**(1/TRAIN_EP),
                         batch_size=64, target_update_freq=100, h1=64, h2=32)
        for ep in range(TRAIN_EP):
            run_episode(env, agent, training=True)
 
        r = eval_agent(agent, env)
        results[condition] = r
        print(f"    Reward={r['reward']:.1f} ±{r['reward_ci']:.1f}  "
              f"CR={r['cr']:.1%} ±{r['cr_ci']:.1%}  "
              f"Stores={r['stores']:.1f}  Time={r['time']:.0f}min")
 
    # Save
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved ablation_results.json")
 
    # Plot
    conditions = list(results.keys())
    rewards    = [results[c]['reward'] for c in conditions]
    cis        = [results[c]['reward_ci'] for c in conditions]
    crs        = [results[c]['cr'] * 100 for c in conditions]
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Reward Component Ablation Study — DQN Agent',
                 fontsize=13, fontweight='bold')
 
    colors = ['#2ecc71' if c == 'Full (baseline)' else '#e74c3c' for c in conditions]
 
    axes[0].barh(conditions, rewards, xerr=cis, color=colors,
                 edgecolor='white', capsize=4)
    axes[0].set_title('Mean Reward (95% CI)', fontweight='bold')
    axes[0].axvline(rewards[0], color='green', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].set_xlabel('Mean Reward')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
 
    axes[1].barh(conditions, crs, color=colors, edgecolor='white')
    axes[1].set_title('Completion Rate (%)', fontweight='bold')
    axes[1].set_xlabel('Completion Rate (%)')
    axes[1].grid(axis='x', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('../figures/fig_ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/fig_ablation_study.png")
 
    return results
 
 
if __name__ == '__main__':
    run_ablation()
 