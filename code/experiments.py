"""
============================================================
Market Search and Purchase Scheduling Using Reinforcement Learning
A00051705 — MSc Data Science, University of Roehampton

Unified Experiment Runner
============================================================

This file trains and evaluates all agents and runs every experiment
reported in the dissertation:

  SECTION 1  Configuration and hyperparameters
  SECTION 2  Environment and agent factories
  SECTION 3  Training and evaluation utilities
  SECTION 4  Statistical significance testing (Welch's t-test)
  SECTION 5  Baseline experiment
  SECTION 6  Oracle benchmark and optimality gap analysis
  SECTION 7  Policy function analysis (store-visit patterns)
  SECTION 8  Environment parameter documentation
  SECTION 9  Parametric experiments (duration / availability / goods)
  SECTION 10 Graph structure experiments
  SECTION 11 Brand configuration experiments
  SECTION 12 Heuristic policy comparison
  SECTION 13 Training-size convergence sweep
  SECTION 14 Figure generation
  SECTION 15 Interactive dashboard
  SECTION 16 Entry point

Run:
    python experiments.py                  # full run (all experiments)
    python experiments.py --no-sweep       # skip convergence sweep
    python experiments.py --quick          # N=100 episodes, fast smoke test
============================================================
"""

import sys
import os
import json
import time
import argparse
from collections import Counter

import numpy as np
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Agent imports ────────────────────────────────────────────────────────────
from environment       import MarketEnvironment
from agents            import BaseAgent, encode_action, decode_action, get_valid_action_mask
from trainer           import run_episode
from oracle_agent      import OracleAgent
from qlearning_agent   import QLearningAgent
from sarsa_agent       import SARSAAgent
from expected_sarsa_agent import ExpectedSARSAAgent
from dqn_agent         import DQNAgent
from double_dqn_agent  import DoubleDQNAgent
from dueling_dqn_agent import DuelingDQNAgent
from ppo_agent         import PPOAgent
from reinforce_agent   import REINFORCEAgent
from a2c_agent         import A2CAgent
from value_iteration_agent import ValueIterationAgent
from heuristic_agents  import (HeuristicAgent, TimeAwareAgent,
                                FreshnessFirstAgent, RandomAgent)

# ── NumPy-safe JSON encoder ───────────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars and arrays to native Python types for JSON."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)




# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

SEED             = 42
TRAIN_TABULAR    = 3_000
TRAIN_DEEP       = 800
TRAIN_PG         = 800
TEST_EPISODES    = 1_000      # all main evaluations use 1,000 episodes
EP_SEED_BASE     = 99_000     # fixed seeds: 99000 … 99999

TABULAR_SWEEP_SIZES = [3_000, 10_000, 50_000, 100_000, 200_000, 300_000]
DEEP_SWEEP_SIZES    = [200, 500, 800, 1_500, 3_000, 5_000]

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(RESULTS_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

AGENT_COLORS = {
    'Q-Learning':         '#e74c3c',
    'SARSA':              '#e67e22',
    'ExpSARSA':           '#f39c12',
    'DQN':                '#2ecc71',
    'DoubleDQN':          '#27ae60',
    'DuelingDQN':         '#1abc9c',
    'PPO':                '#3498db',
    'REINFORCE':          '#9b59b6',
    'A2C':                '#8e44ad',
    'ValueIter':          '#e91e63',
    'Greedy':             '#95a5a6',
    'TimeAware':          '#7f8c8d',
    'Freshness':          '#bdc3c7',
    'BrandQuality':       '#16a085',
    'Random':             '#34495e',
    'Oracle':             '#f1c40f',
}

AGENT_GROUPS = {
    'Tabular RL':      ['Q-Learning', 'SARSA', 'ExpSARSA'],
    'Deep RL':         ['DQN', 'DoubleDQN', 'DuelingDQN'],
    'Policy Gradient': ['PPO', 'REINFORCE', 'A2C'],
    'Model-Based':     ['ValueIter'],
    'Heuristic':       ['Greedy', 'TimeAware', 'Freshness', 'BrandQuality'],
    'Baseline':        ['Random'],
}

# Global store for training curves (filled during training)
_TRAINING_CURVES = {}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ENVIRONMENT AND AGENT FACTORIES
# ════════════════════════════════════════════════════════════════════════════

def make_env(duration=120, avail_prob=0.60, n_goods=20,
             n_brands=5, brands_per_store=3, seed=SEED):
    """Create a MarketEnvironment with the given parameters."""
    env = MarketEnvironment(seed=seed)
    env.TOTAL_DURATION   = duration
    env.AVAIL_PROB        = avail_prob
    env.N_GOODS           = n_goods
    env.N_BRANDS          = n_brands
    env.BRANDS_PER_STORE  = brands_per_store
    env._generate_static_params()
    return env


def eps_decay_for_budget(n_episodes, eps_start=1.0, eps_end=0.02):
    """Compute epsilon decay so epsilon reaches eps_end after n_episodes steps."""
    return float((eps_end / eps_start) ** (1.0 / max(n_episodes, 1)))


def make_agents(env, seed=SEED,
                train_tabular=TRAIN_TABULAR,
                train_deep=TRAIN_DEEP,
                train_pg=TRAIN_PG):
    """
    Return list of (name, agent, n_train_episodes) tuples.
    Agents with n_train=0 require no training (heuristics, oracle, model-based).
    """
    tab_decay  = eps_decay_for_budget(train_tabular, eps_end=0.02)
    deep_decay = eps_decay_for_budget(train_deep,    eps_end=0.05)

    return [
        # ── Tabular RL ──────────────────────────────────────────────────────
        ('Q-Learning', QLearningAgent(
            env, seed=seed,
            lr=0.3, gamma=0.99,
            eps_start=1.0, eps_end=0.02, eps_decay=tab_decay,
        ), train_tabular),

        ('SARSA', SARSAAgent(
            env, seed=seed+1,
            lr=0.3, gamma=0.99,
            eps_start=1.0, eps_end=0.02, eps_decay=tab_decay,
        ), train_tabular),

        ('ExpSARSA', ExpectedSARSAAgent(
            env, seed=seed+2,
            lr=0.3, gamma=0.99,
            eps_start=1.0, eps_end=0.02, eps_decay=tab_decay,
        ), train_tabular),

        # ── Deep RL ─────────────────────────────────────────────────────────
        ('DQN', DQNAgent(
            env, seed=seed+3,
            lr=3e-4, gamma=0.99,
            eps_start=1.0, eps_end=0.05, eps_decay=deep_decay,
            batch_size=64, target_update_freq=100, h1=64, h2=32,
        ), train_deep),

        ('DoubleDQN', DoubleDQNAgent(
            env, seed=seed+4,
            lr=3e-4, gamma=0.99,
            eps_start=1.0, eps_end=0.05, eps_decay=deep_decay,
            batch_size=64, target_update_freq=100, h1=64, h2=32,
        ), train_deep),

        ('DuelingDQN', DuelingDQNAgent(
            env, seed=seed+5,
            lr=3e-4, gamma=0.99,
            eps_start=1.0, eps_end=0.05, eps_decay=deep_decay,
            batch_size=64, target_update_freq=100, h1=64, h2=32,
        ), train_deep),

        # ── Policy Gradient ─────────────────────────────────────────────────
        ('PPO', PPOAgent(
            env, seed=seed+6,
            lr_actor=3e-4, lr_critic=1e-3, gamma=0.99,
            lam=0.97, clip_eps=0.15, n_epochs=4,
            rollout_steps=256,
        ), train_pg),

        ('REINFORCE', REINFORCEAgent(
            env, seed=seed+7,
            lr=3e-4, gamma=0.99, h1=64, h2=32,
        ), train_pg),

        ('A2C', A2CAgent(
            env, seed=seed+8,
            lr_actor=3e-4, lr_critic=1e-3, gamma=0.99,
            entropy_coef=0.05, h1=64, h2=32,
        ), train_pg),

        # ── Model-Based ─────────────────────────────────────────────────────
        ('ValueIter', ValueIterationAgent(
            env, seed=seed+9, gamma=0.99, n_iter=200,
        ), 0),

        # ── Heuristics ──────────────────────────────────────────────────────
        ('Greedy',       HeuristicAgent(env,         seed=seed+10), 0),
        ('TimeAware',    TimeAwareAgent(env,          seed=seed+11), 0),
        ('Freshness',    FreshnessFirstAgent(env,     seed=seed+12), 0),
        ('BrandQuality', BrandQualityHeuristic(env,   seed=seed+13), 0),

        # ── Baseline ────────────────────────────────────────────────────────
        ('Random',       RandomAgent(env,             seed=seed+14), 0),
    ]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRAINING AND EVALUATION UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def train_agent(agent, env, n_episodes, name):
    """Train agent for n_episodes. Returns per-episode reward list."""
    if name == 'ValueIter':
        agent.train()
        return []
    rewards = []
    for ep in range(n_episodes):
        stats = run_episode(env, agent, training=True)
        if hasattr(agent, 'post_episode'):
            agent.post_episode()
        rewards.append(stats['total_reward'])
    return rewards


def evaluate_agent(agent, env, n=TEST_EPISODES, seed_base=EP_SEED_BASE,
                   record_routes=False):
    """
    Evaluate agent on n fixed episodes.
    Returns summary dict; if record_routes=True also records store sequences.
    """
    rewards, crs, stores_visited_counts = [], [], []
    time_used_list, premium_list, pref_list, expiry_list = [], [], [], []
    first_stores, second_stores, route_patterns = [], [], []

    for i in range(n):
        if hasattr(agent, 'reset_plan'):
            agent.reset_plan()

        state     = env.reset(rng_seed=seed_base + i)
        ep_reward = 0.0
        ep_route  = [0]   # store 0 is always starting store

        while not env.done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            if record_routes and action['type'] == 'travel':
                ep_route.append(action['store'])
            state      = next_state
            ep_reward += reward

        ep_stats = env.compute_episode_stats()
        rewards.append(ep_reward)
        crs.append(ep_stats['completion_rate'])
        stores_visited_counts.append(ep_stats['stores_visited'])
        time_used_list.append(ep_stats['time_used'])
        premium_list.append(ep_stats['avg_premium_pct'])
        pref_list.append(ep_stats['avg_pref_score'])
        expiry_list.append(ep_stats['avg_expiry_days'])

        if record_routes:
            first_stores.append(ep_route[0] if ep_route else -1)
            second_stores.append(ep_route[1] if len(ep_route) > 1 else -1)
            route_patterns.append(tuple(ep_route))

    arr = np.array(rewards)
    n_ep = len(arr)

    def _ci(x): return float(1.96 * np.std(x) / np.sqrt(n_ep))
    def _m(x):  return float(np.mean(x))
    def _sd(x): return float(np.std(x))

    result = {
        'mean_reward':   round(_m(rewards), 4),
        'std_reward':    round(_sd(rewards), 4),
        'ci95_reward':   round(_ci(rewards), 4),
        'mean_cr':       round(_m(crs), 4),
        'std_cr':        round(_sd(crs), 4),
        'ci95_cr':       round(_ci(crs), 4),
        'mean_stores':   round(_m(stores_visited_counts), 4),
        'ci95_stores':   round(_ci(stores_visited_counts), 4),
        'mean_time':     round(_m(time_used_list), 4),
        'mean_premium':  round(_m(premium_list), 4),
        'mean_pref':     round(_m(pref_list), 4),
        'mean_expiry':   round(_m(expiry_list), 4),
        'success_rate':  round(float(np.mean(np.array(crs) == 1.0)), 4),
        '_raw_rewards':  rewards,
    }

    if record_routes:
        n_stores = env.N_STORES
        visit_freq   = [float(np.mean([s in r for r in route_patterns]))
                        for s in range(n_stores)]
        second_freq  = [float(np.mean(np.array(second_stores) == s))
                        for s in range(n_stores)]
        top_routes   = [(list(r), cnt)
                        for r, cnt in Counter(route_patterns).most_common(5)]
        result['route_info'] = {
            'visit_freq':        visit_freq,
            'second_store_freq': second_freq,
            'dominant_store':    int(np.argmax(visit_freq[1:]) + 1),
            'avg_route_length':  round(float(np.mean([len(r) for r in route_patterns])), 2),
            'top_5_routes':      top_routes,
        }

    return result


def run_config(env_kwargs, label='',
               train_tabular=TRAIN_TABULAR,
               train_deep=TRAIN_DEEP,
               train_pg=TRAIN_PG,
               collect_curves=False,
               record_routes=False):
    """
    Train and evaluate all agents on one environment configuration.
    Returns dict: {agent_name: metrics_dict}.
    """
    global _TRAINING_CURVES
    out = {}
    env = make_env(**env_kwargs)

    for name, agent, n_train in make_agents(
            env, seed=SEED,
            train_tabular=train_tabular,
            train_deep=train_deep,
            train_pg=train_pg):

        env2 = make_env(**env_kwargs, seed=SEED + hash(name) % 97)
        agent.env = env2
        t0 = time.time()

        curve = train_agent(agent, env2, n_train, name)
        if collect_curves and curve:
            _TRAINING_CURVES[name] = curve

        r = evaluate_agent(agent, env2,
                           record_routes=record_routes)
        out[name] = r

        print(f"    {name:<16} R={r['mean_reward']:>8.1f} ±{r['ci95_reward']:.1f}"
              f"  CR={r['mean_cr']:.1%} ±{r['ci95_cr']:.1%}"
              f"  Stores={r['mean_stores']:.1f}"
              f"  SR={r['success_rate']:.1%}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    return out


def _strip_raw(d):
    """Remove internal '_raw_*' keys before JSON serialisation."""
    if isinstance(d, dict):
        return {k: _strip_raw(v) for k, v in d.items()
                if not (isinstance(k, str) and k.startswith('_'))}
    if isinstance(d, list):
        return [_strip_raw(x) for x in d]
    return d


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STATISTICAL SIGNIFICANCE TESTING
# ════════════════════════════════════════════════════════════════════════════

def compute_significance_table(agent_raw_rewards: dict) -> dict:
    """
    Pairwise Welch's t-test (unequal variance) between all agent pairs.
    Returns dict: {
        'AgentA vs AgentB': {
            't_stat', 'p_value', 'significant' (p<0.05),
            'better', 'cohens_d'
        }
    }
    """
    agents  = list(agent_raw_rewards.keys())
    results = {}
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if j <= i:
                continue
            ra = np.array(agent_raw_rewards[a])
            rb = np.array(agent_raw_rewards[b])
            t_stat, p_value = scipy_stats.ttest_ind(ra, rb, equal_var=False)
            pooled_std = np.sqrt((ra.std()**2 + rb.std()**2) / 2)
            cohens_d   = abs(ra.mean() - rb.mean()) / (pooled_std + 1e-10)
            results[f'{a} vs {b}'] = {
                't_stat':      round(float(t_stat), 4),
                'p_value':     round(float(p_value), 6),
                'significant': bool(p_value < 0.05),
                'better':      a if ra.mean() > rb.mean() else b,
                'cohens_d':    round(float(cohens_d), 4),
            }
    return results


def print_significance_summary(sig_table: dict):
    sig_pairs = [(k, v) for k, v in sig_table.items() if v['significant']]
    total     = len(sig_table)
    print(f"\n  Significant differences (p<0.05): {len(sig_pairs)} of {total} pairs")
    print(f"  {'Pair':<30} {'p-value':>10}  {'t-stat':>8}  {'d':>6}  Better")
    print(f"  {'-'*70}")
    for k, v in sorted(sig_pairs, key=lambda x: x[1]['p_value'])[:15]:
        stars = '***' if v['p_value'] < 0.001 else ('**' if v['p_value'] < 0.01 else '*')
        print(f"  {k:<30} {v['p_value']:>10.4f}  {v['t_stat']:>8.3f}"
              f"  {v['cohens_d']:>6.3f}  {v['better']} {stars}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BASELINE EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def run_baseline():
    """
    Train and evaluate all agents under default environment settings.
    120 min | 20 goods | 60% availability | 5 brands / 3 per store | star graph.
    """
    print(f"\n{'═'*65}")
    print("BASELINE  —  120 min | 20 goods | 60% availability")
    print(f"{'═'*65}")

    env_kw  = {'duration': 120, 'avail_prob': 0.60, 'n_goods': 20}
    results = run_config(env_kw, collect_curves=True, record_routes=True)

    # Significance testing
    print("\n  Pairwise Welch's t-test...")
    raw = {n: r['_raw_rewards'] for n, r in results.items() if '_raw_rewards' in r}
    sig_table = compute_significance_table(raw)
    print_significance_summary(sig_table)

    sig_path = os.path.join(RESULTS_DIR, 'significance_table.json')
    with open(sig_path, 'w') as f:
        json.dump(sig_table, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Significance table saved → {sig_path}")

    return results, sig_table


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ORACLE BENCHMARK AND OPTIMALITY GAP ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

"""
Oracle algorithm (greedy perfect-information planner)
------------------------------------------------------
At episode start the oracle receives ground-truth availability, expiry, and
price arrays.  It scores every (store, good, brand) triple available:

    score(s, g, b) = W_PREF · pref(g,b)
                   + W_EXP  · expiry(s,g,b)
                   + W_PREM · 100 · premium(s,g,b)

Each good is assigned to its globally best (store, brand) pair.  Stores are
visited in assignment order subject to the time budget.

Optimality gap per episode:
    gap(ω) = 100 × (R*(ω) − R^π(ω)) / |R*(ω)|     if R*(ω) ≠ 0

Mean optimality gap: Δ̄ = (1/N) Σ_ω gap(ω)
"""

def run_oracle_benchmark(baseline_results: dict, env_kw: dict) -> dict:
    """
    Compare every agent against the oracle over TEST_EPISODES episodes.
    Computes optimality gap for each agent.
    """
    print(f"\n{'═'*65}")
    print("ORACLE BENCHMARK  —  Optimality Gap Analysis")
    print(f"  N = {TEST_EPISODES} episodes, seed base = {EP_SEED_BASE}")
    print(f"{'═'*65}")

    env = make_env(**env_kw)

    # Run oracle
    oracle = OracleAgent(env, seed=SEED)
    t0 = time.time()
    oracle_res = evaluate_agent(oracle, env, record_routes=False)
    oracle_rewards = np.array(oracle_res['_raw_rewards'])
    print(f"  {'Oracle':<18} R={oracle_res['mean_reward']:>8.2f}"
          f" ±{oracle_res['ci95_reward']:.2f}"
          f"  CR={oracle_res['mean_cr']:.1%}"
          f"  SR={oracle_res['success_rate']:.1%}  [{time.time()-t0:.0f}s]")

    agent_rows = {}
    for name, res in baseline_results.items():
        if '_raw_rewards' not in res:
            continue
        agent_arr = np.array(res['_raw_rewards'])
        gaps = np.where(
            oracle_rewards != 0,
            100.0 * (oracle_rewards - agent_arr) / np.abs(oracle_rewards),
            0.0
        )
        agent_rows[name] = {
            'mean_reward':        res['mean_reward'],
            'ci95_reward':        res['ci95_reward'],
            'mean_cr':            res['mean_cr'],
            'success_rate':       res['success_rate'],
            'optimality_gap_pct': round(float(gaps.mean()), 2),
            'gap_std_pct':        round(float(gaps.std()),  2),
        }
        print(f"  {name:<18} R={res['mean_reward']:>8.2f}"
              f" ±{res['ci95_reward']:.2f}"
              f"  CR={res['mean_cr']:.1%}"
              f"  Gap={gaps.mean():+.1f}% ±{gaps.std():.1f}%")

    return {
        'oracle': {
            'mean_reward':  oracle_res['mean_reward'],
            'ci95_reward':  oracle_res['ci95_reward'],
            'mean_cr':      oracle_res['mean_cr'],
            'success_rate': oracle_res['success_rate'],
        },
        'agents':      agent_rows,
        'n_episodes':  TEST_EPISODES,
        'seed_base':   EP_SEED_BASE,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — POLICY FUNCTION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def run_policy_analysis(baseline_results: dict, env_kw: dict) -> dict:
    """
    Analyse which stores each agent visits and why.
    Records first store, second store, visit frequencies, and route patterns
    from the baseline results (record_routes=True was set in run_baseline).
    """
    print(f"\n{'═'*65}")
    print("POLICY FUNCTION ANALYSIS  —  Store-Visit Patterns")
    print(f"  N = {TEST_EPISODES} episodes")
    print(f"{'═'*65}")

    env = make_env(**env_kw)

    # Store characteristics (fixed, explains WHY policies prefer certain stores)
    store_chars = []
    for s in range(env.N_STORES):
        stocked = env.store_stocks[s]
        best_prefs = [env.brand_preference[g, np.where(stocked[g])[0]].max()
                      for g in range(env.N_GOODS) if stocked[g].any()]
        avg_best_pref = float(np.mean(best_prefs)) if best_prefs else 0.0
        avg_premium   = float(np.mean(env.price_premium[s][stocked])) * 100
        store_chars.append({
            'store_id':              s,
            'avg_best_pref_score':   round(avg_best_pref, 3),
            'avg_price_premium_pct': round(avg_premium, 2),
            'brands_stocked_per_good': int(stocked.sum(axis=1).mean()),
            'travel_time_min':       env.TRAVEL_TIME,
        })
        print(f"  Store {s}:  avg_pref={avg_best_pref:.3f}"
              f"  avg_premium={avg_premium:+.2f}%")

    # Extract route info from baseline results
    agent_policy = {}
    for name, res in baseline_results.items():
        if 'route_info' not in res:
            continue
        ri = res['route_info']
        agent_policy[name] = {
            'dominant_store':    ri['dominant_store'],
            'avg_route_length':  ri['avg_route_length'],
            'visit_freq':        ri['visit_freq'],
            'second_store_freq': ri['second_store_freq'],
            'top_5_routes':      ri['top_5_routes'],
            'mean_reward':       res['mean_reward'],
            'mean_cr':           res['mean_cr'],
        }
        print(f"  {name:<18} dominant_store={ri['dominant_store']}"
              f"  avg_route_len={ri['avg_route_length']:.2f}")

    return {
        'agent_policies':      agent_policy,
        'store_characteristics': store_chars,
        'n_episodes':          TEST_EPISODES,
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ENVIRONMENT PARAMETER DOCUMENTATION
# ════════════════════════════════════════════════════════════════════════════

def document_environment_parameters(env_kw: dict) -> dict:
    """
    Return all fixed environment parameters as structured tables.
    Stochastic variables (expiry realisations, availability outcomes)
    are excluded — they vary per episode and are not fixed.
    """
    print(f"\n{'═'*65}")
    print("ENVIRONMENT PARAMETER DOCUMENTATION  —  Fixed Parameters")
    print(f"{'═'*65}")

    env = make_env(**env_kw)

    global_params = {
        'n_stores':            env.N_STORES,
        'n_goods':             env.N_GOODS,
        'n_brands_per_good':   env.N_BRANDS,
        'brands_per_store':    env.BRANDS_PER_STORE,
        'total_duration_min':  env.TOTAL_DURATION,
        'travel_time_min':     env.TRAVEL_TIME,
        'visit_time_min':      env.VISIT_TIME,
        'availability_prob':   env.AVAIL_PROB,
        'price_premium_range': '[-30%, +30%]  (uniform, fixed per store-good-brand)',
        'expiry_range_days':   '[0, 7]  (uniform integer, stochastic per episode)',
        'graph_structure':     'Star — all stores equidistant at 5 min travel',
        'reward_missing_item': env.W_MISSING_ITEM,
        'reward_premium':      env.W_PREMIUM,
        'reward_pref_score':   env.W_PREF_SCORE,
        'reward_expiry':       env.W_EXPIRY,
        'reward_store_visit':  env.W_STORE_VISIT,
    }

    store_table = []
    for s in range(env.N_STORES):
        stocked = env.store_stocks[s]
        best_prefs = [env.brand_preference[g, np.where(stocked[g])[0]].max()
                      for g in range(env.N_GOODS) if stocked[g].any()]
        avg_best_pref = float(np.mean(best_prefs)) if best_prefs else 0.0
        avg_premium   = float(np.mean(env.price_premium[s][stocked])) * 100
        store_table.append({
            'store_id':                s,
            'brands_stocked_per_good': int(stocked.sum(axis=1).mean()),
            'avg_best_pref_score':     round(avg_best_pref, 3),
            'avg_price_premium_pct':   round(avg_premium, 2),
            'travel_time_min':         env.TRAVEL_TIME,
            'graph_role':              'Start node' if s == 0 else 'Spoke',
        })
        print(f"  Store {s}:  avg_pref={avg_best_pref:.3f}"
              f"  avg_premium={avg_premium:+.2f}%"
              f"  brands/good={int(stocked.sum(axis=1).mean())}")

    return {
        'global_parameters':       global_params,
        'store_table':             store_table,
        'note': (
            'All parameters listed here are fixed at environment initialisation '
            '(seed=42). Stochastic variables — per-episode brand availability '
            'and expiry date realisations — are not listed as they vary each episode.'
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PARAMETRIC EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

def run_parametric_experiments():
    """
    Vary one environment parameter at a time:
      - Episode duration:      60 / 120 / 180 min
      - Brand availability:    40% / 60% / 80%
      - Number of goods:       10 / 20 / 40
    All other parameters held at baseline values.
    """
    param_grids = {
        'duration':   [60, 120, 180],
        'avail_prob': [0.40, 0.60, 0.80],
        'n_goods':    [10, 20, 40],
    }
    all_results = {}

    for param, values in param_grids.items():
        print(f"\n{'═'*65}")
        print(f"PARAMETRIC — {param.upper()}  values = {values}")
        print(f"{'═'*65}")
        all_results[param] = {}
        for v in values:
            kw = {'duration': 120, 'avail_prob': 0.60, 'n_goods': 20}
            kw[param] = v
            print(f"\n  {param} = {v}")
            raw = run_config(kw)
            all_results[param][str(v)] = _strip_raw(raw)

    return all_results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — GRAPH STRUCTURE EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

"""
Graph structure experiment design
----------------------------------
Baseline: all stores at 5 min travel (standard star graph).
Variant A: one distant store  — store 5 costs 20 min to travel to.
Variant B: two distant stores — stores 4 and 5 each cost 20 min.

All other parameters (goods, availability, rewards) are held constant.
The DistantStoreEnvironment subclass overrides travel costs per destination.
"""

class DistantStoreEnvironment(MarketEnvironment):
    """
    MarketEnvironment with configurable per-store travel times.
    Replaces the constant TRAVEL_TIME with per-store lookup.

    distant_stores: dict {store_id: travel_time_minutes}
    Stores not listed retain the default TRAVEL_TIME (5 min).
    """

    def __init__(self, distant_stores: dict, seed: int = SEED,
                 duration: int = 120, avail_prob: float = 0.60, n_goods: int = 20):
        self._distant = distant_stores
        super().__init__(seed=seed)
        self.TOTAL_DURATION = duration
        self.AVAIL_PROB     = avail_prob
        self.N_GOODS        = n_goods
        self._generate_static_params()

    def _travel_cost(self, dest: int) -> int:
        return self._distant.get(dest, self.TRAVEL_TIME)

    def get_valid_actions(self):
        if self.done:
            return []
        actions = [{'type': 'end'}]
        cur  = self.current_store
        info = self.revealed_info.get(cur, {})
        avail = info.get('avail', np.zeros((self.N_GOODS, self.N_BRANDS), dtype=bool))
        for g in self.items_needed:
            for b in range(self.N_BRANDS):
                if avail[g, b]:
                    actions.append({'type': 'buy', 'good': g, 'brand': b})
        for s in range(self.N_STORES):
            if s != cur:
                cost = self._travel_cost(s) + self.VISIT_TIME
                if self.time_remaining >= cost:
                    actions.append({'type': 'travel', 'store': s})
        return actions

    def step(self, action):
        assert not self.done
        reward = 0.0
        info   = {}
        if action['type'] == 'end':
            self.done = True
        elif action['type'] == 'buy':
            g, b  = action['good'], action['brand']
            cur   = self.current_store
            if g in self.items_needed and self.actual_avail[cur, g, b]:
                prem_pct = self.price_premium[cur, g, b] * 100
                expiry   = self.expiry_dates[cur, g, b]
                pref     = self.brand_preference[g, b]
                self.items_bought[g] = (b, prem_pct, expiry, pref)
                self.items_needed.discard(g)
                reward += self.W_PREMIUM * prem_pct
                reward += self.W_PREF_SCORE * pref
                reward += self.W_EXPIRY * expiry
            else:
                reward -= 1.0
        elif action['type'] == 'travel':
            s = action['store']
            self.time_remaining -= self._travel_cost(s)
            self.current_store   = s
            self._visit_store(s)
            reward += self.W_STORE_VISIT
        if self.time_remaining <= 0:
            self.done = True
        if self.done:
            missing = len(self.items_needed)
            reward += self.W_MISSING_ITEM * missing
            info['missing_items']  = missing
            info['items_bought']   = len(self.items_bought)
            info['stores_visited'] = len(self.visited_stores)
            info['time_used']      = self.TOTAL_DURATION - self.time_remaining
        return self._get_state(), reward, self.done, info


def _eval_on_env(env, agent, name) -> dict:
    """
    Evaluate a single agent on a given env without retraining.
    Skips agents whose action space doesn't match the env
    (e.g. tabular agents trained on 5 brands cannot run on 10 brands).
    Returns None if incompatible.
    """
    orig_env  = agent.env
    agent.env = env
    # Action-space compatibility check
    env_action_dim = env.get_action_dim()
    agent_action_dim = getattr(agent, 'action_dim', None)
    if agent_action_dim is not None and agent_action_dim != env_action_dim:
        agent.env = orig_env
        return None
    try:
        res = evaluate_agent(agent, env)
    except (IndexError, ValueError):
        agent.env = orig_env
        return None
    agent.env = orig_env
    return res


def run_graph_experiments(trained_agents: dict) -> dict:
    """
    REQ-4a: Three graph configurations.
    Evaluated agents: Oracle + all trained RL agents + key heuristics.
    """
    print(f"\n{'═'*65}")
    print("GRAPH STRUCTURE EXPERIMENTS")
    print(f"  N = {TEST_EPISODES} episodes per configuration")
    print(f"{'═'*65}")

    configs = {
        'baseline_star':      {},
        'one_distant_store':  {5: 20},
        'two_distant_stores': {4: 20, 5: 20},
    }

    results = {}
    for cfg_name, distant_cfg in configs.items():
        label = distant_cfg or 'all 5 min'
        print(f"\n  Configuration: {cfg_name}  (distant={label})")

        if distant_cfg:
            env = DistantStoreEnvironment(distant_stores=distant_cfg, seed=SEED)
        else:
            env = make_env()

        cfg_results = {}

        # Oracle
        oracle = OracleAgent(env, seed=SEED)
        t0 = time.time()
        r = evaluate_agent(oracle, env)
        cfg_results['Oracle'] = {
            'mean_reward': r['mean_reward'], 'ci95_reward': r['ci95_reward'],
            'mean_cr': r['mean_cr'],         'success_rate': r['success_rate'],
        }
        print(f"    {'Oracle':<18} R={r['mean_reward']:>8.2f}"
              f"  CR={r['mean_cr']:.1%}  SR={r['success_rate']:.1%}"
              f"  [{time.time()-t0:.0f}s]")

        # All trained agents
        for name, agent in trained_agents.items():
            t0 = time.time()
            r  = _eval_on_env(env, agent, name)
            if r is None:
                print(f"    {name:<18} skipped (action space mismatch)")
                cfg_results[name] = {
                    'mean_reward': None, 'ci95_reward': None,
                    'mean_cr': None, 'success_rate': None,
                    'note': 'action_space_mismatch',
                }
                continue
            cfg_results[name] = {
                'mean_reward': r['mean_reward'], 'ci95_reward': r['ci95_reward'],
                'mean_cr': r['mean_cr'],         'success_rate': r['success_rate'],
            }
            print(f"    {name:<18} R={r['mean_reward']:>8.2f}"
                  f"  CR={r['mean_cr']:.1%}  [{time.time()-t0:.0f}s]")

        results[cfg_name] = {
            'distant_stores': distant_cfg,
            'agents':         cfg_results,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — BRAND CONFIGURATION EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

def run_brand_experiments(trained_agents: dict) -> dict:
    """
    REQ-4b: Three brand configurations.
      baseline:  5 brands, 3 per store  (original)
      narrow:    5 brands, 2 per store  (fewer options per store)
      rich:     10 brands, 3 per store  (more brand diversity)
    """
    print(f"\n{'═'*65}")
    print("BRAND CONFIGURATION EXPERIMENTS")
    print(f"  N = {TEST_EPISODES} episodes per configuration")
    print(f"{'═'*65}")

    configs = {
        'baseline_5brands_3per': {'n_brands': 5, 'brands_per_store': 3},
        'narrow_5brands_2per':   {'n_brands': 5, 'brands_per_store': 2},
        'rich_10brands_3per':    {'n_brands': 10, 'brands_per_store': 3},
    }

    results = {}
    for cfg_name, brand_kw in configs.items():
        print(f"\n  Configuration: {cfg_name}  {brand_kw}")
        env = make_env(**brand_kw)

        cfg_results = {}

        oracle = OracleAgent(env, seed=SEED)
        t0 = time.time()
        r = evaluate_agent(oracle, env)
        cfg_results['Oracle'] = {
            'mean_reward': r['mean_reward'], 'ci95_reward': r['ci95_reward'],
            'mean_cr': r['mean_cr'],
        }
        print(f"    {'Oracle':<22} R={r['mean_reward']:>8.2f}"
              f"  CR={r['mean_cr']:.1%}  [{time.time()-t0:.0f}s]")

        for name, agent in trained_agents.items():
            t0 = time.time()
            r  = _eval_on_env(env, agent, name)
            if r is None:
                print(f"    {name:<22} skipped (action space mismatch: "
                      f"agent trained on {agent.env.get_action_dim()} actions, "
                      f"env has {env.get_action_dim()})")
                cfg_results[name] = {
                    'mean_reward': None, 'ci95_reward': None,
                    'mean_cr': None, 'note': 'action_space_mismatch',
                }
                continue
            cfg_results[name] = {
                'mean_reward': r['mean_reward'], 'ci95_reward': r['ci95_reward'],
                'mean_cr': r['mean_cr'],
            }
            print(f"    {name:<22} R={r['mean_reward']:>8.2f}"
                  f"  CR={r['mean_cr']:.1%}  [{time.time()-t0:.0f}s]")

        results[cfg_name] = {
            'n_brands':        brand_kw['n_brands'],
            'brands_per_store': brand_kw['brands_per_store'],
            'agents':          cfg_results,
        }

    return results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — HEURISTIC POLICY: BRAND QUALITY HEURISTIC
# ════════════════════════════════════════════════════════════════════════════

"""
Brand Quality Heuristic — formal specification
-----------------------------------------------
Buy rule (two conditions must both hold):
  (a) pref(g, b) > mean(brand_preference)          [above-average preference]
  (b) expiry(s, g, b) < mean_expiry_revealed        [below-average expiry]

  Score for qualifying (g, b): W_PREF · pref(g,b) − 0.3 · expiry(g,b)
  Select highest-scoring qualifying item.

Travel rule:
  Go to store s* = argmax_s Σ_{g ∈ needed} max_{b ∈ stock(s,g)} pref(g,b)
  (store with highest total expected brand preference for remaining goods)
  Add small bonus (+0.5) for unvisited stores to encourage coverage.

Rationale:
  Condition (a) ensures only brands in the top 40% of the preference
  distribution are purchased. Condition (b) ensures urgency-weighted buying
  (fresher items preferred). The travel rule routes to maximum brand value.
"""

class BrandQualityHeuristic(BaseAgent):
    """
    Heuristic agent implementing a dual quality-gate buy rule.

    Buy only when:
      - Brand preference is strictly above the global mean (~3.0).
      - Expiry date is strictly below the running average of revealed items.

    Travel to the store with the highest total expected brand preference
    for remaining goods.
    """

    def get_name(self) -> str:
        return 'BrandQuality'

    def select_action(self, state: dict, training: bool = True) -> dict:
        env      = self.env
        cur      = state['current_store']
        needed   = state['items_needed']
        time_rem = state['time_remaining']
        info     = state['revealed_info'].get(cur, {})
        avail    = info.get('avail',  np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        expiry   = info.get('expiry', np.zeros((env.N_GOODS, env.N_BRANDS)))
        pref     = env.brand_preference

        # Thresholds
        pref_thresh = float(pref.mean())   # global mean ≈ 3.0

        # Dynamic expiry threshold: mean of all available items revealed so far
        all_exp = []
        for s_info in state['revealed_info'].values():
            e = s_info.get('expiry', np.zeros((env.N_GOODS, env.N_BRANDS)))
            m = s_info.get('avail',  np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
            if m.any():
                all_exp.extend(e[m].tolist())
        expiry_thresh = float(np.mean(all_exp)) if all_exp else 3.5

        # Buy rule
        best_buy, best_score = None, -np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if (avail[g, b]
                        and pref[g, b] > pref_thresh
                        and expiry[g, b] < expiry_thresh):
                    score = env.W_PREF_SCORE * pref[g, b] - 0.3 * expiry[g, b]
                    if score > best_score:
                        best_score, best_buy = score, {'type': 'buy', 'good': g, 'brand': b}
        if best_buy:
            return best_buy

        # Travel rule
        tc = env.TRAVEL_TIME + env.VISIT_TIME
        best_store, best_util = None, -np.inf
        for s in range(env.N_STORES):
            if s == cur or time_rem < tc:
                continue
            utility = sum(
                pref[g, np.where(env.store_stocks[s, g])[0]].max()
                for g in needed if env.store_stocks[s, g].any()
            )
            if s not in state['visited_stores']:
                utility += 0.5
            if utility > best_util:
                best_util, best_store = utility, s
        if best_store is not None and best_util > 0:
            return {'type': 'travel', 'store': best_store}
        return {'type': 'end'}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — TRAINING-SIZE CONVERGENCE SWEEP
# ════════════════════════════════════════════════════════════════════════════

def run_training_size_sweep():
    """
    Vary training budget for tabular and deep RL agents.
    Shows how reward and completion rate improve with more training.
    """
    env_kw = {'duration': 120, 'avail_prob': 0.60, 'n_goods': 20}

    sweep = {
        'tabular': {},
        'deep':    {},
        'metadata': {
            'tabular_sizes': TABULAR_SWEEP_SIZES,
            'deep_sizes':    DEEP_SWEEP_SIZES,
            'test_episodes': TEST_EPISODES,
            'env':           env_kw,
        }
    }

    print(f"\n{'═'*65}")
    print("CONVERGENCE SWEEP — Tabular Agents")
    print(f"{'═'*65}")
    for N in TABULAR_SWEEP_SIZES:
        print(f"\n  N = {N:,} episodes")
        sweep['tabular'][str(N)] = {}
        decay = eps_decay_for_budget(N, eps_end=0.02)
        for off, name, Cls in [(0,'Q-Learning',QLearningAgent),
                               (1,'SARSA',SARSAAgent),
                               (2,'ExpSARSA',ExpectedSARSAAgent)]:
            env   = make_env(**env_kw, seed=SEED + hash(name) % 97)
            agent = Cls(env, seed=SEED+off, lr=0.3, gamma=0.99,
                        eps_start=1.0, eps_end=0.02, eps_decay=decay)
            t0 = time.time()
            train_agent(agent, env, N, name)
            r  = evaluate_agent(agent, env)
            r.pop('_raw_rewards', None)
            sweep['tabular'][str(N)][name] = r
            print(f"    {name:<14} N={N:>7,}  R={r['mean_reward']:>7.1f}"
                  f" ±{r['ci95_reward']:.1f}  CR={r['mean_cr']:.1%}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'═'*65}")
    print("CONVERGENCE SWEEP — Deep RL Agents")
    print(f"{'═'*65}")
    for N in DEEP_SWEEP_SIZES:
        print(f"\n  N = {N:,} episodes")
        sweep['deep'][str(N)] = {}
        decay = eps_decay_for_budget(N, eps_end=0.05)
        for i, (name, Cls) in enumerate([('DQN',DQNAgent),
                                          ('DoubleDQN',DoubleDQNAgent),
                                          ('DuelingDQN',DuelingDQNAgent)]):
            env   = make_env(**env_kw, seed=SEED + hash(name) % 97)
            agent = Cls(env, seed=SEED+3+i, lr=3e-4, gamma=0.99,
                        eps_start=1.0, eps_end=0.05, eps_decay=decay,
                        batch_size=64, target_update_freq=100, h1=64, h2=32)
            t0 = time.time()
            train_agent(agent, env, N, name)
            r  = evaluate_agent(agent, env)
            r.pop('_raw_rewards', None)
            sweep['deep'][str(N)][name] = r
            print(f"    {name:<14} N={N:>7,}  R={r['mean_reward']:>7.1f}"
                  f" ±{r['ci95_reward']:.1f}  CR={r['mean_cr']:.1%}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)

    return sweep


# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — FIGURE GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_figures(results: dict, sweep: dict = None):
    agents_ordered = [
        'Q-Learning','SARSA','ExpSARSA',
        'DQN','DoubleDQN','DuelingDQN',
        'PPO','REINFORCE','A2C',
        'ValueIter',
        'Greedy','TimeAware','Freshness','BrandQuality',
        'Random',
    ]
    baseline = results.get('baseline', {})

    # ── Fig 1: Baseline comparison ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Algorithm Comparison — Baseline  (120 min | 20 goods | 60% availability)',
                 fontsize=13, fontweight='bold')
    for ax, (key, title, ylabel) in zip(axes, [
        ('mean_reward', 'Mean Reward ± 95% CI', 'Reward'),
        ('mean_cr',     'Completion Rate',       'Completion Rate'),
        ('mean_stores', 'Stores Visited',        'Stores Visited'),
    ]):
        names  = [a for a in agents_ordered if a in baseline]
        vals   = [baseline[a][key] for a in names]
        colors = [AGENT_COLORS.get(a, '#aaa') for a in names]
        errs   = [baseline[a].get('ci95_reward', 0) for a in names] if key == 'mean_reward' else None
        ax.bar(range(len(names)), vals, color=colors, edgecolor='white',
               linewidth=0.8, yerr=errs, capsize=3,
               error_kw={'elinewidth': 1.2, 'ecolor': '#333'})
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        if key == 'mean_cr':
            ax.set_ylim(0, 1.08)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_baseline_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ fig_baseline_comparison.png")

    # ── Fig 2: Oracle optimality gap ────────────────────────────────────────
    oracle_data = results.get('oracle_benchmark', {})
    if oracle_data:
        agent_rows = oracle_data.get('agents', {})
        names  = [a for a in agents_ordered if a in agent_rows]
        gaps   = [agent_rows[a]['optimality_gap_pct'] for a in names]
        colors = [AGENT_COLORS.get(a, '#aaa') for a in names]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(names, gaps, color=colors, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title('Optimality Gap vs Oracle (%)  — Baseline',
                     fontweight='bold', fontsize=13)
        ax.set_ylabel('Mean Optimality Gap (%)')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_optimality_gap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ fig_optimality_gap.png")

    # ── Fig 3: Graph structure experiments ──────────────────────────────────
    graph_data = results.get('graph_experiments', {})
    if graph_data:
        cfg_names = list(graph_data.keys())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Graph Structure Experiments', fontsize=13, fontweight='bold')
        all_agents_g = [a for a in agents_ordered
                        if any(a in graph_data[c]['agents'] for c in cfg_names)]
        all_agents_g = ['Oracle'] + [a for a in all_agents_g if a != 'Oracle']
        for ax, metric, ylabel in zip(axes,
                                       ['mean_reward', 'mean_cr'],
                                       ['Mean Reward', 'Completion Rate']):
            for aname in all_agents_g:
                vals = [graph_data[c]['agents'].get(aname, {}).get(metric)
                        for c in cfg_names]
                valid = [(i, v) for i, v in enumerate(vals) if v is not None]
                if not valid: continue
                xi, yi = zip(*valid)
                ax.plot(list(xi), list(yi), marker='o', label=aname,
                        color=AGENT_COLORS.get(aname, '#aaa'), linewidth=2)
            ax.set_xticks(range(len(cfg_names)))
            ax.set_xticklabels(cfg_names, rotation=20, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            if metric == 'mean_cr':
                ax.set_ylim(0, 1.08)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_graph_experiments.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ fig_graph_experiments.png")

    # ── Fig 4: Brand configuration experiments ───────────────────────────────
    brand_data = results.get('brand_experiments', {})
    if brand_data:
        cfg_names = list(brand_data.keys())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Brand Configuration Experiments', fontsize=13, fontweight='bold')
        all_agents_b = [a for a in agents_ordered
                        if any(a in brand_data[c]['agents'] for c in cfg_names)]
        all_agents_b = ['Oracle'] + [a for a in all_agents_b if a != 'Oracle']
        for ax, metric, ylabel in zip(axes,
                                       ['mean_reward', 'mean_cr'],
                                       ['Mean Reward', 'Completion Rate']):
            for aname in all_agents_b:
                vals = [brand_data[c]['agents'].get(aname, {}).get(metric)
                        for c in cfg_names]
                valid = [(i, v) for i, v in enumerate(vals) if v is not None]
                if not valid: continue
                xi, yi = zip(*valid)
                ax.plot(list(xi), list(yi), marker='s', label=aname,
                        color=AGENT_COLORS.get(aname, '#aaa'), linewidth=2)
            ax.set_xticks(range(len(cfg_names)))
            ax.set_xticklabels(cfg_names, rotation=20, ha='right')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            if metric == 'mean_cr':
                ax.set_ylim(0, 1.08)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_brand_experiments.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ fig_brand_experiments.png")

    # ── Fig 5: Convergence curves ────────────────────────────────────────────
    if _TRAINING_CURVES:
        WINDOW = 50
        groups = {
            'Tabular RL':      ['Q-Learning', 'SARSA', 'ExpSARSA'],
            'Deep RL':         ['DQN', 'DoubleDQN', 'DuelingDQN'],
            'Policy Gradient': ['PPO', 'REINFORCE', 'A2C'],
        }
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Training Convergence Curves (50-episode rolling average)',
                     fontsize=13, fontweight='bold')
        for ax, (gname, gagents) in zip(axes, groups.items()):
            for aname in gagents:
                if aname not in _TRAINING_CURVES:
                    continue
                raw = np.array(_TRAINING_CURVES[aname])
                smoothed = (np.convolve(raw, np.ones(WINDOW)/WINDOW, mode='valid')
                            if len(raw) >= WINDOW else raw)
                ax.plot(np.arange(len(smoothed)), smoothed, label=aname,
                        color=AGENT_COLORS.get(aname, '#aaa'), linewidth=2)
            ax.set_title(gname, fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward (50-ep rolling avg)')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig_convergence_curves.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ fig_convergence_curves.png")

    # ── Fig 6: Parametric results ────────────────────────────────────────────
    param_data = results.get('parametric', {})
    if param_data:
        param_info = [
            ('duration',   'Duration (min)'),
            ('avail_prob', 'Availability Probability'),
            ('n_goods',    'Number of Goods'),
        ]
        for metric, mlabel in [('mean_reward','Mean Reward'), ('mean_cr','Completion Rate')]:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'{mlabel} vs Parameter — All Algorithms',
                         fontsize=13, fontweight='bold')
            for ax, (param, xlabel) in zip(axes, param_info):
                exp_d = param_data.get(param, {})
                pvals = sorted(exp_d.keys(), key=float)
                for aname in agents_ordered:
                    y = [exp_d.get(pv, {}).get(aname, {}).get(metric)
                         for pv in pvals]
                    valid = [(i, v) for i, v in enumerate(y) if v is not None]
                    if not valid: continue
                    xi, yi = zip(*valid)
                    ax.plot(list(xi), list(yi), marker='o', label=aname,
                            color=AGENT_COLORS.get(aname, '#aaa'),
                            linewidth=2, markersize=5)
                ax.set_xticks(range(len(pvals)))
                ax.set_xticklabels(pvals)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(mlabel)
                ax.set_title(f'{mlabel} vs {xlabel}', fontweight='bold')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=6, ncol=2)
                if metric == 'mean_cr':
                    ax.set_ylim(0, 1.08)
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            fname = f"fig_parametric_{'reward' if metric=='mean_reward' else 'cr'}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {fname}")

    print(f"\n✓ All figures saved → {FIGURES_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 15 — INTERACTIVE HTML DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

def generate_dashboard(results: dict, sweep: dict = None):
    agents_ordered = [
        'Q-Learning','SARSA','ExpSARSA',
        'DQN','DoubleDQN','DuelingDQN',
        'PPO','REINFORCE','A2C',
        'ValueIter',
        'Greedy','TimeAware','Freshness','BrandQuality',
        'Random',
    ]

    n_test = TEST_EPISODES
    sweep_json = json.dumps(sweep or {})
    baseline   = results.get('baseline', {})
    sig_table  = results.get('significance_table', {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Market Search RL — Experiment Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f1117;color:#e8eaf6;font-family:'Segoe UI',sans-serif;font-size:14px}}
  h1{{padding:22px 32px 6px;font-size:20px;color:#7c83fd}}
  .sub{{padding:0 32px 18px;color:#9fa8da;font-size:12px}}
  .tabs{{display:flex;gap:4px;padding:0 32px;border-bottom:1px solid #1e2030;flex-wrap:wrap}}
  .tab{{padding:9px 18px;cursor:pointer;border-radius:8px 8px 0 0;color:#9fa8da;
        background:#1a1d2e;font-size:12px;transition:all .2s}}
  .tab.active,.tab:hover{{background:#7c83fd;color:#fff}}
  .panel{{display:none;padding:24px 32px}}
  .panel.active{{display:block}}
  .kpis{{display:flex;gap:14px;margin-bottom:24px;flex-wrap:wrap}}
  .kpi{{background:#1a1d2e;border-radius:10px;padding:14px 20px;flex:1;min-width:140px}}
  .kv{{font-size:24px;font-weight:700;color:#7c83fd}}
  .kl{{font-size:11px;color:#9fa8da;margin-top:3px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin-bottom:20px}}
  .box{{background:#1a1d2e;border-radius:10px;padding:18px}}
  .box h3{{font-size:12px;color:#9fa8da;margin-bottom:12px}}
  canvas{{max-height:260px}}
  table{{width:100%;border-collapse:collapse;font-size:12px;background:#1a1d2e;border-radius:8px;overflow:hidden}}
  th{{background:#13152b;color:#9fa8da;padding:9px 11px;text-align:left;font-weight:600}}
  td{{padding:8px 11px;border-bottom:1px solid #1e2030}}
  tr:hover td{{background:#1f2340}}
  .g{{color:#52c77a;font-weight:600}} .a{{color:#f0a500;font-weight:600}} .r{{color:#f05050;font-weight:600}}
  .r1{{color:#ffd700;font-weight:700}} .r2{{color:#c0c0c0;font-weight:700}} .r3{{color:#cd7f32;font-weight:700}}
  .sig{{font-size:11px;padding:1px 5px;border-radius:4px;background:#1b3a1b;color:#52c77a;margin-left:4px}}
  .ns{{font-size:11px;padding:1px 5px;border-radius:4px;background:#2a1010;color:#f05050;margin-left:4px}}
</style>
</head>
<body>
<h1>Market Search and Purchase Scheduling — RL Experiment Dashboard</h1>
<p class="sub">MSc Data Science | University of Roehampton | A00051705<br>
N={n_test} test episodes (fixed seeds) | 95% CI on all metrics | Welch's t-test significance</p>
<div class="tabs">
  <div class="tab active" onclick="show('overview','Overview')">Overview</div>
  <div class="tab" onclick="show('baseline','Baseline')">Baseline + CI</div>
  <div class="tab" onclick="show('oracle','Oracle')">Oracle Gap</div>
  <div class="tab" onclick="show('policy','Policy')">Policy Analysis</div>
  <div class="tab" onclick="show('graph','Graph')">Graph Experiments</div>
  <div class="tab" onclick="show('brand','Brand')">Brand Experiments</div>
  <div class="tab" onclick="show('param','Parametric')">Parametric</div>
  <div class="tab" onclick="show('sig','Significance')">Significance</div>
  <div class="tab" onclick="show('conv','Convergence')">Convergence</div>
  <div class="tab" onclick="show('rank','Rankings')">Rankings</div>
</div>

<div id="tab-overview" class="panel active">
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="box"><h3>Mean Reward — all agents</h3><canvas id="c-ov-r"></canvas></div>
    <div class="box"><h3>Completion Rate — all agents</h3><canvas id="c-ov-cr"></canvas></div>
  </div>
</div>

<div id="tab-baseline" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    95% confidence intervals: 1.96 × σ / √n, n={n_test} episodes.
  </p>
  <div class="grid">
    <div class="box"><h3>Premium vs Preference Score</h3><canvas id="c-bl-pp"></canvas></div>
    <div class="box"><h3>Avg Expiry Days Achieved</h3><canvas id="c-bl-ex"></canvas></div>
  </div>
  <table id="t-baseline"></table>
</div>

<div id="tab-oracle" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Oracle has perfect information (availability, expiry, prices) before each episode.
    Gap = 100 × (R_oracle − R_agent) / |R_oracle| per episode.
  </p>
  <div class="grid">
    <div class="box"><h3>Optimality Gap (%)</h3><canvas id="c-or-gap"></canvas></div>
    <div class="box"><h3>Reward vs Oracle</h3><canvas id="c-or-r"></canvas></div>
  </div>
  <table id="t-oracle"></table>
</div>

<div id="tab-policy" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Store-visit frequencies and route patterns across {n_test} test episodes.
  </p>
  <div class="grid">
    <div class="box"><h3>Store Visit Frequency by Agent</h3><canvas id="c-pol-freq"></canvas></div>
    <div class="box"><h3>Avg Route Length by Agent</h3><canvas id="c-pol-len"></canvas></div>
  </div>
  <table id="t-policy"></table>
  <br>
  <table id="t-store-chars"></table>
</div>

<div id="tab-graph" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Graph structure experiments: baseline (all 5 min) vs one/two distant stores (20 min).
  </p>
  <div class="grid">
    <div class="box"><h3>Reward vs Graph Configuration</h3><canvas id="c-gph-r"></canvas></div>
    <div class="box"><h3>Completion Rate vs Graph Configuration</h3><canvas id="c-gph-cr"></canvas></div>
  </div>
  <table id="t-graph"></table>
</div>

<div id="tab-brand" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Brand configuration experiments: narrow (5B/2), baseline (5B/3), rich (10B/3).
  </p>
  <div class="grid">
    <div class="box"><h3>Reward vs Brand Configuration</h3><canvas id="c-br-r"></canvas></div>
    <div class="box"><h3>Completion Rate vs Brand Configuration</h3><canvas id="c-br-cr"></canvas></div>
  </div>
  <table id="t-brand"></table>
</div>

<div id="tab-param" class="panel">
  <div class="grid">
    <div class="box"><h3>Completion Rate vs Duration</h3><canvas id="c-dur-cr"></canvas></div>
    <div class="box"><h3>Completion Rate vs Availability</h3><canvas id="c-av-cr"></canvas></div>
    <div class="box"><h3>Completion Rate vs Goods Count</h3><canvas id="c-gd-cr"></canvas></div>
  </div>
  <table id="t-param"></table>
</div>

<div id="tab-sig" class="panel">
  <p style="color:#9fa8da;font-size:12px;margin-bottom:14px">
    Welch's t-test (two-tailed, α=0.05). Cohen's d: small &lt;0.2, medium 0.2–0.8, large &gt;0.8.
  </p>
  <table id="t-sig"></table>
</div>

<div id="tab-conv" class="panel">
  <div class="grid">
    <div class="box"><h3>Tabular — Reward vs Training Size</h3><canvas id="c-tab-r"></canvas></div>
    <div class="box"><h3>Deep RL — Reward vs Training Size</h3><canvas id="c-deep-r"></canvas></div>
  </div>
  <table id="t-conv"></table>
</div>

<div id="tab-rank" class="panel">
  <table id="t-rank"></table>
</div>

<script>
const R={json.dumps(results)};
const SW={sweep_json};
const AG={json.dumps(agents_ordered)};
const C={json.dumps(AGENT_COLORS)};

function show(id){{
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  event.target.classList.add('active');
}}
function rgba(h,a){{const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);return`rgba(${{r}},${{g}},${{b}},${{a}})`}}
function n(v,d=1){{return v==null?'—':Number(v).toFixed(d)}}
function cr(v){{if(v==null)return'—';const cl=v>=.95?'g':v>=.80?'a':'r';return`<span class="${{cl}}">${{(v*100).toFixed(1)}}%</span>`}}
const defs={{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{labels:{{color:'#e8eaf6',font:{{size:10}},boxWidth:10}}}}}},
  scales:{{x:{{ticks:{{color:'#9fa8da'}},grid:{{color:'rgba(255,255,255,.05)'}}}},
           y:{{ticks:{{color:'#9fa8da'}},grid:{{color:'rgba(255,255,255,.08)'}}}}}}
}};
function bar(id,labels,ds,yO={{}}){{
  const ctx=document.getElementById(id)?.getContext('2d');if(!ctx)return;
  new Chart(ctx,{{type:'bar',data:{{labels,datasets:ds}},
    options:{{...defs,scales:{{...defs.scales,y:{{...defs.scales.y,...yO}}}}}}}});
}}
function line(id,labels,ds,yO={{}}){{
  const ctx=document.getElementById(id)?.getContext('2d');if(!ctx)return;
  new Chart(ctx,{{type:'line',data:{{labels,datasets:ds}},
    options:{{...defs,scales:{{...defs.scales,y:{{...defs.scales.y,...yO}}}}}}}});
}}

const bl=R.baseline||{{}};
const names=AG.filter(a=>bl[a]);
const rewards=names.map(a=>bl[a].mean_reward);
const best=names[rewards.indexOf(Math.max(...rewards))];

// KPIs
document.getElementById('kpis').innerHTML=`
  <div class="kpi"><div class="kv">${{best}}</div><div class="kl">Best agent (reward)</div></div>
  <div class="kpi"><div class="kv">${{n(Math.max(...rewards),0)}}</div><div class="kl">Peak reward ±${{n(bl[best]?.ci95_reward,1)}}</div></div>
  <div class="kpi"><div class="kv">{n_test}</div><div class="kl">Test episodes (95% CI)</div></div>
  <div class="kpi"><div class="kv">${{names.length}}</div><div class="kl">Algorithms compared</div></div>`;

// Overview charts
bar('c-ov-r',names,[{{label:'Reward',data:names.map(a=>bl[a].mean_reward),
  backgroundColor:names.map(a=>rgba(C[a]||'#7c83fd',.8)),borderColor:names.map(a=>C[a]||'#7c83fd'),borderWidth:1}}]);
bar('c-ov-cr',names,[{{label:'Completion',data:names.map(a=>bl[a].mean_cr),
  backgroundColor:names.map(a=>rgba(C[a]||'#7c83fd',.8)),borderColor:names.map(a=>C[a]||'#7c83fd'),borderWidth:1}}],
  {{min:0,max:1.05,ticks:{{callback:v=>(v*100).toFixed(0)+'%'}}}});

// Baseline table
let h='<tr><th>Agent</th><th>Reward</th><th>±CI</th><th>CR</th><th>±CI</th><th>Stores</th><th>Time</th><th>Premium%</th><th>Pref</th><th>Success</th></tr>';
names.forEach(a=>{{const d=bl[a];
  h+=`<tr><td><b>${{a}}</b></td><td>${{n(d.mean_reward,1)}}</td><td>±${{n(d.ci95_reward,1)}}</td>
  <td>${{cr(d.mean_cr)}}</td><td>±${{n(d.ci95_cr,3)}}</td><td>${{n(d.mean_stores,1)}}</td>
  <td>${{n(d.mean_time,0)}}m</td><td>${{n(d.mean_premium,1)}}%</td>
  <td>${{n(d.mean_pref,2)}}</td><td>${{n(d.success_rate*100,1)}}%</td></tr>`}});
document.getElementById('t-baseline').innerHTML=h;

// Premium vs Pref scatter
const ctx2=document.getElementById('c-bl-pp')?.getContext('2d');
if(ctx2)new Chart(ctx2,{{type:'scatter',
  data:{{datasets:names.map(a=>{{const d=bl[a];return{{label:a,
    data:[{{x:d.mean_premium,y:d.mean_pref}}],backgroundColor:rgba(C[a]||'#7c83fd',.85),
    pointRadius:9,pointHoverRadius:12}}}})}}}},
  options:{{...defs,scales:{{
    x:{{...defs.scales.x,title:{{display:true,text:'Avg Premium (%)',color:'#9fa8da'}}}},
    y:{{...defs.scales.y,title:{{display:true,text:'Avg Pref Score',color:'#9fa8da'}}}}}}}}}});
bar('c-bl-ex',names,[{{label:'Expiry Days',data:names.map(a=>bl[a].mean_expiry||0),
  backgroundColor:names.map(a=>rgba(C[a]||'#7c83fd',.8)),borderColor:names.map(a=>C[a]||'#7c83fd'),borderWidth:1}}]);

// Oracle tab
const orb=R.oracle_benchmark||{{}};
const ora=orb.agents||{{}};
const ornames=[...Object.keys(ora)];
bar('c-or-gap',ornames,[{{label:'Optimality Gap (%)',
  data:ornames.map(a=>ora[a]?.optimality_gap_pct),
  backgroundColor:ornames.map(a=>rgba(C[a]||'#7c83fd',.8)),borderWidth:1}}]);
const orvals=[{{mean_reward:(orb.oracle||{{}}).mean_reward,name:'Oracle'}},...ornames.map(a=>{{return{{name:a,mean_reward:ora[a]?.mean_reward}}}})];
bar('c-or-r',orvals.map(v=>v.name),[{{label:'Mean Reward',
  data:orvals.map(v=>v.mean_reward),
  backgroundColor:orvals.map(v=>rgba(C[v.name]||'#f1c40f',.8)),borderWidth:1}}]);
let ot='<tr><th>Agent</th><th>Mean Reward</th><th>±CI</th><th>CR</th><th>Success Rate</th><th>Opt. Gap (%)</th><th>Gap Std</th></tr>';
ot+=`<tr><td><b>Oracle</b></td><td>${{n((orb.oracle||{{}}).mean_reward,1)}}</td><td>±${{n((orb.oracle||{{}}).ci95_reward,1)}}</td><td>${{cr((orb.oracle||{{}}).mean_cr)}}</td><td>${{n((orb.oracle||{{}}).success_rate*100,1)}}%</td><td>0.0%</td><td>—</td></tr>`;
ornames.forEach(a=>{{const d=ora[a];
  ot+=`<tr><td>${{a}}</td><td>${{n(d?.mean_reward,1)}}</td><td>±${{n(d?.ci95_reward,1)}}</td><td>${{cr(d?.mean_cr)}}</td><td>${{n(d?.success_rate*100,1)}}%</td><td>${{n(d?.optimality_gap_pct,1)}}%</td><td>±${{n(d?.gap_std_pct,1)}}%</td></tr>`}});
document.getElementById('t-oracle').innerHTML=ot;

// Policy analysis
const pol=R.policy_analysis||{{}};
const polAgents=Object.keys(pol.agent_policies||{{}});
const storeLabels=['Store 0','Store 1','Store 2','Store 3','Store 4','Store 5'];
if(polAgents.length>0){{
  const vf=polAgents.map(a=>pol.agent_policies[a]?.visit_freq||[]);
  const datasets=polAgents.map((a,i)=>{{return{{label:a,data:vf[i],backgroundColor:rgba(C[a]||'#7c83fd',.7),borderColor:C[a]||'#7c83fd',borderWidth:1}}}});
  bar('c-pol-freq',storeLabels,datasets);
  bar('c-pol-len',polAgents,[{{label:'Avg Route Length',data:polAgents.map(a=>pol.agent_policies[a]?.avg_route_length),
    backgroundColor:polAgents.map(a=>rgba(C[a]||'#7c83fd',.8)),borderWidth:1}}]);
  let pt='<tr><th>Agent</th><th>Dominant Store</th><th>Avg Route Length</th><th>Mean Reward</th><th>CR</th></tr>';
  polAgents.forEach(a=>{{const d=pol.agent_policies[a];
    pt+=`<tr><td><b>${{a}}</b></td><td>Store ${{d?.dominant_store}}</td><td>${{n(d?.avg_route_length,2)}}</td><td>${{n(d?.mean_reward,1)}}</td><td>${{cr(d?.mean_cr)}}</td></tr>`}});
  document.getElementById('t-policy').innerHTML=pt;
  const sc=pol.store_characteristics||[];
  let st='<tr><th>Store</th><th>Avg Best Pref Score</th><th>Avg Price Premium (%)</th><th>Brands/Good</th><th>Travel Time (min)</th></tr>';
  sc.forEach(s=>{{st+=`<tr><td>Store ${{s.store_id}}</td><td>${{n(s.avg_best_pref_score,3)}}</td><td>${{n(s.avg_price_premium_pct,2)}}%</td><td>${{s.brands_stocked_per_good}}</td><td>${{s.travel_time_min}}</td></tr>`}});
  document.getElementById('t-store-chars').innerHTML=st;
}}

// Graph experiments
const gd=R.graph_experiments||{{}};
const gcfgs=Object.keys(gd);
if(gcfgs.length>0){{
  const gallA=[...new Set(gcfgs.flatMap(c=>Object.keys(gd[c]?.agents||{{}})))];
  const mkGDs=(metric)=>gallA.map(a=>{{return{{label:a,data:gcfgs.map(c=>gd[c]?.agents?.[a]?.[metric]??null),
    borderColor:C[a]||'#7c83fd',backgroundColor:rgba(C[a]||'#7c83fd',.1),borderWidth:2,pointRadius:5,tension:.2,fill:false}}}});
  line('c-gph-r',gcfgs,mkGDs('mean_reward'));
  line('c-gph-cr',gcfgs,mkGDs('mean_cr'),{{min:0,max:1.05,ticks:{{callback:v=>(v*100).toFixed(0)+'%'}}}});
  let gt='<tr><th>Agent</th>'+gcfgs.map(c=>`<th colspan="2">${{c}}</th>`).join('')+'</tr>';
  gt+='<tr><th></th>'+gcfgs.map(()=>'<th>Reward</th><th>CR</th>').join('')+'</tr>';
  gallA.forEach(a=>{{gt+=`<tr><td><b>${{a}}</b></td>`;
    gcfgs.forEach(c=>{{const d=gd[c]?.agents?.[a];gt+=d?`<td>${{n(d.mean_reward,1)}}</td><td>${{cr(d.mean_cr)}}</td>`:'<td>—</td><td>—</td>'}});gt+='</tr>'}});
  document.getElementById('t-graph').innerHTML=gt;
}}

// Brand experiments
const bd=R.brand_experiments||{{}};
const bcfgs=Object.keys(bd);
if(bcfgs.length>0){{
  const ballA=[...new Set(bcfgs.flatMap(c=>Object.keys(bd[c]?.agents||{{}})))];
  const mkBDs=(metric)=>ballA.map(a=>{{return{{label:a,data:bcfgs.map(c=>bd[c]?.agents?.[a]?.[metric]??null),
    borderColor:C[a]||'#7c83fd',backgroundColor:rgba(C[a]||'#7c83fd',.1),borderWidth:2,pointRadius:5,tension:.2,fill:false}}}});
  line('c-br-r',bcfgs,mkBDs('mean_reward'));
  line('c-br-cr',bcfgs,mkBDs('mean_cr'),{{min:0,max:1.05,ticks:{{callback:v=>(v*100).toFixed(0)+'%'}}}});
  let bt='<tr><th>Agent</th>'+bcfgs.map(c=>`<th colspan="2">${{c}}</th>`).join('')+'</tr>';
  bt+='<tr><th></th>'+bcfgs.map(()=>'<th>Reward</th><th>CR</th>').join('')+'</tr>';
  ballA.forEach(a=>{{bt+=`<tr><td><b>${{a}}</b></td>`;
    bcfgs.forEach(c=>{{const d=bd[c]?.agents?.[a];bt+=d?`<td>${{n(d.mean_reward,1)}}</td><td>${{cr(d.mean_cr)}}</td>`:'<td>—</td><td>—</td>'}});bt+='</tr>'}});
  document.getElementById('t-brand').innerHTML=bt;
}}

// Parametric
const pd=R.parametric||{{}};
function paramLine(cid,param,metric,yO={{}}){{
  const ed=pd[param]||{{}};const pv=Object.keys(ed).sort((a,b)=>+a-+b);
  const ag=AG.filter(a=>pv.some(p=>ed[p]?.[a]));
  const ds=ag.map(a=>{{return{{label:a,data:pv.map(p=>ed[p]?.[a]?.[metric]??null),
    borderColor:C[a]||'#7c83fd',backgroundColor:rgba(C[a]||'#7c83fd',.1),
    borderWidth:2,pointRadius:5,tension:.2,fill:false}}}});
  line(cid,pv,ds,yO);
}}
const crY={{min:0,max:1.05,ticks:{{callback:v=>(v*100).toFixed(0)+'%'}}}};
paramLine('c-dur-cr','duration','mean_cr',crY);
paramLine('c-av-cr','avail_prob','mean_cr',crY);
paramLine('c-gd-cr','n_goods','mean_cr',crY);
let paramt='<tr><th>Parameter</th><th>Value</th>'+AG.filter(a=>names.includes(a)).map(a=>`<th>${{a}}</th>`).join('')+'</tr>';
['duration','avail_prob','n_goods'].forEach(param=>{{const ed=pd[param]||{{}};
  Object.keys(ed).sort((a,b)=>+a-+b).forEach(v=>{{paramt+=`<tr><td>${{param}}</td><td>${{v}}</td>`+
    AG.filter(a=>names.includes(a)).map(a=>`<td>${{n(ed[v]?.[a]?.mean_cr*100,1)}}%</td>`).join('')+'</tr>'}});
}});
document.getElementById('t-param').innerHTML=paramt;

// Significance
const sig=R.significance_table||{{}};
let sh='<tr><th>Pair</th><th>t-stat</th><th>p-value</th><th>Significant?</th><th>Cohen\'s d</th><th>Better</th></tr>';
Object.entries(sig).sort((a,b)=>a[1].p_value-b[1].p_value).forEach(([k,v])=>{{
  const stars=v.p_value<.001?'***':v.p_value<.01?'**':v.p_value<.05?'*':'';
  const sc=v.significant?'sig':'ns';
  sh+=`<tr><td>${{k}}</td><td>${{n(v.t_stat,3)}}</td><td>${{n(v.p_value,5)}}</td>
  <td><span class="${{sc}}">${{v.significant?'✓ '+stars:'✗'}}</span></td>
  <td>${{n(v.cohens_d,3)}}</td><td><b>${{v.better}}</b></td></tr>`}});
document.getElementById('t-sig').innerHTML=sh;

// Convergence
const tab=SW.tabular||{{}};const deep=SW.deep||{{}};
const ts=Object.keys(tab).sort((a,b)=>+a-+b);
const ds2=Object.keys(deep).sort((a,b)=>+a-+b);
const mkConvDs=(data,sizes,agts)=>agts.filter(a=>sizes.some(s=>data[s]?.[a])).map(a=>{{return{{label:a,
  data:sizes.map(s=>data[s]?.[a]?.mean_reward??null),
  borderColor:C[a]||'#7c83fd',backgroundColor:rgba(C[a]||'#7c83fd',.1),
  borderWidth:2,pointRadius:5,tension:.15,fill:false}}}});
line('c-tab-r',ts.map(s=>parseInt(s).toLocaleString()),mkConvDs(tab,ts,['Q-Learning','SARSA','ExpSARSA']));
line('c-deep-r',ds2.map(s=>parseInt(s).toLocaleString()),mkConvDs(deep,ds2,['DQN','DoubleDQN','DuelingDQN']));
const allS=[...new Set([...ts,...ds2])].sort((a,b)=>+a-+b);
let ct='<tr><th>Agent</th>'+allS.map(s=>`<th>${{parseInt(s).toLocaleString()}}</th>`).join('')+'</tr>';
['Q-Learning','SARSA','ExpSARSA','DQN','DoubleDQN','DuelingDQN'].forEach(a=>{{ct+=`<tr><td><b>${{a}}</b></td>`;
  allS.forEach(s=>{{const d=tab[s]?.[a]||deep[s]?.[a];ct+=d?`<td>${{n(d.mean_reward,0)}}</td>`:'<td>—</td>'}});ct+='</tr>'}});
document.getElementById('t-conv').innerHTML=ct;

// Rankings
const sorted=[...names].sort((a,b)=>bl[b].mean_reward-bl[a].mean_reward);
let rt='<tr><th>Rank</th><th>Agent</th><th>Reward</th><th>±CI</th><th>CR</th><th>Success Rate</th><th>Stores</th><th>Category</th></tr>';
sorted.forEach((a,i)=>{{const d=bl[a];const rc=i===0?'r1':i===1?'r2':i===2?'r3':'';
  const cat=Object.entries({json.dumps(AGENT_GROUPS)}).find(([_,v])=>v.includes(a))?.[0]||'';
  rt+=`<tr><td class="${{rc}}">${{i===0?'🥇':i===1?'🥈':i===2?'🥉':i+1}}</td>
  <td><b>${{a}}</b></td><td class="${{rc}}">${{n(d.mean_reward,1)}}</td><td>±${{n(d.ci95_reward,1)}}</td>
  <td>${{cr(d.mean_cr)}}</td><td>${{n(d.success_rate*100,1)}}%</td>
  <td>${{n(d.mean_stores,1)}}</td><td>${{cat}}</td></tr>`}});
document.getElementById('t-rank').innerHTML=rt;
</script>
</body>
</html>"""

    dash_path = os.path.join(RESULTS_DIR, '..', 'dashboard.html')
    with open(dash_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Dashboard saved → {dash_path}")
    return dash_path


# ════════════════════════════════════════════════════════════════════════════
# SECTION 16 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Market Search RL — Full Experiment Suite')
    parser.add_argument('--no-sweep', action='store_true',
                        help='Skip training-size convergence sweep')
    parser.add_argument('--quick', action='store_true',
                        help='N=100 episodes — fast smoke test')
    args = parser.parse_args()

    if args.quick:
        global TEST_EPISODES, TRAIN_TABULAR, TRAIN_DEEP, TRAIN_PG
        TEST_EPISODES = 100
        TRAIN_TABULAR = 300
        TRAIN_DEEP    = 80
        TRAIN_PG      = 80
        print("⚠  Quick mode: TEST_EPISODES=100, reduced training budgets")

    env_kw = {'duration': 120, 'avail_prob': 0.60, 'n_goods': 20}

    print("=" * 65)
    print("  Market Search and Purchase Scheduling — RL Experiment Suite")
    print(f"  TEST_EPISODES = {TEST_EPISODES}")
    print(f"  Seed = {SEED}")
    print("=" * 65)

    all_results = {}
    sweep_results = None

    # ── STEP 1: Baseline ──────────────────────────────────────────────────
    print("\n[1] Baseline experiment...")
    baseline_results, sig_table = run_baseline()
    all_results['baseline']         = _strip_raw(baseline_results)
    all_results['significance_table'] = sig_table

    # Collect trained agents for reuse in later experiments
    trained_agents = {}
    env_base = make_env(**env_kw)
    for name, agent, n_train in make_agents(env_base, seed=SEED):
        env2 = make_env(**env_kw, seed=SEED + hash(name) % 97)
        agent.env = env2
        train_agent(agent, env2, n_train, name)
        trained_agents[name] = agent

    # ── STEP 2: Oracle benchmark ──────────────────────────────────────────
    print("\n[2] Oracle benchmark and optimality gap analysis...")
    oracle_results = run_oracle_benchmark(baseline_results, env_kw)
    all_results['oracle_benchmark'] = oracle_results

    # ── STEP 3: Policy analysis ───────────────────────────────────────────
    print("\n[3] Policy function analysis...")
    policy_results = run_policy_analysis(baseline_results, env_kw)
    all_results['policy_analysis'] = policy_results

    # ── STEP 4: Environment parameters ───────────────────────────────────
    print("\n[4] Environment parameter documentation...")
    env_params = document_environment_parameters(env_kw)
    all_results['environment_parameters'] = env_params

    # ── STEP 5: Parametric experiments ────────────────────────────────────
    print("\n[5] Parametric experiments (duration / availability / goods)...")
    parametric_results = run_parametric_experiments()
    all_results['parametric'] = parametric_results

    # ── STEP 6: Graph structure experiments ───────────────────────────────
    print("\n[6] Graph structure experiments...")
    graph_results = run_graph_experiments(trained_agents)
    all_results['graph_experiments'] = graph_results

    # ── STEP 7: Brand configuration experiments ───────────────────────────
    print("\n[7] Brand configuration experiments...")
    brand_results = run_brand_experiments(trained_agents)
    all_results['brand_experiments'] = brand_results

    # ── STEP 8: Convergence sweep (optional) ──────────────────────────────
    if not args.no_sweep:
        print("\n[8] Training-size convergence sweep...")
        sweep_results = run_training_size_sweep()
        sweep_path = os.path.join(RESULTS_DIR, 'training_size_results.json')
        with open(sweep_path, 'w') as f:
            json.dump(sweep_results, f, indent=2, cls=_NumpyEncoder)
        print(f"✓ Sweep results saved → {sweep_path}")
    else:
        sweep_path = os.path.join(RESULTS_DIR, 'training_size_results.json')
        if os.path.exists(sweep_path):
            with open(sweep_path) as f:
                sweep_results = json.load(f)
            print(f"  Loaded existing sweep → {sweep_path}")

    # ── STEP 9: Save all results ──────────────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)
    print(f"\n✓ All results saved → {results_path}")

    # ── STEP 10: Figures ──────────────────────────────────────────────────
    print("\n[9] Generating figures...")
    generate_figures(all_results, sweep_results)

    # ── STEP 11: Dashboard ────────────────────────────────────────────────
    print("\n[10] Generating interactive dashboard...")
    generate_dashboard(all_results, sweep_results)

    print("\n" + "=" * 65)
    print("✅  ALL EXPERIMENTS COMPLETE")
    print(f"   experiment_results.json     — all results")
    print(f"   training_size_results.json  — convergence sweep")
    print(f"   significance_table.json     — Welch's t-test")
    print(f"   figures/                    — all plots")
    print(f"   dashboard.html              — interactive dashboard")
    print("=" * 65)
    print("\nUsage:")
    print("  python experiments.py             # full run")
    print("  python experiments.py --no-sweep  # skip convergence sweep")
    print("  python experiments.py --quick     # fast smoke test (N=100)")


if __name__ == '__main__':
    main()
