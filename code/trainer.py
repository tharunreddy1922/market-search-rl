"""
Training and evaluation pipeline.
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Optional
from environment import MarketEnvironment
from agents import BaseAgent


def run_episode(env: MarketEnvironment, agent: BaseAgent,
                training: bool = True, ep_seed: Optional[int] = None) -> Dict:
    """Run one episode, return stats dict."""
    state = env.reset(rng_seed=ep_seed)
    total_reward = 0.0
    steps = 0

    while not env.done:
        action = agent.select_action(state, training=training)
        next_state, reward, done, info = env.step(action)

        if training:
            agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1

    if hasattr(agent, 'post_episode'):
        agent.post_episode()

    stats = env.compute_episode_stats()
    stats['total_reward'] = total_reward
    stats['steps'] = steps
    return stats


def train_agent(agent: BaseAgent, n_episodes: int = 10000,
                log_interval: int = 1000, seed_offset: int = 0,
                save_curve: bool = False) -> List[Dict]:
    """Train agent for n_episodes. Returns history of stats."""
    history = []
    start = time.time()
 
    for ep in range(n_episodes):
        stats = run_episode(agent.env, agent, training=True, ep_seed=None)
        stats['episode'] = ep
        history.append(stats)
 
        if (ep + 1) % log_interval == 0:
            recent = history[-log_interval:]
            avg_reward = np.mean([s['total_reward'] for s in recent])
            avg_cr = np.mean([s['completion_rate'] for s in recent])
            eps = getattr(agent, 'eps', float('nan'))
            elapsed = time.time() - start
            print(f"  Ep {ep+1:>6d} | Reward: {avg_reward:>8.2f} | "
                  f"Completion: {avg_cr:.2%} | ε={eps:.3f} | {elapsed:.0f}s")
 
    # NEW: Save training curve so plots can be regenerated without retraining
    if save_curve:
        os.makedirs('results/curves', exist_ok=True)
        curve_data = {
            'agent': agent.get_name(),
            'n_episodes': n_episodes,
            'rewards': [s['total_reward'] for s in history],
            'completion_rates': [s['completion_rate'] for s in history],
            'stores_visited': [s['stores_visited'] for s in history],
        }
        fname = f"results/curves/{agent.get_name()}_curve.json"
        with open(fname, 'w') as f:
            json.dump(curve_data, f)
 
    return history


def evaluate_agent(agent: BaseAgent, n_episodes: int = 1000, seed_offset: int = 99999) -> Dict:
    """Evaluate agent on fixed test episodes. Returns aggregated stats."""
    all_stats = []
    for i in range(n_episodes):
        stats = run_episode(agent.env, agent, training=False, ep_seed=seed_offset + i)
        all_stats.append(stats)

    keys = ['total_reward', 'completion_rate', 'avg_premium_pct', 'avg_pref_score',
            'avg_expiry_days', 'stores_visited', 'time_used', 'items_bought', 'missing_items']

    result = {}
    for k in keys:
        vals = [s[k] for s in all_stats]
        arr  = np.array(vals)
        n    = len(arr)
        result[f'{k}_mean']   = float(np.mean(arr))
        result[f'{k}_std']    = float(np.std(arr))
        result[f'{k}_median'] = float(np.median(arr))
        result[f'{k}_ci95']   = float(1.96 * np.std(arr) / np.sqrt(n))  # ADD THIS LINE
        result[f'{k}_min']    = float(np.min(arr))                        # ADD THIS LINE
        result[f'{k}_max']    = float(np.max(arr))                        # ADD THIS LINE
 
    return result, all_stats
