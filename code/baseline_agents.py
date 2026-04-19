"""
Baseline agents for comparison:
- RandomAgent: random valid action
- GreedyAgent: always buy best available at current store, travel to unvisited store with most items, end when time low
"""

import numpy as np
from typing import Dict
from environment import MarketEnvironment
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


class RandomAgent(BaseAgent):
    """Selects uniformly random valid action."""

    def get_name(self) -> str:
        return "Random"

    def select_action(self, state: Dict, training: bool = True) -> Dict:
        valid_actions = self.env.get_valid_actions()
        idx = self.rng.integers(0, len(valid_actions))
        return valid_actions[idx]


class GreedyAgent(BaseAgent):
    """
    Heuristic greedy agent:
    1. Buy all needed items available at current store (prefer highest pref score, then lowest premium)
    2. Travel to unvisited store that stocks the most needed items
    3. End when no useful stores remain or time is low
    """

    def get_name(self) -> str:
        return "Greedy"

    def select_action(self, state: Dict, training: bool = True) -> Dict:
        env = self.env
        cur = state['current_store']
        needed = state['items_needed']
        info = state['revealed_info'].get(cur, {})
        avail = info.get('avail', np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        pref = env.brand_preference
        premium = info.get('premium', np.zeros((env.N_GOODS, env.N_BRANDS)))

        # Step 1: Buy best available needed item at current store
        best_action = None
        best_score = -np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if avail[g, b]:
                    score = pref[g, b] * 10 - premium[g, b] * 100  # prefer high pref, low premium
                    if score > best_score:
                        best_score = score
                        best_action = {'type': 'buy', 'good': g, 'brand': b}
        if best_action is not None:
            return best_action

        # Step 2: Travel to store with most needed items (unvisited preferred)
        time_rem = state['time_remaining']
        cost = env.TRAVEL_TIME + env.VISIT_TIME  # exactly 20 min needed
        if time_rem < cost:
            return {'type': 'end'}

        # Score stores by expected number of needed items available
        best_store = None
        best_store_score = -np.inf
        for s in range(env.N_STORES):
            if s == cur:
                continue
            cost = env.TRAVEL_TIME + env.VISIT_TIME
            if time_rem < cost:
                continue
            # Use known stock info (store_stocks) as proxy before visiting
            expected = 0
            for g in needed:
                if env.store_stocks[s, g].any():
                    expected += env.AVAIL_PROB  # expected brands available
            # Bonus for unvisited
            if s not in state['visited_stores']:
                expected += 2.0
            if expected > best_store_score:
                best_store_score = expected
                best_store = s

        if best_store is not None and best_store_score > 0:
            return {'type': 'travel', 'store': best_store}

        return {'type': 'end'}
