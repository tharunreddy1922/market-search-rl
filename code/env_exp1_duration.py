"""
Experiment 1: Extended Duration (3 hours / 180 minutes)

Change from baseline:
  - TOTAL_DURATION: 120 min → 180 min
  - Store revisiting: ALLOWED (agent can return to same store)
    The key rule change: revisiting a store does NOT charge the visit time again
    but DOES allow buying newly-desired items found in a re-check.
    However, re-visiting is still free of the 15-min visit cost (already paid),
    but the 5-min travel cost still applies each way.

Rationale:
  With 3 hours, an agent can realistically circle back to a store it visited
  earlier once more items have been revealed. The visit cost (15 min) is only
  charged once per store (first visit), but travel (5 min) applies every time
  the agent moves between stores.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from environment import MarketEnvironment


class MarketEnvExp1(MarketEnvironment):
    """
    Experiment 1: 3-hour duration, store revisiting allowed.
    First visit to a store costs 15 min. Subsequent visits cost only 5 min travel.
    """

    TOTAL_DURATION = 180   # 3 hours

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)

    def reset(self, rng_seed: Optional[int] = None) -> Dict:
        state = super().reset(rng_seed=rng_seed)
        # Track stores visited so far (inherited), plus allow revisits
        self._revisit_count = {}  # store_idx → number of times visited
        self._revisit_count[self.current_store] = 1
        return state

    def _visit_store(self, store_idx: int):
        """
        Override: first visit deducts VISIT_TIME; subsequent visits
        only deduct travel time (already paid in step() via travel action).
        """
        first_visit = store_idx not in self.visited_stores
        if first_visit:
            self.visited_stores.add(store_idx)
            self.time_remaining -= self.VISIT_TIME  # 15 min first visit only
        # Always reveal info — store copies (not views) so mutations
        # to actual_avail cannot silently corrupt revealed inventory
        self.revealed_info[store_idx] = {
            'avail':   self.actual_avail[store_idx].copy(),
            'expiry':  self.expiry_dates[store_idx].copy(),
            'premium': self.price_premium[store_idx].copy(),
        }

    def get_valid_actions(self) -> List[Dict]:
        """
        Same as base but travel actions are valid to ANY other store
        (including previously visited ones) if time allows.
        """
        if self.done:
            return []

        actions = [{'type': 'end'}]
        cur = self.current_store
        info = self.revealed_info.get(cur, {})
        avail = info.get('avail', np.zeros((self.N_GOODS, self.N_BRANDS), dtype=bool))

        # Buy actions
        for g in self.items_needed:
            for b in range(self.N_BRANDS):
                if avail[g, b]:
                    actions.append({'type': 'buy', 'good': g, 'brand': b})

        # Travel: revisited stores cost TRAVEL_TIME (5 min) only
        #          unvisited stores cost TRAVEL_TIME + VISIT_TIME (20 min)
        for s in range(self.N_STORES):
            if s == cur:
                continue
            if s in self.visited_stores:
                # Revisit: only travel cost (5 min)
                if self.time_remaining >= self.TRAVEL_TIME:
                    actions.append({'type': 'travel', 'store': s})
            else:
                # First visit: travel + visit time (20 min)
                if self.time_remaining >= self.TRAVEL_TIME + self.VISIT_TIME:
                    actions.append({'type': 'travel', 'store': s})

        return actions

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Override travel to apply correct revisit logic."""
        assert not self.done, "Episode already ended"
        reward = 0.0
        info = {}

        if action['type'] == 'end':
            self.done = True

        elif action['type'] == 'buy':
            g, b = action['good'], action['brand']
            cur = self.current_store
            if g in self.items_needed and self.actual_avail[cur, g, b]:
                premium_pct = self.price_premium[cur, g, b] * 100
                expiry = self.expiry_dates[cur, g, b]
                pref = self.brand_preference[g, b]
                self.items_bought[g] = (b, premium_pct, expiry, pref)
                self.items_needed.discard(g)
                reward += self.W_PREMIUM * premium_pct
                reward += self.W_PREF_SCORE * pref
                reward += self.W_EXPIRY * expiry

        elif action['type'] == 'travel':
            s = action['store']
            self.time_remaining -= self.TRAVEL_TIME   # always 5 min travel
            self.current_store = s
            self._visit_store(s)                      # 15 min only on first visit
            reward += self.W_STORE_VISIT              # efficiency penalty still applies

        if self.time_remaining <= 0:
            self.done = True

        if self.done:
            missing = len(self.items_needed)
            reward += self.W_MISSING_ITEM * missing
            info['missing_items'] = missing
            info['items_bought'] = len(self.items_bought)
            info['stores_visited'] = len(self.visited_stores)
            info['time_used'] = self.TOTAL_DURATION - self.time_remaining

        return self._get_state(), reward, self.done, info
