"""
Value Iteration Agent — Dynamic Programming (Bellman, 1957)

Model-based algorithm: exploits known transition probabilities.
Since our environment's stochastic factors (availability, expiry) have
known distributions, we can compute expected values analytically.

Value Iteration computes the optimal value function V*(s) by:
    V_{k+1}(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) V_k(s') ]

For the shopping environment:
- States are discretized: (time_bucket, current_store, items_needed_bitmask)
- Transition probabilities for availability are known (AVAIL_PROB)
- Expected rewards can be computed analytically

Since the full state space (items bitmask = 2^20) is too large for exact VI,
we use an approximation:
  - Discretize time into buckets
  - Represent items as a count (not full bitmask) for tractability
  - Run VI for a limited number of iterations, then use the value function
    as a heuristic guide for action selection

This is "approximate dynamic programming" — common in large state spaces.
The key insight: because probabilities are KNOWN, VI can still achieve
near-optimal solutions even with state abstraction.
"""

import numpy as np
from collections import defaultdict
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


class ValueIterationAgent(BaseAgent):
    """
    Approximate Value Iteration agent.

    State abstraction: (time_bucket, current_store, n_items_remaining)
    This reduces the 2^20 bitmask to a count, making VI tractable.
    Action selection: greedy w.r.t. computed V table + local heuristic for buy decisions.
    """

    def __init__(self, env, seed=42, gamma=0.95,
                 n_iter=200, time_buckets=8):
        super().__init__(env, seed)
        self.gamma = gamma
        self.n_iter = n_iter
        self.time_buckets = time_buckets
        self.V = {}   # (time_bucket, store, n_items) -> value
        self._trained = False
        self.eps = 0.0  # for compatibility with logging

    def get_name(self): return "ValueIteration"

    def _time_bucket(self, t):
        """Discretize remaining time into buckets."""
        total = self.env.TOTAL_DURATION
        bucket = int(t / total * self.time_buckets)
        return min(bucket, self.time_buckets - 1)

    def _state_key(self, time_remaining, store, n_items):
        return (self._time_bucket(time_remaining), store, n_items)

    def _get_V(self, key):
        return self.V.get(key, 0.0)

    def train(self, n_episodes=None):
        """
        Run Value Iteration over the abstract state space.
        Transitions modelled analytically using known probabilities.
        """
        env = self.env
        N_STORES = env.N_STORES
        N_GOODS = env.N_GOODS
        AVAIL_PROB = env.AVAIL_PROB
        BRANDS_PER_STORE = env.BRANDS_PER_STORE
        TRAVEL_TIME = env.TRAVEL_TIME
        VISIT_TIME = env.VISIT_TIME
        TOTAL = env.TOTAL_DURATION

        # Expected items buyable per store visit
        # P(at least one brand available for a good at a store)
        p_buy_item = 1.0 - (1.0 - AVAIL_PROB) ** BRANDS_PER_STORE
        # Expected items to find per store visit
        expected_items_per_store = N_GOODS * p_buy_item  # across all goods

        # Expected reward per item bought (heuristic average)
        # premium: E[premium] ≈ 0 (uniform [-30,+30])
        # pref: E[pref] ≈ 3.0 (average of 1..5)
        # expiry: E[expiry] ≈ 3.5 (uniform [0,7])
        avg_reward_per_item = (env.W_PREMIUM * 0.0 +
                               env.W_PREF_SCORE * 3.0 +
                               env.W_EXPIRY * 3.5)

        # Build state space
        time_steps = list(range(self.time_buckets + 1))
        stores = list(range(N_STORES))
        item_counts = list(range(N_GOODS + 1))

        # Initialize V
        for tb in time_steps:
            for s in stores:
                for n in item_counts:
                    key = (tb, s, n)
                    # Terminal value estimate: missing items penalty
                    self.V[key] = env.W_MISSING_ITEM * n

        # Value Iteration
        for iteration in range(self.n_iter):
            delta = 0.0
            new_V = {}

            for tb in time_steps:
                t_remaining = tb / self.time_buckets * TOTAL
                for s in stores:
                    for n in item_counts:
                        key = (tb, s, n)
                        old_val = self.V[key]

                        if n == 0 or t_remaining <= 0:
                            new_V[key] = 0.0
                            continue

                        best_val = env.W_MISSING_ITEM * n  # end action

                        # Action: stay and buy
                        # Expected items bought this visit (can't buy more than needed)
                        exp_buy = min(n, expected_items_per_store)
                        buy_reward = exp_buy * avg_reward_per_item
                        n_after = max(0, n - round(exp_buy))
                        tb_after = self._time_bucket(t_remaining)
                        future_key = (tb_after, s, n_after)
                        stay_val = buy_reward + self.gamma * self._get_V(future_key)
                        best_val = max(best_val, stay_val)

                        # Action: travel to another store
                        travel_cost = TRAVEL_TIME + VISIT_TIME
                        if t_remaining >= travel_cost:
                            t_after_travel = t_remaining - travel_cost
                            tb_after_travel = self._time_bucket(t_after_travel)
                            for s2 in stores:
                                if s2 == s:
                                    continue
                                travel_reward = env.W_STORE_VISIT  # penalty
                                future_key2 = (tb_after_travel, s2, n)
                                travel_val = travel_reward + self.gamma * self._get_V(future_key2)
                                best_val = max(best_val, travel_val)

                        new_V[key] = best_val
                        delta = max(delta, abs(new_V[key] - old_val))

            self.V = new_V
            if delta < 1e-3:
                break

        self._trained = True

    def select_action(self, state, training=True):
        """
        Greedy action selection w.r.t. V table.
        For buy decisions: use heuristic (best available brand).
        For travel decisions: use VI value function.
        """
        env = self.env
        cur = state['current_store']
        needed = state['items_needed']
        time_rem = state['time_remaining']
        n = len(needed)

        if n == 0:
            return {'type': 'end'}

        info = state['revealed_info'].get(cur, {})
        avail = info.get('avail', np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        expiry = info.get('expiry', np.zeros((env.N_GOODS, env.N_BRANDS)))
        pref = env.brand_preference

        # First: buy best available item at current store
        best_buy, best_score = None, -np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if avail[g, b]:
                    score = (pref[g, b] * env.W_PREF_SCORE +
                             expiry[g, b] * env.W_EXPIRY)
                    if score > best_score:
                        best_score, best_buy = score, {'type': 'buy', 'good': g, 'brand': b}
        if best_buy is not None:
            return best_buy

        # No items to buy here — decide whether to travel
        if not self._trained:
            # Fallback: travel to unvisited store with most remaining items
            valid = env.get_valid_actions()
            travel_actions = [a for a in valid if a['type'] == 'travel']
            if not travel_actions:
                return {'type': 'end'}
            # Prefer unvisited stores
            unvisited = [a for a in travel_actions
                         if a['store'] not in state['visited_stores']]
            candidates = unvisited if unvisited else travel_actions
            return candidates[self.rng.integers(len(candidates))]

        # Use V table to select best travel destination
        tb_now = self._time_bucket(time_rem)
        best_travel, best_v = None, -np.inf

        # End action value
        end_val = env.W_MISSING_ITEM * n
        best_v = end_val

        travel_cost = env.TRAVEL_TIME + env.VISIT_TIME
        if time_rem >= travel_cost:
            t_after = time_rem - travel_cost
            tb_after = self._time_bucket(t_after)
            for s2 in range(env.N_STORES):
                if s2 == cur:
                    continue
                key = (tb_after, s2, n)
                v = env.W_STORE_VISIT + self.gamma * self._get_V(key)
                if v > best_v:
                    best_v, best_travel = v, s2

        if best_travel is not None:
            return {'type': 'travel', 'store': best_travel}
        return {'type': 'end'}

    def update(self, state, action, reward, next_state, done):
        pass  # Model-based: no online updates needed

    def post_episode(self):
        pass
