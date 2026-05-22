"""
Market Search & Purchase Scheduling Environment
Implements the MDP as described in the project specification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class MarketEnvironment:
    """
    MDP Environment for market search and purchase scheduling.
    
    Spec:
    - 6 stores, star-shaped graph, 5 min travel between any pair
    - 2 hour (120 min) duration, 15 min per store visit
    - 20 goods, 5 brands each, each store stocks 3 brands per good
    - Brand availability probability: 60%
    - Price premium: uniform in [-30%, +30%]
    - Expiry date: uniform in [0-7] days
    """

    # Environment constants
    N_STORES = 6
    N_GOODS = 20
    N_BRANDS = 5          # brands per good
    BRANDS_PER_STORE = 3  # brands stocked per good per store
    TOTAL_DURATION = 120  # minutes
    TRAVEL_TIME = 5       # minutes between any two stores (star graph)
    VISIT_TIME = 15       # minutes per store visit
    AVAIL_PROB = 0.60     # probability a stocked brand is available
    MAX_PREMIUM = 0.30    # ±30% price premium

    # Reward weights
    W_MISSING_ITEM = -20.0
    W_PREMIUM = -1.0       # per 1% premium paid
    W_PREF_SCORE = 2.0     # per preference unit
    W_EXPIRY = 0.5         # per day of expiry (higher = better)
    W_STORE_VISIT = -2.0   # penalty per store visited (encourage efficiency)

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self._generate_static_params()

    # ------------------------------------------------------------------
    # Static parameter generation (fixed per environment instance)
    # ------------------------------------------------------------------

    def _generate_static_params(self):
        """Generate fixed parameters: store inventories, premiums, brand preferences."""

        # Brand preference per good: brand 0 is most preferred (score 5) down to brand 4 (score 1)
        # Shuffled per good to add variety
        self.brand_preference = np.zeros((self.N_GOODS, self.N_BRANDS))
        for g in range(self.N_GOODS):
            perm = self.rng.permutation(self.N_BRANDS)
            # scores 5,4,3,2,1 assigned to brands
            for rank, b in enumerate(perm):
                self.brand_preference[g, b] = self.N_BRANDS - rank

        # Each store stocks BRANDS_PER_STORE brands per good (known beforehand)
        # shape: (N_STORES, N_GOODS, N_BRANDS) boolean
        self.store_stocks = np.zeros((self.N_STORES, self.N_GOODS, self.N_BRANDS), dtype=bool)
        for s in range(self.N_STORES):
            for g in range(self.N_GOODS):
                chosen = self.rng.choice(self.N_BRANDS, size=self.BRANDS_PER_STORE, replace=False)
                self.store_stocks[s, g, chosen] = True

        # Price premium per (store, good, brand): uniform [-30%, +30%]
        # Only meaningful for stocked brands
        self.price_premium = self.rng.uniform(
            -self.MAX_PREMIUM, self.MAX_PREMIUM,
            (self.N_STORES, self.N_GOODS, self.N_BRANDS)
        )
        # Zero out premiums for non-stocked brands
        self.price_premium[~self.store_stocks] = 0.0

    # ------------------------------------------------------------------
    # Episode generation (stochastic factors)
    # ------------------------------------------------------------------

    def reset(self, rng_seed: Optional[int] = None) -> Dict:
        """
        Start a new episode. Returns initial state dict.
        Stochastic factors: brand availability, expiry dates.
        """
        if rng_seed is not None:
            ep_rng = np.random.default_rng(rng_seed)
        else:
            ep_rng = self.rng

        # Actual availability this episode: stocked AND available (60% chance)
        # shape: (N_STORES, N_GOODS, N_BRANDS)
        self.actual_avail = (
            self.store_stocks &
            (ep_rng.random((self.N_STORES, self.N_GOODS, self.N_BRANDS)) < self.AVAIL_PROB)
        )

        # Expiry dates: uniform [0, 7] days, per (store, good, brand)
        self.expiry_dates = ep_rng.integers(0, 8, (self.N_STORES, self.N_GOODS, self.N_BRANDS)).astype(float)
        self.expiry_dates[~self.store_stocks] = 0.0

        # Agent state
        self.current_store = 0          # starts at store 0
        self.time_remaining = self.TOTAL_DURATION
        self.visited_stores = set()
        self.items_bought = {}          # {good_idx: (brand_idx, premium, expiry, pref_score)}
        self.items_needed = set(range(self.N_GOODS))
        self.revealed_info = {}         # {store_idx: (avail_matrix, expiry_matrix)}
        self.done = False

        # Visit initial store automatically (pay visit cost)
        self._visit_store(self.current_store)

        return self._get_state()

    def _visit_store(self, store_idx: int):
        """Mark store as visited, deduct time, reveal info (copies, not views)."""
        if store_idx not in self.visited_stores:
            self.visited_stores.add(store_idx)
            self.time_remaining -= self.VISIT_TIME
        # Store copies so revealed_info can't be silently mutated
        self.revealed_info[store_idx] = {
            'avail':   self.actual_avail[store_idx].copy(),
            'expiry':  self.expiry_dates[store_idx].copy(),
            'premium': self.price_premium[store_idx].copy(),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def get_valid_actions(self) -> List[Dict]:
        """
        Returns list of valid actions.
        Action types:
          - {'type': 'buy', 'good': g, 'brand': b}  — buy good g brand b at current store
          - {'type': 'travel', 'store': s}            — travel to store s
          - {'type': 'end'}                           — finalize shopping
        """
        if self.done:
            return []

        actions = [{'type': 'end'}]
        cur = self.current_store
        info = self.revealed_info.get(cur, {})
        avail = info.get('avail', np.zeros((self.N_GOODS, self.N_BRANDS), dtype=bool))

        # Buy actions: goods still needed, brand available at current store
        for g in self.items_needed:
            for b in range(self.N_BRANDS):
                if avail[g, b]:
                    actions.append({'type': 'buy', 'good': g, 'brand': b})

        # Travel actions: can reach store and still have time to visit
        for s in range(self.N_STORES):
            if s != cur:
                cost = self.TRAVEL_TIME + self.VISIT_TIME
                if self.time_remaining >= cost:
                    actions.append({'type': 'travel', 'store': s})

        return actions

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute action, return (next_state, reward, done, info)."""
        assert not self.done, "Episode already ended"

        reward = 0.0
        info = {}

        if action['type'] == 'end':
            self.done = True

        elif action['type'] == 'buy':
            g, b = action['good'], action['brand']
            cur = self.current_store
            # Check still available (should be — agent only chooses from valid actions)
            if g in self.items_needed and self.actual_avail[cur, g, b]:
                premium_pct = self.price_premium[cur, g, b] * 100  # convert to %
                expiry = self.expiry_dates[cur, g, b]
                pref = self.brand_preference[g, b]
                self.items_bought[g] = (b, premium_pct, expiry, pref)
                self.items_needed.discard(g)
                # Immediate reward for buying
                reward += self.W_PREMIUM * premium_pct
                reward += self.W_PREF_SCORE * pref
                reward += self.W_EXPIRY * expiry
            else:
                reward -= 1.0  # small penalty for invalid buy attempt

        elif action['type'] == 'travel':
            s = action['store']
            self.time_remaining -= self.TRAVEL_TIME
            self.current_store = s
            self._visit_store(s)
            reward += self.W_STORE_VISIT  # efficiency penalty per visit

        # Check if time ran out
        if self.time_remaining <= 0:
            self.done = True

        # Terminal reward
        if self.done:
            missing = len(self.items_needed)
            reward += self.W_MISSING_ITEM * missing
            info['missing_items'] = missing
            info['items_bought'] = len(self.items_bought)
            info['stores_visited'] = len(self.visited_stores)
            info['time_used'] = self.TOTAL_DURATION - self.time_remaining

        return self._get_state(), reward, self.done, info

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------

    def _get_state(self) -> Dict:
        return {
            'current_store': self.current_store,
            'time_remaining': self.time_remaining,
            'items_needed': frozenset(self.items_needed),
            'visited_stores': frozenset(self.visited_stores),
            'revealed_info': self.revealed_info,
        }

    def state_to_vector(self, state: Optional[Dict] = None) -> np.ndarray:
        """
        Convert state dict to flat numpy vector for function approximation.
        Features:
          - one-hot current store (6)
          - time remaining normalized (1)
          - items needed binary (20)
          - visited stores binary (6)
          - for each store×good: best available pref score, min premium, max expiry (3×6×20=360 — compressed)
        Total: 6 + 1 + 20 + 6 + 6*20*3 = 393 features
        """
        if state is None:
            state = self._get_state()

        feats = []

        # Current store one-hot
        store_oh = np.zeros(self.N_STORES)
        store_oh[state['current_store']] = 1.0
        feats.append(store_oh)

        # Time remaining (normalized, clamped to [0, 1])
        feats.append(np.array([max(0.0, state['time_remaining'] / self.TOTAL_DURATION)]))

        # Items needed
        needed_vec = np.zeros(self.N_GOODS)
        for g in state['items_needed']:
            needed_vec[g] = 1.0
        feats.append(needed_vec)

        # Visited stores
        visited_vec = np.zeros(self.N_STORES)
        for s in state['visited_stores']:
            visited_vec[s] = 1.0
        feats.append(visited_vec)

        # Per store per good: best pref score of available brand, min premium, max expiry
        store_good_feats = np.zeros((self.N_STORES, self.N_GOODS, 3))
        for s, info in state['revealed_info'].items():
            avail = info['avail']   # (N_GOODS, N_BRANDS)
            prem = info['premium']
            exp = info['expiry']
            for g in range(self.N_GOODS):
                avail_brands = np.where(avail[g])[0]
                if len(avail_brands) > 0:
                    store_good_feats[s, g, 0] = self.brand_preference[g, avail_brands].max() / self.N_BRANDS
                    store_good_feats[s, g, 1] = prem[g, avail_brands].min() / self.MAX_PREMIUM
                    store_good_feats[s, g, 2] = exp[g, avail_brands].max() / 7.0
        feats.append(store_good_feats.flatten())

        return np.concatenate(feats)

    def get_state_dim(self) -> int:
        return self.N_STORES + 1 + self.N_GOODS + self.N_STORES + self.N_STORES * self.N_GOODS * 3

    def get_action_dim(self) -> int:
        """Max possible discrete actions: end(1) + buy(N_GOODS*N_BRANDS) + travel(N_STORES)"""
        return 1 + self.N_GOODS * self.N_BRANDS + self.N_STORES

    def compute_episode_stats(self) -> Dict:
        """Compute full statistics at end of episode."""
        assert self.done, "Episode not finished"
        bought = len(self.items_bought)
        missing = self.N_GOODS - bought
        total_premium = sum(v[1] for v in self.items_bought.values()) if self.items_bought else 0.0
        avg_premium = total_premium / bought if bought > 0 else 0.0
        total_pref = sum(v[3] for v in self.items_bought.values()) if self.items_bought else 0.0
        avg_pref = total_pref / bought if bought > 0 else 0.0
        avg_expiry = sum(v[2] for v in self.items_bought.values()) / bought if bought > 0 else 0.0
        completion_rate = bought / self.N_GOODS

        return {
            'items_bought': bought,
            'missing_items': missing,
            'completion_rate': completion_rate,
            'avg_premium_pct': avg_premium,
            'total_premium_pct': total_premium,
            'avg_pref_score': avg_pref,
            'avg_expiry_days': avg_expiry,
            'stores_visited': len(self.visited_stores),
            'time_used': self.TOTAL_DURATION - self.time_remaining,
        }
