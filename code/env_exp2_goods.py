"""
Experiment 2: Restricted Store Coverage — Each store stocks only 10 goods.

Change from baseline:
  - Each store permanently stocks only 10 out of 20 goods (3 brands each).
  - The 10 goods per store are assigned with partial overlap:
      Shop 1: goods  0– 9  (goods 0–9)
      Shop 2: goods 10–19  (goods 10–19)
      Shop 3: goods  0– 4, 15–19  (overlap with shops 1 & 2)
      Shop 4: goods  5– 9, 10–14  (overlap with shops 1 & 2)
      Shop 5: goods  0– 4, 10–14  (mixed overlap)
      Shop 6: goods  5– 9, 15–19  (mixed overlap)

  This guarantees:
  - Every good is stocked in at least 2 stores (ensuring reachability)
  - No single store covers all 20 goods (forcing multi-store visits)
  - Stores have meaningful specialisation

  For non-stocked goods at a store:
  - store_stocks[s, g, :] = all False
  - No buy actions for good g available at store s
  - state_to_vector: store-good slot stays zero (no info)

Why this is harder:
  - Agents must plan coverage more carefully — a store might not have a good at all
  - Greedy/random agents that pick stores hoping for broad coverage will fail
  - Learned agents must route to the specific stores that carry missing goods
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from environment import MarketEnvironment


# Goods stocked per store (indices into range(20))
STORE_GOODS = {
    0: list(range(0, 10)),       # goods  0–9
    1: list(range(10, 20)),      # goods 10–19
    2: list(range(0, 5)) + list(range(15, 20)),   # 0–4, 15–19
    3: list(range(5, 10)) + list(range(10, 15)),  # 5–9, 10–14
    4: list(range(0, 5)) + list(range(10, 15)),   # 0–4, 10–14
    5: list(range(5, 10)) + list(range(15, 20)),  # 5–9, 15–19
}


class MarketEnvExp2(MarketEnvironment):
    """
    Experiment 2: Each store stocks only 10 goods (restricted coverage).
    Store-good assignments are deterministic per STORE_GOODS mapping above.
    """

    GOODS_PER_STORE = 10   # new constant

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)

    def _generate_static_params(self):
        """
        Override: store_stocks reflects the restricted coverage mapping.
        Each store stocks BRANDS_PER_STORE brands for each of its 10 goods only.
        """
        # Brand preferences (unchanged)
        self.brand_preference = np.zeros((self.N_GOODS, self.N_BRANDS))
        for g in range(self.N_GOODS):
            perm = self.rng.permutation(self.N_BRANDS)
            for rank, b in enumerate(perm):
                self.brand_preference[g, b] = self.N_BRANDS - rank

        # Store stocks: only goods in STORE_GOODS[s] are stocked
        self.store_stocks = np.zeros((self.N_STORES, self.N_GOODS, self.N_BRANDS), dtype=bool)
        for s in range(self.N_STORES):
            stocked_goods = STORE_GOODS[s]
            for g in stocked_goods:
                chosen = self.rng.choice(self.N_BRANDS, size=self.BRANDS_PER_STORE, replace=False)
                self.store_stocks[s, g, chosen] = True

        # Price premiums (only for stocked goods)
        self.price_premium = self.rng.uniform(
            -self.MAX_PREMIUM, self.MAX_PREMIUM,
            (self.N_STORES, self.N_GOODS, self.N_BRANDS)
        )
        self.price_premium[~self.store_stocks] = 0.0

    def get_store_goods(self, store_idx: int) -> List[int]:
        """Returns list of goods stocked at this store."""
        return STORE_GOODS[store_idx]
