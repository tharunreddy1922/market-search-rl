"""
Oracle (Perfect-Information) Benchmark Policy
=============================================
Implements an omniscient greedy planner that knows, at episode start,
the full realisation of stochastic variables (availability, expiry dates,
price premiums).  Used to compute the theoretical maximum obtainable
reward (TMOR) against which RL agents are benchmarked.

Algorithm description
---------------------
At reset time the oracle receives the complete ground-truth arrays
  - actual_avail  [N_STORES × N_GOODS × N_BRANDS]
  - expiry_dates  [N_STORES × N_GOODS × N_BRANDS]
  - price_premium [N_STORES × N_GOODS × N_BRANDS]

It then solves a greedy set-cover plan:

  For each good g still needed:
      candidate = argmax over (s, b) of oracle_score(s, g, b)
              s.t. actual_avail[s,g,b] = True
                   reaching s is feasible within remaining time
      oracle_score = W_PREF * pref[g,b] + W_EXP * expiry[s,g,b]
                   + W_PREM * premium[s,g,b]*100  + W_VISIT * visits_to_s

  The plan produces an ordered list of (store, good, brand) triples.
  At execution time the oracle follows the plan exactly.

Pseudocode
----------
  Oracle.plan(env):
      plan ← []
      assigned ← {}          // good → (store, brand)
      time_budget ← TOTAL_DURATION

      FOR each good g in priority order (highest best-reachable score first):
          FOR each store s reachable within time_budget:
              FOR each brand b s.t. actual_avail[s,g,b]:
                  compute score(s,g,b)
          best ← (s*, g, b*) with highest score
          IF best exists AND still feasible:
              assigned[g] ← (s*, b*)
              time_budget -= travel_cost(s*) if s* not yet in plan

      plan ← group assigned by store, sort stores by visitation order
      RETURN plan

  Oracle.act(state):
      execute next action in pre-computed plan
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from environment import MarketEnvironment
from agents import BaseAgent


class OracleAgent(BaseAgent):
    """
    Perfect-information greedy oracle.

    Constructs a complete shopping plan before the first action using
    ground-truth stochastic realisations.  No learning is involved.
    The oracle serves strictly as an upper-bound benchmark.
    """

    def get_name(self) -> str:
        return "Oracle"

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self) -> List[Dict]:
        """
        Build an ordered action sequence using full ground-truth knowledge.

        Returns list of action dicts that, if executed in order, maximises
        expected reward given perfect information.
        """
        env = self.env
        W_PREF  =  env.W_PREF_SCORE          # +2.0 per pref unit
        W_EXP   =  env.W_EXPIRY              # +0.5 per expiry day
        W_PREM  =  env.W_PREMIUM             # -1.0 per % premium
        W_VISIT =  env.W_STORE_VISIT         # -2.0 per store visit

        # --- score every (store, good, brand) triple that is truly available ---
        # score[s,g,b] = pref + expiry_bonus + premium_cost
        # (missing-item penalty avoided implicitly by buying the good)
        scores = np.full((env.N_STORES, env.N_GOODS, env.N_BRANDS), -np.inf)
        for s in range(env.N_STORES):
            for g in range(env.N_GOODS):
                for b in range(env.N_BRANDS):
                    if env.actual_avail[s, g, b]:
                        prem_pct = env.price_premium[s, g, b] * 100.0
                        exp_days = env.expiry_dates[s, g, b]
                        pref     = env.brand_preference[g, b]
                        scores[s, g, b] = (W_PREF * pref
                                           + W_EXP  * exp_days
                                           + W_PREM * prem_pct)

        # --- greedy assignment: assign each good to its best (store, brand) ---
        assignment: Dict[int, Tuple[int, int]] = {}   # good → (store, brand)
        needed = set(range(env.N_GOODS))

        # Priority: goods whose best-available score is highest (most to gain)
        def best_score_for_good(g):
            s_flat = scores[:, g, :]
            if s_flat.max() == -np.inf:
                return -np.inf
            return s_flat.max()

        priority_order = sorted(needed,
                                key=best_score_for_good,
                                reverse=True)

        for g in priority_order:
            best_idx = np.unravel_index(
                np.argmax(scores[:, g, :]), (env.N_STORES, env.N_BRANDS)
            )
            s_best, b_best = int(best_idx[0]), int(best_idx[1])
            if scores[s_best, g, b_best] > -np.inf:
                # Adjust score to penalise adding a new store visit
                # (approximate: if store already assigned, no extra penalty)
                already_in = s_best in {v[0] for v in assignment.values()}
                adjusted = scores[s_best, g, b_best]
                if not already_in:
                    adjusted += W_VISIT   # penalty for this store visit
                assignment[g] = (s_best, b_best)

        # --- build feasibility-aware visit order ---
        # Group goods by store, then order stores to minimise travel
        store_goods: Dict[int, List[int]] = {}
        for g, (s, b) in assignment.items():
            store_goods.setdefault(s, []).append(g)

        # Greedy nearest-store ordering (all travel times equal → any order)
        stores_to_visit = list(store_goods.keys())
        # Start from store 0 (initial position); keep store 0 first if present
        if 0 in stores_to_visit:
            stores_to_visit.remove(0)
            stores_to_visit = [0] + stores_to_visit

        # --- simulate feasibility: drop stores we cannot reach in time ---
        time_budget = env.TOTAL_DURATION - env.VISIT_TIME  # store 0 already visited
        current_store = 0
        feasible_plan: List[Dict] = []
        visited = {0}

        for s in stores_to_visit:
            if s == 0:
                # Already at store 0 — buy goods assigned here
                for g in store_goods.get(0, []):
                    b = assignment[g][1]
                    feasible_plan.append({'type': 'buy', 'good': g, 'brand': b})
            else:
                travel_cost = env.TRAVEL_TIME + env.VISIT_TIME
                if time_budget < travel_cost:
                    break
                time_budget -= travel_cost
                feasible_plan.append({'type': 'travel', 'store': s})
                visited.add(s)
                for g in store_goods.get(s, []):
                    b = assignment[g][1]
                    feasible_plan.append({'type': 'buy', 'good': g, 'brand': b})

        feasible_plan.append({'type': 'end'})
        return feasible_plan

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: Dict, training: bool = False) -> Dict:
        """
        Execute the pre-computed plan.
        Replanning occurs if a buy action turns out invalid at runtime
        (should not happen for oracle, included for robustness).
        """
        if not hasattr(self, '_plan_queue') or len(self._plan_queue) == 0:
            # Build plan from scratch using ground-truth env state
            self._plan_queue = self._plan()

        while self._plan_queue:
            action = self._plan_queue.pop(0)

            # Validate: skip buys for goods already purchased
            if action['type'] == 'buy':
                if action['good'] not in state['items_needed']:
                    continue
                cur = state['current_store']
                info = state['revealed_info'].get(cur, {})
                avail = info.get('avail',
                                 np.zeros((self.env.N_GOODS, self.env.N_BRANDS), dtype=bool))
                if not avail[action['good'], action['brand']]:
                    continue  # brand not actually available — skip

            # Validate: skip travel if insufficient time
            if action['type'] == 'travel':
                cost = self.env.TRAVEL_TIME + self.env.VISIT_TIME
                if state['time_remaining'] < cost:
                    return {'type': 'end'}

            return action

        return {'type': 'end'}

    def update(self, state, action, reward, next_state, done):
        """Oracle does not learn."""
        pass

    def reset_plan(self):
        """Called at start of each new episode."""
        if hasattr(self, '_plan_queue'):
            del self._plan_queue
