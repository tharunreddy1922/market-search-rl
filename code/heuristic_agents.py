"""
Multiple Heuristic Agents for comparison.
Heuristic agents use fixed rules — no learning, no training.

1. Heuristic —  Buy if preference >= 3 (top 3 of 5) AND expiry < 4 days.
    Travel to store with highest sum of brand ratings for remaining goods.
2. FreshnessFirst     — prioritise freshest items with good brands
3. BestDealFirst      — always buy cheapest available preferred brand
4. TimeAware          — adjusts strategy based on remaining time
"""
import numpy as np
from agents import BaseAgent


class HeuristicAgent(BaseAgent):
    """
    Buy if preference >= 3 (top 3 of 5) AND expiry < 4 days.
    Travel to store with highest sum of brand ratings for remaining goods.
    """
    def get_name(self): return "Heuristic"

    def select_action(self, state, training=True):
        env = self.env
        cur = state['current_store']
        needed = state['items_needed']
        time_rem = state['time_remaining']
        info = state['revealed_info'].get(cur, {})
        avail = info.get('avail', np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        expiry = info.get('expiry', np.zeros((env.N_GOODS, env.N_BRANDS)))
        pref = env.brand_preference

        # Buy rule: preference >= 3 AND expiry < 4
        best_buy, best_score = None, -np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if avail[g,b] and pref[g,b] >= 3.0 and expiry[g,b] < 4.0:
                    s = pref[g,b] * 2.0 - expiry[g,b] * 0.3
                    if s > best_score:
                        best_score, best_buy = s, {'type':'buy','good':g,'brand':b}
        if best_buy: return best_buy

        # Travel: highest sum of best brand ratings for remaining goods
        tc = env.TRAVEL_TIME + env.VISIT_TIME
        best_store, best_ss = None, -np.inf
        for s in range(env.N_STORES):
            if s == cur or time_rem < tc: continue
            ss = sum(pref[g, np.where(env.store_stocks[s,g])[0]].max()
                     for g in needed if env.store_stocks[s,g].any())
            if s not in state['visited_stores']: ss += 1.0
            if ss > best_ss: best_ss, best_store = ss, s
        if best_store is not None and best_ss > 0:
            return {'type':'travel','store':best_store}
        return {'type':'end'}


class FreshnessFirstAgent(BaseAgent):
    """
    Prioritise buying items with the shortest expiry date first
    (use-it-or-lose-it logic), as long as brand preference is acceptable (>=2).
    """
    def get_name(self): return "Freshness-First"

    def select_action(self, state, training=True):
        env = self.env
        cur = state['current_store']
        needed = state['items_needed']
        time_rem = state['time_remaining']
        info = state['revealed_info'].get(cur, {})
        avail = info.get('avail', np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        expiry = info.get('expiry', np.zeros((env.N_GOODS, env.N_BRANDS)))
        pref = env.brand_preference

        # Buy: pick item with LOWEST expiry (most urgent) with acceptable brand (>=2)
        best_buy, best_score = None, np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if avail[g,b] and pref[g,b] >= 2.0:
                    # Lower expiry = more urgent = higher priority
                    score = expiry[g,b] - pref[g,b] * 0.1  # expiry dominant
                    if score < best_score:
                        best_score, best_buy = score, {'type':'buy','good':g,'brand':b}
        if best_buy: return best_buy

        # Travel to nearest unvisited store
        tc = env.TRAVEL_TIME + env.VISIT_TIME
        for s in range(env.N_STORES):
            if s != cur and s not in state['visited_stores'] and time_rem >= tc:
                return {'type':'travel','store':s}
        # If all visited, go to store with most unmet needs
        best_store, best_ss = None, -np.inf
        for s in range(env.N_STORES):
            if s == cur or time_rem < tc: continue
            ss = sum(1 for g in needed if env.store_stocks[s,g].any())
            if ss > best_ss: best_ss, best_store = ss, s
        if best_store: return {'type':'travel','store':best_store}
        return {'type':'end'}


class BestDealAgent(BaseAgent):
    """
    Always buy the combination of (good, brand) that gives
    the highest preference score at the LOWEST price premium.
    Purely economically rational — gets best value every purchase.
    """
    def get_name(self): return "Best-Deal"

    def select_action(self, state, training=True):
        env = self.env
        cur = state['current_store']
        needed = state['items_needed']
        time_rem = state['time_remaining']
        info = state['revealed_info'].get(cur, {})
        avail = info.get('avail', np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        pref = env.brand_preference
        premium = info.get('premium', np.zeros((env.N_GOODS, env.N_BRANDS)))

        # Buy: maximise preference, minimise premium
        best_buy, best_score = None, -np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if avail[g,b]:
                    # High pref good, low premium also good
                    score = pref[g,b] * 3.0 - premium[g,b] * 100
                    if score > best_score:
                        best_score, best_buy = score, {'type':'buy','good':g,'brand':b}
        if best_buy: return best_buy

        # Travel to store with best brand ratings AND lowest premiums
        tc = env.TRAVEL_TIME + env.VISIT_TIME
        best_store, best_ss = None, -np.inf
        for s in range(env.N_STORES):
            if s == cur or time_rem < tc: continue
            ss = 0.0
            for g in needed:
                stocked = np.where(env.store_stocks[s,g])[0]
                if len(stocked) > 0:
                    best_pref = pref[g,stocked].max()
                    avg_prem = env.price_premium[s,g,stocked].mean()
                    ss += best_pref * 2.0 - avg_prem * 50
            if ss > best_ss: best_ss, best_store = ss, s
        if best_store: return {'type':'travel','store':best_store}
        return {'type':'end'}


class TimeAwareAgent(BaseAgent):
    """
    Adjusts strategy based on time pressure:
    - Lots of time: be picky (only buy top 2 brands, expiry >= 3)
    - Medium time: moderate (top 3 brands, expiry >= 2)
    - Low time: desperate (buy anything available to complete the list)
    """
    def get_name(self): return "Time-Aware"

    def select_action(self, state, training=True):
        env = self.env
        cur = state['current_store']
        needed = state['items_needed']
        time_rem = state['time_remaining']
        info = state['revealed_info'].get(cur, {})
        avail = info.get('avail', np.zeros((env.N_GOODS, env.N_BRANDS), dtype=bool))
        expiry = info.get('expiry', np.zeros((env.N_GOODS, env.N_BRANDS)))
        pref = env.brand_preference
        tc = env.TRAVEL_TIME + env.VISIT_TIME

        # Adjust thresholds based on time remaining
        time_fraction = time_rem / env.TOTAL_DURATION
        if time_fraction > 0.6:       # plenty of time — be very picky
            pref_thresh, exp_thresh = 4.0, 3.0
        elif time_fraction > 0.3:     # medium time — moderate
            pref_thresh, exp_thresh = 3.0, 2.0
        else:                          # low time — buy anything
            pref_thresh, exp_thresh = 1.0, 0.0

        best_buy, best_score = None, -np.inf
        for g in needed:
            for b in range(env.N_BRANDS):
                if avail[g,b] and pref[g,b] >= pref_thresh and expiry[g,b] >= exp_thresh:
                    score = pref[g,b] * 2.0 + expiry[g,b] * 0.5
                    if score > best_score:
                        best_score, best_buy = score, {'type':'buy','good':g,'brand':b}
        if best_buy: return best_buy

        # Travel to most promising store
        best_store, best_ss = None, -np.inf
        for s in range(env.N_STORES):
            if s == cur or time_rem < tc: continue
            ss = sum(pref[g, np.where(env.store_stocks[s,g])[0]].max()
                     for g in needed if env.store_stocks[s,g].any())
            if s not in state['visited_stores']: ss += 2.0
            if ss > best_ss: best_ss, best_store = ss, s
        if best_store: return {'type':'travel','store':best_store}
        return {'type':'end'}
