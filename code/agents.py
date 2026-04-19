"""
Action encoding and base agent utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from environment import MarketEnvironment


def encode_action(action: Dict, env: MarketEnvironment) -> int:
    """Map action dict -> integer index."""
    if action['type'] == 'end':
        return 0
    elif action['type'] == 'buy':
        # indices 1 .. N_GOODS*N_BRANDS
        return 1 + action['good'] * env.N_BRANDS + action['brand']
    elif action['type'] == 'travel':
        # indices N_GOODS*N_BRANDS+1 .. end
        return 1 + env.N_GOODS * env.N_BRANDS + action['store']
    raise ValueError(f"Unknown action: {action}")


def decode_action(idx: int, env: MarketEnvironment) -> Dict:
    """Map integer index -> action dict."""
    if idx == 0:
        return {'type': 'end'}
    buy_end = 1 + env.N_GOODS * env.N_BRANDS
    if idx < buy_end:
        idx_buy = idx - 1
        good = idx_buy // env.N_BRANDS
        brand = idx_buy % env.N_BRANDS
        return {'type': 'buy', 'good': good, 'brand': brand}
    else:
        store = idx - buy_end
        return {'type': 'travel', 'store': store}


def get_valid_action_mask(state: Dict, env: MarketEnvironment) -> np.ndarray:
    """Returns boolean mask of valid action indices."""
    mask = np.zeros(env.get_action_dim(), dtype=bool)
    for a in env.get_valid_actions():
        mask[encode_action(a, env)] = True
    return mask


class BaseAgent:
    def __init__(self, env: MarketEnvironment, seed: int = 42):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: Dict, training: bool = True) -> Dict:
        raise NotImplementedError

    def update(self, state, action, reward, next_state, done):
        pass

    def get_name(self) -> str:
        return self.__class__.__name__
