"""
SARSA Agent — REVIEWED & IMPROVED
Same parameter improvements as Q-Learning:
  - lr: 0.1 -> 0.2
  - eps_end: 0.05 -> 0.02
  - eps_decay: 0.9995 -> 0.99995
  - Coarser state hash (20-min buckets, no avail_bits)
"""

import numpy as np
from collections import defaultdict
from environment import MarketEnvironment
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask
from qlearning_agent import _state_hash


class SARSAAgent(BaseAgent):
    def __init__(self, env, seed=42,
                 lr=0.2,
                 gamma=0.95,
                 eps_start=1.0,
                 eps_end=0.02,
                 eps_decay=0.99995):
        super().__init__(env, seed)
        self.lr        = lr
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(env.get_action_dim()))
        self._next_action = None

    def get_name(self): return "SARSA"

    def _hash(self, state): return _state_hash(state, self.env)

    def _eps_greedy(self, state):
        valid_mask    = get_valid_action_mask(state, self.env)
        valid_indices = np.where(valid_mask)[0]
        if self.rng.random() < self.eps:
            return int(self.rng.choice(valid_indices))
        q = self.Q[self._hash(state)].copy()
        q[~valid_mask] = -np.inf
        return int(np.argmax(q))

    def select_action(self, state, training=True):
        if training:
            if self._next_action is None:
                idx = self._eps_greedy(state)
            else:
                idx = self._next_action
                self._next_action = None
        else:
            valid_mask = get_valid_action_mask(state, self.env)
            q = self.Q[self._hash(state)].copy()
            q[~valid_mask] = -np.inf
            idx = int(np.argmax(q))
        return decode_action(idx, self.env)

    def update(self, state, action, reward, next_state, done):
        s = self._hash(state)
        a = encode_action(action, self.env)
        if done:
            target = reward
        else:
            next_a_idx    = self._eps_greedy(next_state)
            self._next_action = next_a_idx
            target = reward + self.gamma * self.Q[self._hash(next_state)][next_a_idx]
        self.Q[s][a] += self.lr * (target - self.Q[s][a])
        if self.eps > self.eps_end:
            self.eps *= self.eps_decay

    def post_episode(self):
        self._next_action = None
