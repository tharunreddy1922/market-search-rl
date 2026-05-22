

import numpy as np
from collections import defaultdict
from typing import Dict
from environment import MarketEnvironment
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


def _state_hash(state: Dict, env: MarketEnvironment) -> tuple:
    needed_mask  = int(sum(1 << g for g in state['items_needed']))
    visited_mask = int(sum(1 << s for s in state['visited_stores']))
    time_bucket  = int(state['time_remaining'] // 10)   # 10-min buckets: finer time resolution
    return (state['current_store'], time_bucket, needed_mask, visited_mask)


class QLearningAgent(BaseAgent):
    def __init__(self, env, seed=42,
                 lr=0.2,
                 gamma=0.95,
                 eps_start=1.0,
                 eps_end=0.02,
                 eps_decay=0.999975):
        super().__init__(env, seed)
        self.lr        = lr
        self.gamma     = gamma
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(env.get_action_dim()))

    def get_name(self): return "Q-Learning"

    def _hash(self, state): return _state_hash(state, self.env)

    def select_action(self, state, training=True):
        valid_mask   = get_valid_action_mask(state, self.env)
        valid_indices = np.where(valid_mask)[0]
        if training and self.rng.random() < self.eps:
            idx = self.rng.choice(valid_indices)
        else:
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
            next_valid = get_valid_action_mask(next_state, self.env)
            q_next = self.Q[self._hash(next_state)].copy()
            q_next[~next_valid] = -np.inf
            target = reward + self.gamma * np.max(q_next)
        self.Q[s][a] += self.lr * (target - self.Q[s][a])
        if self.eps > self.eps_end:
            self.eps *= self.eps_decay

    def post_episode(self): pass
