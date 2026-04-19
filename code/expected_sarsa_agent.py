"""
Expected SARSA Agent
Improvement over SARSA:
  - Standard SARSA uses the actual next action chosen (random or greedy)
  - Expected SARSA uses the EXPECTED Q-value across all actions weighted by their probability
  - This reduces variance in the update — smoother, more stable learning
  - Formula: target = r + γ × Σ_a [ π(a|s') × Q(s', a) ]
    where π(a|s') = ε/|valid| for random actions, (1-ε) + ε/|valid| for greedy action
"""
import numpy as np
from collections import defaultdict
from environment import MarketEnvironment
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask
from qlearning_agent import _state_hash


class ExpectedSARSAAgent(BaseAgent):
    def __init__(self, env, seed=42, lr=0.2, gamma=0.95,
                 eps_start=1.0, eps_end=0.02, eps_decay=0.999975):
        super().__init__(env, seed)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(env.get_action_dim()))

    def get_name(self): return "Expected-SARSA"

    def _hash(self, state): return _state_hash(state, self.env)

    def select_action(self, state, training=True):
        valid_mask = get_valid_action_mask(state, self.env)
        valid_indices = np.where(valid_mask)[0]
        if training and self.rng.random() < self.eps:
            return decode_action(int(self.rng.choice(valid_indices)), self.env)
        q = self.Q[self._hash(state)].copy()
        q[~valid_mask] = -np.inf
        return decode_action(int(np.argmax(q)), self.env)

    def _expected_q(self, state):
        """Compute E[Q(s,a)] under epsilon-greedy policy."""
        valid_mask = get_valid_action_mask(state, self.env)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        if n_valid == 0:
            return 0.0

        q = self.Q[self._hash(state)].copy()
        q[~valid_mask] = -np.inf
        best_a = int(np.argmax(q))

        # Expected Q = prob of random × average Q of valid actions
        #            + prob of greedy × Q of best action
        eps = self.eps
        expected = 0.0
        for a in valid_indices:
            prob = eps / n_valid  # random exploration probability
            if a == best_a:
                prob += (1.0 - eps)  # extra greedy probability
            expected += prob * self.Q[self._hash(state)][a]
        return expected

    def update(self, state, action, reward, next_state, done):
        s = self._hash(state)
        a = encode_action(action, self.env)

        if done:
            target = reward
        else:
            # KEY DIFFERENCE: use expected Q-value instead of specific action Q-value
            target = reward + self.gamma * self._expected_q(next_state)

        self.Q[s][a] += self.lr * (target - self.Q[s][a])
        if self.eps > self.eps_end:
            self.eps *= self.eps_decay

    def post_episode(self): pass
