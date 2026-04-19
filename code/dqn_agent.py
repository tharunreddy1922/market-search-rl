"""
Deep Q-Network (DQN) agent.
Neural network implemented with pure NumPy (no external DL framework).
Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
With: experience replay, target network, epsilon-greedy.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple
from environment import MarketEnvironment
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


# -----------------------------------------------------------------------
# Minimal Neural Network (NumPy)
# -----------------------------------------------------------------------

class NumpyNN:
    """2-hidden-layer MLP: input -> h1 -> h2 -> output, ReLU activations."""

    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He initialisation
        self.W1 = rng.standard_normal((in_dim, h1)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = rng.standard_normal((h2, out_dim)) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass. x shape: (batch, in_dim) or (in_dim,)"""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        a1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, a1)
        a2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, a2)
        out = h2 @ self.W3 + self.b3
        cache = {'x': x, 'a1': a1, 'h1': h1, 'a2': a2, 'h2': h2}
        return out, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        out, _ = self.forward(x)
        return out

    def backward(self, cache: dict, dout: np.ndarray) -> dict:
        """Compute gradients. dout: (batch, out_dim)"""
        x, a1, h1, a2, h2 = cache['x'], cache['a1'], cache['h1'], cache['a2'], cache['h2']
        batch = x.shape[0]
        dW3 = h2.T @ dout / batch
        db3 = dout.mean(axis=0)
        dh2 = dout @ self.W3.T
        da2 = dh2 * (a2 > 0)
        dW2 = h1.T @ da2 / batch
        db2 = da2.mean(axis=0)
        dh1 = da2 @ self.W2.T
        da1 = dh1 * (a1 > 0)
        dW1 = x.T @ da1 / batch
        db1 = da1.mean(axis=0)
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}

    def apply_gradients(self, grads: dict, lr: float):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']
        self.W3 -= lr * grads['dW3']
        self.b3 -= lr * grads['db3']

    def copy_weights_from(self, other: 'NumpyNN'):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# -----------------------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """DQN with experience replay and target network."""

    def __init__(self, env: MarketEnvironment, seed: int = 42,
                 lr: float = 3e-4, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.9997,
                 buffer_size: int = 10000, batch_size: int = 64,
                 target_update_freq: int = 200, h1: int = 64, h2: int = 32):
        super().__init__(env, seed)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()

        self.online_net = NumpyNN(state_dim, h1, h2, action_dim, seed=seed)
        self.target_net = NumpyNN(state_dim, h1, h2, action_dim, seed=seed + 1)
        self.target_net.copy_weights_from(self.online_net)

        self.replay_buffer: deque = deque(maxlen=buffer_size)

    def get_name(self) -> str:
        return "DQN"

    def select_action(self, state: Dict, training: bool = True) -> Dict:
        valid_mask = get_valid_action_mask(state, self.env)
        valid_indices = np.where(valid_mask)[0]

        if training and self.rng.random() < self.eps:
            idx = int(self.rng.choice(valid_indices))
        else:
            sv = self.env.state_to_vector(state)
            q_vals = self.online_net.predict(sv)[0]
            q_vals[~valid_mask] = -np.inf
            idx = int(np.argmax(q_vals))

        return decode_action(idx, self.env)

    def update(self, state: Dict, action: Dict, reward: float, next_state: Dict, done: bool):
        sv = self.env.state_to_vector(state)
        nsv = self.env.state_to_vector(next_state)
        a_idx = encode_action(action, self.env)
        self.replay_buffer.append((sv, a_idx, reward, nsv, done,
                                   get_valid_action_mask(next_state, self.env)))
        self.step_count += 1

        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

        if self.step_count % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)

        if self.eps > self.eps_end:
            self.eps *= self.eps_decay

    def _train_step(self):
        indices = self.rng.integers(0, len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        states_arr = np.array(states)           # (B, state_dim)
        next_states_arr = np.array(next_states)
        rewards_arr = np.array(rewards)
        dones_arr = np.array(dones, dtype=float)
        actions_arr = np.array(actions)

        # Current Q values
        q_online, cache = self.online_net.forward(states_arr)

        # Target Q values (using target network)
        q_target_next = self.target_net.predict(next_states_arr)  # (B, action_dim)

        # Mask invalid actions in next states
        next_masks_arr = np.array(next_masks)  # (B, action_dim)
        q_target_next[~next_masks_arr] = -np.inf

        max_q_next = np.max(q_target_next, axis=1)
        max_q_next[dones_arr.astype(bool)] = 0.0

        targets = rewards_arr + self.gamma * max_q_next  # (B,)

        # Compute TD error and gradient
        q_pred = q_online[np.arange(self.batch_size), actions_arr]  # (B,)
        td_errors = q_pred - targets  # (B,)

        dout = np.zeros_like(q_online)
        dout[np.arange(self.batch_size), actions_arr] = td_errors  # MSE gradient

        grads = self.online_net.backward(cache, dout)
        self.online_net.apply_gradients(grads, self.lr)

    def post_episode(self):
        pass
