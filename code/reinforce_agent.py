

import numpy as np
from collections import deque
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


class MLP:
    """Two-hidden-layer MLP with He init."""
    def __init__(self, in_dim, h1, h2, out_dim, rng):
        def he(fan_in, fan_out):
            return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
        self.W1 = he(in_dim, h1); self.b1 = np.zeros(h1)
        self.W2 = he(h1, h2);     self.b2 = np.zeros(h2)
        self.W3 = he(h2, out_dim); self.b3 = np.zeros(out_dim)
        self._cache = {}

    def forward(self, x, store_cache=True):
        a1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, a1)
        a2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, a2)
        out = h2 @ self.W3 + self.b3
        if store_cache:
            self._cache = {'x': x, 'h1': h1, 'h2': h2, 'a1': a1, 'a2': a2}
        return out

    def backward(self, grad_out):
        c = self._cache
        dW3 = c['h2'].reshape(-1,1) @ grad_out.reshape(1,-1)
        db3 = grad_out
        dh2 = grad_out @ self.W3.T
        da2 = dh2 * (c['a2'] > 0)
        dW2 = c['h1'].reshape(-1,1) @ da2.reshape(1,-1)
        db2 = da2
        dh1 = da2 @ self.W2.T
        da1 = dh1 * (c['a1'] > 0)
        dW1 = c['x'].reshape(-1,1) @ da1.reshape(1,-1)
        db1 = da1
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def apply_grads(self, grads, lr):
        self.W1 += lr * grads['W1']; self.b1 += lr * grads['b1']
        self.W2 += lr * grads['W2']; self.b2 += lr * grads['b2']
        self.W3 += lr * grads['W3']; self.b3 += lr * grads['b3']


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -30, 30))
    return e / (e.sum() + 1e-10)


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE with baseline variance reduction.
    Collects full episode trajectory, then updates at episode end.
    """
    def __init__(self, env, seed=42, lr=1e-3, gamma=0.95, h1=64, h2=32):
        super().__init__(env, seed)
        self.lr = lr
        self.gamma = gamma
        rng = np.random.default_rng(seed)
        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()
        self.policy = MLP(state_dim, h1, h2, action_dim, rng)
        # Running baseline (mean return)
        self.baseline = 0.0
        self.baseline_alpha = 0.01
        # Trajectory buffer
        self._traj = []

    def get_name(self): return "REINFORCE"

    def select_action(self, state, training=True):
        env = self.env
        s_vec = env.state_to_vector(state)
        logits = self.policy.forward(s_vec)
        mask = get_valid_action_mask(state, env)
        logits[~mask] = -1e9
        probs = softmax(logits)

        if training:
            valid_indices = np.where(mask)[0]
            valid_probs = probs[valid_indices]
            valid_probs = valid_probs / valid_probs.sum()
            idx = self.rng.choice(valid_indices, p=valid_probs)
        else:
            idx = np.argmax(probs * mask)

        action = decode_action(idx, env)
        if training:
            self._traj.append({'s_vec': s_vec, 'action_idx': idx, 'probs': probs.copy()})
        return action

    def update(self, state, action, reward, next_state, done):
        if self._traj:
            self._traj[-1]['reward'] = reward

    def post_episode(self):
        """Called at episode end — perform REINFORCE update."""
        if len(self._traj) == 0:
            return
        # Ensure all steps have reward
        for t in self._traj:
            if 'reward' not in t:
                t['reward'] = 0.0

        T = len(self._traj)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._traj[t]['reward'] + self.gamma * G
            returns[t] = G

        # Per-episode advantage normalisation (standard practice, lower variance)
        # Subtract episode mean and divide by episode std — no slow EMA needed
        adv_mean = returns.mean()
        adv_std  = returns.std() + 1e-8
        advantages = (returns - adv_mean) / adv_std

        # Policy gradient update
        for t in range(T):
            s_vec = self._traj[t]['s_vec']
            idx = self._traj[t]['action_idx']
            adv = advantages[t]
            probs = self._traj[t]['probs']

            # Recompute logits for gradient
            _ = self.policy.forward(s_vec)
            # grad log pi(a|s) = (1[a] - pi(a|s)) for softmax
            grad_log = -probs.copy()
            grad_log[idx] += 1.0
            # Policy gradient: ascend alpha * adv * grad_log_pi
            grad_out = self.lr * adv * grad_log
            grads = self.policy.backward(grad_out)
            self.policy.apply_grads(grads, 1.0)  # lr already in grad_out

        self._traj = []
