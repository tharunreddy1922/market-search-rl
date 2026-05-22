"""
Proximal Policy Optimisation (PPO) Agent — NumPy implementation from scratch.

PPO is an on-policy policy gradient algorithm (Schulman et al., 2017).
Unlike Q-Learning / DQN which learn a value function and derive a policy,
PPO directly optimises the policy using gradient ascent, constrained by a
clipped surrogate objective that prevents destructively large policy updates.

Architecture:
  - Actor network  (393 → 64 → 32 → action_dim) — outputs action logits
  - Critic network (393 → 64 → 32 → 1)          — outputs state value V(s)
  Both networks are separate MLP objects (same NumPy implementation as DQN).

Key PPO ideas implemented:
  - Rollout buffer: collect N steps of experience with the current policy
  - Advantage estimation (GAE): A_t = δ_t + (γλ)δ_{t+1} + ...
  - Clipped surrogate loss: L_CLIP = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
  - Value function loss: MSE between V(s) and discounted returns
  - Entropy bonus: encourages exploration
  - Multiple update epochs per rollout (K epochs)
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask
from environment import MarketEnvironment


# ── Tiny MLP (shared with DQN) ─────────────────────────────────────────────
class MLP:
    """Two-hidden-layer MLP with He init and ReLU activations."""

    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int, rng):
        def he(fan_in, fan_out):
            return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))

        self.W1 = he(in_dim, h1);  self.b1 = np.zeros(h1)
        self.W2 = he(h1, h2);      self.b2 = np.zeros(h2)
        self.W3 = he(h2, out_dim); self.b3 = np.zeros(out_dim)
        self._cache = {}

    def forward(self, x: np.ndarray, store_cache: bool = True) -> np.ndarray:
        a1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, a1)
        a2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, a2)
        out = h2 @ self.W3 + self.b3
        if store_cache:
            self._cache = {'x': x, 'a1': a1, 'h1': h1, 'a2': a2, 'h2': h2}
        return out

    def backward(self, d_out: np.ndarray) -> Dict:
        c = self._cache
        B = d_out.shape[0]
        dW3 = c['h2'].T @ d_out / B
        db3 = d_out.mean(0)
        dh2 = d_out @ self.W3.T
        da2 = dh2 * (c['a2'] > 0)
        dW2 = c['h1'].T @ da2 / B
        db2 = da2.mean(0)
        dh1 = da2 @ self.W2.T
        da1 = dh1 * (c['a1'] > 0)
        dW1 = c['x'].T @ da1 / B
        db1 = da1.mean(0)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def apply_grads(self, grads: Dict, lr: float, clip: float = 0.5):
        """Gradient ascent with gradient clipping."""
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            g = np.clip(grads[key], -clip, clip)
            setattr(self, key, getattr(self, key) + lr * g)

    def get_params(self):
        return {k: getattr(self, k).copy() for k in ('W1','b1','W2','b2','W3','b3')}

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v.copy())


# ── Rollout Buffer ─────────────────────────────────────────────────────────
class RolloutBuffer:
    """Stores one rollout of experience for PPO update."""

    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.masks: List[np.ndarray] = []

    def add(self, state_vec, action_idx, log_prob, reward, value, done, mask):
        self.states.append(state_vec)
        self.actions.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


# ── PPO Agent ──────────────────────────────────────────────────────────────
class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimisation agent.

    Parameters
    ----------
    lr_actor  : learning rate for the actor (policy) network
    lr_critic : learning rate for the critic (value) network
    gamma     : discount factor
    lam       : GAE lambda for advantage estimation
    clip_eps  : PPO clipping parameter ε
    n_epochs  : number of update epochs per rollout
    rollout_steps : steps to collect per rollout before updating
    entropy_coef  : coefficient of entropy bonus in actor loss
    h1, h2   : hidden layer sizes
    """

    def __init__(
        self,
        env: MarketEnvironment,
        seed: int = 42,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        rollout_steps: int = 512,
        entropy_coef: float = 0.01,
        h1: int = 64,
        h2: int = 32,
    ):
        super().__init__(env, seed)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.rollout_steps = rollout_steps
        self.entropy_coef = entropy_coef
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()

        self.actor  = MLP(state_dim, h1, h2, action_dim, self.rng)
        self.critic = MLP(state_dim, h1, h2, 1, self.rng)

        self.buffer = RolloutBuffer()
        self._step_count = 0
        self._update_count = 0

        # Current episode tracking
        self._current_state_vec: Optional[np.ndarray] = None
        self._pending_action: Optional[int] = None
        self._pending_log_prob: Optional[float] = None
        self._pending_value: Optional[float] = None
        self._pending_mask: Optional[np.ndarray] = None

        # For logging
        self.eps = 0.0  # PPO doesn't use epsilon; kept for compatibility

    # ── Action selection ───────────────────────────────────────────────────
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-8)

    def select_action(self, state: Dict, training: bool = True) -> Dict:
        state_vec = self.env.state_to_vector(state)
        mask = get_valid_action_mask(state, self.env)

        # Actor forward pass → logits → masked softmax → sample
        logits = self.actor.forward(state_vec[np.newaxis, :], store_cache=False)[0]
        logits[~mask] = -1e9  # mask invalid actions
        probs = self._softmax(logits)
        probs[~mask] = 0.0
        probs /= probs.sum() + 1e-8

        if training:
            action_idx = self.rng.choice(len(probs), p=probs)
        else:
            action_idx = int(np.argmax(probs))

        log_prob = float(np.log(probs[action_idx] + 1e-8))

        # Critic value estimate
        value = float(self.critic.forward(state_vec[np.newaxis, :], store_cache=False)[0, 0])

        if training:
            self._current_state_vec = state_vec
            self._pending_action = action_idx
            self._pending_log_prob = log_prob
            self._pending_value = value
            self._pending_mask = mask

        return decode_action(action_idx, self.env)

    # ── Update (called after each step) ────────────────────────────────────
    def update(self, state, action, reward, next_state, done):
        if self._pending_action is None:
            return

        self.buffer.add(
            state_vec=self._current_state_vec,
            action_idx=self._pending_action,
            log_prob=self._pending_log_prob,
            reward=reward,
            value=self._pending_value,
            done=done,
            mask=self._pending_mask,
        )
        self._step_count += 1

        # Trigger PPO update when rollout is full
        if len(self.buffer) >= self.rollout_steps or done:
            if done:
                next_value = 0.0
            else:
                sv = self.env.state_to_vector(next_state)
                next_value = float(
                    self.critic.forward(sv[np.newaxis, :], store_cache=False)[0, 0]
                )
            self._ppo_update(next_value)
            self.buffer.clear()
            self._update_count += 1

        self._pending_action = None

    # ── GAE + PPO update ───────────────────────────────────────────────────
    def _ppo_update(self, next_value: float):
        n = len(self.buffer)
        rewards  = np.array(self.buffer.rewards,   dtype=float)
        values   = np.array(self.buffer.values,    dtype=float)
        dones    = np.array(self.buffer.dones,     dtype=float)
        log_probs_old = np.array(self.buffer.log_probs, dtype=float)
        actions  = np.array(self.buffer.actions,   dtype=int)
        states   = np.array(self.buffer.states,    dtype=float)

        # ── GAE advantage computation ──────────────────────────────────────
        advantages = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            nv = next_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── K epochs of mini-batch updates ────────────────────────────────
        indices = np.arange(n)
        for _ in range(self.n_epochs):
            self.rng.shuffle(indices)
            batch_size = min(64, n)
            for start in range(0, n, batch_size):
                idx = indices[start:start + batch_size]
                s_batch = states[idx]           # (B, state_dim)
                a_batch = actions[idx]          # (B,)
                lp_old  = log_probs_old[idx]    # (B,)
                adv     = advantages[idx]       # (B,)
                ret     = returns[idx]          # (B,)

                # ── Actor update ───────────────────────────────────────────
                logits = self.actor.forward(s_batch)   # (B, action_dim)
                # Masked softmax per sample
                probs_batch = np.zeros_like(logits)
                for i, (lgt, mask) in enumerate(
                    zip(logits, [self.buffer.masks[j] for j in idx])
                ):
                    lgt[~mask] = -1e9
                    p = self._softmax(lgt)
                    p[~mask] = 0.0
                    p /= p.sum() + 1e-8
                    probs_batch[i] = p

                # Log probs for taken actions
                lp_new = np.log(probs_batch[np.arange(len(a_batch)), a_batch] + 1e-8)

                # Entropy bonus
                entropy = -(probs_batch * np.log(probs_batch + 1e-8)).sum(1).mean()

                # Clipped surrogate
                ratio = np.exp(lp_new - lp_old)
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -np.minimum(surr1, surr2) - self.entropy_coef * entropy

                # Gradient of actor loss w.r.t. logits
                # d(loss)/d(logit_k) = sum_i d(loss_i)/d(logit_k)
                # For each sample, the gradient through softmax cross-entropy:
                d_logits = np.zeros_like(logits)
                for i, (a, p, surr_min_grad, mask) in enumerate(
                    zip(a_batch, probs_batch,
                        np.minimum(ratio, np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)) * adv,
                        [self.buffer.masks[j] for j in idx])
                ):
                    # Gradient of -ratio * adv w.r.t. log_prob_a = -adv (when not clipped)
                    # Gradient of log_prob_a w.r.t. logit_k = p_k - [k==a]
                    r = np.exp(lp_new[i] - lp_old[i])
                    clipped = (r < 1 - self.clip_eps or r > 1 + self.clip_eps)
                    if clipped:
                        grad_lp = 0.0
                    else:
                        grad_lp = -adv[i]   # gradient ascent on -loss → descent on loss
                    # Entropy gradient
                    grad_ent = -(np.log(p + 1e-8) + 1)
                    d_log_pa = np.zeros(logits.shape[1])
                    d_log_pa[a] = grad_lp
                    # Jacobian of softmax: d(log p_a)/d(logit_k) = [k==a] - p_k (valid only)
                    d_logits[i] = d_log_pa[a] * (np.eye(logits.shape[1])[a] - p)
                    d_logits[i] += self.entropy_coef * (p - np.eye(logits.shape[1])[a])
                    d_logits[i, ~mask] = 0.0

                grads_actor = self.actor.backward(-d_logits)  # ascent → negate for backward
                self.actor.apply_grads(grads_actor, self.lr_actor)

                # ── Critic update ──────────────────────────────────────────
                values_pred = self.critic.forward(s_batch)[:, 0]  # (B,)
                critic_loss = (values_pred - ret) ** 2             # MSE
                d_value = 2 * (values_pred - ret) / len(a_batch)   # (B,)
                grads_critic = self.critic.backward(d_value[:, np.newaxis])
                # Gradient descent on critic loss (negate to use apply_grads as ascent)
                grads_critic_neg = {k: -v for k, v in grads_critic.items()}
                self.critic.apply_grads(grads_critic_neg, self.lr_critic)

    def get_name(self) -> str:
        return "PPO"
