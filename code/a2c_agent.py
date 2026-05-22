"""
A2C Agent — Advantage Actor-Critic (Mnih et al., 2016)

Combines policy gradient (actor) with value function (critic):
  - Actor:  pi(a|s; theta) — outputs action probabilities
  - Critic: V(s; phi)      — estimates state value

Key advantage over REINFORCE:
  - Critic provides a bootstrapped baseline at every step (not just episode end)
  - Lower variance than Monte Carlo returns
  - No need to wait until episode end for updates (online updates)

Update rules:
  Actor loss:  L_actor  = -E[log pi(a|s) * A(s,a)]
  Critic loss: L_critic = MSE(V(s), r + gamma*V(s'))
  Entropy:     L_entropy = -sum(pi * log pi)  [encourages exploration]
  Total:       L = L_actor + c1*L_critic - c2*L_entropy
"""

import numpy as np
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


class MLP:
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

    def apply_grads_adam(self, grads, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for key in ['W1','b1','W2','b2','W3','b3']:
            g = grads[key]
            m[key] = beta1*m[key] + (1-beta1)*g
            v[key] = beta2*v[key] + (1-beta2)*g**2
            m_hat = m[key]/(1-beta1**t)
            v_hat = v[key]/(1-beta2**t)
            getattr(self, key).__iadd__(lr * m_hat/(np.sqrt(v_hat)+eps))


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -30, 30))
    return e / (e.sum() + 1e-10)


class A2CAgent(BaseAgent):
    """
    Online A2C: update actor and critic at every step using TD error as advantage.
    A(s,a) = r + gamma*V(s') - V(s)
    """
    def __init__(self, env, seed=42, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.95, entropy_coef=0.01, h1=64, h2=32):
        super().__init__(env, seed)
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.entropy_coef = entropy_coef

        rng = np.random.default_rng(seed)
        state_dim = env.get_state_dim()
        action_dim = env.get_action_dim()

        self.actor  = MLP(state_dim, h1, h2, action_dim, rng)
        self.critic = MLP(state_dim, h1, h2, 1, rng)

        # Adam optimizer states for actor
        self.actor_m  = {k: np.zeros_like(getattr(self.actor,  k)) for k in ['W1','b1','W2','b2','W3','b3']}
        self.actor_v  = {k: np.zeros_like(getattr(self.actor,  k)) for k in ['W1','b1','W2','b2','W3','b3']}
        self.critic_m = {k: np.zeros_like(getattr(self.critic, k)) for k in ['W1','b1','W2','b2','W3','b3']}
        self.critic_v = {k: np.zeros_like(getattr(self.critic, k)) for k in ['W1','b1','W2','b2','W3','b3']}
        self.t = 1  # Adam step counter

        self._last = None  # (s_vec, action_idx, probs)

    def get_name(self): return "A2C"

    def select_action(self, state, training=True):
        env = self.env
        s_vec = env.state_to_vector(state)
        logits = self.actor.forward(s_vec)
        mask = get_valid_action_mask(state, env)
        logits[~mask] = -1e9
        probs = softmax(logits)

        if training:
            valid_indices = np.where(mask)[0]
            valid_probs = probs[valid_indices]
            valid_probs = valid_probs / (valid_probs.sum() + 1e-10)
            idx = self.rng.choice(valid_indices, p=valid_probs)
            self._last = (s_vec, idx, probs)
        else:
            idx = int(np.argmax(probs * mask))

        return decode_action(idx, env)

    def update(self, state, action, reward, next_state, done):
        if self._last is None:
            return
        s_vec, action_idx, probs = self._last

        # Critic: compute V(s) and V(s')
        v_s = self.critic.forward(s_vec)[0]
        if done:
            v_s_next = 0.0
        else:
            s_next_vec = self.env.state_to_vector(next_state)
            v_s_next = self.critic.forward(s_next_vec, store_cache=False)[0]

        # TD error = advantage estimate
        td_target = reward + self.gamma * v_s_next
        td_error = td_target - v_s  # A(s,a) approximation

        # ── Critic update ──────────────────────────────────────────────
        _ = self.critic.forward(s_vec)  # refresh cache
        grad_critic_out = np.array([2.0 * (v_s - td_target)])  # MSE gradient
        critic_grads = self.critic.backward(grad_critic_out)
        # Negate for gradient descent (minimise loss)
        critic_grads = {k: -v for k, v in critic_grads.items()}
        self.critic.apply_grads_adam(critic_grads, self.lr_critic,
                                     self.critic_m, self.critic_v, self.t)

        # ── Actor update ───────────────────────────────────────────────
        _ = self.actor.forward(s_vec)  # refresh cache
        # grad log pi(a|s) = (1[a==idx] - pi(a|s))
        grad_log = -probs.copy()
        grad_log[action_idx] += 1.0
        # Entropy gradient: -sum(pi * log pi) -> grad = -(log pi + 1)
        log_probs = np.log(probs + 1e-10)
        entropy_grad = -(log_probs + 1.0)

        # Actor gradient: ascend policy gradient + entropy bonus
        actor_grad_out = self.lr_actor * (td_error * grad_log + self.entropy_coef * entropy_grad)
        actor_grads = self.actor.backward(actor_grad_out)
        self.actor.apply_grads_adam(actor_grads, 1.0,  # lr already in grad
                                    self.actor_m, self.actor_v, self.t)

        self.t += 1
        self._last = None

    def post_episode(self):
        self._last = None
