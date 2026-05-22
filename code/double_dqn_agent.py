"""
Double DQN Agent
Fix: use ONLINE network to SELECT the best action in next state,
     but TARGET network to EVALUATE its Q-value.
This prevents the overestimation bias in standard DQN.
Standard DQN: target = r + γ * max_a Q_target(s', a)
Double DQN:   target = r + γ * Q_target(s', argmax_a Q_online(s', a))
"""
import numpy as np
from collections import deque
from dqn_agent import NumpyNN
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


class DoubleDQNAgent(BaseAgent):
    def __init__(self, env, seed=42, lr=3e-4, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9997,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=200, h1=64, h2=32):
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
        self.target_net = NumpyNN(state_dim, h1, h2, action_dim, seed=seed+1)
        self.target_net.copy_weights_from(self.online_net)
        self.replay_buffer = deque(maxlen=buffer_size)

    def get_name(self): return "Double-DQN"

    def select_action(self, state, training=True):
        valid_mask = get_valid_action_mask(state, self.env)
        valid_indices = np.where(valid_mask)[0]
        if training and self.rng.random() < self.eps:
            return decode_action(int(self.rng.choice(valid_indices)), self.env)
        sv = self.env.state_to_vector(state)
        q = self.online_net.predict(sv)[0]
        q[~valid_mask] = -np.inf
        return decode_action(int(np.argmax(q)), self.env)

    def update(self, state, action, reward, next_state, done):
        sv = self.env.state_to_vector(state)
        nsv = self.env.state_to_vector(next_state)
        a_idx = encode_action(action, self.env)
        next_mask = get_valid_action_mask(next_state, self.env)
        self.replay_buffer.append((sv, a_idx, reward, nsv, done, next_mask))
        self.step_count += 1
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
        if self.step_count % self.target_update_freq == 0:
            self.target_net.copy_weights_from(self.online_net)
        if self.eps > self.eps_end:
            self.eps *= self.eps_decay

    def _train_step(self):
        idx = self.rng.integers(0, len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in idx]
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        sa = np.array(states)
        nsa = np.array(next_states)
        ra = np.array(rewards)
        da = np.array(dones, dtype=float)
        aa = np.array(actions)
        nma = np.array(next_masks)

        q_online, cache = self.online_net.forward(sa)

        # DOUBLE DQN KEY DIFFERENCE:
        # Step 1: Use ONLINE network to find best action in next state
        q_online_next = self.online_net.predict(nsa)
        q_online_next[~nma] = -np.inf
        best_actions_online = np.argmax(q_online_next, axis=1)  # chosen by online

        # Step 2: Use TARGET network to evaluate those actions
        q_target_next = self.target_net.predict(nsa)
        # Pick the Q-value of the action selected by online network
        best_q_target = q_target_next[np.arange(self.batch_size), best_actions_online]
        best_q_target[da.astype(bool)] = 0.0

        targets = ra + self.gamma * best_q_target
        q_pred = q_online[np.arange(self.batch_size), aa]
        td_errors = q_pred - targets
        dout = np.zeros_like(q_online)
        dout[np.arange(self.batch_size), aa] = td_errors
        grads = self.online_net.backward(cache, dout)
        self.online_net.apply_gradients(grads, self.lr)

    def post_episode(self): pass
