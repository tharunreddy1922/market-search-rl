"""
Dueling DQN Agent
Splits the network output into two streams:
  V(s)        = value of being in state s (scalar)
  A(s,a)      = advantage of taking action a in state s (vector)
  Q(s,a)      = V(s) + A(s,a) - mean(A(s,:))

Why? The agent can learn how good a state is WITHOUT needing to
know the effect of each action. Very useful when many actions
have similar values (e.g. different buy options at the same store).
"""
import numpy as np
from collections import deque
from agents import BaseAgent, encode_action, decode_action, get_valid_action_mask


class DuelingNN:
    """Neural net with two output heads: Value and Advantage."""
    def __init__(self, in_dim, h1, h2, action_dim, seed=42):
        rng = np.random.default_rng(seed)
        s = lambda fan: np.sqrt(2.0/fan)
        # Shared layers
        self.W1 = rng.standard_normal((in_dim, h1)) * s(in_dim)
        self.b1 = np.zeros(h1)
        self.W2 = rng.standard_normal((h1, h2)) * s(h1)
        self.b2 = np.zeros(h2)
        # Value stream: h2 -> 1
        self.Wv = rng.standard_normal((h2, 1)) * s(h2)
        self.bv = np.zeros(1)
        # Advantage stream: h2 -> action_dim
        self.Wa = rng.standard_normal((h2, action_dim)) * s(h2)
        self.ba = np.zeros(action_dim)

    def predict(self, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        V = h2 @ self.Wv + self.bv          # (batch, 1)
        A = h2 @ self.Wa + self.ba          # (batch, actions)
        # Q = V + A - mean(A)  — subtracting mean stabilises training
        Q = V + A - A.mean(axis=1, keepdims=True)
        return Q

    def forward(self, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        a1 = x @ self.W1 + self.b1; h1 = np.maximum(0, a1)
        a2 = h1 @ self.W2 + self.b2; h2 = np.maximum(0, a2)
        V = h2 @ self.Wv + self.bv
        A = h2 @ self.Wa + self.ba
        Q = V + A - A.mean(axis=1, keepdims=True)
        cache = {'x':x,'a1':a1,'h1':h1,'a2':a2,'h2':h2,'V':V,'A':A}
        return Q, cache

    def backward(self, cache, dQ):
        x,a1,h1,a2,h2 = cache['x'],cache['a1'],cache['h1'],cache['a2'],cache['h2']
        A = cache['A']
        B = dQ.shape[0]
        n = dQ.shape[1]
        # dQ/dA = I - 1/n (ones matrix)  →  dA = dQ * (I - 1/n)
        dA = dQ - dQ.mean(axis=1, keepdims=True)
        dV = dQ.sum(axis=1, keepdims=True)
        dWa = h2.T @ dA / B; dba = dA.mean(axis=0)
        dWv = h2.T @ dV / B; dbv = dV.mean(axis=0)
        dh2 = dA @ self.Wa.T + dV @ self.Wv.T
        da2 = dh2 * (a2 > 0)
        dW2 = h1.T @ da2 / B; db2 = da2.mean(axis=0)
        dh1 = da2 @ self.W2.T
        da1 = dh1 * (a1 > 0)
        dW1 = x.T @ da1 / B; db1 = da1.mean(axis=0)
        return {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2,'dWv':dWv,'dbv':dbv,'dWa':dWa,'dba':dba}

    def apply_gradients(self, g, lr):
        self.W1-=lr*g['dW1']; self.b1-=lr*g['db1']
        self.W2-=lr*g['dW2']; self.b2-=lr*g['db2']
        self.Wv-=lr*g['dWv']; self.bv-=lr*g['dbv']
        self.Wa-=lr*g['dWa']; self.ba-=lr*g['dba']

    def copy_weights_from(self, other):
        for attr in ['W1','b1','W2','b2','Wv','bv','Wa','ba']:
            setattr(self, attr, getattr(other, attr).copy())


class DuelingDQNAgent(BaseAgent):
    def __init__(self, env, seed=42, lr=3e-4, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9997,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=200, h1=64, h2=32):
        super().__init__(env, seed)
        self.lr=lr; self.gamma=gamma; self.eps=eps_start
        self.eps_end=eps_end; self.eps_decay=eps_decay
        self.batch_size=batch_size; self.target_update_freq=target_update_freq
        self.step_count=0
        sd=env.get_state_dim(); ad=env.get_action_dim()
        self.online_net = DuelingNN(sd,h1,h2,ad,seed=seed)
        self.target_net = DuelingNN(sd,h1,h2,ad,seed=seed+1)
        self.target_net.copy_weights_from(self.online_net)
        self.replay_buffer = deque(maxlen=buffer_size)

    def get_name(self): return "Dueling-DQN"

    def select_action(self, state, training=True):
        vm = get_valid_action_mask(state, self.env)
        vi = np.where(vm)[0]
        if training and self.rng.random() < self.eps:
            return decode_action(int(self.rng.choice(vi)), self.env)
        sv = self.env.state_to_vector(state)
        q = self.online_net.predict(sv)[0]
        q[~vm] = -np.inf
        return decode_action(int(np.argmax(q)), self.env)

    def update(self, state, action, reward, next_state, done):
        sv=self.env.state_to_vector(state); nsv=self.env.state_to_vector(next_state)
        ai=encode_action(action,self.env); nm=get_valid_action_mask(next_state,self.env)
        self.replay_buffer.append((sv,ai,reward,nsv,done,nm))
        self.step_count+=1
        if len(self.replay_buffer)>=self.batch_size: self._train_step()
        if self.step_count%self.target_update_freq==0:
            self.target_net.copy_weights_from(self.online_net)
        if self.eps>self.eps_end: self.eps*=self.eps_decay

    def _train_step(self):
        idx=self.rng.integers(0,len(self.replay_buffer),size=self.batch_size)
        batch=[self.replay_buffer[i] for i in idx]
        sa,aa,ra,nsa,da,nma=zip(*batch)
        sa=np.array(sa); nsa=np.array(nsa); ra=np.array(ra)
        da=np.array(da,dtype=float); aa=np.array(aa); nma=np.array(nma)
        q_online,cache=self.online_net.forward(sa)
        qt=self.target_net.predict(nsa); qt[~nma]=-np.inf
        mq=np.max(qt,axis=1); mq[da.astype(bool)]=0.0
        targets=ra+self.gamma*mq
        qp=q_online[np.arange(self.batch_size),aa]
        td=qp-targets
        dout=np.zeros_like(q_online); dout[np.arange(self.batch_size),aa]=td
        grads=self.online_net.backward(cache,dout)
        self.online_net.apply_gradients(grads,self.lr)

    def post_episode(self): pass
