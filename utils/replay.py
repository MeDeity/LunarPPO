import torch
import numpy as np

class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, gamma, gae_lambda):
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32)
        dones = torch.as_tensor(self.dones, dtype=torch.float32)
        values = torch.as_tensor(self.values, dtype=torch.float32)
        advantages = torch.zeros_like(rewards)
        last_adv = 0.0
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            last_adv = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_adv
            advantages[t] = last_adv
        targets = advantages + values
        return advantages[:-1], targets[:-1]

    def as_tensors(self):
        states_np = np.asarray(self.states[:-1], dtype=np.float32)
        actions_np = np.asarray(self.actions[:-1], dtype=np.int64)
        log_probs_np = np.asarray(self.log_probs[:-1], dtype=np.float32)
        states = torch.as_tensor(states_np, dtype=torch.float32)
        actions = torch.as_tensor(actions_np, dtype=torch.int64)
        old_log_probs = torch.as_tensor(log_probs_np, dtype=torch.float32)
        return states, actions, old_log_probs

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
