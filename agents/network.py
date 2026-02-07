import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, activation="relu"):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation="relu"):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_sizes, action_dim, activation)

    def forward(self, state):
        logits = self.mlp(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def log_prob(self, state, action):
        dist = self.forward(state)
        return dist.log_prob(action), dist.entropy()

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(64, 64), activation="relu"):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_sizes, 1, activation)

    def forward(self, state):
        return self.mlp(state).squeeze(-1)
