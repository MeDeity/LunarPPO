import torch
import torch.nn as nn

# 定义 PPO 中使用的多层感知机、Actor 策略网络和 Critic 价值网络

class MLP(nn.Module):
    # 通用多层感知机，用于构建 Actor 和 Critic 的主干网络
    def __init__(self, input_dim, hidden_sizes, output_dim, activation="relu"):
        # input_dim: 输入特征的维度，例如状态向量的长度
        # hidden_sizes: 每一层隐藏层神经元个数的列表，例如 [64, 64]
        # output_dim: 输出特征的维度，例如动作数量或价值输出维度
        # activation: 隐藏层激活函数，支持 "relu" 或 "tanh"
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
        # 输出层不加激活函数，由上层网络决定如何使用
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: 输入张量，一般形状为 [batch_size, input_dim]
        return self.net(x)

class ActorNetwork(nn.Module):
    # Actor：根据状态输出离散动作的策略分布（Categorical）
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), activation="relu"):
        # state_dim: 状态向量的维度（观测空间维度）
        # action_dim: 离散动作的数量（动作空间大小）
        # hidden_sizes: 隐藏层规模列表，例如 [64, 64]
        # activation: 隐藏层激活函数类型，通常使用 "relu"
        super().__init__()
        self.mlp = MLP(state_dim, hidden_sizes, action_dim, activation)

    def forward(self, state):
        # state: 状态张量，一般形状为 [batch_size, state_dim]
        # 返回给定状态下的动作分布（用于采样和计算 log_prob）
        logits = self.mlp(state)
        return torch.distributions.Categorical(logits=logits)

    def act(self, state):
        # state: 当前环境状态张量，可以是单个状态或一批状态
        # 用当前策略对状态采样动作，并返回动作、对应的 log_prob 和熵
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def log_prob(self, state, action):
        # state: 状态张量
        # action: 已执行（或采样得到）的动作张量
        # 在给定状态和动作下，计算动作的 log_prob 和策略熵
        dist = self.forward(state)
        return dist.log_prob(action), dist.entropy()

class CriticNetwork(nn.Module):
    # Critic：根据状态估计状态价值 V(s)，用于计算优势函数
    def __init__(self, state_dim, hidden_sizes=(64, 64), activation="relu"):
        # state_dim: 状态向量的维度
        # hidden_sizes: 价值网络隐藏层规模列表
        # activation: 隐藏层激活函数类型
        super().__init__()
        self.mlp = MLP(state_dim, hidden_sizes, 1, activation)

    def forward(self, state):
        # state: 状态张量，一般形状为 [batch_size, state_dim]
        # 返回每个状态对应的标量价值，形状为 [batch_size]
        return self.mlp(state).squeeze(-1)
