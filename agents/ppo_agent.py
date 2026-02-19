import torch
import torch.nn.functional as F
from torch.optim import Adam
from .network import ActorNetwork, CriticNetwork

# PPOAgent 封装了 PPO 算法中策略网络（Actor）和价值网络（Critic）的
# 创建、前向推理（选动作）、以及基于采样批次的参数更新逻辑。


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        # state_dim: 环境状态向量的维度，例如 CartPole 中为 4
        # action_dim: 动作空间中离散动作的数量
        # config: 字典形式的超参数配置，通常从配置文件或命令行读取

        # 从配置中读取网络结构相关参数
        hidden_sizes = tuple(config.get("network", {}).get("hidden_sizes", [64, 64]))
        activation = config.get("network", {}).get("activation", "relu")

        # 创建策略网络 Actor 和价值网络 Critic
        self.actor = ActorNetwork(state_dim, action_dim, hidden_sizes, activation)
        self.critic = CriticNetwork(state_dim, hidden_sizes, activation)

        # 训练相关超参数：学习率
        lr = config.get("training", {}).get("learning_rate", 3e-4)

        # 使用 Adam 优化器，同时更新 Actor 和 Critic 的参数
        self.optimizer = Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # PPO 相关超参数
        # clip_epsilon: PPO 剪切范围，控制新旧策略差异的上限
        self.clip_epsilon = config.get("ppo", {}).get("clip_epsilon", 0.2)
        # entropy_coef: 熵系数，用于鼓励策略保留一定随机性（探索）
        self.entropy_coef = config.get("ppo", {}).get("entropy_coef", 0.01)
        # value_coef: 价值损失在总损失中的权重
        self.value_coef = config.get("ppo", {}).get("value_coef", 0.5)
        # update_epochs: 每次收集到一批数据后，重复更新多少个 epoch
        self.update_epochs = config.get("ppo", {}).get("update_epochs", 10)
        # batch_size: 每次更新时使用的小批量大小（如果实现了 mini-batch）
        self.batch_size = config.get("ppo", {}).get("batch_size", 64)

        # 折扣因子和 GAE（广义优势估计）中的 lambda
        self.gamma = config.get("training", {}).get("gamma", 0.99)
        self.gae_lambda = config.get("training", {}).get("gae_lambda", 0.95)

        # 自动选择设备：优先使用 GPU，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def select_action(self, state_tensor):
        # 根据当前策略，从给定状态中采样动作
        # state_tensor: 状态张量，一般形状为 [batch_size, state_dim]
        # 返回值：
        #   action: 采样到的动作张量（在 CPU 上）
        #   log_prob: 对应动作的对数概率
        #   entropy: 策略熵（用来衡量随机性）
        #   value: Critic 估计的状态价值 V(s)
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():  # 选动作时不需要计算梯度
            action, log_prob, entropy = self.actor.act(state_tensor)
            value = self.critic(state_tensor)
        return action.cpu(), log_prob.cpu(), entropy.cpu(), value.cpu()

    def evaluate(self, states, actions):
        # 在训练过程中，给定一批状态和对应的动作，计算：
        #   new_log_probs: 新策略下这些动作的 log_prob
        #   entropy: 新策略的熵
        #   values: Critic 估计的状态价值
        # states: [batch_size, state_dim]
        # actions: [batch_size]，为离散动作的索引
        dist = self.actor(states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        return new_log_probs, entropy, values

    def update(self, batch):
        # 使用一批采样数据对 Actor 和 Critic 进行参数更新
        # batch: 字典，通常包含以下张量（都在采样时缓存）：
        #   "states": 状态序列
        #   "actions": 对应执行的动作
        #   "advantages": 预先计算好的优势函数估计 A_t
        #   "targets": 价值网络的目标值，一般为折扣回报或 GAE 目标
        #   "old_log_probs": 采样时旧策略下动作的 log_prob
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        targets = batch["targets"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)

        # 对优势进行标准化处理，可以稳定训练
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        metrics = {}
        # 重复多次遍历同一批数据进行更新（PPO 的常见做法）
        for _ in range(self.update_epochs):
            # 基于当前策略重新计算 log_prob、熵和价值
            new_log_probs, entropy, values = self.evaluate(states, actions)

            # 计算概率比值 ratio = π_new(a|s) / π_old(a|s)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO 的两个目标：未剪切目标和剪切后的目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            ) * advantages

            # 策略损失取两者中较小的（对应原论文中的 clipped objective）
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数使用均方误差损失
            value_loss = F.mse_loss(values, targets)

            # 熵损失，鼓励策略保持一定随机性（负号是因为我们最小化总损失）
            entropy_loss = -self.entropy_coef * entropy.mean()

            # 总损失：策略损失 + 价值损失（加权）+ 熵损失
            loss = policy_loss + self.value_coef * value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
            )
            self.optimizer.step()

            # 记录当前 epoch 的训练指标，便于日志和可视化
            metrics = {
                "loss_policy": policy_loss.item(),
                "loss_value": value_loss.item(),
                "entropy": entropy.mean().item(),
            }
        return metrics

    def save(self, path):
        # 将当前 Actor 和 Critic 的参数保存到指定路径
        # path: 存储模型的文件路径（例如 "ppo_ckpt.pth"）
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path, map_location=None):
        # 从文件中加载已保存的 Actor 和 Critic 参数
        # path: 模型文件路径
        # map_location: 指定加载到的设备，默认加载到当前 agent 的 device
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
