import torch
import numpy as np


# 这个文件实现了一个简单的回放缓冲区 ReplayBuffer，
# 用于暂存一段连续的轨迹数据（states, actions, rewards, dones, log_probs, values），
# 然后基于这些数据计算 GAE 优势函数以及价值目标。


class ReplayBuffer:
    def __init__(self):
        # 使用 Python 列表依次存储采集到的每一步数据
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        # 将一次环境交互得到的 transition 添加进缓冲区
        # state: 当前状态（obs）
        # action: 当前执行的动作（离散整型）
        # reward: 当前步得到的即时奖励
        # done: 当前步是否终止（episode 是否结束）
        # log_prob: 旧策略下当前动作的对数概率 log π_old(a|s)
        # value: 价值网络在当前状态下估计的 V(s)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, gamma, gae_lambda):
        # 使用 GAE（Generalized Advantage Estimation）计算优势函数 advantages
        # 同时得到价值网络的回归目标 targets = advantages + values
        # gamma: 折扣因子 γ
        # gae_lambda: GAE 中的 λ 超参数

        # 将列表转换为 PyTorch 张量，方便进行向量化计算
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32)
        dones = torch.as_tensor(self.dones, dtype=torch.float32)
        values = torch.as_tensor(self.values, dtype=torch.float32)

        # advantages 的形状与 rewards 相同，初始化为 0
        advantages = torch.zeros_like(rewards)
        last_adv = 0.0

        # 从倒数第二个时间步开始向前递推（最后一个 value 通常作为“bootstrap”用）
        for t in reversed(range(len(rewards) - 1)):
            # TD 残差（temporal-difference error）：
            # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]

            # GAE 递推公式：
            # A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
            last_adv = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_adv
            advantages[t] = last_adv

        # 价值网络的目标值：V_target = A_t + V(s_t)
        targets = advantages + values

        # 最后一条通常是“额外加的 bootstrap 状态”，不参与训练，因此去掉末尾元素
        return advantages[:-1], targets[:-1]

    def as_tensors(self):
        # 将缓冲区中存储的 numpy/标量数据统一整理成张量格式，
        # 方便直接送入 PPOAgent.update 进行训练

        # 同样去掉最后一个“bootstrap”状态
        states_np = np.asarray(self.states[:-1], dtype=np.float32)
        actions_np = np.asarray(self.actions[:-1], dtype=np.int64)
        log_probs_np = np.asarray(self.log_probs[:-1], dtype=np.float32)

        states = torch.as_tensor(states_np, dtype=torch.float32)
        actions = torch.as_tensor(actions_np, dtype=torch.int64)
        old_log_probs = torch.as_tensor(log_probs_np, dtype=torch.float32)
        return states, actions, old_log_probs

    def clear(self):
        # 清空所有存储的数据，为下一批采样做准备
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
