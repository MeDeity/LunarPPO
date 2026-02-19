import gymnasium as gym
import numpy as np
import pygame

# 这个文件对 Gymnasium 提供的 "LunarLander-v3" 环境做了一层简单封装，
# 主要目的是：
#   1. 统一 observation（观测）和 action（动作）的数据类型，方便送入 PyTorch 网络
#   2. 对 reset / step / render / close 等接口做轻量包装，便于在强化学习代码中使用


class LunarLanderWrapper:
    def __init__(self, render_mode=None, seed=42):
        # render_mode: 环境的渲染模式，例如 None（不渲染）、"human"（弹窗可视化）等
        # seed: 随机种子，用于控制环境的随机性，方便实验可复现

        # 创建 LunarLander-v3 环境实例
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)

        # 设置环境随机种子
        self.env.reset(seed=seed)

        # 记录状态空间维度：observation_space 是一个 Box，
        # 其中 shape[0] 表示观测向量的维度大小
        self.state_dim = self.env.observation_space.shape[0]

        # 记录动作空间维度：对于离散动作空间，n 表示可选动作的数量
        self.action_dim = self.env.action_space.n

    def reset(self):
        # 将环境重置到初始状态，并返回初始观测
        # 返回值 obs 通常是长度为 state_dim 的一维数组
        obs, _ = self.env.reset()

        # 转换为 numpy 数组并统一为 float32，方便后续转为 PyTorch 张量
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        # 与环境交互一步：
        #   action: 当前时间步要执行的动作（这里是离散的整型动作）
        # 返回：
        #   obs: 下一时刻的观测
        #   reward: 当前步获得的即时奖励
        #   done: 回合是否结束（终止或截断）
        #   info: 额外信息字典，一般可以忽略

        # gymnasium 要求传入的动作类型是原生 Python int，这里确保类型正确
        obs, reward, terminated, truncated, info = self.env.step(int(action))

        # terminated: 因为达到终止条件（成功/失败）而结束
        # truncated: 因最大步数等原因被截断而结束
        done = terminated or truncated

        # 同样统一 obs 的数据类型，reward 转成 float，done 转成 bool
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def render(self):
        # 渲染当前环境状态（如果环境支持渲染）
        if hasattr(self.env, "render"):
            self.env.render()
            try:
                # 对于使用 pygame 渲染的环境，手动处理一次事件队列，
                # 避免窗口“假死”（不响应关闭等事件）
                pygame.event.pump()
            except Exception:
                # 某些平台上可能没有 pygame 窗口，此时忽略异常即可
                pass

    def close(self):
        # 关闭环境，释放资源
        self.env.close()
