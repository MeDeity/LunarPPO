import argparse
import os
import sys

# 将项目根目录加入到模块搜索路径，方便导入自定义的包
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pygame

from environments.lunar_lander_wrapper import LunarLanderWrapper
from agents.ppo_agent import PPOAgent


# 这个脚本的作用：
#   1. 让人类通过键盘控制 LunarLander（左、上、右方向键）
#   2. 或者让已经训练好的 PPO 智能体来自动玩游戏
# 便于学生直观感受“人类策略”和“智能体策略”的差异。


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # --model: 如果提供模型路径，就加载 PPO 模型自动玩；否则为人类手动控制
    parser.add_argument("--model", type=str, default=None)
    # --render: 是否渲染画面（默认 True）
    parser.add_argument("--render", action="store_true", default=True)
    args = parser.parse_args()

    # 创建环境，通常使用 "human" 渲染模式显示画面
    env = LunarLanderWrapper(render_mode="human" if args.render else None)
    state_dim = env.state_dim
    action_dim = env.action_dim

    agent = None
    if args.model:
        # 如果指定了模型路径，则构造一个 PPOAgent 并加载训练好的参数
        agent = PPOAgent(
            state_dim,
            action_dim,
            {
                "network": {"hidden_sizes": [64, 64], "activation": "relu"},
                "training": {
                    "learning_rate": 3e-4,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                },
                "ppo": {
                    "clip_epsilon": 0.2,
                    "entropy_coef": 0.01,
                    "value_coef": 0.5,
                    "update_epochs": 10,
                    "batch_size": 64,
                },
            },
        )
        agent.load(args.model, map_location="cpu")

    # 重置环境，开始新的一局
    obs = env.reset()
    done = False
    total_reward = 0.0

    # 用 pygame 自带的时钟控制循环频率，使画面更流畅
    clock = pygame.time.Clock()

    # 一直循环，直到这一局结束
    while not done:
        if agent is None:
            # 没有加载模型时，使用键盘控制飞船
            pygame.event.pump()  # 处理事件队列，避免窗口无响应
            keys = pygame.key.get_pressed()

            # LunarLander 的动作含义（离散动作空间）：
            #   0: 不点火
            #   1: 左边辅助发动机
            #   2: 主发动机（向上）
            #   3: 右边辅助发动机
            if keys[pygame.K_LEFT]:
                a = 1
            elif keys[pygame.K_UP]:
                a = 2
            elif keys[pygame.K_RIGHT]:
                a = 3
            else:
                a = 0
        else:
            # 如果加载了 PPO 模型，则由智能体自动选择动作
            state_tensor = torch.from_numpy(obs).unsqueeze(0)
            action, _, _, _ = agent.select_action(state_tensor)
            a = int(action.item())

        # 执行动作，与环境交互一步
        obs, reward, done, info = env.step(a)
        total_reward += reward

        if args.render:
            env.render()

        # 控制帧率为每秒 60 帧左右，避免运行过快
        clock.tick(60)

    print("episode_reward", total_reward)
    env.close()


if __name__ == "__main__":
    main()
