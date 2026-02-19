import os
import sys

# 将项目根目录加入到模块搜索路径中，方便使用 "agents"、"environments" 等包
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
import torch

from agents.ppo_agent import PPOAgent
from environments.lunar_lander_wrapper import LunarLanderWrapper
from utils.visualization import plot_evaluation


# 这个脚本用于加载训练好的 PPO 模型，在 LunarLander 环境中进行多次评估：
#   1. 运行若干个完整回合（episodes），记录每回合总奖励、步数和是否“成功”
#   2. 打印平均奖励、奖励标准差和成功率等指标
#   3. 可选地将每回合数据保存到 CSV，并画出奖励分布图


def main():
    # 使用 argparse 从命令行读取参数，方便在终端灵活控制评估行为
    parser = argparse.ArgumentParser()
    # --model: 已训练模型文件路径（必需参数）
    parser.add_argument("--model", type=str, required=True)
    # --episodes: 评估的回合数（默认 20 回合）
    parser.add_argument("--episodes", type=int, default=20)
    # --render: 是否渲染环境（加上该参数即开启渲染）
    parser.add_argument("--render", action="store_true", default=False)
    # --save: 是否保存每回合的评估结果到文件
    parser.add_argument("--save", action="store_true", default=False)
    # --results-dir: 存放评估结果（CSV、图像）的目录
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    # 创建包裹好的 LunarLander 环境
    # 如果传入 --render，则使用 "human" 模式实时显示飞船降落过程
    env = LunarLanderWrapper(render_mode="human" if args.render else None)
    state_dim = env.state_dim
    action_dim = env.action_dim

    # 创建一个 PPOAgent，并使用与训练时相同的超参数（这里是硬编码的）
    # 注意：评估阶段只会用到前向推理，不会再训练
    agent = PPOAgent(
        state_dim,
        action_dim,
        {
            "network": {"hidden_sizes": [64, 64], "activation": "relu"},
            "training": {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
            "ppo": {
                "clip_epsilon": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "update_epochs": 10,
                "batch_size": 64,
            },
        },
    )

    # 从指定路径加载训练好的模型参数到 CPU 上
    agent.load(args.model, map_location="cpu")

    rewards = []      # 每个回合的总奖励
    steps_list = []   # 每个回合的步数
    successes = []    # 每个回合是否成功（1 表示成功，0 表示未成功）

    # 进行多次评估（episodes 回合）
    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        # 在当前回合中不断与环境交互，直到回合结束
        while not done:
            # 将 numpy 状态转为 PyTorch 张量，并增加一维 batch 维度
            state_tensor = torch.from_numpy(obs).unsqueeze(0)

            # 使用当前策略选择动作，这里忽略 log_prob、entropy 和 value，只用动作
            action, _, _, _ = agent.select_action(state_tensor)

            # 与环境交互一步，获得下一观测和奖励
            obs, reward, done, info = env.step(int(action.item()))

            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)

        # 将累计奖励 >= 200 视为“成功”一次
        # 对于 LunarLander 任务，平均回报达到 200 左右通常被视为“已学会”
        successes.append(1 if total_reward >= 200 else 0)

    # 计算评估指标：平均奖励、奖励标准差和成功率
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    success_rate = float(np.mean(successes))
    print("avg_reward", avg_reward)
    print("std_reward", std_reward)
    print("success_rate", success_rate)

    # 如需保存详细评估结果，则写入 CSV，并绘制奖励分布图
    if args.save:
        os.makedirs(args.results_dir, exist_ok=True)
        df = pd.DataFrame(
            {
                "episode": list(range(1, args.episodes + 1)),
                "reward": rewards,
                "success": successes,
                "steps": steps_list,
            }
        )
        csv_path = os.path.join(args.results_dir, "eval_episodes.csv")
        df.to_csv(csv_path, index=False)

        # 根据保存的 CSV 画出如直方图/箱线图等分布图（具体取决于实现）
        plot_evaluation(csv_path, os.path.join(args.results_dir, "distribution.png"))

    env.close()


if __name__ == "__main__":
    main()
