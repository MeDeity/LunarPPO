import os
import sys

# 将项目根目录加入到模块搜索路径，方便在脚本中直接 import 项目内部模块
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from environments.lunar_lander_wrapper import LunarLanderWrapper
from utils.replay import ReplayBuffer
from utils.logger import Logger


# 这个脚本是 PPO 训练的主入口，整体流程为：
#   1. 读取配置和命令行参数，设置随机种子
#   2. 创建环境和 PPOAgent
#   3. 循环采集一批轨迹数据（horizon 步），存入 ReplayBuffer
#   4. 使用 GAE 计算优势和价值目标，并调用 agent.update 进行多轮参数更新
#   5. 记录训练日志，并定期保存模型


def load_config(path):
    # 从 YAML 配置文件中读取超参数配置
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 解析命令行参数，方便在不同实验中复用同一训练脚本
    parser = argparse.ArgumentParser()
    # --config: 配置文件路径，默认使用 config/hyperparameters.yaml
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml")
    # --resume: 若提供模型路径，则从该 checkpoint 继续训练
    parser.add_argument("--resume", type=str, default=None)
    # --seed: 随机种子，影响环境、网络初始化等
    parser.add_argument("--seed", type=int, default=42)
    # --total-timesteps: 覆盖配置中的总交互步数，方便快速实验
    parser.add_argument("--total-timesteps", type=int, default=None)
    # --log-dir: 日志保存目录
    parser.add_argument("--log-dir", type=str, default="logs")
    # --exp-name: 实验名（日志和模型会按该名称区分）
    parser.add_argument("--exp-name", type=str, default=None)
    # --save-interval: 每多少次 update 保存一次模型
    parser.add_argument("--save-interval", type=int, default=50)
    # --tb: 是否启用 TensorBoard（具体使用取决于 Logger 的实现）
    parser.add_argument("--tb", action="store_true", default=True)
    args = parser.parse_args()

    # 设置随机种子，尽量保证结果可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 读取基础配置
    config = load_config(args.config)
    # 如有需要，可通过命令行覆盖总步数
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps

    # 创建环境和智能体
    env = LunarLanderWrapper(render_mode=None, seed=args.seed)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = PPOAgent(state_dim, action_dim, config)

    # 构造实验名称（包含学习率、剪切系数、网络结构和随机种子等信息）
    exp_name = args.exp_name or f"ppo_lr{config['training']['learning_rate']}_clip{config['ppo']['clip_epsilon']}_h{'-'.join(map(str, config['network']['hidden_sizes']))}_seed{args.seed}"
    log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)

    # 若指定 --resume，则从已有模型继续训练
    if args.resume:
        agent.load(args.resume, map_location="cpu")

    # 总的交互步数（与环境交互的次数）
    total_timesteps = config["training"]["total_timesteps"]
    # horizon 表示每次收集多少步数据后进行一次 PPO 更新
    horizon = 2048
    # 创建用于存储轨迹数据的回放缓冲区
    buffer = ReplayBuffer()

    # 初始化环境并将起始状态存入 buffer（方便 GAE 计算时处理 episode 边界）
    obs = env.reset()
    buffer.add(obs, 0, 0.0, False, torch.tensor(0.0), torch.tensor(0.0))
    update_step = 0
    collected = 0

    # 使用 tqdm 显示训练进度条
    pbar = tqdm(total=total_timesteps, desc="Training", unit="step", dynamic_ncols=True)
    while collected < total_timesteps:
        # 1. 采集 horizon 步的数据
        for _ in range(horizon):
            # 将 numpy 状态转换为张量，并增加 batch 维度
            state_tensor = torch.from_numpy(obs).unsqueeze(0)
            # 使用当前策略采样动作，同时得到 log_prob、熵和状态价值
            action, log_prob, entropy, value = agent.select_action(state_tensor)
            next_obs, reward, done, _ = env.step(int(action.item()))

            # 将当前 transition 存入回放缓冲区
            buffer.add(
                obs,
                int(action.item()),
                reward,
                done,
                log_prob.item(),
                value.item(),
            )

            obs = next_obs
            collected += 1
            pbar.update(1)

            if done:
                # 若 episode 结束，重置环境并插入一个“虚拟”起点，
                # 方便在 compute_gae 时正确处理下一条轨迹的开头
                obs = env.reset()
                buffer.add(
                    obs,
                    0,
                    0.0,
                    False,
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                )

        # 2. 使用 GAE 计算优势函数和价值目标
        advantages, targets = buffer.compute_gae(agent.gamma, agent.gae_lambda)
        # 从 buffer 中取出状态、动作和旧 log_prob，转换为张量
        states, actions, old_log_probs = buffer.as_tensors()

        # 构造传入 PPOAgent.update 的 batch
        batch = {
            "states": states,
            "actions": actions,
            "advantages": advantages,
            "targets": targets,
            "old_log_probs": old_log_probs,
        }

        # 3. 调用 PPOAgent 执行多轮更新，并返回损失和熵等指标
        metrics = agent.update(batch)

        # 4. 记录训练指标到日志系统（用于可视化）
        logger.log_metrics(
            {
                "train/loss_policy": metrics["loss_policy"],
                "train/loss_value": metrics["loss_value"],
                "train/entropy": metrics["entropy"],
            },
            update_step,
        )
        pbar.set_postfix(
            {
                "policy": f"{metrics['loss_policy']:.3f}",
                "value": f"{metrics['loss_value']:.3f}",
                "entropy": f"{metrics['entropy']:.3f}",
            }
        )
        logger.flush()

        # 清空 buffer，为下一轮数据收集做准备
        buffer.clear()
        update_step += 1

        # 5. 按间隔保存模型 checkpoint，便于中途评估或恢复训练
        if update_step % args.save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(os.path.join("models", f"{exp_name}_step{update_step}.pt"))

    env.close()
    pbar.close()


if __name__ == "__main__":
    main()
