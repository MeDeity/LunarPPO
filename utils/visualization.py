import os
import matplotlib.pyplot as plt
import pandas as pd

# 这个文件提供了一些简单的可视化工具函数，
# 用于将训练日志和评估结果画成图片，方便观察算法表现。


def plot_training_curves(log_csv_path, output_path):
    # 根据训练过程记录的 CSV 日志文件，绘制训练曲线图
    # log_csv_path: 训练日志 CSV 路径（例如包含 reward、success_rate 等列）
    # output_path: 输出图片路径
    if not os.path.exists(log_csv_path):
        return
    df = pd.read_csv(log_csv_path)

    # 创建一个包含两个子图的画布：左边画平均奖励，右边画成功率
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # 如果日志中包含 train/reward_avg 列，则绘制平均奖励随时间变化曲线
    if "train/reward_avg" in df.columns:
        ax[0].plot(df["train/reward_avg"])
        ax[0].set_title("Reward Avg")

    # 如果日志中包含 train/success_rate 列，则绘制成功率曲线
    if "train/success_rate" in df.columns:
        ax[1].plot(df["train/success_rate"])
        ax[1].set_title("Success Rate")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)


def plot_evaluation(results_csv_path, output_path):
    # 根据评估脚本保存的结果 CSV，绘制每回合总奖励的分布图
    # results_csv_path: evaluate.py 生成的 CSV 文件路径
    # output_path: 输出图片路径
    if not os.path.exists(results_csv_path):
        return
    df = pd.read_csv(results_csv_path)

    # 绘制奖励的直方图，观察模型在多个 episode 上的表现分布
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["reward"], bins=20)
    ax.set_title("Reward Distribution")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
