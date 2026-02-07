import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_curves(log_csv_path, output_path):
    if not os.path.exists(log_csv_path):
        return
    df = pd.read_csv(log_csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if "train/reward_avg" in df.columns:
        ax[0].plot(df["train/reward_avg"])
        ax[0].set_title("Reward Avg")
    if "train/success_rate" in df.columns:
        ax[1].plot(df["train/success_rate"])
        ax[1].set_title("Success Rate")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

def plot_evaluation(results_csv_path, output_path):
    if not os.path.exists(results_csv_path):
        return
    df = pd.read_csv(results_csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["reward"], bins=20)
    ax.set_title("Reward Distribution")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
