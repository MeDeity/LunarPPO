import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import numpy as np
import pandas as pd
import torch
from agents.ppo_agent import PPOAgent
from environments.lunar_lander_wrapper import LunarLanderWrapper
from utils.visualization import plot_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    env = LunarLanderWrapper(render_mode="human" if args.render else None)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = PPOAgent(state_dim, action_dim, {
        "network": {"hidden_sizes": [64, 64], "activation": "relu"},
        "training": {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
        "ppo": {"clip_epsilon": 0.2, "entropy_coef": 0.01, "value_coef": 0.5, "update_epochs": 10, "batch_size": 64},
    })
    agent.load(args.model, map_location="cpu")

    rewards = []
    steps_list = []
    successes = []
    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            state_tensor = torch.from_numpy(obs).unsqueeze(0)
            action, _, _, _ = agent.select_action(state_tensor)
            obs, reward, done, info = env.step(int(action.item()))
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(1 if total_reward >= 200 else 0)

    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    success_rate = float(np.mean(successes))
    print("avg_reward", avg_reward)
    print("std_reward", std_reward)
    print("success_rate", success_rate)

    if args.save:
        os.makedirs(args.results_dir, exist_ok=True)
        df = pd.DataFrame({"episode": list(range(1, args.episodes + 1)), "reward": rewards, "success": successes, "steps": steps_list})
        csv_path = os.path.join(args.results_dir, "eval_episodes.csv")
        df.to_csv(csv_path, index=False)
        plot_evaluation(csv_path, os.path.join(args.results_dir, "distribution.png"))

    env.close()

if __name__ == "__main__":
    main()
