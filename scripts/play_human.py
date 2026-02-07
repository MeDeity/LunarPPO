import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import pygame
from environments.lunar_lander_wrapper import LunarLanderWrapper
from agents.ppo_agent import PPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--render", action="store_true", default=True)
    args = parser.parse_args()

    env = LunarLanderWrapper(render_mode="human" if args.render else None)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = None
    if args.model:
        agent = PPOAgent(state_dim, action_dim, {
            "network": {"hidden_sizes": [64, 64], "activation": "relu"},
            "training": {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95},
            "ppo": {"clip_epsilon": 0.2, "entropy_coef": 0.01, "value_coef": 0.5, "update_epochs": 10, "batch_size": 64},
        })
        agent.load(args.model, map_location="cpu")

    obs = env.reset()
    done = False
    total_reward = 0.0
    clock = pygame.time.Clock()
    while not done:
        if agent is None:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                a = 1
            elif keys[pygame.K_UP]:
                a = 2
            elif keys[pygame.K_RIGHT]:
                a = 3
            else:
                a = 0
        else:
            state_tensor = torch.from_numpy(obs).unsqueeze(0)
            action, _, _, _ = agent.select_action(state_tensor)
            a = int(action.item())
        obs, reward, done, info = env.step(a)
        total_reward += reward
        if args.render:
            env.render()
        clock.tick(60)
    print("episode_reward", total_reward)
    env.close()

if __name__ == "__main__":
    main()
