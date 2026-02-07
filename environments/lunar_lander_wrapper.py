import gymnasium as gym
import numpy as np
import pygame

class LunarLanderWrapper:
    def __init__(self, render_mode=None, seed=42):
        self.env = gym.make("LunarLander-v3", render_mode=render_mode)
        self.env.reset(seed=seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self):
        obs, _ = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = terminated or truncated
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def render(self):
        if hasattr(self.env, "render"):
            self.env.render()
            try:
                pygame.event.pump()
            except Exception:
                pass

    def close(self):
        self.env.close()
