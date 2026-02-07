import os
import sys
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

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--tb", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    if args.total_timesteps is not None:
        config["training"]["total_timesteps"] = args.total_timesteps

    env = LunarLanderWrapper(render_mode=None, seed=args.seed)
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = PPOAgent(state_dim, action_dim, config)

    exp_name = args.exp_name or f"ppo_lr{config['training']['learning_rate']}_clip{config['ppo']['clip_epsilon']}_h{'-'.join(map(str, config['network']['hidden_sizes']))}_seed{args.seed}"
    log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir)

    if args.resume:
        agent.load(args.resume, map_location="cpu")

    total_timesteps = config["training"]["total_timesteps"]
    horizon = 2048
    buffer = ReplayBuffer()

    obs = env.reset()
    buffer.add(obs, 0, 0.0, False, torch.tensor(0.0), torch.tensor(0.0))
    update_step = 0
    collected = 0

    pbar = tqdm(total=total_timesteps, desc="Training", unit="step", dynamic_ncols=True)
    while collected < total_timesteps:
        for _ in range(horizon):
            state_tensor = torch.from_numpy(obs).unsqueeze(0)
            action, log_prob, entropy, value = agent.select_action(state_tensor)
            next_obs, reward, done, _ = env.step(int(action.item()))
            buffer.add(obs, int(action.item()), reward, done, log_prob.item(), value.item())
            obs = next_obs
            collected += 1
            pbar.update(1)
            if done:
                obs = env.reset()
                buffer.add(obs, 0, 0.0, False, torch.tensor(0.0), torch.tensor(0.0))
        advantages, targets = buffer.compute_gae(agent.gamma, agent.gae_lambda)
        states, actions, old_log_probs = buffer.as_tensors()
        batch = {
            "states": states,
            "actions": actions,
            "advantages": advantages,
            "targets": targets,
            "old_log_probs": old_log_probs,
        }
        metrics = agent.update(batch)
        logger.log_metrics({
            "train/loss_policy": metrics["loss_policy"],
            "train/loss_value": metrics["loss_value"],
            "train/entropy": metrics["entropy"],
        }, update_step)
        pbar.set_postfix({"policy": f"{metrics['loss_policy']:.3f}", "value": f"{metrics['loss_value']:.3f}", "entropy": f"{metrics['entropy']:.3f}"})
        logger.flush()
        buffer.clear()
        update_step += 1
        if update_step % args.save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(os.path.join("models", f"{exp_name}_step{update_step}.pt"))

    env.close()
    pbar.close()

if __name__ == "__main__":
    main()
