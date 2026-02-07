import torch
import torch.nn.functional as F
from torch.optim import Adam
from .network import ActorNetwork, CriticNetwork

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        hidden_sizes = tuple(config.get("network", {}).get("hidden_sizes", [64, 64]))
        activation = config.get("network", {}).get("activation", "relu")
        self.actor = ActorNetwork(state_dim, action_dim, hidden_sizes, activation)
        self.critic = CriticNetwork(state_dim, hidden_sizes, activation)
        lr = config.get("training", {}).get("learning_rate", 3e-4)
        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.clip_epsilon = config.get("ppo", {}).get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("ppo", {}).get("entropy_coef", 0.01)
        self.value_coef = config.get("ppo", {}).get("value_coef", 0.5)
        self.update_epochs = config.get("ppo", {}).get("update_epochs", 10)
        self.batch_size = config.get("ppo", {}).get("batch_size", 64)
        self.gamma = config.get("training", {}).get("gamma", 0.99)
        self.gae_lambda = config.get("training", {}).get("gae_lambda", 0.95)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def select_action(self, state_tensor):
        state_tensor = state_tensor.to(self.device)
        with torch.no_grad():
            action, log_prob, entropy = self.actor.act(state_tensor)
            value = self.critic(state_tensor)
        return action.cpu(), log_prob.cpu(), entropy.cpu(), value.cpu()

    def evaluate(self, states, actions):
        dist = self.actor(states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        return new_log_probs, entropy, values

    def update(self, batch):
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        targets = batch["targets"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        metrics = {}
        for _ in range(self.update_epochs):
            new_log_probs, entropy, values = self.evaluate(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, targets)
            entropy_loss = -self.entropy_coef * entropy.mean()
            loss = policy_loss + self.value_coef * value_loss + entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()
            metrics = {
                "loss_policy": policy_loss.item(),
                "loss_value": value_loss.item(),
                "entropy": entropy.mean().item(),
            }
        return metrics

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
