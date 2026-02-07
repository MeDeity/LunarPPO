# PPO 算法讲解

本页系统讲解近端策略优化（Proximal Policy Optimization, PPO）以及广义优势估计（Generalized Advantage Estimation, GAE），以支持后续的工程实现与教学。

## PPO 目标
PPO 通过限制新旧策略差异（裁剪/惩罚）来稳定训练，兼顾样本利用率与收敛稳定性。

### 核心思想
- 最大化期望回报的同时，约束策略更新步长
- 使用裁剪的概率比（ratio）避免过度更新
- 结合熵正则鼓励探索，提升策略多样性

## 损失函数
给定旧策略概率 `old_prob` 与新策略概率 `new_prob`，优势 `A`：
```python
ratio = new_prob / old_prob
clip_loss = -torch.min(
    ratio * A,
    torch.clamp(ratio, 1 - eps, 1 + eps) * A
)
value_loss = F.mse_loss(V(s), R_t)  # 价值函数拟合到回报/目标
entropy_loss = -entropy_coef * dist.entropy().mean()
total_loss = clip_loss + value_coef * value_loss + entropy_loss
```
其中 `eps` 为裁剪阈值（如 0.2），`value_coef` 控制价值损失权重，`entropy_coef` 控制熵正则强度。

## GAE 优势估计
GAE 在偏差与方差之间折中，提升优势估计的稳定性：
```python
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
```
向后递推获得每一步的优势，`lambda` 为平衡参数（如 0.95）。

## 训练流程（高层伪代码）
```python
for iteration in range(num_updates):
    trajectories = rollout(policy, env, horizon)
    states, actions, rewards, dones, log_probs, values = trajectories

    advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
    targets = advantages + values

    for epoch in range(update_epochs):
        for batch in batches(states, actions, advantages, targets, log_probs):
            new_log_probs, entropy, new_values = policy(batch.states, batch.actions)
            ratio = torch.exp(new_log_probs - batch.log_probs)

            clip_loss = -torch.min(
                ratio * batch.advantages,
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch.advantages
            ).mean()
            value_loss = F.mse_loss(new_values, batch.targets)
            entropy_loss = -entropy_coef * entropy.mean()

            total_loss = clip_loss + value_coef * value_loss + entropy_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

## 实践建议
- 归一化优势（zero-mean, unit-std）可提升稳定性
- 使用小批次多轮更新（如 batch_size=64, update_epochs=10）
- 监控 KL 散度与策略熵，必要时自适应调整学习率
- 值函数和策略网络可共享部分前馈层，但需谨慎调参

## 参考
- Spinning Up: PPO 教程
- 原始论文：Schulman et al., 2017 
