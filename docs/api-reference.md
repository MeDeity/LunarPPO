# 模块 API 说明

本页给出计划中的主要模块与类/函数接口说明，作为实现与教学参考。实际代码落地后，接口将根据仓库文件更新。

## agents.network
### ActorNetwork
- 初始化：`ActorNetwork(state_dim: int, action_dim: int, hidden_sizes: List[int])`
- 前向：`forward(state) -> dist`（返回策略分布，如 Normal/Categorical）
- 抽样：`sample_action(state) -> action, log_prob, entropy`
### CriticNetwork
- 初始化：`CriticNetwork(state_dim: int, hidden_sizes: List[int])`
- 前向：`forward(state) -> value`

## agents.ppo_agent
### PPOAgent
- 初始化：`PPOAgent(state_dim, action_dim, config)`
  - 关键配置：`clip_epsilon, entropy_coef, value_coef, gae_lambda, gamma, learning_rate, update_epochs, batch_size`
- 采样接口（可选）：`rollout(env, horizon) -> trajectories`
- 更新接口：`update(batch) -> metrics`
  - 输入 batch 建议包含：`states, actions, advantages, targets, old_log_probs`
  - 返回 metrics：`loss_policy, loss_value, entropy, kl, lr`
- 保存/加载：`save(path) / load(path)`

## environments.lunar_lander_wrapper
### LunarLanderWrapper
- 初始化：`LunarLanderWrapper(render_mode=None)`
- 接口：
  - `reset() -> state`
  - `step(action) -> (next_state, reward, done, info)`
  - `render()`

## utils.logger
### Logger
- 初始化：`Logger(log_dir)`
- 记录指标：`log_scalar(tag, value, step)`
- 记录字典：`log_metrics(metrics_dict, step)`
- 同步输出：`flush()`

## utils.visualization
### Visualization
- 曲线绘制：`plot_training_curves(log_dir, output_path)`
- 评估图表：`plot_evaluation(results_dir, output_path)`

## utils.replay
### ReplayBuffer
- `add(state, action, reward, done, log_prob, value)`
- `compute_gae(gamma, gae_lambda) -> advantages, targets`
- `as_batches(batch_size) -> Iterator[Batch]`

## scripts.train
### 入口参数（示例）
- `--config config/hyperparameters.yaml`
- `--resume models/lunar_ppo_best.pt`
### 主要流程
- 读取配置、初始化 env/agent/logger
- 采样 + 更新 + 日志记录 + 模型保存

## scripts.evaluate
### 入口参数（示例）
- `--model models/lunar_ppo_best.pt`
- `--render`
- `--episodes 100`
- `--save`

## scripts.play_human
### 运行
- `python scripts/play_human.py`
- 按键控制（将在实现中定义） 
