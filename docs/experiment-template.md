# 实验模板（脚本参数与日志字段约定）

本页提供统一的脚本参数规范、日志字段命名与输出结构，便于开展可复现的实验与教学。

## 目标
- 统一训练/评估脚本的命令行参数
- 统一日志字段与文件输出命名
- 提供实验记录模板，支持快速对比与复现

## 目录与命名约定
- 运行产出目录：
  - `logs/`：训练过程日志与 TensorBoard
  - `models/`：模型与检查点
  - `results/`：评估与图表输出
- 试验命名：
  - 使用短标签，如：`ppo_lr3e-4_clip0.2_h64x2_seed42`
  - 作为子目录名：`logs/ppo_lr3e-4_clip0.2_h64x2_seed42/`

## 训练脚本参数约定（train.py）
- `--config` 路径（默认 `config/hyperparameters.yaml`）
- `--resume` 模型路径（可选）
- `--seed` 随机种子（默认 `42`）
- `--total-timesteps` 覆盖配置中的总步数（可选）
- `--log-dir` 日志目录（默认 `logs/`）
- `--exp-name` 实验名（默认自动生成）
- `--save-interval` 模型保存间隔（单位：更新轮次，默认 `50`）
- `--tb` 是否启用 TensorBoard（默认启用）

## 评估脚本参数约定（evaluate.py）
- `--model` 加载模型路径或目录
- `--episodes` 评估轮次（默认 `20`）
- `--render` 是否渲染（默认关闭）
- `--save` 是否保存评估结果（CSV/图表，默认关闭）
- `--results-dir` 输出目录（默认 `results/`）

## 人类试玩脚本参数约定（play_human.py）
- `--render` 开启渲染（默认开启）
- `--model` 可选，载入训练好的策略对战（默认空）

## 日志字段约定（训练）
- 标量（按更新轮次记录）
  - `train/reward_avg`：每窗口平均奖励
  - `train/reward_std`：奖励标准差
  - `train/success_rate`：成功率（定义为着陆成功的比例）
  - `train/loss_policy`：策略损失
  - `train/loss_value`：价值损失
  - `train/entropy`：策略熵
  - `train/kl`：KL 散度（新旧策略）
  - `train/clip_fraction`：裁剪触发比例
  - `train/lr`：学习率
  - `train/episode_length_avg`：平均轨迹长度
- 文本/配置
  - `config/*`：记录各关键超参数（写入 JSON/YAML）
- 文件输出
  - `logs/{exp-name}/scalars.csv`：按列存储上述标量
  - `logs/{exp-name}/events/`：TensorBoard 事件文件

## 日志字段约定（评估）
- 标量（整体）
  - `eval/reward_avg`：平均奖励
  - `eval/reward_std`：奖励标准差
  - `eval/success_rate`：成功率
  - `eval/episode_length_avg`：平均步长
- 每回合结果（CSV）
  - 列：`episode, reward, success, steps`
  - 文件：`results/{exp-name}/eval_episodes.csv`
- 图表
  - `results/{exp-name}/curves.png`：奖励/成功率曲线
  - `results/{exp-name}/distribution.png`：奖励分布直方图

## 实验记录模板
```csv
exp_name,seed,total_timesteps,learning_rate,clip_epsilon,entropy_coef,value_coef,hidden_sizes,update_epochs,batch_size,gamma,gae_lambda,reward_avg,success_rate,notes
ppo_lr3e-4_clip0.2_h64x2_seed42,42,1000000,0.0003,0.2,0.01,0.5,"[64,64]",10,64,0.99,0.95,215.3,0.96,"稳定着陆，熵逐步下降"
```

## 示例命令合集
```bash
# 训练（默认配置）
python scripts/train.py --exp-name ppo_lr3e-4_clip0.2_h64x2_seed42 --seed 42

# 训练（覆盖总步数与日志输出位置）
python scripts/train.py --total-timesteps 1500000 --log-dir logs --exp-name ppo_big_run

# 评估（渲染并保存结果）
python scripts/evaluate.py --model models/lunar_ppo_best.pt --episodes 50 --render --save --results-dir results
```

## 注意事项
- 保持参数命名与日志字段一致性，便于脚本与可视化工具复用
- 训练窗口与统计口径需固定（例如每 N 更新为一窗口）
- 建议同时输出 CSV 与 TensorBoard，兼顾可读性与交互性

更多运行示例见 [usage.md](file:///d:/Project/LunarPPO/docs/usage.md)。 
