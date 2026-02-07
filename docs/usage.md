# 使用指南：训练与评估

本文档整理常用脚本的运行方式，包括训练、评估与人类试玩。

## 开始训练
```bash
# 默认参数训练
python scripts/train.py

# 使用自定义配置
python scripts/train.py --config config/hyperparameters.yaml

# 在已有模型基础上继续训练
python scripts/train.py --resume models/lunar_ppo_best.pt
```

## 评估模型
```bash
# 基本评估
python scripts/evaluate.py --model models/lunar_ppo_best.pt

# 渲染/回放
python scripts/evaluate.py --model models/lunar_ppo_best.pt --render

# 批量评估并保存结果
python scripts/evaluate.py --model models/ --episodes 100 --save
```

## 人类玩家试玩
```bash
python scripts/play_human.py
```

## 日志与可视化
- 训练日志与指标输出建议写入 `logs/`
- 使用 TensorBoard 可视化：`tensorboard --logdir logs/`
- 评估图表与曲线存放建议为 `results/`

上述目录与脚本将随着项目实现逐步补齐；具体结构以仓库最新代码为准。 
