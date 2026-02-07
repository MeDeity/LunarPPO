# LunarPPO - 基于PPO算法的月球着陆强化学习项目

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-0081a5.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌙 项目概述

LunarPPO 是一个使用近端策略优化（PPO）算法训练AI智能体完成月球着陆任务的强化学习项目。该项目基于Gymnasium的LunarLander-v2环境，实现了从零开始的PPO算法，并提供了完整的训练、评估和可视化工具。

## ✨ 核心特性

- 🚀 **完整的PPO实现**：从零实现PPO算法（Clip版本），包含GAE优势估计
- 📊 **训练过程可视化**：实时监控训练进度、奖励曲线和网络参数
- 🎮 **交互式演示**：训练后可播放AI着陆过程，支持人类玩家试玩
- ⚙️ **超参数调优**：模块化设计，便于调整网络结构和训练参数
- 📈 **性能分析**：详细记录训练指标，支持TensorBoard可视化

## 📁 项目结构

```
LunarPPO/
├── agents/
│   ├── ppo_agent.py          # PPO智能体核心实现
│   └── network.py            # 策略网络和价值网络定义
├── environments/
│   └── lunar_lander_wrapper.py  # 环境封装和预处理
├── utils/
│   ├── logger.py             # 训练日志记录
│   ├── visualization.py      # 结果可视化工具
│   └── replay.py             # 轨迹回放缓冲区
├── config/
│   └── hyperparameters.yaml  # 超参数配置
├── scripts/
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── play_human.py         # 人类玩家试玩
├── models/                   # 保存的训练模型
├── logs/                     # 训练日志和TensorBoard文件
├── results/                  # 评估结果和图表
└── requirements.txt          # 项目依赖
```

## 📖 文档入口

- 教学与实现文档主页：[docs/README.md](file:///d:/Project/LunarPPO/docs/README.md)
- 入门指南：[docs/getting-started.md](file:///d:/Project/LunarPPO/docs/getting-started.md)
- 架构设计：[docs/architecture.md](file:///d:/Project/LunarPPO/docs/architecture.md)
- 模块 API 说明：[docs/api-reference.md](file:///d:/Project/LunarPPO/docs/api-reference.md)
- 使用指南（训练与评估）：[docs/usage.md](file:///d:/Project/LunarPPO/docs/usage.md)
## 🛠️ 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/LunarPPO.git
cd LunarPPO

# Windows（推荐使用 Python 3.10）
# 安装 Python 3.10（已安装可跳过）
winget install Python.Python.3.10
# 创建并激活虚拟环境
py -3.10 -m venv venv310
.\venv310\Scripts\activate
# 安装依赖（3.10 环境下）
.\venv310\Scripts\python -m pip install -r requirements.txt

# Linux/Mac
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 开始训练

```bash
# 基础训练（默认参数）
python scripts/train.py

# 使用自定义配置
python scripts/train.py --config config/hyperparameters.yaml

# 继续训练已有模型
python scripts/train.py --resume models/lunar_ppo_best.pt
```

### 3. 评估模型

```bash
# 评估模型性能
python scripts/evaluate.py --model models/lunar_ppo_best.pt

# 渲染AI着陆过程
python scripts/evaluate.py --model models/lunar_ppo_best.pt --render

# 批量评估并保存结果
python scripts/evaluate.py --model models/ --episodes 100 --save
```

### 4. 人类玩家试玩

```bash
# 与AI对战（看看你能不能比AI飞得更好！）
python scripts/play_human.py
```
- 控制键：← 左引擎、↑ 主引擎、→ 右引擎；无按键=0
- 渲染采用非阻塞事件轮询；若窗口卡顿，请使用 pygame-ce 并确保 60 FPS

## 🔧 核心PPO算法实现

```python
class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        # 策略网络和价值网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # PPO超参数
        self.clip_epsilon = config['clip_epsilon']
        self.entropy_coef = config['entropy_coef']
        self.gae_lambda = config['gae_lambda']
        
    def update(self, states, actions, rewards, dones):
        # 1. 计算GAE优势估计
        advantages = self.compute_gae(rewards, dones)
        
        # 2. PPO-Clip目标函数
        ratio = new_prob / old_prob
        clip_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        )
        
        # 3. 总损失 = 策略损失 + 价值损失 - 熵正则化
        total_loss = clip_loss + value_loss - entropy * self.entropy_coef
        
        # 4. 多轮优化
        for _ in range(self.update_epochs):
            self.optimizer.step(total_loss)
```

## 📈 预期训练结果

经过充分训练后，智能体应该能够：

1. **稳定着陆**：在目标平台中央平稳着陆
2. **高效燃料使用**：优化燃料消耗（通常获得200+奖励）
3. **高成功率**：在100次尝试中成功着陆95次以上

典型的训练曲线：
- 前1000步：随机探索，奖励在-200到-100之间
- 1000-10000步：逐渐学习基本控制，奖励提升到0-100
- 10000+步：掌握精细控制，奖励稳定在200+

## ⚙️ 超参数配置

```yaml
# config/hyperparameters.yaml
training:
  total_timesteps: 1000000
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95

ppo:
  clip_epsilon: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  update_epochs: 10
  batch_size: 64

network:
  hidden_sizes: [64, 64]
  activation: "relu"
```

## 🎯 挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| 稀疏奖励问题 | 使用GAE进行优势估计，稠密化奖励 |
| 训练不稳定 | PPO的Clipping机制，多轮小批量更新 |
| 探索不足 | 熵正则化，鼓励策略多样性 |
| 收敛速度慢 | 自适应学习率，价值函数基线 |

## 📚 学习资源

### 前置知识
- [OpenAI Spinning Up - PPO教程](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Gymnasium LunarLander文档](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [PPO原始论文](https://arxiv.org/abs/1707.06347)

### 扩展学习
- [动手学强化学习](https://hrl.boyuai.com/)
- [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)

## 🤝 贡献指南

欢迎提交Issue和Pull Request！贡献步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- OpenAI 和 Farama Foundation 提供Gymnasium环境
- LunarLander环境由Erwin Coumans基于Box2D开发
- 感谢所有强化学习开源社区的贡献者

---

**⭐ 如果这个项目对你有帮助，请给个Star！**
