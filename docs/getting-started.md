# 入门指南

本文档帮助你在本地快速搭建运行 LunarPPO 的开发环境，并为后续教学与实现打好基础。

## 环境准备
- Python 3.8+（建议使用 3.10 或以上）
- 推荐使用虚拟环境（venv 或 conda）
- 依赖：PyTorch、Gymnasium（含 Box2D）、TensorBoard 等

## 安装步骤
```bash
# 克隆项目（如已存在本地目录可跳过）
git clone https://github.com/yourusername/LunarPPO.git
cd LunarPPO

# 创建并激活虚拟环境
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

如你尚未生成 requirements.txt，可在实现过程中逐步补充依赖并更新文件。

## 验证安装
```bash
python -c "import torch, gymnasium; print('OK', torch.__version__)"
pip list | grep gymnasium
```

若 Gymnasium Box2D 安装失败，需确认系统具备构建工具或改用预编译包。

## 目录建议
项目规划目录结构参考根目录 [README.md](file:///d:/Project/LunarPPO/README.md) 中的“项目结构”章节；随着实现推进，将逐步在仓库中创建对应模块与脚本。

## 下一步
- 阅读 [algorithm-ppo.md](file:///d:/Project/LunarPPO/docs/algorithm-ppo.md) 了解 PPO 与 GAE
- 参考 [usage.md](file:///d:/Project/LunarPPO/docs/usage.md) 运行训练与评估脚本
- 按照 [project-plan.md](file:///d:/Project/LunarPPO/docs/project-plan.md) 进行周度学习与实现 
