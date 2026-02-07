# 架构设计

本页描述 LunarPPO 的整体架构设计、模块划分与数据流，便于教学与实现时形成统一认识。

## 总览
- 训练脚本（scripts/）：组织训练流程、读取配置、调度智能体与环境
- 智能体（agents/）：PPO 算法实现（含 actor-critic 网络与更新逻辑）
- 环境封装（environments/）：对 Gymnasium LunarLander-v2 进行包装与预处理
- 工具库（utils/）：日志、可视化、回放等通用辅助组件
- 配置（config/）：超参数与网络结构等配置文件
- 产出目录：models/（模型）、logs/（训练日志/TensorBoard）、results/（评估图表）

## 模块职责
- agents/ppo_agent.py
  - 持有策略网络与价值网络
  - 采样与收集轨迹（或配合脚本进行采样）
  - 计算 GAE、构建目标（advantages/returns）
  - 执行 PPO-Clip 更新与优化
- agents/network.py
  - 定义 ActorNetwork 与 CriticNetwork
  - 初始化权重、前向计算与分布抽样（日志概率、熵等）
- environments/lunar_lander_wrapper.py
  - 统一状态/动作空间接口
  - 处理渲染、归一化、裁剪等预处理（按需）
- utils/logger.py
  - 统一的日志接口，记录训练指标（奖励、损失、熵、KL 等）
  - 输出到控制台、文件或 TensorBoard
- utils/visualization.py
  - 训练曲线绘制与结果图导出
- utils/replay.py
  - 轨迹/回放缓冲区（状态、动作、奖励、done、log_prob、value 等）
- scripts/train.py
  - 解析配置、初始化环境与智能体
  - 组织采样、调用更新、记录指标
- scripts/evaluate.py
  - 加载模型、运行评估、渲染/保存结果
- scripts/play_human.py
  - 人类控制接口，与 AI 对比体验

## 数据流（训练环）
1. 初始化环境与智能体，读取超参数
2. 采样：按时间步收集 (s, a, r, done, log_prob, value)
3. 计算优势与回报（GAE 与 targets）
4. 多轮小批量更新（ratio、clip、value、entropy）
5. 记录指标并可视化（奖励、损失、熵、成功率）
6. 定期保存模型与检查点

## 依赖与边界
- 与外部库的边界：Gymnasium 环境、PyTorch 模型与优化器、TensorBoard 可视化
- 内部模块间的接口清晰：智能体不直接依赖具体脚本实现；环境包装保持独立

## 扩展方向
- 多环境并行采样（向量化环境）
- KL 自适应或学习率调度
- 网络共享/分离结构的实验对比 
