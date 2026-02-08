# 强化学习数学基础

本页为 PPO/LunarLander 项目的数学教学基础，覆盖 MDP、价值函数、策略梯度、PPO 目标与 GAE 等关键概念与公式，并与工程实现建立映射关系。

## 符号与设定
- 环境 MDP：状态空间 S、动作空间 A、转移概率 P(s'|s,a)、奖励函数 r(s,a)、折扣因子 γ∈(0,1)
- 轨迹 τ：s₀,a₀,r₀,s₁,a₁,r₁,...
- 策略 π(a|s；θ)：参数为 θ 的概率分布
- 回报（折扣累计奖励）：G_t = Σ_{k=0}^∞ γ^k r_{t+k}

## 价值函数与优势
- 状态价值：V^π(s) = E[G_t | s_t=s]
- 动作价值：Q^π(s,a) = E[G_t | s_t=s, a_t=a]
- 优势函数：A^π(s,a) = Q^π(s,a) − V^π(s)
- 工程映射：在代码中 V 由价值网络估计；A 由 GAE 计算得到

## 策略梯度与 REINFORCE
- 目标：最大化 J(θ) = E_{τ∼π_θ} [Σ_t r_t]
- 基本策略梯度：
  ∇_θ J(θ) = E_{s,a∼π_θ} [∇_θ log π_θ(a|s) · G_t]
- 使用优势替代回报可减小方差：
  ∇_θ J(θ) ≈ E [∇_θ log π_θ(a|s) · A^π(s,a)]
- 工程映射：loss_policy 与 log_prob·adv 的乘积相关

## PPO-Clip 目标
- 设旧策略 π_old，新策略 π_new，重要性比率：
  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) = exp(logπ_new − logπ_old)
- Clipping 目标：
  L^CLIP(θ) = E_t [ min( r_t · A_t, clip(r_t, 1−ε, 1+ε) · A_t ) ]
- 直觉：约束更新步长，避免过度优化导致策略崩溃
- 总损失：L = L^CLIP + c_v · L^value − c_e · H(π)
  - L^value 为均方误差(价值回归)
  - H(π) 为策略熵，鼓励探索
- 工程映射：agents/ppo_agent.py 中 ratio、clip、value_loss、entropy

## Generalized Advantage Estimation（GAE）
- 设 TD 残差：δ_t = r_t + γ V(s_{t+1}) − V(s_t)
- GAE 递推：
  A_t = δ_t + γ λ (1−done_t) A_{t+1}
- 其中 λ∈[0,1] 调控偏差/方差权衡
- 目标值：y_t = A_t + V(s_t)
- 工程映射：utils/replay.py 的 compute_gae 函数

## KL 与裁剪触发率
- 监控 KL(π_old || π_new) 以及裁剪触发比例，可诊断更新是否过大
- 工程建议：在训练日志中记录 kl、clip_fraction 以辅助调参

## 训练稳定性要点
- 优势归一化：A ← (A − mean)/std
- 多轮小批量更新：epoch×batch_size
- 熵正则：防止过早收敛
- 学习率与 ε：影响收敛速度与稳定性

## 与代码的对应关系
- 策略网络与价值网络：[agents/network.py](file:///d:/Project/LunarPPO/agents/network.py)
- PPO 更新逻辑：[agents/ppo_agent.py](file:///d:/Project/LunarPPO/agents/ppo_agent.py)
- GAE 与批数据打包：[utils/replay.py](file:///d:/Project/LunarPPO/utils/replay.py)
- 环境封装（LunarLander-v3）：[environments/lunar_lander_wrapper.py](file:///d:/Project/LunarPPO/environments/lunar_lander_wrapper.py)
- 训练脚本与日志：[scripts/train.py](file:///d:/Project/LunarPPO/scripts/train.py)

## 进一步阅读
- Spinning Up: PPO
- Schulman 等人的 PPO 论文 (arXiv:1707.06347)
- GAE 论文 (arXiv:1506.02438)
