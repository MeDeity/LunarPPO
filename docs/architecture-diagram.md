# 架构图（图片）

本页提供项目架构的可视化图示，用于教学展示与实现参考。使用 Mermaid 绘制数据流与模块边界，便于在 Markdown 中直接渲染。

## 系统数据流图
```mermaid
flowchart LR
    subgraph Scripts[scripts/]
        T[train.py]
        E[evaluate.py]
        H[play_human.py]
    end

    subgraph Agents[agents/]
        A[PPOAgent]
        AN[ActorNetwork]
        CN[CriticNetwork]
    end

    subgraph Env[environments/]
        L[LunarLanderWrapper]
    end

    subgraph Utils[utils/]
        LG[logger]
        VS[visualization]
        RB[replay buffer]
    end

    subgraph Artifacts
        M[models/]
        LOGS[logs/]
        RSLT[results/]
    end

    T -->|配置/初始化| A
    T -->|创建| L
    T -->|日志| LG
    A --> AN
    A --> CN
    A <-->|轨迹/批次| RB
    L -->|step/obs/reward| A
    A -->|更新/指标| LG
    LG --> LOGS
    T -->|保存/加载| M
    E -->|加载模型| M
    E --> L
    E --> VS
    VS --> RSLT
    H --> L
```

## 目录关系图
```mermaid
flowchart TB
    root[LunarPPO/]
    agents[agents/]
    env[environments/]
    utils[utils/]
    cfg[config/]
    scripts[scripts/]
    models[models/]
    logs[logs/]
    results[results/]
    docs[docs/]

    root --> agents
    root --> env
    root --> utils
    root --> cfg
    root --> scripts
    root --> models
    root --> logs
    root --> results
    root --> docs
```

## 使用说明
- 若渲染环境不支持 Mermaid，可将上述图导出为 SVG/PNG 放置在 `docs/images/` 并在页面中引用
- 教学展示时可配合模块职责与数据流讲解，帮助学生建立整体认知 
