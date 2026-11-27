# ESkill-VLA · 运动规划

Current-only KLAL 接三种 Planner。八槽结构、十槽边界提前分流和十槽联合结构共用 Current、路由和数据接口。

使用当前 checkout 中的 `python -m scripts.train --config <recipe.json>`。多卡从同一个入口启动，例如 `torchrun --standalone --nproc_per_node=2 -m scripts.train --config <recipe.json>`。计算资产路径、batch 和设备按运行服务器填写。

| 研究分支 | 可用方法 |
| --- | --- |
| visual-primitive | VAE、BBox、GAC、KLAL 视觉定位实验 |
| motion-planner | KLAL 和三种事件 Planner |
| action-execution | KLAL、十槽联合 Planner 和动作接口消融 |
| unified | KLAL、十槽联合 Planner 和双先验执行 |

训练入口、微调策略和相机约定见 [训练说明](src/openpi/training/README.md)。本分支结构与配方见 [设计说明](docs/design.md)。数据格式见 [Episode 说明](docs/data.md)。各分支都保留原生 π0.5 SFT 和 LoRA 基线。
