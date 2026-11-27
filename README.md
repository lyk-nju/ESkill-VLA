# ESkill-VLA · 完整系统

集成从三个专项分支选定的方法，当前组合为 Current KLAL → 十槽 Skill MoE Planner → 双先验 π0.5 Action Expert。后续跨模块训练、整套模型与基线的比较，以及完整系统的闭环评测，都在本分支开展。

Motion Planner 的架构统一称为轨迹规划、边界优先和联合解码。实验表单列训练方式。架构与配置值的对应关系见[规划器设计](https://github.com/lyk-nju/ESkill-VLA/blob/motion-planner/docs/design.md)。

新实验的方案、对照、结果摘要和产物路径记录在本分支的 [experiments/](experiments/) 下，每个实验一个目录。可复用的 LIBERO 训练配方放在 [examples/libero/](examples/libero/)，权重、完整日志和视频放在数据盘。

正式 benchmark 由 `unified` 维护。在当前分支运行 `python -m scripts.benchmark --help`，使用本分支的模型进行预测，由统一协议计分，结果记入本分支的实验目录。[测评协议](docs/evaluation.md)。

四个研究分支的职责见[分支索引](https://github.com/lyk-nju/ESkill-VLA/tree/main#研究分支)。

使用当前 checkout 中的 `python -m scripts.train --config <recipe.json>`。多卡从同一个入口启动，例如 `torchrun --standalone --nproc_per_node=2 -m scripts.train --config <recipe.json>`。计算资产路径、batch 和设备按运行服务器填写。

训练入口、微调策略和相机约定见 [训练说明](src/openpi/training/README.md)。本分支结构与配方见 [设计说明](docs/design.md)。数据格式见 [Episode 说明](docs/data.md)。各分支都保留原生 π0.5 SFT 和 LoRA 基线。
