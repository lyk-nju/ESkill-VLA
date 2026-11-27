# ESkill-VLA · 视觉先验

研究模型如何根据图像和完整任务指令定位当前交互目标。后续视觉先验实验都在本分支开展，包括 Current／Subject 监督、VAE 与 BBox 读出、GAC 与 KLAL 对照，以及相机视角和数据增强对定位的影响。

新实验的方案、对照、结果摘要和产物路径记录在本分支的 [experiments/](experiments/) 下，每个实验一个目录。可复用的 LIBERO 训练配方放在 [examples/libero/](examples/libero/)，权重、完整日志和视频放在数据盘。

正式 benchmark 由 `unified` 维护。在当前分支运行 `python -m scripts.benchmark --help`，使用本分支的模型进行预测，由统一协议计分，结果记入本分支的实验目录。[测评协议](docs/evaluation.md)。

四个研究分支的职责见[分支索引](https://github.com/lyk-nju/ESkill-VLA/tree/main#研究分支)。

使用当前 checkout 中的 `python -m scripts.train --config <recipe.json>`。多卡从同一个入口启动，例如 `torchrun --standalone --nproc_per_node=2 -m scripts.train --config <recipe.json>`。计算资产路径、batch 和设备按运行服务器填写。

训练入口、微调策略和相机约定见 [训练说明](src/openpi/training/README.md)。本分支结构与配方见 [设计说明](docs/design.md)。数据格式见 [Episode 说明](docs/data.md)。各分支都保留原生 π0.5 SFT 和 LoRA 基线。
