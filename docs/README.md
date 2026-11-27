# 文档索引

当前主线是冻结 π0.5 VLM 的 Current／Subject 视觉 Grounding。文档按用途划分；具体运行的配置与结果以该实验记录为准。

| 用途 | 文档 |
| --- | --- |
| 研究目标 | [idea.md](idea.md) |
| 当前架构与训练范围 | [design.md](design.md) |
| Grounding 标签、画布与 VAE | [target_grounding.md](target_grounding.md) |
| 数据字段与存储 | [data.md](data.md) · [保存格式](../data/README.md) |
| Event 与目标绑定 | [event_organization.md](event_organization.md) |
| 在线编辑、反事实、Patch 与经典增强 | [augmentation.md](augmentation.md) |
| 训练接口 | [training/README.md](../src/openpi/training/README.md) |
| 远端操作步骤 | [remote_grounding_training.md](remote_grounding_training.md) |
| 验收与能力边界 | [evaluation.md](evaluation.md) |
| 分支、模型开关与 checkpoint | [versions.md](versions.md) |
| 论文参考 | [ref_papers/](ref_papers/README.md) |
| 历史动作方案 | [history/](history/README.md) |

文档职责：设计页解释结构，数据页解释保存事实与标签，增强页解释变换，操作页给命令。历史实验与核验报告保留当时设置；不要从历史文档提取当前默认参数。
