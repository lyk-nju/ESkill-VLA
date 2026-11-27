# 实验入口

当前工作为 Current／Subject 视觉 Grounding。每次新运行显式传配置，独立输出与 W&B run；模板预算不自动成为正式实验预算。

## 当前视觉专项

| 内容 | 入口／状态 |
| --- | --- |
| Visual Primitive 多层注意力 | [E0–E3 实验与评测入口](visual_primitive/README.md) |
| Current 定位实验设计 | [BBox、KLAL 与 GAC](current_grounding_localization/README.md)，含论文源码核对、验证集几何统计和算子检查 |
| 远端准备和训练 | [使用说明](../docs/remote_grounding_training.md) |
| 增强／无增强配置 | [train_current_grounding.json](../examples/libero/train_current_grounding.json)／[baseline](../examples/libero/train_current_grounding_baseline.json)，均为 32 步接口模板 |
| 真实权重启动 | [current_grounding_fused](current_grounding_fused/README.md)，π0.5／FLUX，双卡 BF16 8 步与 checkpoint 回读 |
| 数据与增强检查 | [check/test 索引](../data/libero/atomic-vla/check/test/README.md)，包含标记补齐、候选、位置、原目标优先及 Region 保护 |
| 训练入口 | `python -m scripts.train --config /path/to/run.json`；恢复另传 `--resume` |

真实启动已通过，不代表正式训练收敛或泛化验证。当前专项不执行动作闭环；需要先完成模型验收，再研究接回动作系统。

## 历史联合动作与基线

以下记录保留各自数据、代码和结果语义，不能将其中的“新版”或 Stage 1／Stage 2 预算直接套到当前视觉专项。

| 记录 | 用途 |
| --- | --- |
| [eskill_libero40](eskill_libero40/README.md) | 早期 ESkill 训练、GPU10 与结果 |
| [pi05_libero40](pi05_libero40/README.md) | 纯动作基线 |
| [eskill_libero40_mirror](eskill_libero40_mirror/README.md) | 历史镜像先验20k＋联合动作40k路线 |
| [eskill_libero40_direct](eskill_libero40_direct/README.md) | 直接联合动作与颜色／裁切／旋转对照 |
| [grounding_motion_diagnostic](grounding_motion_diagnostic/README.md) | 历史 Grounding／Motion 干预 |
| [model_prediction_visualization](model_prediction_visualization/README.md) | 历史动作 rollout 预测／Attention 面板，跟随真实请求更新 |
| [composition_step060000](composition_step060000/README.md)／[composition_eskill_step030000](composition_eskill_step030000/README.md) | 当时版本的新组合测评 |

这些目录继续放在原位置，避免破坏配置引用、归一化路径和外部报告。旧 checkpoint 复现使用对应 commit 与配置，不做静默兼容转换。动作训练及闭环命令集中在 [历史训练](../docs/history/joint_training.md) 与 [历史测评](../docs/history/joint_evaluation.md)。
