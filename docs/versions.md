# 运行版本与模型入口

当前开发分支为 **`visual-primitive`**。运行身份由 Git commit、训练配置和数据内容确定。

## 仓库与工作目录

| 本地目录 | 分支 | 用途 |
| --- | --- | --- |
| `visual-primitive` | `visual-primitive` | 当前 Grounding 开发与训练准备 |
| `ESkill-VLA` | `main` | 此前联合动作与诊断代码 |
| `ESkill-VLA-episode` | `simplify-episode` | 更早的 Episode 整理分支 |

这些目录是同一仓库的 worktree。其他服务器可在现有 `ESkill-VLA` 目录直接切换到 `visual-primitive`，不需要复制本机目录布局。远端操作见 [使用说明](remote_grounding_training.md)。

## 模型与训练入口

ESkill 有两条研究主线，当前的 Visual Primitive 和历史联合动作模型。仓库同时保留旧的 Grounding 对照与纯 π₀.₅ 动作基线。

| 用途 | 模型配置 | 信息流 |
| --- | --- | --- |
| Visual Primitive | `eskill` / `visual_primitive` | 主相机和完整任务 → 多层原生注意力 Current；融合前视觉 → 独立 Subject。[训练与评测](../experiments/visual_primitive/README.md) |
| 历史联合动作 | `eskill` / `joint`、`stage1` 或 `stage2` | 图文特征 → Skill 与目标 Grounding → Motion → Action。[历史说明](../experiments/eskill_libero40_direct/README.md) |
| 旧 Grounding 对照 | `eskill` / `grounding` | 冻结 VLM 的 Current／Subject Query Decoder，输出 VAE latent 或 BBox，可加 GAC。[对照配置](../examples/libero/README.md) |
| 纯 π₀.₅ 基线 | `pi05` / `joint` | 图文特征 → π₀.₅ Action Expert。[历史配方](../experiments/pi05_libero40/README.md) |

Visual Primitive 的 E0–E3 共用空间读出结构，比较冻结基座、GAC、多层对齐和两项联合监督。

## Checkpoint 与数据

- Visual Primitive 从原生 π₀.₅ 初始化 Subject 和视觉分支，训练所需的权重通过精简 checkpoint 保存。策略加载和训练恢复按 checkpoint 的模型格式识别归档权重。
- 旧 VAE Grounding 对照使用独立的 Current／Subject 视觉投影和冻结的 FLUX VAE。各历史实验的完整复现使用当时的 Git commit 与配置。
- 同一实验恢复匹配模型、数据内容、划分、增强、优化器、batch 与卡数。资源可以换路径，但实际内容须一致；刷新数据标记或改变增强规则后新开运行。
- `eskill.episode.v2-draft` 是已有数据 schema 标识，继续保留；它与模型命名无关。包版本 0.1.0 也不代替实验的 commit 与配置。

历史实验保留原始结果和资源身份。
