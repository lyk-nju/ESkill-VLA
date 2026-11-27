# LIBERO：Current／Subject Grounding

当前 Visual Primitive 使用 `training_stage=visual_primitive`。历史 Grounding 对照使用 `training_stage=grounding`。两者均接收主相机和完整任务。[架构入口](../../docs/versions.md) · [历史对照的远端数据准备](../../docs/remote_grounding_training.md)

## Current Grounding 数据与训练接口

| 配置 | 用途 |
| --- | --- |
| [Visual Primitive E0–E3](../../experiments/visual_primitive/README.md) | 多层原生 Attention、GAC 和新初始化的 Subject，基座使用原生 π₀.₅ |
| [train_current_grounding.json](train_current_grounding.json) | 在线编辑／反事实／Patch＋经典图像增强 |
| [train_current_grounding_baseline.json](train_current_grounding_baseline.json) | 相同结构与监督，无增强对照 |
| [train_vae_gac.json](train_vae_gac.json) | Current／Subject VAE 配方加入同图实体对比，三项权重为 0.1、0.1、0.02 |
| [train_bbox.json](train_bbox.json) | Current 目标框＋Subject 夹爪框，独立回归 |
| [train_bbox_gac.json](train_bbox_gac.json) | 相同配方，加同图实体对比 |

以上均为 32 步接口模板。复制到本次运行目录，填写数据、tokenizer、原生 π₀.₅ 和输出路径。VAE 对照还需要填写 FLUX VAE 权重。固定 split 与训练统计随仓库同步。[BBox 定位实验与评测](../../experiments/current_grounding_localization/README.md)

数据为完整 `atomicdata/libero40_grounding/`，1,693 条轨迹；训练不需要仿真或原始 HDF5。若已有完整目录，直接同步后检查；旧副本缺标记时执行：

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.interaction_locks \
  --index /path/to/libero40_grounding/index.json \
  --output /path/to/reports/eligibility_new --refresh-copy-eligibility
```

开启反事实缺少 `interaction_locked` 会启动报错，Copy 缺机器人 mask 同样报错。补齐工具复用已有 mask／框，不能生成缺失的原生标注。标签语义和有效性见 [数据协议](../../docs/data.md)。

## 检查、预览与训练

从项目根目录执行，输出目录每次使用新路径：

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.grounding_check \
  --config /path/to/run.json --output /path/to/reports/input_new --workers 2
PYTHONPATH=src:. python -m scripts.preprocess.libero.grounding_coverage \
  --config /path/to/run.json --output /path/to/reports/coverage_new --samples-per-task 200
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python -m scripts.preprocess.libero.grounding_review \
  --config /path/to/run.json --output /path/to/reports/review_new --vae-device cuda:0
```

`grounding_check` 检查全量数据并生成各任务标签预览；`grounding_coverage` 统计 Dataset 阶段增强；`grounding_review` 通过训练 batch 变换生成组合增强与真实 VAE 重建。`--benchmark` 使用随机小骨干检查吞吐，不代表完整 π0.5 速度。历史 `--export-samples` 只支持关闭第二阶段变换的 Dataset 导出，完整效果使用默认预览。

确认数据与可视化后先用真实权重短测，再按正式预算训练：

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m scripts.train --config /path/to/run.json
```

GPU／batch 按空闲资源设定，`batch_size` 为每卡值。正式 W&B 和恢复命令见 [训练接口](../../src/openpi/training/README.md) 与 [远端步骤](../../docs/remote_grounding_training.md)。

## 从原始资产重新补标

只有重新生成数据时需要独立 LIBERO 原生环境。来源导出与逐帧基础区域复用 `scripts.preprocess.libero.atomic_vla.batch`；交互目标／全场景框用 [review_interaction_targets.json](review_interaction_targets.json) 的 inventory／extract／render，提取时开启 `protect_robot=true`。已有全场景框后，用 [prepare_grounding_full.json](prepare_grounding_full.json) 补齐可见 mask、编辑资格及交互区：

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.grounding_batch \
  --config examples/libero/prepare_grounding_full.json
```

需先将配置中的来源资产、输出和仿真环境路径改为本机；EGL 和原生来源流程见 [离线处理](../../scripts/preprocess/README.md)。[prepare_grounding.json](prepare_grounding.json) 是较早十条试验配置，不能当全量结果。

全量目录包括 `episode.json`、`arrays.npz`、`images.npz`、`scene_regions.npz`、`online_grounding.npz` 等实际引用文件。先保存再回读渲染；检查报告进入 [AtomicVLA check/test](../../data/libero/atomic-vla/check/test/README.md)，视频放其 `video/`。原始记录与人工审核状态继续保留。

## 在线与历史动作入口

在线只读主相机：

```python
from openpi.policies.current_grounding import create_current_grounding_policy
policy = create_current_grounding_policy('/path/to/grounding/checkpoint.pt', device='cuda')
prediction = policy.infer([main_rgb_uint8], task_instruction)
# prediction['grounding'] / prediction['subject_grounding'] 是两路 tokens。
```

本接口不返回动作，不能直接用于动作 rollout。原生 π0.5、`train_eskill*.json`、动作客户端和两阶段配方仍保留供已有实验显式使用，操作说明已移至 [历史 LIBERO 数据／动作训练／测评](../../docs/history/libero_joint.md)，版本边界见 [versions](../../docs/versions.md)。
