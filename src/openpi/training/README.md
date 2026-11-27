# 训练接口

多层原生注意力实验使用 `training_stage=visual_primitive`。下文的 `grounding` 配置用于冻结 VLM 的 Grounding 对照。跨服务器操作与实验配方见[数据准备](../../../docs/remote_grounding_training.md)、[Visual Primitive 训练与评测](../../../experiments/visual_primitive/README.md)和[历史联合动作训练](../../../docs/history/joint_training.md)。

## Current Grounding 专项训练

统一执行 `python -m scripts.train --config /path/to/run.json`。当前配置从 [增强模板](../../../examples/libero/train_current_grounding.json) 或 [无增强对照](../../../examples/libero/train_current_grounding_baseline.json) 复制，再填写本机资源与正式预算。

| 项目 | Current／Subject 专项 |
| --- | --- |
| 模型／阶段 | `model_type=eskill`、`training_stage=grounding` |
| 结构 | `model.grounding_subject=true` |
| 冻结 | `model.freeze_vlm=true`，视觉塔／原视觉投影／语言主干及 VAE 均冻结 |
| 训练参数 | 两路视觉投影、文本投影／Attention、Queries／Decoder、共享 latent 读出头 |
| 输入 | 当前主相机＋完整任务；无状态／投影／标定／真实目标输入 |
| 目标 | `current`／`subject`，各自全局有效数量归一化的区域加权 MSE |
| 权重 | `current=0.1`、`subject=0.1` |
| 数据 | 完整 `libero40_grounding`；固定父轨迹划分 |
| 预算 | 仓库模板为 32 步接口配置，正式预算需指定 |

Current 读取融合图文特征，Subject 读取融合前视觉 Tokens；两路各 16×256。专项不实例化 Skill、Future、Motion、Motion→Action 投影，不执行继承的冻结动作专家。当前架构由 Grounding 阶段与 Subject 开关确定，见 [运行版本说明](../../../docs/versions.md)。

## 数据与增强

数据前置及补齐命令见 [数据协议](../../../docs/data.md)。反事实开启时初始化检查逐帧布尔 `interaction_locked`；Copy 开启时检查逐帧 `robot::single` mask。缺失直接报错并提示补齐；不要静默改概率绕过。模型和 Dataset 不导入仿真。

增强先执行在线 Copy／单任务反事实／保护区外 Patch，再执行同步裁切／旋转及输入颜色扰动。位置增强、原 Current 优先和文本替换的概率见 [增强说明](../../../docs/augmentation.md)。

Dataset 返回 `inputs`、`targets`、`metadata`；两路标签 `role_images/weights/valid` 顺序为 `(current, subject)`，形状包含两臂槽位但只使用第一槽。`role_core_boxes` 保存最终主相机坐标系中的归一化紧框。BBox 配方将第一臂槽的 Current 与 Subject 紧框分别用于独立回归，使用纯框配方时无需提供 FLUX VAE；BBox＋GAC 和 [VAE＋GAC](../../../examples/libero/train_vae_gac.json) 另外使用同图实体框、有效掩码和正例索引。这些字段均属于训练标签。

验证关闭所有增强，用完整任务与真实 Event 目标；固定验证子集损失不等于任务成功率。`val_max_batches` 为每卡最多验证 batch 数，不自动遍历全部验证帧。

## 初始化、优化与恢复

专项从明确来源的原生 π0.5 `pretrained_path` 加载，VAE 用 `vae_path`，tokenizer 用 `tokenizer_path`。不要将旧动作或共享投影专项 checkpoint 直接当作当前结构初始化。当前模板 FP32 参数＋BF16 autocast，AdamW LR=5e-5、warmup=1000，此后恒定；具体以该运行实际配置为准。共享配置仍要求 `norm_stats` 资源路径；它不代表专项会监督动作或输入本体状态。

`batch_size` 为每卡 batch，当前无梯度累积；全局 batch＝每卡 batch×world size。配置不会自动选择空闲 GPU，启动前检查资源。GPU 技术短测保持 `smoke=false` 使用真实权重，仅显式禁用该次 W&B；`smoke=true` 是随机小模型，不能当真实基座验证。

首次输出目录须为空。定期保存完整模型、优化器、各卡 RNG、实际配置与资源身份，最新 `checkpoint.pt` 与对应里程碑采用硬链接；不同里程碑仍占空间。使用 `--resume checkpoint.pt` 恢复同一实验，`num_steps` 是总目标步数，必须大于已保存步数。

恢复要求相同 world size、batch、模型、划分、资源内容、增强与优化器语义。可迁移资源路径，但内容需一致；更新数据标记或增强规则后新开实验。恢复沿用 W&B run，新实验使用独立输出和 run。

## 日志与接口职责

正式训练默认读取仓库根目录本机 `wandb.json`；技术检查显式设 disabled。Current 损失记录为 `train/current_object_loss`／`val/current_object_loss`，Subject 为 `train/subject_grounding_loss`／`val/subject_grounding_loss`；另记总损失、学习率、梯度范数和单步耗时。BBox 配方记录 bbox_loss、IoU、中心像素误差、预测与真值面积比。两种 GAC 配方另记 gac_loss 和具有有效候选的 Recall@1。

本地 `metrics.jsonl`／`validation.jsonl` 保存有效监督、增强事件与几何诊断；仅汇报有效字段，不伪造 Action／Skill 指标。增强检查不能替代独立验证集的泛化评估。

| 位置 | 职责 |
| --- | --- |
| `config.py` | 结构／运行配置与入口校验 |
| `data_loader.py` | 固定划分、采样键、worker 与组 batch |
| `checkpoints.py` | 资源身份、权重、优化器、RNG 保存恢复 |
| `metrics.py` | 有效损失及诊断聚合 |
| `scripts/train.py` | 单卡／DDP 优化、验证、日志和保存 |

专项在线读取使用 [current_grounding](../policies/current_grounding.py)，不走动作 rollout。历史 joint／stage1／stage2、Motion 梯度与 Action 训练说明集中在 [历史训练接口](../../../docs/history/joint_training.md)。
