# 训练入口

运行前设置配方中的 Episode index、固定父轨迹划分、训练集动作统计、tokenizer、π0.5 基座、输出目录和所需的上游 checkpoint。训练、验证和恢复使用同一个配置入口。

```bash
python -m scripts.train --config examples/libero/train_visual_primitive_multiview.json
torchrun --standalone --nproc_per_node=2 -m scripts.train --config examples/libero/train_visual_primitive_multiview.json
python -m scripts.train --config examples/libero/train_visual_primitive_multiview.json --resume /path/to/checkpoint.pt
```

### 参数与相机

`vlm_finetune.mode` 可取 `frozen`、`lora`、`full`。`layers` 取 `"all"` 或语言模型层编号数组。`full` 配合 `[10,11,12,13,14,15,16,17]` 仅训练 π0.5 的最后八层。LoRA 另设 `target_modules`、`lora_rank`、`lora_alpha`。同一配置下，视觉塔和图像投影分别由 `vlm_finetune.freeze_vision_tower`、`freeze_image_projector` 控制。KLAL 读取层由 `model.attention_layers` 指定。原生 π0.5 全参基线用 `train_pi05_full.json`，LoRA 用 `train_pi05_lora.json`。历史 `train_pi05_sft*.json` 也启用了 LoRA。原有 LoRA 配方在语言与动作两条流上安装 adapter。

LIBERO 使用 `["image"]` 或 `["image","wrist_image"]`。主相机对应 KLAL 和二维轨迹的空间网格。两视角输入共用一次图文编码，缺少的腕帧通过有效性 mask 屏蔽。文本替换沿用普通样本的腕图 dropout 概率，实际移动像素后会屏蔽腕图。

在线 Copy-Paste 只编辑主图。当前双视角配方对同帧腕图以 20% 概率随机丢弃，成功移动像素时一律丢弃。腕图可见意味着该样本没有成功位移。`metrics.jsonl` 的 `wrist_pairing` 分别记录有、无位移时源腕帧可用数、实际保留数和比例，跨步统计请先合计分子分母。物理一致的双视角评测使用同一场景状态下的主图和腕图，另行报告冲突腕图与移除腕图的结果。

原生 π0.5 同时使用 LIBERO 和 RoboTwin 的动作数据适配器。当前研究配方使用 LIBERO 的单臂事件、实体 mask 和主相机 EEF 标注。

验收清单可先运行 `python -m scripts.analysis.visual_primitive_eval --audit-wrist --manifest MANIFEST --episode-index INDEX --output DIR`。输出按 split 给出同帧匹配、主图与源场景冲突、缺失或无法核实的数量，并附逐行结果。三维重排样本须在索引中引用编辑后场景自己的 Episode 与帧；若清单另存腕图，配对时还会核对其像素。

### 监督与权重

Planner 从 KLAL 初始化时，`vlm_finetune` 控制主干更新，`model.train_current` 控制 Current 文本权重聚合器，`loss_weights.current_align` 控制 KLAL 的直接监督。只微调 VLM 并设 `current_align=0` 时，主干仍从规划损失获得梯度。历史 `planner_joint=true` 配方从基座初始化并启用聚合器。三者的设置随训练配置保存。

在支持动作执行的分支，训练首步、日志步、末步及验证记录各动作专家层对主图、腕图、文本、Motion 和动作自身的注意力质量。各层值位于本地 `aux.action_layer_XX_*_mass`，跨层均值进入 W&B 的 `aux_train` 和 `aux_val`。统计对参与 FM 监督的动作步及所有头取平均，腕图缺席按零质量计入。

| 在线编辑 | Current | Start | Skill、End、未来轨迹、动作 |
| --- | --- | --- | --- |
| 原始观测 | 按标注 | 按 EEF 有效性 | 按对应标签或动作行 |
| 同图换指令或移动物体 | 按编辑后的目标 | 执行端可见且标注有效时保留 | 屏蔽原示范 |
| 镜像、裁剪、旋转 | 与主图同步 | 与主图同步 | 轨迹同步变换，动作行在几何变化时屏蔽 |

训练数据、增强、采样、分布式有效项归约、优化器、日志以及 checkpoint 读写通过公共代码维护。在 `unified` 将公共修改单独提交，再将相应提交同步至研究分支。每次实验固定分支和 commit。显式指定 `vlm_finetune` 的训练保存完整模型，可独立恢复；新阶段初始化会继承源主干和所需模块，并重新创建优化器。旧的紧凑 adapter 需要匹配的基座。
