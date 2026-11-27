# Visual Primitive 多层 Attention Grounding

视觉读出实验的当前入口见 [visual-primitive 分支](https://github.com/lyk-nju/ESkill-VLA/tree/visual-primitive/experiments/visual_primitive)。

模型入口为 `training_stage=visual_primitive`。Current 从完整任务指令对视觉 patch 的原生注意力中读出当前交互目标。已有单视角实验同时训练独立的夹爪 Subject。双视角配方只训练 Current。[实现](../../src/openpi/models/visual_primitive.py)

## 主相机与腕部相机的 Learn to See

数据按同一个 Episode 的源帧号读取主图和腕图。每路 224×224 RGB 编码成 256 个 patch，与有效任务 token 一起经过一次双向 PaliGemma。每层每个文本查询、每个头先在全部有效 key 上做 softmax，再截取并条件归一化主图的 256 项。各头取均值，同一组学习得到的文本权重 α 聚合各层文本查询。L10–L17 各自产生一张 16×16 图，对主图可见目标计算八个 KL 的均值。SigLIP、投影层、语言嵌入和低层参数冻结，AdamW 更新完整的 L10–L17 和 α。

训练以 `visual_primitive_full_vlm/run_002` 为历史参考。两组都从 gpu10 原始 π₀.₅ 起步，使用相同的父轨迹划分、全局 batch 128、学习率 5e-5 和 20k 预算。历史组的权重为 Current 0.1、Subject 1，双视角组只有 Current 0.1。Subject 读取 detach 后的视觉特征，两个配方的总损失和全局梯度裁剪范围仍有差异。正式对照沿用服务器上的同一份基座文件，分别取 10k、20k 权重。[双视角训练配方](../../examples/libero/train_visual_primitive_multiview.json)

```bash
PYTHONPATH=src:. torchrun --standalone --nproc-per-node=4 scripts/train.py \
    --config examples/libero/train_visual_primitive_multiview.json
```

腕图开关依据主图里是否实际搬动了物体。只要一次 Copy-Paste 成功，即使只移动了干扰物，训练也屏蔽腕图的全部 key。其余有效腕图以 20% 概率随机屏蔽，包括只换任务文本、只做背景遮挡和 Copy 尝试失败的样本。随机数取自专用子序列，同一源帧在同一 epoch 的不同指令共享一次决定。验证默认读取两路原图。镜像同步作用于两路图像，主图裁剪和旋转同步应用于主图标签，两路各自采样颜色增强。

本机 1,693 条轨迹的 273,465 个有效主相机帧均能按源帧号找到有效腕图。训练包只有主图实体标注，缺少腕部实例 mask 和逐帧腕部标定，无法验证二维位移在腕图中的新位置。以下计数使用正式配方和 seed 42，从每个训练任务抽一条轨迹、每条抽十个不同源帧。

| 实际增强 | 样本 | 保留腕图 | 随机屏蔽 | 位移后强制屏蔽 |
| --- | ---: | ---: | ---: | ---: |
| 原文本，无位移 | 169 | 133 | 36 | 0 |
| 只换文本 | 4 | 3 | 1 | 0 |
| 原文本，有位移 | 209 | 0 | 0 | 209 |
| 位移且换文本 | 18 | 0 | 0 | 18 |

在 173 个无位移样本中，37 个随机丢腕（21.4%）。模型看到有效腕图时仍可推断主图没有发生 Copy 位移。每步跨卡总数见 `metrics.jsonl` 的 `wrist_samples`，四组分项见 `wrist_by_augmentation`。原始腕图缺帧单列为 `missing`。

GPU 3 的技术检查在留出轨迹 `libero_object_no_noops_s016_r001` 第 105 帧运行了一次本地 π₀.₅ 前向和反向。八层各输出 256 个主图 patch，Current KL 为 2.6372，腕部原始注意力质量为 0.1304，屏蔽后为 0。该检查使用 batch 1 和 BF16 骨干参数。本机基座文件 SHA-256 以 `ce648b1f` 开头，gpu10 的 `run_002` 记录以 `bbb94502` 开头；正式配方指向后者的原路径。

推理接口是 `policy.infer([main_rgb, wrist_rgb], task)`，移除腕部使用 `[main_rgb, None]`。训练日志和评测记录条件归一化前的主图、腕图 attention mass。两项之和可以小于 1，其余质量落在文本上。配对评测优先看 B 同图换任务的双目标命中率，以及 E 真实晚期 Place 的接收器命中率。

```bash
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_eval \
    --checkpoint "$CKPT" --manifest "$MANIFEST" --episode-index "$INDEX" \
    --wrist-mode paired --output "$REPORT/paired"
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_eval \
    --checkpoint "$CKPT" --manifest "$MANIFEST" --wrist-mode removed --output "$REPORT/removed"
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_eval \
    --checkpoint "$MONO_CKPT" --manifest "$MANIFEST" --output "$REPORT/mono"
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_compare \
    --run paired="$REPORT/paired" removed="$REPORT/removed" mono="$REPORT/mono" \
    --reference paired --output "$REPORT/comparison"
```

每个真实双视角样本在清单中提供 `episode_id`（默认取 `parent`）和整数 `frame`。评测核对主图与该源帧逐像素相同，再读取腕图。人工编辑主图的 A–D 样本可以单独使用 `removed`，或在只含编辑样本的清单中使用 `conflict` 配原始腕图。同图换任务的一对查询会核对两路实际输入的哈希。

本机旧版扩展技术清单共 36 行。逐像素比对后，27 行主图等于原观测，含 12 行原任务、9 行换文本和 6 行查询不存在的目标。其余 9 行在主图移动了目标或干扰物，构造记录只保存了二维位移。这项核查覆盖本机技术清单，正式 A–D 仍需核对其各自的源文件。

E 组从固定验证集的原生 Place 事件抽样。用 LIBERO 仿真环境运行下列命令，生成可供上述三组评测共用的清单、同帧双视角 PNG 与掩码。每个任务选一帧，位于 Place 最后四分之一，接收实体在主图有可见像素，而且原演示状态回放连续三帧确认夹爪抓着正确的源物体。清单保留源物体、接收实体、事件边界、原始 HDF5 行号和状态文件校验和。此组用于检查持物期间的接收器定位。

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.late_place \
    --index "$INDEX" --split data/libero/atomic-vla/split.json --output "$E_CASES"
export MANIFEST="$E_CASES/manifest.json"
```

本机 175 条验证轨迹中，36 个任务含符合实体绑定条件的 Place 事件。回放 80 条轨迹后，28 个任务各取得一帧，覆盖 LIBERO-10／Goal／Object／Spatial 各 8／4／10／6 个任务。其中 18 帧的源物体与接收器框中心相距超过 20 个源图像像素，两个 10 px 命中区域可区分。CPU 小模型已跑通同一清单上的双视角、移除腕图和单目三条评测路径。正式 checkpoint 的定位结果待测。[E 组清单与逐帧依据](/data1/yuankai/robot-research/workspace/reports/multiview_klal_late_place_20261127_001/manifest.json)

连续帧可额外标记同一 `timeline_id` 和真实 `place_start_frame`，提供每帧的 `skill`、`target` 及物体和接收器的框。报告记录峰值质心首次单独命中源物体、随后首次单独命中接收器的采样帧；命中半径沿用清单坐标系的 10 px。`switch_delay_frames` 是后者相对真实 Place 起点的帧差，负值代表提前，未观察到这次切换记为 null。采样间隔决定时间分辨率。

## 两种空间图

- **Current** 在 PaliGemma L10–L17 提取各层 8 个头的文本查询行。FP32 softmax 的分母涵盖所有允许访问的 Key。记录分配给图像的原始质量，再对每个头、每个文本 token 的 256 个视觉 patch 做条件归一化。共享的文本权重 α 聚合有效指令 token，各头均值生成该层空间图。训练对每层单独计算目标分布到空间图的 KL，再平均 8 个 KL。部署时平均 8 张图，输出 16×16 Current。
- **GAC** 使用同一份 α 汇聚融合后的文本，作为语义查询。正例与干扰物取自同图标注区域，区域特征取自融合前的视觉 token。至少有两个有效候选才计入对比损失。α 反映文本聚合权重，其空间效果通过定位指标衡量。
- **Subject** 读取冻结的融合前视觉 token。16 个新初始化的 256 维 Query 经过两层 Decoder，池化后与各 patch 打分，输出独立的 16×16 夹爪图。用当前帧的夹爪标签计算空间 KL。

训练侧把物理实体的可见 mask 转成 patch 面积软标签。Visual Primitive 在 Dataset 初始化时校验采样帧上每个有效物理框对应的可见 mask。缺失或标记无效时报告轨迹和实体。有效的全零 mask 表示完全不可见，该实体跳过 Current KL 和 GAC。投影 Region 可用其几何区域监督，Subject 在无可见 mask 字段时使用夹爪框。

Patch Masking 在选择查询前按原始任务判断。含空间关系的任务跳过 Patch Masking，普通任务只保护夹爪。Pick 使用原指令和原目标，Place 的反事实查询共用宿主图像，每个训练样本输入一条指令。Event、框和实体 ID 用于构造监督，每项损失独立按有效样本数归一化。推理入口 `create_current_grounding_policy(...).infer([rgb], task)` 返回 `current_map`、`subject_map`、`visual_tokens`、`vlm_context`、`current_query`，以及 α、逐层图和原始视觉注意力质量。

## 对照与运行

| 配方 | GAC | Current KL | Subject KL | 文本权重 | PaliGemma |
| --- | ---: | ---: | ---: | --- | --- |
| [E0](../../examples/libero/train_visual_primitive_e0.json) | 0 | 0 | 1 | 均匀 | 冻结 |
| [E1](../../examples/libero/train_visual_primitive_e1.json) | 1 | 0 | 1 | 可学习 | L10–L17 Q-LoRA |
| [E2](../../examples/libero/train_visual_primitive_e2.json) | 0 | 0.1 | 1 | 可学习 | L10–L17 Q-LoRA |
| [E3](../../examples/libero/train_visual_primitive_e3.json) | 1 | 0.1 | 1 | 共享可学习 | L10–L17 Q-LoRA |

四份配置均是从原生 π₀.₅ 启动的技术短训。使用 LIBERO40 Grounding、固定父轨迹划分、单卡 BF16、batch 2、32 次更新、seed 42、学习率 5e−5。LoRA 的 rank 和 alpha 均为 8，GAC 温度 0.07。技术配置仅写本地日志。跨机器运行时，在配置中填写真实数据、基座、tokenizer 和独立输出路径。

```bash
PYTHONPATH=src:. python -m scripts.train --config examples/libero/train_visual_primitive_e3.json
PYTHONPATH=src:. python -m scripts.train --config examples/libero/train_visual_primitive_e3.json --resume "$RUN/checkpoint.pt"
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_eval --checkpoint "$RUN/checkpoint.pt" --manifest "$MANIFEST" --output "$REPORT/e3"
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_compare --run "E0=$REPORT/e0" "E1=$REPORT/e1" "E2=$REPORT/e2" "E3=$REPORT/e3" --output "$REPORT/comparison"
```

精简 checkpoint 保存本次视觉模块和 LoRA，重新加载时校验基座与 tokenizer 内容。评测清单声明坐标系、RGB、完整任务、目标、父轨迹、场景和 split。四组对照校验清单与图像、标签内容相同。逐层面板同时展示条件热图和视觉质量，结果给出 3×3 局部峰值中心、10 px 命中、同图双查询命中、位移响应和按父轨迹的配对 Bootstrap。

## Attention Allocation Loss 对照

[E3 Allocation 配方](../../examples/libero/train_visual_primitive_e3_allocation.json) 将 Current 的逐层 KL 换成 A³-VLA 式分配损失。原论文用末层的语言到图像注意力和均匀文本聚合。这里沿用 E3 的 L10–L17 条件归一化视觉注意力图和共享 α，在每层分别计算损失后取平均。GAC 和 Subject KL 沿用 E3 的设置。[A³-VLA 第 3.3 节及附录 A.6](https://arxiv.org/html/2611.06447v2#S3.SS3)

覆盖目标区域超过 25% 的 patch 纳入 ROI。每层损失为 `max(0, 0.25 - ROI mass)²`，达到 25% 时该层损失为零。没有达标 patch 的样本跳过这项损失。阈值可通过 `model.allocation_min_patch_overlap` 和 `model.allocation_min_mass` 调节。训练和验证日志记录 ROI 注意力质量、达到下限的层比例以及有效样本数。配方沿用 `loss_weights.current_align=0.1` 作为技术起点，正式对照时可结合共享 Q-LoRA 的梯度标定损失权重。

## 定位口径

- 主中心指标 `peak3_hit10` 和 `peak3_error_px` 使用质量最高的 3×3 窗内加权质心。单峰诊断 `argmax_hit10` 和 `argmax_error_px` 使用最大权重 patch 的中心。训练与验证日志使用 `grounding_current_`／`grounding_subject_` 前缀。评测保留 `hit10` 和 `center_error_px` 作为主读出的兼容字段，历史 BBox 模型的主读出采用直接回归框中心。
- Current 和 Subject 都以各自标注框的中心计算距离，要求空间标注非空且有有效框。训练日志将距离换算成源图像像素，增强样本按增强画面的原始像素尺度计量。锁定清单继续使用原有坐标系和显式 `center_xy`。训练验证的中心指标与使用源图像坐标及相同标注框的评测可直接核对。
- 区域 KL 和 ROI mass 按可见 mask 或可用区域统计。训练日志的 `grounding_current_map_kl` 在最终 Current 图上计算，与评测的 `current.kl` 对应。遮挡和非对称形状会使可见区域偏离框中心，区域指标和中心命中分别反映覆盖与定位。

独立 CNN Probe 仅消费评测保存的空间图。在原 Fit 上训练，在原 Check 选步，然后逐组报告中心误差和 IoU。主模型训练完成后，使用同一协议的原始分区运行：


## 2026-11-23 真实权重技术对照

四组均完成 32 次更新，每组 64 次训练样本抽取。逐步核对了四组的父轨迹、帧、任务文本、目标和增强参数。基于代码 ce9568ae 加运行时补丁，E0 定稿权重在 `e0_final`。训练记录含命令、进程、配置与 checkpoint。[复现实验记录](/data1/yuankai/robot-research/workspace/experiment/visual_primitive/technical32_20261123_001/)

固定 16 帧的损失验证中，有 9 帧满足同图实体对比条件。

| 组 | Current 逐层 KL | GAC Recall@1 | Subject KL |
| --- | ---: | ---: | ---: |
| E0 | 3.914 | 未训练 | 0.709 |
| E1 | 3.912 | 6/9 | 0.713 |
| E2 | 3.899 | 未训练 | 0.709 |
| E3 | 3.904 | 5/9 | 0.713 |

在 E3 首批上，GAC 和 Current KL 对共享 Q-LoRA 的梯度范数经 loss 加权后分别为 0.02324 和 0.01443，Subject 对该组参数为零。真实 π₀.₅ 单帧、8 层全部头的提取核对，以及两进程参数同步均已验证。[基座数值对照](/data1/yuankai/robot-research/workspace/reports/visual_primitive_technical32_20261123_001/real_backbone_equivalence.json)

6 条验证父轨迹共生成 36 条技术查询。四组 Current 均命中 3/30，9 组同图双查询均为 0/9。另有 6 条目标缺席查询单独统计。E0、E1、E2、E3 的 Subject 分别命中 20、23、20、23 次，分母均为 36。实测 E3 将 Milk 目标水平移走 50 px 后，预测中心基本停留在原处。在这次训练预算与输入范围内，E3 未改善 Current 定位。

这份技术清单可独立复评。本机已检查的路径未找到旧 E0 的 Fit/Check 和锁定 A–D 样本清单，正式历史对照与 Probe 拟合仍待原始清单。[完整结果与错误面板](/data1/yuankai/robot-research/workspace/reports/visual_primitive_technical32_20261123_001/README.md)
