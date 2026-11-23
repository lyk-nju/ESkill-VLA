# Visual Primitive 多层 Attention Grounding

模型接收一张主相机 RGB 和完整任务指令。Current 使用 PaliGemma 原生文本到图像注意力定位当前交互目标。Subject 使用独立随机初始化的分支定位夹爪。模型入口是 `training_stage=visual_primitive`。[实现](../../src/openpi/models/visual_primitive.py)

## 两种空间图

- **Current** 在 PaliGemma L10–L17 提取各层 8 个头的文本查询行。FP32 softmax 的分母涵盖所有允许访问的 Key。记录分配给图像的原始质量，再对每个头、每个文本 token 的 256 个视觉 patch 做条件归一化。共享的文本权重 α 聚合有效指令 token，各头均值生成该层空间图。训练对每层单独计算目标分布到空间图的 KL，再平均 8 个 KL。部署时平均 8 张图，输出 16×16 Current。
- **GAC** 使用同一份 α 汇聚融合后的文本，作为语义查询。正例与干扰物取自同图标注区域，区域特征取自融合前的视觉 token。至少有两个有效候选才计入对比损失。α 反映文本聚合权重，其空间效果通过定位指标衡量。
- **Subject** 读取冻结的融合前视觉 token。16 个新初始化的 256 维 Query 经过两层 Decoder，池化后与各 patch 打分，输出独立的 16×16 夹爪图。用当前帧的夹爪标签计算空间 KL。

训练侧把可见 mask 或有效框转成 patch 面积软标签。每个损失独立按有效数归一化。Event、框和实体 ID 仅用于构造监督。反事实查询共用同一图像编辑，训练每个样本只输入一条指令。推理入口 `create_current_grounding_policy(...).infer([rgb], task)` 返回 `current_map`、`subject_map`、`visual_tokens`、`vlm_context`、`current_query`，以及 α、逐层图和原始视觉注意力质量。

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

## 定位口径

- 主中心指标 `peak3_hit10` 和 `peak3_error_px` 使用质量最高的 3×3 窗内加权质心。单峰诊断 `argmax_hit10` 和 `argmax_error_px` 使用最大权重 patch 的中心。训练与验证日志使用 `grounding_current_`／`grounding_subject_` 前缀。评测保留 `hit10` 和 `center_error_px` 作为主读出的兼容字段，历史 BBox 模型的主读出采用直接回归框中心。
- Current 和 Subject 都以各自标注框的中心计算距离，要求空间标注非空且有有效框。训练日志将距离换算成源图像像素，增强样本按增强画面的原始像素尺度计量。锁定清单继续使用原有坐标系和显式 `center_xy`。训练验证的中心指标与使用源图像坐标及相同标注框的评测可直接核对。
- 区域 KL 和 ROI mass 按可见 mask 或可用区域统计。训练日志的 `grounding_current_map_kl` 在最终 Current 图上计算，与评测的 `current.kl` 对应。遮挡和非对称形状会使可见区域偏离框中心，区域指标和中心命中分别反映覆盖与定位。

独立 CNN Probe 仅消费评测保存的空间图。在原 Fit 上训练，在原 Check 选步，然后逐组报告中心误差和 IoU。主模型训练完成后，使用同一协议的原始分区运行：

```bash
PYTHONPATH=src:. python -m scripts.analysis.visual_primitive_probe --run "$REPORT/e3" --output "$REPORT/probe"
```

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
