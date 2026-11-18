# Current 定位实验：BBox、空间注意力与 GAC

第一轮已实现 BBox 和 BBox＋GAC，均从冻结的 π0.5 开始。两组共用单主相机、完整任务、Subject 夹爪紧框、采样和增强。模板为单卡 batch 2、32 步，正式预算待定。KLAL 属于后续设计。

| 配方 | 总损失 |
| --- | --- |
| [BBox](../../examples/libero/train_bbox.json) | BBox＋Subject 框损失 |
| [BBox＋GAC](../../examples/libero/train_bbox_gac.json) | BBox＋Subject 框损失＋0.2 × GAC |

BBox 为归一化中心宽高上的四坐标 L1 之和加 1−GIoU。GAC 使用初始温度 0.07。Subject 用独立视觉查询和独立三层 MLP 回归夹爪紧框，同样使用 L1＋1−GIoU。两组可训练参数相同，GAC 复用 Current 的视觉投影。推理返回归一化 xyxy、框内 16 个局部特征和 16 个全图采样位置。

从仓库根目录运行。训练使用本次运行专属的输出目录。评测用同一个 seed 从固定验证集的 Pick／Place 中点取可生成第二条任务的帧，每帧两次推理共用原始 RGB。输出含逐对指令、真值、预测框、两条均命中的比例和可评分覆盖率。两组 checkpoint 各运行一次。

```bash
PYTHONPATH=src:. python -m scripts.train --config examples/libero/train_bbox.json
PYTHONPATH=src:. python -m scripts.train --config examples/libero/train_bbox_gac.json
PYTHONPATH=src:. python -m scripts.analysis.grounding_pairs --checkpoint /path/to/checkpoint.pt --output /path/to/pairs.json --max-pairs 100
```

## 双框梯度校准（2026-11-19）

Current 和 Subject 的框损失权重均为 1。在 GPU 1 上从原生 π0.5 初始化，冻结主干以 BF16 存储，头部 FP32，batch 2，seed 42。每步在全部可训练参数上分别求各损失的梯度。前 32 步用 GAC 系数 0.1，在 GAC 有效的 27 步中，取两项框梯度范数几何平均与原始 GAC 范数之比的中位数，得到 0.381。

后续 64 个新训练批次用系数 0.4，54 步有有效对比样本。这 54 步中，裁剪前三项加权梯度范数的中位数依次为 Current 4.35、Subject 7.93、GAC 5.29。将无有效 GAC 的步数也计入，按各步实际全局裁剪系数计算后，三项范数平均为 0.453、0.718、0.443。此处衡量进入 AdamW 前的梯度尺度，正式选型另看同图换任务的定位结果。[逐步记录及汇总](/data1/yuankai/robot-research/workspace/experiment/bbox_weight_calibration_20261119_VD9zOPAg/summary.json)

## 早期 Subject VAE 配方的技术记录

在 GPU 0 用原生 π0.5、FLUX VAE、FP32 参数和 BF16 前向，两组各做 1 次 AdamW 更新。同一条 LIBERO Object 轨迹取第 10、30 帧，batch 2，seed 42，两组初始 BBox 损失均为 1.8883，Subject 均为 3.5950。GAC 损失为 1.9125。此 batch 上 BBox 与加权 GAC 对 Current learned queries 的梯度范数分别为 0.03239、0.00800。峰值已分配显存均约 13.2 GiB。推理接口用同一帧的橙汁和布丁指令输出两组框，Subject 输出逐元素一致。[BBox 单步记录](/data1/yuankai/robot-research/workspace/experiment/bbox_gac_matched_Dt8j0aZm/bbox.json) · [GAC 单步记录](/data1/yuankai/robot-research/workspace/experiment/bbox_gac_matched_Dt8j0aZm/bbox_gac.json)

按固定训练集的完整任务指令分组，每个任务抽 1 条轨迹、8 帧，开启两阶段增强。320/320 帧有回归监督，190/320 帧有可比较的实体。其余 130 帧中，106 帧的正例未进入外观候选，21 帧的正例在几何变换后未完整保留，3 帧缺少合格负例。逐帧检查了全部 190 个有效正例与回归真值的最终坐标。正式训练后的同图换任务命中率待测。[分任务覆盖记录](/data1/yuankai/robot-research/workspace/experiment/bbox_gac_matched_Dt8j0aZm/coverage.json)

## 论文与源码核对

| 工作 | 核对到的机制 | 对本实验的取舍 |
| --- | --- | --- |
| TransVG | 在图文序列前加入一个可学习的 REG token，融合后取该位置，三层 MLP 加 sigmoid 回归一个归一化的中心宽高框。实际训练调用 `trans_vg_loss`，四坐标绝对误差按框求和、跨样本平均，再加逐框的 1−GIoU，两项直接相加。 | 复用单目标回归和损失定义。ESkill 已有 Current Decoder，一帧一个查询目标，直接沿用其输出。 [论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.pdf) · [模型](https://github.com/djiajunustc/TransVG/blob/c8624277018fc9786a39181e229f155f933b2b59/models/trans_vg.py#L44-L56) · [损失](https://github.com/djiajunustc/TransVG/blob/c8624277018fc9786a39181e229f155f933b2b59/utils/loss_utils.py#L109-L127) · [训练调用](https://github.com/djiajunustc/TransVG/blob/c8624277018fc9786a39181e229f155f933b2b59/engine.py#L39-L40) |
| KLAL | 论文取最后一个回答 token 到视觉 tokens 的注意力，各 head 在视觉位置重新归一化，平均 heads，逐层计算 KL(GT‖预测) 再平均。RefCOCO 用框内沿长轴的缩短中心线生成目标 patches，再做平滑。作者的 Qwen 可视化取最后生成步、最后层，平均 heads，截取视觉段后除以最大值画图。 | 作者公开仓库可核对 Notebook 和数据处理；完整 KLAL 训练循环未在该仓库核实。ESkill 使用可训练的 Current 空间读出，训练与推理读同一张概率图。 [论文](https://openaccess.thecvf.com/content/WACV2026/papers/Esmaeilkhani_Direct_Visual_Grounding_by_Directing_Attention_of_Visual_Tokens_WACV_2026_paper.pdf) · [补充材料 §1.5](https://openaccess.thecvf.com/content/WACV2026/supplemental/Esmaeilkhani_Direct_Visual_Grounding_WACV_2026_supplemental.pdf) · [作者 Notebook](https://github.com/ParsaKhani/visual-grounding-datasets/blob/d2cb2f5efd7a048aa1505b6bb22a9ff9f859e301/visualization/qwen_line_tracing_attention.ipynb) |
| ProGAL-VLA | 论文 Eq. 9–12 用符号子目标作 query、场景实体作 key，经温度缩放计算正例的对比交叉熵。论文用 3D 轨迹和阶段末夹爪距离构造弱配对。附录表 6 给出温度 0.07、相对动作损失权重 0.1。 | 使用已有 Event 绑定和增强后的目标 ID。查阅时论文所列项目页返回 404，在第一作者公开仓库中未找到训练源码；这里的二维 ROI 特征和同图候选过滤属于 ESkill 设计。 [方法与附录](https://arxiv.org/html/2604.09824v1) · [作者公开仓库](https://github.com/nstrndrbi?tab=repositories) |

GuidedVLA 的 `compute_object_mass_loss` 对动作注意力落在目标内的总质量取负对数；相同的框内总质量可以对应不同的框内分布。该方法可作注意力质量诊断。MDETR 的公开损失提供了归一化向量、温度和有效正例掩码的实现参考，其双向 token–box 对齐使用另一套标签。[GuidedVLA 源码](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/models_pytorch/pi0_pytorch.py#L412-L455) · [MDETR 源码](https://github.com/ashkamath/mdetr/blob/main/models/mdetr.py#L461-L520)

## 两条路线共用的接口

继续输入主相机 RGB 和完整任务。冻结的 π0.5 一次前向提供图文融合特征，以及按行排列的融合前图像特征。复用 Current 的 16 个 256 维输出，沿 token 维平均得到一个目标查询 `q`。这些 token 当前用于解码 VAE 画布，其序号不代表输入图像的 16 个位置。实际空间网格来自视觉塔，224×224 输入产生 16×16 个 patch。当前两组 Subject 使用原始夹爪紧框和框回归损失。后续联合训练再接已有 Skill MoE 的预测表征。[Current/Subject 实现](../../src/openpi/models/grounding.py) · [模型输入](../../src/openpi/models/pi0_eskill.py)

**路线 A（单目标框）。** `MLP(q).sigmoid()` 输出 `[cx,cy,w,h]`。在有效 Current 样本上计算四坐标 L1 之和及 1−GIoU，GIoU 使用转换后的 xyxy。返回预测框及框内的 4×4 局部视觉 tokens，同时保留采样点在全图的归一化坐标。计算损失用原预测框，读取特征时取其与输入画面的交集。训练时直接对预测框施加定位损失，在线输出用预测框采样。

**路线 B（受监督的空间注意力）。** 用同一个 `q` 经线性投影与 256 个图像 patch 的 key 做缩放点积，直接保留 FP32 `log_softmax`。目标概率取最终 Current 紧框与各 patch 的相交面积并归一化，分母为零的样本跳过定位损失。对每个有效样本在 256 个位置求和得到 KL，再对有效样本平均。此设计借鉴 KLAL，并把监督位置移到可训练的 Current 读出。模型输出用同一张图读取融合前的视觉内容。

若下游需要固定数量的 tokens，B 首版可把 16×16 全图划为 4×4 个区块，返回每块的概率质量、以该概率为权重的局部特征及全图坐标。质量为零时特征置零。保留各块质量，避免在低关注背景块内部重新归一化后给予它与目标块相同的重要性。A 的 16 个位置是框内坐标，B 的 16 个位置是全图分块；本轮的共同指标直接评估定位，后续接 Motion 时单独检验读出的影响。

**GAC（训练时可开关）。** 对每个候选的 GT 框，从融合前图像特征用每个 bin 中心的双线性采样读取 4×4 特征后池化为一个向量，再做共享线性投影。同图换任务且输入像素相同时，各实体的 key 一致。将实体 key 与同一个定位查询 `q`做 L2 归一化，计算 `CE(cosine(q, keys) / τ, positive_index)`。分母包括本帧正例和有效负例。GT 实体表仅进入训练损失及离线评测；在线接口只接收图像和任务。完整损失为定位损失、固定的 Subject 框损失及可选的 GAC 项。温度可先取 0.07。GAC 权重需按本项目共享 Current 参数上的梯度检查，ProGAL 表中的 0.1 对应其动作目标。

## 数据与增强

1. Dataset 在 Copy-Paste 与文本替换完成后，从最终的 `boxes` 和 `key` 返回归一化实体框、候选有效性及正例索引。每张图按实体 ID 绑定，collate 时沿实体维补齐。Copy 资格、物体是否接触夹爪均不能替代视觉可区分性。负例来自同一张图。反事实替换源物体后，以替换后的目标作为正例。[数据入口](../../src/openpi/policies/grounding_dataset.py)
2. 完整屏内框继续监督定位，包括原生遮挡。GAC 首版只用可见证据充足的物理实体，例如原生可见比例至少 80%；排除无可见比例或被编辑、Patch Mask、裁切完全遮没的候选。父实体与其当前部件不互作负例；Copy 改变父位置后，未同步移动的子标注也不参与实体对比。纯空间 region 继续参与定位。正例无效或合格负例为零时，只跳过该样本的 GAC。
3. 按 RGB 的同一变换处理所有候选和正例。Copy 后的源框经过 resize/pad、镜像、裁切及旋转；先裁到有效裁切窗口，再旋转和裁到最终画面，取剩余区域的轴对齐框。使用这一框生成 A/B 的共同标签。`role_core_boxes` 保存最终变换后的坐标。回归可用部分出画框；GAC 只使用完整保留的候选。Patch 保护圈用于输入增强，定位 GT 每边留量为 0。[增强代码](../../src/openpi/augmentation.py) · [现有标签](../../docs/target_grounding.md)

## 已做的小检查

| 检查 | 结果及其影响 |
| --- | --- |
| 完整固定验证集，关闭增强，按主相机 resize 到 224，半开紧框与 14 像素网格比较 | 175 条轨迹、27,912 个有效 Current 框；1,947 个短边小于 14 像素，485 个完全不含 patch 中心。最高注意力 patch 中心在框内这一指标的理论上限为 98.26%。评测须报告该子集，并补充连续中心误差与峰值 patch 是否和框相交。 |
| CPU 梯度，Torch 2.7.1、torchvision 0.22.1 | 原生 `roi_align(aligned=True,sampling_ratio=1)` 对特征有梯度，对测试框坐标梯度为 None。4×4 bin 中心的 `grid_sample(align_corners=False)` 在内部测试框上最大前向误差约 1.53e−5，对坐标有非零梯度。GT ROI 可直接用 ROIAlign；若未来 Motion 损失须影响预测框，可用后一读取方式。[算子定义](https://docs.pytorch.org/vision/main/generated/torchvision.ops.roi_align.html) |
| 小框 KL 目标 | 两个处于同一个 patch 内但位置不同的小框给出相同离散目标；跨四个 patch 的小框给四处质量。三例目标概率和均为 1，反传得到有限非零梯度。单格监督能表达“在哪一格”，无法单独辨识格内边界。 |

## 实施与验收顺序

第一轮使用最终坐标系的实体表，已加入几何变换、反事实正例、无有效对比样本和 checkpoint 推理的回归测试。归约复用项目的全局有效样本计数，确保一个 rank 没有 GAC 有效样本时也能正常训练。在线推理单测直接调用 `infer(images, task)`，不构造任何实体标签。

完整规划包含 BBox、BBox＋GAC、KLAL、KLAL＋GAC。四组共用初始化来源、父轨迹 split、采样序列、增强和训练预算。另保留同紧框的 VAE Current 作参考。固定验证帧按同图换任务、同目标移位、未见场景组合分组。主指标为配对两条指令均选对的比例、预测位移与真实位移的差、跨场景实体命中率；重叠而无法判定唯一实体的样本单列。BBox 同时记录 IoU 与面积比，KLAL 记录目标内概率质量。给 GT 实体框的 GAC Recall@1 单列为训练机制诊断。在线定位效果以移除全部 GT 候选后的预测为准。

建议预先固定实体命中的评分规则。模型先独立预测，再由评分器把 A 的预测框按 patch 相交面积转成概率图，B 直接用输出概率图。对每个可区分实体的 GT 框也生成同网格分布，与预测图计算 Bhattacharyya 系数（逐 patch 将两分布乘积开方再求和），最高者作为该帧选中的实体。GT 在相同网格上不可区分时报告样本数及评测覆盖率。计算位移时，A 用预测框中心，B 用 patch 中心按预测概率加权。同图换指令的配对样本须共用完全相同的输入像素，关闭会随查询改变的 Patch 保护增强。
