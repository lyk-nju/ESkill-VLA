# Target Grounding：当前与终点对象状态

ESkill 使用 **Current Object／Future Object 两组视觉 tokens**，为 Motion 提供操作对象的当前状态与完整 Event 目标。结构与实现状态见 [Design](design.md)。

## 两组目标的语义

| 分支 | 标签来源 | 含义 |
| --- | --- | --- |
| Current Object | 当前主相机 RGB＋当前 Event 对象 bbox | 当前操作实例／部件的视觉状态 |
| Future Object | 当前 Event 的 `goal_frame` RGB＋同一对象 bbox，目标区域额外外扩 | 此次 Event 记录的终点状态及邻近落位上下文 |

Future Object 始终属于当前操作对象，不用容器或参照物替代，也不是固定 40 帧后的对象状态。同一对象先 Pick 后 Place 时，每个 Event 使用自己的终点。

原始 Event 不必包含完整释放；目标是已核验记录中的阶段终点，不额外补造成功终态。已知不可信终点只排除相关目标监督，不连带丢弃其他有效字段。

当前 EEF 位姿、实际夹爪状态及其相机投影直接进入 Motion。夹爪 bbox 可继续用于数据检查，但不属于本模块的训练目标。

## 预测接口

每臂 Current／Future 各 16 个 256 维 Query，合并为 `[B,2,2,16,256]`。共享两层 Grounding Decoder 读取当前 VLM 上下文；同一臂两组 Query 可交换信息，以 role／arm embedding 区分。

Grounding 与 Skill 并行预测，两组 Object tokens 进入 Motion。训练与推理均使用预测 tokens，不把真实终点图像或 VAE latent 输入 Motion／Action。Action 不直接读取 Object tokens。

## 原位画布与完整屏内区域

只对主相机做区域监督，VLM 仍可使用腕部 RGB。区域采用 `full_in_frame`：指定对象几何独立渲染，忽略外部遮挡，裁掉屏外部分，再由 mask 求紧致 bbox。完全出画时 bbox 无效；被遮挡本身不使框缩小。

Current 沿用已有画布：bbox 每边扩展 4 个源图像素，在原图位置保留框内 RGB，框外置黑，保持比例 resize／pad 到 256×256。对象不裁剪放大到中央。

Future 在上述范围上，按 **256×256 Grounding 画布每边额外增加 8 px**，即宽、高各额外增加 16 px；从同一终点 RGB 保留新增区域，监督权重同步外扩。区域裁到有效图像范围，不扩入补黑边。这样可保留部分承托面、容器边缘等局部上下文，但不保证包括完整参照物。

配置为 `model.future_bbox_expand_px=8`，单位是最终 Grounding 画布像素，设为 0 即关闭额外外扩。源 bbox、实例绑定与 `goal_frame` 保持原值；仅在训练目标变换时生成扩大后的区域。原图中的背景、遮挡和遮挡物仍是真实观测内容；完整 bbox 不等于无遮挡外观真值。

区域有效性、实例绑定与目标帧有效性分别控制监督。当前对象不可见或出画，不应自动否定可用的 Event 终点目标。

## FLUX VAE 潜变量监督

沿用冻结 FLUX VAE 编码标签，FP32、posterior mode 和固定 scaling／shift；权重版本写入实验配置。**冻结的是 VAE，共享预测头参与训练。**

目标 latent 默认 `[16,32,32]`。16 个视觉 tokens 对应 4×4 网格，各自经共享 LayerNorm＋MLP（256→512→`16×8×8`）输出局部 latent patch，再按空间位置拼接。

Current／Future 分别计算区域加权 MSE，前景／背景默认权重 1／0.1；区域权重通过同一画布变换对应到 latent 网格。只在各角色有效目标上归一化。默认值沿用现有监督构造，ESkill 的损失总权重仍需在 recipe 中确定。

视觉损失更新 Decoder 与预测头；Action FM 可通过 Motion 更新 Grounding tokens 的连续生成路径。VAE 目标停止梯度。纯动作推理不运行 VAE 或 latent 读出头，诊断时才解码比较。

## 采用边界与核验

[AffordanceVLA](ref_papers/target_grounding/affordancevla.md) 提供 Query→FLUX latent 的直接监督参考；本项目采用两组对象状态、原位画布与 Event 终点目标，不能将原论文结果直接归为这些改造的证据。[ReconVLA](ref_papers/target_grounding/reconvla.md) 提供操作区域视觉监督的动机。

先检查同一实例的当前／终点绑定、画布位置与真实遮挡，再检查预测误差。视觉重建准确不代表目标组合泛化；还需验证不同指令下的目标变化以及其通过 Motion 对实际动作的影响。数据约定见 [Data](data.md)，参考汇总见 [索引](ref_papers/target_grounding/README.md)。
