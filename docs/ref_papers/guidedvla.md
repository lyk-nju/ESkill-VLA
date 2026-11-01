# GuidedVLA：仿真验证与对照方法

核对日期：2026-10-30。已读论文 v2 的方法、仿真实验、数据构建和训练细节，检查公开配置、数据加载、技能变换与评测入口；另读取官方 LIBERO 发布包的元数据及一个实际数据文件。未安装其环境、加载模型或运行训练／评测。本文记录研究建议，不替换当前 Design。

[论文](https://arxiv.org/html/2605.12369v2) · [代码](https://github.com/GuidedVLA/GuidedVLA) · [LIBERO 数据](https://huggingface.co/datasets/ybwowen/libero) · [已发布权重](https://huggingface.co/ybwowen/pi0-libero-object-depth-skill)

代码固定 revision：`04be059e0d6bd448be5cb45fdbafc775f7eb5e38`。

**2026-10-30 用户确定的当前范围：基座统一 π0.5；暂不复现 GuidedVLA，先参考其数据构建和训练方法建立 ESkill-VLA 的仿真数据链路。** 下文三组对照为先前建议，不是当前执行任务；不直接照搬其注意力头、技能分布或深度分支。RoboTwin 的候选构建方案见 [数据草案](../../data/robotwin/construction_plan.md)。

## 它做了什么，与我们的区别

GuidedVLA 基于 OpenPI，把对象、技能和几何相关信息明确引入动作解码器。它采用注意力头分工，并提供 ControlNet 风格的零初始化残差分支。

| 方向 | GuidedVLA | ESkill-VLA 当前设计 |
| --- | --- | --- |
| 对象 | 用阶段相关对象／目的地区域约束动作注意力，损失为区域内注意力总量的负对数 | 每臂预测当前执行器、当前对象、同一对象事件终态的三组视觉 Token，使用局部视觉 latent 去噪监督 |
| 技能 | 汇聚指定注意力头的特征，预测未来动作窗口中技能 ID 的分布，以 KL 监督 | 每臂识别当前 Event 的共享技能类别，查可训练 SkillEmbedding，并显式条件化目标、运动及动作 |
| 空间 | 冻结 Depth Anything V3 编码当前 RGB，深度特征作为特定头的 K/V；没有三维路径损失 | 监督 Event 内真实剩余末端三维粗路径，学习 Latent Motion |

它与我们关注的信息高度相关，因此是有价值的强对照；机制相似不能证明我们的 Event、未来目标或运动 Token 已经得到验证。GuidedVLA 的深度分支还在推理时运行，比较时需记录推理成本。

官方 LIBERO 数据中的技能实际为：`other`、`approach target`、`interact at target`、`transport with object`，不是现成的 Pick／Place／Hold 事件词表。源码 `ComputeSkillSoftLabel` 对动作窗口内 skill_id 计数并归一化；这与我们对当前事件的硬类别监督不同。

## 仿真设置与发布情况

| 路线 | 论文设置／现有发布 | 对 Stage 1 的意义 |
| --- | --- | --- |
| LIBERO → LIBERO-Plus | 在 LIBERO 训练，在 LIBERO-Plus 测试七类扰动；已发布带 mask／技能的 LeRobot v3 数据和完整方法权重 | 公开材料较齐，方便核对官方实现；Panda 单臂，不能覆盖双臂并发 |
| RoboTwin 2.0 | 8 个任务，每任务 1,000 条 randomized 示范，100 次评测／任务；训练数据在仓库 release checklist 中仍为未发布 | 与我们的双臂目标更接近，可以在本地共同数据上重训对照；现有 clean50 不等同于论文训练集 |

八个任务对应本地已有任务目录：adjust_bottle、beat_block_hammer、click_bell、dump_bin_bigbin、move_playingcard_away、lift_pot、place_burger_fries、place_can_basket。

论文作者报告 LIBERO-Plus 的 π0 68.2% → GuidedVLA 75.4%，RoboTwin 平均 77.38% → 90.63%。均为作者报告，未在本地复现；本地不同数据预算下的结果不能直接与这些数字比较。

论文 LIBERO 配方先做无 guidance 微调，再继续训练 guidance；使用 batch=64、chunk=50、8×H200。RoboTwin 描述为 30k steps、batch=16、chunk=50、4×H200。公开默认配置不是完整论文协议的自动执行器，不能只复制命令就宣称复现。

## 后续纳入 GuidedVLA 对照时的比较条件（暂不执行）

若后续开展三组比较，分别为纯动作基线、GuidedVLA、ESkill-VLA。统一基座、初始化 checkpoint、训练数据与拆分、示范预算、视角、动作变换、训练步数和评测种子；GuidedVLA 与我们额外使用的标注来源各自说明。

- 论文主要结果基于 π0。若我们的首轮继续采用 π0.5，应使另外两组也采用同一 π0.5 初始化，并称为迁移后的对照实现；不能拿不同基座的数字直接归因于新方法。
- 若采用论文两阶段方案，所有组的共同预微调和后续优化预算需明确，避免只有某组多训练一阶段。仿真优先指本项目新增训练优先使用仿真示范，并不抹去 π0／π0.5 基座已有的机器人预训练。
- 原生 LIBERO 先看基础任务闭环结果；LIBERO-Plus 是 OOD 评测，不额外从中取训练标签。是否做完整扰动评测随具体方案确定，不作为当前启动门槛。

## 已发现的配置差异

1. [LIBERO 默认配置](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/config.py#L745)：动作基线指向 physical-intelligence/libero，完整 GuidedVLA 指向 ybwowen/libero；基线默认 batch 也不同。公平比较应显式统一数据、batch 和训练预算。
2. [PyTorch 初始化](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/scripts/train_pytorch.py#L899)：PyTorch 不使用配置中的 JAX weight_loader，需显式设置 pytorch_weight_path。源码会提示缺失时从零初始化，不能误以为已经加载 π0 基座。
3. [LIBERO 动作变换](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/config.py#L434)：代码说明原始动作已经是 delta，但公开实验配置启用 extra_delta_transform。与发布 checkpoint 或自训基线对齐前，要确认这套额外转换与归一化，不能自行删改后仍称原协议。
4. [RoboTwin 配置](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/config.py#L1108)：完整方法默认 batch=32，而论文写 batch=16；损失权重为对象／技能各 0.01。源码已提供 14 维双臂关节／夹爪适配与 delta→absolute 回转，但额外标注仍需准备。
5. [验证划分](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/data_loader.py)：_split_dataset 对整库行索引随机划分，邻帧及重叠动作窗口可能进入两侧。它不能作为独立轨迹泛化的证据；我们采用哪种划分应写进对照协议，所有组保持一致。

## 对仿真优先的建议

若首轮保留双臂研究目标，优先考虑本地 RoboTwin 数据，对同一批数据分别训练三组；LIBERO 可用于核对官方公开实现。若优先降低官方复现的准备工作量，则可以先单臂 LIBERO，再进入 RoboTwin，但单臂结果不能验证双臂并发设计。

仿真仍需核对监督是否齐全。本地 RoboTwin 提供 Segmentation 纹理和 actor／mesh 分割读取入口，但现有封装会转成伪彩色图；若从仿真补对象／夹爪区域，需要保留原始实例 ID 并建立 actor／link 映射。已有 clean50 文件不会自动多出 mask；重新回放或采集前应另写方案。

数据字段初查见 [LIBERO 候选目录](../../data/libero/guidedvla/README.md)。正式选任务、预算、训练与闭环评测后，按工作区实验规则建立具体方案并确认；本轮只完成论文、代码和有限样本核对。

## 数据管线专项核对

2026-10-30 用户进一步明确：分别构建 LIBERO 和 RoboTwin 两套仿真数据引擎，最后各抽 10 条渲染并人工验收。专项记录分别放在 [LIBERO](../../data/libero/guidedvla/guidedvla_pipeline.md) 和 [RoboTwin](../../data/robotwin/guidedvla_pipeline.md)，保留来源差异和未公开步骤。

从公开实现看，它们共用“阶段相关标注 → LeRobot → 数据适配 → 训练加载”的逻辑，而非两套已公开的端到端标注引擎。主仓库公开了标注结果的消费接口；固定 RoboTwin 子模块还有基础示范转换器，但未提供完整阶段切分、Qwen3-VL／SAM2 批量标注及人工修订入口。阶段起止如何产生，是目前不能从发布材料补成既定算法的环节。

加载部分可借鉴以下组织方式，但不能原样作为我们的标签契约：

| 环节 | 公开实现 | 我们的处理依据 |
| --- | --- | --- |
| 时间窗口 | 根据元数据 fps 读取未来 H 帧动作和 skill_id | 保留来源真实帧映射；action horizon 不等于 Event 区间 |
| 技能目标 | 窗口内类别直方图，原 skill_id 替换为窗口中点 ID | 我们监督当前时刻每臂 Event 技能，需单独组织 |
| 区域目标 | 两来源字段统一成三视角 × 256 patch，另附视角有效性 | 视角轴与每臂三角色轴不同；需保留像素区域及对象关联 |
| 缺失标签 | loader 拒绝整批没有有效非零对象区域；loss 对没有对象权重的样本不计入分母 | 本项目逐臂、逐角色标有效性，保留仍有真值的动作监督 |
| 转换顺序 | repack → 来源适配／delta → normalize → resize／tokenize／pad | 来源动作各自适配；统计应来自实际训练 split |
| 验证拆分 | 对整库行索引随机分 train／val；norm_stats 脚本默认读取 all | 本项目先按完整 Episode 拆分，避免相邻窗口泄漏，统计仅用训练数据 |

另一个需核验的空间对应问题：[训练前向](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/models_pytorch/pi0_pytorch.py#L850) 会调用图像随机几何增强，而对象区域另行传递；在该路径未见同步变换区域的实现。这里只是静态代码检查，未量化其影响。我们构造三角色监督时，需要明确图像、区域、相机与画布各使用哪次几何变换。

加载证据：[时间窗口与行拆分](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/data_loader.py#L181)、[变换顺序](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/data_loader.py#L281)、[区域打包与检查](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/data_loader.py#L885)、[归一化统计入口](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/scripts/compute_norm_stats.py#L36)。
