# Action Representation：按当前方法组织的论文索引

当前方法见 [docs/action_representation.md](../../docs/action_representation.md)，整体接口见 [design.md](../../docs/design.md)。

先读 RoboTracer 的有序稀疏 Spatial Trace，理解当前整段粗运动表示的来源；WALL-WM 提供 Event 组织参照，LAPA / ConLA 保留跨本体学习动机。时长与变长生成路线转为按需补充。

本方向主卡：4 核心、27 补充、4 归档。跨方向工作只链接同一主卡，不重复计数。优先级更新于 2026-10-28；“核心”按当前设计用途选择，不代表已复现或整篇方法都会采用。

## 当前核心

| 工作 | 当前角色 | 直接服务什么设计 |
| --- | --- | --- |
| [RoboTracer：稀疏三维 Spatial Trace](robotracer.md) | 主要表征参考 | 用有序粗空间路径连接任务理解与执行；固定查询辅助监督为本项目改造。 |
| [WALL-WM](wall_wm.md) | 事件组织参照 | Event 对齐视频与动作；当前不沿用时长头和变长生成塔。 |
| [LAPA 从视频构造伪动作监督](lapa.md) | 跨本体学习出发点 | 从无机器人动作标签视频获得动作相关监督；当前不直接采用其量化教师。 |
| [ConLA 用类别与时序先验削弱外观捷径](conla.md) | 跨本体学习约束 | 语义与时序先验、外观捷径问题；当前不直接引入其完整对比学习模块。 |

<details>
<summary>按需补充与替代路线（27 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [VH-Diffuser](vh_diffuser.md) | 按需时域参照 |
| [Adaptive Action Chunking](adaptive_action_chunking.md) | 按需时域参照 |
| [UniVLA 任务相关潜动作与显式解码](univla.md) | 按需补充与替代路线 |
| [XSkill 跨身体技能与示范对齐](xskill.md) | 按需补充与替代路线 |
| [RynnVLA-Latent / RynnLAM：三维运动监督与本体适配](rynnvla_latent.md) | 按需补充与替代路线 |
| [CLAM：连续潜动作与真实动作联合接地](clam.md) | 按需补充与替代路线 |
| [野外潜动作世界模型：迁移的可能是相机相对运动](latent_wm_in_the_wild.md) | 按需补充与替代路线 |
| [LaWAM：action representation 经未来视觉特征连接动作专家](lawam.md) | 按需补充与替代路线 |
| [LAC-WM：真实候选动作进入潜空间，再预测后果](lac_wm.md) | 按需补充与替代路线 |
| [Moto：运动 token 有训练价值，也可以在推理时省略](moto.md) | 按需补充与替代路线 |
| [RLA-WM：把状态转移残差压缩后建模](rla_wm.md) | 按需补充与替代路线 |
| [Fast-WAM：训练预测未来，测试直接控制](fast_wam.md) | 按需补充与替代路线 |
| [DreamDojo：人类视频迁移到机器人世界模型](dreamdojo.md) | 按需补充与替代路线 |
| [AdaWorld：预训练世界模型的可迁移动作接口](adaworld.md) | 按需补充与替代路线 |
| [What Do LAMs Actually Learn：低重建误差不能保证动作语义](what_lam_learn.md) | 按需补充与替代路线 |
| [LAOM：控制信息存在，不等于可用](laom.md) | 按需补充与替代路线 |
| [FAST：动作压缩的必要基线](fast.md) | 按需补充与替代路线 |
| [From Pixels to Tokens：把 teacher 来源与监督接口分开比较](pixels_to_tokens.md) | 按需补充与替代路线 |
| [X-Tokenizer：语义约束与运动残差可以分工](x_tokenizer.md) | 按需补充与替代路线 |
| [ActionPiece：同类动作的细微区别需要保留](actionpiece.md) | 按需补充与替代路线 |
| [UniLAP：从连续视频后验到可采样策略](unilap.md) | 按需补充与替代路线 |
| [Faster-WAM（Zhao）：单次未来条件与稀疏融合](faster_wam_future.md) | 按需补充与替代路线 |
| [Faster-WAM（Ma）：当前帧世界模型与浅动作专家](faster_wam_shallow.md) | 按需补充与替代路线 |
| [CoMo：连续运动教师与时序方向诊断](como.md) | 按需补充与替代路线 |
| [CLAP：机器人动作锚定与人类视频伪正例](clap.md) | 按需补充与替代路线 |
| [MV-LAM：多视角运动可观测性与动作监督上界](mv_lam.md) | 按需补充与替代路线 |
| [AffordanceWAM：哪些未来 World 信息应该传给动作](affordancewam.md) | 按需补充与替代路线 |

</details>

<details>
<summary>历史归档（4 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [UniPi 生成视频作为运动计划接口](../_archive/action_representation/unipi.md) | 历史归档 |
| [RynnWorld-Latent：把同一潜动作接到世界模型](../_archive/action_representation/rynnworld_latent.md) | 历史归档 |
| [DynaMo：动力学预训练也可以只改善视觉表征](../_archive/action_representation/dynamo.md) | 历史归档 |
| [IGOR：图像目标转移同时服务策略和世界模型](../_archive/action_representation/igor.md) | 历史归档 |

</details>

[历史溯源与核查](evidence.md)保留各轮证据；其中旧分层不代表当前阅读顺序。当前级别与调整前记录统一在 [catalog.json](../catalog.json)。

返回 [论文库入口](../README.md)。
