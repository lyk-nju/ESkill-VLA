# Action Representation：方法来源索引

当前设计：[docs/action_representation.md](../../action_representation.md)。整体实现：[design.md](../../design.md)。本方向关注有序粗空间路径与机器人动作条件。

论文描述与项目实现分开维护。以下卡片提供机制依据；只有当前设计文档确定的模块、损失和接口才进入 Stage 1。

## 当前方法参考

| 工作 | 参考价值 |
| --- | --- |
| [RoboTracer：有序三维粗路径](robotracer.md) | 主要参考其有序、稀疏三维 Spatial Trace；本项目采用固定查询的相对 XYZ 辅助监督，具体接口见 [Action Representation](../../action_representation.md)。 |
| [WALL-WM：事件对齐的数据组织](wall_wm.md) | 当前仅借鉴事件对齐的数据组织，将标签关联到有语义意义的区间。原动作目标包含物理三维末端位姿和夹爪信息。 |

## 按需参考

| 工作 | 保留价值 |
| --- | --- |
| [LAPA：视频监督与机器人动作适配](lapa.md) | 保留未来从无机器人动作标签视频学习动作相关知识的出发点；中间监督目标与最终机器人动作空间可以不同。 |
| [ConLA：动作信息与外观捷径](conla.md) | 保留区分交互运动与静态外观的学习动机；同技能仍可能对应不同几何实现，动作类别一致不要求整个运动表示相同。 |
| [RynnLAM：三维运动与相机分支](rynnvla_latent.md) | 保留三维运动监督、相机运动与本体适配分别处理的接口价值，服务后续人类视频数据设计。 |
| [CoMo：连续运动与时序方向](como.md) | 保留有向运动与时序变化的监督价值。当前 Motion Queries 的进度编码表示路径次序；它与视频时序对比属于不同实现。 |
| [CLAP：动作锚定与人类视频教师](clap.md) | 保留“机器人动作锚定共享运动”与“人类视频自监督”使用不同监督来源的思路，帮助界定未来跨本体数据条件。 |
| [ActionPiece：运动几何差异的保留](actionpiece.md) | 保留同语义下细粒度运动差异的价值，支持 Atomic Skill 的类别共享与 Latent Motion 的具体几何信息分工。 |
| [X-Tokenizer：语义约束与运动细节](x_tokenizer.md) | 保留语义约束与精细运动重建可以作用于不同部分的认识，关联当前可复用 Atomic Skill 与条件运动 Latent Motion 的职责分工。 |
| [LAOM：控制相关监督与视觉干扰](laom.md) | 保留视觉重建可能容纳无关动态、运动监督能提供控制相关约束的证据。当前末端 XYZ 辅助目标直接表达控制相关空间信息。 |
| [Faster-WAM：条件缓存与动作融合](faster_wam_future.md) | 保留一次条件编码、在动作生成迭代中缓存复用的工程思路。当前缓存的是预测的 Atomic Skill、Target Grounding 与 Latent Motion 条件，不是视频 rollout。 |

## 跨方向关联

[AWE：路径几何与事件粒度](../event_organization/awe.md)（语义粒度与路径几何） · [3D HAMSTER：末端路径与深度尺度](../target_grounding/hamster_3d.md)（三维坐标与深度）

[背景机制笔记](background.md) 收录其余 24 项工作，供后续调研检索。

返回 [论文库入口](../README.md)。
