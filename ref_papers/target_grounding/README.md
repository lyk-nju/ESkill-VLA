# Target Grounding：按当前方法组织的论文索引

当前方法见 [docs/target_grounding.md](../../docs/target_grounding.md)，整体接口见 [design.md](../../docs/design.md)。

以 ReconVLA 的局部去噪为主要监督来源，AffordanceVLA 提供角色化 query 与空间条件参照。三视觉锚点及终点目标是本项目扩展。

本方向主卡：2 核心、13 补充、3 归档。跨方向工作只链接同一主卡，不重复计数。优先级更新于 2026-10-28；“核心”按当前设计用途选择，不代表已复现或整篇方法都会采用。

## 当前核心

| 工作 | 当前角色 | 直接服务什么设计 |
| --- | --- | --- |
| [ReconVLA 用当前区域重建监督感知](reconvla.md) | 直接方法来源 | 训练期局部视觉 latent 去噪；三锚点及事件终点目标是本项目扩展。 |
| [AffordanceVLA：ROI、接触位置和对象几何共同连接动作](affordancevla.md) | 直接设计参照 | 按交互职责组织 query、区域监督与动作条件；不默认移植其所有空间头。 |

## 跨方向核心接口

| 工作 | 当前角色 | 对本方向的作用 |
| --- | --- | --- |
| [AR-WAM：离散操作与视觉接地条件](../atomic_skill/ar_wam.md) | 直接接口参照 | 离散操作嵌入与对象接地分别条件化策略；外部技能条件不等于自主技能预测。 |

<details>
<summary>按需补充与替代路线（13 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [RoboPoint 空间关系与交互点监督](robopoint.md) | 按需补充与替代路线 |
| [RT-Affordance 关键交互姿态作为条件](rt_affordance.md) | 按需补充与替代路线 |
| [RoboGround 的对象与放置区域接地](roboground.md) | 按需补充与替代路线 |
| [GraspVLA 的感知与动作监督对齐](graspvla.md) | 按需补充与替代路线 |
| [HAMSTER 的二维轨迹引导接口](hamster.md) | 按需补充与替代路线 |
| [3D HAMSTER 的度量轨迹与坐标一致性](hamster_3d.md) | 按需补充与替代路线 |
| [3D-LOTUS++ 的技能与对象几何接口](lotus_3d.md) | 按需补充与替代路线 |
| [VISTA 的视觉依赖与轨迹偏好训练](vista_visual_conditioning.md) | 按需补充与替代路线 |
| [Mind-VLA：按阶段对象建立空间监督](mind_vla.md) | 按需补充与替代路线 |
| [AffordVLA：以功能区域教师监督替代 ROI 重建](affordvla.md) | 按需补充与替代路线 |
| [RIGID：对齐空间关系是否比对齐特征更稳](rigid.md) | 按需补充与替代路线 |
| [Encoded but Not in Control：知道目标与按照目标行动分开测](encoded_not_control.md) | 按需补充与替代路线 |
| [SG-VLA：当前空间监督与分阶段梯度介入](sg_vla.md) | 按需补充与替代路线 |

</details>

<details>
<summary>历史归档（3 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [MimicPlay 的三维手轨迹与潜在计划](../_archive/target_grounding/mimicplay.md) | 历史归档 |
| [SuSIE 的未来子目标图像与控制](../_archive/target_grounding/susie.md) | 历史归档 |
| [CoT-VLA 的未来图像与动作联合预测](../_archive/target_grounding/cot_vla.md) | 历史归档 |

</details>

[历史溯源与核查](evidence.md)保留各轮证据；其中旧分层不代表当前阅读顺序。当前级别与调整前记录统一在 [catalog.json](../catalog.json)。

返回 [论文库入口](../README.md)。
