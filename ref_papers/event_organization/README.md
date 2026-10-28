# Event Organization：按当前方法组织的论文索引

当前方法见 [docs/event_organization.md](../../docs/event_organization.md)，整体接口见 [design.md](../../docs/design.md)。

先读 AtomSkill / DeCo 的边界与语义，再读 WALL-WM 的事件时序组织；StructVLA / StaKe 支持结构监督，RDD 检查事件衔接。

本方向主卡：5 核心、11 补充、2 归档。跨方向工作只链接同一主卡，不重复计数。优先级更新于 2026-10-28；“核心”按当前设计用途选择，不代表已复现或整篇方法都会采用。

## 当前核心

| 工作 | 当前角色 | 直接服务什么设计 |
| --- | --- | --- |
| [AtomSkill 语义事件与技能对齐](atomskill.md) | 直接方法来源 | Event 分段与技巧语义；借鉴组织方式，不直接沿用动作量化码本。 |
| [DeCo：完整交互、关键帧与技能衔接](deco.md) | 边界依据 | 追踪 AtomSkill 的物理候选边界与完整交互划分来源。 |
| [StructVLA：把关键交互状态组织成稀疏未来](structvla.md) | 结构监督来源 | 以稀疏交互结构提供训练信号；物理关键帧不直接等于语义 Event。 |
| [StaKe：π0.5 的阶段与关键帧辅助监督](stake.md) | 轻量监督参照 | 阶段辅助头与关键帧监督；二元阶段预测不等于在线多类别技巧选择。 |
| [RDD：让分段对齐低层策略真正会做的事](rdd.md) | 衔接验证依据 | 检查事件进入条件与首尾交接的可执行性，不要求首版引入检索规划器。 |

## 跨方向核心接口

| 工作 | 当前角色 | 对本方向的作用 |
| --- | --- | --- |
| [WALL-WM：事件时序、剩余时间与变长运动](../action_representation/wall_wm.md) | 主要方法来源 | 事件对齐时序、剩余时间回归与长度分桶；粗 XYZ Token 是本项目的简化接口。 |

<details>
<summary>按需补充与替代路线（11 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [BUDS 从重复结构发现可复用技能](buds.md) | 按需补充与替代路线 |
| [UVD 通过视觉进展发现子目标](uvd.md) | 按需补充与替代路线 |
| [StageWAM：预测下一阶段的潜在状态](stagewam.md) | 按需补充与替代路线 |
| [TaKSIE：子目标选择、进度记忆和切换条件](taksie.md) | 按需补充与替代路线 |
| [HULC：没有显式事件边界也能学习层次计划](hulc.md) | 按需补充与替代路线 |
| [Hi Robot：从开放任务到可执行子指令](hi_robot.md) | 按需补充与替代路线 |
| [AWE：用重构误差定义 waypoint，而非预设语义边界](awe.md) | 按需补充与替代路线 |
| [Knowing When to Stop：把执行时长与语义事件边界分开](knowing_when_to_stop.md) | 按需补充与替代路线 |
| [ATOM-Bench：单项能力与组合表现的评测对照](atom_bench.md) | 按需补充与替代路线 |
| [依赖覆盖：正确 Event 阶段不等于充分任务信息](diagnosing_compositional_generalization.md) | 按需补充与替代路线 |
| [LOTUS：视觉分段与持续扩展技能库](lotus_continual.md) | 按需补充与替代路线 |

</details>

<details>
<summary>历史归档（2 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [RT-H：语言动作层级与技能粒度](../_archive/event_organization/rt_h.md) | 历史归档 |
| [OpenHOUSE：人类视频中的在线、多粒度事件理解](../_archive/event_organization/openhouse.md) | 历史归档 |

</details>

[历史溯源与核查](evidence.md)保留各轮证据；其中旧分层不代表当前阅读顺序。当前级别与调整前记录统一在 [catalog.json](../catalog.json)。

返回 [论文库入口](../README.md)。
