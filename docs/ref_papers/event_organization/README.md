# Event Organization：方法来源索引

当前设计：[docs/event_organization.md](../../event_organization.md)。整体实现：[design.md](../../design.md)。本方向关注 Event 的语义边界、交接与标签组织；当前按臂维护事件流，保留跨臂并发和共享目标关系。

论文描述与项目实现分开维护。以下卡片提供机制依据；只有当前设计文档确定的模块、损失和接口才进入 ESkill。

## 当前方法参考

| 工作 | 参考价值 |
| --- | --- |
| [AtomSkill：事件边界与技能语义](atomskill.md) | 提供 Event 边界与技能语义分别组织的依据。一个 Event 对应一次 Skill 执行，同一 Skill 类型可跨 Event 复用；当前采用脚本／状态记录得到可核验标签。 |
| [DeCo：完整交互与事件衔接](deco.md) | 说明 Event 的粒度与进入状态、结束效果相关，而非按固定帧数或每次夹爪跳变机械切割。AtomSkill §III-A 明确引用其分段方法。 |

## 按需参考

| 工作 | 保留价值 |
| --- | --- |
| [StructVLA：稀疏结构的训练监督](structvla.md) | 提供通过训练监督吸收任务结构的实例。物理里程碑与语义 Event 可以具有不同粒度，稀疏结构监督也不要求部署时生成完整未来视频。 |
| [StaKe：辅助监督与部署 Token](stake.md) | 直接说明“训练期辅助头监督、部署保留中间 token”的接口如何成立，关联当前／终点对象和 Motion tokens 的训练／推理分工。 |
| [RDD：事件交接与可执行性](rdd.md) | 保留事件交接状态与低层可执行分布之间的关系；语言上合理的边界未必产生适合后续控制的初始状态。 |
| [AWE：路径几何与事件粒度](awe.md) | 帮助区分 Event 语义粒度与粗路径几何精度：一个语义事件可以包含多个必要转弯，几何静止也可能伴随抓放语义变化。 |
| [任务依赖：技能类别与完整上下文](diagnosing_compositional_generalization.md) | 解释当前架构为什么保留全局任务上下文进入动作专家；可复用的技能类别不意味着当前行为只依赖孤立的子任务。 |

## 跨方向关联

[WALL-WM：事件对齐的数据组织](../action_representation/wall_wm.md)（Event 对齐数据） · [LADS：目标表示与在线预测](../atomic_skill/lads.md)（离线目标与在线预测）

[背景机制笔记](background.md) 收录其余 11 项工作，供后续调研检索。

返回 [论文库入口](../README.md)。
