# ESkill-VLA

**E** 代表 Event。项目以一个 Event 对应一次 Skill 执行为核心，学习技能、交互目标与运动实现。

研究怎样的数据组织与学习目标，能够让机器人从有限示范中理解任务目的，形成可泛化、可依据实际效果持续推进的行为能力。

从 **[docs/idea.md](docs/idea.md)** 开始。文档按“研究主线 → 各方向方法 → 整体实现”组织：

| 文档 | 职责 |
| --- | --- |
| [Idea](docs/idea.md) | 稳定的研究问题、归纳偏好、能力目标与验证假设 |
| [Event Organization](docs/event_organization.md) | 事件怎样定义、为什么这样组织、参考来源 |
| [Atomic Skill](docs/atomic_skill.md) | 语义分类、可训练技能先验、监督理由与参考来源 |
| [Target Grounding](docs/target_grounding.md) | 三视觉锚点的当前设计、监督理由与参考来源 |
| [Action Representation](docs/action_representation.md) | 事件级整段粗运动的当前设计、监督理由与参考来源 |
| [Design](docs/design.md) | Stage 1 架构图、张量接口、数据结构、梯度及训练／推理流程 |

[ref_papers](ref_papers/README.md) 保存详细论文卡、原文证据与方法溯源。阅读结果先沉淀在论文库，改变当前方法选择时再更新方向文档与 Design；只有研究主张变化才修改 Idea。各方法仍待实现与实验验证。

当前 Stage 1：完善模块并跑通联合训练与闭环评测，以完整方案相对动作基线的平均任务成功率为主，不预设大批消融为完成条件。
