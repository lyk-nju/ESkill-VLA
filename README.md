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
| [Data](docs/data.md) | 各模块需要的外部数据 |
| [Evaluation](docs/evaluation.md) | LIBERO／LIBERO-Plus 环境、π0.5 服务与闭环测评 |

[ref_papers](ref_papers/README.md) 保存当前方法来源、按需参考卡与压缩背景笔记。交给实现 agent 时，先读 [Design](docs/design.md)，再按模块查参考；论文卡说明原方法和借鉴边界，不增加实现或实验要求。改变方法选择时同步更新方向文档与 Design，研究主张变化才修改 Idea。各方法仍待实现与实验验证。

当前 Stage 1 按臂组织并发 Event，每臂预测技能、三锚点和粗运动，网络共享并由单一动作专家联合执行。完善模块并跑通联合训练与闭环评测，以完整方案相对动作基线的平均任务成功率为主，不预设大批消融为完成条件。

数据方案见 **[docs/data.md](docs/data.md)**；候选来源、局部检查记录及来源字段讨论保留在 **[data/README.md](data/README.md)**。具体来源、训练规模与配比尚未确定。

## LIBERO／LIBERO-Plus 测评

已接入基于官方 OpenPi 的 π0.5 checkpoint 服务及两套独立仿真客户端。命令见 [LIBERO](examples/libero/README.md) 与 [Plus](examples/libero_plus/README.md)；官方源码固定版本和环境材料由工作区主仓库管理。15 项测评测试及 11 例零动作技术闭环通过，真实微调 checkpoint 与成功率待另行验证。
