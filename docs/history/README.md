# 历史联合动作路线

当前设计见 [Current／Subject](../design.md)。这里保存 Skill→Current／Future→Motion→Action 路线的说明，用于已有 checkpoint、实验和诊断工具的追溯；不是当前 Grounding 专项的默认训练计划。

| 文档 | 内容 |
| --- | --- |
| [joint_design.md](joint_design.md) | 原始联合动作模型的信息流 |
| [atomic_skill.md](atomic_skill.md) | 六类 Skill-MoE 与连续技能先验 |
| [action_representation.md](action_representation.md) | 八点未来轨迹、夹爪命令、终点保持和 Action K/V |
| [joint_data.md](joint_data.md) | 动作窗口、未来标签、标定与历史镜像约定 |
| [joint_training.md](joint_training.md) | joint／stage1／stage2、初始化、损失和梯度诊断 |
| [libero_joint.md](libero_joint.md) | 原 LIBERO 数据、动作训练、服务与测评命令 |
| [joint_evaluation.md](joint_evaluation.md) | LIBERO／Plus、Motion Donor 与真实 rollout Attention 面板 |

文档整理只改变说明入口；运行时代码和 checkpoint 字段保留。旧配方仍需显式选择，当前 Grounding checkpoint 不能直接作为这些动作策略加载。具体历史结果保留在 [experiments/](../../experiments/README.md)，核验报告保留在各数据集 `check/`，论文参考位于 [ref_papers/](../ref_papers/README.md)。
