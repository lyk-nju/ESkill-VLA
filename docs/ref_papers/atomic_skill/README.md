# Atomic Skill：方法来源索引

当前设计见 [Atomic Skill](../../atomic_skill.md) 与 [Design](../../design.md)。ESkill 使用连续视觉 Query、六类语义路由及共享／专门化小型专家，技能先验通过 Motion 参与动作生成。

## 当前方法参考

| 工作 | 参考价值与边界 |
| --- | --- |
| [AtomicVLA：技能条件与参数路由](atomic_vla.md) | 语义 Top-1 路由与共享专家；本项目将小型专家放在 VLM 输出端，而非 Action 内部 FFN |
| [w²VLA：条件调制与视觉遮蔽](w2vla.md) | FiLM 与训练时视觉遮蔽；本项目使用连续 Skill Prior 与 Action-side 原始视觉 K/V mask |
| [CRAFT：技能表征与动作学习](craft.md) | 技能必须参与控制；本项目通过 Skill→Motion→Action 的连续路径接受动作梯度 |

## 按需参考

| 工作 | 保留价值 |
| --- | --- |
| [AR-WAM：可学习操作条件](ar_wam.md) | 操作知识与对象空间条件的分工；外部操作指令与本项目自主识别不同 |
| [LADS：目标表示与在线预测](lads.md) | 连续表示、在线类别选择及离散路由的梯度边界 |
| [DASL：训练辅助与部署条件](dasl.md) | 辅助读出与部署中间表征的区别 |

[AtomSkill](../event_organization/atomskill.md) 提供事件语义参考，[StaKe](../event_organization/stake.md) 提供训练辅助与部署 token 的机制参考。[背景笔记](background.md) 保留其他路线。

论文原方法不增加 ESkill 实现要求。返回 [论文库](../README.md)。
