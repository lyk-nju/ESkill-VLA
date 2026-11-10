# ESkill-VLA：研究主线

**E 代表 Event。** 项目研究如何从有限示范中学习可复用的视觉技能，并结合当前对象、任务目标与本体状态，生成能指导实际控制的交互运动。

当前设计为 **ESkill-VLA**：Skill → Goal-conditioned Motion → Action。本文说明研究假设；结构与实现计划见 [Design](design.md)。ESkill 代码已接入并通过技术回归，研究假设仍待真实训练与闭环实验验证。

## 研究问题

多任务示范中的同一技能会出现在不同对象、位置与目标关系中。我们希望策略能利用共享的操作知识，在 Object–Goal 关系改变时重新组织行为，而不只是复现训练组合。

现有辅助表征的预测误差下降，也不必然意味着动作专家使用了它。近期 Motion Donor 诊断提示需要检查表征变化能否带来实际运动变化；这不足以单独证明失败源于视觉捷径，更不能代替新的闭环验证。

## 三项建模选择

| 部分 | 学习内容 | 作用 |
| --- | --- | --- |
| Continuous Visual Skill | 每臂连续 Query，经语义路由的共享／专门化专家形成技能先验 | 在相同操作类别间共享参数，保留随视觉状态变化的连续信息 |
| Object State Grounding | 当前操作对象与同一对象的 Event 终点状态 | 表达对象应发生的变化，并绑定当前实例 |
| Event Interaction Motion | 从当前 EEF／夹爪状态出发，未来 40 个保存帧的路径与开合行为 | 将技能和目标实例化为可用于动作生成的局部计划 |

Event 是数据监督与目标关联的单元。Skill 是可复用的操作类别；同一 Skill 可以在一条轨迹内执行多次，各次 Event 的对象、目标与路径可以不同。

每次重新观察后，模型预测当前 Skill、Object 目标和局部 Motion，再生成动作并执行短前缀。当前 Event 的目标可以超出 40 帧窗口，Motion 无需一次覆盖整个 Event。

## 架构原则

- 使用六个有数据支持的小型技能专家及一个 Shared Expert，保留九类数据编号；训练和推理都由模型预测路由。
- Grounding 监督 Current／Future Object；当前 EEF 使用真实位姿和相机投影。
- Motion 每臂八个固定帧偏移槽位，联合监督二维 EEF 与夹爪命令，只使用当前 Event 内的有效未来标签。
- 只有高维 Motion 作为新增结构化条件进入 Action。Action 保留原始 π0.5 VLM 条件，训练时对其读取的视觉 K/V 做受控 Patch Masking。
- 辅助目标约束表征，真实控制效果决定方法是否成立。未来标签不进入在线输入。

## 需要验证的假设

1. **技能复用**：语义专家能否在新对象／目标组合上提供有用先验，而不仅提高当前技能分类准确率。
2. **目标绑定**：Future Object 能否随指令中的 Object–Goal 关系改变，而非只预测常见的训练终态。
3. **运动控制**：同状态、同噪声下改变 Motion，能否带来符合目标的 EEF 方向和夹爪时机变化。
4. **闭环收益**：这些机制能否提高组合任务表现，同时保持基础 LIBERO／Plus 的任务能力。

必须核对基座及后续各阶段的训练数据后，才能将某个组合称为未见组合。当前先做原生 π0.5 初始化的镜像先验 20k 与动作训练 40k，不带 Functional Reuse；复用作为后续独立对照。类别分工、视觉重建和二维轨迹监督都不能单独证明技能解耦或因果控制。

## 阅读顺序

- [Event Organization](event_organization.md)：事件、对象与时间的对应。
- [Atomic Skill](atomic_skill.md)：连续 Query 与小型 Skill-MoE。
- [Target Grounding](target_grounding.md)：当前／终点对象监督。
- [Action Representation](action_representation.md)：固定时间 Motion 与夹爪行为。
- [Design](design.md)：完整接口、梯度、训练推理与实现顺序。
- [Data](data.md)、[Evaluation](evaluation.md)：数据准备与行为验收。
- [参考工作](ref_papers/README.md)：原论文机制及本项目采用边界。
