# ESkill-VLA：方法参考库

当前架构以 [Design](../design.md) 为准，研究假设见 [Idea](../idea.md)。论文卡保留原方法与采用边界，不将参考工作中的模块、训练阶段或结果自动视为本项目实现。

## ESkill 阅读入口

| 模块 | 当前设计 | 主要参考 |
| --- | --- | --- |
| [Event](event_organization/README.md) | 每臂事件、实例绑定与真实终点；事件内局部运动监督 | [AtomSkill](event_organization/atomskill.md)、[DeCo](event_organization/deco.md)、[WALL-WM](action_representation/wall_wm.md) |
| [Skill](atomic_skill/README.md) | 连续 Query、六类预测路由、6+1 小型专家；Skill 经 Motion 影响 Action | [AtomicVLA](atomic_skill/atomic_vla.md)、[w²VLA](atomic_skill/w2vla.md)、[CRAFT](atomic_skill/craft.md) |
| [Grounding](target_grounding/README.md) | Current／Future Object 各 16×256；冻结 FLUX VAE 目标与可训练投影头 | [AffordanceVLA](target_grounding/affordancevla.md)、[ReconVLA](target_grounding/reconvla.md) |
| [Motion](action_representation/README.md) | 每臂八个固定帧偏移查询，未来 40 个保存帧；联合监督 EEF 2D 与 Gripper Command | [SeeTraceAct](https://arxiv.org/html/2606.02745v2)、[RoboTracer](action_representation/robotracer.md) |

[GuidedVLA](guidedvla.md) 保留仿真数据、训练接口与空间对齐的参考。上述工作不构成自动启动的复现清单；ESkill 的参数、信息流与验收以本项目设计为准。

## 背景与维护

| 方向 | 重点 | 背景笔记 |
| --- | --- | --- |
| Event | 事件粒度、交接、完整任务上下文 | [Event](event_organization/background.md) |
| Skill | 语义路由、连续先验、选择与梯度边界 | [Skill](atomic_skill/background.md) |
| Grounding | 实例、目标状态、几何与监督 | [Grounding](target_grounding/background.md) |
| Motion | 动作相关表示、时间与空间、条件缓存 | [Motion](action_representation/background.md) |

原论文机制保留原义，本项目采用方式随 Design 同步；被替换的项目设计从正文移除，历史由 Git 保存。区分原文、代码静态检查、复现实验与未核验内容；不把重建或探针改善写成控制收益。

2026-11-10 同步 ESkill 采用方式；论文的原阅读范围与版本仍见各卡片。
