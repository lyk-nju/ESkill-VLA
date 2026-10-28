# DASL：慢速技能与快速动作，不要求部署时生成轨迹

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Dual-Process Atomic Skill Learning: Decoupling Semantic Reasoning and Real-Time Control*，Jun Chen 等。arXiv 2607.10625；v1 **2026-07-12**，实读 v2 **2026-09-11**。

**来源和范围：** [元数据](https://arxiv.org/abs/2607.10625)、[原文 §2.2–2.3、§3.1–3.5](https://arxiv.org/html/2607.10625v2)。核心；核对方法，未复现。

## 核心机制与证据边界

低频策略以语言及稀疏观测预测 VQ 技能，在预设间隔内重复使用；高频 Decision Transformer 生成动作。latent trajectory diffusion 在训练中规整表示，部署时移除。总目标仍联合优化。低频与高频接口，不等于先后两阶段冻结训练，也不等于学到了事件边界。

§2.2 明确讨论 LISA、LCSD、SkillDiffuser、LADS。方法比较见 [机制溯源证据](evidence.md#trace)；不由这些引用推断所有模块都直接继承。

## 对本项目的价值

本项目可以把 action representation 的训练监督与部署作用拆开。若视频变化有助于形成运动知识，它既可以训练一个推理必经的 latent，也可以作为动作专家的辅助训练目标；两条路线的延迟、在线预测难度与误差传播不同。

建议在相同动作底座与数据预算下比较：直接动作专家、带辅助运动表征监督的专家、必须经过显式 action representation 的专家。保持 atomic skill / target grounding 输入一致，主要看闭环成功、组合泛化与延迟。只有后者带来额外收益时，才有理由扩大部署接口。

还需将固定间隔与事件结束驱动的切换分开测量。DASL 不能直接作为 Event-Centric Data Organization 已完成的证据。
