# SPiRL：技能身份稳定，不等于技能选择不依赖状态

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Accelerating Reinforcement Learning with Learned Skill Priors*，Karl Pertsch、Youngwoon Lee、Joseph J. Lim，CoRL 2020（论文集 2021）。arXiv:2010.11944；本轮核对 v1（2020-10-22），调研日期 2026-10-27。

**原始来源与阅读范围：** [论文 §3.2–3.3](https://arxiv.org/html/2010.11944v1)、[会议页面](https://proceedings.mlr.press/v155/pertsch21a.html)、[作者项目页](https://clvrai.github.io/spirl/)。已核对原版方法；后续代码库的闭环变体不自动计入此版本，未复现。

## 原工作做了什么

用固定长度动作片段学习连续高斯 VAE 技能空间，并学习状态条件 skill prior。下游高层策略通过 RL 在 latent 空间选择技能，同时向已学先验正则化；低层 decoder 执行动作片段。预训练不要求语义技能标签，但需要动作数据，下游有奖励与交互训练。

## 对 atomic skill / target grounding / action representation 的意义与边界

SPiRL 中的 skill 更接近连续运动抽象与可执行策略先验，不能直接当成本项目离散语义 atomic skill 的现成实现。它仍指出一个重要的定义问题：**技能是什么，与当前状态下什么技能可执行，是两个问题。**

我们希望 pick 的语义跨对象稳定，但预测当前是否应 pick、能否 pick，当然应依赖观测、接触和 target grounding。不能为了“对象无关”而要求整个技能选择器忽略状态，也不能仅因先验以状态为条件就判定它没有学到共享技能。

其先验约束可以提示我们如何避免在线选择离训练分布太远的 atomic skill / action representation；但 Phase 1 不必据此新增 RL。首先应分清表示迁移、行为克隆适配和交互强化三种收益来源。

## 可以区分机制的实验

固定 atomic skill 的语义词表和低层动作专家，比较无技能先验与当前状态/target grounding 条件的可行性先验；使用接触条件改变、目标不可达、同类对象位置变化等情形。分别检查语义身份是否稳定、技能选择是否校准以及任务成功率。

若状态先验改善控制但没有改变 atomic skill 表示的跨对象一致性，这仍是有价值的结果，只应把贡献表述为技能选择而非语义解耦。凡涉及新任务 RL，单列交互次数和奖励信息，避免与无更新迁移混比。

## 下一步精读

确认原版 latent 的时间跨度、prior 的训练目标及下游正则强度；阅读代码时明确使用原版还是后续闭环 decoder，避免把仓库新增能力归于原论文。

相关：[SAILOR 2022](sailor_2022.md)、[QueST](quest.md)、[方向论文索引](README.md)。
