# SAILOR 2022：技能检索与在线可预测性也是重要对照

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Learning and Retrieval from Prior Data for Skill-based Imitation Learning*，Soroush Nasiriany、Tian Gao、Ajay Mandlekar、Yuke Zhu，CoRL 2022（论文集 2023）。arXiv:2210.11435；本轮核对 v2（2022-11-15），调研日期 2026-10-27。

**原始来源与阅读范围：** [论文 §3–4.2](https://arxiv.org/html/2210.11435v2)、[会议页面](https://proceedings.mlr.press/v205/nasiriany23a.html)、[作者项目页](https://ut-austin-rpl.github.io/sailor/)。已核对技能模型、检索与目标任务训练，未复现。勿与 NeurIPS 2025 的同名 SAILOR 混淆。

## 原工作做了什么

从固定时窗示范学习 VAE 技能、观测条件低层策略和起止观测条件先验；额外预测子轨迹间的时间间隔。根据目标示范的技能相似性检索已有数据，再训练目标策略和调整技能模型。部署由观测历史预测 latent，低层策略在窗口内闭环执行。新任务需要示范与参数更新。

## 对 atomic skill / target grounding / action representation 的意义与边界

SAILOR 的价值不只在“有一个 skill latent”。它把数据复用和技能表示连接起来，提醒我们在增加 atomic skill / target grounding / action representation 结构前比较更简单的数据检索方案。同样的数据总量，不代表同样有效的训练分布。

另一个要点是 posterior 能从完整示范提取技能，并不保证在线 policy 能从当前历史预测它。本项目由未来帧训练的 action representation teacher 也有这一风险；重建误差下降而在线 prior 不可预测，可能无法转化为控制收益。

时间可预测性可能编码任务进度或典型顺序。在重新排列 Event 时，这既可能帮助长时序控制，也可能限制技能身份的不变性；这是待检验的项目推论，不是该论文已经证明的缺陷。

## 可以区分机制的实验

保持被检索数据量和训练步数相同，比较不检索、随机匹配检索、技能空间检索；再比较语义 atomic skill、运动 action representation 和视觉特征作为检索依据。评测已见顺序与未见 Event 顺序，并单列目标示范数和参数更新量。

同时比较 teacher latent 与在线预测 latent 条件下的控制表现。若大部分差距来自在线预测，优先修复训练—部署条件差异，而不是扩大 latent 容量或增加更多教师监督。

## 下一步精读

核对时间间隔目标的采样方式、检索阈值与数据配比、技能微调范围。明确哪些成功率来自更相关的数据，哪些来自表示与策略改进。

相关：[SPiRL](spirl.md)、[QueST](quest.md)、[方向论文索引](README.md)。
