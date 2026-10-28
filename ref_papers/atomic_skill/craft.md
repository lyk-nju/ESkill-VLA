# CRAFT：用动作监督检验技能表示能否跨执行复用

> **当前定位（2026-10-28）：核心 · 复用验证依据。** 用控制及合法条件交换检验技巧复用；operation–entity 与本项目对象无关技巧定义不同。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Same Scene, Different Task: Skill Alignment for Compositional Generalization in VLAs*，Taegeun Yang、Youngju Na、Yoonki Cho、Sung-Eui Yoon。arXiv 2610.00524v1，首次提交 **2026-09-30**，不是根据编号猜测的 10 月日期。

**来源和范围：** [元数据](https://arxiv.org/abs/2610.00524)、[原文 §2–4.4、§5.1、Appendix B.1](https://arxiv.org/html/2610.00524v1)。已读方法及相关设定，未查代码、未复现。核心，研究日期 2026-10-27。

## 核心机制与证据边界

skill queries 看视觉与语言，state queries 只看视觉；动作专家共同使用两组表示。训练交换不同执行中的 skill 表示，在动作预测误差上做对齐。反事实指令缺少直接动作标签时，借用所需技能已有演示的 state 表示与动作监督。梯度路径另作限制，避免动作专家/状态分支替技能表征吸收训练信号。

这里的 skill 是 **operation–entity**；实验主要固定操作顺序、重组实体。它没有证明纯动词表示跨任意对象和事件顺序都成立。π 系实现还包含两次共享 backbone 编码（Appendix B.1），不能默认计算开销不变。

## 与本项目最直接的关系

CRAFT 提供的是一个强竞争定义：把实体留在 skill 内部，仍可能改善组合。我们若显式区分 atomic skill 与 target grounding，就需要证明这种拆分在更广的对象交叉划分、局部空间精度或表示可迁移性上有收益。

值得移植的是**用控制结果检验表示交换**的思路。固定一个可行状态，将相同语义、不同对象条件下产生的 atomic skill 交换；target grounding 保持与该状态兼容。分别报告行为类型、交互位置、成功率，而非只展示聚类。

另设两种干预：只换 atomic skill，验证行为变化；只换 target grounding，验证目标变化。所有干预都需要合法的对应状态与监督，不能把另一场景的原始动作直接当本场景标签。上述实验是本项目建议，论文没有验证本项目三分结构。

## 文献关系

§2.1 明确引用 SkillVLA、SkillNet 和 ACT-VLA；已对读 ACT-VLA 与 SkillVLA 原文，见 [机制溯源证据](evidence.md#trace)。SkillNet 只读到正式摘要/作者方法页，不宣称全文关系已核。推荐与 LCSD、AtomSkill 和 X-Tokenizer 联合比较，以区分语言一致性、动作一致性和量化层级语义。
