# LCSD：语言重建能建立技能，也可能混入对象参数

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Rethinking Mutual Information for Language Conditioned Skill Discovery on Imitation Learning*，Zhaoxun Ju 等，ICAPS 2024，正式发表于 2024-05-30。arXiv:2402.17511；本轮核对 v1（2024-02-27），调研日期 2026-10-27。LCSD 为论文的方法简称。

**原始来源与阅读范围：** [论文 Approach / Mutual Information Skill Learning / Diffusion Policy](https://arxiv.org/html/2402.17511v1)、[ICAPS 页面](https://ojs.aaai.org/index.php/ICAPS/article/view/31488)。已核对方法，未检查代码或复现。

## 原工作做了什么

输入是带整条语言指令的离线状态—动作轨迹，不要求逐帧技能标签。状态与语言产生量化技能；连续相同码被压缩，用技能表示恢复指令的 CLIP 语义。论文还使用码本重初始化，并训练状态与技能条件的 diffusion policy。语言恢复 decoder 是训练辅助模块，不是机器人动作 decoder。

## 对 atomic skill / target grounding / action representation 的意义与边界

LCSD 直接对应“用语言组织离散技能”的问题，应与 LISA、AtomSkill 放在一起阅读。它不是完全没有标注：轨迹级指令本身就提供监督；也不能从重建损失直接推导出对象无关的语义技能。

**本项目的关键张力是：完整指令包含行为、对象和关系，而我们希望将其中部分信息交给 target grounding。** 如果要求 atomic skill 单独重建整句“把红杯放到左侧托盘”，模型有动力把红杯与左侧关系也写进 atomic skill。这是目标函数层面的潜在冲突，不是已经观察到该论文出现信息泄漏。

另外，连续相同 atomic skill 不代表同一个 Event：先推 A 再推 B 可以具有相同动作类别，却有目标绑定或事件实例的改变。不能仅以技能码变化定义事件边界。

## 可以区分机制的实验

固定数据、码本大小和模型，比较完整指令恢复、只恢复行为语义、由 atomic skill / target grounding 联合恢复指令三种监督。对任务语句拆分的质量也应抽样审查，避免“动词标签”遗漏工具或接触语义。

测试同动词不同对象、同对象不同动词以及连续同技能切换目标；同时评价 atomic skill 的可复用性、target grounding 的目标绑定、闭环成功率。若全指令恢复表现最好且没有损害组合泛化，应允许语义分工适度重叠，而非强求统计独立。

## 下一步精读

检查语言恢复究竟消费哪些时序技能信息、重复码压缩的边界效应、码本重初始化敏感性，以及训练时语言和当前状态分别贡献了什么。

相关：[LISA](lisa.md)、[SkillDiffuser](skilldiffuser.md)、[方向论文索引](README.md)。
