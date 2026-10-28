# AtomSkill 语义事件与技能对齐

> **当前定位（2026-10-28）：核心 · 直接方法来源。** Event 分段与技巧语义；借鉴组织方式，不直接沿用动作量化码本。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/event_organization.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Learning Semantic Atomic Skills for Multi-Task Robotic Manipulation  
**时间与版本：** 首次公开 2025；本次核对 arXiv v2，2026-07-02。  
**归属：** Insight 0、Insight 1；Phase 1 优先。

[论文原文](https://arxiv.org/html/2512.18368v2) · [论文入口及 PDF](https://arxiv.org/abs/2512.18368) · [作者项目页](https://atom-skill.github.io/)

## 原工作机制

正文方法核对：§III-A 至 III-C。夹爪状态变化用于分段，VLM 生成技能语义；事件动作重采样后经 VQ 编码，再用语义与时间对比目标对齐。策略结合技能先验、keypose 和动作预测。

## 证据边界

原系统含 keypose 驱动的技能衔接。其技能码本承载运动信息，不能直接等同于本项目纯类别式 atomic skill。可变长分段也不意味着编码器原样处理变长序列。

## 对本项目的推论

优先借鉴边界与语义标签分开生成的做法。RoboTwin 可先用脚本和物理状态获得标签，再比较自动标注。将语义类别 atomic skill 与连续或量化运动 action representation 分开，是我们的设计选择。

最小实验：相同事件片段分别使用类别分类、动作码本和带语义约束的动作码本；测试跨对象的技能选择及未见组合成功率。另做边界对照，避免把标注质量和分段作用混在一起。

## 下一次深读

核对技能标签在推理时由谁提供，以及跨任务标签字典、边界阈值和相邻事件的衔接处理。语义正样本还受量化 token 条件约束，应结合公式核验实现。

返回 [事件组织](README.md) · 关联 [原子技能](../atomic_skill/README.md)。

## 两报告复核补充（2026-10-27）

[核读 v2 §III-B、§IV-D](https://arxiv.org/html/2512.18368v2)：语义正对除相同标签外还要求相同量化 token。Table V 的 keypose 消融总体为 73.3→74.2，但定位组改善、运动模式组下降，不能说所有类别均受益。夹爪跳变是分段启发式，不等于接触成功或任意技能的真值边界。详细数字与限定见 [技能复核](../atomic_skill/evidence.md#review)。
