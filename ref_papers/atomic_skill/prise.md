# PRISE 动作序列压缩与变长技能

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** PRISE: LLM-Style Sequence Compression for Learning Temporal Action Abstractions in Control  
**时间：** ICML 2024。  
**归属：** Insight 1、Insight 0、Insight 3；Phase 1 压缩对照。

[正式论文页](https://proceedings.mlr.press/v235/zheng24b.html) · [作者方法页](https://ruijiezheng.com/project/PRISE/index.html) · [论文入口及 PDF](https://arxiv.org/abs/2402.10450)

## 原工作机制

作者方法页核对：Method。先做状态条件的动作量化，再用 BPE 将动作码序列压成变长技能 token。下游策略预测 token 并解码动作；新任务设置可微调解码器。

## 证据边界

序列压缩得到时间抽象，不自动给出语言行为类别。其少样本新任务适配应与零更新组合分开。

## 对本项目的推论

这是检验“语义事件是否比纯运动规律更有用”的对照。我们不必完整替换 π0.5，可先比较由不同组织方式得到的片段标签和复用结构。

最小实验：在同一多任务数据上统计压缩片段与语义边界的关系，再以匹配容量的接口测试未见技能与对象组合。压缩率作为诊断，成功率作为主要判据。

## 下一次深读

核对状态条件量化是否吸收对象特征，BPE 词表大小与持续时间分布如何影响闭环反馈。

返回 [原子技能](README.md) · 关联 [事件组织](../event_organization/README.md)。
