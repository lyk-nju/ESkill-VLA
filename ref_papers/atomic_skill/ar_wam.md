# AR-WAM：离散操作与视觉接地条件

> **当前定位（2026-10-28）：核心 · 直接接口参照。** 离散操作嵌入与对象接地分别条件化策略；外部技能条件不等于自主技能预测。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

**论文：** AR-WAM: A Visual-Conditioned Agent-Ready World Action Model for Robotic Manipulation。Yicheng Jiang、Zesen Gan 等；arXiv 2609.23578v2，2026-10-02。

[原文](https://arxiv.org/html/2609.23578v2) · [官方代码](https://github.com/AR-WAM/AR-WAM)

**核查范围：** 本次复核 §III-A 操作条件接口，沿用此前方法记录；未复现。

## 可借鉴机制

外部 agent 提供操作类别与目标框；可学习操作嵌入和空间条件经条件调制进入策略。共享策略支持多个操作类别。

## 对当前设计的作用

它是“技巧与目标分别提供条件”的直接参照。其操作向量不是冻结文本原型，也不是我们根据当前上下文学习的自主选择器。

我们的三锚点局部去噪、事件剩余运动 Token 与该接口不同；首版不因此引入外部 agent、完整世界模型或其失败恢复系统。
