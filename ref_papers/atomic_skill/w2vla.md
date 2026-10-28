# w²VLA：文本技巧与 FiLM 条件化

> **当前定位（2026-10-28）：核心 · 直接方法来源。** 文本技巧嵌入、适配与 FiLM 条件化；对象无关原型分类器是我们的改造。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

**论文：** Decoupling the Declarative from the Procedural in Vision-Language-Action Models。Nikolaos Tsagkas 等；arXiv 2606.21496v2，2026-10-09。

[原文](https://arxiv.org/html/2606.21496v2) · [论文入口](https://arxiv.org/abs/2606.21496)

**核查范围：** 延续此前方法阅读，本次复核 §II-B–C 与附录提示词表；未复现代码。

## 可借鉴机制

冻结文本特征经 MLP 形成技巧条件，并通过 FiLM 调制控制特征；空间条件另行注入。整体方法还依赖视觉路由与训练时的视觉遮蔽。

## 对当前设计的作用

参考其技巧调制接口。我们的 Event 分类、硬原型查表与技巧直连 Action Expert 属于另外的设计选择，不能由其结果推出文本原型优于可学习嵌入。

附录部分技巧短语含目的地或实体信息，因此不能将原方法概括成严格无对象名称。我们的对象无关词表需要自行检验。
