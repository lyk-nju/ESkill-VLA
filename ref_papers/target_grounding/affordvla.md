# AffordVLA：以功能区域教师监督替代 ROI 重建

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原文：** AffordVLA: Injecting Affordance Representations into Vision-Language-Action Models via Implicit Feature Alignment；arXiv 2605.17517，v1 2026-05-17。[版本](https://arxiv.org/abs/2605.17517) · [方法 §III](https://arxiv.org/html/2605.17517v1)

Qwen3-VL 产生任务相关部件概念，SAM3 派生教师提供中间 affordance 特征，学生视觉 tokens 做对齐；教师仅训练时使用。§II-C 明确将其与重建监督比较。这里不是把预测 mask 直接塞给部署策略，也不应称为完全没有外部监督成本。

**对项目的意义。** 这是 ROI decoder 之外应优先实现的轻量对照。固定对象区域与教师预算后，比较重建外观和学习功能部位，才能回答 target grounding 到底需要何种信息。增加“同对象、不同 atomic skill”的部位变化测试；如果表示只识别对象名称，不能算学到了技能相关交互区。

**局限与下一步。** 作者在 §V 自述精确接触、持续动作调整和跨平台泛化仍有边界。项目可以在相同动作专家上增加目标替换和局部接触误差评测；表示更像教师本身不是行为控制证据。本文读到 §II-C、III、V，未复现。

与 [AffordanceVLA](affordancevla.md) 是不同工作。返回 [机制链](evidence.md#trace)。
