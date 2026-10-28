# LAC-WM：真实候选动作进入潜空间，再预测后果

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Cross-Embodiment Robot Foundation World Models with Latent Actions  
**版本：** arXiv 2610.10846v1，2026-10-07。  
**归属：** Insight 3，世界模型与跨本体适配。

[正文](https://arxiv.org/html/2610.10846v1) · [作者页](https://lacwm.github.io/)

## 已核对的机制

正文 §3、§4 与任务评估概述。IDM/FDM 学视觉转移，cross-augmentation 抑制捷径，motion decoder 用人手、机器人末端及相机运动标签约束 latent。迁移新本体时学习 action projector：真实动作映射到 action representation，保留 FDM 预测后果，并用于候选动作评估。

## 证据边界

这条箭头是 A→action representation→未来，不是从观测直接 action representation→A 的策略。跨本体预训练包含运动标签；UMAP 混合不足以证明统一可控语义。规划收益还受候选策略、采样数量和额外推理预算影响。

## 对本项目的推论

action representation 可以是跨本体动力学接口，却未必适合直接充当政策输出。应把这两种用途分开实验。若未来要用世界模型挑选动作，先检查候选动作覆盖率，再判断预测器的增益。

可固定候选集，对比真实控制条件与潜动作条件的预测排序精度；在动作预算不变时评估最终成功率。此路线适合作为 Phase 2/后续扩展，不应未经验证就放入 Phase 1 主链。

## 后续精读

完整核对 §6 候选选择、价值指标和 unseen embodiment 微调协议，计入所有运动标签与适配数据。

关联 [AdaWorld](adaworld.md) · [DreamDojo](dreamdojo.md)。
