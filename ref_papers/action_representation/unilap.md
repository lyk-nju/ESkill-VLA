# UniLAP：从连续视频后验到可采样策略

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** UniLAP: A Unified Latent Action Policy for Cross-Embodiment Manipulation  
**论文标识：** [arXiv 2610.16969v1](https://arxiv.org/abs/2610.16969v1) · [原文](https://arxiv.org/html/2610.16969v1)  
**日期：** v1 2026-10-12。核查 2026-10-27。  
**本轮分层：** core；Insight 3 action representation。

## 已读证据

相关方法核读：§2–3.3；§4.2；附录 A.2–A.3。使用冻结 DreamDojo LAM teacher，训练当前观测/语言条件的连续 flow policy；DINOv2 跨帧对应残差提供运动辅助目标。目标本体阶段训练新动作 decoder 并轻量微调策略。

## 证据边界

无需外部 flow estimator 不等于无视觉 teacher；对应残差是运动代理而非严格相机不变量。不同语料采样频率不同，latent 也不是自然具有统一物理单位。

## 对当前项目的推论

优先比较多模态 prior 与点预测，固定 atomic skill / target grounding 后评价可行路径多样性、末端精度及teacher/prior差距；不把多本体视频预训练等同于零适配执行。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
