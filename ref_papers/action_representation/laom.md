# LAOM：控制信息存在，不等于可用

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** Latent Action Learning Requires Supervision in the Presence of Distractors  
**论文标识：** [arXiv 2502.00379v5](https://arxiv.org/abs/2502.00379v5) · [原文](https://arxiv.org/html/2502.00379v5)  
**日期：** 历史诊断；2025-06-12版本。核查 2026-10-27。  
**本轮分层：** core；Insight 3 action representation。

## 已读证据

相关方法核读：§3–5。基于 LAPO，去量化、增加容量、使用多步 IDM、特征预测与增强；进一步在表示学习期使用少量真实动作监督。实验在带干扰的连续控制任务中进行。

## 证据边界

不能将论文标题理解为所有视频学习都必须监督的普遍定理。大 latent 的探针变好也可能只因它保留了更多无关动态。

## 对当前项目的推论

Phase 1 比较相同标签预算在 teacher 学习期使用或只在动作 decoder 使用；报告在线 prior 差距和干扰条件下闭环表现。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
