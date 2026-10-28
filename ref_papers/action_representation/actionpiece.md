# ActionPiece：同类动作的细微区别需要保留

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models  
**论文标识：** [arXiv 2609.18487v1](https://arxiv.org/abs/2609.18487v1) · [原文](https://arxiv.org/html/2609.18487v1)  
**日期：** v1 2026-09-16。核查 2026-10-27。  
**本轮分层：** core；Insight 3 action representation。

## 已读证据

相关方法核读：§3–4；§5.5；附录 B。使用平移、旋转和夹爪的物理距离监督近远排序；约束编码/量化表示与码字分配，并在解码动作空间用 PRC 衡量邻域次序。部署通过冻结 decoder 得到真实动作。

## 证据边界

方法框架可用 RVQ，但默认实验配置 Q=1，不能写成多层 RVQ 是收益来源。PRC 保序不保证碰撞安全、任务成功或语义不变性。

## 对当前项目的推论

将物理保序作为 action representation 诊断；atomic skill 对象不变性应在另一个投影检验，避免对完整运动表征一味拉近同类技能。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
