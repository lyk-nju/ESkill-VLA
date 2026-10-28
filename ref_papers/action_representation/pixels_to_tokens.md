# From Pixels to Tokens：把 teacher 来源与监督接口分开比较

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models  
**论文标识：** [arXiv 2605.04678v1](https://arxiv.org/abs/2605.04678v1) · [原文](https://arxiv.org/html/2605.04678v1)  
**日期：** v1 2026-05-06。核查 2026-10-27。  
**本轮分层：** core；Insight 3 action representation。

## 已读证据

相关方法核读：§2–4；附录 A.1/B.2/C。在统一 VLA 中比较视觉转移与真实动作来源的表征，以及隐式对齐、直接预测、条件预测和动作 token 监督。保留真实动作损失；附录 A.1 的 image teacher 还使用 5% 动作标签。

## 证据边界

不同接口的 placeholder 数量不同，作者提供长度匹配对照；排行榜部分取自论文/公开榜单，并非所有模型重新统一训练。不能宣称所有比较完全同预算。

## 对当前项目的推论

先采用其因素拆分思路，增加我们需要的当前 target grounding 条件；将 teacher 标签预算、token预算、额外训练和推理成本分别报告。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
