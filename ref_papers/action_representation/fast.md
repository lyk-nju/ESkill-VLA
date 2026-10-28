# FAST：动作压缩的必要基线

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** FAST: Efficient Action Tokenization for Vision-Language-Action Models  
**论文标识：** [arXiv 2501.09747v1](https://arxiv.org/abs/2501.09747v1) · [原文](https://arxiv.org/html/2501.09747v1)  
**日期：** 历史基线；2025-01-16。核查 2026-10-27。  
**本轮分层：** support；Insight 3 action representation。

## 已读证据

相关方法核读：§V。对归一化动作序列做 DCT、量化与 BPE，得到可逆解码的压缩 token；数据驱动的 BPE 不等于语义技能发现。

## 证据边界

压缩效率、重建精度和控制可学性是此路线的指标；不能把一个 token 当成一个 Event 或 atomic skill。

## 对当前项目的推论

保留直接动作和 FAST 两种基线，检查额外语义结构的收益是否仅来自动作时序压缩。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
