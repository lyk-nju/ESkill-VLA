# X-Tokenizer：语义约束与运动残差可以分工

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining  
**论文标识：** [arXiv 2606.14752v2](https://arxiv.org/abs/2606.14752v2) · [原文](https://arxiv.org/html/2606.14752v2)  
**日期：** v1 2026-06-07；v2 06-28。核查 2026-10-27。  
**本轮分层：** core；Insight 3 action representation。

## 已读证据

相关方法核读：§1–3.4。非对称 residual quantization：第一层 masked-action 目标，深层保留重建细节；结合冻结 VLM 特征对齐和下一帧特征预测。动作编码用于离线监督，下游连续策略推理不调用 tokenizer。

## 证据边界

coarse motion intent 不是已证明的对象无关 atomic skill；完整 action representation 仍须包含精细几何。cross-embodiment 编码包含本体条件。

## 对当前项目的推论

分别对语义投影与全部运动表征施加语义约束，检查后者是否压掉同目标多路径和接触精度。与 ActionPiece 对照语义与几何保真目标的冲突。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
