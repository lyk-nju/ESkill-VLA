# X-Tokenizer：语义约束与运动细节

**用途：补充机制，按需阅读。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../docs/action_representation.md)，实现以 [design.md](../../docs/design.md) 为准。

## 原工作机制

非对称 residual quantization 的首层使用 masked-action 目标，深层保留重建细节，结合冻结 VLM 对齐与未来特征监督。tokenizer 用于离线监督，连续策略推理不调用它。

## 对本项目的参考价值

保留语义约束与精细运动重建可以作用于不同部分的认识，关联当前可复用 K 与条件运动 Z 的职责分工。

## 采用边界

粗 motion intent 并非已证明的对象无关 Skill；编码包含本体条件。当前不增加 RVQ、masked-action 或新对齐损失。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2606.14752v2)

既有阅读记录：v2 §1–3.4；与 ActionCodec 为引用和差异化关系，未确认代码继承。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

来源关系：X-Tokenizer 引用并比较 [ActionCodec v1 §4–5](https://arxiv.org/html/2602.15397v1)；后者再与 [FAST §V](https://arxiv.org/html/2501.09747v1) 的压缩路线比较。后续版本不能倒推为早期论文的来源。

返回 [本方向索引](README.md)。
