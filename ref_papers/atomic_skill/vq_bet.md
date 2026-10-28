# VQ-BeT 动作量化与多模态控制

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Behavior Generation with Latent Actions  
**时间与版本：** ICML 2024；核对 arXiv v2。  
**归属：** Insight 1 的区分对照、Insight 3 的动作编码参照。

[论文原文](https://arxiv.org/html/2403.03181) · [正式论文页](https://proceedings.mlr.press/v235/lee24y.html) · [论文入口及 PDF](https://arxiv.org/abs/2403.03181)

## 原工作机制

正文方法核对：§2.3、§3。以真实动作或 action chunk 训练残差 VQ，再从观测预测动作码，并用 offset 修正量化误差。方法针对多模态行为生成。

## 证据边界

它的 latent 来自有标签动作，不能直接用于无动作视频。粗细两级动作码也不自动对应行为类别 atomic skill 与实现 action representation。

## 对本项目的推论

用该路线检查离散化是否损害局部控制，并明确语义一致性和动作保真是不同目标。对 atomic skill 做稳定性约束时，不应顺带要求 action representation 丢弃所有类内变化。

最小实验：比较不同码本容量下的动作重建误差、成功率和有效轨迹多样性；保留连续动作专家作为控制精度参照。

## 下一次深读

核对动作归一化、chunk 时长、码本使用率、offset 对精度的贡献，以及多模态评测的具体定义。

返回 [原子技能](README.md) · 关联 [潜在动作](../action_representation/README.md)。
