# UniVLA 任务相关潜动作与显式解码

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** UniVLA: Learning to Act Anywhere with Task-centric Latent Actions  
**时间与版本：** 2025；核对 arXiv v3，2025-11-03，正文标题省略 UniVLA 前缀。  
**归属：** Insight 3；Phase 1 显式 action representation 接口、Phase 2 迁移重点。

[论文原文](https://arxiv.org/html/2505.06111) · [论文入口及 PDF](https://arxiv.org/abs/2505.06111)

## 原工作机制

正文方法核对：§III-A 至 III-C。在 DINOv2 特征上学习潜动作，借助语言与两阶段码本区分任务相关和无关变化。部署保留潜动作预测，用视觉条件的动作解码器适配机器人。

## 证据边界

该接口支持显式 latent 解码的可行性，但尚不能替我们证明 atomic skill / target grounding 并行分解或 Event 组织的价值。

## 对本项目的推论

应与 LAPA 的替换头路线并列比较。对 action representation 的设计除离散或连续外，还要比较预测目标为像素、视觉特征或对象局部变化时的信息差异。

最小实验：保持同一 π0.5 底座和数据，比较训练辅助 action representation 与显式解码 action representation；再改变 teacher 的预测目标，观察背景干扰、局部控制及组合成功率。

## 下一次深读

核对两阶段训练中的冻结范围、潜动作预测与动作损失的关系、历史 latent 使用方式，以及训练和部署条件是否一致。

返回 [action representation](README.md)。
