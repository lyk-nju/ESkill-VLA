# ReconVLA 用当前区域重建监督感知

> **当前定位（2026-10-28）：核心 · 直接方法来源。** 训练期局部视觉 latent 去噪；三锚点及事件终点目标是本项目扩展。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver  
**时间与版本：** 2025；核对 arXiv v1。  
**归属：** Insight 2；Phase 1 优先。

[论文原文](https://arxiv.org/html/2508.10333) · [论文入口及 PDF](https://arxiv.org/abs/2508.10333)

## 原工作机制

正文方法核对：§3.2–3.3。用当前被操作区域的 VAE latent 作为重建目标，由 reconstructive tokens 条件化扩散去噪，并联合动作监督。训练目标区域通过检测模型构造。

## 证据边界

原工作围绕当前 gaze region；不是事件完成图生成。显式 object 与 effector 双区域是我们的扩展设想。

## 对本项目的推论

在共享骨干上加 target grounding 的视觉辅助监督，可作为较小的第一步。部署时是否保留重建组件，应按实际依赖与延迟决定，不把完整图像生成作为默认前置步骤。

最小实验：对象 ROI、双区域 ROI、加入原图位置关系三种监督，对比目标绑定、抓取精度和组合任务成功率；将重建质量与控制质量分别报告。

## 下一次深读

核对区域标注错误、裁剪坐标保留方式、重建 token 与动作 token 的连接，以及推理代码是否移除去噪器。

返回 [目标接地](README.md)。
