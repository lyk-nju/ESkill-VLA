# UniPi 生成视频作为运动计划接口

> **当前定位（2026-10-28）：归档 · 历史归档。** 像素视频规划历史路线，当前优先研究紧凑特征接口；从优先清单移除。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Learning Universal Policies via Text-Guided Video Generation  
**时间：** NeurIPS 2023。  
**归属：** Insight 3 的替代路线；非 Phase 1 必需依赖。

[论文入口及 PDF](https://arxiv.org/abs/2302.00111) · [作者官方方法介绍](https://research.google/blog/unipi-learning-universal-policies-via-text-guided-video-generation/)

## 原工作机制

作者方法介绍核对：UniPi implementation 及流程。依据当前图像与语言生成未来视频计划，再用逆动力学模型提取执行动作。

## 证据边界

其显式中间接口是视频，不等同于紧凑的运动 latent。视频外观合理与动作物理可执行需要分别验证。

## 对本项目的推论

该路线有助于追问 action representation 应保留多少未来变化信息。我们可以研究较轻量的潜在运动接口，是否在相同执行预算下提供足够控制条件。

最小实验：若后续加入视频计划基线，统一任务与初始状态，分别记录计划一致性、闭环成功率和推理延迟；不要只比较生成画面。

## 下一次深读

核对动作适配所需监督、视频计划的重规划频率以及未见任务的具体定义。工程实现优先级低于 LAPA、ConLA 和 UniVLA。

返回 [action representation](../../action_representation/README.md)。
