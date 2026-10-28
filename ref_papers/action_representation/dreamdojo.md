# DreamDojo：人类视频迁移到机器人世界模型

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos  
**版本：** arXiv 2602.06949v1，2026-02-06。  
**归属：** Insight 3，Phase 2 世界模型参照。

[正文](https://arxiv.org/html/2602.06949v1) · [作者页](https://dreamdojo-world.github.io/) · [会议入口](https://proceedings.mlr.press/v306/gao26q.html)

## 已核对的机制

正文 §3.1–3.3。连续 latent 为人类视频提供动作代理条件；之后重置动作条件层，用目标机器人数据适配真实动作，另可进行蒸馏。机器人条件使用相对动作并与视频 latent 的时间压缩对齐。

## 证据边界

这是动作条件世界模型，不是直接输出控制的 VLA。预训练知识迁移、视频生成质量、规划和策略评估是不同层次。目标机器人适配仍需数据；数据规模优势不能全部归于 latent 设计。

## 对本项目的推论

可把它视为 Phase 2 的另一种目标：让人类数据改善动作后果预测，再研究控制收益。当前 Phase 1 不必复制大型视频生成器。

建议在小规模下控制视频数量与任务覆盖，比较被动未来预测、潜动作条件预测和动作监督预测；报告目标本体所需动作样本数。对跨本体测试，留出本体、对象和交互类型应分别设计。

## 后续精读

潜动作瓶颈容量、对照所用人类数据是否一致、真实动作后训练与蒸馏的损失。本文未验证作者数据或模型部署可用性。

关联 [RynnWorld-Latent](../_archive/action_representation/rynnworld_latent.md) · [AdaWorld](adaworld.md)。
