# RoboTracer：以稀疏三维 Spatial Trace 连接理解与执行

> **当前定位（2026-10-28）：核心 · 当前运动表征的主要参考。** 借鉴有序粗空间路径；固定 Motion Queries、相对 XYZ 辅助监督与 Action Expert 接口是本项目改造。

**论文：** Towards Spatial Trace with Reasoning in Vision-Language Models for Robotics；早期题名为 RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics。  
**版本：** arXiv:2512.13660v4，2026-07-03；初版 2025-12-15。  
[原文](https://arxiv.org/html/2512.13660v4) · [项目](https://zhoues.github.io/RoboTracer/) · [代码入口](https://github.com/Zhoues/RoboTracer)

## 已核对的机制

依据正文 §3.1–3.3、附录 D、E.7–E.8 及 C。Spatial Trace 用约 6–12 个有序三维关键点描述粗空间路径，每点以图像坐标与绝对深度 \((u,v,d)\) 表达；可采用对象或执行器中心的轨迹语义，不是固定时间采样的密集控制动作。

原模型使用基于冻结 MapAnything 编码器的空间编码、尺度解码与特征投影；SFT 包含自回归输出与尺度回归，RFT 使用 GRPO 和空间轨迹相关奖励。不能把它描述成与本项目相同的并行连续 XYZ 回归网络。

## 对当前设计的启发

完整事件的运动可以用稀疏空间路径表达，作为任务理解与本体执行之间的中间接口。本项目进一步使用固定数量的联合 Motion Queries，从当前到 Event 结束按归一化弧长构造末端相对 XYZ 标签，以 SmoothL1 辅助监督运动 Token，并将预测 Token 输入 Flow Matching Action Expert。

这借鉴了表示层次，不复制其自回归坐标输出、强化训练或外部规划执行栈；原文的点数也不直接确定本项目的查询数量。当前方法见 [Action Representation](../../docs/action_representation.md)。

## 证据边界

RoboTwin 中多数任务采用对象中心轨迹，结合其抓取／放置 API；点击类任务采用执行器轨迹。真实平台使用规划、抓取感知与修正，部分接触点深度来自传感器。因此其结果不能直接证明本项目端到端动作条件接口的收益。

论文明确参考路径并非唯一合法路径。稀疏监督不能自动保证多模态保持，确定性回归依然存在路径平均风险。本项目 Stage 1 先验证整体方案的任务成功率，出现实际问题再定位。

公开项目入口不等于完整权重与训练实现已核验；本轮未复现代码或实验。
