# VH-Diffuser：预测时域后生成变长轨迹

> **当前定位（2026-10-28）：补充 · 按需时域参照。** 预测时域与变长轨迹的替代方向；当前固定 Motion Queries 不依赖长度头。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

**论文：** VH-Diffuser: Variable Horizon Diffusion Planner for Time-Aware Goal-Conditioned Trajectory Planning。Ruijia Liu、Ancheng Hou、Shaoyuan Li、Xiang Yin；arXiv 2509.11930v1，2025-09-15。

[原文](https://arxiv.org/html/2509.11930v1) · [论文入口](https://arxiv.org/abs/2509.11930)

**核查范围：** 本次复核 §V 的时域与序列构造；未复现。虽不在此前半年检索窗口内，因直接服务当前设计而收为核心。

## 可借鉴机制

先估计起止状态间最短步数，再设置 diffusion 初始噪声序列长度；训练采用随机长度片段。长度预测与轨迹内容分开学习。

## 对当前设计的作用

借鉴“先长度、后内容”的计算顺序。我们的标签是示范 Event 剩余时间，不是最短可达步数；采用轻量时间查询 Transformer，不直接采用其 diffusion planner。

该文序列含起点，所以点数比转移步数多一；我们的未来查询不含当前起点，不能直接照抄长度公式。
