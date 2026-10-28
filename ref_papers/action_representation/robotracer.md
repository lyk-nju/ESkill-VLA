# RoboTracer：有序三维粗路径

**用途：当前方法参考。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../docs/action_representation.md)，实现以 [design.md](../../docs/design.md) 为准。

## 原工作机制

用有序稀疏 Spatial Trace 描述粗路径，每点为图像坐标与绝对深度 (u,v,d)。原模型包含冻结 MapAnything 空间编码、自回归输出、尺度回归及 GRPO 训练，连接规划执行栈。

## 对本项目的参考价值

当前运动表征的主要来源是有序粗路径这一层次。本项目改为固定 M 个联合 Motion Queries 加显式弧长进度编码，辅助回归当前末端到 Event 结束的相对 XYZ 位移，预测 Z 作为动作条件。

## 采用边界

并行 Query、XYZ、弧长采样和 SmoothL1 都是本项目改造，不是原文实现。原文不少任务采用对象中心 trace；当前标签明确采用末端轨迹。确定性辅助回归不等于已经学得多峰路径分布。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2512.13660v4) · [出版／作者入口](https://zhoues.github.io/RoboTracer/)

既有阅读记录：v4 §3.1–3.3、附录 C/D/E.7–E.8；未核实完整公开训练实现。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
