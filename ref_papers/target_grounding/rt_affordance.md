# RT-Affordance 关键交互姿态作为条件

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation  
**时间与版本：** 2024；核对 arXiv v1。  
**归属：** Insight 2，关联 Insight 0；Phase 1 几何接口参照。

[论文原文](https://arxiv.org/html/2411.02704) · [论文入口及 PDF](https://arxiv.org/abs/2411.02704)

## 原工作机制

正文方法核对：§III。以关键时刻的末端姿态组成 affordance plan，和语言共同条件化策略；计划可由用户提供或模型预测。图像投影形式含位置、朝向与时序信息。

## 证据边界

这里的 affordance 是关键交互姿态计划，比当前对象 ROI 包含更多未来约束。视觉投影依赖相机标定。

## 对本项目的推论

可用它检验“知道对象在哪里”之外是否还需要抓取侧、朝向和放置姿态。但若加入计划式条件，应明确扩展了 target grounding 的定义，并另报所用监督。

最小实验：相同对象与语言下，比较 ROI、交互点、姿态关系的收益；同时提供真值条件上限和预测条件结果，分离规划误差与执行误差。

## 下一次深读

核对计划是一次生成还是在线更新、如何处理目标移动，以及真实轨迹与低成本 affordance 图像各自贡献多少收益。

返回 [目标接地](README.md)。
