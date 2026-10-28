# HAMSTER 的二维轨迹引导接口

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

HAMSTER 说明高层视觉语言知识可以通过空间路径进入控制器。它同时挑战本项目的结构选择：粗轨迹可能已承载一部分目标选择和运动实现，显式 atomic skill / target grounding / action representation 分解需要证明超出这种简单接口的价值。

## 论文与阅读范围

- 正式题名：HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation。
- 版本：arXiv 2502.05485v4，2025-05-10；ICLR 2025。
- 来源：[原文](https://arxiv.org/html/2502.05485v4)、[版本记录](https://arxiv.org/abs/2502.05485)、[作者项目页](https://hamster-robot.github.io/)。
- 阅读深度：主文 §4.1–4.2 的高低层接口与数据来源；未核验所有附录消融或复现。
- 关联：Insight 2 与 Insight 3；可作为跨层接口替代方案。

## 原工作的接口与证据

高层 VILA 从 RGB 与语言产生归一化二维末端路径，waypoint 含 x、y 和夹爪开合状态；低层三维控制策略结合实际观测执行。论文采用一次高层规划、低层反馈控制，路径经过 Ramer–Douglas–Peucker 简化。[§4](https://arxiv.org/html/2502.05485v4)

| 训练或部署问题 | 原工作范围 |
| --- | --- |
| 高层数据 | RoboPoint、RLBench、Bridge/DROID 轨迹及 VQA 等混合来源 |
| 路径监督 | 机器人 proprioception 与相机参数提供投影轨迹 |
| 高层坐标 | 图像平面归一化坐标 |
| 低层接口 | 三维感知策略，如 RVT2、3D Diffuser Actor |
| 人类视频范围 | 文中讨论视频跟踪的可行性，不能据此写成已证明纯人类视频训练 |

## 对本项目的推论

路径同时回答“朝哪里走”和“一段时间里怎么走”，不宜直接命名为 target grounding，也不适合直接归为离散技能 atomic skill。它是 target grounding 与 action representation 之间有解释性的候选接口，可以检验本项目是否需要更细的因子分解。

二维路径有利于跨数据来源共享，却可能把深度、抓取朝向和接触约束留给低层猜测。在抓取前移动路径看似合理时，控制失败不一定是高层语义错误；必须区分路径目标正确、路径几何可执行和低层跟随成功。

## 可区分机制的实验

在同一低层专家上比较二维路径、三维路径、当前目标 target grounding 和 atomic skill / target grounding / action representation。分别测试真值投影路径与预测路径，避免把感知质量差异当作接口优劣。以新对象、新布局和新技能组合三个拆分评估，观察哪一类泛化真正需要语义 atomic skill。

对执行中目标被移动的场景，比较一次规划、事件边界重规划和固定频率重规划。报告重规划延迟与路径过期率；更高调用频率的优势不能免费归到表征设计。

下一步重点结合 [3D HAMSTER](hamster_3d.md) 追问深度缺失，以及非接触空中运动如何映射到三维控制空间。

返回 [方向论文索引](README.md)。
