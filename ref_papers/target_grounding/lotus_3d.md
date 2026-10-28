# 3D-LOTUS++ 的技能与对象几何接口

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

3D-LOTUS++ 是本项目需要正面对照的已有分解：任务规划、对象 grounding 和运动控制已经被拆开。因而 atomic skill / target grounding / action representation 的主要新意不能只是三个模块的命名与连接。

## 论文与阅读范围

- 正式题名：Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy。
- 版本：arXiv 2410.01345v2，2025-03-01；ICRA 2025，首次发布于 2024 年。
- 来源：[原文](https://arxiv.org/html/2410.01345v2)、[版本记录](https://arxiv.org/abs/2410.01345)、[GemBench 作者页](https://www.di.ens.fr/willow/research/gembench/)。
- 阅读深度：主文 §IV-A–C 与 §V-A；未复现。
- 关联：Insight 1、2、3；GemBench 的泛化分层也关联 Insight 0。

## 原工作的接口与证据

| 项目 | 原文方法 |
| --- | --- |
| 任务规划 | LLaMA3-8B 按六种预定义 primitive 生成计划 |
| 接地输入输出 | OWLv2 框、SAM mask 与标定 RGB-D 得到对象点云，CLIP 匹配计划中的对象 |
| 控制表示 | 点的 XYZ 与四类角色：操作对象、目标位置、机器人、障碍物；角色编码替代 RGB |
| 动作监督 | 模仿轨迹；预测离散位姿、夹爪状态与停止概率 |
| 部署依赖 | 多视角 RGB-D、相机参数、外部语言与视觉模块、位姿执行器 |

GemBench 包含位置、刚体对象、关节对象与长时序泛化层次。需区分 benchmark 的七类通用 primitive 与 ++ 系统使用的六种 primitive。[§III–V](https://arxiv.org/html/2410.01345v2)

## 对本项目的推论

四类点角色给出了比“对象+夹爪 ROI”更完整的交互接口：接受对象和障碍物都可能决定运动。用角色与几何替代外观，也是削弱对象纹理捷径的一条直接路径。它是否足以覆盖材质、摩擦或可变形物体，则需要保留外观条件的对照。

本项目仍可区别于固定 primitive、外接工具和串行对象匹配：研究可学习技能的不变性、并行 atomic skill / target grounding 的共享接口、事件监督与无动作视频迁移。但每项区别都要有实验证据，不能把端到端或 latent 自动视为更优。

## 可区分机制的实验

在同一 RoboTwin 数据和控制专家上，对比预定义技能加角色点云、学习 atomic skill 加相同角色几何、并行学习 atomic skill / target grounding，以及加入 action representation 的版本。先固定目标真值查看控制上界，再替换成预测接地，拆开技能选错、对象选错、几何错与执行错。

沿 GemBench 的思路分别保留新位置、新物体、新机构和新序列，避免把所有 OOD 合为一个成功率。如果去 RGB 的角色几何在新纹理上很好、在接触任务上下降，应追查外观中真正有物理作用的信息，而非继续增加外观不变性。

后续需核验：开放词汇匹配的部件级错误、停止信号的监督来源，以及固定 primitive 与学习技能在相同数据下的可扩展性。

返回 [方向论文索引](README.md)。

## 名称消歧

本卡与 [LOTUS 持续技能发现](../event_organization/lotus_continual.md)（2311.02058）是不同工作，不合并条目或互换实验结论。
