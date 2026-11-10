# RoboGround：对象与目的地区分

**用途：补充机制，按需阅读。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../target_grounding.md)，实现以 [design.md](../../design.md) 为准。

## 原工作机制

GLaMM 预测操作对象和放置区域两个 mask；动作策略将其与观测、语言和状态共同使用。论文主要 RoboCasa 设置在 episode 开始时预测一次 mask，之后复用。

## 对本项目的参考价值

借鉴对象区域为动作策略提供空间条件的机制。当前事件目标采用 Event-boundary Object State Grounding，仅用被操作对象在事件终点的区域构造视觉监督。

## 采用边界

原文的双 mask 接口只作为论文机制保留，不进入当前事件目标定义。原文一次预测依赖其任务设定，不是对所有长任务都有效或无效的证明。当前每次新观察更新视觉锚点，不增加部署 mask 或 GLaMM 模块；训练真值区域不进入在线预测器。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2504.21530v1) · [出版／作者入口](https://robo-ground.github.io/)

既有阅读记录：v1 §4.1–4.4、§5.1；对象标记前驱见 [MOO v2 §3](https://arxiv.org/html/2303.00905v2)。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

来源关系：[CLIPort §3](https://arxiv.org/html/2109.12098) 的语义／空间分工启发 [MOO v2 §2–3](https://arxiv.org/html/2303.00905v2)；MOO 给策略的是对象中心标记，RoboGround 扩展为对象和放置区域 mask。它们不是当前／终点对象潜变量回归的现成实现。

返回 [本方向索引](README.md)。
