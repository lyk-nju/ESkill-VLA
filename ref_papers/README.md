# ESkill-VLA：方法参考库

这里说明当前设计借鉴了什么，以及相关工作有哪些有用机制。**实现基准是 [docs/design.md](../docs/design.md)**；研究出发点见 [idea.md](../docs/idea.md)，各模块的选择理由见四份方向文档。论文卡不增加模块、训练阶段、实验任务或验收条件。

## 面向实现的阅读入口

先读 Design，再按需要查下表中的机制。原论文架构与 ESkill-VLA 的改造在各卡中分别说明。

| 模块 | 当前设计摘要 | 主要来源 |
| --- | --- | --- |
| [Event](event_organization/README.md) | 一个 Event 对应一次 Skill 执行，脚本／状态记录组织可核验标签 | [AtomSkill](event_organization/atomskill.md)、[DeCo](event_organization/deco.md)、[WALL-WM](action_representation/wall_wm.md) |
| [Skill](atomic_skill/README.md) | 冻结文本原型分类，预测类别查独立可训练 U；进入 Goal、FiLM 和动作专家 | [w²VLA](atomic_skill/w2vla.md)、[AR-WAM](atomic_skill/ar_wam.md)、[CRAFT](atomic_skill/craft.md) |
| [Grounding](target_grounding/README.md) | 三角色各 4 个视觉 Token，共享 Query Decoder，以原图布局画布进行训练期去噪监督 | [ReconVLA](target_grounding/reconvla.md)、[AffordanceVLA](target_grounding/affordancevla.md) |
| [Motion](action_representation/README.md) | 固定数量、显式空间进度编码的 Motion Queries；相对 XYZ 粗路径辅助监督 | [RoboTracer](action_representation/robotracer.md) |

这些是 9 项主要来源，不代表要复现 9 套模型。Stage 1 先完成当前整体算法的联合训练与闭环评测，以相对动作基线的平均任务成功率为主。

## 按需参考与背景

共保留 **30 张独立卡片**：9 项主要来源、5 项研究依据（StructVLA、StaKe、RDD、LAPA、ConLA），以及 16 项具有具体接口、几何或监督价值的补充。其余 **61 项既有工作**压缩到四份背景笔记，保留机制与原文入口，不再保存旧实现建议或调研排期。

| 方向 | 按需参考的重点 | 较远路线 |
| --- | --- | --- |
| Event | 语义区间与几何点的区别、事件交接、保留全局任务依赖 | [背景笔记](event_organization/background.md) |
| Skill | 表示学习与在线选择、梯度边界、共享条件与专家路由的区别 | [背景笔记](atomic_skill/background.md) |
| Grounding | 对象与目的地角色、三维尺度、关系信息与梯度路径 | [背景笔记](target_grounding/background.md) |
| Motion | 动作相关监督、运动差异、条件缓存，以及未来跨本体数据来源 | [背景笔记](action_representation/background.md) |

## 维护约定

1. 卡片只写原方法、对项目的参考价值、采用边界和原文阅读范围。当前方案由方向文档与 Design 维护，论文原方法不自动成为实现要求。
2. 实现从 Design 确定的接口出发；按需材料可以帮助理解问题，不能据此添加当前方案之外的头、损失或训练流程。Stage 1 的完成标准也以 Design 为准。
3. 新研究结论只有经过明确设计更新才进入实现基准。被替换的项目设计从当前正文移除，历史由 Git 保存，不与现行方案并列。
4. 区分已读正文、仅作者介绍、官方技术项目和未核实部分。保留证据限制，不把机制相似写成代码继承，不把重建／探针改善写成控制收益。

整理日期：2026-10-28。本次依据已存论文阅读记录整合，没有重新联网核验、运行论文代码或验证性能数字。原有 91 项条目均保留机制与来源，优先级历史、旧实验清单与重复调研流水已移除。
