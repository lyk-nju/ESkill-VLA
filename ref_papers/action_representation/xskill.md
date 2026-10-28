# XSkill 跨身体技能与示范对齐

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** XSkill: Cross Embodiment Skill Discovery  
**时间：** CoRL 2023。  
**归属：** Insight 3、Insight 1；Phase 2 与 Phase 3 的连接参照。

[作者方法与演示页](https://xskill.cs.columbia.edu/) · [正式论文页](https://proceedings.mlr.press/v229/xu23a.html) · [论文入口及 PDF](https://arxiv.org/abs/2307.09955)

## 原工作机制

作者方法页核对：Abstract、One shot learning。由人和机器人视频发现技能 prototypes，再接条件扩散策略。Skill Alignment Transformer 根据当前状态对齐人类提示视频中的技能。

## 证据边界

发现共享表征与训练机器人控制是不同阶段。作者展示的任务组合支持研究示范对齐，但不能外推为任意互联网视频的零样本控制。

## 对本项目的推论

跨身体迁移和在线进度对齐应分开设计。共享的 atomic skill 或事件表示可以作为候选接口；对象关系 target grounding 与本体动作映射仍需各自建模。

最小实验：固定参数与初始状态，提供不同事件顺序的 Context；另做仅改变示范速度的对照，区分内容理解与时间匹配。

## 下一次深读

核对人机视频是否需要配对、技能 prototypes 的学习目标、提示视频与训练任务的重叠，以及低层策略的训练数据。

返回 [action representation](README.md) · 关联 [原子技能](../atomic_skill/README.md)。
