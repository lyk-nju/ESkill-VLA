# BUDS 从重复结构发现可复用技能

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为替代分段、进度或组合评测的按需参考；当前先落实可核验事件边界与监督接口。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/event_organization.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation  
**时间：** 2021 预印本；RA-L 2022。  
**归属：** Insight 0、Insight 1；Phase 1 粒度与复用参照。

[论文入口及 PDF](https://arxiv.org/abs/2109.13841) · [作者方法与实验页](https://ut-austin-rpl.github.io/BUDS-website/)

## 原工作机制

作者方法页核对：Method Overview、Hierarchical Policy Model、Multi-task Domain。通过凝聚聚类建立示范层次结构，从重复模式中发现技能，再训练目标条件技能策略和高层控制器。

## 证据边界

作者的新组合实验通过重新训练 meta-controller 调用已有技能，属于带适配的复用证据，不能据此主张冻结模型的零更新重组。

## 对本项目的推论

技能粒度可以结合跨任务重复性选择，而非只按固定帧数决定。需要同时检查过细片段难以独立执行、过粗片段绑定完整任务的问题。

最小实验：比较粗、中、细三档语义粒度的复用率和成功率；分别报告冻结全部策略与只训练高层选择器两种设置。

## 下一次深读

核对聚类距离、层次切分规则、技能数量选择和新高层控制器所需数据。确认不同任务中相似片段是否包含相同交互关系。

返回 [事件组织](README.md) · 关联 [原子技能](../atomic_skill/README.md)。
