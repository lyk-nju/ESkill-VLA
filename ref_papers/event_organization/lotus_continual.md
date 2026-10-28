# LOTUS：视觉分段与持续扩展技能库

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为替代分段、进度或组合评测的按需参考；当前先落实可核验事件边界与监督接口。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/event_organization.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**题名：** *LOTUS: Continual Imitation Learning for Robot Manipulation Through Unsupervised Skill Discovery*，Weikang Wan、Yifeng Zhu、Rutav Shah、Yuke Zhu，ICRA 2024。arXiv 2311.02058，v1 2023-11-03；实读 v4 2024-11-23。

**来源和范围：** [元数据](https://arxiv.org/abs/2311.02058)、[原文 §III–V](https://arxiv.org/html/2311.02058v4)。建议归入 Event 支持材料，跨链 atomic skill；未复现。与 2410.01345 的 **3D-LOTUS++ 是不同论文**。

## 机制与直接前驱

DINOv2 特征用于层次时序分段，再做增量技能分区：silhouette 高于阈值归旧分区，低于阈值开新分区。低层是目标条件策略，高层选择技能与 latent 子目标，旧技能更新使用回放。高层输出预留最大技能数，通过 mask 逐步开放。

§IV-A 明确沿用 [BUDS v2 §III-B–C](https://arxiv.org/html/2109.13841v2) 的自底向上分段思路；后者先由多模态表征构造层次，再跨演示聚类。LOTUS 增加持续分区与保留旧技能的机制。这条链补充了夹爪关键帧之外的事件组织路线。

## 本项目值得学习什么

先分开研究“哪里发生事件”“这是什么行为”“需要新增哪种技能”。同一视觉段落被聚在一起，不自动证明对象无关语义；技能标签变化也不自动证明新的物理能力。

可以先把 LOTUS 用作标注与数据组织对照：相同动作数据分别按脚本、夹爪、视觉层次分段；控制片段数量及长度分布，再测语义一致性与闭环成功。若未来转向持续学习，再测复用/新增决策、回放成本及跨多个增量阶段的旧任务保留。

其未来图像 latent 子目标与本项目“当前交互空间”意义下的 target grounding 不同，不应直接改名接入。保留该区别，比仅把所有子目标方案归入空间接地更有解释力。

## 与现有卡片的关系

与 CompILE/TACO 比较分段监督；与 AtomicVLA 比较增量聚类加回放和新增专家路线。保留独立文件名 `lotus_continual.md`，在 3D-LOTUS++ 卡增加消歧链接即可，不合并两项工作。报告中阈值方向和固定滑窗分类的修正见 [评审](../atomic_skill/evidence.md#review)。
