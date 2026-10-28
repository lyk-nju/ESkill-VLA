# AdaWorld：预训练世界模型的可迁移动作接口

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** AdaWorld: Learning Adaptable World Models with Latent Actions  
**版本：** arXiv 2503.18938v4，2025-06-02；ICML 2025。  
**归属：** Insight 3。

[正文](https://arxiv.org/html/2503.18938v4) · [正式会议页](https://proceedings.mlr.press/v267/gao25u.html) · [作者页](https://adaptable-world-model.github.io/)

## 已核对的机制

正文 §2.1–2.3。beta-VAE 从帧对获得连续 latent，作为自回归视频世界模型的条件。迁移示范时复用 latent 序列；适配真实控制时，离散动作可初始化对应 latent，连续控制用 MLP 映射并微调。

## 证据边界

示范动作的零训练视频迁移，与真实控制接口的有监督适配分开。视频中的动作复现不能直接解释为机器人零样本执行。论文的连续/离散比较也受具体码本容量与任务设定约束。

## 对本项目的推论

与 LAPA 对照，它强调保留前向预测能力。可评估同一个预训练 action representation 用于行为克隆或世界建模时需要保留哪些信息；二者最优瓶颈未必相同。

最小实验应同时测控制可解码性、未来预测和对象不变性。如果优化某一项明显牺牲另一项，可以采用共享底座加用途投影，而非强制一个 latent 承担全部职责。

## 后续精读

适配数据量、动作接口初始化、规划预算与初始模型的预训练差异。若复现，先选择与机器人相关的连续控制部分。

关联 [LAC-WM](lac_wm.md) · [DreamDojo](dreamdojo.md)。
