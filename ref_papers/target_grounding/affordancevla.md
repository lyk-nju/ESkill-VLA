# AffordanceVLA：ROI、接触位置和对象几何共同连接动作

> **当前定位（2026-10-28）：核心 · 直接设计参照。** 按交互职责组织 query、区域监督与动作条件；不默认移植其所有空间头。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原文：** AffordanceVLA: A Vision-Language-Action Model Empowering Action Generation through Affordance-Aware Understanding；arXiv 2606.06155。v1 2026-06-04，本轮读 v2 2026-10-18。[版本](https://arxiv.org/abs/2606.06155) · [§2–3、附录 H](https://arxiv.org/html/2606.06155v2)

并行 queries 分别监督对象 ROI 的 VAE latent、二维交互热图、三维 shape 与 10-DoF layout；动作专家读取理解与 affordance 表征。它采用多专家和多阶段数据训练，推理仍保留 affordance generation 路径。§2.1 明确引用 ReconVLA，但没有据此证实代码继承。

**对项目的意义。** “ROI query 再给动作专家”已经存在非常近的结构。值得检验的是 Event 分配、atomic skill 与 target grounding 的互补条件，以及 object–effector 当前关系如何支持组合泛化。不要把该文 How2Act 仅凭名称归为本项目的 action representation：它监督的内容仍主要是对象几何。

**公平对照。** 第一阶段可缩成同骨干的 ROI-only、ROI+热图、ROI+关系三组；不必复制整套大规模训练。分别核算额外视觉数据、教师处理和参数预算。对照对象位置互换与同物不同交互部位，观察作用是正确选目标还是改善运动精度。

本轮读 §2、3.1–3.3、附录 C/D/H；尚未逐项核查其完整数据去重、伪标签流水线及所有消融，不将总体分数解释成模块独立贡献。返回 [机制链](evidence.md#trace)。

## 两报告复核补充（2026-10-27）

本次补核 [v2 §3.1.4、附录 F/G](https://arxiv.org/html/2606.06155v2)。单向 attention mask 约束前向可见性，后两个训练阶段仍端到端联合更新，不能据 mask 声称动作梯度被阻断。不同阶段的记录数、轨迹数和损失权重应分列；三阶段不能压缩成一个固定配方。具体核查及与 AffordanceWAM、SG-VLA 的比较见 [梯度与接口记录](evidence.md#review)。
