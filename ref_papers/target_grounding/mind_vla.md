# Mind-VLA：按阶段对象建立空间监督

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原文：** Mind-VLA: Instruction-Aware Spatial Representation Alignment for Vision-Language-Action Models；arXiv 2608.04633。v1 2026-08-05，本轮读 v2 2026-08-27。[版本](https://arxiv.org/abs/2608.04633) · [§III–IV](https://arxiv.org/html/2608.04633v2)

指令提供有序对象，抓取/推动阶段用夹爪与运动线索匹配；对象 canonical 多视图用于 VAE 与几何监督，部署保留对象 queries 供动作读取。对象 mesh 做中心和尺度归一化，不能直接代表当前 metric 相对位姿。

**对项目的意义。** 事件对应目标、对象重建 queries、动作消费这些表示已有直接近邻。当前选题应验证同技能跨对象重组时，执行器与目标关系是否提供额外贡献，而非仅报告加入对象 supervision 比 action-only 更好。

**建议控制。** 固定事件边界比较当前对象与下一对象标签；另比较相同事件分配下的 canonical shape、当前 ROI、object–effector 关系。重点观察过渡期间提前换目标与接近阶段监督归属。保留形状监督但打乱其当前位姿，可检验模型究竟依赖何种信息。

本轮读 §II、III-B–D、IV-A；DreamVLA 基底还保留其他未来/动态目标，公平比较需控制这些辅助信号。v1/v2 指标不可混用。返回 [机制链](evidence.md#trace)。
