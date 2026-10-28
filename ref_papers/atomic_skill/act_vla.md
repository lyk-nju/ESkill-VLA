# ACT-VLA：通过 latent 组合补齐训练数据

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Unleashing More Actions via Action Compositional Training for VLA Models*，Kai Peng、Jie Lu、Xiaojiang Peng。arXiv 2607.00351v1，**2026-07-01**，方法名 ACT-VLA。

**来源和范围：** [元数据](https://arxiv.org/abs/2607.00351)、[原文 §II-B、§III-A–B](https://arxiv.org/html/2607.00351v1)。核心竞争路线；已核方法，未复现。

## 核心机制与证据边界

以 text latent interpolation 组合已有任务行为，在模拟器生成候选轨迹，经成功筛选形成训练数据，再训练 VLA。它包含环境 rollout 和筛选成本，不能记成零交互的数据扩充。

§III-A 明确采用 Li 的 TLI；[TLI v5 §3.2–3.3](https://arxiv.org/html/2505.03500v5) 从演示得到任务文本 latent，并在推理残差流中插值。ACT-VLA 将此能力用于离线合成。CRAFT §2.1 将该合成示范路线列为相关工作。完整两端证据见 [机制溯源证据](evidence.md#trace)。

## 对本项目的价值

latent 可以组合并不自动意味着 latent 是可命名、对象无关的原子技能；表示的可操纵性与语义可识别性需要分别验证。

对当前方案，关键是避免把组合能力的提升全部归因于分解结构。给基线同样的组合覆盖或合成数据后，显式 atomic skill / target grounding / action representation 还剩多少收益，是更强的检验。

建议同时报告原始示范量、合成尝试量、成功接受量、训练步数和模拟器特权信息。再将 ACT-VLA 的数据路线与 CRAFT 的表示监督路线、Decompose and Recompose 的检索路线对齐比较。本卡保留它是因为它能推翻“只有显式原子结构才能改善组合”的过强解释。
