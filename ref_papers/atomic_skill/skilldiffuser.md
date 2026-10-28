# SkillDiffuser：语义技能与多模态运动生成已有直接先例

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *SkillDiffuser: Interpretable Hierarchical Planning via Skill Abstractions in Diffusion-Based Task Execution*，Zhixuan Liang 等，2023 年预印本 / CVPR 2024。arXiv:2312.11598；本轮核对 v3（2024-03-28），调研日期 2026-10-27。

**原始来源与阅读范围：** [论文 §4.2–4.4](https://arxiv.org/html/2312.11598v3)、[CVPR 页面](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_SkillDiffuser_Interpretable_Hierarchical_Planning_via_Skill_Abstractions_in_Diffusion-Based_Task_CVPR_2024_paper.html)、[作者项目页](https://skilldiffuser.github.io/)。已核对方法及任务范围，未复现。

## 原工作做了什么

观测与语言驱动离散技能预测；技能条件的 state-only diffusion 产生未来观测表示，再由逆动力学模型结合当前视觉恢复动作。训练同时涉及技能量化、状态扩散与真实动作监督。技能作用于固定时窗，同一码可连续重复以覆盖较长子任务。论文在 LOReL、MetaWorld 等设置中研究解释性和组合。

## 对 atomic skill / target grounding / action representation 的意义与边界

这是比“用了 diffusion”更接近本项目的先例：**可解释技能条件化、多模态执行以及状态变化到动作的桥梁，已经被放在同一系统里。** 因此，“离散 atomic skill 条件化生成模型”本身不足以承载我们的创新叙事。

尚可研究的差别包括：可变长 Event 是否比固定时窗重复更有利；并行的当前空间 target grounding 是否提供独立信息；跨视频、本体迁移是否真的需要单独的 action representation。不能把该论文的状态生成器直接等同于当前局部 grounding，也不能把上述研究目标写成它未能解决的已证实缺陷。

## 对架构选择的反向约束

至少保留两条可比路线：直接预测动作的现有 Action Expert 加 atomic skill / target grounding 条件；预测状态转移再用逆动力学解码。比较时共享感知底座、训练数据和动作监督，并记录生成步数及控制延迟。否则更复杂系统的性能差别无法定位到知识分解。

可先在同一技能和目标下采样多种可行动作，检验运动多样性是否真的改善成功率；若加入显式 action representation 只增加采样噪声或延迟，则应把 action representation 留作训练期辅助目标。

## 下一步精读

检查技能聚类与语言条件之间的依赖、逆动力学面对预测状态误差时的鲁棒性、长任务中技能重复及切换的实现。将闭环行为证据与技能可视化证据分开整理。

相关：[LCSD](lcsd.md)、[LISA](lisa.md)、[方向论文索引](README.md)。
