# TaKSIE：子目标选择、进度记忆和切换条件

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为替代分段、进度或组合评测的按需参考；当前先落实可核验事件边界与监督接口。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/event_organization.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文**：Incorporating Task Progress Knowledge for Subgoal Generation in Robotic Manipulation through Image Edits；Xuhui Kang、Yen-Ling Kuo；WACV 2025；arXiv:2410.11013v2，2024-12-17。归属：Insight 0；关联未来目标、action representation。

**原始来源与阅读范围**：[论文](https://arxiv.org/html/2410.11013)、[会议论文](https://openaccess.thecvf.com/content/WACV2025/papers/Kang_Incorporating_Task_Progress_Knowledge_for_Subgoal_Generation_in_Robotic_Manipulation_WACV_2025_paper.pdf)、[作者项目页](https://live-robotics-uva.github.io/TaKSIE/)。已核对 §3.1–3.4；未复现。2026-10-27 检索。

## 论文实际做了什么

用视觉表征到终态的距离曲线、平滑和斜率变化选关键子目标；递归进度编码器条件化图像编辑模型生成下一子目标，目标条件策略负责执行。在线进度评价器比较当前与生成目标的表征相似度，达到阈值或超时后更新子目标。

因此它确实包含部署时的切换模块。生成的未来目标图像与本项目定义的“当前空间 target grounding”不同，进度相似度也不是可靠物理完成的保证。

## 对本项目的判断

最有价值的启发是将三个原本容易混写的决策拆开：离线选择哪些状态做监督；模型是否记得已经选择的动作模式；在线何时刷新中间目标。Event 标签质量提高不意味着完成判断也更准确，进度记忆有效也不意味着必须生成整张未来图像。

对 atomic skill / target grounding / action representation 而言，同一个 atomic skill 和 target grounding 可以对应多种合理 action representation。若每个动作窗都独立重采样，机器人可能在推和拉、左右绕行之间来回切换。可以先测试事件内 action representation 的连续性或低频刷新，而不把整个子目标图像生成器加入 Phase 1。

同时，终态距离曲线可能受背景、摄像机运动、遮挡以及“先远离再接近”的必要步骤影响。本项目应检查这些失败模式，而不是把视觉进度直接当作单调的真实任务进展。

## 最小验证与证伪条件

做两因素对照：边界来源使用固定间隔或进度关键点；更新规则使用固定频率或相似度/超时。其余模型、目标预算与监督保持一致，才能区分好子目标与好刷新规则。

针对同技能的多条可行轨迹，记录模式切换次数、路径连续性与成功率，比较无记忆、事件内持久 latent、递归进度记忆。若简单持久 latent 已实现相同收益，完整图像生成可留到后续阶段；若误判完成频繁，则应单独改进评价器，不能通过扩大 atomic skill 词表掩盖问题。

**后续核查**：生成目标与可达目标的偏差；相似度阈值跨任务迁移；超时强制推进在失败恢复中的作用与风险。

返回 [方向论文索引](README.md)。
