# VISTA 的视觉依赖与轨迹偏好训练

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

这篇 VISTA 提醒我们先检验控制器是否使用视觉信息。target grounding 可以被训练得很好，却没有影响动作；改善训练目标也可能比增加新模块更有效。

## 论文与阅读范围

- 正式题名：VISTA: Enhancing Visual Conditioning via Track-Following Preference Optimization in Vision-Language-Action Models。
- 版本：arXiv 2602.05049v1，2026-02-04；预印本。
- 来源：[原文](https://arxiv.org/html/2602.05049v1)、[版本记录](https://arxiv.org/abs/2602.05049)、[作者项目页](https://vista-vla.github.io/)。
- 阅读深度：主文 §3.2–3.3、§4.1–4.3 与实验主表；未复现。
- 关联：Insight 2、3 的训练机制与评价；不是事件分段方法。
- 同名区分：世界模型 VISTA 是另一篇 [Scaling World Model for Hierarchical Manipulation Policies](https://arxiv.org/abs/2602.10983)。

## 原工作的接口与证据

论文用视觉扰动前后动作分布变化衡量视觉依赖。训练先将离线未来点轨迹画到当前图像，以对应动作与 batch 内错配动作构造 DPO；再从带轨迹的冻结 teacher 蒸馏给不带轨迹的指令策略。[§3–4](https://arxiv.org/html/2602.05049v1)

| 项目 | 原文范围 |
| --- | --- |
| 监督来源 | 原有机器人数据、离线点跟踪和动作偏好配对 |
| 轨迹坐标 | 当前图像中的二维轨迹叠加 |
| 部署输入 | 普通图像与指令，不要求未来轨迹 |
| 模型范围 | 离散 OpenVLA 与连续 OpenVLA-OFT |
| 结果边界 | 视觉依赖与成功存在关联；不同任务和序列长度的增益不完全一致 |

## 对本项目的推论

target grounding 的评价应同时包含表征质量与动作使用情况。仅报告 ROI 重建、点定位或注意力热图，会遗漏动作专家忽略 target grounding 的情况。VISTA 提供无需新增部署结构的对照，因此“加 target grounding 后提升”还需要与“相同骨干改善对齐训练”比较。

扰动敏感并不自动等于正确 grounding：模型可能对背景、遮挡或噪声同样敏感。需要控制变化的语义，把目标物体移动、角色交换和无关背景变化分开，检验动作响应是否符合任务。

## 可区分机制的实验

使用同一动作骨干比较 ROI 重建、track teacher 蒸馏、二者结合与基础训练。匹配训练步数，并记录离线跟踪和 teacher 的额外成本。“没有新增采集数据”不代表没有额外标签处理。

构造四种干预：移动正确目标、移动干扰物、交换同类对象角色、替换背景。观察动作端点与目标绑定的变化；进一步遮蔽或交换 target grounding latent，检验动作是否以任务相关方式依赖 target grounding。较高视觉敏感度只有与正确响应同步时，才支持有效接地。

后续需研究连续动作专家的依赖度量，以及单纯蒸馏是否已经达到显式 target grounding 的效果。

返回 [方向论文索引](README.md)。
