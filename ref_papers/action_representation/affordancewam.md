# AffordanceWAM：哪些未来 World 信息应该传给动作

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原文：** *AffordanceWAM: Affordance-Aware Joint World–Action Modeling for Robot Manipulation*，arXiv 2609.22332。v1 2026-09-16；本轮读 v3，2026-10-18。[版本记录](https://arxiv.org/abs/2609.22332) · [原文](https://arxiv.org/html/2609.22332v3)。

**机制证据。** §3.2–3.4 将未来 RGB、scalar affordance 和 RGB-rich affordance heatmap 纳入 World 学习，并限制动作只读取指定条件通路。人类视频参与 World 学习，机器人样本再提供动作监督。任务后训练显式冻结 World、停止接口梯度；部署仍使用预测 World。Table 11 显示打开 heatmap→action 后 RoboCasa 由 69.4 降为 63.5，支持保留该通路限制在其设置下的价值。

**和项目的关系。** 它将“学习什么信息”和“向动作开放什么信息”分开，适合检查 target grounding 的辅助重建是否必须直接成为动作条件。它并未证明 RGB 信息无用，也未证明重建表征天然产生外观捷径。本文仍允许动作访问 RGB；未来 World 条件也不能当作免费附加信息。

**建议的项目实验。** 固定预测器与数据，比较三组：重建仅作训练监督、读取紧凑交互量、读取重建特征。每组分别测试正确目标替换、无关外观替换和相同目标的多种合理动作。另行比较联合更新与 detach，才能将条件容量、前向泄漏与梯度冲突分清。若引入未来 World，需单列其生成成本与预测误差，不能把收益全部归给 atomic skill 或 target grounding。

**证据边界与关系类型。** 本轮读 §2、§3.1–3.4、§4.3；没有复现。与项目及 AffordanceVLA 的联系在此是机制比较，不声称后者被本文直接沿用。此卡应链接 World/action representation 主线，而不因名称中的 affordance 只放入视觉接地清单。
