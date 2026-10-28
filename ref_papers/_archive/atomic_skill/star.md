# STAR：码本利用率与语义多样性应分别验证

> **当前定位（2026-10-28）：归档 · 历史归档。** 主要增加量化训练/码本覆盖技巧；本阶段由 QueST、VQ-BeT 及主分支 X-Tokenizer 覆盖强动作抽象对照，暂不占主阅读位。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization*，Hao Li 等，ICML 2025。arXiv:2506.03863；本轮核对 HTML v3（2026-04-07），调研日期 2026-10-27。该版本晚于会议版，后续复现须记录采用哪一版。

**原始来源与阅读范围：** [论文 §3.2.1–3.2.4](https://arxiv.org/html/2506.03863v3)、[ICML 页面](https://proceedings.mlr.press/v267/li25bo.html)。已核对方法结构与训练接口，未核对代码实现或复现数值。

## 原工作做了什么

先把动作序列编码为经过旋转增强的残差量化表示，再冻结抽象模型训练 Causal Skill Transformer。后者根据观测和任务条件，沿残差码本层级自回归预测粗到细的码；另有连续 offset 修正，解码后滚动执行。这里的 causal 首先指预测依赖和注意力结构，不是识别了操作因果关系。

## 对 atomic skill / target grounding / action representation 的意义与边界

STAR 扩展了 VQ-BeT、QueST 所代表的动作抽象路线。它提示我们先排除量化器塌缩或容量不足，再讨论离散空间是否学到技能。

但更高的码本利用率、perplexity 或动作覆盖范围，不等于更丰富的行为语义。残差量化的首层也不能未经检验就命名为 atomic skill、后续层命名为 action representation：粗码可能主要表示绝对位置或运动幅度，细码也可能承载决定动作类型的接触变化。

对本项目，STAR 更适合作为强运动 token 对照。若语义 atomic skill 仅改善量化训练稳定性，而没有改善跨对象组合，则研究结论应明确落在优化或表示容量上。

## 可以区分机制的实验

在相同数据、总 token 比特预算、连续 offset 预算和动作 decoder 下，比较普通 VQ、FSQ、残差量化及旋转增强。分别测量码本利用率、给定语义类别内的运动覆盖、跨类别混淆以及闭环成功率。

再检查残差各层的条件探针：已知 target grounding 后首层是否仍主要编码空间位置；固定首层、交换细层是否保持行为类型。交换应在物理可行状态内进行，不能把明显非法的动作解释成语义失败。

## 下一步精读

跟踪会议版与 v3 的差异；核对旋转参数、量化解码及 offset 的实现一致性，控制总容量与微调预算。不要把自回归“因果性”写成我们的因果干预证据。

相关：[QueST](../../atomic_skill/quest.md)、[VQ-BeT](../../atomic_skill/vq_bet.md)、[方向论文索引](../../atomic_skill/README.md)。
