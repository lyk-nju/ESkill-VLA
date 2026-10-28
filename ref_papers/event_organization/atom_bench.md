# ATOM-Bench：单项能力与组合表现的评测对照

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为替代分段、进度或组合评测的按需参考；当前先落实可核验事件边界与监督接口。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/event_organization.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**定位：评测支线，建议归入 Insight 0；不是 Event 分段方法。**

**论文：** *ATOM-Bench: A Real-World Benchmark for Atomic Skills and Compositional Generalization in Manipulation Policies*，Zenan Wu 等，arXiv:2606.16826v1，2026-06-15。复核日期：2026-10-27。

[原文](https://arxiv.org/html/2606.16826v1) · [作者项目](https://flageval-baai.github.io/AtomBenchPage/)

## 已核对的机制

§3–4：两个机器人平台共有 30 个原子任务、24 个保留组合任务。原子分 motor 与 instruction 两类；训练仅用原子任务。PSR 是预定义过程得分点的完成比例；对组合任务 X：

\[
\mathrm{AS}(X)=\frac{1}{|A(X)|}\sum_{a\in A(X)}\mathrm{PSR}(a),\qquad
\mathrm{CFS}(X)=\frac{\max(0,\mathrm{AS}(X)-\mathrm{PSR}(X))}{1-\mathrm{PSR}(X)}.
\]

指标逐任务计算后宏平均；AS 衡量孤立原子的表现。论文 §2 明确引用 LIBERO-PRO；不是继承其扰动实现。

## 本项目如何使用，哪些解释不成立

建议先报告整任务成功率、分项成功和过程得分，再考虑差距指标。CFS 是操作性诊断；它不是经干预识别的“组合导致失败的百分比”。AS 是平均原子进度，不能当作串行任务成功概率的理论上界。即使每段成功率相同，长序列也会累积失败；起点变化、目标难度和评分权重亦可能造成差距。

本地实现需要预注册三项规则：PSR=1 时 CFS 记为不适用并报告数量；近满分时标出分母敏感性；先逐任务算比值再宏平均，不能代入总体均值计算。低 CFS 也不等于可组合，可能只是原子能力太低。

最小迁移是借用分项能力评测思想，在同一 RoboTwin 任务族配对检查单事件与组合；加示范初态/真实交接终态对照，检验差距是否主要来自起点分布。无需复刻两个真机平台，也不能把自定义结果称为官方 benchmark 成绩。

阅读：§2、§3、§4.1–4.4、§6；未复现、未核查评分代码。返回[本轮复核](evidence.md#review)。
