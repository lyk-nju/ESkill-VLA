# LADS：分开学习什么技能与如何在线预测它

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *Discrete Latent Plans via Semantic Skill Abstractions*，Haobin Jiang、Jiangxing Wang、Zongqing Lu，ICLR 2025，方法名 LADS。本轮未核实 arXiv 标识，按正式出版版本引用。

**来源和范围：** [OpenReview](https://openreview.net/forum?id=L66G39JrM4)、[正式 PDF §1–2、§4.1–4.5](https://proceedings.iclr.cc/paper_files/paper/2025/file/e7c786024ca718f2487712bfe9f51030-Paper-Conference.pdf)。核心；只核相关章节，未复现。

## 核心机制与证据边界

未来轨迹编码成离散 latent plan，低层结合状态重建动作；高层依据当前信息和语言预测离散技能分布。stop-gradient 区分目标技能与高层预测路径，整条技能序列另与语言做对比对齐。§4.5 明确联合优化，不能写成先冻结 tokenizer 再训练高层的顺序流程。

§2 明确建立在 LISA 上；引言引用 SPiRL 类潜技能学习及 LCSD 的稳定性问题。两端核查见 [机制溯源证据](evidence.md#trace)。

## 对本项目的价值

这个接口提醒我们同时定义训练 teacher 与部署 predictor。利用未来轨迹构造更好的 atomic skill 或 action representation 标签，不代表这些标签在部署时都可准确预测。

建议测三种条件：teacher 表征、在线预测表征、无显式表征的同容量底座。teacher 与在线版本的控制差距说明瓶颈在可预测性；teacher 本身也无增益，则应重新检查抽象定义。不能只扩大表示容量来解决两类完全不同的问题。

还可把“表示相互独立”“梯度路径分开”“训练分阶段”“部署频率不同”作为四个独立设计轴。LADS 与 DASL 提供其中若干组合，但没有替本项目证明哪一套是最优结构。
