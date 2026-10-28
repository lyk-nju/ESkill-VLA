# CLAP：机器人动作锚定与人类视频伪正例

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos。Chubin Zhang 等。  
**首发：** 2026-01-07；本轮依据 **v2，2026-06-14**，属于旧作新修订。  
**定位：** action representation / support；优先作为 Phase 2 的监督来源对照。

[版本入口](https://arxiv.org/abs/2601.04061) · [v2 原文](https://arxiv.org/html/2601.04061v2)

## 已核机制与来源

核读 v1/v2 §II、§III-C 与 Algorithm 1–2，并阅读 v2 策略预训练概述。Act-VAE 先用真实机器人动作建立码本，再冻结动作编码器及码本。视觉转移分为动作相关与环境两路；机器人样本的对比正例来自配对真实动作，人类样本在 v2 中来自处理干净帧对的 EMA teacher，在线分支看增强输入。

v1 的人类样本正例描述存在同次编码自对比问题；v2 明确改为 EMA teacher。复用方法应以 v2 为准。§II 对视觉重建式 latent action 作出批评；其机器人动作锚定与本库 CLAM 属机制对照，此处没有认定直接继承关系。

## 不宜从报告直接推出的结论

“把人类视频映射入机器人锚定空间”可以作为机制概括，但没有人类视频逐段对应的真实机器人动作标签。共享码本及自蒸馏不保证任意人类动作都能在目标机器人上执行，也不等同于对象不变的 atomic skill。原文仍使用视觉重建辅助，不能概括为完全不用重建。

## 对项目的增量判断

在 Phase 2 比较“只由视频学教师”与“机器人动作锚定教师”，动作标签总预算保持一致。为每条监督记录来源：真实配对、视频自蒸馏或伪标签，防止把监督优势归结为架构优势。

Phase 1 暂不整体移植其训练链；先检验同等机器人标签预算在 teacher 端或最终控制端使用的差异。当前 atomic skill / target grounding 的并行定义保持独立。

没有核验代码实现与论文 v2 是否完全一致，也没有复现。返回 [action representation](README.md)。
