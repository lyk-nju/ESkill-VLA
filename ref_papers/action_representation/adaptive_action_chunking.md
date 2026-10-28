# Adaptive Action Chunking：轻量长度头与动作联合学习

> **当前定位（2026-10-28）：补充 · 按需时域参照。** 动作执行长度的参考；当前整段粗空间路径不引入长度预测。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

**论文：** Adaptive Action Chunking for Robotic Imitation Learning。Qingpeng Wen、Haomin Zhu 等；Biomimetics 2026, 11(5), 316；DOI: 10.3390/biomimetics11050316。

[期刊原文](https://www.mdpi.com/2313-7673/11/5/316) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13204913/) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/42187383/)

**核查范围：** 整合此前方法核查；本次核对出版入口与期刊索引。期刊正文返回 429、PMC 返回验证页，本次未重新完整打开正文，未复现。

## 可借鉴机制

共享视觉特征上联合学习动作预测与 chunk 长度分类。已有方法记录为最大长度动作输出后选择前缀，长度标签来自难度启发式。

## 对当前设计的作用

支持使用轻量长度头，而不是必须加入大型规划器。执行 chunk 长度不同于 Event 剩余时长；固定输出后裁剪也不等于变长 Token 训练。

具体实现前仍应回到可访问正文复核标签规则与输出裁剪代码。本卡不将它作为变长时序训练的直接证据。
