# Faster-WAM：条件缓存与动作融合

**用途：补充机制，按需阅读。** 本卡解释来源，不增加实施要求；历史动作路线见 [方向文档](../../history/action_representation.md)，历史实现见 [joint_design.md](../../history/joint_design.md)。

## 原工作机制

一次视频专家前向保留带噪未来 slots，形成可缓存条件，以 SparseMoT 和 Interval KV-Fusion 与动作专家交互。

## 对本项目的参考价值

保留一次条件编码、在动作生成迭代中缓存复用的工程思路。当前动作块缓存原始 VLM K/V 与预测 Motion K/V，不进行视频 rollout。

## 采用边界

不移植其未来 slots 或融合架构；真实 KV 接口仍以所选基座为准。该文与另一篇讨论浅动作模块的 Faster-WAM 不同。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2608.04404v1)

既有阅读记录：v1 Introduction、Method、Future Conditioning 分析。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
