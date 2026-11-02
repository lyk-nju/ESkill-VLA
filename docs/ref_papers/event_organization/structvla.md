# StructVLA：稀疏结构的训练监督

**用途：研究依据，按需阅读。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../event_organization.md)，实现以 [design.md](../../design.md) 为准。

## 原工作机制

结合夹爪变化、速度信号、间隔约束和 VLM 筛选得到关键帧；先学习稀疏未来帧，再转向 FAST 动作 token 训练。离线筛选 VLM 不进入部署。

## 对本项目的参考价值

提供通过训练监督吸收任务结构的实例。物理里程碑与语义 Event 可以具有不同粒度，稀疏结构监督也不要求部署时生成完整未来视频。

## 采用边界

当前不引入其未来帧预训练或 FAST 输出。该文不证明本项目 Atomic Skill、Target Grounding 与 Latent Motion 分工的独立收益，也不自动把每个关键帧变成一个 Skill。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2603.12553v1) · [出版／作者入口](https://wm-planner.github.io/structvla/)

既有阅读记录：v1 §3.1–3.3；未审计代码或完整实验配置。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
