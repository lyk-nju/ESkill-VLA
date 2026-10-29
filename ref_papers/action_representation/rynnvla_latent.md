# RynnLAM：三维运动与相机分支

**用途：补充机制，按需阅读。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../docs/action_representation.md)，实现以 [design.md](../../docs/design.md) 为准。

## 原工作机制

官方项目的 RynnLAM 用帧对学习连续运动 latent，结合未来特征、动态 3D flow 和相机分支；VLA 先预测潜动作序列，再适配真实本体动作。

## 对本项目的参考价值

保留三维运动监督、相机运动与本体适配分别处理的接口价值，服务后续人类视频数据设计。

## 采用边界

这是官方技术项目，原卡未核实论文 ID。ktoken_zcam 的 608 维含相机分量，K-token 不是本项目语义 Atomic Skill；预训练检查点不等于可部署策略。当前不导入此教师。

## 原文与阅读范围

[来源入口](https://github.com/alibaba-damo-academy/RynnVLA-Latent) · [补充来源](https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io/) · [补充来源](https://huggingface.co/Alibaba-DAMO-Academy/RynnLAM)

既有阅读记录：官方 README、项目页、RynnLAM 模型卡和 inference.py 局部代码。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
