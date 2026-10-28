# ReconVLA：局部视觉 latent 去噪

**用途：当前方法参考。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../docs/target_grounding.md)，实现以 [design.md](../../docs/design.md) 为准。

## 原工作机制

当前被操作 gaze 区域经 VAE 编码成视觉目标，reconstructive tokens 条件化 latent 扩散去噪，并与动作目标联合训练；区域标签由检测模型构造。

## 对本项目的参考价值

当前三角色视觉 latent 的主要监督来源：用局部视觉信息训练预测 Token，保留外观与空间关系。ESkill-VLA 将目标扩展为当前执行器、当前对象及事件终点区域，并采用保留原图布局的画布。

## 采用边界

三角色共享 Query Decoder、目标画布与区域加权损失属于本项目设计。推理按 design.md 移除监督用 VAE／去噪器，保留预测锚点；不从原文推导部署必须生成图像。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2508.10333v1)

既有阅读记录：v1 §3.2–3.3；ReconVLA §1 明确受 [ROSS v2 §4](https://arxiv.org/html/2410.09575v2) 的重建式视觉指令训练启发。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

来源关系：[MAE v3 §3](https://arxiv.org/html/2111.06377v3) 提供遮挡重建动机，[ROSS v2 §4](https://arxiv.org/html/2410.09575v2) 转为视觉 latent 去噪，ReconVLA 将监督聚焦到操作区域；概念继承不等于复制相同网络。

返回 [本方向索引](README.md)。
