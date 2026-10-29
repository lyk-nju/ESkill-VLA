# AffordanceVLA：角色 Query 与空间条件

**用途：当前方法参考。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../docs/target_grounding.md)，实现以 [design.md](../../docs/design.md) 为准。

## 原工作机制

不同 queries 分别学习对象 ROI 的 VAE latent、二维交互热图、三维 shape 与 layout，动作专家读取相关表征。原文使用多专家、多阶段训练，部署保留 affordance generation。

## 对本项目的参考价值

提供按交互职责组织 queries、让空间表征参与动作条件的参照。当前将三个角色统一为视觉 latent，使用共享预测器和训练期去噪器。

## 采用边界

不移植其热图、OBB、接触方向或完整生成架构。How2Act 主要监督对象几何，不等同于本项目的运动 Latent Motion。单向 attention mask 约束前向可见性，不自动阻断动作梯度。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2606.06155v2)

既有阅读记录：v2 §2、§3.1–3.3、§3.1.4 与附录 C/D/F/G/H；引用 ReconVLA 不等于已确认代码继承。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
