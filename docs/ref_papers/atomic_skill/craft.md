# CRAFT：技能表征与动作学习

**用途：当前方法参考。** 本卡解释来源，不增加实施要求；历史动作路线见 [方向文档](../../history/atomic_skill.md)，历史实现见 [joint_design.md](../../history/joint_design.md)。

## 原工作机制

skill queries 读取视觉与语言，state queries 读取视觉，动作专家共同使用。训练跨执行交换 skill 表征，以对应 state 和动作监督对齐，并限制部分梯度路径。

## 对本项目的参考价值

技能表示需要参与实际动作计算，不能仅有语义聚类。当前通过连续 Skill Prior→Motion→Action 通路，让操作先验接受执行相关梯度；硬路由选择本身由 Skill CE 训练。

## 采用边界

CRAFT 的 skill 是 operation–entity，主要重组实体；本项目将类别与对象角色分开。其反事实交换、双次骨干编码和梯度限制不属于当前实现。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2610.00524v1)

既有阅读记录：v1 §2–4.4、§5.1、Appendix B.1。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
