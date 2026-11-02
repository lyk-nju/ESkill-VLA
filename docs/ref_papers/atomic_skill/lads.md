# LADS：目标表示与在线预测

**用途：补充机制，按需阅读。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../atomic_skill.md)，实现以 [design.md](../../design.md) 为准。

## 原工作机制

未来轨迹编码离散 plan，低层结合状态重建动作；高层用当前信息和语言预测技能分布。stop-gradient 区分目标与预测路径，技能序列另做语言对比，整体联合优化。

## 对本项目的参考价值

保留“学习什么表示”与“在线如何选择”的区分，有助理解当前分类器、离散选择与下游技能表的职责及梯度边界。

## 采用边界

本项目不读取未来轨迹产生在线 Atomic Skill，也不引入其 plan tokenizer 或对比损失。LADS 的 stop-gradient 不能直接替换当前 argmax 的梯度语义。

## 原文与阅读范围

[来源入口](https://openreview.net/forum?id=L66G39JrM4) · [补充来源](https://proceedings.iclr.cc/paper_files/paper/2025/file/e7c786024ca718f2487712bfe9f51030-Paper-Conference.pdf)

既有阅读记录：正式出版 PDF §1–2、§4.1–4.5；§2 明确建立在 LISA 上。 本次仅据已存阅读记录整理，未重新联网核验或复现实验。

返回 [本方向索引](README.md)。
