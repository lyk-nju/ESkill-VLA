# CLAM：连续潜动作与真实动作联合接地

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations  
**版本：** arXiv 2505.04999v2，2026-07-30；初稿 2025，arXiv 标注 IROS 2026。  
**归属：** Insight 3。

[论文与 PDF](https://arxiv.org/abs/2505.04999v2) · [作者方法页](https://clamrobot.github.io/)

## 已核对的机制

作者方法页与官方摘要核对。连续 IDM/FDM 从观测转移学习 latent，同时利用少量带动作的 task-agnostic play 联合训练动作 decoder。专家示范可以只有观测；部署时由策略产生 latent，再解码为控制。FDM 服务训练。

## 证据边界

不能写成全流程无动作监督：play 数据仍有真实动作。论文在所测控制任务中支持连续表征与联合接地，不能推出任何视频都能直接迁移到任意机器人。旧项目页与 v2 的任务范围不同，本卡以新版为准。

## 对本项目的推论

Phase 1 已有真实动作，没必要刻意把 action representation 学习限制成纯无监督。应比较连续/离散表征和是否联合动作监督这两个因素，避免把改进全部归因于连续性。

同一 atomic skill / target grounding 下做多个可行路径示范，检查 decoder 能否保留差异；还应测不同初始关节状态的可执行性。连续 latent 本身并不保证多模态，prior 的分布建模需另行设计。

## 后续精读

核对 v2 全文中 action decoder 的条件、联合训练梯度、play/expert 划分与状态观测假设，之后再决定是否适合作为直接复现基线。

关联 [UniVLA](univla.md) · [理论诊断](what_lam_learn.md)。
