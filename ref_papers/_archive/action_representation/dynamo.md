# DynaMo：动力学预训练也可以只改善视觉表征

> **当前定位（2026-10-28）：归档 · 历史归档。** 仅视觉表征预训练的旁支，当前不占独立核心阅读位。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Control  
**时间：** NeurIPS 2024，arXiv 2409.12192。  
**归属：** Insight 3，关联控制相关视觉特征。

[作者方法页](https://dynamo-ssl.github.io/) · [论文入口](https://arxiv.org/abs/2409.12192) · [正式会议页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b8db54b629e00537b59cbc6612026d7-Abstract-Conference.html)

## 已核对的机制

作者方法页核对。用领域内示范观测联合学习视觉编码器、latent IDM 和 FDM，以特征空间前向一致性进行预训练，不要求该阶段真实动作标签、对比采样或外部视频。学得视觉表征再供不同下游模仿策略使用。

## 证据边界

主要验证对象是视觉预训练，而非部署时显式输出 action representation 的策略。预训练不用动作不等于下游控制也不用动作监督。本卡核对作者方法页，未完成全文及代码细节审计。

## 对本项目的推论

这是必须保留的简单对照：如果动态辅助训练改善共享视觉特征已经获得同等收益，就应谨慎增加独立 action representation 接口。

最小实验：同一批机器人视频、同一训练步数，比较无辅助、重建当前图、预测未来特征、显式 action representation 四种方式。冻结/解冻视觉骨干分别记录，防止把可训练参数变化归因于目标函数。

## 后续精读

编码器防坍缩机制、IDM 的历史/未来访问范围、下游冻结策略和数据划分。优先看正式六环境版本，避免混入早期匿名项目页结果。

关联 [LaWAM](../../action_representation/lawam.md) · [RLA-WM](../../action_representation/rla_wm.md)。
