# Moto：运动 token 有训练价值，也可以在推理时省略

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos  
**版本：** arXiv 2412.04445v4，2025-10-16；ICCV 2025。  
**归属：** Insight 3，关联离散运动码。

[正文](https://arxiv.org/html/2412.04445v4) · [正式论文入口](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Moto_Latent_Motion_Token_as_the_Bridging_Language_for_Learning_ICCV_2025_paper.html)

## 已核对的机制

正文 §3.2–3.4。帧对经 M-Former 与 VQ 得到运动 token，自回归预训练学习运动序列；真实动作微调插入 action query。训练时随机屏蔽 action query 对运动 token 的部分注意力，因此推理可直接查询真实动作而不先生成运动 token。

## 证据边界

运动 token 不等同于对象无关的离散语义技能 atomic skill。模型在所测任务中获益，仍不足以证明显式中间 token 是部署必需，也不能把生成多种 token 等同于有效动作多样性。

## 对本项目的推论

用相同预训练知识比较“显式 action representation”和“训练时学 action representation、测试直接动作”，可以避免把额外延迟当作结构贡献。需要同时报告动作精度、组合泛化和每秒控制更新。

可借鉴训练时条件屏蔽，但应监测完全绕过 atomic skill / target grounding / action representation 的可能；条件替换与因果干预比注意力图更能诊断依赖关系。

## 后续精读

核对各实验的 token 推理开关、历史长度与联合损失权重；复现时固定同一动作数据和骨干。

关联 [LAPA](lapa.md) · [Fast-WAM](fast_wam.md)。
