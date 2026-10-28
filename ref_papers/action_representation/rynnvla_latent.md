# RynnVLA-Latent / RynnLAM：三维运动监督与本体适配

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**来源状态：** 官方技术项目；截至 2026-10-27，未核实到对应 arXiv ID。README 的论文链接为空，不把项目介绍冒充论文全文。  
**名称：** README 使用 Scaling Robotic Manipulation Foundation Models with 3D-Grounded Latent Actions；项目页使用 Latent Action Pretraining for Robotic Manipulation Foundation Models。  
**归属：** Insight 3，关联 target grounding、Phase 2。

[官方代码与 README](https://github.com/alibaba-damo-academy/RynnVLA-Latent) · [作者项目页](https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io/) · [RynnLAM 模型卡](https://huggingface.co/Alibaba-DAMO-Academy/RynnLAM) · [推理代码](https://github.com/alibaba-damo-academy/RynnVLA-Latent/blob/main/rynnlam/inference.py)

## 已核对的原工作

官方说明与局部代码核对：README、项目站 index.html、RynnLAM 模型卡及 inference.py。RynnLAM 从帧对提取连续运动 latent，结合未来视觉特征、动态 3D flow 和相机运动分支。VLA 先用 flow matching 学潜动作序列，再把同一动作专家适配到真实本体动作；项目说明没有要求额外独立动作 decoder。

发布数据使用的 ktoken_zcam 是 608 维：8×64 motion tokens、64 维 global latent、32 维 camera latent。代码同时支持多种 representation；编码器函数默认 ktoken，不能与数据配置混淆。2B/4B 发布权重标为 Stage 1 预训练检查点。

## 证据边界

没有原生控制标签，不等于没有深度、3D flow、相机等几何监督及教师成本。代码中的 K-token 不是本项目的行为语义 atomic skill；完整 descriptor 含 camera 支路，也不能整体称为相机不变表示。已发布预训练权重不能直接视为闭环控制策略。

## 对本项目的推论

它使 action representation 的候选设计从“像素重建瓶颈”扩展到“几何约束的连续运动”。但必须先明确迁移对象：相机坐标中的视觉运动、物体相对运动，还是执行器可实现的控制。

最小比较可用相同机器人数据训练 RGB-only、RGB+2D flow、RGB+3D motion 三种 teacher；分别控制相机与物体运动，测动作可解码性、背景敏感性及控制成功率。对 camera 分支做可审查的保留/移除对照，不能只看聚类图。

## 待核实

正式论文与完整消融、几何标签的生成误差、Stage 2 的训练预算。源码核对锚点：README blob 9d3615de1452db14e253d72e6411a1c8b5ee3c83；inference.py blob 62aa42965416b792fa8c4479a514d15dec17cc6b。

关联 [RynnWorld-Latent](../_archive/action_representation/rynnworld_latent.md) · [方向论文索引](README.md)。
