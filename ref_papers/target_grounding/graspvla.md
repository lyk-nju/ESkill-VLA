# GraspVLA 的感知与动作监督对齐

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

GraspVLA 值得用来检验：target grounding 的收益究竟来自一个新模块，还是来自互联网定位数据与机器人动作数据共享了可学习的中间接口。

## 论文与阅读范围

- 正式题名：GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data。
- 版本：arXiv 2505.03233v3，2025-08-27；CoRL 2025。
- 来源：[原文](https://arxiv.org/html/2505.03233v3)、[版本记录](https://arxiv.org/abs/2505.03233)、[作者项目页](https://pku-epic.github.io/GraspVLA-web/)、[会议论文](https://proceedings.mlr.press/v305/deng25a.html)。
- 阅读深度：主文 §4 模型与联合训练，§5.1 真实抓取评估设置；未复现。
- 关联：主要是 Insight 2；与 Insight 3 的连续动作专家接口相邻。

## 原工作的接口与证据

| 项目 | 原文方法 |
| --- | --- |
| 数据监督 | SynGrasp-1B 合成动作数据与 GRIT 互联网 grounding 数据联合训练 |
| 中间输出 | 二维对象框；机器人数据另有 robot base frame 下的 grasp pose |
| 动作接口 | Flow-matching 动作专家读取包括中间 token 的 VLM KV cache |
| 部署输入 | 图像、语言及机器人 proprioception；中间框与抓取姿态由模型预测 |
| 评估口径 | 真实抓取成功包含最多三次尝试；不是任意长任务的零样本组合评估 |

互联网样本主要承担框监督，合成机器人样本连接框、抓取姿态和动作。其感知到动作的串行预测是该模型的具体设计。[§4–5.1](https://arxiv.org/html/2505.03233v3)

## 对本项目的推论

应该把“统一监督接口”作为 atomic skill / target grounding / action representation 架构之外的竞争解释。假如普通动作模型通过 bbox 与关键位姿联合训练就能获得同等泛化，新增 target grounding query 的必要性便需重新论证。大规模合成数据也意味着不同模型间的分数差异不能直接解释为结构优势。

抓取姿态包含未来可执行交互约束，信息量高于当前对象定位。对比时应把 bbox、当前几何与未来 grasp pose 分开标记。否则所谓更好的 target grounding 可能实际来自额外未来信息。原论文的串行接口也不要求本项目改成 atomic skill→target grounding；本项目仍可以让 atomic skill、target grounding 并行，再共同约束 action representation。

## 可区分机制的实验

固定动作数据、图像分辨率和专家规模，依次加入 bbox、对象相对夹爪位姿、未来 grasp pose、ROI 重建。标签来自同一仿真轨迹，分别统计生成成本与部署依赖。保持未来信息量一致后，再比较显式 token 与隐式 query。

抓取评估同时报告单次成功和三次内成功；组合任务另外测试 pick 后的目标转移与 place，避免用抓取专门化结果替代跨事件复用证据。后续优先查看感知监督消融，以及新外观、新几何和多实例目标的误差是否被分别统计。

返回 [方向论文索引](README.md)。
