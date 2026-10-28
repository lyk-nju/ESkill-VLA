# RLA-WM：把状态转移残差压缩后建模

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** Learning Visual Feature-Based World Models via Residual Latent Action  
**来源：** arXiv 2605.07079，查阅当前 HTML 于 2026-10-27。  
**归属：** Insight 3，特征空间世界模型。

[正文](https://arxiv.org/html/2605.07079) · [作者页](https://mlzxy.github.io/rla-wm/) · [代码](https://github.com/mlzxy/rla-wm)

## 已核对的机制

正文 §3、§4 概述。先把未来与当前 DINO 特征之差压缩成 RLA，结合当前特征解码未来。世界模型以当前状态和真实动作块为条件，通过 flow matching 生成紧凑 RLA，再解码未来特征。论文还探讨辅助行为克隆与模型内 RL。

## 证据边界

RLA 的直接学习对象是视觉转移残差，不能先验称为纯 motor code。自编码阶段可用观测，动作条件世界模型则仍需要动作；不可把全流程描述为完全无动作标签。预测质量与控制收益需分别评价。

## 对本项目的推论

残差接口值得作为 action representation 的第三类 teacher，与动作编码、帧对编码比较。它是否减少静态外观信息，应使用背景、视角和对象替换检验，不能仅凭“做了减法”推断不变性。

在同一 atomic skill / target grounding 下采样多个 RLA，测不同未来是否对应可执行轨迹。若未来分布的变化主要来自遮挡或外界随机性，应避免误算成机器人运动多样性。

## 后续精读

§4.2 中 BC-RLA 的梯度接口、专家数据与 play 数据划分，以及生成模型的不确定性如何影响动作选择。本文尚未进行训练复现。

关联 [LaWAM](lawam.md) · [理论诊断](what_lam_learn.md)。
