# RynnWorld-Latent：把同一潜动作接到世界模型

> **当前定位（2026-10-28）：归档 · 历史归档。** 与 RynnLAM 重复且侧重视频生成，闭环控制证据不足；保留追溯入口。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**项目：** RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions  
**状态：** 官方 README 核对，2026-10-27；未核实正式论文 ID，未复现。  
**归属：** Insight 3，关联跨本体迁移。

[官方仓库](https://github.com/alibaba-damo-academy/RynnWorld-Latent)

## 已核对的机制

用 RynnLAM 连续潜动作条件化视频世界模型。公开说明的输入包括初始帧与 20 个潜动作，预测 80 个后续帧；样本共 81 帧，每 4 个 RGB 帧对应一个 latent。接口使用含 camera 分量的 608 维 descriptor。

README 描述的应用包括目标机器人适配后的数字遥操作，以及编辑初始图、复用潜动作的数据合成。模型接口不依赖文字条件。

## 证据边界

这些是作者官方技术说明，不能代替正式实验或第三方复现。跨外观生成、跨本体视频迁移、真实机器人可执行性是不同证据。输出视频的稳定性也不足以保证接触动力学正确。

## 对本项目的推论

同一 action representation 同时服务 policy 与 world prediction 是值得测试的接口选择。可在不改变主控制链的前提下，增加小型前向预测头，观察它能否改善 action representation；只有独立收益成立后，再考虑部署世界模型。

建议先做反事实诊断：固定初始图更换 action representation、固定 action representation 更换对象与视角，检查对象作用、运动方向与相机变化是否可分。Phase 2 应另测适配真实动作的样本效率。

## 待核实

完整训练配方、动作适配器、公开权重对应的数据与许可、长滚动误差。README blob：e844e598a37922b31090b064929b01a7c480bdce。

关联 [RynnVLA-Latent](../../action_representation/rynnvla_latent.md) · [AdaWorld](../../action_representation/adaworld.md)。
