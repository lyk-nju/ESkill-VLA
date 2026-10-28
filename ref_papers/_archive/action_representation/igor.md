# IGOR：图像目标转移同时服务策略和世界模型

> **当前定位（2026-10-28）：归档 · 历史归档。** 保留历史机制卡；核心位置让给更直接的近期接口与多模态策略比较。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** IGOR: Image-GOal Representations are the Atomic Control Units for Foundation Models in Embodied AI  
**来源：** arXiv 2411.00785；2024。  
**归属：** Insight 3，关联 atomic skill 与未来子目标。

[正文](https://arxiv.org/html/2411.00785) · [官方方法页](https://www.microsoft.com/en-us/research/project/igor-image-goal-representations/)

## 已核对的机制

正文 §2.1–2.3。IDM/VQ 压缩视觉转移；IDM 与 FDM 使用不同 crop，FDM 只取当前帧以减少历史旁路。latent 同时供世界模型和高层策略使用。控制微调加入低层动作预测模块，并冻结其他已训练组件。

## 证据边界

这里的 atomic control unit 是视觉转移表示，不能仅因名称含 atomic 就等同于 pick/place 类别 atomic skill。跨本体视频迁移与低数据机器人控制证据应分别描述；真实动作条件世界模型微调在所读版本属于后续工作。

## 对本项目的推论

这一先例提示：同时有高层 latent、目标效果与低层控制本身不是充分新意。需要证明 atomic skill 与 target grounding 分离对未见组合提供了超出单个转移 latent 的收益。

可构造联合 latent 与 atomic skill / target grounding / action representation 分工的容量匹配对照，固定训练数据和损失预算，测对象替换、技能替换与顺序重组。跨 crop 约束需小心处理抓取点等不能被丢掉的局部几何。

## 后续精读

latent 条件进入低层策略的位置、时间间隔、跨域数据过滤与低样本协议。观察是否仍绑定屏幕位置或对象效果。

关联 [方向论文索引](../../atomic_skill/README.md) · [LaWAM](../../action_representation/lawam.md)。
