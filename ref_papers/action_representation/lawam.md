# LaWAM：action representation 经未来视觉特征连接动作专家

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies  
**版本：** arXiv 2606.15768v2，2026-10-25。  
**归属：** Insight 3，关联 future subgoal。

[正文](https://arxiv.org/html/2606.15768v2) · [作者页](https://rlinf.github.io/LaWAM/)

## 已核对的机制

正文 §3.2–3.3。IDM 从当前/未来特征得到 action representation，前向 decoder 学未来特征，并有末端状态辅助预测。第二阶段 prior 从当前观测和语言回归 action representation，经保留的 decoder 得到未来特征，再条件化动作专家。未来真值只提供监督；动作损失通过 stop-gradient 与 subgoal 支路隔离。

## 证据边界

这是显式未来特征接口，不是target grounding（当前交互空间）。prior 是确定性点预测，不能据此声称 action representation 保留了多峰动作分布。末端辅助信号和物理时长对齐也应计入对照；附录具体混合数据掩码仍需进一步核查。

## 对本项目的推论

这是“action representation 仅用于预训练”和“action representation 直接解码为动作”之外的第三种实现。可以把当前 target grounding 与预测未来 U 分开，比较动作专家接收 action representation 或 U 的控制精度、组合收益及延迟。

优先做相同 backbone 下的小规模接口对照；不要因不同论文总榜结果直接迁移整个架构。对多样性另比较分布 prior 与点回归，并固定采样次数。

## 后续精读

附录 E 的物理时长编码、不同数据的辅助监督可用性、预测接口与容量匹配消融。尤其要确认当前观测旁路是否让 action representation 成为可忽略条件。

关联 [RLA-WM](rla_wm.md) · [Fast-WAM](fast_wam.md)。
