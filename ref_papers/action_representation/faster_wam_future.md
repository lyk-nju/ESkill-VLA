# Faster-WAM（Zhao）：单次未来条件与稀疏融合

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models  
**论文标识：** [arXiv 2608.04404v1](https://arxiv.org/abs/2608.04404v1) · [原文](https://arxiv.org/html/2608.04404v1)  
**日期：** v1 2026-08-05。核查 2026-10-27。  
**本轮分层：** core；Insight 3 action representation。

## 已读证据

相关方法核读：Introduction；Method；Future Conditioning 消融。保留带噪未来 slots，以单次视频专家前向得到可缓存的未来条件，用 SparseMoT 和 Interval KV-Fusion 控制与动作专家的交互。作者另有保留融合设计的 current-only 消融，支持该设置中的未来条件贡献。

## 证据边界

未来条件不是完整视频 rollout。OOD 增益来自具体受控实现，不能推出所有 policy 都必须显式想象。

## 对当前项目的推论

与仅当前帧 KV、LaWAM 未来特征接口做同预算比较，单独控制动作专家深度，避免把融合优化误当未来预测收益。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
