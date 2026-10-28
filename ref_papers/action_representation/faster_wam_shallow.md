# Faster-WAM（Ma）：当前帧世界模型与浅动作专家

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为视频教师、生成式运动或本体适配的扩展比较；当前先验证剩余时长与变长粗三维运动 Token。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/action_representation.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**原题：** Faster-WAM: Do World Action Models Need Deep Action Modules?  
**论文标识：** [arXiv 2608.02365v2](https://arxiv.org/abs/2608.02365v2) · [原文](https://arxiv.org/html/2608.02365v2)  
**日期：** v1 2026-08-03；v2 10-06。核查 2026-10-27。  
**本轮分层：** support；Insight 3 action representation。

## 已读证据

相关方法核读：§2.1–2.5；训练/推理附录定位。继承 Fast-WAM 当前帧 prefill，利用 DoT/Lite KV-Fusion 让浅动作专家访问各层视频 KV；语言和本体状态经视频模型传入。测试不生成未来视频。

## 证据边界

与 2608.04404 是两篇不同工作，v2 §2.5 明确比较该并行路线；两者测量设置不同，速度倍率不能直接相除。

## 对当前项目的推论

支持一个更便宜的反证基线：有视频预训练表征即可改善控制时，应进一步证明部署 action representation 或未来生成的必要性。

来源关系见 [本轮溯源](evidence.md#trace)；结果未经本项目复现。
