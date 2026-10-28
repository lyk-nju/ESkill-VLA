# OpenHOUSE：人类视频中的在线、多粒度事件理解

> **当前定位（2026-10-28）：归档 · 历史归档。** 属于人类流式视频定位/描述，没有机器人执行迁移证据；Phase 2 启动前暂归历史扩展，当前由 GEBD 概念链解释粒度问题已足够。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/event_organization.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文**：Open-ended Hierarchical Streaming Video Understanding with Vision Language Models；Hyolim Kang 等；ICCV 2025；arXiv:2509.12145v1，2025-09-15。归属：Insight 0；关联 Phase 2 人类视频。

**原始来源与阅读范围**：[论文](https://arxiv.org/html/2509.12145)、[会议记录](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html)、[作者项目页](https://sites.google.com/view/yunsupark/projects/openhouse)。已核对 §3、§4.1；未复现。2026-10-27 检索。

## 论文实际做了什么

轻量流式 RNN 检测 substep/step 边界，动作状态和进度下降分别帮助识别开始与结束；稀疏调用冻结 VLM，结合历史记忆生成自由文本描述。进度头使用时间边界派生的监督。数据层级由已有原子标注、LLM 分组和人工核验构建。

论文评估在线视频定位与描述，不提供机器人控制迁移证据。“class-agnostic”不等于没有边界标注，“VLM 冻结”也不等于全系统没有训练。

## 对本项目的判断

这是机器人文献之外值得纳入的方向：当未来引入人类长视频时，Event 的困难包括连续操作之间几乎没有背景帧、一个动作属于多个层级，以及在线不能回看未来修改已输出边界。只讨论离线分段准确率会遗漏这些实际约束。

它提供了“何时发生变化”和“变化叫什么”的模块分工。对于我们的数据整理，可以先用便宜的时序模型给候选区间，再让 VLM 做语义确认；这只是可测试的工程选择，不能假定其边界会天然符合机器人低层可执行单元。

本项目的 Phase 1 仍可以完全使用离线 Event 标签。若 Phase 2 只需要从完整视频提取训练信号，没有必要为了引用本工作而强制引入在线 detector。真正需要在线模块时，应同时评估误切、漏切、结束延迟及描述一致性。

## 最小验证与证伪条件

先在小规模人工核验的人类/机器人配对片段上，比较等间隔采样、离线视觉变化、流式边界候选。分别评估边界与文本语义，不把描述流畅度当作分段正确性。

将人类视频按相机视角、手部遮挡、物体外观和动作速度分组，检查同一 atomic skill 的跨域一致性。再将提取片段用于同一 action representation 预训练，测机器人执行迁移。若只有描述指标提升而控制迁移不变，便应把它定位为数据标注辅助工具，而不是动作知识迁移的证据。

**后续核查**：人类边界监督量；边界延迟的实际分布；分层标注的嵌套一致性；与对象接触变化信号结合能否降低误切。

返回 [方向论文索引](../../event_organization/README.md)。
