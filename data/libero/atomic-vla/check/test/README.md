# AtomicVLA 当前验收

Current／Subject 与在线增强的数据补齐结果见 [全量 Grounding 核验](full_grounding/README.md)：沿用全部 1,693 条轨迹，新增场景实体完整区域和可见 mask，原数据保持不变。

直接查看 [55 条合并验收视频与章节](full_frame_v1/merged55/index.html)，或打开 [全量标签与核验汇总](full_frame_v1/README.md)。

当前生成覆盖 40 任务、1,693 条／273,465 帧，人工结论填写 [manual_review.csv](full_frame_v1/manual_review.csv)。

| 保留目录 | 用途 |
| --- | --- |
| `full_grounding/` | 全场景框／mask、在线增强字段与训练输入核验 |
| `full_frame_v1/` | 当前全量标签、汇总、章节与人工表 |
| `video/full_frame_v1/` | 当前 55 条单视频和合并视频 |
| `all_rules_v1/` | 当前批处理依赖的来源清单、状态与规则核对依据 |
| `manual_binding12/` | 合并实例标签、34 条助手补充及视觉依据 |
| `annotation_review/` | 事件边界修订与用户确认的 Push 技能依据 |
| `review100_v1/` | 官方 LIBERO-40 对照仍需的两份固定来源清单 |

本目录只保留当前验收与运行依赖；原始抽样在 [check/raw](../raw/README.md)。

740 条状态对齐警示仍保留，生成完成与人工质量验收分别记录。
