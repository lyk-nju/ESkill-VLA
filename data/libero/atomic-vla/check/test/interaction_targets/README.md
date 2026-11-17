# 交互目标与全场景 grounding 核验

2026-11-17：全量 **1,693 条／40 任务／3,873 个 Event** 已生成唯一目标候选，尚待人工确认。旧数据中 2,073 个 Event 可沿用已有目标区域；1,800 个 Place Event 的目标变为目的地，需要重新补框。

本轮完成 **10 条独立轨迹、27 个 Event、2,198 帧**的逐帧补标，覆盖 Pick／Place／Open／Close／Turn／Push。共 13,456 组 bbox＋mask（含部件与空间区域）；2,198 帧的当前目标框均有效。原始数据未改写；训练接口及 Future 分支尚未迁移。

**[完整合并视频：3 分 39.8 秒](../video/interaction_targets/all_10.mp4)** · [视频目录](../video/interaction_targets/) · [全量 Event 清单](artifacts/events.csv) · [统计](artifacts/summary.json) · [派生 Episode](artifacts/episodes/)

| 需要什么 | 有没有 | 备注 |
| --- | --- | --- |
| 当前 Event 交互目标 | 有候选 | Pick 指操作对象；Place 指目的地；部件／空间区域保留独立身份 |
| 场景其他实例及区域 | 十条已补齐 | 全部 BDDL 物体与有限固定装置；桌面、地面、墙面、背景装饰不计入对象列表 |
| 遮挡时保持完整框 | 有 | 实体隔离渲染；完整屏内区域，不因外部遮挡缩框 |
| 红／绿框视频 | 有 | 右图：目标红，其余物体绿；左图：源 RGB；目标部件的父物体整框不重复绘制 |
| 人工验收 | 待确认 | 唯一绑定与字段完整不代表语义或图像对齐已人工通过 |

| 编号 | 样本与关键切换 | 视频 |
| --- | --- | --- |
| 01 | Milk → Basket | [播放](../video/interaction_targets/01_libero_object_no_noops_s000_r007.mp4) |
| 02 | Black bowl → Plate；同类干扰物 | [播放](../video/interaction_targets/02_libero_spatial_no_noops_s000_r000.mp4) |
| 03 | Top drawer → Bowl → Top drawer | [播放](../video/interaction_targets/03_libero_goal_no_noops_s000_r003.mp4) |
| 04 | Bowl → Bottom drawer；放置后关闭 | [播放](../video/interaction_targets/04_libero_10_no_noops_s001_r002.mp4) |
| 05 | Stove knob → Moka pot → Stove | [播放](../video/interaction_targets/05_libero_10_no_noops_s000_r006.mp4) |
| 06 | Mug → Microwave → Microwave door | [播放](../video/interaction_targets/06_libero_10_no_noops_s000_r002.mp4) |
| 07 | Book → Caddy 后格 | [播放](../video/interaction_targets/07_libero_10_no_noops_s002_r003.mp4) |
| 08 | 两个 Mug 分别指向左／右 Plate | [播放](../video/interaction_targets/08_libero_10_no_noops_s000_r000.mp4) |
| 09 | Mug → Plate；Pudding → 盘子右侧区域 | [播放](../video/interaction_targets/09_libero_10_no_noops_s000_r001.mp4) |
| 10 | Push：Plate 保持为目标 | [播放](../video/interaction_targets/10_libero_goal_no_noops_s000_r007.mp4) |

07／09 的红框来自仿真原生空间区域的投影，不是整物体分割；09 的“右侧”沿用任务／场景坐标语义，不按当前图像左右重新命名。Push 首版仍监督被推动物体。以上三种语义请重点核验。

场景状态仍沿用原先的动作匹配结果，源质量警示保留；抽查帧的重渲染 RGB 误差保存在各 Episode 的 `interaction_annotation.alignment`。已逐帧核对 mask 与 bbox，并解码核验视频帧数；规则及既有绑定回归 40 项通过。

代码入口和运行方式见 [预处理说明](../../../../../../scripts/preprocess/README.md#交互目标与全场景区域核验)。本目录 `artifacts` 和视频入口指向数据盘；这些派生 Episode 复用原始 RGB／动作文件，暂非独立训练数据包。人工确认后再安排全量逐帧导出。
