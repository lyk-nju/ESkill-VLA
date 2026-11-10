# Motion 最短监督窗口：10 个保存帧

2026-11-10，固定 +5、10…40 帧八点不变，最短监督窗口设为 10 个保存主相机帧（按名义 20 Hz 约 0.5 秒）。查询仍在当前 Event 内，或偏移不超过 10 帧时，可用真实记录监督；前两点可跨 Event，后六点不放宽。Episode 尾部、缺失记录、出画及当前起点无效仍按原规则排除，不重复终点。

当前 Skill 与 Future Object 的 Event 终点保持原定义。跨界点记录 `future_cross_event`，实际监督使用下一段真实 EEF／夹爪命令；不会将下一段技能标签改写成当前技能。

## 全量覆盖率

1,693 条、40 任务、273,465 个动作样本；各任务／轨迹与 train／val 划分、原数据均未修改。下表按原始动作行统计，不按技能加权抽样。

| 指标 | 严格 Event（0 帧） | 最短窗口（10 帧） |
| --- | ---: | ---: |
| 无有效二维点 | 21,110（7.72%） | 10,224（3.74%） |
| 无有效夹爪命令 | 19,365（7.08%） | 8,465（3.10%） |

补回 **10,886 条**原先无有效二维点的样本。+15…40 的有效数量与旧规则完全一致；新增跨界有效二维点分别为 +5：10,886，+10：21,750。

剩余零二维点样本按“当前起点无效→没有可用未来窗口→未来记录无效”的优先次序归因，分类互斥：

| 原因 | 样本数 |
| --- | ---: |
| 允许窗口内未来投影均无效 | 248 |
| 当前起点投影无效 | 1,753 |
| Episode 尾部无未来查询 | 8,223 |

Episode 最后五帧合计 8,465 条没有 +5 未来查询，其中与当前起点无效重叠的部分归在“当前起点”类。最短窗口只消除 Event 边界截断，不能保证每条都有 0.5 秒真值。

[旧规则完整统计](minimum_motion/before.json) · [新规则完整统计](minimum_motion/after.json)

## 十条核验视频

左侧原图、右侧镜像，含方向词时两侧保持原图。`X`／黄色外圈为最短窗口内的有效跨 Event 轨迹点；青色=Open 命令，粉色=Close 命令。当前／终点对象与状态门控仍显示在原位置。对照图优先选择存在有效跨界点的帧。

| 序号 | 视频 | 跨界有效二维点数 | 对照图 |
| --- | --- | ---: | --- |
| 01 | [push the plate to the front of the stove](../video/motion_min10f/01_libero_goal_no_noops_s002_r009.mp4) | 0 | [图片](minimum_motion/review/01_libero_goal_no_noops_s002_r009.jpg) |
| 02 | [turn on the stove](../video/motion_min10f/02_libero_goal_no_noops_s002_r017.mp4) | 0 | [图片](minimum_motion/review/02_libero_goal_no_noops_s002_r017.jpg) |
| 03 | [open the top drawer and put the bowl inside](../video/motion_min10f/03_libero_goal_no_noops_s013_r025.mp4) | 30 | [图片](minimum_motion/review/03_libero_goal_no_noops_s013_r025.jpg) |
| 04 | [put the yellow and white mug in the microwave and close it](../video/motion_min10f/04_libero_10_no_noops_s000_r003.mp4) | 30 | [图片](minimum_motion/review/04_libero_10_no_noops_s000_r003.jpg) |
| 05 | [pick up the milk and place it in the basket](../video/motion_min10f/05_libero_object_no_noops_s024_r000.mp4) | 15 | [图片](minimum_motion/review/05_libero_object_no_noops_s024_r000.jpg) |
| 06 | [pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate](../video/motion_min10f/06_libero_spatial_no_noops_s004_r014.mp4) | 15 | [图片](minimum_motion/review/06_libero_spatial_no_noops_s004_r014.jpg) |
| 07 | [put both the alphabet soup and the cream cheese box in the basket](../video/motion_min10f/07_libero_10_no_noops_s010_r006.mp4) | 45 | [图片](minimum_motion/review/07_libero_10_no_noops_s010_r006.jpg) |
| 08 | [pick up the butter and place it in the basket](../video/motion_min10f/08_libero_object_no_noops_s000_r012.mp4) | 15 | [图片](minimum_motion/review/08_libero_object_no_noops_s000_r012.jpg) |
| 09 | [pick up the black bowl next to the cookie box and place it on the plate](../video/motion_min10f/09_libero_spatial_no_noops_s013_r022.mp4) | 15 | [图片](minimum_motion/review/09_libero_spatial_no_noops_s013_r022.jpg) |
| 10 | [put the white mug on the plate and put the chocolate pudding to the right of the plate](../video/motion_min10f/10_libero_10_no_noops_s000_r011.mp4) | 45 | [图片](minimum_motion/review/10_libero_10_no_noops_s000_r011.jpg) |

十条共 1,739 帧，1920×1340、10 fps；每条已完整解码核对帧数与时间。视频播放速度不当作物理控制时钟。标签先保存再回读，实际产物位于数据盘；[完整清单](minimum_motion/review/manifest.json)。人工验收：**待用户查看**。

96 项相关 CPU 回归通过，包含越界真值与独立 mask、Episode 尾部、双臂各自边界、源索引跳号、严格 Event 对照、DDP 与训练恢复、阶段交接。最短窗口属于训练语义，历史缺失字段的 checkpoint 记为 0，禁止静默切换窗口后 resume；新配方仍从原生 π0.5 开始。未执行正式长训练或闭环效果评测。
