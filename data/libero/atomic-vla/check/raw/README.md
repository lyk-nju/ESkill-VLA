# 原始发布数据核验

状态：2026-10-31 已完成 10 段原版视频及技术检查；用户抽看后认可数据大体可用并选定为 LIBERO 主来源，逐项人工表仍待填写。

- 来源及版本见 [来源说明](../../README.md)；需求以 [docs/data.md](../../../../../docs/data.md) 为准。
- 固定候选：seed=42。LiLo-VLA／CycleVLA 各无放回随机选 2 个分片；AtomicVLA 按作者转换器顺序（10／Goal／Object／Spatial）各随机选 1 个分片。合计下载 790,026,033 字节，官方分片 SHA-256 均需匹配。
- 在上述有限候选中按任务文本排序抽 10 个不同任务，再各抽一条；不足 10 个任务时如实记录。LiLo-VLA 用主相机／状态／动作指纹排除背景擦除副本；CycleVLA 每个父轨迹最多一片段。记录候选范围，不声称全库均匀抽样。
- 视频展示完整发布记录的主相机 RGB、已有文本／事件或进度、夹爪状态、动作和独立 XYZ 曲线。终点预览只展示原图；没有对象区域则明确缺失，不生成框／mask，不补事件。
- 发布 RLDS 未携带逐帧时间戳／fps。本轮 10 fps 仅为展示帧率，一条记录的一帧对应一输出帧，不据此恢复物理时长。CycleVLA 原有重复尾帧保持原样。
- 预算：CPU 读取／渲染，最多 3 个并发渲染（每进程 2 个编码线程）；不运行仿真、视觉补标模型或策略训练；三套视频合计目标小于 300 MiB。
- 输出：本目录 manifest.json 固定来源及样本；video/ 保存 MP4；verification.json 记录 CRC、字段与完整视频解码检查；manual_review.csv 初始全部 pending。
- AtomicVLA 的跨容器 episode 关联先验证任务与帧数；不匹配就保留基础 RGB 并显示未关联，不按近似任务或默认 ID 兜底。
- 这里只检验字段与技术可读性，事件边界、目标对象和整体适用性由人工核验。

## 结果

[播放索引](index.html) · [视频目录](video/) · [字段自查](../field_check.md) · [人工核验表](manual_review.csv) · [技术检查](verification.json)

候选 80 条、33 个不同任务文本；从去重后的 80 个单元中抽取 10 条。完整发布记录共 1,575 帧；1280×1000、H.264、10 fps 展示，视频合计 7,197,738 字节。已核对官方分片 SHA256、TFRecord 两项 CRC32C、各序列长度／有限数值、全部原 RGB 解码以及输出视频的全部帧／PTS／尺寸。

10 条完整基础任务轨迹，候选 JSON 关联的任务文本、all_frames 均与源 RGB 对应；10 条的最后 end_frame 均为 N，若直接当图像下标则超过末索引 N−1；本轮渲染按闭区间显示，故末段终点预览标为不可用。作者是否采用右开边界尚待核实，这不是已确认的源标签错误。原标注保留，尚未转换端点或补 bbox。完整 JSON 的 1,693 条均以 all_frames 作为最后 end，另有 45 条存在相邻段不连续；这两项是静态 JSON 检查，不能替代对应 RGB 的全量核对。官方训练配置仍指向未发布的 output_report_test.json，本轮实际核对发布的 libero_lerobot.json。

| 视频 | 发布记录 | 帧数 | 任务／子任务 |
| --- | --- | --- | --- |
| [01](video/01_libero_spatial_no_noops_s007_r022_raw.mp4) | `libero_spatial_no_noops_s007_r022` | 97 | pick up the black bowl on the cookie box and place it on the plate |
| [02](video/02_libero_spatial_no_noops_s007_r006_raw.mp4) | `libero_spatial_no_noops_s007_r006` | 75 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| [03](video/03_libero_goal_no_noops_s000_r023_raw.mp4) | `libero_goal_no_noops_s000_r023` | 90 | put the bowl on the plate |
| [04](video/04_libero_spatial_no_noops_s007_r016_raw.mp4) | `libero_spatial_no_noops_s007_r016` | 110 | pick up the black bowl on the ramekin and place it on the plate |
| [05](video/05_libero_10_no_noops_s007_r004_raw.mp4) | `libero_10_no_noops_s007_r004` | 263 | turn on the stove and put the moka pot on it |
| [06](video/06_libero_goal_no_noops_s000_r001_raw.mp4) | `libero_goal_no_noops_s000_r001` | 155 | put the wine bottle on the rack |
| [07](video/07_libero_spatial_no_noops_s007_r001_raw.mp4) | `libero_spatial_no_noops_s007_r001` | 124 | pick up the black bowl next to the cookie box and place it on the plate |
| [08](video/08_libero_10_no_noops_s007_r001_raw.mp4) | `libero_10_no_noops_s007_r001` | 247 | put the yellow and white mug in the microwave and close it |
| [09](video/09_libero_spatial_no_noops_s007_r000_raw.mp4) | `libero_spatial_no_noops_s007_r000` | 156 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| [10](video/10_libero_10_no_noops_s007_r008_raw.mp4) | `libero_10_no_noops_s007_r008` | 258 | put both the cream cheese box and the butter in the basket |

蓝／绿／黄折线分别只是 state 前三维的 X／Y／Z 原数值；没有夹爪或物体框。AtomicVLA 底部色块表示源事件段，CycleVLA 黄色尾段表示源 stop 标志，LiLo-VLA 为单技能段，均不是新增视觉标注。

运行环境：`/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python`；从 ESkill-VLA 根目录执行：

```bash
python -m data.libero.scripts.check_raw prepare --source atomic-vla
python -m data.libero.scripts.check_raw render --source atomic-vla
python -m data.libero.scripts.check_raw verify --source atomic-vla
```

这些是本轮入口，已有 manifest／视频／人工表时不直接重跑。原数据位于 manifest.raw_root；候选分片的随机选择与官方大小／哈希另保存在该处 candidate_shards.json。技术通过不代表语义正确，manual_review.csv 的 10 条结论均为 pending。
