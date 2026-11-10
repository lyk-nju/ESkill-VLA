# 水平镜像核验视频

2026-11-10，从已保存的 AtomicVLA LIBERO40 Episode 选取十条完整轨迹，覆盖六类技能和四套任务。左侧原图，右侧镜像；含方向词的第 10 条两侧均为原图。符合条件时固定展示镜像以便人工比较，实际 Stage 1 以 50% 概率翻转。

主相机与腕部同步翻转；白点为当前 EEF，青色未来点=Open 命令、粉色=Close 命令，编号 1–8 对应 +5、10…40 个保存帧。绿色框为 Current 监督范围（含原留量）；下方黄色框为 Event 终点的 Future 范围（额外每边 8 px）。轨迹与夹爪命令各按自己的有效性显示，越过当前 Event 的查询不填充。

物理状态、技能、臂身份、夹爪命令不变；这是二维训练增强的标签检查，不是仿真世界镜像，也不是模型预测。数值标签先保存再回读；RGB 与目标画布复用训练侧的确定性空间变换。人工验收状态：**待查看**。

[视频目录](../video/mirror/) · [完整清单](mirror/manifest.json)

| 序号 | 视频 | 技能 | 帧数 | 对照图 |
| --- | --- | --- | --- | --- |
| 01 | [push the plate to the front of the stove](../video/mirror/01_libero_goal_no_noops_s002_r009.mp4) | Push | 142 | [图片](mirror/01_libero_goal_no_noops_s002_r009.jpg) |
| 02 | [turn on the stove](../video/mirror/02_libero_goal_no_noops_s002_r017.mp4) | Turn | 86 | [图片](mirror/02_libero_goal_no_noops_s002_r017.jpg) |
| 03 | [open the top drawer and put the bowl inside](../video/mirror/03_libero_goal_no_noops_s013_r025.mp4) | Open / Pick / Place | 196 | [图片](mirror/03_libero_goal_no_noops_s013_r025.jpg) |
| 04 | [put the yellow and white mug in the microwave and close it](../video/mirror/04_libero_10_no_noops_s000_r003.mp4) | Close / Pick / Place | 285 | [图片](mirror/04_libero_10_no_noops_s000_r003.jpg) |
| 05 | [pick up the milk and place it in the basket](../video/mirror/05_libero_object_no_noops_s024_r000.mp4) | Pick / Place | 146 | [图片](mirror/05_libero_object_no_noops_s024_r000.jpg) |
| 06 | [pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate](../video/mirror/06_libero_spatial_no_noops_s004_r014.mp4) | Pick / Place | 165 | [图片](mirror/06_libero_spatial_no_noops_s004_r014.jpg) |
| 07 | [put both the alphabet soup and the cream cheese box in the basket](../video/mirror/07_libero_10_no_noops_s010_r006.mp4) | Pick / Place | 219 | [图片](mirror/07_libero_10_no_noops_s010_r006.jpg) |
| 08 | [pick up the butter and place it in the basket](../video/mirror/08_libero_object_no_noops_s000_r012.mp4) | Pick / Place | 152 | [图片](mirror/08_libero_object_no_noops_s000_r012.jpg) |
| 09 | [pick up the black bowl next to the cookie box and place it on the plate](../video/mirror/09_libero_spatial_no_noops_s013_r022.mp4) | Pick / Place | 130 | [图片](mirror/09_libero_spatial_no_noops_s013_r022.jpg) |
| 10 | [仅原图：put the white mug on the plate and put the chocolate pudding to the right of the plate](../video/mirror/10_libero_10_no_noops_s000_r011.mp4) | Pick / Place | 218 | [图片](mirror/10_libero_10_no_noops_s000_r011.jpg) |

合计 1,739 帧，1920×1340、10 fps，视频均已完整解码核对帧数、尺寸与播放时间。播放速度不代表数据的精确物理时钟。

当前 40 个任务中，以下含方向词的任务不做镜像：

- `put the white mug on the left plate and put the yellow and white mug on the right plate`
- `put the white mug on the plate and put the chocolate pudding to the right of the plate`

生成入口（只需 CPU，输出新目录）：

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.review_mirror \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --output /data1/yuankai/robot-research/workspace/reports/atomicvla_training/mirror \
  --count 10
```

产物实际保存在数据盘，本目录及 `check/test/video/mirror` 为入口。测试覆盖镜像几何、无效标签、物理字段不变、两阶段梯度范围、分布式归约、恢复与交接；不替代视频人工验收或真实模型效果测评。

状态门控更新：当前 Stage 1 镜像行关闭 Motion 的八维状态投影（含 bias），二维起点保持可用。上述已有视频展示图像／目标／路径几何，未显示该门控；`review_mirror.py` 后续生成的标签包含 `arm_state_mask`，画面显示 `BODY STATE: ON / MASKED`，不改写旧视频。
