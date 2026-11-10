# 每 4 帧 Motion 标签：十条人工核验视频

本页保留此前每 4 帧的渲染结果；当前训练已改为每 5 帧，这些视频不能代替新间隔的标签验收。

每条视频保留全部源帧，10 fps 仅为播放速度。下方显示当前 Event 内八个未来查询；真值监督标签，不是模型预测。人工验收待完成。

| 序号 | 技能 | 视频 |
| --- | --- | --- |
| 01 | Push | [libero_goal_no_noops_s002_r009](../video/motion_stride4/01_libero_goal_no_noops_s002_r009.mp4) |
| 02 | Turn | [libero_goal_no_noops_s002_r017](../video/motion_stride4/02_libero_goal_no_noops_s002_r017.mp4) |
| 03 | Open / Pick / Place | [libero_goal_no_noops_s013_r025](../video/motion_stride4/03_libero_goal_no_noops_s013_r025.mp4) |
| 04 | Close / Pick / Place | [libero_10_no_noops_s000_r003](../video/motion_stride4/04_libero_10_no_noops_s000_r003.mp4) |
| 05 | Pick / Place | [libero_object_no_noops_s024_r000](../video/motion_stride4/05_libero_object_no_noops_s024_r000.mp4) |
| 06 | Pick / Place | [libero_spatial_no_noops_s004_r014](../video/motion_stride4/06_libero_spatial_no_noops_s004_r014.mp4) |
| 07 | Pick / Place | [libero_10_no_noops_s010_r006](../video/motion_stride4/07_libero_10_no_noops_s010_r006.mp4) |
| 08 | Pick / Place | [libero_object_no_noops_s000_r012](../video/motion_stride4/08_libero_object_no_noops_s000_r012.mp4) |
| 09 | Pick / Place | [libero_spatial_no_noops_s013_r022](../video/motion_stride4/09_libero_spatial_no_noops_s013_r022.mp4) |
| 10 | Pick / Place | [libero_10_no_noops_s000_r011](../video/motion_stride4/10_libero_10_no_noops_s000_r011.mp4) |

白点：当前起点；青色点：打开命令；粉色点：闭合命令。上方蓝／绿／黄框分别为夹爪、当前物体和 Event 终点物体。

后续真实训练短测尚未启动；当前四张 GPU 被已有进程占用约 19.6 GB／张，本次未停止这些进程。
