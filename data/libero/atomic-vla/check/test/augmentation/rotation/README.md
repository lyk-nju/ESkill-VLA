# 同步旋转人工核验

2026-11-13；十条完整train轨迹，另有32张独立随机角度预览。人工验收待用户查看。

视频从左到右：A0原图、A2颜色＋裁切、A3再加旋转。同seed下A2与A3的裁切和颜色参数完全一致。正角度为图像坐标顺时针，旋转中心(0.5,0.5)。绿色框表示Current监督区域外接框，青色编号点表示未来EEF真值；VAE画布同步变换、保留原色，腕图不做几何变换。

32张预览中，前16张优先选择轨迹点最靠近边缘的帧，后16张随机抽样；每张独立抽样角度。预览绿线绘制监督权重区域轮廓，便于核对旋转方向。

| 序号 | 轨迹 | 固定角度 | 视频 |
| --- | --- | --- | --- |
| 01 | libero_goal_no_noops_s013_r002 | +0.335° | [视频](../../video/augmentation_rotation/01_libero_goal_no_noops_s013_r002.mp4) |
| 02 | libero_goal_no_noops_s002_r017 | -4.392° | [视频](../../video/augmentation_rotation/02_libero_goal_no_noops_s002_r017.mp4) |
| 03 | libero_goal_no_noops_s013_r025 | -2.627° | [视频](../../video/augmentation_rotation/03_libero_goal_no_noops_s013_r025.mp4) |
| 04 | libero_10_no_noops_s000_r003 | -0.892° | [视频](../../video/augmentation_rotation/04_libero_10_no_noops_s000_r003.mp4) |
| 05 | libero_object_no_noops_s024_r000 | -2.342° | [视频](../../video/augmentation_rotation/05_libero_object_no_noops_s024_r000.mp4) |
| 06 | libero_spatial_no_noops_s004_r014 | +2.694° | [视频](../../video/augmentation_rotation/06_libero_spatial_no_noops_s004_r014.mp4) |
| 07 | libero_10_no_noops_s010_r006 | -1.266° | [视频](../../video/augmentation_rotation/07_libero_10_no_noops_s010_r006.mp4) |
| 08 | libero_object_no_noops_s000_r012 | +4.987° | [视频](../../video/augmentation_rotation/08_libero_object_no_noops_s000_r012.mp4) |
| 09 | libero_spatial_no_noops_s013_r022 | +0.049° | [视频](../../video/augmentation_rotation/09_libero_spatial_no_noops_s013_r022.mp4) |
| 10 | libero_10_no_noops_s000_r011 | +4.966° | [视频](../../video/augmentation_rotation/10_libero_10_no_noops_s000_r011.mp4) |

| 监督保留率 | A2 | A3 |
| --- | --- | --- |
| 有效轨迹点 | 100.000% | 100.000% |
| 有效对象角色 | 100.000% | 100.000% |
| 校正后的前景面积估计 | 99.936% | 99.905% |

十条共1,744帧，1920×1340／10FPS，约2.91分钟；覆盖六类技能。统计仅针对这批固定变换样本，不能代表全量训练分布。

| 样本 | 选择方式 | 角度 | 预览 |
| --- | --- | --- | --- |
| 01 | near_edge | -2.879° | [图片](artifacts/samples/01.jpg) |
| 02 | near_edge | +3.015° | [图片](artifacts/samples/02.jpg) |
| 03 | near_edge | +1.459° | [图片](artifacts/samples/03.jpg) |
| 04 | near_edge | -0.866° | [图片](artifacts/samples/04.jpg) |
| 05 | near_edge | -3.445° | [图片](artifacts/samples/05.jpg) |
| 06 | near_edge | -4.362° | [图片](artifacts/samples/06.jpg) |
| 07 | near_edge | +0.656° | [图片](artifacts/samples/07.jpg) |
| 08 | near_edge | -2.661° | [图片](artifacts/samples/08.jpg) |
| 09 | near_edge | -4.353° | [图片](artifacts/samples/09.jpg) |
| 10 | near_edge | +3.942° | [图片](artifacts/samples/10.jpg) |
| 11 | near_edge | +3.045° | [图片](artifacts/samples/11.jpg) |
| 12 | near_edge | -3.113° | [图片](artifacts/samples/12.jpg) |
| 13 | near_edge | -0.120° | [图片](artifacts/samples/13.jpg) |
| 14 | near_edge | +2.053° | [图片](artifacts/samples/14.jpg) |
| 15 | near_edge | -4.533° | [图片](artifacts/samples/15.jpg) |
| 16 | near_edge | +3.718° | [图片](artifacts/samples/16.jpg) |
| 17 | random | +1.017° | [图片](artifacts/samples/17.jpg) |
| 18 | random | -0.403° | [图片](artifacts/samples/18.jpg) |
| 19 | random | +1.448° | [图片](artifacts/samples/19.jpg) |
| 20 | random | +4.298° | [图片](artifacts/samples/20.jpg) |
| 21 | random | +3.893° | [图片](artifacts/samples/21.jpg) |
| 22 | random | +2.498° | [图片](artifacts/samples/22.jpg) |
| 23 | random | +1.940° | [图片](artifacts/samples/23.jpg) |
| 24 | random | +1.759° | [图片](artifacts/samples/24.jpg) |
| 25 | random | -4.289° | [图片](artifacts/samples/25.jpg) |
| 26 | random | -3.923° | [图片](artifacts/samples/26.jpg) |
| 27 | random | -3.247° | [图片](artifacts/samples/27.jpg) |
| 28 | random | -4.524° | [图片](artifacts/samples/28.jpg) |
| 29 | random | +1.877° | [图片](artifacts/samples/29.jpg) |
| 30 | random | -4.086° | [图片](artifacts/samples/30.jpg) |
| 31 | random | -1.536° | [图片](artifacts/samples/31.jpg) |
| 32 | random | -4.996° | [图片](artifacts/samples/32.jpg) |

74项CPU回归通过，包括224／256方向一致、正负90°、±5°边界、黑角保持、裁切视野约束、单次插值、同seed裁切／颜色一致、增强后精确恢复及DDP。十条视频和32张预览的保存标签均已独立按NumPy公式回读核对；A2参数和标签与之前核验逐值相同。

标签、角度、随机种子和源数据记录见 [artifacts](artifacts)。未启动正式训练或闭环效果测评。
