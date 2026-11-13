# 单阶段增强：十条轨迹人工核验

2026-11-13；固定 train split，十条完整轨迹。技术核验完成，人工验收待用户查看。

新增旋转的 A0／A2／A3 视频和32张独立角度预览见[旋转核验](rotation/README.md)。下面保留原A0／A1／A2对照。

每条视频从左到右为 A0 原图、A1 颜色扰动、A2 颜色＋主图轻裁切。绿色框表示 Current 对象监督区域；青色编号点表示未来 EEF 真值。下方是 Current／Future VAE 目标和腕图。VAE目标保留原色，Action／Skill／Gripper命令不变。

每条轨迹固定一组随机变换（seed43～52）以保持视频稳定；训练时每个采样行重新抽样。A2按0.5概率裁切，部分视频窗口为[0,0,1,1]。95%指保留边长，不是面积。

| 序号 | 轨迹／技能 | 视频 |
| --- | --- | --- |
| 01 | libero_goal_no_noops_s013_r002；Push | [视频](../video/augmentation/01_libero_goal_no_noops_s013_r002.mp4) · [预览](artifacts/01_libero_goal_no_noops_s013_r002.jpg) |
| 02 | libero_goal_no_noops_s002_r017；Turn | [视频](../video/augmentation/02_libero_goal_no_noops_s002_r017.mp4) · [预览](artifacts/02_libero_goal_no_noops_s002_r017.jpg) |
| 03 | libero_goal_no_noops_s013_r025；Open／Pick／Place | [视频](../video/augmentation/03_libero_goal_no_noops_s013_r025.mp4) · [预览](artifacts/03_libero_goal_no_noops_s013_r025.jpg) |
| 04 | libero_10_no_noops_s000_r003；Close／Pick／Place | [视频](../video/augmentation/04_libero_10_no_noops_s000_r003.mp4) · [预览](artifacts/04_libero_10_no_noops_s000_r003.jpg) |
| 05 | libero_object_no_noops_s024_r000；Pick／Place | [视频](../video/augmentation/05_libero_object_no_noops_s024_r000.mp4) · [预览](artifacts/05_libero_object_no_noops_s024_r000.jpg) |
| 06 | libero_spatial_no_noops_s004_r014；Pick／Place | [视频](../video/augmentation/06_libero_spatial_no_noops_s004_r014.mp4) · [预览](artifacts/06_libero_spatial_no_noops_s004_r014.jpg) |
| 07 | libero_10_no_noops_s010_r006；Pick／Place | [视频](../video/augmentation/07_libero_10_no_noops_s010_r006.mp4) · [预览](artifacts/07_libero_10_no_noops_s010_r006.jpg) |
| 08 | libero_object_no_noops_s000_r012；Pick／Place | [视频](../video/augmentation/08_libero_object_no_noops_s000_r012.mp4) · [预览](artifacts/08_libero_object_no_noops_s000_r012.jpg) |
| 09 | libero_spatial_no_noops_s013_r022；Pick／Place | [视频](../video/augmentation/09_libero_spatial_no_noops_s013_r022.mp4) · [预览](artifacts/09_libero_spatial_no_noops_s013_r022.jpg) |
| 10 | libero_10_no_noops_s000_r011；Pick／Place | [视频](../video/augmentation/10_libero_10_no_noops_s000_r011.mp4) · [预览](artifacts/10_libero_10_no_noops_s000_r011.jpg) |

| 汇总 | A1 | A2 |
| --- | --- | --- |
| 有效轨迹点 | 9,866/9,866（100.000%） | 9,866/9,866（100.000%） |
| 有效对象角色 | 3,488/3,488（100.000%） | 3,488/3,488（100.000%） |
| 前景面积估计保留率 | 100.000% | 99.936% |

共1,744帧，10 FPS，约2.91分钟；1920×1340，视频总大小18.2 MiB。面积按裁切宽×高校正，仍受离散插值误差影响；以上只代表该十条固定变换样本，不是全量训练覆盖率。

保存标签回读已核对：A1坐标／有效性不变；A2符合窗口坐标变换与出画mask；夹爪命令／有效性不变。68项CPU回归通过，含多进程／DDP、增强后恢复一致性和旧配置默认关闭。未启动正式训练或闭环效果实验。

源标签、变换参数、预览和manifest保存在 [artifacts](artifacts)，实际文件位于数据盘。视频入口为 `check/test/video/augmentation`。

渲染代码：[review_augmentation.py](../../../../../../scripts/preprocess/libero/review_augmentation.py)；训练与配方见[直接联合训练对照](../../../../../../experiments/eskill_libero40_direct/README.md#视觉增强对照)。
