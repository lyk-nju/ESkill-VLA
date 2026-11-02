# 遮挡下完整夹爪框核验

视频只展示叠加后的画面及 crop，不包含原图对照栏。人工验收待完成。

| 任务 | 视频 | 帧数 | 原生任务 | 技术检查 |
| --- | --- | --- | --- | --- |
| handover_block | [播放](../video/occlusion/handover_block_seed_0_repeat_001_clean_full_gripper.mp4) | 289 | True | success |
| place_burger_fries | [播放](../video/occlusion/place_burger_fries_seed_0_repeat_001_randomized_full_gripper.mp4) | 241 | True | success |

蓝＝左夹爪，黄＝右夹爪；细实线＝实际可见框，粗虚线＝忽略遮挡的屏内完整框。绿／紫保留原先物体／参照物的可见框。

部分出画裁到画面边缘，完全出画不标框。crop 取真实 RGB，完整遮挡时仍可包含遮挡物；没有合成夹爪外观。百分比是可见像素／完整夹爪像素。

[人工核验表](manual_review.csv) · [协议与源记录](../../../../../workspace/experiment/robotwin_occluded_gripper_regions/README.md)
