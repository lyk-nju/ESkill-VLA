# 原语事件时间轴视频

六任务各一条 random；原生区域＋候选事件，人工验收待完成。

[集中播放](index.html) · [人工核验表](manual_review.csv)

| 任务 | seed | 视频 | 左臂 | 右臂 | 状态 |
| --- | --- | --- | --- | --- | --- |
| beat_block_hammer | 0 | [播放](../video/beat_block_hammer/random/beat_block_hammer_seed_0_repeat_001_random_primitive_events.mp4) | Pick > ToolContact | Idle | success |
| click_bell | 0 | [播放](../video/click_bell/random/click_bell_seed_0_repeat_001_random_primitive_events.mp4) | Press | Idle | success |
| move_playingcard_away | 0 | [播放](../video/move_playingcard_away/random/move_playingcard_away_seed_0_repeat_001_random_primitive_events.mp4) | Pick > Reposition > Release | Idle | success |
| lift_pot | 1 | [播放](../video/lift_pot/random/lift_pot_seed_1_repeat_001_random_primitive_events.mp4) | Pick | Pick | success |
| place_burger_fries | 0 | [播放](../video/place_burger_fries/random/place_burger_fries_seed_0_repeat_001_random_primitive_events.mp4) | Pick > Place > Return > Idle | Pick > Hold > Place > Return | success |
| place_can_basket | 0 | [播放](../video/place_can_basket/random/place_can_basket_seed_0_repeat_001_random_primitive_events.mp4) | Idle > Pick | Pick > Place > Return > Idle | success |

仅显示已标注画面。蓝／黄＝左右夹爪；粗虚线＝屏内完整框，细实线＝可见框；绿／紫＝任务物体／参照物。下方左右两条时间轴显示候选技能和当前游标。

候选来自原语映射／合并，Hold 另核验逐帧接触、夹爪命令与相对运动；Unknown 保持不确定。相机画面按标称 15 fps 核验播放，时间轴和时间读数使用真实仿真时间。

原语和 goal 帧映射保留在候选记录；这组是预览格式，正式 EventData 仍需显式转换与语义核验。任务别名 beat_hammer_block → beat_block_hammer；moving_playingcard_away → move_playingcard_away。

[协议、原始记录与候选标注](../../../../../workspace/experiment/robotwin_primitive_event_preview/README.md)
