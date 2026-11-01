# 八任务 randomized 视频核验

8 任务 × seed 0、1；原生专家轨迹，人工验收待完成。

[浏览全部视频](index.html) · [人工核验表](manual_review.csv)

| 任务 | seed | MP4 | 帧数 | 技术状态 | 原生任务 |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | 0 | 无（初始化／运行异常） | 0 | exception | 未完成 |
| adjust_bottle | 1 | 无（初始化／运行异常） | 0 | exception | 未完成 |
| beat_block_hammer | 0 | [播放](../video/randomized/beat_block_hammer_seed_0_repeat_001_randomized_native_ids.mp4) | 116 | success | 成功 |
| beat_block_hammer | 1 | [播放](../video/randomized/beat_block_hammer_seed_1_repeat_001_randomized_native_ids.mp4) | 115 | success | 成功 |
| click_bell | 0 | [播放](../video/randomized/click_bell_seed_0_repeat_001_randomized_native_ids.mp4) | 78 | success | 成功 |
| click_bell | 1 | [播放](../video/randomized/click_bell_seed_1_repeat_001_randomized_native_ids.mp4) | 77 | success | 成功 |
| dump_bin_bigbin | 0 | [播放](../video/randomized/dump_bin_bigbin_seed_0_repeat_001_randomized_native_ids.mp4) | 169 | success | 成功 |
| dump_bin_bigbin | 1 | [播放](../video/randomized/dump_bin_bigbin_seed_1_repeat_001_randomized_native_ids.mp4) | 167 | success | 成功 |
| move_playingcard_away | 0 | [播放](../video/randomized/move_playingcard_away_seed_0_repeat_001_randomized_native_ids.mp4) | 116 | success | 成功 |
| move_playingcard_away | 1 | [播放](../video/randomized/move_playingcard_away_seed_1_repeat_001_randomized_native_ids.mp4) | 118 | success | 成功 |
| lift_pot | 0 | [播放](../video/randomized/lift_pot_seed_0_repeat_001_randomized_native_ids.mp4) | 82 | partial | 失败 |
| lift_pot | 1 | [播放](../video/randomized/lift_pot_seed_1_repeat_001_randomized_native_ids.mp4) | 112 | success | 成功 |
| place_burger_fries | 0 | [播放](../video/randomized/place_burger_fries_seed_0_repeat_001_randomized_native_ids.mp4) | 242 | success | 成功 |
| place_burger_fries | 1 | [播放](../video/randomized/place_burger_fries_seed_1_repeat_001_randomized_native_ids.mp4) | 245 | success | 成功 |
| place_can_basket | 0 | [播放](../video/randomized/place_can_basket_seed_0_repeat_001_randomized_native_ids.mp4) | 247 | success | 成功 |
| place_can_basket | 1 | [播放](../video/randomized/place_can_basket_seed_1_repeat_001_randomized_native_ids.mp4) | 249 | success | 成功 |

蓝＝左夹爪；黄＝右夹爪；绿＝任务物体（名称区分）；紫＝参考物。三个相机各一行，原 RGB／mask＋bbox／crop。

EMPTY 表示当前不可见。15 fps 为核验播放，真实仿真时间与物理步号见画面。任务失败、场景异常、区域技术失败和人工验收分别记录；未替换失败 seed。

区域只是原生实例证据，当前对象归属、Event、技能、并发与 Hold 仍待 annotation。倒桶各内容物独立保存；右侧仅展示夹爪与主要对象，其余 crop 见无损抽帧。

本地入口映射：用户的 beat_hammer_block → beat_block_hammer；moving_playingcard_away → move_playingcard_away。部分轨迹的原生任务成功由 check_success 判定，仍待人工核验实际行为。

[完整协议与源证据](../../../../../workspace/experiment/robotwin_randomized_native_regions/README.md)
