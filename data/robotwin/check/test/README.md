# RoboTwin 定点人工复核

本轮用户明确要求提供刚讨论的堆三个碗视频。复用已采集的 clean 轨迹，只做离线渲染，不新采数据、运行仿真或新增模型标注，也不改写正式十条抽样清单。

- 来源：`/data1/yuankai/robot-research/data/robotwin/clean_native/stack_bowls_three/aloha_agilex/raw/episode_0000000.hdf5`，seed 1，443 源帧。
- 目的：核验三个相同外观碗的实例绑定，以及三个源 Place 结束时刻取的是否为同一操作对象。
- 呈现：沿用主相机和双腕 RGB、左右臂原语时间轴及真实仿真时间；主相机显示原生夹爪框和来源对象区域。底部追加三个源 Place 终点主相机画面，按各自对象实例 ID 高亮可见 mask／bbox，显示 bowl 名、ID、源结束行；当前 Place 的对应终点面板加亮。显示的是来源原语终点预览，正式 Event／goal 标签仍待后续装配。
- 时序：完整保留 443 帧、15 fps 标称核验播放；真实时间另显，不称为实时播放。
- 预算：CPU 离线，最多 2 分钟／200 MB，输出一条渲染后 MP4；不额外输出原始 MP4。
- 产物：`video/stack_bowls_three__episode_0000000__seed_1__clean__place_goals_review.mp4`，以及对应 JSON 报告与 `stack_bowls_three_manifest.json`。
- 技术核对：完整解码帧数与源帧数相同，终点实例 ID、源行、同刻时间和 bbox 回链源记录。人工结果保持 pending，由用户审核。

已生成 [堆三个碗复核视频](video/stack_bowls_three__episode_0000000__seed_1__clean__place_goals_review.mp4)：1280×1240、443 帧、约 29.53 秒、2.92 MB；渲染耗时约 72.46 秒，源文件 SHA 校验不变。三格分别为 bowl1／ID 68／源行 138、bowl2／ID 69／源行 271、bowl3／ID 70／源行 426，对应原生 bbox 均有效。完整解码及三终点数据核对通过；已查看第 250 帧，碗2操作对应中间终点面板高亮。人工验收仍待用户填写。
