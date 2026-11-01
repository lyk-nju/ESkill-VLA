# 原生 mask／bbox／crop 核验

2026-10-30；两个任务各 seed 0、1，共 4 条新采专家轨迹，分别使用 GPU 0–3。技术检查通过，**人工验收待完成**。本轮是区域能力试点，不计入此前 10 条历史数据核验。

新增批次：[八任务 randomized 视频与核验入口](randomized/README.md)。与以下 clean 试点分开记录，不合并统计。

新增定义验证：[遮挡时仍保留完整夹爪框](occlusion/README.md)，两条仅含已标注画面的 MP4；忽略遮挡并裁掉屏外部分，与以下可见框记录分开。

新增联合预览：[原语 Event＋完整夹爪框＋左右臂时间轴](events/README.md)，六任务各一条 random，视频放 `video/<任务名>/random/`，候选标签人工待验收。

| 任务 | seed | MP4 | 源帧数 | 仿真时长 | 技术检查／原生任务 |
| --- | --- | --- | --- | --- | --- |
| place_object_stand | 0 | [播放](video/place_object_stand_seed_0_repeat_001_native_ids.mp4) | 138 | 7.232 s | 通过／成功 |
| place_object_stand | 1 | [播放](video/place_object_stand_seed_1_repeat_001_native_ids.mp4) | 138 | 7.188 s | 通过／成功 |
| handover_block | 0 | [播放](video/handover_block_seed_0_repeat_001_native_ids.mp4) | 287 | 15.140 s | 通过／成功 |
| handover_block | 1 | [播放](video/handover_block_seed_1_repeat_001_native_ids.mp4) | 279 | 14.696 s | 通过／成功 |

每个视频三行分别为主相机、左腕相机、右腕相机；每行从左到右为原 RGB、mask＋bbox、三个 RGB crop。**蓝色＝左夹爪，黄色＝右夹爪，绿色＝指定操作物体**。绿色在放置任务中是鼠标，在交接任务中是红色方块。夹爪取配置中的两片手指 link，不包含整条机械臂；具体名称与 ID 见 [records/](records/)。

框只包围当前可见像素；遮挡或出画时显示 `EMPTY`，bbox／crop 为 null。crop 是同帧矩形 RGB 裁剪，包含框内背景；无损 mask 另存。视频以标称 15 fps 顺序播放，源采样间隔不恒定，顶部同时给出实际仿真时间及物理步号。

842 帧、三视角共 7,578 个区域记录的独立像素核对无不一致；四个视频已完整解码，32 个抽样帧的无损 mask／crop 与源像素一致。左右夹爪及对象在每条轨迹至少一个视角可见，不要求每个视角始终可见。这里的“通过”只表示技术一致性，人工请检查对象／手指归属、交接中的身份、遮挡及裁剪是否合理，记录到 [manual_review.csv](manual_review.csv)。

在这两类任务上，原生相机链路可以生成区域；Event／技能、操作臂归属、并发与 Hold 仍需 annotation。尚未接入全量采集或训练导出，也未验证其它任务、关节物体及其它机器人本体。

完整协议、源 RGB／整数 ID、配置、日志及无损示例见 [实验入口](../../../../workspace/experiment/robotwin_native_visual_regions/README.md)。
