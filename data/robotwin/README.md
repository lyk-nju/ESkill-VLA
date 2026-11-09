# RoboTwin 数据

clean 50 任务 × 50 条，共 **2,500 条原始轨迹**。已用固定十条接通统一 Episode、内容回读核对与视频渲染；全量转换和人工验收尚未完成。

| 内容 | 位置 |
| --- | --- |
| 原始数据 | `/data1/yuankai/robot-research/data/robotwin/clean_native/`；修复续采在 `supplement/`，保留独立来源身份 |
| 采集约定与来源 | [check/collection.md](check/collection.md) |
| 配额、来源版本与结果 | [check/summary.json](check/summary.json) · [原语覆盖](check/primitive_coverage.csv) |
| 当前字段与缺项 | [check/field_check.md](check/field_check.md) |
| 统一 Episode 十条验收 | [说明](check/test/episode_v1/README.md) · [视频](check/test/video/) · [人工表](check/test/episode_v1/manual_review.csv) |
| 原采集十条验收 | [视频](processed/video/clean/) · [固定样本清单](processed/clean_manifest.json) · [原人工表](processed/clean_manual_review.csv) |
| 处理命令与接口 | [examples/robotwin/](../../examples/robotwin/README.md) |

统一 Episode 试点在 `/data1/yuankai/robot-research/data/robotwin/check/episode_v1/`，每条自包含图像与数值。动作采用 14 维绝对关节目标，当前观测配下一次独立观测的驱动目标。正式每臂 Event／技能和对象完整区域待补；源可见物体框仅作为证据，不能用于完整区域监督。

`config.yaml` 是历史 `event_data` 读取配置；当前 clean 批次通过 `check collection --source-root` 显式指定。
