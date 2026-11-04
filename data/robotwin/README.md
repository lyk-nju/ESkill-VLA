# RoboTwin 数据

clean 50 任务 × 50 条，共 **2,500 条**。十条验收视频已生成，逐项人工结论仍以验收表为准。

| 内容 | 位置 |
| --- | --- |
| 原始数据 | `/data1/yuankai/robot-research/data/robotwin/clean_native/`；修复续采在 `supplement/`，保留独立来源身份 |
| 采集约定与来源 | [check/collection.md](check/collection.md) |
| 配额、来源版本与结果 | [check/summary.json](check/summary.json) · [原语覆盖](check/primitive_coverage.csv) |
| 当前字段与缺项 | [check/field_check.md](check/field_check.md) |
| 十条验收 | [视频](processed/video/clean/) · [样本清单](processed/clean_manifest.json) · [人工表](processed/clean_manual_review.csv) |
| 处理命令与接口 | [examples/robotwin/](../../examples/robotwin/README.md) |

当前保存原生实例区域；夹爪有完整屏内框，物体目前仍是可见区域。正式每臂 Event／技能、对象完整区域与公共 JSON＋NPZ 训练导出尚待整理。

`config.yaml` 是历史 `event_data` 读取配置；当前 clean 批次通过 `check collection --source-root` 显式指定。
