# AtomicVLA 数据核验

当前结果为 40 任务、1,693 条／273,465 帧的主相机逐帧完整屏内区域。

| 内容 | 入口 | 用途 |
| --- | --- | --- |
| 当前人工验收 | [55 条合并视频与章节](test/full_frame_v1/merged55/index.html) | 覆盖全部 40 任务，20 分 13.7 秒 |
| 全量标签与检查 | [full_frame_v1](test/full_frame_v1/README.md) | 标签索引、保存回读、核验汇总与人工表 |
| Motion 几何语义 | [元数据与入口核验](test/motion_semantics_v1/README.md) | 1,693 条元数据回填；原数组不变，未知语义屏蔽 Motion |
| 全量相机与 Motion | [相机参数与八点标签](test/calibration_full_v1/README.md) | 全量保存与 Dataset 逐帧回读 |
| 原版字段与抽样 | [字段表](field_check.md) · [原版视频](raw/index.html) | 保留原数据核对依据 |
| 历史试验 | [归档索引](archive/README.md) | 旧模型补框、诊断、100 条／v2 和保存试点 |

当前生成完成，人工逐项结论以验收表为准；740 条原状态对齐警示保留。
