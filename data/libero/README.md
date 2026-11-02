# LIBERO 数据

当前主来源为 [AtomicVLA](atomic-vla/README.md) 的事件标注与作者指定的基础 RGB、状态、动作；官方 HDF5 提供匹配场景状态。已完成 40 任务、1,693 条／273,465 帧的主相机逐帧区域保存，验收入口见 [55 条合并视频](atomic-vla/check/test/full_frame_v1/merged55/index.html)。

| 目录 | 用途 |
| --- | --- |
| [atomic-vla/](atomic-vla/README.md) | 当前数据、实例绑定依据、字段表与验收入口 |
| [libero-40/](libero-40/README.md) | 官方 Spatial／Object／Goal／LIBERO-10 的 HDF5 来源与场景对应 |
| [guidedvla/](guidedvla/README.md) | GuidedVLA 发布版的来源调研、字段与历史核验 |
| [libero-90/](libero-90/README.md) | 后续扩展备选 |

## 处理入口

| 入口 | 职责 |
| --- | --- |
| [prepare_data.py](../../scripts/prepare_data.py) | 将已有事件与区域导出为预保存标签 |
| [atomic_vla/batch.py](../../scripts/preprocess/libero/atomic_vla/batch.py) | 全量原生区域生成、保存回读与验收渲染 |
| [libero_render.py](../../scripts/libero_render.py) | 从已保存的 `episode.json` 渲染视频 |
| [check_raw.py](../../scripts/preprocess/libero/check_raw.py) | 原始 RLDS 字段抽样、渲染及技术核验 |

命令、环境和完整批处理流程见 [LIBERO 示例](../../examples/libero/README.md)，标签契约见 [data/README.md](../README.md)。共享模型补标 CLI 的 `--dataset libero` 仍对应 GuidedVLA 发布版。

## 共用人工核验渲染器

[render.py](../../scripts/preprocess/libero/render.py) 负责 `Episode/Event/Region` 展示与编码；[view.py](../../scripts/preprocess/libero/view.py) 适配来源及预保存字段；[render_review.py](../../scripts/preprocess/libero/render_review.py) 用于原版或已有补标结果的重渲。渲染只读取现有标签。

画布为 1920×1340，展示任务、主相机、当前 Event／技能、终点图、夹爪状态、动作和三维末端路径。蓝色为夹爪，绿色为当前对象，黄色为同一对象在当前 Event 的实际终点区域。三维图灰色为整段路径、绿色为本 Event 路径、白点为当前位置、黄圈为终点。

当前区域使用忽略外部遮挡的完整屏内几何。Event 当前区间为 `[start_frame, stop_frame)`，终点按显式 `goal_frame` 读取；目标区域始终查询本 Event 的同一操作对象。坐标、帧顺序和源图像保持原义；缺字段显示 `MISSING`，缺失点不插值，越界终点不截到上一帧。有框不自动表示人工通过。RLDS 的 10 fps 是展示帧率，不据此推定物理时间。
