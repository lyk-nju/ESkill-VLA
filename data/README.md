# 数据集与标签

数据需求见 [docs/data.md](../docs/data.md)，离线处理在 [scripts/preprocess/](../scripts/preprocess/README.md)，训练读取在 [src/openpi/policies/](../src/openpi/policies/README.md)。本目录按数据来源保存配置、字段自查、标签依据和人工核验入口。

| 来源 | 用途 | 当前状态 |
| --- | --- | --- |
| [LIBERO／AtomicVLA](libero/atomic-vla/README.md) | LIBERO 主来源：事件标注＋基础 RGB、状态、动作 | 40 任务、1,693 条／273,465 帧，逐帧主相机区域已保存；[55 条验收视频](libero/atomic-vla/check/test/full_frame_v1/merged55/index.html)覆盖全部任务 |
| [官方 LIBERO-40](libero/libero-40/README.md) | 为 AtomicVLA 匹配原始 HDF5 场景状态，支持原生区域提取 | 全量对应结果见 [规则检查](libero/atomic-vla/check/test/all_rules_v1/README.md) |
| [RoboTwin](robotwin/README.md) | 原生双臂仿真数据 | clean 50×50 已采齐；语义 Event 整理及统一训练导出待完成 |
| [AgiBot World 2026](agibot_world/README.md) | 真实双臂数据候选 | 初步调研；PhysicalReasoning 已完成字段自查和十条渲染，人工核验待完成 |
| [GuidedVLA 发布版](libero/guidedvla/README.md) | 数据管线与字段参考 | 保留来源核对和十条原版核验 |
| [LIBERO-90](libero/libero-90/README.md) | 后续扩展备选 | 尚未构建 |

AtomicVLA 基础标签位于 `/data1/yuankai/robot-research/data/atomicdata/libero40/episodes/`，批次索引为同级 `index.json`。包含全量相机参数的训练入口为 `libero40/motion_2d/index.json`，复用基础标签与 RGB；见 [全量相机核验](libero/atomic-vla/check/test/calibration_full_v1/README.md)。数据处理与回读示例见 [examples/libero/](../examples/libero/README.md)。

正式 train／val 使用每个任务内的完整父轨迹 90%／10% 划分。LIBERO 的固定清单为 [split.json](libero/atomic-vla/split.json)：40 任务共 1,518 train／175 val，SFT 与事件格式共用；逐任务数量、跨格式核对和新训练配置见 [task_split_v1](libero/atomic-vla/check/test/task_split_v1/README.md)。已有全量 SFT 与短程检查保留原划分，不能作为新独立验证结果。

## 预保存格式（v2 草案）

2026-11-01：先统一预处理交付，再批量补标，最后**只回读已保存标签进行可视化**。这是可修订的保存接口；π0.5 的训练 batch、动作归一化、VAE latent 和 Motion 监督目标仍在适配层处理；当前 Motion 使用每臂八点二维接口，相机参数已覆盖全量 1,693 条。十条可视化样本见 [八点核验](libero/atomic-vla/check/test/motion_2d_8x4_v1/README.md)。

| 文件 | 保存内容 |
| --- | --- |
| `index.json` | 本批次每条轨迹的 `episode.json` 路径、帧数、Event 数；不将逐臂或多视角副本算成独立轨迹 |
| `episodes/<id>/episode.json` | 来源版本、任务、臂、时间／坐标语义、Event、数组引用、修订依据和质量记录 |
| `episodes/<id>/arrays.npz` | 原数值状态／动作、逐臂末端 XYZ、帧索引、bbox、压缩 mask 及各项有效性 |
| 原始 RGB 缓存 | 通过路径、数组键及 SHA256 引用已有 JPEG 字节；不从带框视频反解图像、不重复压缩 |

正式标签目录通过导出参数指定，大体积文件放对应数据集的 `/data1/.../processed/`；项目内保留入口和核验记录。试点在该来源 `check/test/`，视频在 `check/test/video/`。当前 [AtomicVLA 保存接口试点](libero/atomic-vla/check/archive/history_20261101/test/saved_format/README.md)复用已有框，不重新补标。

核心层级为 **Episode → 每臂 Event → 该 Event 的对象区域序列**；执行器区域单独按臂保存。物体跨 Event 复用时可以引用同一组数组，不必另建全局对象身份标签。实际实例名／部件及绑定依据保留在 `source_annotation` 中，用于追溯。

```text
Episode
  source, instructions, arm_ids, source_arm_mapping
  frame_count, timeline, timestamps_s?, alignment, nominal_fps?
  cameras, state, actions, effector_positions
  effector_regions[arm][camera]
  events[], cooperative_goal_groups[], quality

Event
  event_id, arm, start_frame, stop_frame, goal_frame
  description, skill, boundary_valid, skill_valid, object_present
  object_regions[camera], evidence, source_annotation
```

- **区间与终点分开**：当前帧归属 `[start_frame, stop_frame)`；`goal_frame` 显式指定真实记录。AtomicVLA 使用 `goal=stop−1`；允许其他来源以 `goal=stop` 指向额外终点观察。读写器不猜、不移动端点。同臂不重叠，跨臂允许并发。
- **goal 不重复标注或存图**：按当前 Event 自己的 `object_regions` 在 `goal_frame` 查询，即使此帧已经属于下一 Event，也不能切换到下一个对象。
- **区域**：bbox 为源 RGB 像素 `xyxy`，右／下边界不含；当前主线为 `full_in_frame`。mask 逐帧展平后用 `np.packbits(..., bitorder='big')` 保存；引用记录解码后的 `[N,H,W] bool`。bbox／mask 各自带有效性。完整对象出画时，已渲染的空 mask 有效，bbox 无效；未生成的 mask 则无效。遮挡不缩框。
- **时间与臂**：已核实时间用秒数；只有逐步对应时用 `source_steps`，时间戳／fps 保持 `null`，10 fps 仅在渲染调用里指定。单臂保留 `single`，双臂保留 `left/right`；模型槽位映射在训练适配时明确。
- **来源和质量**：原边界／技能、修订、自动或助手实例绑定、对齐警示分别保留。`valid` 表示记录可用，`manual_review` 表示人工验收，两者不混用；字段齐全不自动等于训练就绪。

保存与读取分别属于离线处理和运行时接口：

```python
from openpi.policies.eskill_dataset import load_episode
from scripts.preprocess.export import EpisodeWriter, write_index, validate_episode

writer = EpisodeWriter()
ref = writer.add("eef_xyz", xyz)  # metadata 中引用 ref
path = writer.save(output_dir, metadata)
write_index(index_path, episode_paths)

episode = load_episode(path)
validate_episode(episode)  # 导出／核验阶段检查，不在每个训练样本上重复全轨迹校验。
event = episode.event_at("single", frame)
image = episode.rgb("image", frame)
region = episode.region(event["object_regions"]["image"], frame)
goal = episode.goal(event["event_id"], "image")
```

[AtomicVLA export.py](../scripts/preprocess/libero/atomic_vla/export.py)负责把已有完整区域及最新实例标签变成此接口；[LIBERO view.from_saved](../scripts/preprocess/libero/view.py)只适配已保存字段，再调用原有 [render.py](../scripts/preprocess/libero/render.py)。RoboTwin 可复用读写层，其来源适配器尚未迁移。旧草案的 `end_frame` 不自动重命名：每个来源必须明确转换为 `stop_frame` 和 `goal_frame`。

AtomicVLA 当前已完成 40 任务、1,693 条／273,465 帧的逐帧原生区域保存与回读；55 条验收视频覆盖全部 40 任务。入口为 [full_frame_v1](libero/atomic-vla/check/test/full_frame_v1/index.html)，结果与来源哈希核验见 [verification.json](libero/atomic-vla/check/test/full_frame_v1/verification.json)。全量生成完成，用户验收 pending；740 条原状态对齐警示保留。

## 维护约定

- 每个新来源按 [docs/data.md](../docs/data.md) 和 [fields.py](../scripts/preprocess/fields.py) 做三列字段自查：需要什么／有没有／备注，缺项写明补充办法；抽十条独立轨迹渲染，再人工核验。
- 原版核验放对应来源的 `check/raw/`，新处理试点放 `check/test/`，视频放 `check/test/video/`。来源版本、样本身份、原始标签及修订依据保留。
- 官方声明、有限样本检查和人工结论分别记录；技术检查通过不自动填写人工验收。历史检查的范围与统计口径保持原义。
- 代码、当前说明、来源配置及少量核验摘要进入 Git；原始数据、全量标签、帧图、视频、权重和运行日志保存在本机／数据盘。运行时代码放 `src/openpi/`，离线处理放 `scripts/preprocess/`。

过时方案与整理前文档集中保存在 [历史快照](../examples/archive/README.md#数据调研文档)，当前入口不再逐轮追加过程记录。
