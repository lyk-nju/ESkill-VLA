# 数据集与数据链路调研

更新于 2026-11-01。当前工作是了解来源数据能提供什么、标注质量如何，以及可能怎样支持 Event、Atomic Skill、Target Grounding、Latent Motion 与 Action 学习。

**状态：Stage 1 统一使用 π0.5；LIBERO 已选定 [AtomicVLA 事件标注＋基础数据](libero/atomic-vla/README.md)，主要补主相机夹爪／物体 bbox；RoboTwin 使用原生实例区域。** 具体训练任务、规模及预算待确定。参考 GuidedVLA 的数据构建与训练方法，其复现暂不开展。两条链路均执行字段自查、10 条抽样渲染和最终人工验收。当前数据需求见 [Data](../docs/data.md)，模型接口见 [Design](../docs/design.md)。本目录保留来源证据与接入缺口；以下旧调研记录不覆盖当前来源选择。

各数据集的调研说明、字段核对、统计清单及产物放入各自目录；代码已移到 [scripts/preprocess/](../scripts/preprocess/README.md) 与 [src/openpi/policies/](../src/openpi/policies/README.md)，当前运行示例见 [examples/](../examples/libero/README.md)。整理规则见 [AGENTS.md](AGENTS.md)。根目录保留公共索引和跨数据集讨论。

阅读时区分三类内容：官方资料的声明、本地有限样本／代码的检查结果、尚待讨论的设计建议。局部检查保留具体范围，不直接外推整个数据集；候选方案可以继续比较、替换或放弃。

后续每个候选统一执行：**按 [docs/data.md](../docs/data.md) 及 [fields.py](../scripts/preprocess/fields.py) 逐项自查 → 标明已有／缺失／待核实字段和补充途径 → 抽取 10 条独立轨迹 → 在对应数据集的 `check/` 输出核验视频 → 人工检查后记录适用性。** `docs/data.md` 说明模块需要什么，`fields.py` 定义示范数据字段及对应预处理清单。运行 `python -m scripts.preprocess.fields` 可打印三列自查空表。具体记录要求见 [AGENTS.md](AGENTS.md)。

| 入口 | 内容 |
| --- | --- |
| [Data](../docs/data.md) | 各模块需要的外部数据 |
| [fields.py](../scripts/preprocess/fields.py) | 来源无关的预保存字段接口 v2 草案、11 项需求与预处理清单 |
| [eskill_dataset.py](../src/openpi/policies/eskill_dataset.py) | JSON＋NPZ 回读与 Event／区域查询；不运行仿真、模型或实例绑定规则 |
| [scripts/preprocess/export.py](../scripts/preprocess/export.py) | 保存、导出前结构校验与批次索引 |
| [CHECK_TEMPLATE.md](CHECK_TEMPLATE.md) | 各候选统一的字段自查、10 条渲染与人工核验记录模板 |
| [source.md](source.md) | 外部数据来源、公开字段和接入限制 |
| [agibot_world/](agibot_world/README.md) | AgiBot World 2026 专项调研；四主题初查、第四期 18 任务／29 分片续查及 Dense Prompt 状态 |
| [PhysicalReasoning 核验](agibot_world/check/README.md) | 已完成 11 项字段自查和 10 条视频渲染、技术检查；待人工核验 |
| [LIBERO 当前主来源：AtomicVLA](libero/atomic-vla/README.md) | 用户已选定；逐帧区域已保存，进入人工验收；[原版视频](libero/atomic-vla/check/raw/index.html) |
| [LIBERO／GuidedVLA 发布版](libero/guidedvla/README.md) | 已核对官方代码、元数据及样本；字段初查、[10 条原始视频](libero/guidedvla/check/raw/README.md)渲染与技术检查完成，待人工验收 |
| [LIBERO-90 自建链路](libero/libero-90/README.md) | 后续扩展备选；当前优先接入 AtomicVLA |
| [其他 LIBERO 发布来源](libero/README.md) | LiLo-VLA／AtomicVLA／CycleVLA 分别保存字段自查及 10 条原版核验；来源单元和缺口不混用 |
| [RoboTwin 仿真数据链路](robotwin/README.md) | 历史数据核对、原生区域及完整夹爪框试采视频；[当前字段自查](robotwin/check/native_field_check.md)，语义 annotation 与训练接入待完成 |

## 预保存格式（v2 草案）

2026-11-01：先统一预处理交付，再批量补标，最后**只回读已保存标签进行可视化**。这是可修订的保存接口；π0.5 的训练 batch、动作归一化、VAE latent 和弧长采样仍在训练适配层处理。

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

## 1. 正在了解的数据来源

| 数据 | 已有了解 | 仍需了解的问题 |
| --- | --- | --- |
| 本地 RoboTwin `event_data`，50 任务 × 50 条 clean 示范 | 已有本地数据与事件记录 | primitive 与语义 Event 的关系、对象区域、终点图像与状态的对齐、在训练中的作用 |
| RoboInter 的 DROID 部分（历史候选，本项目目录已移除） | 曾完成局部审计 | 动作语义、技能与对象覆盖、区域质量、不同视角如何利用、能提供哪些有效监督 |
| RoboCOIN Agilex 子集（历史候选，本项目目录已移除） | 官方数据卡提供双臂、末端位姿与子任务字段线索 | 实际子任务编码、并发事件覆盖、末端参考系／参考点、动作时间语义 |
| AgiBot World 2026 | 四主题有限样本已读；有双臂末端、技能区间、部分空间标注与接管反馈 | 分臂并发与持物事件、标注有效覆盖、坐标／动作语义、主题差异；详见[专项调研](agibot_world/README.md) |
| ABC-130K 原版及 Argus 标注 | 已查到双臂数据及分臂标注来源 | 原版访问、标注回链、分臂标签质量、对象区域及接入成本 |

独立轨迹数、技能—对象覆盖、有效监督覆盖和转换成本是可用于比较的维度，尚未据此确定最终组合或接入顺序。公开条目数、多视角副本数和标注段数的口径需要分清。更多候选及 RoboCOIN 的字段说明见 [source.md](source.md)。

## 2. 当前训练方向与前序讨论

共享模型预处理入口：[visual_grounding](../scripts/preprocess/visual_grounding/README.md)（Qwen3-VL 定位、SAM3 实例跟踪）、[text_annotation](../scripts/preprocess/text_annotation/README.md)（源区间与候选技能／对象关联）。这些路径已有历史试点；当前 LIBERO 和 RoboTwin 的主线均为原生完整屏内区域，具体结果以各来源报告为准。测试视频放来源的 `check/test/video/`，正式产物使用 `processed/`，原版 `check/raw/` 保留。

后续路线更新：用户已选择 **LIBERO／RoboTwin 原生相机实例 bbox／mask**，包括忽略遮挡、只保留屏内范围的完整夹爪与对象框。模型补标代码作为其他来源的备选保留。RoboTwin 的时间／动作语义与统一训练目标导出仍需接入；见 [当前原生字段表](robotwin/check/native_field_check.md)。以下早期 VLM＋SAM 讨论保留为历史，不覆盖此决定。

2026-10-30 后续标注方向已明确：借鉴 GuidedVLA 的构建流程，视觉补标采用 **VLM＋SAM3** 作为拟验证方案，参考 ReconVLA 构造项目所需局部视觉监督；子事件标注参考 AtomSkill，结合已有阶段／脚本／状态整理每臂 Event 与技能。两部分通过源帧与对象关联汇合，生成终点对象区域和剩余末端路径。具体 VLM、SAM3 权重／接口版本和试点协议尚未确定，未启动模型补标。来源专项安排见 [LIBERO](libero/guidedvla/guidedvla_pipeline.md) 和 [RoboTwin](robotwin/construction_plan.md)。

2026-10-30 用户明确：统一 π0.5，暂不复现 GuidedVLA，借鉴其方法分别构建 LIBERO 和 RoboTwin 仿真数据引擎。当前优先整理原始轨迹、按臂事件、区域与路径标签，再形成训练样本。原方法管线分别见 [LIBERO](libero/guidedvla/guidedvla_pipeline.md)、[RoboTwin](robotwin/guidedvla_pipeline.md)，RoboTwin 构建草案见 [construction_plan.md](robotwin/construction_plan.md)；具体来源版本、任务、clean／randomized、规模和训练配置尚未确定。

此前讨论的“外部机器人数据 mid-training → RoboTwin 适配 → Piper 微调”保留为后续候选路线，当前不把外部数据阶段作为仿真验证的前置条件。GuidedVLA 的纯动作微调后继续辅助监督可作为训练组织参考；是否采用两段训练及各段预算，仍需对 π0.5 和本项目联合目标具体确定。

主对照仍为 π0.5 动作基线与完整模型，两者使用一致的数据、初始化及训练预算；若后续加入外部训练阶段，也保持一致。动作映射、mask 和归一化需在所选基座中落实，并记录新增计算开销。

已知现有 clean20 checkpoint 与 `event_data` 的样本不一致，因此直接比较时存在数据差异；如何设置基线和使用榜单参照也应随具体研究问题确定。

## 3. 一种可讨论的链路分层

下面的分层用于梳理信息流。[fields.py](../scripts/preprocess/fields.py) 描述统一记录层的语义字段；预保存格式采用上面的 JSON＋NPZ 草案，训练容器和各来源迁移另行接入。

```text
来源快照与校验记录
  → 来源适配器：读取原始状态、动作、时间、图像索引与标注
  → 统一记录：轨迹／视角身份、按臂事件、对象角色、字段语义与有效性
  → 拆分及采样索引：轨迹分组、共同样本序列、视角／指令选择
  → 标签构造：Atomic Skill 映射、Target Grounding 画布、各臂 Latent Motion 剩余路径、动作窗口
  → 模型 batch：输入、监督、有效性 mask、追溯元数据
```

预保存层保留原始 RGB 引用，物化结构化标签和数值数组。是否再转换为 LeRobot 等训练容器，按后续读取需求决定；字段同名并不能证明动作或时间语义一致。

已有数据位于 `/data1/yuankai/robot-research/`：本地 RoboTwin 在 `data/robotwin/event_data/`，已下载 RoboInter 在 `data/external/robointer/raw/`。统一索引、缓存和转换产物的具体目录结构尚未确定。

## 4. 本次只读检查新增的接入证据

1. **RoboTwin 终点不一定有同索引 HDF5 图像。** `place_object_stand/aloha_agilex/data/episode_0000000.hdf5` 有 135 行 state/action 和图像，事件记录保留 136 帧，末 primitive 结束于 135。生成器将原始状态 `[:-1]` 写为 state、`[1:]` 写为 action，图像只保存 `[:-1]`。因此可以在验证配对后从 action 侧恢复末端终点状态，但终点图像需要独立定位原视频／其他原始证据，不能夹到第 134 帧替代。依据：[生成器](../../RoboTwin/envs/utils/pkl2hdf5.py)。该结论已核对一个样本及生成代码，未逐条审计全部 2,500 条。
2. **现成 DROID 适配器不直接接受 RoboInter 的主 state/action。** 本地 [DroidInputs](../../RoboTwin/XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/droid_policy.py) 读取 7 维关节与夹爪，输出接口为 8 维动作；RoboInter 主字段为 7 维末端状态和增量动作。本次 `episode_000000.parquet` 确认还保存了 7 维 `other_information.action_joint_position` 和 `observation_joint_position`，值得继续核实是否能沿用关节动作路径；夹爪命令及对齐关系尚不足以确定该转换。
3. **本地 FM 损失的归一化方式。** 本地 [pi0.py](../../RoboTwin/XPolicyLab/policy/Pi_05/openpi/src/openpi/models/pi0.py) 对动作末维直接取均值，未读取逐维标签 mask。如果混合不同维度的动作，padding 对目标和损失尺度有什么影响、是否采用有效维度 mask，需要结合最终动作表示研究。

RoboInter 的 500 条轨迹／1,000 个视角条目及质量统计沿用交接中的 chunk-000 审计，未在本次重复全量计算。RoboCOIN 本次只核对官方数据卡，尚未完成 Parquet 样本审计。

## 5. 可以继续调查的问题

1. 各候选实际能提供哪些标签、覆盖哪些技能和对象，质量与补标成本如何？已有分片是否具有代表性？
2. 不同来源的事件粒度、动作、时间与参考点差异，适合通过转换解决，还是需要调整研究表示？
3. 保留原生格式、建立统一索引或物化训练格式，哪种方式适合后续训练与反复修订？

这些是开放问题，不构成已排定的实现任务或验收门槛。

重放补 mask、SAM2／检测补标等新试点继续遵守 [实验管理规则](../../workspace/experiment/README.md)：先写具体方案并取得确认。本文保留调研问题及证据记录。

## Git 中保存的范围

代码、当前文档、来源版本、人工规则补充和少量核验摘要进入 Git。原始示范、全量标签及数组、帧图、视频、模型权重和运行日志保留在原本机／数据盘位置；本文链接的这类核验材料需在原工作区或单独恢复数据后查看。当前训练缓存与 checkpoint 不随代码发布。
