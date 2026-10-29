# ESkill-VLA 数据格式调研笔记

更新于 2026-10-29。本文保留来源字段、时间／坐标转换的讨论与检查记录。数据需求见 [Data](../docs/data.md)，模型与标签语义见 [Design §2–3](../docs/design.md)；具体来源适配和存储字段仍可修订。

字段名和分层方式用于举例；来源声明、本地检查结果与加工建议分别说明。具体数据组成、动作适配和技能映射仍需样本证据，不能将某个来源的原始格式直接作为所有来源的接口。

## 1. 来源描述：数值形状之外还要保存语义

一种可能的组织方式是来源注册表，分别描述数据来源、机器人配置与动作解释。下列 `SourceSpec` 等结构只是字段示意，是否采用这套抽象及其具体粒度仍可讨论。

```text
SourceSpec
  source_id, source_revision, local_root, license_reference
  artifact_paths_and_checksums, adapter_version
  embodiment_id, physical_arm_ids, arm_slot_mapping, camera_roles
  observation_time_spec, action_time_spec, annotation_time_spec
  state_spec, action_spec, effector_spec, camera_calibration_spec
  semantic_status, evidence_references

ActionSpec
  action_spec_id, ordered_channels, native_dimension
  representation                  # joint_position / eef_delta / ...
  command_or_recorded_next_state
  coordinate_frame, rotation_representation, units
  gripper_convention
  observation_action_alignment, action_step_duration_s
  model_transform_version, normalization_id, model_dimension_mapping

EffectorSpec
  arm_id, position_field, orientation_field
  coordinate_frame, units, orientation_convention
  reference_point, reference_point_transform_if_known
  position_semantics_valid, orientation_semantics_valid
```

未核实语义写为 unknown 并附原因，不能仅凭相关性或字段名称标为 verified。保留原始数值；只有在坐标、单位和参数化明确后才执行对应转换。仅位置语义已知时可以构造位置路径，不宣称姿态或参考点变换也已完成。

需要显式记录的例子：

- RoboTwin 当前 action 是下一记录时刻的状态；转换后不得再次机械移动一帧。
- RoboInter 主 action 的末端增量参考系、旋转增量组合规则与夹爪命令语义仍需核实。辅助列中的关节动作是另一个候选接口，不能把当前夹爪观测直接当作动作命令补齐。
- RoboCOIN 的 26 维同时包含关节和末端描述，不等于 26 个独立控制自由度；是否选择其中 14 维关节／夹爪通道需验证后决定。
- RoboTwin 的 `extrinsics_matrix` 写入逻辑优先取 `cam2world_gl`。使用投影筛查时必须核对矩阵方向及 OpenGL / CV 坐标约定，不能仅按键名解释为 world-to-camera。

## 2. 轨迹、视角和拆分身份

```text
TrajectoryRecord
  trajectory_id, source_id, source_revision, embodiment_id
  origin_episode_id?, source_clip_span?, split_group_id, split
  source_episode_records[]
  timeline, state_reference, action_reference, effector_references
  arm_presence[2], view_records[], instruction_records[]
  events_by_arm: {L: EventRecord[], R: EventRecord[]}
  cooperative_goal_groups[], provenance

ViewRecord
  view_id, source_episode_id, camera_role, media_reference
  frame_timestamp_mapping, image_geometry, calibration_reference?
  annotation_references, per_frame_role_validity
```

已审计 RoboInter 分片中的两个外部视角对应同一轨迹。评估独立样本数量和划分 split 时需要考虑这层关系，避免泄漏。训练时随机选择一个视角、同时使用多视角或有意设置视角采样权重，都是可以比较的方案；交接中讨论过先选轨迹再选一个外部视角，但尚不作为固定采样规则。

如果发布条目本身是源 episode 的裁剪，还需通过原始来源映射归入同一 `split_group_id`。仅删除视角后缀并不能证明已经恢复原始 DROID episode 身份；父轨迹映射无法核实时明确记录这一限制。所有跨来源重复数据也按可识别的原始来源分组。

图像、框和 mask 保存各自的原始分辨率及 resize / crop 变换。RoboInter 框使用的像素域需依据 `origin_shape` 等字段核对，不能仅按 320×180 视频尺寸猜测。

## 3. 时间、事件区间和终点

帧索引、秒时间、区间开闭及终点观测之间的关系值得单独记录。下面是一种将这些概念分开表达的结构，尚未确定最终字段形式。

```text
EventRecord
  event_id, trajectory_id, arm_id
  sample_start_index, sample_stop_index_exclusive
  source_interval, source_interval_convention
  semantic_terminal_time_s, goal_time_s, trace_end_time_s
  raw_skill, mapped_skill?, skill_mapping_version, skill_validity
  manipulated_object_id?
  cooperative_goal_group_id?
  expected_effect?, observed_outcome?
  boundary_source, source_confidence, validation_status, invalid_reasons
```

`[sample_start_index, sample_stop_index_exclusive)` 只决定哪些当前时刻属于该事件；终点时刻单独保存。来源的 inclusive end、下一段的开始索引与实际终点观测不能用一个 `end_frame` 混指。保留原始区间以便核验转换。

本地 RoboTwin 样例包含连续 primitive `[0,70]`、`[71,82]`、`[83,135]`，HDF5 却只有索引 0–134 的当前图像。其来源末端状态可以通过验证 `action_ee[:-1] == state_ee[1:]` 后，以 `state_ee` 加最后一个 `action_ee` 恢复完整记录；这不恢复末帧 RGB 或相机标定。构造 goal 时独立检查目标时刻的图像与区域，缺失则标无效，不能用最近可用帧悄悄代替。该样例及生成代码见 [数据入口](README.md#4-本次只读检查新增的接入证据)。

原始观测、图像、动作行数允许不同，需通过时间映射关联。原始末状态可作为 Latent Motion 的终点，即使它不是一个有有效动作的训练采样点。动作窗口按 episode 构造，不能因当前 Event 结束就把后续真实动作伪造为零；是否限制窗口跨 Event 属于采样协议，当前未新增这一限制。

**协作终点：** 按当前设计，共同追求同一对象终态的事件对齐语义终点，goal 与各臂真实剩余路径均取该终点。独立并发或效果不同的交接保留各自终点。样本审计需确认共享效果和技能范围；不能为了对齐而跨过该臂下一次不同操作。上述目标／路径时间字段用于记录观测关联，不代表另设不同预测时域。

## 4. 技能和对象映射

如果采用显式技能映射，可以记录来源标签、上下文、合并条件、对象关系及原始依据，以便回溯。下面列的是前序讨论过的候选，尚未证明它们适合各来源的事件粒度。

| 交接中的候选规则 | 需要调查的条件与反例 |
| --- | --- |
| RoboTwin grasp + lift → Pick | 同臂、同一对象、时间相邻且没有中间释放／新操作；不能将所有 lift 无条件并入前一 grasp |
| RoboInter transfer 并入 Place，直到释放 | 需要同对象、同臂、后续确有放置／释放证据；倒水、交接等前的搬运不能仅按动词并入 Place |
| manipulate 不监督技能 | 保留原事件和其他有效标签；映射未知不意味着整条样本无效 |
| 空 actor 继承前一抓取对象 | 必须有持续抓持／支撑证据，在释放、换物或不确定处停止继承 |
| 间隙映射为 idle / hold / return | 按交互关系与任务语义判断，未知空缺不能自动填 idle |

共享词表、上述规则及长尾技能处理尚待确认。未知标签不伪造新类别 ID，也不借用 `arm_presence` 屏蔽该臂。

## 5. 视觉角色与有效性

每个角色标签记录：臂、角色、视角、当前／目标时间、被操作对象引用、区域类型（bbox / mask）、区域坐标变换、原始质量等级、独立检查结果和无效原因。原始 `Primary` 等评级保留为来源信息，不直接等同于最终有效性。

`ee`、`obj`、`goal` 分别判断：当前执行器区域可用，不代表目标对象或事件目标也可用。RoboInter 的 `placement_proposal` 在已审计分片中等于终点对象框；若与本臂事件、同一对象身份及终点图像对应，可构造 goal 监督，仍需核对实际区域质量。夹爪框也不能仅凭 `Primary` 接受。外部视角缺标时可选择同轨迹另一视角，但当前与目标标签必须遵守一致的视角选择和空间对应规则。

缺失、未标注、不可见、区域无定义和质量检查失败分别保存原因。训练时由版本化规则生成 bool mask；其阈值与策略尚未由本草案确定。

来源 mask 或 bbox 用于构造保持原图布局的目标画布，并生成区域加权损失权重。在线预测器不读取真值区域、未来图像、事件 ID 或标签有效性。

## 6. 三维路径、参考系与物理时间

当前 Latent Motion 设想使用各臂到事件终点的三维末端路径并进行弧长采样。调研要看各来源能否提供合适的时间、位置与参考点，以及这种表达是否适合它们的行为。二维 `annotation.trace` 不能直接视为三维路径，也不能独立验证用同一夹爪框生成的标注。

**坐标适配：** 当前设计使用双臂共同固定参考系与统一固定 `s_xyz`。来源记录保留原始 frame、单位和可用变换；若原始位置位于各臂基座系，需先变换到共同参考系。具体轴向与参考点映射仍需核实，减去当前末端位置不等于完成旋转或本体对齐。尺度统计只使用训练 split，不按单条路径长度缩放。

目前了解到 RoboTwin 为 15 Hz、RoboInter 为 10 Hz，所查 RoboCOIN 样例声明为 30 Hz。相同 H 不对应相同秒数。保留原频率可减少重采样转换，但模型可能需要区分时间尺度；统一频率便于解释窗口时长，但要核实状态插值与增量动作重算。两条路线都值得讨论，尚未选定默认值。

## 7. 一种训练样本接口示意

下表用于检查当前设想涉及哪些信息。来源／本体条件、统一末端状态、动作 mask 以及具体形状都是候选设计，并非已经决定要新增的模块或字段。

| 层 | 主要字段与形状 | 约束 |
| --- | --- | --- |
| 输入 | 当前／历史图像、图像有效性、完整任务指令 | 只有当前及历史可得信息；视角选择在 B / C 相同 |
| 输入 | 原生状态经来源适配后的 `current_state`、`body_state[2,D_body]` | 原生动作专家状态与辅助末端状态分别定义，不能用同一个 state 名称掩盖语义差异 |
| 输入／静态配置 | `arm_presence[2]`、来源／本体／动作接口标识 | 物理臂存在性来自配置；单臂放 L 的建议仍待定，映射需保留原臂身份 |
| Atomic Skill 目标 | `skill_labels[2]`、`skill_validity[2]` | 缺失标签不影响该臂在线预测 |
| Target Grounding 目标 | 画布或缓存 VAE latent `[2,3,...]`、`role_validity[2,3]`、区域权重 | 缓存必须包含视角、图像变换、画布规则和 VAE 版本键 |
| Latent Motion 目标 | `trace_displacements_m[2,M,3]`、`trace_validity[2,M]` | 参考点、参考系、终点与采样规则均可追溯 |
| 动作目标 | `actions[H,D_max]`、`action_validity[H,D_max]` | mask 同时表达时间、真实通道、缺失／无效值；padding 不增加监督维数 |
| 元数据 | source / trajectory / split group / view / event IDs、采样时间、目标时间、规则版本 | 用于构造、审计与复现；未来标签信息不进入在线输入 |

需要查清 `annotation.instruction_add`、`substask` 与 episode 任务指令各自的生成和使用语境。若把逐阶段离线文本作为策略输入，应解释部署时是否能获得相同信息；使用全局任务描述、将阶段文本用于监督是另一种可讨论的选择。

动作转换、归一化和 padding 的顺序需要结合最终表示确定。按来源／本体／动作接口维护统计量是一种候选；保留物理尺度还是使用统计尺度也值得比较。调查时应留意重复差分，例如把已有 delta 再送入绝对关节的 `DeltaActions`，以及归一化统计是否混入评测数据。

本地 FM 对动作末维直接取均值。异构维度下，固定 padding 分母、有效元素归一化和来源权重会怎样改变目标，需要结合实际动作布局分析；有效维度 mask 是一个候选做法，尚未选定 reduction。若采用 B / C 对照，动作处理和损失权重差异也需要纳入控制。

## 8. 采样与追溯方面的调研问题

如果后续希望复现相同样本流，可以考虑记录 `sample_id, source_id, trajectory_id, current_index, view_id, instruction_id, augmentation_seed` 及处理规则版本。保存索引清单和确定性生成各有存储、灵活性上的取舍。只记录同一个全局 seed 不足以保证不同取数流程得到相同样本。

标签缺失时，保留样本并屏蔽部分损失、调整采样或补标都是待比较的处理方式；如对照组采用不同过滤，解释结果时需计入样本差异。原始标注与后续修订的来源关系也值得保留。

继续调研可关注：能否追溯原始样本，是否存在跨 split 的重复轨迹，按臂事件如何表达，时间／坐标是否明确，终点状态与图像是否齐全，以及有效监督覆盖怎样随候选方案变化。这些问题用于积累证据，不作为当前阶段的实现验收清单。专项试点仍按实验规范另行确认。
