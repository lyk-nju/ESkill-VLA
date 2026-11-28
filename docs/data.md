# Data：ESkill 的数据需求与监督接口

LIBERO 研究配方读取 `atomicdata/libero40_grounding`。主图和完整任务是必需输入，可加入同帧腕图。Current 使用当前 Event 的目标可见区域，Planner 使用技能和主图 EEF 投影，动作训练按记录的控制行读取对应观测与动作窗口。 [模型设计](design.md) · [数据格式](../data/README.md)

正式 train／val 在每个任务内部按完整父轨迹约 90%／10% 划分，帧、视角、双臂与派生记录跟随父轨迹。LIBERO 共 1,693 条，固定为 1,518 train／175 val；SFT 与 ESkill 共用 [split.json](../data/libero/atomic-vla/split.json)，归一化只读 train。

Motion Planner 的正式与技术配方均设置 `skill_sampling=false`。历史联合动作配方还支持按技能重采样，权重为 `w_s=min(3, sqrt(N_max/N_s))`，`N_s` 按实际动作样本配对的当前观察所属 Event 统计。[训练接口](../src/openpi/training/README.md)

启用状态一致反事实时，数据集另读 `index.json` 同目录下的 `counterfactual_states.json`。v3 为支持文字替换的轨迹保存源／目的实例、可替换帧区间以及需要保护的接触实体区间。`null` 表示该轨迹沿用原始指令，可继续对独立实体执行位置增强。 [生成和训练配置](augmentation.md#状态一致反事实)

## 事件边界标签

`boundary_xy` 的两个 `(x,y)` 分别代表当前观测 Start 和当前 Event 的 End，在主相机模型画布上归一化。End 使用 `[start_frame, stop_frame)` 内最后一个有效相机帧的同帧 EEF 投影。`boundary_valid` 逐点屏蔽缺失或越界投影，`boundary_source_frame_indices` 保留源帧。八个未来点具有独立的 `trace_valid`。短事件的 End 可以早于最短 10 帧运动窗口的终点。镜像、裁剪和旋转同步变换 RGB、两个边界和未来点。[标签实现](../src/openpi/policies/eskill_dataset.py)

## 外部数据自查表

| 数据内容 | 对应模块 | 具体需要什么 |
| --- | --- | --- |
| RGB 图像 | VLM、Object Grounding | 同步当前主／腕相机 RGB；完整序列可查询 Event 终点主相机，保留帧与视角对应 |
| 任务指令 | VLM | 完整 Episode 的自然语言任务描述 |
| 机器人状态 | 离线监督／核验 | EEF 三维轨迹用于投影监督；明确执行臂、参考点、坐标系及单位，不作为模型输入 |
| 机器人动作与运动记录 | Motion、Action | 源动作序列、各臂逐帧 EEF 位置、夹爪命令；区分实际开度与控制命令，核对符号和执行时序 |
| 每臂 Event 边界 | Skill、Grounding、Motion | 每臂起止区间及真实 `goal_frame`；同臂不重叠、跨臂可并发 |
| 子事件描述与技能标签 | Skill-MoE | 当前 Event 语义；保留九类编号，首版只用六类活跃路由，缺失与未支持标签显式记录 |
| 执行器视觉区域 | Subject（专项训练）／核验 | 已存夹爪完整屏内 bbox／mask；启用 Subject 时构造夹爪局部 RGB 画布，原联合动作路径不依赖此项 |
| 当前对象视觉区域 | Current Object | 各帧操作实例／部件的完整屏内 bbox／mask，具有跨帧及 Event 终点的一致绑定 |
| Event 终点对象区域 | Future Object | 按当前 Event 的 `goal_frame` 查询同一对象区域和 RGB，不需要额外重复标注 |
| 时间与同步信息 | Motion、Action 对齐 | 当前使用保存主相机帧序列；保留删除 no-op／重采样的索引关系，真实时间戳可选 |
| 坐标与相机信息 | 离线 Motion 投影 | 主相机内外参、EEF 参考点、图像方向及 resize／pad；仅用于生成轨迹标签 |

Grounding 标注位于主相机。完整屏内框 `full_in_frame` 描述对象的屏内几何范围，忽略外部遮挡；原生可见 mask 描述当前图像中的实际可见像素。Motion Planner 使用后者构造物理实体的 Current 分布。具备几何投影的任务区域可使用区域 mask 或框覆盖率。

### 交互目标修订：数据核验中

新语义以当前 Event 要交互的目标为准：Pick milk → milk；Place milk into basket → basket；Open／Close／Turn → 对应抽屉、门或旋钮；Push 首版仍指被推动物体。目标在原 Event 边界切换，取当前帧区域。明确的空间目标（如盘子右侧、caddy 后格）保留原生区域，不替换成整张桌子。

核验格式在 Episode 增加 `scene_entities[entity_id]`，集中保存实例／部件身份和逐帧 `regions.image`；Event 的 `interaction_target` 引用实体，并记录绑定状态、候选、依据和人工审核状态。全场景有限物体和固定装置都生成 mask／bbox，目标在视频中为红框，其余实例为绿框。桌面、地面、墙面及 arena 装饰不计入物体清单；原夹爪区域继续保留。

全量 40 任务、1,693 条／273,465 帧已补齐场景区域与在线增强元数据，见 [全量 Grounding 核验](../data/libero/atomic-vla/check/test/full_grounding/README.md)。独立目录 `atomicdata/libero40_grounding` 保留原 RGB／动作及来源依据，训练所需文件均在各 Episode 内；原 `object_regions` 仍表示被操作物体。`GroundingDataset` 在 `grounding`、`visual_primitive` 和 `motion_planner` 三个训练入口使用 `interaction_target`。历史联合动作训练使用下述 Current／Future 接口。自动生成完成与人工质量验收分别记录，实例绑定的人工状态仍为 pending。

以下增强参数记录历史 Current／Subject 专项。Motion Planner 的反事实与 Copy 概率见其[联合配方](../examples/libero/train_motion_planner_joint.json)。

专项派生数据保留所有实体的完整屏内框／mask，包括原生遮挡下的区域。`scene_entities[*].regions.image.visible_masks` 单独保存同帧原生可见像素；另有 `visible_fraction`、`copy_eligible`、`interaction_locked`、`online_grounding.interaction_boxes` 和 Event 文本模板。完整任务反事实仅对可唯一解析的简单搬运任务替换源物体；候选须 `visible_fraction >= 0.8`、描述唯一且 `interaction_locked=false`；可定位、任务可替换和 `copy_eligible` 三种条件独立。缺少锁定标记或无合格候选时不替换。复杂任务保留原指令，Copy-Paste 不再按容器类别设黑名单，可移动物体统一检查非交互锁定、至少 98% 可见及完整区域有效，并由独立的小型空间词保护决定，并先计算无碰撞、满足位移范围的二维可行落点，再采样粘贴；原 Event 目标可安全移动时以 80% 概率优先选择，其余候选平分剩余概率，原目标不可移动时在其他候选中均匀选择。文本替换保留 20% 条件触发概率；真实 Event 和 Subject 的原生遮挡标签仍保留。可见 mask 另用于编辑及区分增强新造成的完全遮挡。`robot::single` 完整 mask 用于阻止 Copy-Paste 擦除／覆盖机器人，不再保护全部机器人免受随机 Patch Masking。

每次读取只采一个目标；先编辑源 RGB、更新框，再构造干净 Current VAE 画布，最后对输入做 Patch Masking；当前文本目标与夹爪的完整框各自每边外扩 8 源像素，与局部交互区一同保护，其他实体可被随机遮挡。新文本和对应完整框同时替换，不带反事实 Skill／Motion／Action 标签。模型直接读取完整任务，不生成中间短句；真实标签按 Event 边界切换。开启 `model.grounding_subject` 后，第二个 role 是夹爪 Subject，直接复用 `effector_regions`，每边 4 源像素的真实 RGB 上下文可包含抓取物体；不需要新增持物身份。Subject 和 Current 各有有效性掩码。

全量目录沿用原父轨迹 split（1,518 train／175 val），不把派生样本当作新轨迹。全部 Event 目标框有效，34 帧夹爪完整出画，仅关闭对应 Subject 监督。原生可见 mask 用于反事实筛选和编辑，不能替代完整框。先前十条／2,198 帧的增强实验保留在 `workspace/reports/atomicvla_online_grounding_20261117_001`，作为历史 [在线增强核验](../data/libero/atomic-vla/check/test/online_grounding/README.md)，不作为独立泛化评测。正式训练用法见 [训练说明](../src/openpi/training/README.md#current-grounding-专项训练)。

Current／Subject 在线增强按两个处理阶段串联：第一阶段为反事实与局部编辑（Copy-Paste、单目标文本替换、干净画布、受保护 Patch Mask），第二阶段为经典图像增强（按官方 π0.5 PyTorch 参数同步裁切／旋转主图及两项标签，再仅扰动输入颜色；当前配置关闭翻转）。两个增强阶段在同一训练样本上依次执行，与模型训练的 Stage 1／Stage 2 无关。Dataset 从已有完整 bbox 生成仅供增强使用的 `role_core_boxes`；几何增强后必须仍有未外扩目标区域，不能仅凭背景边距判定标签有效。完全出画只屏蔽对应 Current／Subject，部分出画与原生遮挡标签继续保留。方向词查询跳过几何变换；验证关闭全部增强。参数见训练说明。

训练增强不修改已保存数据。轻裁切与可选±5°中心旋转同步变换主图、Current／Future画布及权重和未来二维轨迹；采用共享归一化参数，先裁切再旋转，合成一次采样。被裁掉的点不会因旋转重新进入监督，旋转后出画点或完整区域也会失效；原Event／Skill／Action／Gripper标签不变。腕图几何不变。颜色扰动只用于输入RGB，监督画布保留原色；填充区域保持黑色。参数与人工核验见[增强对照](../experiments/eskill_libero40_direct/README.md#视觉增强对照)。

## 从保存数据构造监督

Motion Planner 按主相机保存帧构造一条训练记录。一次成功的目标文本替换、实体移位，或遮住当前目标或交互区的随机遮挡，会屏蔽该行的 Skill 和 Trace 监督。Current 依照编辑后的目标及剩余可见区域计算。轨迹使用未来第 5、10、…、40 个保存帧上的二维 EEF 标签，超过有效窗口后可重复终点。实现见 [GroundingDataset](../src/openpi/policies/grounding_dataset.py)，训练与诊断见 [Motion Planner 实验](../experiments/motion_planner/README.md)。

下表保留历史联合动作模型的监督接口，供其数据导出和已有实验复核。

| 监督 | 构造方式 | 有效性 |
| --- | --- | --- |
| 当前技能 | 当前臂 Event → 全局 ID → 六类活跃索引 | 技能有效且属于活跃类别 |
| Current Object | 当前对象 bbox＋当前主相机 → 256×256 原位画布 → 冻结 VAE latent | 对象绑定、区域与图像有效 |
| Future Object | 当前 Event 的对象引用＋`goal_frame` → 原位画布每边额外外扩 8 px → latent | 当前 Event 终点目标有效；不改用下一 Event 对象；源 bbox 不改写 |
| 未来 EEF 2D | 当前后 +5、10…40 个保存主相机帧 → Event＋最短 10 帧窗口的 EEF → 主相机 224×224 归一化坐标；V3 超过窗口后保持终点 | 来源坐标有效且在画面内；派生保持点单独标记 |
| 未来夹爪命令 | +5、10…40 查询帧 → 对齐的演示控制命令 | Event 内或偏移不超过 10 帧，不跨 Episode，不使用二维保持点替代命令 |
| Action | 当前动作行起的 10 步源控制命令 → 训练归一化／补维 | 沿用 recipe 的动作窗口与尾部规则，不按 Event 截断 |

Future Object 始终是完整 Event 终点，Motion 是未来 40 个保存帧的局部计划。没有释放标签时不从技能名称补造 Open；末端坐标、当前实际夹爪状态和未来夹爪命令分别保存其语义。

Motion 使用当前 Event＋最短 10 个保存帧窗口（名义约 0.5 秒），前两个点可跨 Event。V3 配方 `motion_terminal_hold=true`：超过 Event 终点与最短窗口终点中较晚者的二维查询，保持该位置；Episode 越界同样保持最后观察。保持点是派生停止目标，以相同点权重进入 Trace loss，夹爪仍只使用真实命令。当前 Skill 与 Goal 不切换到下一 Event。关闭开关（旧配方默认）时保持原有越界屏蔽。内部缺测不填补，终点投影无效则派生点无效。

`trace_terminal_hold` 与 `terminal_padded_mask` 标记派生点，`observed_mask`／`future_observed` 仅标真实未来查询，两者互斥。`trace_source_frame_indices` 记录二维标签的真实来源（重复终点时可与查询帧不同）；`future_source_frame_indices` 仍记录允许窗口内的真实未来查询，越界为 -1。Episode 文件保存来源事实，不改写；这些目标在 Dataset 内按配置派生。完整实例见 [边界规则](action_representation.md#event-边界与缺失标签)。

诊断分别使用 `future_effector_missing`（无对应 EEF 记录）、`future_projection_invalid`（记录存在但投影无效）及 `future_command_missing`（无可用且无歧义的命令）。`future_event_tail`／`future_episode_tail` 标记越界，两者可重叠。`future_cross_event` 标记最短窗口内允许跨 Event 的查询；此时 `future_event_tail` 可为 true 且主标签有效。`future_observed` 仅表示查询位于已保存相机序列及允许监督的窗口内，不保证 RGB、EEF 或命令都有效。未知／无效 Event 对应元数据 `event_ids=None`，路径与夹爪 mask 排除该监督。

启用保持后，前两项诊断检查实际二维来源帧（包括重复的终点），`future_command_missing` 仍检查原查询命令。`future_episode_tail=true` 的二维点可因有效终点保持进入 loss，但不能计为真实未来观测。覆盖率和误差均区分真实点、派生点，避免把补充监督解释为新增数据记录。

删除含义混杂的 `future_missing`：保存相机行的偏移查询不存在中间的索引匹配失败，`matched=false` 只表示 Episode 尾部；源帧索引跳号也不应被当作保存序列缺帧。此项删除不改变路径、夹爪或其有效 mask。

真实技能、Event 边界、目标帧、未来状态／动作与标签 mask 只用于损失。模型条件只来自当前 RGB、任务指令与固定平台臂身份；标签缺失不关闭预测分支，未来有效性不屏蔽 Action 中的 Motion tokens。

## 当前资产与待补接口

历史联合动作模型读取下列基础 Episode，Motion Planner 读取同级的 `libero40_grounding/`。

~~~text
/data1/yuankai/robot-research/data/atomicdata/libero40/
  index.json
  episodes/<id>/
    episode.json
    arrays.npz
    images.npz
~~~

已有 40 任务、1,693 条／273,465 帧，保存了事件、实例关系、逐帧完整屏内区域、EEF 源像素坐标与全量相机参数。保留来源质量记录；帧偏移标签已可直接构造，真实训练效果仍需实验验证。

这些完整状态指源服务器数据盘；代码更新不会同步其他机器的外部数据。旧副本缺少夹爪声明时，按[跨机器迁移说明](../data/libero/atomic-vla/check/test/training/README.md#跨机器补齐夹爪声明)使用源核验清单比对动作／状态，再补充元数据；不要求迁移原始缓存，也不静默按数值猜测约定。

| 内容 | 状态与下一步 |
| --- | --- |
| Event、对象区域与相机标定 | 复用现有资产；按 ESkill 的当前／终点对象语义读取 |
| 时间轴 | 当前 `timeline=source_steps`；按已保存主相机帧序列采样，不强制补时间戳 |
| 夹爪标签 | 1,693 条原动作与源缓存逐值一致；第 7 维 -1=Open、+1=Close，观察后执行，已写入 `actions.gripper_convention`；未来采样使用相同帧偏移 |
| 模型输入 | RGB＋任务指令；不读取状态、二维起点或相机标定，平台臂身份固定 |
| 轨迹投影 | 仅作为离线未来点监督；在线不输入标定或起点，当前点无效不屏蔽未来有效标签 |
| 覆盖率统计 | 审计工具统计 Event 数、帧长分布、任务／技能／划分及开合命令；与 Dataset 共用有效标签计算。已按 +5…+40 帧统计真实有效监督 |

按 LIBERO 默认 20 Hz 估算，间隔约 0.25 秒、最远约 2 秒；这是量级参考，数据仍保留帧偏移定义。渲染帧率仅控制播放速度。

## 文件职责与人工核验

Episode 继续保存来源事实与逐帧字段，固定窗口、VAE latent 和训练张量由读取／变换层构造；不另建重复数据格式。保存层的区间仍为 `[start_frame, stop_frame)`，`goal_frame` 独立。具体 schema 见 [数据说明](../data/README.md)，现有字段检查入口为 [fields.py](../scripts/preprocess/fields.py)，ESkill 新需求需同步接入后才能视为通过。

预处理先核对保存帧配对与命令语义依据，Dataset 生成查询和 mask，transforms 统一图像／坐标变换，模型只消费规范化 batch。RoboTwin 等来源使用相同职责划分，但先完成各自事件、实例与时钟核验。

每个来源使用“需要什么／有没有／备注”三列自查。新处理抽十条独立轨迹，只回读保存结果渲染：视频位于对应来源 `check/test/video/`，附查询时刻、Event 边界、xy／grip 有效性、当前／终点对象。先做人工核验，再安排正式训练。

## Stage 1 镜像派生样本

镜像在训练 batch 内即时生成，不改写 Episode 或复制一份数据。主／腕相机同步水平翻转；256×256 Current／Future 画布及权重同步翻转后才进行 VAE 编码，Future 的每边 8 px 外扩先按原规则完成。224×224 画布的归一化有效未来 EEF 点使用 `x'=1-x, y'=y`，其时间偏移、有效 mask 不变。

臂身份、Event、技能、夹爪命令保持原值；模型不输入本体状态，不需要 arm_state_mask。原图、镜像、Stage 2 和在线推理均只使用 RGB＋任务指令。镜像不提供动作真值，只用于无 Action loss 的 Stage 1。方向性语言样本只保留原图；Stage 2 和验证不翻转。同一原轨迹及镜像仍属于相同 split。既有[核验视频](../data/libero/atomic-vla/check/test/training/mirror_videos.md)中的状态门控／起点展示属于历史实现。
