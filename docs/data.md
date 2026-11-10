# Data：ESkill 的数据需求与监督接口

本表用于逐来源核对已有字段与缺项。当前设计见 [Design](design.md)；保存格式与数据入口见 [data/README.md](../data/README.md)。ESkill Dataset 已实现以下派生标签；LIBERO 夹爪命令语义已全量核对；第一版直接使用固定保存帧偏移，物理时间不再是训练前置。

正式 train／val 在每个任务内部按完整父轨迹约 90%／10% 划分，帧、视角、双臂与派生记录跟随父轨迹。LIBERO 共 1,693 条，固定为 1,518 train／175 val；SFT 与 ESkill 共用 [split.json](../data/libero/atomic-vla/split.json)，归一化只读 train。

当前 ESkill 配方对训练样本按技能做有上限的平方根倒频率重采样：`w_s=min(3, sqrt(N_max/N_s))`，`N_s` 按实际动作样本配对的当前观察所属 Event 统计。只改变训练抽样概率，源数据、划分、归一化和验证分布保持原定义；实现与多卡恢复见 [训练接口](../src/openpi/training/README.md)。

## 外部数据自查表

| 数据内容 | 对应模块 | 具体需要什么 |
| --- | --- | --- |
| RGB 图像 | VLM、Object Grounding | 同步当前主／腕相机 RGB；完整序列可查询 Event 终点主相机，保留帧与视角对应 |
| 任务指令 | VLM | 完整 Episode 的自然语言任务描述 |
| 机器人状态 | Motion | 当前 EEF 三维位置、姿态、实际夹爪状态；明确执行臂、参考点、坐标系及单位 |
| 机器人动作与运动记录 | Motion、Action | 源动作序列、各臂逐帧 EEF 位置、夹爪命令；区分实际开度与控制命令，核对符号和执行时序 |
| 每臂 Event 边界 | Skill、Grounding、Motion | 每臂起止区间及真实 `goal_frame`；同臂不重叠、跨臂可并发 |
| 子事件描述与技能标签 | Skill-MoE | 当前 Event 语义；保留九类编号，首版只用六类活跃路由，缺失与未支持标签显式记录 |
| 执行器视觉区域 | 数据核验，可选 | 夹爪完整屏内 bbox／mask 可保留；ESkill 不要求其视觉重建监督 |
| 当前对象视觉区域 | Current Object | 各帧操作实例／部件的完整屏内 bbox／mask，具有跨帧及 Event 终点的一致绑定 |
| Event 终点对象区域 | Future Object | 按当前 Event 的 `goal_frame` 查询同一对象区域和 RGB，不需要额外重复标注 |
| 时间与同步信息 | Motion、Action 对齐 | 当前使用保存主相机帧序列；保留删除 no-op／重采样的索引关系，真实时间戳可选 |
| 坐标与相机信息 | Motion 投影与在线接口 | 主相机内外参、EEF 参考点、图像方向及 resize／pad；训练和推理使用同一投影定义 |

Grounding 只标主相机；这不改变策略使用双相机的配置。区域统一为忽略外部遮挡的完整屏内区域 `full_in_frame`，出画部分裁掉；完整出画时 bbox 无效，遮挡不缩框。RGB 保留真实遮挡，不生成或假定无遮挡像素。

## 从保存数据构造监督

| 监督 | 构造方式 | 有效性 |
| --- | --- | --- |
| 当前技能 | 当前臂 Event → 全局 ID → 六类活跃索引 | 技能有效且属于活跃类别 |
| Current Object | 当前对象 bbox＋当前主相机 → 256×256 原位画布 → 冻结 VAE latent | 对象绑定、区域与图像有效 |
| Future Object | 当前 Event 的对象引用＋`goal_frame` → 原位画布每边额外外扩 8 px → latent | 当前 Event 终点目标有效；不改用下一 Event 对象；源 bbox 不改写 |
| 未来 EEF 2D | 当前后 +5、10…40 个保存主相机帧 → Event＋最短 10 帧窗口的真实 EEF → 主相机 224×224 归一化坐标 | 帧对应明确、处于允许窗口、坐标有效且在画面内 |
| 未来夹爪命令 | 与 EEF 相同查询帧 → 对齐的演示控制命令 | Event 内或偏移不超过 10 帧，命令语义与执行时序有效 |
| Action | 当前动作行起的 10 步源控制命令 → 训练归一化／补维 | 沿用 recipe 的动作窗口与尾部规则，不按 Event 截断 |

Future Object 始终是完整 Event 终点，Motion 是未来 40 个保存帧的局部计划。没有释放标签时不从技能名称补造 Open；末端坐标、当前实际夹爪状态和未来夹爪命令分别保存其语义。

Motion 使用当前 Event＋最短 10 个保存帧窗口（名义约 0.5 秒），前两个点可跨 Event；Episode 尾部仍排除。当前 Skill 与 Goal 不切换到下一 Event。Motion 维护 `observed_mask`、`future_cross_event`、`terminal_padded_mask`、各类缺失原因及独立 `xy_valid`／`gripper_valid`。终点填充默认不进入主辅助损失；内部缺测不当作终点保持。短 Event 无有效查询时仍可保留其他监督和 Action loss。

诊断分别使用 `future_effector_missing`（无对应 EEF 记录）、`future_projection_invalid`（记录存在但投影无效）及 `future_command_missing`（无可用且无歧义的命令）。`future_event_tail`／`future_episode_tail` 标记越界，两者可重叠。`future_cross_event` 标记最短窗口内允许跨 Event 的查询；此时 `future_event_tail` 可为 true 且主标签有效。`future_observed` 仅表示查询位于已保存相机序列及允许监督的窗口内，不保证 RGB、EEF 或命令都有效。未知／无效 Event 对应元数据 `event_ids=None`，路径与夹爪 mask 排除该监督。

删除含义混杂的 `future_missing`：保存相机行的偏移查询不存在中间的索引匹配失败，`matched=false` 只表示 Episode 尾部；源帧索引跳号也不应被当作保存序列缺帧。此项删除不改变路径、夹爪或其有效 mask。

真实技能、Event 边界、目标帧、未来状态／动作与标签 mask 只用于损失。模型条件来自当前观察与在线可获得的本体／相机信息；标签缺失不关闭预测分支，未来有效性不屏蔽 Action 中的 Motion tokens。

## 当前资产与待补接口

正式 LIBERO 数据入口：

~~~text
/data1/yuankai/robot-research/data/atomicdata/libero40/
  index.json
  episodes/<id>/
    episode.json
    arrays.npz
    images.npz
~~~

已有 40 任务、1,693 条／273,465 帧，保存了事件、实例关系、逐帧完整屏内区域、EEF 源像素坐标与全量相机参数。保留来源质量记录；帧偏移标签已可直接构造，真实训练效果仍需实验验证。

| 内容 | 状态与下一步 |
| --- | --- |
| Event、对象区域与相机标定 | 复用现有资产；按 ESkill 的当前／终点对象语义读取 |
| 时间轴 | 当前 `timeline=source_steps`；按已保存主相机帧序列采样，不强制补时间戳 |
| 夹爪标签 | 1,693 条原动作与源缓存逐值一致；第 7 维 -1=Open、+1=Close，观察后执行，已写入 `actions.gripper_convention`；未来采样使用相同帧偏移 |
| 当前本体输入 | 已有源状态；已统一 EEF／夹爪适配，保留物理投影所需的未归一化值 |
| 在线相机／起点 | 离线／在线接口已接入；十条真实轨迹起点投影最大误差 0 源像素，不依赖未来字段 |
| 覆盖率统计 | 审计工具统计 Event 数、帧长分布、任务／技能／划分及开合命令；与 Dataset 共用有效标签计算。已按 +5…+40 帧统计真实有效监督 |

按 LIBERO 默认 20 Hz 估算，间隔约 0.25 秒、最远约 2 秒；这是量级参考，数据仍保留帧偏移定义。渲染帧率仅控制播放速度。

## 文件职责与人工核验

Episode 继续保存来源事实与逐帧字段，固定窗口、VAE latent 和训练张量由读取／变换层构造；不另建重复数据格式。保存层的区间仍为 `[start_frame, stop_frame)`，`goal_frame` 独立。具体 schema 见 [数据说明](../data/README.md)，现有字段检查入口为 [fields.py](../scripts/preprocess/fields.py)，ESkill 新需求需同步接入后才能视为通过。

预处理先核对保存帧配对与命令语义依据，Dataset 生成查询和 mask，transforms 统一图像／坐标变换，模型只消费规范化 batch。RoboTwin 等来源使用相同职责划分，但先完成各自事件、实例与时钟核验。

每个来源使用“需要什么／有没有／备注”三列自查。新处理抽十条独立轨迹，只回读保存结果渲染：视频位于对应来源 `check/test/video/`，附查询时刻、Event 边界、xy／grip 有效性、当前／终点对象。先做人工核验，再安排正式训练。

## Stage 1 镜像派生样本

镜像在训练 batch 内即时生成，不改写 Episode 或复制一份数据。主／腕相机同步水平翻转；256×256 Current／Future 画布及权重同步翻转后才进行 VAE 编码，Future 的每边 8 px 外扩先按原规则完成。224×224 画布的归一化 EEF 起点和有效未来点使用 `x'=1-x, y'=y`，其时间偏移、有效 mask 不变。

物理状态数值、臂身份、Event、技能、夹爪命令保持原值；镜像样本用 `arm_state_mask=false` 关闭 Motion 的八维状态分支（包括投影 bias），保留镜像后的二维 EEF 起点。原图、Stage 2 和在线推理使用真实臂状态；Skill／Grounding 本来只从视觉与任务提取先验。不提供镜像动作真值，镜像只用于无 Action loss 的 Stage 1。方向性语言样本只保留原图；Stage 2 和验证不翻转。同一原轨迹及镜像仍属于相同 split。核验视频和标签见 [mirror_videos.md](../data/libero/atomic-vla/check/test/training/mirror_videos.md)。
