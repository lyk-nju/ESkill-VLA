# Data：ESkill 的数据需求与监督接口

本表用于逐来源核对已有字段与缺项。当前设计见 [Design](design.md)；保存格式与数据入口见 [data/README.md](../data/README.md)。ESkill Dataset 已实现以下派生标签；用于真实训练前仍须补齐来源时间与命令语义声明。

正式 train／val 在每个任务内部按完整父轨迹约 90%／10% 划分，帧、视角、双臂与派生记录跟随父轨迹。LIBERO 共 1,693 条，固定为 1,518 train／175 val；SFT 与 ESkill 共用 [split.json](../data/libero/atomic-vla/split.json)，归一化只读 train。

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
| 时间与同步信息 | Motion、Action 对齐 | 真实时间戳，或可核验的原始控制时钟＋保存行映射；保留删除 no-op／重采样的索引关系 |
| 坐标与相机信息 | Motion 投影与在线接口 | 主相机内外参、EEF 参考点、图像方向及 resize／pad；训练和推理使用同一投影定义 |

Grounding 只标主相机；这不改变策略使用双相机的配置。区域统一为忽略外部遮挡的完整屏内区域 `full_in_frame`，出画部分裁掉；完整出画时 bbox 无效，遮挡不缩框。RGB 保留真实遮挡，不生成或假定无遮挡像素。

## 从保存数据构造监督

| 监督 | 构造方式 | 有效性 |
| --- | --- | --- |
| 当前技能 | 当前臂 Event → 全局 ID → 六类活跃索引 | 技能有效且属于活跃类别 |
| Current Object | 当前对象 bbox＋当前主相机 → 256×256 原位画布 → 冻结 VAE latent | 对象绑定、区域与图像有效 |
| Future Object | 当前 Event 的对象引用＋`goal_frame` → 同样画布与 latent | 当前 Event 终点目标有效；不改用下一 Event 对象 |
| 未来 EEF 2D | 当前后 +0.25s…+2s 八时刻 → 同 Event 的 EEF → 主相机 224×224 归一化坐标 | 时间可核验、同 Event、坐标有效且在画面内 |
| 未来夹爪命令 | 与 EEF 相同查询时刻 → 对齐的演示控制命令 | 同 Event、命令语义与执行时序有效 |
| Action | 当前动作行起的 10 步源控制命令 → 训练归一化／补维 | 沿用 recipe 的动作窗口与尾部规则，不按 Event 截断 |

Future Object 始终是完整 Event 终点，Motion 是最长两秒的局部计划。没有释放标签时不从技能名称补造 Open；末端坐标、当前实际夹爪状态和未来夹爪命令分别保存其语义。

Motion 维护 `observed_mask`、`terminal_padded_mask`、各类缺失原因及独立 `xy_valid`／`gripper_valid`。终点填充默认不进入主辅助损失；内部缺测不当作终点保持。短 Event 无有效查询时仍可保留其他监督和 Action loss。

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

已有 40 任务、1,693 条／273,465 帧，保存了事件、实例关系、逐帧完整屏内区域、EEF 源像素坐标与全量相机参数。保留来源质量记录；已有标签不等于 ESkill 的秒级监督已经就绪。

| 内容 | 状态与下一步 |
| --- | --- |
| Event、对象区域与相机标定 | 复用现有资产；按 ESkill 的当前／终点对象语义读取 |
| 时间轴 | 当前 `timeline=source_steps`，时间戳／标称频率尚未补齐；核对原始控制时钟及 no-op 删除映射后才能生成秒级标签 |
| 夹爪标签 | 原动作中已有命令；八时刻采样代码已接入；仍需确认来源命令语义与执行顺序并写入声明 |
| 当前本体输入 | 已有源状态；已统一 EEF／夹爪适配，保留物理投影所需的未归一化值 |
| 在线相机／起点 | 离线／在线接口已接入；十条真实轨迹起点投影最大误差 0 源像素，不依赖未来字段 |
| 覆盖率统计 | 审计工具支持 all／train／val。全量 H10 跨 Event 为 7.17%，前缀5为 3.19%；秒级覆盖率待时间声明后统计 |

原始时钟尚未核实时，可以保留来源索引做诊断，但不能把它标为已核验的两秒目标。渲染帧率仅控制播放速度。

## 文件职责与人工核验

Episode 继续保存来源事实与逐帧字段，固定窗口、VAE latent 和训练张量由读取／变换层构造；不另建重复数据格式。保存层的区间仍为 `[start_frame, stop_frame)`，`goal_frame` 独立。具体 schema 见 [数据说明](../data/README.md)，现有字段检查入口为 [fields.py](../scripts/preprocess/fields.py)，ESkill 新需求需同步接入后才能视为通过。

预处理先补时间映射与语义依据，Dataset 生成查询和 mask，transforms 统一图像／坐标变换，模型只消费规范化 batch。RoboTwin 等来源使用相同职责划分，但先完成各自事件、实例与时钟核验。

每个来源使用“需要什么／有没有／备注”三列自查。新处理抽十条独立轨迹，只回读保存结果渲染：视频位于对应来源 `check/test/video/`，附查询时刻、Event 边界、xy／grip 有效性、当前／终点对象。先做人工核验，再安排正式训练。
