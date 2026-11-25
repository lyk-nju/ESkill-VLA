# Event Organization：每臂事件与对象目标

Event 是一次具体的 Skill 执行，也是对象、目标和局部运动监督的关联单元。Motion Planner 使用 LIBERO 单臂 Event。抓取 milk 后把它放入 basket 时，两个 Event 搬运的实例都是 milk，`interaction_target` 依次指向 milk 和 basket。以下每臂字段也用于其他来源及历史联合动作模型。[Motion Planner 设计](design.md)

## 每臂维护事件流

同一臂的 Event 按时间排序、互不重叠；左右臂可以并发，分别拥有自己的 Skill、对象与终点。单臂数据保留真实臂身份，双臂模型槽位在适配层映射。

一个 Event 至少关联：

| 信息 | 含义 |
| --- | --- |
| `arm`、`event_id` | 执行臂与该次事件 |
| `[start_frame, stop_frame)` | 当前帧所属区间，右端不含 |
| `goal_frame` | 该事件的真实终点观察 |
| `description`、`skill` | 原始子事件描述与统一技能类别 |
| `object_regions`、来源实例绑定 | 当前 Event 操作的具体实例／部件及其逐帧区域 |
| `interaction_target.entity_id` | 当前 Event 的交互目标，引用 `scene_entities` 中的实体、部件或区域 |
| 有效性与依据 | 边界、技能、对象、区域分别记录，不用一个总开关替代 |

`stop_frame` 决定事件归属，`goal_frame` 决定终点目标，两者不必是同一帧。AtomicVLA 采用 `goal_frame=stop_frame−1`；其他来源若有额外终点观察，应显式保存，不能由读取器猜测。

同一对象可跨多个 Event 复用。多个同类物体须绑定具体实例，不能只按名词选择。终点查询始终使用当前 Event 自己的对象引用，即使目标帧邻接下一个 Event，也不切换对象。

## 数据标签与物理完成

沿用已核验的 AtomicVLA 事件与修订依据。事件名称描述操作语义，不保证其区间覆盖接触、释放和稳定落位全过程。Place 末段仍闭爪时，不据此自动延长事件、补开爪命令或宣布失败。

历史联合动作模型的 Future Object 取记录的 Event 终点，作为该阶段目标的监督代理。已知错标、失败或不可信的终点单独标记／排除对应监督；不把“缺少完整释放”本身视为必须修复的错误。成功、数据有效性与人工验收是不同状态。

缺失事件不补成 Idle；未知技能不重映射为已支持类别。Hold 可保留为数据语义，但当前六类路由尚不支持它。

## 三种时间范围

| 范围 | 定义 | 用途 |
| --- | --- | --- |
| Event | 实际每臂语义区间 | 当前技能、操作对象、完整事件终点 |
| Motion | 固定 +5 到 +40 个保存主相机帧的八个查询 | Motion Planner 的二维 EEF 监督 |
| Action | 历史 LIBERO 控制配方的 10 步动作块，执行前缀 5 步 | π0.5 控制与滚动重规划 |

Motion Planner 保留最短 10 个保存帧的窗口，靠近 Event 边界时可在同一 Episode 内跨入下一段。超过有效窗口的查询取窗口终点，投影有效时以与实测点相同的权重计入二维损失，并单列为派生标签。长 Event 的 +40 点仍使用当时的实测位置。[边界与缺失标签](action_representation.md#event-边界与缺失标签)

Action 保持标准动作窗口监督，可以跨 Event；统计完整窗口与执行前缀的跨界比例，再评估是否需要调整。第一版按保存帧顺序查询，不要求物理时间戳；物理时长估算与视频播放速度分开说明。

## 双臂关系与在线边界

两臂可以操作不同对象，也可以共同操作同一对象。共享对象不自动意味着共享终点或相同技能；协作关系有明确依据时才关联。各臂的边界与监督有效性独立，Motion 通过联合注意力学习协作。

训练中用 Event 生成标签，在线由当前观察预测技能、对象与运动，不请求真实边界或剩余时长。真实标签有效性只控制损失；Action 中的 Motion 槽位只能按在线可知的臂存在性等信息屏蔽。

## 参考

[AtomSkill](ref_papers/event_organization/atomskill.md) 提供边界与技能语义分别组织的参考；[DeCo](ref_papers/event_organization/deco.md) 有助于讨论交互粒度。[AWE](ref_papers/event_organization/awe.md) 提醒语义事件与路径转弯不是同一粒度。采用这些机制不要求把每个现有事件重标为完整的物理交互。

字段与来源约定见 [Data](data.md)，其他工作见 [参考索引](ref_papers/event_organization/README.md)。
