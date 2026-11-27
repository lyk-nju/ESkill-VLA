# Event：离线阶段与当前交互目标

Event 是数据监督单元，不是模型在线输入。每个执行臂维护自己的事件流，时间区间采用 `[start_frame, stop_frame)`；同臂不重叠，不同臂可以并行。当前 Grounding 专项只启用 LIBERO 单臂槽位。

## 实体绑定

`scene_entities` 集中保存具体实例、部件和目标区；每个 Event 的 `interaction_target.entity_id` 引用其中一个实体，另存候选、绑定依据和审核状态。不能仅凭同类名称区分多只碗或多个柜子，无法唯一绑定时保留不确定性。

| Event | 当前交互目标 |
| --- | --- |
| Pick | 被抓取实例 |
| Place | 放置目的实例，或明确的原生空间 Region |
| Open／Close／Turn | 对应抽屉、门、旋钮等部件 |
| Push | 第一版指被推动物体 |

例如同一条 `put milk in basket` 轨迹，Pick 的 Current 是 milk，Place 的 Current 是 basket。标签在原 Event 边界切换，不根据在线预测重新划分来源事件。明确的“盘子右侧”或“收纳盒后格”保留原生目标区，不改成整个承托物。

Subject 由当前帧夹爪框确定，不跟随目标文本替换。反事实任务仅在语义明确、源／目的绑定一致时改写；标签依照实际观察与替换任务确定，不能把所有新任务都解释为原 Place 目标。

## 数据事实与能力边界

原始技能标签、边界修订、操作对象和交互目标分别保存。`boundary_valid`、绑定状态与人工审核不是同一个概念；自动 resolved 不等于人工通过。Event 可以只覆盖阶段动作，不强行补造完整释放或成功终态。

来源 `object_regions` 表示操作对象，不等同于专项 Current；`goal_frame` 供历史终点监督使用，不参与当前视觉专项。原区间、实例与 RGB 保留，训练标签由读取器派生。

模型只接收图像和完整任务，需要自己推断当前应关注什么。Event 不控制在线选择，也不提供额外阶段分类输入。当前表征尚未证明具备错误物体检测或恢复动作能力。

字段见 [数据需求](data.md)，标签画布见 [Grounding](target_grounding.md)，历史 Motion 时间窗与双臂计划见 [历史动作表征](history/action_representation.md)。
