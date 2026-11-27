# 运动规划

Current-only KLAL 接三种 Planner。轨迹规划、边界优先和联合解码共用 Current、路由和数据接口。

模型输入为 RGB 和完整任务指令。主相机的 Current 取原生文本到主图 patch 的逐层注意力，通过学习的文本权重聚合。配置可加入同帧腕相机。八个未来点默认每五帧取一点。

| 架构名 | planner_version | 模型 | 单目配方 |
| --- | --- | --- | --- |
| 轨迹规划 | `trajectory` | 八个未来槽位，首次读取主图后不加原查询残差 | train_motion_planner_frozen.json |
| 边界优先 | `boundary_first` | Start 和 End 提前解码，八个未来槽继续计算 | train_motion_planner_boundary.json |
| 联合解码 | `joint_decode` | 十个槽一起完成时序混合和稀疏专家计算，再用同一个头解码 | train_motion_planner_joint_slots.json |

这些历史结构同时改变了多个计算步骤。多视角十槽配方为 `train_motion_planner_multiview.json`。初始化选择摄像头顺序匹配的 KLAL checkpoint。评测使用 `python -m scripts.analysis.motion_planner --help`。

## Uniform α

[Visual Primitive 配方](../examples/libero/train_visual_primitive_multiview_uniform.json)用均匀文本权重训练 Current KLAL。[Planner 配方](../examples/libero/train_motion_planner_multiview_uniform.json)继承其 10k checkpoint，使用每个有效 token 等权的 Current 热图和 Skill Router 文本输入。聚合器有 0 个参数。这里沿用既有的双视角联合解码设置，使用独立的输出目录。单视角和其他 Planner 结构也支持 `model.learned_alpha=false`，初始化时要求上游 checkpoint 使用相同的聚合方式。
