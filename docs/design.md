# 运动规划

Current-only KLAL 接三种 Planner。八槽结构、十槽边界提前分流和十槽联合结构共用 Current、路由和数据接口。

模型输入为 RGB 和完整任务指令。主相机的 Current 取原生文本到主图 patch 的逐层注意力，通过学习的文本权重聚合。配置可加入同帧腕相机。八个未来点默认每五帧取一点。

| planner_version | 模型 | 单目配方 |
| --- | --- | --- |
| v1 | 八个未来槽位，首次读取主图后不加原查询残差 | train_motion_planner_frozen.json |
| v2 | Start 和 End 提前解码，八个未来槽继续计算 | train_motion_planner_boundary.json |
| v2_joint | 十个槽一起完成时序混合和稀疏专家计算，再用同一个头解码 | train_motion_planner_joint_slots.json |

这些历史结构同时改变了多个计算步骤。多视角十槽配方为 `train_motion_planner_multiview.json`。初始化选择摄像头顺序匹配的 KLAL checkpoint。评测使用 `python -m scripts.analysis.motion_planner --help`。
