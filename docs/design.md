# 动作接口

固定 Current KLAL 和十槽联合 Planner，比较 Motion K/V、空间先验与任务文本直通对动作的影响。

模型输入为 RGB 和完整任务指令。主相机的 Current 取原生文本到主图 patch 的逐层注意力，通过学习的文本权重聚合。配置可加入同帧腕相机。八个未来点默认每五帧取一点。

See 与 Plan 读取完整任务和当前各视角。Start、End 和八个未来槽经过联合时序混合和同一路由专家。Act 读取融合后的主图、有效腕图、十个最终 Motion 槽和动作序列。直接任务文本 K/V 由执行器屏蔽，RoPE 位置仍按完整有效前缀计算。视觉和 Motion 特征可携带上游任务语义。

在主图上，每个 patch 到十个有效预测位置的最近距离决定运动先验，距离以 patch 为单位，高斯标准差为 1，最后在整张网格归一化。重复终点只产生一个空间峰。对每个头和动作步，Current 和 Motion 各分配原生主图注意力质量的 15%，其余 70% 沿用原生分布。十点全无效时，Motion 份额回到原生分布。

配方依次是 `train_visual_primitive_multiview.json`、`train_motion_planner_multiview.json`、`train_action_execution.json`。各阶段的 `init_checkpoint` 必须指向实际训练产物，动作配方默认冻结 See、Plan 和 VLM，只训练动作专家与 K/V 投影。要联合微调，显式设置 `model.train_current`、`model.train_planner`、对应辅助损失权重和 `vlm_finetune`。本地策略用 `create_trained_policy`，LIBERO 服务用 `python -m scripts.serve_policy --runtime local --checkpoint ...`。

### 动作对照

| 配方 | Motion K/V | 直接 Text K/V | Current / Motion |
| --- | --- | --- | --- |
| train_action_native.json | 关闭 | 开启 | 0 / 0 |
| train_action_motion_kv.json | 开启 | 屏蔽 | 0 / 0 |
| train_action_current.json | 开启 | 屏蔽 | 0.30 / 0 |
| train_action_motion.json | 开启 | 屏蔽 | 0 / 0.30 |
| train_action_execution.json | 开启 | 屏蔽 | 0.15 / 0.15 |
| train_action_execution_text.json | 开启 | 开启 | 0.15 / 0.15 |

各组共用同一个十槽 Planner 初始化、数据和动作损失。最后两行只改变文本可见性，十个 Motion K 的旋转位置和动作序列的起始位置一致。原生 π0.5 对照保留文本。旧权重加载时使用训练当时的可见性与八点高斯混合先验。

配对动作干预可缓存 `model.plan_features(inputs, cache_prefix=True)`。向 `condition_from_features` 传入 `current_prior` 会同时改变规划与执行的 Current，`skill_id` 指定规划专家，`motion_plan` 替换成另一份预测的十槽 latent 和二维轨迹。用 `sample_from_condition` 和同一个初始噪声比较动作。供体计划携带自己的专家选择。
