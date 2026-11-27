# 动作接口

固定 Current KLAL 和十槽联合 Planner，分别对照原生 Action、Motion K/V、Current 先验、Motion 先验和双先验。

模型输入为 RGB 和完整任务指令。主相机的 Current 取原生文本到主图 patch 的逐层注意力，通过学习的文本权重聚合。配置可加入同帧腕相机。八个未来点默认每五帧取一点。

See 读取主图 Current KLAL。Plan 对所有有效视角和任务编码，使用 Current 指引读取主图 memory。Start、End 和八个未来点的十个槽经过一次联合时序混合和同一路由专家。Act 保留原生前缀 K/V 并加入最终十个槽的 K/V。对每个头、每个动作步，原生主图注意力的 70% 保留，Current 和轨迹先验各按原主图质量分配 15%。腕图、文本和其他位置保持其原生概率。轨迹先验由八个预测点在主图网格上叠加，每个有效点使用标准差为一个 patch 的高斯核。有效点缺失时，该部分质量留给原生注意力。

配方依次是 `train_visual_primitive_multiview.json`、`train_motion_planner_multiview.json`、`train_action_execution.json`。各阶段的 `init_checkpoint` 必须指向实际训练产物，动作配方默认冻结 See、Plan 和 VLM，只训练动作专家与 K/V 投影。要联合微调，显式设置 `model.train_current`、`model.train_planner`、对应辅助损失权重和 `vlm_finetune`。本地策略用 `create_trained_policy`，LIBERO 服务用 `python -m scripts.serve_policy --runtime local --checkpoint ...`。

### 动作对照

| 配方 | Motion K/V | Current / Motion |
| --- | --- | --- |
| train_action_native.json | 关闭 | 0 / 0 |
| train_action_motion_kv.json | 开启 | 0 / 0 |
| train_action_current.json | 开启 | 0.30 / 0 |
| train_action_motion.json | 开启 | 0 / 0.30 |
| train_action_execution.json | 开启 | 0.15 / 0.15 |

各组共用同一个十槽 Planner 初始化、数据和动作损失。

配对动作干预可缓存 `model.plan_features(inputs, cache_prefix=True)`。向 `condition_from_features` 传入 `current_prior` 会同时改变规划与执行的 Current，`skill_id` 指定规划专家，`motion_plan` 替换成另一份预测的十槽 latent 和二维轨迹。用 `sample_from_condition` 和同一个初始噪声比较动作。供体计划携带自己的专家选择。
