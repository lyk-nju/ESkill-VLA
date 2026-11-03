# ESkill LIBERO 服务与精度校验

2026-11-03：修复 `serve_local` 拒绝 ESkill 的限制，沿用现有 WebSocket 服务和 LIBERO／LIBERO-Plus 共用客户端。

| checkpoint | 客户端观测与服务适配 |
| --- | --- |
| `pi05/libero_sft` | 主／腕 RGB、源 8 维 EEF／夹爪状态 |
| `eskill/episodes` | 按 `camera` 选择 RGB；带 `sft_data_index` 时再加入腕部 RGB；同帧真实关节位置 7 维＋源状态 8 维 |

服务握手公布 `cameras`、`requires_joint_state` 和模型 `state_dim`；客户端从 `robot0_joint_pos` 读取真实值，以 `observation/joint_state` 发送。客户端与服务端均拒绝缺失、错误维度或非有限关节值。已有训练归一化由 checkpoint 恢复，在线输入不含事件标签或未来信息。

精度校验在训练入口、训练模型构造入口执行：ESkill 的 BF16 基座必须配合 BF16 autocast；FP32 训练应使用 FP32 基座。推理继续使用 Policy 自身的 autocast 规则。

相关测试：30 项通过，1 项跳过。覆盖原有 SFT 协议、ESkill 单／双相机、真实关节顺序、缺失值拒绝、rollout 逐次更新关节状态、SFT 初始化与恢复，以及三种受支持精度组合的前向／反向。错误精度组合在读取数据或初始化多卡前即被拒绝。

真实闭环使用已有 `task_split_v1/smoke/auxiliary_full/checkpoint.pt`：tiny CPU ESkill、随机测试规模 VAE 训练所得的第 3 步 checkpoint，加载推理时不创建 VAE。它不是预训练 π0.5 的正式效果检查。

- LIBERO Spatial 任务 0、官方初态 0，等待 10 步。
- 主／腕双相机，真实七维关节位置，源八维状态；服务拼接为十五维。
- Flow Matching 2 次迭代，每次执行动作块前 5 步。
- 共执行 6 步、请求模型 2 次，错误数 0；技术模式 `success_rate=null`。
- 启动命令见 [commands.json](commands.json)，逐回合记录见 [run/records/libero_spatial_t0000_r000.json](run/records/libero_spatial_t0000_r000.json)，汇总见 [run/summary.json](run/summary.json)。

服务日志中的一次 HTTP 握手报错来自测试脚本用裸 TCP 探测端口就绪；真实客户端的 WebSocket 推理正常。核验后另将评测配置的 `state`（保存的 8 维记录）与 `policy_state`（模型实际输入）分开命名，已有这次运行记录保持原样。
