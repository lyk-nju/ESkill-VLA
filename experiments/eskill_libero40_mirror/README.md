# LIBERO40：镜像先验与动作两阶段训练

2026-11-10 已确定：Stage 1 从官方原生 π0.5 初始化训练 20k，Stage 2 接续训练 40k；先不加入 Functional Reuse。本机仅做代码测试与真实数据可视化，正式训练尚未启动，计划在远端执行。

| 项目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 配置 | [train_eskill_stage1.json](../../examples/libero/train_eskill_stage1.json) | [train_eskill_stage2.json](../../examples/libero/train_eskill_stage2.json) |
| 初始化 | 官方 π0.5 转换权重，新增分支随机初始化 | Stage 1 `checkpoint_020000.pt`，新优化器／步数 |
| 更新参数 | Skill、Grounding、Motion、输入适配及辅助头 | Action Expert、动作／时间投影、Motion K/V |
| 冻结 | VLM、Action、VAE | VLM、Skill、Grounding、Motion 及其适配与读出 |
| 样本 | 符合条件时 50% 水平镜像，否则原图 | 仅原图，无 latent donor／recipient 复用 |
| 损失 | Skill CE、Current／Future VAE MSE、XY、Gripper，各权重 1 | Action FM，权重 1 |
| 预算 | 20,000 更新步 | 40,000 更新步 |
| 保存／验证 | 每 5k／每 1k | 每 5k／每 1k |

共同配置：固定 train 1,518／val 175 轨迹、仅 train 的 quantile 统计、双相机、H=10；Motion 八点每隔 5 帧，Future 画布额外每边 8 px。skill_sampling=true，平方根倒频率权重封顶 3。冻结 VLM，rank 0；FP32 参数与 BF16 autocast，每卡 batch 8，lr=1e-5、warmup=1k、betas=(0.9,0.95)、weight_decay=1e-10、clip=1。

Stage 2 沿用 Action 视觉 K/V patch mask：各相机概率 0.3，20% 样本完全无遮蔽。Stage 1 不执行动作条件或 patch mask。所有验证使用原图及完整视觉；不自动启动闭环测评。

远端配置需调整资源路径和卡数，全局 batch=每卡 batch×卡数；示例八卡对应 64。上述学习率、权重和 mask 比例是首轮配方，未宣称最优。启动命令见 [LIBERO 示例](../../examples/libero/README.md#两阶段镜像训练不带复用)。正式启动时再记录机器、GPU、全局 batch、Git 状态、命令、PID、W&B 和输出目录。

核验：水平镜像作用于两路 RGB、Current／Future 画布和权重、EEF 起点与未来二维点；物理状态、技能、臂身份、夹爪命令不变。方向词任务只保留原图。Stage 1 不产生镜像 Action loss。视频见[人工核验索引](../../data/libero/atomic-vla/check/test/training/mirror_videos.md)，待用户验收。

技术检查完成：76 项 CPU 回归通过，覆盖镜像双翻转、方向词排除、参数冻结、两进程梯度等价、断点恢复和阶段权重交接；十条真实轨迹的视频均已完整解码。测试仅用微型模型，不代表官方大模型训练收敛或泛化收益。
