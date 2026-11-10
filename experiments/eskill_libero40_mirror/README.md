# LIBERO40：镜像先验与动作两阶段训练

2026-11-10 已确定：Stage 1 从官方原生 π0.5 初始化训练 20k，Stage 2 接续训练 40k；先不加入 Functional Reuse。本机已完成代码测试、真实数据可视化及真实权重短训练检查；正式 20k／40k 训练尚未启动，计划在远端执行。

| 项目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 配置 | [train_eskill_stage1.json](../../examples/libero/train_eskill_stage1.json) | [train_eskill_stage2.json](../../examples/libero/train_eskill_stage2.json) |
| 初始化 | 官方 π0.5 转换权重，新增分支随机初始化 | Stage 1 `checkpoint_020000.pt`，新优化器／步数 |
| 更新参数 | Skill、Grounding、Motion、输入适配及辅助头 | 前述所有模块＋Action Expert、动作／时间投影、Motion K/V |
| 冻结 | VLM、Action、VAE 编码器 | VLM、VAE 编码器 |
| 样本 | 符合条件时 50% 水平镜像，否则原图 | 仅原图，无 latent donor／recipient 复用 |
| 损失 | Skill CE、Current／Future VAE MSE、XY、Gripper，各权重 1 | 保留前五项并加入 Action FM，六项各权重 1 |
| 预算 | 20,000 更新步 | 40,000 更新步 |
| 保存／验证 | 每 5k／每 1k | 每 5k／每 1k |

共同配置：固定 train 1,518／val 175 轨迹、仅 train 的 quantile 统计、双相机、H=10；Motion 八点每隔 5 帧，Future 画布额外每边 8 px。skill_sampling=true，平方根倒频率权重封顶 3。冻结 VLM，rank 0；FP32 参数与 BF16 autocast，每卡 batch 8，lr=1e-5、warmup=1k、betas=(0.9,0.95)、weight_decay=1e-10、clip=1。

Stage 2 沿用 Action 视觉 K/V patch mask：各相机概率 0.3，20% 样本完全无遮蔽。Stage 1 不执行动作条件或 patch mask。所有验证使用原图及完整视觉；不自动启动闭环测评。

远端配置需调整资源路径和卡数，全局 batch=每卡 batch×卡数；示例八卡对应 64。上述学习率、权重和 mask 比例是首轮配方，未宣称最优。启动命令见 [LIBERO 示例](../../examples/libero/README.md#两阶段镜像训练不带复用)。正式启动时再记录机器、GPU、全局 batch、Git 状态、命令、PID、W&B 和输出目录。

核验：水平镜像作用于两路 RGB、Current／Future 画布和权重、EEF 起点与未来二维点；原始物理状态数值、技能、臂身份、夹爪命令不变；镜像样本关闭 Motion 的八维状态投影（含 bias），保留镜像后的二维起点。原图与 Stage 2 使用真实臂状态。方向词任务只保留原图。Stage 1 不产生镜像 Action loss。视频见[人工核验索引](../../data/libero/atomic-vla/check/test/training/mirror_videos.md)，待用户验收。

技术检查完成：76 项 CPU 回归通过，覆盖镜像双翻转、方向词排除、参数冻结、两进程梯度等价、断点恢复和阶段权重交接；十条真实轨迹的视频均已完整解码。测试仅用微型模型，不代表官方大模型训练收敛或泛化收益。

此前本机短测的 Stage 1 20 步产生于新增镜像状态门控之前；该次 Stage 2 12 步采用了现已撤销的冻结先验／仅 FM 配方，不能作为当前 Stage 2 的验收结果。原始事实与产物保留在[历史短测记录](local_check.md)。当前 Stage 2 改为六项损失联合训练，检查状态与结果见[联合训练短测](local_check_joint.md)。

修正后的联合训练短测已通过：76 项 CPU 回归、四卡真实权重 Stage 2 12 步及恢复、先验／动作权重更新与 VLM／VAE 冻结核对、策略回读均正常；峰值 allocated 19.54 GiB／卡。当前正式 20k／40k 训练仍未启动。

镜像状态门控修正：Stage 1 镜像行关闭 Motion 的八维状态投影（含 bias），原图与 Stage 2 保持启用；新语义由独立回归核验。上述历史真实权重短测没有覆盖这项新门控，保留其原始结论，不作为当前两阶段完整验收。

状态门控检查完成：91 项相关 CPU 回归通过，另核对十条真实样本及[状态门控对照图](../../data/libero/atomic-vla/check/test/training/state_mask/preview.jpg)。此次未追加正式长训练。

Motion 最短监督窗口更新为 10 个保存帧：前两个查询点可跨 Event，Episode 尾部不填充；名义约 0.5 秒，不改变 +5…40 八点布局。此前真实权重短测使用严格 Event 标签，不作为新窗口的完整训练验收；新窗口以全量覆盖率重算、独立回归和视频检查确认链路。

最短窗口核验完成：全量零有效二维点7.72%→3.74%，96项相关回归通过，十条新规则视频供人工检查，见[监督覆盖率](../../data/libero/atomic-vla/check/test/training/minimum_motion.md)。
