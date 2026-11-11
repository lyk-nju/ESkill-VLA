# LIBERO40：镜像先验与动作两阶段训练

2026-11-10 已确定：Stage 1 从官方原生 π0.5 初始化训练 20k，Stage 2 接续训练 40k；先不加入 Functional Reuse。本机已完成代码测试、真实数据可视化及真实权重短训练检查；正式 20k／40k 训练尚未启动，计划在远端执行。

| 项目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 配置 | [train_eskill_stage1.json](../../examples/libero/train_eskill_stage1.json) | [train_eskill_stage2.json](../../examples/libero/train_eskill_stage2.json) |
| 初始化 | 官方 π0.5 转换权重，新增分支随机初始化 | Stage 1 `checkpoint_020000.pt`，新优化器／步数 |
| 更新参数 | Skill、Grounding、Motion、输入适配及辅助头 | 前述所有模块＋视觉编码器／视觉投影／语言主干＋Action Expert、动作／时间投影、Motion K/V |
| 冻结 | VLM、Action、VAE 编码器 | VAE 编码器 |
| 样本 | 符合条件时 50% 水平镜像，否则原图 | 仅原图，无 latent donor／recipient 复用 |
| 损失 | Skill=0.02、Current=0.01、Future=0.01、Trace=0.25、Gripper=0.05 | 保持前五项权重，加入 Action=1 |
| 预算 | 20,000 更新步 | 40,000 更新步 |
| 保存／验证 | 每 5k／每 1k | 每 5k／每 1k |

共同配置：固定 train 1,518／val 175 轨迹、仅 train 的 quantile 统计、双相机、H=10；Motion 八点每隔 5 帧，Future 画布额外每边 8 px。skill_sampling=true，平方根倒频率权重封顶 3。LoRA rank=0；FP32 参数与 BF16 autocast，每卡 batch 8，lr=1e-5、warmup=1k、betas=(0.9,0.95)、weight_decay=1e-10、clip=1。Stage 1 冻结 VLM，Stage 2 解冻全部 VLM，暂用统一学习率。

Stage 2 当前为全量 VLM 联合训练；下方真实权重 12 步短测采用了旧冻结 VLM 设置，其 19.54 GiB／卡及耗时不能作为解冻后的资源验收。已有 Stage 1 权重可通过 `init_checkpoint` 转入新 Stage 2，旧冻结 Stage 2 不能直接换配置 `--resume`。当前未在本机运行解冻后的真实权重训练或正式八卡测评。

解冻实现核验：60 项针对性 CPU 回归通过。微型模型验证 FM 与辅助损失均能回传视觉编码器／视觉投影／语言主干，VAE 保持冻结；Stage 1→解冻 Stage 2 严格继承权重并重新建立训练状态，VLM 权重实际更新，恢复与连续训练一致。包含梯度重算、Motion 梯度诊断及两进程 DDP 与整批参考梯度等价检查；冻结 VLM 对照仍通过。这些检查不代表真实大模型资源占用或闭环收益。

Stage 2 沿用 Action 视觉 K/V patch mask：各相机概率 0.3，20% 样本完全无遮蔽。Stage 1 不执行动作条件或 patch mask。所有验证使用原图及完整视觉；不自动启动闭环测评。

远端配置需调整资源路径和卡数，全局 batch=每卡 batch×卡数；示例八卡对应 64。上述学习率、权重和 mask 比例是首轮配方，未宣称最优。启动命令见 [LIBERO 示例](../../examples/libero/README.md#两阶段镜像训练不带复用)。正式启动时再记录机器、GPU、全局 batch、Git 状态、命令、PID、W&B 和输出目录。

当前镜像作用于两路 RGB、Current／Future 画布和权重、未来二维点；技能、臂身份和夹爪命令不变。所有阶段均只使用 RGB＋Language，无本体／起点输入或状态门控。方向词任务只保留原图。Stage 1 不产生镜像 Action loss。既有[核验视频](../../data/libero/atomic-vla/check/test/training/mirror_videos.md)中的状态门控与起点展示属于历史实现，待用户验收。

技术检查完成：76 项 CPU 回归通过，覆盖镜像双翻转、方向词排除、参数冻结、两进程梯度等价、断点恢复和阶段权重交接；十条真实轨迹的视频均已完整解码。测试仅用微型模型，不代表官方大模型训练收敛或泛化收益。

此前本机短测的 Stage 1 20 步产生于新增镜像状态门控之前；该次 Stage 2 12 步采用了现已撤销的冻结先验／仅 FM 配方，不能作为当前 Stage 2 的验收结果。原始事实与产物保留在[历史短测记录](local_check.md)。当前 Stage 2 改为六项损失联合训练，检查状态与结果见[联合训练短测](local_check_joint.md)。

修正后的联合训练短测已通过：76 项 CPU 回归、四卡真实权重 Stage 2 12 步及恢复、先验／动作权重更新与 VLM／VAE 冻结核对、策略回读均正常；峰值 allocated 19.54 GiB／卡。当前正式 20k／40k 训练仍未启动。

历史镜像状态门控曾在镜像行关闭8D投影；当前实现已彻底删除状态分支和门控。上述真实权重短测仅证明当时实现可运行，不作为当前 RGB＋Language 架构的完整验收。当前结构需重新训练 Stage 1，不能直接使用带状态分支的旧 checkpoint。

状态门控检查完成：91 项相关 CPU 回归通过，另核对十条真实样本及[状态门控对照图](../../data/libero/atomic-vla/check/test/training/state_mask/preview.jpg)。此次未追加正式长训练。

Motion 最短监督窗口更新为 10 个保存帧：前两个查询点可跨 Event，Episode 尾部不填充；名义约 0.5 秒，不改变 +5…40 八点布局。此前真实权重短测使用严格 Event 标签，不作为新窗口的完整训练验收；新窗口以全量覆盖率重算、独立回归和视频检查确认链路。

最短窗口核验完成：全量零有效二维点7.72%→3.74%，96项相关回归通过，十条新规则视频供人工检查，见[监督覆盖率](../../data/libero/atomic-vla/check/test/training/minimum_motion.md)。

Stage 2 使用 `motion_grad_interval=100`，在第1步及每100步记录 FM／辅助损失对共享 Motion 的实际加权梯度范数、比例、余弦和所用 `loss_weights`。范围排除轨迹／夹爪读出头，多卡先平均向量，仅写本地 `motion_gradients`。新增诊断不代表已有长期梯度趋势，详见[训练接口](../../src/openpi/training/README.md#motion-梯度诊断)。

梯度诊断与阶段恢复核验：117 项相关回归通过，覆盖 BF16／梯度重算下诊断不改变正常训练梯度和 RNG、双进程全局梯度等价、启用诊断后恢复训练与关闭诊断连续训练的权重一致，以及 stage1／stage2／joint 反复切换时恢复原始 VLM／LoRA 范围并保留 VAE 冻结。阶段改变后须重建优化器与 DDP。

真实权重八批次梯度探针未完成：四张 GPU 被另一组 LIBERO 测评占用，加载模型时 CUDA OOM，没有产生梯度指标或更新权重。失败记录在 `/data1/yuankai/robot-research/workspace/reports/eskill_motion_gradients/probe_002/`；不能据此判断 FM 与辅助梯度的长期比例。

## 损失权重依据

两阶段统一使用 Skill=0.02、Current=0.01、Future=0.01、Trace=0.25、Gripper=0.05。第一阶段按此配比直接学习先验，第二阶段保持辅助监督配比并加入权重为 1 的 FM，供 FM 继续修正先验。Stage 2 全部分支仍联合训练，不通过冻结来实现此调整。Stage 1 的梯度总尺度和裁剪频率可能随新系数改变，旧单位权重短测不能作为当前配方的收敛验收。没有启动新的正式训练。

核对的是具体开源配方，不把它们概括成通用最佳范围：

| 工作 | 公开系数 | 与本项目的关系 |
| --- | --- | --- |
| [GuidedVLA LIBERO](https://github.com/GuidedVLA/GuidedVLA/blob/main/src/openpi/training/config.py) | Object=0.001、Skill=0.001 | attention 区域监督／soft skill KL，与本项目 latent MSE／技能 CE 不等价 |
| [AtomicVLA](https://github.com/zhanglk9/AtomicVLA/blob/main/src/openpi/models/pi0_atomic.py#L269-L278) | FM=1、Decision=0.05、Text=0.02 | 决策 token CE／平均文本 CE，没有本项目的轨迹与 Grounding 目标 |
| [AffordanceVLA Stage 3 LIBERO](https://github.com/Skywalker-yqz/AffordanceVLA/blob/main/configs/stage3%20Libero.yaml) | 在线 main：Which／Where／Shape=0.0441、Layout=0.0176 | 多种视觉／几何目标；不同目标和版本不能逐项照抄 |

AffordanceVLA 本地未修改版本 `3bb77f1a11a05781427458575c6873c147a8a0ab` 的 Stage 3 对应系数为 0.015／0.015／0.015／0.006，与本次在线 main 不同；本记录明确区分版本，不将本地值冒充在线最新配方。其 Stage 1 Which／Where 为 0.1，同样不采用各阶段统一权重。

我们的首轮系数根据监督类型和[既有 12 步短测](local_check_joint.md)的未加权量级选择，粗略让单项辅助加权值落在 0.03–0.04；Trace 是 `[0,1]` 坐标上的 SmoothL1（beta=0.05），因此系数高于 VAE latent MSE。该尺度校准是工程启发式，不证明梯度平衡或最佳权重。

| 损失 | 历史短测原始值 | 新权重 | 代入后的数值 |
| --- | --- | --- | --- |
| Action | 0.065382 | 1 | 0.065382 |
| Skill | 2.025982 | 0.02 | 0.040520 |
| Current | 4.093506 | 0.01 | 0.040935 |
| Future | 3.229530 | 0.01 | 0.032295 |
| Trace | 0.127824 | 0.25 | 0.031956 |
| Gripper | 0.721018 | 0.05 | 0.036051 |

表中仅代入旧日志，没有重训：辅助加权和约 0.181757，FM 约 0.065382；这些是标量值，不能解释成梯度或更新占比。旧权重只训练了少量 warmup 步，标签仍为严格 Event，当前最短窗口与后续收敛阶段的量级可能不同。后续看实际加权梯度趋势、各辅助目标质量及闭环成功率再决定调整；当前不加自动动态加权或额外调度。

此前非单位权重核验：45 项针对性回归通过，覆盖加权梯度的解析参考、零系数、双进程归约、正常反传不受诊断影响，以及新 Stage 2 系数下的阶段交接、日志与恢复／连续训练权重一致。该轮测试尚未包含本次 Stage 1 系数调整；本次另核对两份实际示例配置可加载，五项辅助系数完全一致，Stage 1 不含 Action loss、Stage 2 的 Action 权重为 1。未使用 GPU 或启动正式训练，不宣称新权重的收敛／闭环效果。
