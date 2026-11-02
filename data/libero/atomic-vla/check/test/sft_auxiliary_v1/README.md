# SFT → 辅助联合训练：本机链路检查

2026-11-02：真实 π0.5 SFT 权重、预训练冻结 VAE、AtomicVLA 全部辅助标签已接通四卡训练、保存、恢复和 Python 策略回读。本次为短程工程检查，未进行 ESkill 闭环评测，不表示模型收敛或数据人工验收完成。

| 阶段 | 实际执行 |
| --- | --- |
| SFT 初始化 | 使用已有纯 π0.5 四卡 LoRA 的 `checkpoint_010000.pt`；本轮不重复 SFT |
| 辅助联合训练 | 四张 RTX 3090，32 步，随后恢复 4 步；新优化器从第 1 步开始 |
| 参数 | LoRA rank/alpha=16/16，原基座与 VAE 冻结；41,437,968 个可训练参数 |
| batch / 学习率 | 每卡 2，全局 8；1e-5，前 4 步 warmup；BF16 |
| 监督 | action=1，其余 skill/future/grounding/trace 各 0.1 |
| 数据 | 1,693 条、273,465 帧可采样；实际 288 个窗口覆盖全部 40 个任务 |
| 输入与动作 | 双相机 224×224，无离散状态提示，10 步动作窗；7 维填充至 32 维，完整动作 MSE |
| 视觉目标 | 仅主相机，128×128 三角色画布；冻结预训练 AutoencoderKL |
| 耗时与显存 | 中位每步 0.803 秒；训练／恢复峰值 8.34 GiB/卡，不含初始化耗时 |

动作、状态沿用 SFT 的分位数归一化；新 arm_state 使用训练集均值方差。动作尾部仍重复末条动作，辅助标签的 padding 独立屏蔽。此配置关闭随机图像增强，保持原始图像与区域目标位置一致。

| 检查 | 结果 |
| --- | --- |
| SFT 权重转移 | 504 个 LoRA 张量加载后逐值相等；新增 step_projection 初始为零 |
| 数据配对 | 40 任务 × 首／中／末窗口，共 120 个；图像、状态、动作一致 |
| 完整分支反传 | 五项损失及梯度有限；技能、视觉、运动、动作条件分支均有梯度；冻结参数无梯度 |
| DDP | 32 步、36 步均四卡可训练参数哈希一致 |
| 断点恢复 | 恢复初始参数等于第 32 步保存值；优化器状态为第 36 步；继续更新 504 个 LoRA 张量 |
| 策略回读 | 三个真实观察均输出有限 `[10,7]` 动作；无 VAE，右臂技能保持无效 |
| 自动回归 | 50 项通过，包含初始化完整性、归一化、数据拼接、恢复、DDP 和离线 W&B |

以下损失来自不同 batch，仅用于确认数值与更新正常：

| 损失 | 第 1 步 | 第 32 步 | 第 36 步 |
| --- | ---: | ---: | ---: |
| Action | 0.05750 | 0.03489 | 0.03352 |
| Current skill | 2.09601 | 1.61158 | 1.86983 |
| Future skill | 3.37498 | 1.31826 | 2.11407 |
| Grounding | 3.58798 | 2.61707 | 2.54762 |
| Motion | 0.94174 | 0.08980 | 0.06482 |

W&B run：[`pi05_eskill_lora_bs8_libero40_sft10k_aux32_check_seed42`](https://wandb.ai/laiyuankia-nanjing-university/EskillVLA/runs/q4vjkype)。名称中的 aux32 是初始检查步数；随后同 run 恢复至 36。正式记录使用项目根目录的共享配置，保存产物不含密钥。

- 可复用入口：[train_sft_auxiliary.json](../../../../../../examples/libero/train_sft_auxiliary.json)，[命令与语义](../../../../../../examples/libero/README.md#从-sft-开始辅助联合训练)。
- 本轮配置：[train.json](train.json)、[resume.json](resume.json)；证明记录：[preflight.json](preflight.json)、[report.json](report.json)、[training_32.json](training_32.json)、[training_36.json](training_36.json)。
- 最终 checkpoint：`/data1/yuankai/robot-research/workspace/experiments/sft_auxiliary_v1/train/checkpoint.pt`；同目录保留 `checkpoint_000032.pt`、逐步日志和无密钥配置。
- VAE：[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)，版本 `31f26fdeee1355a5c34592e401dd41e45d25a493`；本地目录 `/data1/yuankai/robot-research/ckpt/vae/sd-vae-ft-mse/` 的 `source.json` 保存文件哈希。

实现借鉴 GuidedVLA 的“加载已有策略权重，再接入辅助分支”顺序。这里使用本项目的技能／视觉／运动分支；`init_checkpoint` 转移权重，`--resume` 恢复已有实验。后者与策略回读不再读取原 SFT checkpoint。当前 ESkill 的 websocket 服务观察映射及闭环评测仍待接入，既有纯 π0.5 服务继续可用。
