# FLUX 潜变量回归训练检查

2026-11-02：新 Visual Grounding 已完成真实 π0.5 LoRA **四卡 8 步训练＋恢复至 10 步＋无 VAE 策略回读**。[完整报告](report.json) · [梯度／恢复／推理核验](verification.json) · [预览](index.html) · [视频](../video/grounding_mse_v1/prediction_probe.mp4)。

| 项目 | 结果 |
| --- | --- |
| 初始化 | 已完成的纯 π0.5 LoRA SFT 10k checkpoint；LoRA rank/alpha=16/16 |
| 数据 | 十条带主相机标定的派生 Episode，与双相机 SFT 缓存配对；八点二维 Motion 标签 |
| Visual Grounding | 每角色 16×256 Token；共享 LayerNorm → 256→512→1024 MLP、GELU；4×4 分块拼成 16×32×32 latent |
| VAE | 已核验的冻结 FLUX 权重，244 个参数严格匹配；FP32、eval、mode；无梯度 |
| 训练 | 每卡 batch=1、全局 4；学习率 1e-5；五项损失与梯度均有限；39,927,051 个可训练参数 |
| DDP | 四卡可训练参数哈希一致；每卡峰值分配显存约 8.62 GB |
| 恢复 | 第 8 步参数与恢复初值一致；优化器恢复到第 10 步，新潜变量头继续更新 |
| 梯度 | Query、潜变量头、未来技能、Motion、动作专家 LoRA 均有非零有限梯度；基础动作输出层按 LoRA 配方冻结 |
| 策略回读 | 不加载 VAE／潜变量头；输出 10×7 动作，与同 checkpoint 的训练模型推理最大差为 0 |
| 自动测试 | 83 项相关测试通过，含不均匀／空标签的 DDP 全局均值、混合精度、保存恢复、SFT 初始化与 W&B 离线记录 |

真实权重检查采用普通训练模式，`smoke=false`。本次只验证接口和数值稳定性，**10 步预测没有收敛**，预览中第三列仍为早期预测；没有测量 ESkill 闭环成功率，人工验收 pending。全量 1,693 条的二维 Motion 标定仍需补充，当前检查使用十条带标定来源。

训练起步权重为 action=1，skill／future／grounding／trace 各 0.1；视觉前景=1、背景=0.1，三角色等权。`grounding_ee/obj/goal` 及各自 `_fg`／`_bg` 指标均写入日志和 W&B，未重复计入总损失。第一步／第十步的 grounding 损失为 11.7135／11.3869；批次不同，这不是收敛比较。

W&B run：[`pi05_eskill_lora_bs4_libero40_sft10k_flux_mse8_motion8x4_check_seed42`](https://wandb.ai/laiyuankia-nanjing-university/EskillVLA/runs/gfg4gtet)。恢复沿用相同 run ID，指标记录至第 10 步。

## 文件与入口

- 当前配置：[train_grounding_check.json](../../../../../../examples/libero/train_grounding_check.json)。本次实际执行配置：[8 步](train_config.json)／[恢复至 10 步](resume_config.json)。
- 权重与逐步日志：`/data1/yuankai/robot-research/workspace/experiments/grounding_mse_v1/train/`；`checkpoint_000008.pt` 保留第 8 步，`checkpoint.pt` 为第 10 步，`metrics.jsonl` 包含连续的 1–10 步。
- 参数一致性：[第 8 步](training_check_8.json)／[第 10 步](training_check_10.json)。梯度和策略输出核验见 `verification.json`。
- 预览三列为目标画布、目标 VAE 重建、模型预测 latent 的解码；三行分别为当前夹爪、当前对象、Event 终点对象。视频将同一检查图显示三秒，供查看接口产物。

实现与使用方法见 [Target Grounding](../../../../../../docs/target_grounding.md) 和 [LIBERO 示例](../../../../../../examples/libero/README.md#从-sft-开始辅助联合训练)。新实验采用新的输出目录；旧去噪辅助 checkpoint 的结构不同，从纯 SFT 初始化新分支。
