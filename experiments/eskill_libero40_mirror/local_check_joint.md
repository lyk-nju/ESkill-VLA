# Stage 2 联合训练：本机真实权重检查

2026-11-10，用户明确 Stage 2 也应训练 Skill／Grounding／Motion。本次修正为：冻结 VLM 与 VAE 编码器，联合更新先验、输入适配、辅助读出、Action Expert、动作／时间投影和 Motion K/V；保留五项辅助损失并加入 Action FM。Stage 2 使用原图，不做 Functional Reuse。

复用[此前检查](local_check.md)中已经验证的 Stage 1 20 步技术权重，单独重跑修正后的 Stage 2。正式 Stage 1 20k／Stage 2 40k 预算不变。

| 项目 | 本次检查 |
| --- | --- |
| 初始化 | `local_check_001/stage1/train/checkpoint.pt`，20 步；继承全部模块与视觉读出头，重建优化器 |
| 硬件 | GPU 0–3，4×RTX 3090 24GB；启动前空闲 |
| 数据 | 原 LIBERO40 train 1,518／val 175，沿用原 train-only quantile 统计 |
| 范围 | 完整模型、真实官方 π0.5 与 FLUX VAE；FP32 参数、BF16 autocast，`smoke=false` |
| Batch／预算 | 每卡 1、全局 4；先 8 步，再恢复至 12 步 |
| 损失 | Action、Skill、Current、Future Object、Trace、Gripper，六项各权重 1 |
| 学习率 | 1e-5，保留 warmup 1k |
| 验证／保存 | 每 8 步及结束；每次固定 4 个验证样本 |
| W&B | 技术检查关闭，仅本地日志 |

产物目录：[local_check_002](../../../workspace/experiment/eskill_libero40_mirror/local_check_002/)。包含精确配置、源码差异、启动命令／PID、状态、日志与 checkpoint。基础提交为 `af6fb09`，实际训练代码为目录内 `source.patch` 所记录的修正后工作区。

状态：**检查完成，全部通过**。76 项相关 CPU 回归通过，包含单独 FM 回传至连续先验、辅助监督梯度、VLM／VAE 冻结、两进程全局梯度等价、阶段交接以及 Stage 2 恢复与连续训练精确一致。真实四卡已完成 12 步及恢复检查。技术检查不作为收敛或闭环效果结论。

## 真实权重结果

- 可训练参数 **491,921,961**，即原动作侧 432,466,976＋先验及辅助模块 59,454,985；冻结参数 3,007,155,091。
- PyTorch 峰值 allocated 为 **19.54 GiB／卡**，无 OOM、NaN 或 Inf。四卡最终可训练参数 SHA256 完全一致。
- 从第 8 步恢复到第 12 步，日志步数连续，六项训练与验证损失均有限。
- 与初始化的 Stage 1 checkpoint 逐项比较：VLM／VAE 的 848 个冻结张量完全相同。Skill 66、Grounding 37、Motion 53 个张量发生更新；输入适配 7、视觉读出头 6、Action Expert 200、动作／时间投影 8、Motion K/V 36 个张量更新。
- checkpoint 回读后，对一条验证轨迹的当前双相机、状态与当前二维起点执行 FM10，输出有限的 `[10,7]` 动作，推理不加载 VAE，不读取未来 RGB 或 Event 标签。

| 损失 | 最后训练步 | 最后验证 |
| --- | --- | --- |
| action | 0.065382 | 0.086526 |
| skill | 2.025982 | 2.013037 |
| current | 4.093506 | 3.914873 |
| future_object | 3.229530 | 3.147375 |
| trace | 0.127824 | 0.163581 |
| gripper | 0.721018 | 0.697807 |
| loss | 10.263243 | 10.023200 |

验证每次只有四个固定样本，训练仍处于 warmup 前段。以上只用于检查数值与梯度链路，不能据此推断训练收敛或泛化表现。正式预算仍为 Stage 1 20k＋Stage 2 40k。

本轮两个训练进程阶段合计 391.4 秒，包含模型加载及完整 checkpoint 写盘。逐项核对 2.5 秒；checkpoint 大小 17.93 GB。

产物：[运行记录](../../../workspace/experiment/eskill_libero40_mirror/local_check_002/runtime.json) · [参数／损失核对](../../../workspace/experiment/eskill_libero40_mirror/local_check_002/audit.json) · [策略回读](../../../workspace/experiment/eskill_libero40_mirror/local_check_002/policy_check.json) · [checkpoint](../../../workspace/experiment/eskill_libero40_mirror/local_check_002/stage2/train/checkpoint.pt)。同目录含 `source.patch`、配置、命令／PID 和日志，W&B 按技术检查约定关闭。

之后增加的 Stage 1 镜像状态门控不在本次短测范围内；本次使用的 Stage 1 初始化权重在镜像时仍接收本体状态。本记录验证 Stage 2 联合更新链路，不代表新门控方案的两阶段收敛验收。
