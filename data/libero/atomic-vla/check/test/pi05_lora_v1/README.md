# 原版 π0.5：AtomicVLA 四卡 LoRA 检查

2026-11-01：**短程训练接口测试通过**。使用完整预训练原版 π0.5，冻结基座、LoRA rank=16、四张 RTX 3090、BF16、每卡 batch=1；未启用 ESkill 分支。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| 数据读取 | 通过 | 处理后的 JSON＋NPZ；40 任务各一条、共 120 个动作窗口 |
| 四卡训练与恢复 | 通过 | 32 步后恢复至 40 步，loss／梯度有限，四卡权重同步 |
| 独立推理 | 通过 | 重新加载基座与 LoRA，10 个真实样本均输出有限的 `[50,7]` 动作 |
| 原始字段对照 | 一致 | 与同源原始 RLDS 缓存的模型输入、动作及 mask 逐项一致 |

每卡峰值 PyTorch 分配显存 7.85 GiB。只使用主相机、源状态和动作，不使用新增事件／bbox 标签。本轮不是全量训练收敛或 LIBERO 任务成功率测试，既有来源质量警示和人工验收状态保持原义。

[完整协议、两组结果及复现入口](/home/yuankai/robot-research/workspace/experiment/pi05_libero_lora/README.md) · [汇总表](/home/yuankai/robot-research/workspace/experiment/pi05_libero_lora/results/summary.csv) · [本组 checkpoint](/data1/yuankai/robot-research/workspace/experiments/pi05_libero_lora/run_001/atomic/checkpoint.pt)。
