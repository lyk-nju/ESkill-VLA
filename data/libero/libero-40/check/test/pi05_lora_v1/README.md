# 原版 π0.5：LIBERO-40 原始字段四卡 LoRA 对照

2026-11-01：**短程训练接口测试通过**。本组读取 `openvla/modified_libero_rlds` 的原始主相机、状态与动作缓存；不是使用本目录官方 HDF5 重新收集的全量训练。

与处理后的 AtomicVLA 组采用同一完整预训练 π0.5、LoRA rank=16、四张 RTX 3090、BF16、每卡 batch=1；40 任务各选一条，抽 120 个窗口。32 步后恢复至 40 步，loss／梯度、rank 权重同步及 10 个样本的独立推理均通过。

两组只更换读取的数据格式，不启用 ESkill 分支。测试不提供训练收敛或任务成功率结论。来源固定修订号 `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`，轨迹与原文件校验信息保存在实验清单。

[完整协议、两组结果及复现入口](/home/yuankai/robot-research/workspace/experiment/pi05_libero_lora/README.md) · [汇总表](/home/yuankai/robot-research/workspace/experiment/pi05_libero_lora/results/summary.csv) · [本组 checkpoint](/data1/yuankai/robot-research/workspace/experiments/pi05_libero_lora/run_001/source/checkpoint.pt)。
