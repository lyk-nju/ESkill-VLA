# 原版 π0.5：10k 步 LoRA SFT 与 LIBERO 测评

用户已确认纯 π0.5、四卡 LoRA；不加入 ESkill 设计。数据范围为现有 1,693 条示范对应的全部源主／腕 RGB、任务与动作，不使用事件、bbox 或路径标签。

训练配置：horizon=10、全局batch32、rank16、BF16、10,000个优化步。完成后测评四套40任务、每任务5个官方初态，共200回合，保存主视角视频与动作记录。

双相机数据回链、137项回归、16步四卡测试及真实模型四套技术闭环均已通过。正式训练与测评当前状态以[实验记录](/home/yuankai/robot-research/workspace/experiment/pi05_libero_sft/README.md)为准；技术检查不计入成功率。

[实际训练配置](/home/yuankai/robot-research/workspace/experiment/pi05_libero_sft/runs/run_001/metadata/train.json) · [日志](/home/yuankai/robot-research/workspace/experiment/pi05_libero_sft/runs/run_001/logs/) · [运行产物](/data1/yuankai/robot-research/workspace/experiments/pi05_libero_sft/run_001/)。
