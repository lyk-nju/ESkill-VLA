# 统一 Episode 链路检查

目的：停止旧 ESkill 长训练，用累计 30k（SFT 20k + ESkill 10k）的已训练权重验证新版数据读取、四卡训练和策略回读。属于技术检查，不报告闭环成功率。

- 原训练停止于 ESkill 第 13,788 步，四卡进程已退出；旧数据和 checkpoint 保留。
- 复用 `eskill_libero_sft20k/run_003/train/checkpoint_010000.pt`，不使用更晚权重。
- 基准代码：`7329601`；新版为当前工作区的统一 Episode 改动。
- 数据：`/data1/yuankai/robot-research/data/atomicdata/libero40_unified/index.json`，由旧三分支全量迁移并逐条校验。
- 产物：`/data1/yuankai/robot-research/workspace/experiments/unified_episode_check/run_001/`。
- recipe：四张 RTX 3090，LoRA rank/alpha 16/16，BF16，每卡 batch 8／全局 32，双相机，horizon 10，lr 5e-5，全量轨迹，不增强，FLUX VAE。保持原优化器、RNG、步数及训练参数。
- 计划：ESkill 第 10,000 → 10,016 步。接口检查按 AGENTS.md 不上传正式 W&B 项目，配置 mode=disabled。
- 归一化：继承旧数值语义，state/actions=quantile、arm_state=standard；转换仅将方法写入各字段。新建实验的 fresh stats 默认全部 quantile，另外保存，不用于本次旧权重续训。

执行入口：

```bash
python -m scripts.preprocess.libero.migrate_unified --source /data1/yuankai/robot-research/data/atomicdata/libero40 --output /data1/yuankai/robot-research/data/atomicdata/libero40_unified --verify
python scripts/convert_episode_checkpoint.py --source /data1/yuankai/robot-research/workspace/experiments/eskill_libero_sft20k/run_003/train/checkpoint_010000.pt --data-index /data1/yuankai/robot-research/data/atomicdata/libero40_unified/index.json --output /data1/yuankai/robot-research/workspace/experiments/unified_episode_check/run_001
python experiments/unified_episode_check/check_data.py /data1/yuankai/robot-research/workspace/experiments/unified_episode_check/run_001/config.json
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py --config /data1/yuankai/robot-research/workspace/experiments/unified_episode_check/run_001/config.json --resume /data1/yuankai/robot-research/workspace/experiments/unified_episode_check/run_001/converted.pt
```

使用现有 Pi_05/openpi/.venv/bin/python，设置 `PYTHONPATH=src:.`。迁移只创建新目录，旧三分支和此前传输包不删除。GPU10 后续应搬运新的 libero40_unified 目录，其训练读取无需 external/ 或旧 pi05_sft/。

结果（2026-11-03）：

- 全量迁移及回读校验通过，1,693 条／273,465 帧，10,328,106,914 bytes；旧目录保留。
- 40 任务各取首／中／末样本，共 120 个；新旧图像、状态、动作及 ESkill 各监督目标一致。π0.5 无辅助目标。两种模式的新统计均生成成功。
- 718 个权重张量转换后逐项一致，归一化前后等价；旧 arm_state=standard 有意保留。
- 四卡 LoRA／全局 batch 32，10,000 → 10,016 步完成，退出码 0。平均 2.45 秒／步；首步 loss=0.0955084，末步=0.1029074，全部损失和梯度有限。四卡最终参数哈希相同，并与初始哈希不同。
- 与旧链路的对应 16 步相比，样本完全一致、首步各损失完全一致；后续总损失最大差 0.000559，不宣称逐位训练复现。
- 新 checkpoint 经 LIBERO 测评输入适配器，对三条真实轨迹执行 10 步 Flow Matching，均输出有限的 `[10,7]` 动作，推理未加载训练 VAE。未运行仿真闭环成功率测评。
- 相关自动测试 70 passed、1 skipped（沙箱本地 IPC 权限）；实际四卡多进程加载、训练及保存已成功。
- 首批前约四分钟包含权重加载和恢复数据迭代位置；训练步耗时不包含这部分。

产物目录含 `conversion.json`、`migration.json`、`data_check.json`、`training_summary.json`、`policy_check.json`、`runtime.json`，以及 `train/checkpoint.pt` 和训练日志。此检查已结束，未启动新的长训练；W&B 按技术检查规范关闭。
