# 统一 Episode：ESkill 30k → 40k

从旧版累计 30k 权重恢复新版训练，累计 40k 时自动运行 LIBERO-40 测评。SFT 已训练 20k，因此本阶段内部步数为 10,000 → 20,000；日志与 W&B 保留阶段步数，累计步数需加 20,000。

配置：[train.json](train.json)；启动器：[run.py](run.py)。初始权重来自旧 `eskill_libero_sft20k/run_003/train/checkpoint_010000.pt` 的已验证转换副本，未使用短程检查后多训练 16 步的版本。

| 项目 | 配置 |
| --- | --- |
| 数据 | libero40，1,693 条／273,465 帧，全量训练，无独立 val |
| 初始化 | 累计 30k checkpoint；保留模型、优化器和各卡 RNG |
| GPU／batch | 4×RTX 3090，每卡 8，全局 32 |
| 方法 | π0.5 ESkill，LoRA rank/alpha 16/16，BF16 |
| 输入 | 主相机＋腕部相机；arm_state 开启；状态不拼入文本；关闭图像增强 |
| 动作 | horizon 10，repeat_last，沿用原完整动作 MSE |
| 优化 | lr 5e-5，betas=(0.9,0.95)，weight decay=1e-10；延续原调度 |
| 归一化 | state/actions=quantile，arm_state=standard；保持旧权重输入含义 |
| 视觉监督 | 冻结 FLUX VAE，原有损失权重不变 |
| 保存／测评 | 每 1,000 阶段步保存；阶段 20k／累计 40k 测评 |
| W&B | pi05_eskill_lora_bs32_libero40，默认读取项目 wandb.json，独立 run |

测评沿用前次协议：Spatial／Object／Goal／LIBERO-10，40 任务，每任务 5 个官方初态，总计 200 回合；seed=7，Flow Matching 10 步，每次执行 5 步，等待 10 步，套件上限 220／280／300／520，保存所有视频。四套件在训练完成后各占一张卡并发测评。

数据盘产物：`/data1/yuankai/robot-research/workspace/experiments/eskill_libero_unified/run_002/`。`status.json` 记录训练／测评／完成／失败阶段；`train.log`、`train/metrics.jsonl` 记录训练；40k 逐回合结果和视频位于 `evaluation/checkpoint_040000/`；`evaluation.json` 汇总四套件。

运行时依赖统一 Episode，不读取旧格式。旧 checkpoint 由 `scripts/convert_episode_checkpoint.py` 一次性转换，旧数据由 `scripts/preprocess/libero/migrate_unified.py` 一次性迁移；历史核验记录保留原义。

2026-11-03 已启动。启动 PID、命令和时间见数据盘 `launch.json`，源码版本及工作区哈希见 `runtime.json`。前置验证见 [统一链路检查](../unified_episode_check/README.md)。本轮与此前 30k 后旧链路续训的主要差别是数据组织和读取代码；本轮不重新计算归一化。

2026-11-03 用户要求取消目录软链接并重新训练：run_001 停止于阶段 11,432／累计 31,432，保留其历史日志。已删除 `atomicdata/libero40_unified` 软链接，只保留新版 `libero40/` 和旧版剩余元数据 `legacy/`。run_002 再次从原累计 30k 权重启动，配置及初始 checkpoint 的 data_index 均为正式 `libero40/index.json`，优化器／RNG／归一化延续 30k 原值。

当前后台流水线 PID=2560673。W&B：[pi05_eskill_lora_bs32_libero40](https://wandb.ai/laiyuankia-nanjing-university/EskillVLA/runs/wz5gsty7)，使用同名独立新 run。首批和云端上传核验完成后记录在数据盘 `wandb_check.json`。累计 40k 后自动测评协议不变。
