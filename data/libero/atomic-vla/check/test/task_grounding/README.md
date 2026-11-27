# 完整任务 Current／Subject 技术检查

2026-11-17，基于 `f18ea3c` 的未提交修改。正式训练未启动。

- 架构：共享未融合视觉特征，Current 使用纯文本条件，Subject 独立；冻结 VLM；仅两项 VAE loss，无目标短句生成及单臂 Arm Embedding。
- 数据：`atomicdata/libero40_grounding`，原父轨迹 train／val split。完整原任务通过 Event 绑定目标；首版支持 12 类简单搬运任务的源物体替换，其他任务保留原指令。
- 预览：40 任务、93 个 Event 中点，186 个有效 role 在两阶段增强后全部保留；6 次反事实替换、2 次 Copy-Paste。人工检查仍需继续；本次没有加载真实 FLUX VAE 进行重建。
- CPU batch size 1 图像增强：平均 21.03 ms、P95 24.75 ms，只计 batch 图像变换，不含 Dataset、VAE、渲染或完整模型。
- 技术训练：真实数据／tokenizer，随机 tiny 主干和随机冻结 tiny VAE，CPU FP32，batch 2，4 步；Current、Subject loss、梯度、验证及 checkpoint 保存正常。末次验证总 loss 0.008743，仅用于接口检查，不能当作正式效果。
- 测试：第一轮扩展回归 139 通过、5 跳过、3 项因沙箱 socket 限制失败；开放本机 IPC 后，两个旧动作模型 DDP 回归、Grounding 多 worker 重放及离线 W&B 四项均通过。另有新架构 checkpoint 回读、冻结参数与输入隔离测试通过。没有验证正式 π₀.₅ GPU BF16 或多卡训练效果。

[预览视频](../video/task_grounding_20261117.mp4) · [报告目录](/data1/yuankai/robot-research/workspace/reports/atomicvla_task_grounding_20261117_001) · [配置与冻结范围](../../../../../../src/openpi/training/README.md#current-grounding-专项训练)

生成预览：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 /data1/yuankai/robot-research/workspace/envs/atomicvla/bin/python -m scripts.preprocess.libero.grounding_review \
  --config examples/libero/train_current_grounding.json \
  --output /data1/yuankai/robot-research/workspace/reports/atomicvla_task_grounding_20261117_001
```

运行命令、源码哈希与配置在报告目录 `run.json`／`config.json`；四步训练配置与日志在 `tiny_training/`。视频为输入及监督标签检查，不是学习后模型预测。新结构需从基座新开训练，不能恢复旧目标文本生成专项 checkpoint。
