# 本机 π0.5 LoRA：新版 Episode 基线

2026-11-04，用户确认从 π0.5 基座第 0 步开始，纯动作 SFT 到 40k；四卡全局 batch32，固定90% train／10% val。配置：[train.json](train.json)。

| 项目 | 配方 |
| --- | --- |
| 模型 | 原版 π0.5，仅动作损失；不构建 ESkill、Grounding、Motion 或 VAE |
| 初始化 | 本地转换好的 π0.5 PyTorch BF16 基座；init_checkpoint=null，优化器与 W&B run 新建 |
| LoRA | 两个 Gemma 的 attention／MLP 线性层，rank=16、alpha=16；原基座、视觉编码器与动作输入／输出层冻结，LoRA 参数 FP32 |
| 设备／精度 | GPU0–3，4×RTX3090；每卡8，全局32，无梯度累积；BF16基座与autocast，梯度重算、fused AdamW、共享DDP梯度缓冲 |
| 数据 | 自包含 libero40；40任务，固定1518 train／175 val，245553／27912动作行；frame_stride=1 |
| 统计 | 复用 ../eskill_libero40/norm_stats.json，仅train计算，quantile；其中arm_state统计在本配方不使用 |
| 输入 | 当前主／腕双相机224×224＋任务文本；不输入离散状态或arm_state；无图像增强、无历史堆叠 |
| 动作 | 源7维OSC命令，不额外做delta变换；填充32维，H=10；repeat_last；完整32维MSE |
| 优化 | lr=5e-5，线性warmup1000后恒定；AdamW betas=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、clip=1；seed42 |
| 预算 | 40000步，约128万次样本抽取、5.2遍训练集；最后不足batch按当前loader保留 |
| 验证 | 每1000步；固定验证子集，每卡最多32batch，合计1024样本；eval/no_grad、无增强；与gpu10的4096样本验证范围不同 |
| 保存 | LoRA首步存盘核验，随后每5000步；checkpoint含LoRA、优化器、各rank RNG、配置与统计，回读仍需原基座 |
| 闭环 | 10k／20k／30k／40k；Spatial／Object／Goal／LIBERO-10各50回合，共200；seed7、官方前5初态、FM10、执行前缀5、等待10、上限220／280／300／520；全部视频 |
| W&B | pi05_lora_bs32_libero40；根目录私有wandb.json，在线；跨测评阶段恢复同一run，横轴0–40k |

使用现有训练／测评编排入口，不复制实现。从项目根目录运行：

```bash
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  experiments/eskill_libero40/run.py \
  --config experiments/pi05_libero40/train.json \
  --gpus 0 1 2 3 \
  --sim-python /data1/yuankai/robot-research/workspace/envs/libero-eval/bin/python \
  --openpi-root /home/yuankai/robot-research/openpi
```

产物：`/data1/yuankai/robot-research/workspace/experiments/pi05_libero40/run_001/`；launch.json／runtime.json记录命令、PID、环境和Git；status.json记录阶段；train/存权重与日志；evaluation/存各里程碑测评。训练与测评顺序运行，阶段间恢复同一优化器／RNG，不重新warmup。

这轮使用新版统一Episode、固定90/10划分及train-only统计；不能把历史全量1693轨迹训练的成功率视为严格同配置对照。与gpu10相比，本轮是纯动作LoRA、bs32、无状态输入，训练样本预算为其四分之一。

## 启动核验

已启动，pipeline PID为2770507。W&B：[pi05_lora_bs32_libero40](https://wandb.ai/laiyuankia-nanjing-university/EskillVLA/runs/78r7rjsy)。首批训练损失与梯度有限，初始稳定步时约1.4秒；该值不包含验证、写盘、测评及初始化，不能作为完整实验耗时。仅训练LoRA共26,542,080参数，参数与Adam状态均FP32。首步checkpoint已回读核验，保存4个rank的RNG，大小约311MB。40k训练及四次闭环测评尚未完成。

W&B云端API已确认收到step1／20的train/loss；并非仅本地离线记录。启动核验时训练已到48步，最近20步均值约1.45秒，pipeline处于training。

## 2026-11-04 读取与推理修复

已核验全部1693条／273465动作行：主／腕相机配对均无负值、越界、无效图像或未来帧。当前训练环境NumPy1.26.4不触发NPZ偏移溢出。无需因此从头训练；已启动进程继续使用其加载的代码，下一10k阶段与测评进程加载新代码。

本机基座 `ckpt/pi05/base/pytorch_bf16/model.safetensors` 的完整SHA256为 `ce648b1f3fb7a2fcf75413568e02db5fde9779106cd77fe7b7cb2ed2e38587be`。当前进程产生的旧式checkpoint仍只有文件头身份，读取会明确警告；下一阶段续训保存后改为完整哈希。该已知基座的完整哈希作为独立核验记录保留。本轮LoRA训练和推理均为BF16计算，精度修复不改变本轮预期推理设置。

## 2026-11-05：10k 测评启动中断与恢复

11 月 4 日 20:34，0–10k 训练正常退出，耗时约 4 小时 19 分钟；checkpoint 保存完整，四卡权重一致。10k 时 train/action_loss=0.02990、val/action_loss=0.02472。20:35 自动测评客户端报 `ModuleNotFoundError: No module named 'numpy'`，编排流程随之退出；未开始任何测评回合。

原因是 `run.py` 对 `--sim-python` 调用 `Path.resolve()`，把虚拟环境的 Python 符号链接解析为独立基础解释器，丢失虚拟环境依赖。修复后保留虚拟环境入口路径，并在训练前用相同解释器运行 LIBERO 全任务列表检查。新增 `--resume-step 10000`：跳过已完成训练，先测评 10k，再用完整 checkpoint 继续 20k／30k／40k。新跑与恢复编排回归检查通过。

已重新核对基座完整 SHA256，以及 10k checkpoint 的 step=10000、world_size=4、494 个优化器参数状态、W&B ID=78r7rjsy。未修改配方、数据划分、batch、状态输入或图像增强。失败日志保存在原产物目录的 `failed_10k/`，恢复命令及启动时间见 `recovery.json`。恢复 pipeline PID=2820362，先补跑原协议 200 回合测评；恢复后的结果以 `status.json` 和实际测评记录为准。

恢复启动核验：四个客户端均使用 `envs/libero-eval/bin/python`，四卡测评服务存活；Spatial 已产生首条正常闭环记录（error=null）。此记录只证明补测链路恢复，不能作为 200 回合最终成功率。
