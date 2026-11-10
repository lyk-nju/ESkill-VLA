# 本机真实权重训练检查（此前冻结先验的 Stage 2）

2026-11-10，用户确认四张显卡已释放，要求本机运行以检查训练是否正常。本次为有限步数的技术检查，正式 Stage 1 20k／Stage 2 40k 预算仍保留。

- 硬件：4×RTX 3090 24GB，GPU 0–3；启动前无计算进程。
- 初始化：真实官方 π0.5 PyTorch 权重与真实 FLUX VAE，完整模型；`smoke=false`。
- 数据：原 LIBERO40 固定父轨迹 split，1,518 train／175 val；重新仅用 train 计算 quantile 统计。
- 配方：继承两阶段配置，FP32 参数／BF16 autocast、冻结 VLM、Motion 每 5 帧八点、Future 每边额外 8 px、技能重采样封顶 3；Stage 1 随机水平镜像。
- 技术预算：Stage 1 先 16 步，再恢复到 20；Stage 2 从这份短测权重初始化先 8 步，再恢复到 12。每卡 batch 1，全局 4；lr=1e-5、warmup=1k。每 8 步及阶段结束验证／保存；每次验证固定 4 个样本。学习率沿用正式配方，不用于判断收敛。
- W&B：按技术检查约定关闭，记录完整本地日志。不开启闭环成功率测评。
- 启动代码：`af6fb09`；启动时工作区干净。环境为数据盘 `workspace/envs/atomicvla`。

运行目录：[local_check_001](../../../workspace/experiment/eskill_libero40_mirror/local_check_001/)，含配置、归一化、各阶段日志、checkpoint、运行命令／PID及状态；实文件位于数据盘。

记录状态：**此前配方的技术检查已完成**。用户随后明确 Stage 2 也应更新 Skill／Grounding／Motion；下文冻结先验、仅 FM 的 Stage 2 已不再作为当前配方或验收依据。Stage 1 检查仍有效，历史产物保持原值。本轮未修改训练代码。运行 PID 与各阶段 PID 保留在目录内 `runtime.json`／`*_process.json`。

## 实际结果

| 阶段 | 完成步数 | 可训练参数 | 最后训练损失 | 最后验证损失 | PyTorch 峰值 allocated／卡 |
| --- | --- | --- | --- | --- | --- |
| Stage 1 | 20 | 59,454,985 | 10.899559 | 9.951834 | 15.83 GiB |
| Stage 2 | 12 | 432,466,976 | 0.065337 | 0.086477 | 18.42 GiB |

这里的 Stage 1 总损失为五项辅助损失之和，Stage 2 只有 Action FM，二者数值不能横向比较。验证每次仅四个固定样本；两阶段均处于 warmup 前段，这些数字用于检查有限性，不能据此判断收敛。峰值为 PyTorch allocated，未包含 CUDA 上下文和 allocator 的全部保留显存。

- Stage 1：原生 π0.5 的所有来源权重逐项相等，VLM／Action 冻结正确。80 次样本抽取中镜像 44 次，含方向词的任务没有被镜像。六类真实标签均被抽到：Pick 35、Place 23、Open 8、Close 2、Turn 7、Push 5。
- Stage 2：所有冻结张量与 Stage 1 严格相等；208 个 Action 张量及 36 个 Motion K/V 张量发生更新。没有实例化 VAE，归一化与 Stage 1 完全相同。
- 两阶段所有训练／验证损失与梯度有限，无 OOM、NaN 或 Inf；四卡最终可训练参数 SHA256 完全一致。Stage 1 从 16 恢复到 20，Stage 2 从 8 恢复到 12，日志步数连续；Stage 2 步数与优化器独立于 Stage 1 重新开始。
- 策略回读：以一条验证轨迹的当前双相机、状态及当前二维起点输入，执行 FM10，输出 `[10,7]` 有限动作；未加载 VAE。没有运行仿真闭环，成功率为 `null`。

训练及恢复过程（含归一化、四次初始化、checkpoint 写盘）合计 868.9 秒；逐项权重核对 6.5 秒，策略构建与推理 51.8 秒。主要等待来自真实大模型初始化和完整 checkpoint 写盘。

## 产物

所有大型文件保留在数据盘：

- [运行配置／环境／命令／PID](../../../workspace/experiment/eskill_libero40_mirror/local_check_001/runtime.json)
- [逐项核对结果](../../../workspace/experiment/eskill_libero40_mirror/local_check_001/audit.json)
- [策略回读结果](../../../workspace/experiment/eskill_libero40_mirror/local_check_001/policy_check.json)
- [Stage 1 checkpoint](../../../workspace/experiment/eskill_libero40_mirror/local_check_001/stage1/train/checkpoint.pt)，14.47 GB，20 步技术检查权重。
- [Stage 2 checkpoint](../../../workspace/experiment/eskill_libero40_mirror/local_check_001/stage2/train/checkpoint.pt)，17.12 GB，12 步技术检查权重。
- 每阶段 `train/metrics.jsonl`、`validation.jsonl`、`sampling.json`、`training_check.json`，以及 `logs/` 中的进程输出。

这验证了当前真实权重的数据→增强→损失→反向→保存／恢复→阶段交接→推理链路。正式 20k／40k 预算、远端 batch 和模型效果尚需独立执行，本次短测不替代收敛与闭环验收。

此前 Stage 2 checkpoint 若需回读，应使用该次记录对应的 `af6fb09` 代码。修正后的联合训练结果另见[当前短测](local_check_joint.md)。
