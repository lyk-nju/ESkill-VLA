# LIBERO-40：π0.5 基座 → ESkill 40k

> 历史实验记录：本目录配置对应重构前代码，保留当时的设置与结果，不能交给当前训练器。新运行统一见 [实验入口](../README.md)。

当前正式目标已改为 **gpu10、8×A100 80GB、全参数、全局bs128、90% train／10% val、每5k保存**。配方与执行说明见 [gpu10.json](gpu10.json) 和 [gpu10.md](gpu10.md)。正式训练尚未启动。

以下保留 s1 的 LoRA 准备方案及性能核验背景；[recipe.md](recipe.md) 描述该本地方案，不是 gpu10 的正式配方。

计划从 π0.5 基座第 0 步直接训练 ESkill 到 40k。使用新版统一 Episode，不加载任何 SFT／ESkill checkpoint，优化器、辅助分支和归一化统计均重新建立。

配置：[train.json](train.json)；训练与自动测评：[历史编排代码](https://github.com/lyk-nju/ESkill-VLA/blob/b2318c2/experiments/eskill_libero40/run.py)；短程性能分析：[历史性能核验代码](https://github.com/lyk-nju/ESkill-VLA/blob/b2318c2/experiments/eskill_libero40/benchmark.py)。

| 项目 | 设置 |
| --- | --- |
| 数据 | `atomicdata/libero40/index.json`，1,693 条／273,465 帧，全量训练，无独立 val |
| 初始化 | π0.5 PyTorch BF16 基座，`init_checkpoint=null` |
| 方法 | π0.5＋ESkill；LoRA rank/alpha=16/16；冻结基座与 FP32 FLUX VAE |
| GPU／batch | 4×RTX 3090，每卡 8，全局 32 |
| 输入 | 主／腕双相机，arm_state 开启，状态不拼入文本，关闭图像增强 |
| 动作 | horizon=10，repeat_last，完整动作 MSE |
| Motion | 预存主相机像素坐标；8 点、间隔 4 来源步、hold_last |
| 归一化 | 当前全量数据独立计算，state/actions/arm_state 均为 quantile |
| 优化 | lr=5e-5，warmup=1000，betas=(0.9,0.95)，weight decay=1e-10，clip=1，seed=42 |
| 损失权重 | action=1，skill/future/grounding/trace 各 0.1 |
| 保存／测评 | 首步保存一次，此后每 1k 保存；10k／20k／30k／40k 各测评 200 回合 |
| W&B | `pi05_eskill_lora_bs32_libero40_base`；默认读取根目录 wandb.json，独立新 run |

模型仍保留 200 tokens 的输入上限；collate 时只裁掉整个 batch 均无效的文本尾部 padding，保留所有真实 tokens、顺序、位置与 mask。注意力共享 KV 存储，LoRA 去掉 autocast 内冗余升降精度复制；五项损失与梯度对照通过。续训直接跳过已消费的采样索引，不再读取、解码旧 batch。梯度重算保持开启，VAE 精度及训练监督不变；首步保存后连续运行，不再用 20 步短段重启检查。四卡最新短测约 1.80 秒／步，详见 [性能核验](performance.md)。

每次测评沿用此前协议：Spatial／Object／Goal／LIBERO-10 各 50 回合，seed=7，官方前 5 个初态，Flow Matching 10 步，执行前缀 5 步，等待 10 步，控制上限 220／280／300／520，保存全部视频。训练与测评依次运行，测评四套件各占一张 GPU，随后恢复同一优化器／RNG／W&B run。

产物：`/data1/yuankai/robot-research/workspace/experiments/eskill_libero40/run_002/`。norm_stats.json、runtime.json、launch.json、status.json 分别记录统计、代码／环境、启动命令／PID、运行阶段。训练日志和 checkpoint 在 train/；四次测评在 evaluation/checkpoint_010000/ 等目录，汇总为 evaluation_010000.json 等文件。step 与 W&B 横轴均为本轮 0–40k。

历史 run_001 从 SFT20k 初始化，已按用户要求停止，停下前日志约 66 步；不作为本轮初始化或训练来源。其配置、初始化证明与日志保留在数据盘原目录。

性能短测及限制见 [performance.md](performance.md)。
