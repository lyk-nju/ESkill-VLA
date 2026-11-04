# 训练性能核验

2026-11-04。真实 π0.5 基座、ESkill／冻结 FLUX VAE、统一 LIBERO-40，4×RTX 3090，每卡 batch 8、全局 32。每种模式预热 3 步、计时 12 步；torchrun 的 OMP_NUM_THREADS=1。没有上传正式 W&B 或产生正式训练 checkpoint，短测状态不用于后续训练。

## 四卡实测

| 项目 | 原实现：padding 200 | 仅裁剪 padding | 裁剪＋减少复制 |
| --- | ---: | ---: | ---: |
| 文本长度 | 200 | 平均 19.75 | 平均 19.75 |
| 整步（最慢 rank 的均值） | 2.328 s | 1.923 s | **1.801 s** |
| 前向总计 | 0.979 s | 0.853 s | 0.803 s |
| 其中语言主干 | 0.501 s | 0.400 s | 0.345 s |
| 其中视觉编码器 | 0.078 s | 0.079 s | 0.079 s |
| 其中 Action Expert | 0.090 s | 0.058 s | 0.056 s |
| 其中 VAE 目标编码 | 0.275 s | 0.276 s | 0.276 s |
| 反向（含重算及梯度通信） | 1.332 s | 1.052 s | 0.979 s |
| 等待 DataLoader | 0.00029 s | 0.00029 s | 0.00031 s |
| Host→GPU | 0.0097 s | 0.0096 s | 0.0096 s |
| 优化器 | 0.0049 s | 0.0049 s | 0.0049 s |
| 诊断汇总 | 0.0028 s | 0.0030 s | 0.0032 s |
| 峰值 allocated 显存 | 11.92 GiB | 11.73 GiB | 11.73 GiB |

各分项为 rank0 均值；四张卡整步均值差异小于 0.00001 秒。“其中”项已包含在前向总计内，不能再次相加。整步包含末尾同步；正式日志及 checkpoint 写入不计入此表。不同模式在同一模型上连续更新，用于性能测量，损失数值不能作为模型效果对比。

新增复制优化相对仅裁剪 padding 减少 **6.4%** 耗时；全部改动相对原实现减少 **22.7%**，吞吐提高约 **29.3%**。按短测外推，40k 纯训练约 **25.9→20.0 小时**，节省约 **5.9 小时**；不包含模型加载、四次测评和机器负载波动，也不承诺实际完成时间。

## 已落地

- `training/data_loader.py`：仅裁掉整个 batch 均无效的右侧文本 padding，保留真实 token、顺序、位置和 mask。`ResumableBatchSampler` 在索引层跳过已消费 batch，恢复训练不再先解码旧样本再丢弃。
- `models/gemma.py`：π0.5 的单个 KV head 用共享存储展开，不再复制八遍；继续调用原 SDPA 算子，支持其他 KV head 数量。整模型峰值仍由其他运算决定，没有宣称峰值显存大幅下降。
- `models/lora.py`：autocast 内让 Linear 自行对齐 dtype，去掉 BF16→FP32→BF16 往返复制；非 autocast 路径保留必要转换。
- `scripts/train.py`：接入索引续训；首步保存一次，之后按原频率保存，避免为验证存盘而单独运行短段再重启模型。

数据格式、相机／状态输入、batch、损失及权重、VAE FP32、优化器、采样顺序、梯度重算和测评频率均保持原 recipe。续训优化依赖当前确定性的 Dataset／CPU transforms；随机图像增强若启用，仍在模型侧并由 checkpoint RNG 恢复。

相关回归检查共 **99 项通过**：五项损失／参数梯度的优化前后对照、不同 KV head 数、LoRA autocast／普通路径、0／2 worker 的 epoch 内及边界恢复、跨 epoch 样本顺序与最终权重逐值一致、多卡有效标签平均、验证与 W&B 保存恢复。CUDA BF16 的 KV 展开单算子检查前向一致，q 梯度存在 BF16 舍入量级的差异；数学目标一致，不宣称跨算子实现的 GPU 更新逐位相同。

## 测过但未采用

- 原生 `enable_gqa=True`：当前 PyTorch 2.10.0／3090／padding mask 组合回退 math attention，Action attention 前反向约 0.95→2.75 ms，未采用。
- VAE channels-last：保持 FP32 时单次编码约 0.275→0.360 s，变慢且存在数值差异，未采用。
- 关闭梯度重算：先前同 batch 单卡实测 OOM，保留重算。
- 增加 workers、异步日志或 checkpoint 框架：数据等待、诊断及存盘摊销占比很小，不增加这些复杂度。既有约 480 MB GPU checkpoint 三次保存均值 2.03 s，每 1k 步摊销约 2 ms／步。

复现入口：[benchmark.py](benchmark.py)，默认比较 `baseline / trim / optimized`；原始四卡结果：[profile_infra.json](/data1/yuankai/robot-research/workspace/experiments/eskill_libero40/run_002/profile_infra.json)。同目录 `profile_ddp.json` 保存此前 padding 单项检查（2.371→1.969 s），`profile_single.json`、`profile_checkpoint.json` 保存单卡与存盘检查。新旧短测分别记录，以上收益全部按本次同场三组结果计算。
