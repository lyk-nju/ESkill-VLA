# gpu10：全参数 LIBERO-40

2026-11-04，用户确认 8×A100 80GB、全局 batch 128、任务内完整父轨迹 90% train／10% val，checkpoint 每 5k 保存。可执行配方为 [gpu10.json](gpu10.json)，沿用 [run.py](run.py) 每 10k 暂停训练并进行闭环测评。正式训练尚未启动，真实大模型的 A100 显存验收待完成。

## 固定配方

| 项目 | 设置 |
| --- | --- |
| 初始化 | π0.5 PyTorch 基座，step 0；不加载旧 SFT／ESkill checkpoint |
| 参数范围 | `lora_rank=0`、`freeze_vlm=false`；视觉编码器、语言主干、动作专家及 ESkill 分支均训练；FLUX VAE 和技能文本原型冻结 |
| 精度 | `backbone_dtype=float32`、`precision=bfloat16`；FP32 参数、梯度、Adam 状态；BF16 autocast；VAE 编码仍为 FP32 |
| 显存措施 | 视觉／语言／动作 Transformer 梯度重算；DDP `gradient_as_bucket_view=true`；`adam_fused=true`，使用 fused AdamW，避免 foreach 的整组参数中间张量 |
| batch | 8 卡×每卡 16＝128，无梯度累积；每 rank 2 个数据 worker |
| 数据 | 固定 `data/libero/atomic-vla/split.json`；1,518 train／175 val，共 245,553／27,912 个动作行；训练保留全部训练轨迹及帧 |
| 统计 | 使用随实验提供的 [norm_stats.json](norm_stats.json)，仅由 train 计算 state／actions／arm_state 的 quantile；不得复用先前 1,693 条全量数据统计 |
| 输入 | 当前主／腕双相机，224×224；不启用图像增强；15维 arm_state 仅直接进入 Motion／未来技能分支，不拼入文本 |
| 动作与监督 | H=10、repeat_last、32维动作 MSE；三角色画布256、冻结FLUX潜变量目标；Motion八点、间隔4来源步、hold_last |
| 优化 | lr=5e-5，线性 warmup 1000步后恒定；betas=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、clip=1、seed42；batch 增加不自动放大学习率 |
| 损失 | action=1，当前技能／未来技能／Grounding／Motion各0.1 |
| 步数 | 40k，约512万次样本抽取，相当于约20.9遍训练集；不是原bs32方案的相同样本预算 |
| val loss | 每1k及阶段末；固定验证集内的抽样，每卡最多32 batch，八卡合计最多4,096样本；不是整套175条验证轨迹逐帧遍历 |
| checkpoint | 5k、10k、15k、20k、25k、30k、35k、40k，共8份；全参数不额外保存首步；包含完整模型、Adam、各rank RNG、配置及统计 |
| 闭环测评 | 10k／20k／30k／40k各200回合；四套件各50回合，前四张指定GPU并发；seed7、FM10步、执行前缀5步、等待10步，沿用原控制上限和全部视频 |
| W&B | `pi05_eskill_bs128_libero40`，根目录本机 `wandb.json`，跨10k阶段恢复同一run |

普通 DDP 每卡保存完整模型状态。约33.66亿可训练参数的 FP32 参数＋梯度＋Adam 状态理论合计约50.2 GiB／卡，另有激活、autocast、通信及VAE开销；共享梯度缓冲和重算不能代替真实显存实测。不自动降低 batch、改精度或切换训练范围。

完整 checkpoint 估计约40GB／份，8份约320GB，另需核验输出及临时写盘空间。`checkpoint.pt` 与最新里程碑为硬链接，不额外复制一份。训练恢复及策略加载使用 mmap，避免每个进程先把包含优化器的完整 checkpoint 都读入 CPU 内存；推理只读取模型部分。

## gpu10 资源与命令

从项目根目录运行。数据根目录按用户指定为 `/data/home/shengqiuProf_user_yuankai/data/robot-research/`：

```text
atomicdata/libero40/index.json
atomicdata/libero40/episodes/<id>/{episode.json,arrays.npz,images.npz}
ckpt/pi05/base/pytorch_bf16/model.safetensors
ckpt/pi05/base/big_vision/paligemma_tokenizer.model
ckpt/vae/flux1-dev/{config.json,diffusion_pytorch_model.safetensors}
experiments/eskill_libero40/run_001/
```

权重、环境及输出路径是新服务器的约定，不代表已经下载或验证存在；若实际安装位置不同，修改配方对应路径。固定 split 随项目同步。训练环境按项目 `.[train]` 依赖准备；LIBERO 仿真环境、官方 OpenPI 的 serving/client 代码及资产按 [测评文档](../../docs/evaluation.md) 准备。`wandb.json` 只在服务器本机配置，不提交。

统计已在 s1 生成并保存为项目内的 [norm_stats.json](norm_stats.json)（约67KB），配方直接引用该文件。随代码同步即可，换服务器、GPU、batch 或全参数／LoRA 不需要重算。只有数据内容、训练划分、字段变换或归一化方法改变时，才应为新实验重新运行 `scripts/compute_norm_stats.py` 并指定新的输出文件。统计内的来源路径用于追溯，不作为运行时数据路径。

迁移清单：最新 ESkill-VLA 代码（包括本实验配置／统计及固定split）；整个自包含 `atomicdata/libero40/`；π0.5 PyTorch基座／tokenizer；FLUX VAE配置／权重；用于自动测评的LIBERO源码／资产及官方OpenPI serving/client代码；本机私有 `wandb.json`。基座、VAE及公开测评依赖可在gpu10下载相同版本；不需要旧SFT／ESkill权重、旧数据目录、external原始数组、标注器模型或历史视频。

Python虚拟环境在目标服务器重建，不直接复制s1的venv路径。已验证的训练环境核心版本为 Python3.11、torch2.10.0、transformers4.53.2、diffusers0.33.1、sentencepiece0.2.0、safetensors0.5.3、wandb0.24.2、numpy1.26.4、Pillow11.2.1；serving/client还需要OpenPI客户端包及其网络依赖。LIBERO单独使用其兼容的仿真环境，见测评文档。CUDA驱动由目标服务器提供，不从s1复制。

A100 短测使用另一个输出目录和禁用 W&B 的配置，保持每卡16和全部模型／精度设置；至少完成两步（包括Adam状态分配）、保存、恢复及验证。不能用 tiny/smoke 模型代替这一步，也不能把短测目录当作正式起点。原配方保持从基座0步开始。

目标机短测通过后，正式自动训练／测评命令为：

```bash
python experiments/eskill_libero40/run.py \
  --config experiments/eskill_libero40/gpu10.json \
  --gpus 0 1 2 3 4 5 6 7 \
  --sim-python /data/home/shengqiuProf_user_yuankai/data/robot-research/envs/libero-eval/bin/python \
  --openpi-root ../openpi
```

仿真解释器与 OpenPI checkout 通过参数指定，不再硬编码 s1 路径。训练卡数直接取 `--gpus` 数量，runtime.json 记录实际全局batch、命令、解释器、Git版本和工作区状态。训练器本身支持 `scripts/train.py --config <阶段配置> --resume <checkpoint.pt>`；当前编排器按10k阶段顺序执行，异常后不自动删除／重试已有测评目录。

## 本地验证

- 真实统一数据的 train-only quantile 已在本地重算核验：1,518条／245,553动作行，验证集交叉为0；结果已从临时目录整理到本实验 [norm_stats.json](norm_stats.json)，gpu10直接复用。
- 60项回归检查通过，包含视觉重算的实际反向重执行及梯度对照、全参数AMP、0／2 worker断点恢复、固定划分、DDP损失、验证与W&B恢复。
- 本机四张3090上，tiny模型以 FP32 参数＋BF16 autocast＋fused AdamW＋共享梯度缓冲运行：2步保存后恢复至4步，与连续4步最终权重逐值相同，各rank一致，Adam状态均FP32。记录：`/tmp/eskill_full_cuda_check/`。这是CUDA功能检查，不是A100整模型显存或速度结果。
- 八卡启动参数、bs128、5k保存配置和四次测评调度已通过模拟调度测试；没有启动gpu10正式任务。
