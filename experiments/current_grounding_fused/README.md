# 融合 Current／未融合 Subject 启动检查

2026-11-18，`feat/visual-primitive-v3`，基于 `0e2017b` 的工作区修改。检查类型为 technical，成功率为 null。

Current 使用冻结 PaliGemma 图文融合后的视觉／文本 Tokens，Subject 使用同一次视觉编码保留的融合前 Tokens。两路使用独立视觉投影、Queries 和 Decoder，共享 latent 读出头。Subject 仍监督已有夹爪局部 RGB，不新增持物身份标签。结构见 [设计说明](../../docs/design.md#current-grounding-专项入口待效果验证)。

| 项目 | 本次实际设置 |
| --- | --- |
| 数据／划分 | `atomicdata/libero40_grounding/index.json`，1,693 条，原父轨迹 split：1,518 train／175 val |
| 输入 | 当前主相机 RGB＋完整任务，无本体状态、bbox 或相机标定输入 |
| 初始化 | 官方 π₀.₅ `ckpt/pi05/base/pytorch_bf16/model.safetensors`，冻结预训练 FLUX VAE `ckpt/vae/flux1-dev` |
| 训练范围 | 两路视觉投影、文本条件模块、Queries／Decoder 和共享 latent 读出头；6,716,160 个可训练参数 |
| 精度／分布式 | FP32 参数＋BF16 autocast，NCCL DDP；物理 GPU 0、3，2×RTX 3090 24GB |
| 训练预算 | 每卡 batch 2，全局 batch 4，共 8 步；每卡 2 个 DataLoader workers |
| 优化／损失 | AdamW，LR 5e-5、warmup 1,000；Current／Subject 各 0.1 |
| 增强 | 反事实、Copy-Paste、保护区外 Patch Mask＋官方强度的裁切／旋转／颜色，水平翻转关闭 |
| 验证／保存 | 第 8 步，验证每卡 2 batch；保存完整模型／优化器／RNG checkpoint |
| W&B | disabled，技术短测不上传正式训练项目 |

双卡 8 步完成，无非有限 loss 或梯度；两卡最终可训练参数哈希完全相同。末步总 loss 0.642940（Current 3.413325、Subject 3.016078），固定验证子集总 loss 0.721847。训练步 2–8 平均 0.16384 秒／步（全局约 24.41 samples/s，不含初始化、验证与保存），峰值已分配显存两卡均为 15.16 GiB。只跑 8 步，这些数值用于判断启动与数值链路。

完整 checkpoint 已回读，并用真实权重 BF16 验证：一次视觉塔前向；同图替换文本时 Subject 完全一致、Current 输出变化；主干无梯度、两路投影均获得梯度。CPU 相关回归 66 项通过，1 项因沙箱 IPC 限制跳过；实际 GPU 双卡及多 worker 路径已运行。检查结束后 GPU 0、3 已释放。

## A／C 计算开销

同一张物理 GPU 0、真实基座／VAE、固定真实验证 batch 2、BF16；按 A→C→C→A 顺序，每次 3 步预热＋5 步计时，计入前向、VAE、反向、裁剪与 AdamW。A 的 Head 和前向取自提交 `0e2017b`，使用相同冻结基座与 VAE；计时产生的更新全部丢弃，不改写 checkpoint。

| 特征路径 | 平均单步 | images/s |
| --- | --- | --- |
| A：两路读取未融合视觉 Tokens，纯文本条件 | 0.11778 s | 16.98 |
| C：Current 融合、Subject 未融合 | 0.15163 s | 13.19 |

C 计算耗时约增加 28.7%。这项比较不含数据加载、增强、DDP 或 checkpoint IO，不代表完整训练吞吐；进程同时保留两份小型 Head，因此 profile 中的显存不是分别隔离进程的测量。双卡实际显存以启动检查的 15.16 GiB／卡为准。

## 配置与产物

[运行目录](/data1/yuankai/robot-research/workspace/experiment/current_grounding_fused_20261118_001) 中保存：

- `config.json`、`run.json`：精确配置、命令、PID、环境、源码哈希、初始化来源和状态。
- `train.log`、`train/metrics.jsonl`、`train/validation.jsonl`：训练与验证指标。
- `train/training_check.json`：两卡参数一致性、训练范围和峰值显存。
- `train/checkpoint.pt`：第 8 步完整权重，可用于技术检查。
- `checkpoint_check.json`、`verify_profile.py`、`reference_a.py`、`profile.json`：回读验证与计算开销对比。

启动命令如下；再次运行需新建输出目录，避免覆盖本次结果。

```bash
CUDA_VISIBLE_DEVICES=0,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo PYTHONPATH=src:. OMP_NUM_THREADS=2 \
/data1/yuankai/robot-research/workspace/envs/atomicvla/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=2 -m scripts.train \
  --config /data1/yuankai/robot-research/workspace/experiment/current_grounding_fused_20261118_001/config.json
```

正式配置入口仍为 [增强](../../examples/libero/train_current_grounding.json)／[无增强对照](../../examples/libero/train_current_grounding_baseline.json)。这两份是 32 步接口模板；本次双卡 8 步配置单独保存在运行目录。新投影结构与旧共享投影专项 checkpoint 不兼容，需要从基座新开训练。
