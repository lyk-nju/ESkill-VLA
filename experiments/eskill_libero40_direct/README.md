# LIBERO-40：新版架构直接联合训练对照

保留当前 Skill MoE → Object-conditioned Motion → Action 架构，关闭视觉增强和技能重采样，从官方 π0.5 基座直接联合训练。用于判断新版在接近旧版训练条件时的表现；这是多项配方调整的整体对照，不是单因素消融，也不是旧模型复现。

配置：[train.json](train.json)。参照仓库保存的旧版 [gpu10.json](../eskill_libero40/gpu10.json) 与 `b2318c2` 的损失实现；它们不代替待比较的实际 checkpoint 配置。尚未启动正式训练。

| 项目 | 本次配方 |
| --- | --- |
| 初始化 | 官方 π0.5 PyTorch 基座；`init_checkpoint=null`，辅助分支随机初始化 |
| 阶段／预算 | `training_stage=joint`，直接 40k；所有步均包含 Action FM，无 Stage 1 |
| 参数范围 | VLM、Action、Skill／Grounding／Motion 及辅助头全部训练；FLUX VAE 冻结；无 LoRA |
| 优化 | 统一 lr=5e-5，warmup 1k 后恒定；fused AdamW，betas=(0.9,0.95)，weight decay=1e-10，clip=1 |
| batch／精度 | 每卡16，八卡全局128；FP32 参数与优化器、BF16 autocast、梯度重算 |
| 数据 | 固定父轨迹 split，1,518 train／175 val；frame_stride=1；普通打乱采样，关闭稀有技能重采样 |
| 归一化 | 复用旧实验提交的 train-only actions quantile；共享文件内的 state／arm_state 不进入模型 |
| 视觉 | 主／腕双相机；图像增强、镜像、Action 视觉 K/V 遮蔽全部关闭；无平移或 latent 复用 |
| 动作 | H=10、7维源动作补到32维、完整动作 MSE、repeat_last |
| 保存／验证 | 每5k保存、每1k验证；验证每卡最多32 batch；seed42 |
| 日志 | 正式训练读取本机 wandb.json；Motion 梯度诊断每100步，仅本地日志 |

## 损失对应

| 当前损失 | 权重 | 与旧版的对应 |
| --- | --- | --- |
| Action FM | 1.0 | 沿用旧值 |
| Skill | 0.1 | 沿用当前技能 CE 的旧值；类别与模型实现已有变化 |
| Current Object | 0.1 | 旧 Grounding 是三个角色损失相加后乘0.1，各角色实际系数均为0.1 |
| Future Object | 0.1 | 同上；不因分开记录而将系数减半 |
| Trace | 0.1 | 沿用旧系数；时间偏移和有效性规则不同；直接预测绝对二维坐标 |
| Gripper | 0.1 | 新增监督没有旧对应项；暂沿用辅助项0.1，不宣称已调优 |

旧版 Future Skill CE 和夹爪视觉 Grounding 已移除，不将其权重转移到名称近似的新损失。新旧总辅助损失和梯度并不因系数相同而等价。

本次保持当前标签定义：8个 Motion 点、每5个保存帧一个点，最短10帧窗口允许跨 Event；Future bbox 额外外扩8px。这是监督定义，不是随机图像增强。模型统一 RGB＋Language，删除本体与2D EEF起点输入，不依赖在线相机标定。输入条件对齐关闭状态文本的 π0.5 SFT；历史 gpu10 ESkill 配方实际启用了15D状态，因此本次与旧 ESkill 比较仍包含架构、输入和监督差异，不能称为只改变训练策略。

## 启动

在目标服务器修改配置中的数据、基座、tokenizer、VAE 和独立输出路径，保留固定 split 与对应统计。数据需完成新版 gripper_convention 迁移及相机／EEF 标注，见[数据检查](../../data/libero/atomic-vla/check/test/training/README.md)。不加载两阶段或旧架构 checkpoint，不继承其优化器与步数。

从仓库根目录运行，下面固定使用八卡，得到全局 batch 128。W&B run 名称为 `pi05_eskill_bs128_libero40_direct`；每个新实验创建独立 run。若修改卡数或每卡 batch，同时修正名称中的全局 batch。

```bash
PYTHONPATH=src:. torchrun --standalone --nproc-per-node=8 \
  scripts/train.py --config experiments/eskill_libero40_direct/train.json
```

纯训练命令不会自动启动闭环测评。先核对八卡实际显存与资源，不能用 CPU tiny 检查代替正式硬件验收。40k×128 名义样本数为512万；若改变卡数或 batch，需记录实际样本预算，避免只按10k步比较。

评测使用本次新 run 的10k／20k／30k／40k checkpoint，按相同任务、初态、seed、FM步数、执行前缀和控制上限与旧版比较，分别报告原版和 Plus。既有抽样结果与全量结果不得混合，详见[测评协议](../../docs/evaluation.md)。

## 技术核验

2026-11-11：配置解析／序列化通过，优化器、预算和 split 设置与历史 gpu10 配方逐项核对一致，本地资源路径存在。按本配方开关构造 CPU tiny 模型，六项损失及加权反向正常，VLM／Action／先验分支获得梯度，VAE 冻结；Action 保留完整视觉且不抽取遮蔽随机数。

现有配置、联合训练和模型测试共25项通过，包含保存／恢复与双进程 CPU DDP。DDP 首次受沙箱本地通信限制失败，在允许本地通信的环境单独重跑通过。未进行真实大模型训练、八卡显存验收或闭环效果测评。

同日移除二维 EEF 起点输入后，119项相关回归通过，涵盖联合训练、两阶段交接、恢复、在线推理及CPU DDP。追加的16项几何／镜像／RoboTwin适配检查包含“当前投影无效不屏蔽未来有效标签”的新回归。真实 Episode `libero_10_no_noops_s000_r000` 的未来标签保存／回读、Motion及镜像单帧渲染通过；未重算全量覆盖率。带旧 `start_projection` 的 ESkill checkpoint 不能直接加载到当前结构，需使用对应旧代码评测；本配方仍从官方基座重新训练。

最终输入统一为 RGB＋Language 后，168项相关回归通过，另补测了 ESkill／无状态 π0.5 的客户端请求和 RoboTwin 保存数据渲染。模型不含 state_projection／arm_state_mask；相同噪声下额外传入 NaN 状态不改变动作。真实训练 Episode 抽查确认没有读取当前状态数组，只有图像、文本和固定臂身份进入模型；无状态镜像标签保存／回读及单帧渲染通过。配置禁止 ESkill 开启状态文本输入。旧状态分支 checkpoint 也不再兼容；以上仅为接口与计算验证，不代表闭环效果改善。
