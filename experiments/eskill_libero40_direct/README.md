# LIBERO-40：新版架构直接联合训练对照

保留当前 Skill MoE → Object-conditioned Motion → Action 架构，关闭视觉增强和技能重采样，从官方 π0.5 基座直接联合训练。用于判断新版在接近旧版训练条件时的表现；这是多项配方调整的整体对照，不是单因素消融，也不是旧模型复现。

配置：[train.json](train.json)。参照仓库保存的旧版 [gpu10.json](../eskill_libero40/gpu10.json) 与 `b2318c2` 的损失实现；它们不代替待比较的实际 checkpoint 配置。尚未启动正式训练。

## 视觉增强对照

2026-11-13 新增 A1／A2／A3，保留 A0 原配方。四组均为单阶段联合40k，官方基座、固定split、八卡全局128、lr=5e-5、Action=1／辅助各0.1；镜像、技能重采样和 Action Patch Masking 均关闭。新增配方没有启动正式训练。

| 实验 | 配置 | 唯一增强差异 |
| --- | --- | --- |
| A0 | [train.json](train.json) | 无增强 |
| A1 | [color.json](color.json) | 每个样本／视角以0.8概率做颜色扰动；brightness／contrast系数0.85～1.15、saturation系数0.9～1.1 |
| A2 | [color_crop.json](color_crop.json) | A1＋每个样本以0.5概率裁切主图，窗口边长为画布的95%～100%，保持方形 |
| A3 | [color_crop_rotation.json](color_crop_rotation.json) | A2＋每个样本的主图旋转，角度均匀采样[-5°,5°]；rotation_prob=1 |

裁切同步变换224主图、256 Current／Future画布及权重、未来二维轨迹；共享连续归一化窗口，避免不同尺寸各自整数取整。腕图不做几何裁切。出画点失效，不钳到边缘；角色区域完全裁出则失效。输入颜色扰动不改变 VAE 监督颜色，黑色目标背景保持-1。Action、Skill、Gripper及时间偏移保持原值。验证与推理不增强。

A3先裁切、再绕画布中心(0.5,0.5)旋转；正角度在y向下的图像坐标中为顺时针。点使用正向R，图像采样使用逆向R；合成同一网格，对每个RGB／权重流只插值一次。保留裁切后的视野，旋转不能重新显示已裁掉的像素或恢复已裁掉的轨迹点。旋转后出画点和裁空角色失效；RGB和目标画布以-1填充，权重以0填充，颜色对比度扰动后也保持人工黑角。腕图只做颜色，Action／Skill／Gripper不变。

`rotation_prob`默认0，`rotation_degrees`默认5。角度在裁切和颜色参数之后抽样；同seed的A2／A3拥有相同crop／color参数，A3只增加旋转。配置仍写入checkpoint并参与resume检查；不把±5°视为已经验证有效的范围。

增强在 batch 移到设备后、模型前向前执行，位于 [augmentation.py](../../src/openpi/augmentation.py)。`model.augment_images` 保持false，使用独立的 `image_augmentation` 配置。其默认关闭；配置写入checkpoint，恢复时禁止改变增强策略，旧checkpoint省略字段时按关闭处理。`metrics.jsonl` 的 `augmentation` 记录全局裁切／颜色应用数量、轨迹／角色有效数量，以及扣除缩放倍率后的前景面积估计；不额外上传W&B。

启动增强对照时，将下方配置换为 `color.json`、`color_crop.json` 或 `color_crop_rotation.json`；各有独立输出目录和run名。先比较标准LIBERO和Plus，再核对Milk任务的首次正确抓取率。轻裁切和小角度旋转改变取景，仍保留原始对象共现关系，不能单独证明或解决语言—对象绑定问题。

人工核验入口：[十条轨迹增强对照](../../data/libero/atomic-vla/check/test/augmentation/README.md)。视频固定每条轨迹的一组随机变换以便观察；训练对每个采样行重新抽样。

旋转另见 [A0／A2／A3核验](../../data/libero/atomic-vla/check/test/augmentation/rotation/README.md)；渲染命令为 `python -m scripts.preprocess.libero.review_augmentation --rotation experiments/eskill_libero40_direct/color_crop_rotation.json --output /path/to/new/review --seed 43`。

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

2026-11-13 增强：68项CPU回归通过，覆盖不同分辨率共享坐标、出画mask、原色VAE目标、黑色背景、Torch RNG恢复、增强配置变更拒绝恢复及旧快照默认关闭。十条真实训练轨迹覆盖六类技能，共1,744帧；A2中轨迹点与角色有效数量均保留100%，前景面积估计保留99.936%。这些是固定核验样本的统计，不代替全量训练分布或效果验证；视频及保存标签已回读核对，人工验收仍待用户查看。

加入旋转后74项CPU回归通过；另生成同十条A0／A2／A3视频和32张独立角度样本，预览显示监督权重轮廓。保存坐标按独立NumPy公式校验，A2旧参数／标签逐值不变；A3该十条的有效轨迹点与角色均保留100%，前景面积估计保留99.905%。角度、裁切、颜色及来源记录已保存，人工验收待用户查看，未启动正式训练。
