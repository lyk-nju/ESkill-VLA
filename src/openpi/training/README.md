# 训练接口

运行时代码沿用 OpenPI 层次：Dataset／平台适配归 `policies/`，张量变换归 `transforms.py`，模型归 `models/`；`data_loader.py` 负责采样、固定父轨迹划分和组 batch，`config.py` 管结构与运行设置，`scripts/train.py` 组织优化、验证和保存。

## ESkill 入口

训练统一使用 `scripts/train.py --config ...`；周期闭环编排使用 `scripts/run_libero.py --config ...`。入口不再默认读取旧实验配置，当前模板与历史记录见 [实验入口](../../../experiments/README.md)。

`model_type=eskill` 选择 `ESkillConfig` 和 `Pi0ESkill`，实现 Skill-MoE → Object-conditioned Motion → Action；`model_type=pi05` 选择纯动作基线。模型结构见 [Design](../../../docs/design.md)。

| 配置 | ESkill 含义 |
| --- | --- |
| `model.action_horizon` | LIBERO 为 10；与 Motion 40 帧范围独立 |
| `model.motion_stride` | 默认 5，八个 token 对应未来 +5、10…40 个保存主相机帧 |
| `model.motion_min_horizon_frames` | 默认 10；最短窗口内允许跨 Event，名义约 0.5 秒；0 为严格 Event 对照，不跨 Episode |
| `model.motion_terminal_hold` | V3 配方 true；二维轨迹超过 Event／最短窗口较晚的终点后保持该位置并计入 loss，Episode 尾部同样保持；缺省 false 保留旧监督 |
| `model.future_bbox_expand_px` | 默认 8；Future 在现有范围上每边额外扩大 8 个 Grounding 画布像素，Current 不变；0 关闭额外外扩 |
| `model.skill_dim / skill_mlp / skill_depth` | 默认 512／2048／2，六类专门化＋共享专家 |
| `cameras` | 输入视角顺序；LIBERO 首视角必须为主相机 `image` |
| `model.action_patch_mask_prob` | 各视角 Action 视觉 K/V 的遮蔽概率，单值可广播 |
| `model.action_unmasked_prob` | 训练中保留完整视觉的样本比例 |
| `action_padding / norm_method` | Action 尾部规则与逐字段归一化，沿用 SFT 语义 |
| `skill_sampling / skill_sampling_max_weight` | 当前 ESkill 示例为 `true / 3.0`，启用有上限的平方根倒频率采样；配置类默认关闭 |
| `motion_grad_interval` | 0 关闭；Stage 2 示例为 100，在第 1 步及每 100 步测量 FM／辅助梯度的范数与余弦，仅本地日志 |
| `image_augmentation` | ESkill batch增强；默认关闭，支持双视角颜色与同步主图裁切，见下文；不使用原生 model.augment_images |

当前保留不带 Functional Reuse 的两阶段配方，并提供[直接联合训练对照](../../../experiments/eskill_libero40_direct/README.md)：从官方基座训练40k，关闭视觉增强／K/V遮蔽和技能重采样，全量解冻，lr=5e-5，八卡全局batch128，Action=1、五项辅助各0.1。该对照沿用当前模型与标签，不是旧架构复现。本机仅运行技术测试与可视化，不启动正式训练。

| 阶段 | 初始化／预算 | 更新参数 | 数据与损失 |
| --- | --- | --- | --- |
| [Stage 1](../../../examples/libero/train_eskill_stage1.json) | 官方原生 π0.5；20k | Skill、Grounding、Motion、输入适配与辅助读出头 | 原图或水平镜像；五项辅助损失，无 Action loss |
| [Stage 2](../../../examples/libero/train_eskill_stage2.json) | Stage 1 的 20k 完整 checkpoint；另训练 40k | VLM、Skill、Grounding、Motion、辅助头及 Action Expert／Motion K/V | 原图；Action FM＋五项辅助损失 |

Stage 1 的 `model.freeze_vlm=true`，冻结 VLM、原生 Action 和 VAE 编码器。Stage 2 示例改为 `model.freeze_vlm=false`，解冻视觉编码器、视觉投影和语言主干，联合更新 Skill／Grounding／Motion、各输入适配、辅助读出头及 Action Expert／动作与时间投影／Motion K/V；VAE 编码器保持冻结。FM 与辅助监督可回传至 VLM，六项监督共同维持标签语义。Motion 按当前原图和任务计算，不使用跨样本或镜像 latent 复用。Stage 1→2 只转移完整模型权重（含已学辅助头），重新建立优化器、步数、RNG 与 W&B run。

Stage 2 仍可显式设置 `freeze_vlm=true` 用于冻结对照；正式示例使用全量解冻，不使用 LoRA。先沿用统一学习率 `1e-5`，未另设 VLM 参数组。视觉编码器及语言层按已有 `gradient_checkpointing` 设置执行梯度重算。旧冻结 VLM 短测的显存／耗时不代表此配方，远端需重新核对实际 batch 的资源占用。

两份 recipe 均为 rank 0、每卡 batch 8、学习率 1e-5、warmup 1k，沿用稀有技能采样与固定 split。实际全局 batch=8×训练卡数；远端运行前按实际资源调整路径和 batch。每 5k 保存，每 1k 验证；不自动启动闭环测评。详见[实验记录](../../../experiments/eskill_libero40_mirror/README.md)。

`train_eskill.json` 保留为 32 步联合训练接口示例（`training_stage=joint`），不代表本次正式配方。联合模式可通过 `lora_rank` 配置 LoRA；当前两阶段仅支持 rank 0。

## 数据前置条件

单阶段增强对照见 [A0～A3](../../../experiments/eskill_libero40_direct/README.md#视觉增强对照)。`image_augmentation` 包括 `color_prob`、`brightness`、`contrast`、`saturation`、`crop_prob`、`crop_scale`、`rotation_prob`、`rotation_degrees`；概率默认0，颜色幅度默认0.15／0.15／0.1、裁切边长范围[0.95,1]、最大旋转5°。A3开启旋转概率1，在[-5°,5°]采样；正方向为图像顺时针。主图、对象画布／权重、未来轨迹共享“先裁切、再旋转”的仿射变换；图像采用逆向采样，点采用正向变换，各流只插值一次。裁切视野和旋转后画面共同约束有效性，黑角在颜色扰动后仍保持-1。腕图只做颜色，VAE目标保留原色。Action／Skill／Gripper不变；不采用遮挡、混合样本或联合训练镜像。

增强仅在训练前向前应用，不进入验证或在线策略。沿用每rank的Torch设备RNG保存／恢复；增强配置改变时拒绝resume，旧快照无字段时默认关闭。训练日志 `augmentation` 汇总所有rank的应用次数与有效监督计数，`role_area_after` 已乘裁切宽×高以扣除resize放大，面积是离散插值后的估计值。W&B指标白名单不变。

ESkill 共用正式 Episode 索引、固定 split 和训练集归一化。当前 LIBERO 为 1,518 train／175 val；同一父轨迹及其视角、帧和派生样本不能跨集合。

ESkill 输入固定为 RGB＋任务指令，模型配置要求 discrete_state_input=false。删除本体投影、状态门控与二维起点输入。离线未来 EEF 点按主图 resize／pad 生成监督，在线不需要真实状态或相机标定。共享统计文件可保留 state 项供其他模型使用，但 ESkill 仅使用 actions 归一化。两组 Grounding 画布在256×256，路径在224×224，不能混用像素值。

Motion 直接按保存主相机帧序列查询，不要求精确时间戳。Dataset 在读取 RGB 前检查夹爪命令声明。`motion_stride=5` 控制未来点间隔；DataLoader 的 `frame_stride` 只选择当前训练样本，二者独立。

LIBERO 夹爪原命令已逐行核对并写入 `actions.gripper_convention`，不重复保存二值数组。按保存帧采样已可直接使用。Event 数量／帧长、开合命令及有效监督统计见 [ESkill 数据核验](../../../data/libero/atomic-vla/check/test/training/README.md)：

以上状态指本机已更新的正式数据。`git pull` 不会更新外部数据盘中的 Episode；gpu10 等机器上的旧副本若缺少声明，需要先执行[跨机器夹爪元数据补齐](../../../data/libero/atomic-vla/check/test/training/README.md#跨机器补齐夹爪声明)。该流程使用随代码提供的源核验清单，比对每条已有动作和状态内容后仅补元数据，不要求旧机器的原始缓存路径可访问。

~~~bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --split data/libero/atomic-vla/split.json --subset train
PYTHONPATH=src:. python scripts/compute_norm_stats.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json
~~~

帧查询映射回真实来源索引，不插值缺失的 EEF 或命令。Motion 保留当前 Event＋最短 10 帧窗口内的真实轨迹与命令；+5、+10 可跨 Event，保留 `future_cross_event`。开启 `motion_terminal_hold` 时，更晚二维点保持该窗口最后观察的位置；Episode 越界也重复最后观察。显式 `trace_terminal_hold` 标记这些派生点，只有有效的二维点参与 Trace loss，夹爪不补造命令。关闭该开关时越界监督仍排除。真实标签 mask 不控制在线 Motion 槽位。Action 使用原动作窗口，包含 recipe 规定的尾部重复，不按 Event 截断。完整规则见 [Motion 监督](../../../docs/action_representation.md#event-边界与缺失标签)。

## 水平镜像

`training_stage=stage1`、`horizontal_flip_prob=0.5`：对每个抽到的训练样本以 50% 概率选择镜像；不是把每个 batch 扩成两倍。主相机和腕部同步翻转，在最终画布上同步翻转 Current／Future RGB 与前景权重，随后由冻结 VAE 编码。有效未来二维点采用 `x'=1-x, y'=y`；像素中心归一化与训练一致，无效点及 mask 保持原值。

镜像只同步翻转 RGB、Current／Future 画布和有效未来二维轨迹；臂 presence、技能和夹爪命令保持不变。模型不输入本体状态，因此没有 arm_state_mask，也不构造镜像3D状态或动作真值。Stage 1 不执行 Action 前向与损失。包含 left／right／leftmost／rightmost／clockwise 等方向词的任务只使用原图，不重写指令。验证与 Stage 2 均不翻转。

翻转抽样在主训练进程中、组 batch 之后执行，使用 checkpoint 保存的 rank RNG；恢复检查概率与阶段一致。核验视频右侧对符合条件的样本固定展示镜像，方便检查，并不代表训练时 100% 翻转。见[十条真实轨迹](../../../data/libero/atomic-vla/check/test/training/mirror_videos.md)。

## 技能采样

`skill_sampling=true` 仅对 train 生效。按固定父轨迹划分及 `frame_stride` 选出的实际训练动作行，查询它配对的当前主相机观察所属 Event；不按动作行时间误取下一 Event，也不按 Event 个数计算频次。只读索引与技能元数据，无需解码 RGB。

令 `N_s` 为技能 s 的有效样本臂数，`N_max` 为最大类计数，单臂样本权重为 `min(skill_sampling_max_weight, sqrt(N_max / N_s))`。只统计六类活跃且边界／技能有效的标签；缺失或未支持标签的样本保留权重 1。双臂样本整体抽取，权重取有效臂权重的最大值，不相乘。

按权重有放回抽样。每个 epoch 仍抽取 `ceil(N / world_size) × world_size` 个样本，不保证遍历所有训练帧；各卡从相同 `seed + epoch` 生成的全局序列中按 rank 切片，允许重复样本。batch、训练步数、动作窗口和 loss 权重保持 recipe 设置。恢复时重建相同序列并跳过已用 batch，采样开关或上限变化不能沿用 `--resume`。

启动后 `sampling.json` 保存各类原始计数、权重、预期占比与每 epoch 抽样量。验证集仍使用原来的固定随机子集；归一化仍只计算原始 train 数据，不按重采样频率重算。纯 π0.5 也可显式开启相同采样以进行公平对照，不需要实例化 ESkill 分支。

## 初始化、优化与恢复

本次 Stage 1 不使用 SFT：`pretrained_path` 指向官方 π0.5 转换权重，`init_checkpoint=null`，新增模块独立初始化。Stage 2 的 `init_checkpoint` 必须为完整 Stage 1 checkpoint，检查结构、数据／split、tokenizer 和动作归一化一致性，连同已学视觉读出头和冻结 VAE 一起严格加载；不继承优化器和步数。Stage 2 的 `vae_path` 与 Stage 1 保持相同内容。ESkill 的 SFT／Stage 1 权重转接均不要求 state 统计存在或一致；统计脚本也只继承目标模型实际使用的字段。

Stage 1→2 允许 `model.freeze_vlm` 从 true 改为 false，因为只改变训练范围、不改变权重结构。已有 Stage 1 权重可用于此初始化。恢复同一个 Stage 2 的 `--resume` 仍要求冻结设置一致；不能直接用新配置恢复旧冻结 VLM Stage 2 的优化器状态。

联合训练接口仍可用 `init_checkpoint` 从本项目纯 π0.5 SFT 加载策略权重。ESkill 不构造文本技能原型。

当前 SFT 转接要求两阶段 `lora_rank`、`lora_alpha`、`backbone_dtype` 一致。完整 SFT 可接当前 rank 0 ESkill 模板；完整 SFT 直接切换为新的 LoRA 存储路线尚未支持，因为仅保存 adapter 无法恢复被 SFT 改变的冻结基座。

`--resume` 恢复同结构 checkpoint 的模型、Adam、各 rank RNG 和采样位置，检查训练语义与资源内容身份；weight_decay 与 Adam betas 必须与保存配置一致，改变时在构造／加载优化器前报错，避免日志配置与实际参数组不一致。ESkill checkpoint 独立保存，不重新读取初始化 SFT 文件。LoRA 恢复／推理仍需对应原始基座与 tokenizer。

`torchrun` 自动启用 DDP／NCCL（CPU 检查用 Gloo），`batch_size` 为每卡大小。SwiGLU 专家只对路由样本执行非空计算，未选中专家保留零梯度连接以对齐分布式归约。FLUX VAE 固定 FP32、eval、无梯度；模型前向可用 BF16 autocast。

## 损失与指标

`loss_weights` 只指定当前阶段生效的项：Stage 1 为五项辅助损失，Stage 2 与联合训练接口均为下列六项。验证与 W&B 同样只记录生效项：

| 损失键 | 监督／汇总 |
| --- | --- |
| `action` | π0.5 Flow Matching，按 recipe 决定是否屏蔽动作维度 |
| `skill` | 六类当前技能 CE，仅支持且有效的标签 |
| `current` | Current Object 加权 VAE latent MSE |
| `future_object` | Future Object 加权 VAE latent MSE |
| `trace` | Event＋最短窗口内有效二维位置 SmoothL1 |
| `gripper` | 同时刻有效夹爪命令 BCEWithLogits |

各项分开按全局有效数量归一化；模型提供 `loss_counts`，验证以相同有效分母累计整个子集。空标签不伪造准确率，终点填充、出画和缺测分别处理。

启用终点保持后，`motion_l2_px` 包含有效真实点与派生停止点；本地诊断另存 `motion_observed_l2_px`／`motion_terminal_hold_l2_px`，不额外上传 W&B 曲线。`audit_training.py --terminal-hold` 与 `review_motion.py --terminal-hold` 使用同一标签实现，分别统计、显示真实来源及保持来源。更改此策略需开启新训练；resume 一致性检查会拒绝改变模型配置。

两阶段示例配方的初始权重如下；配置类未指定权重时仍默认为 1，不改写历史 checkpoint 配置。

| 损失 | Stage 1 | Stage 2 |
| --- | --- | --- |
| `action` | 不计算 | 1 |
| `skill` | 0.02 | 0.02 |
| `current` | 0.01 | 0.01 |
| `future_object` | 0.01 | 0.01 |
| `trace` | 0.25 | 0.25 |
| `gripper` | 0.05 | 0.05 |

两阶段使用相同的五项辅助损失权重，Stage 2 再加入权重为 1 的 Action FM，继续联合更新全部已约定模块。Trace 为归一化二维坐标回归，与 VAE latent MSE 及分类损失量纲不同，保留较大系数。这是参考公开配方与已有短测量级制定的首轮候选，不代表梯度平衡或效果最优；依据和来源见[实验配方](../../../experiments/eskill_libero40_mirror/README.md#损失权重依据)。正式 20k／40k 预算不变。Stage 1 的 loss 与梯度尺度也随本次系数调整，不能将旧单位权重短测的总 loss 或裁剪频率直接作为新配方参考。

| 分区 | ESkill 指标 |
| --- | --- |
| `train`、`val` | `loss`、`action_loss`、`skill_loss`、`current_object_loss`、`future_object_loss`、`traj_loss`、`gripper_loss` |
| `aux_train`、`aux_val` | `current_skill_accuracy`、`motion_l2_px`、`gripper_accuracy`、有效时的 `gripper_transition_accuracy` |
| `chart` | `step`、`learning_rate`、`grad_norm`、`time_per_step` |

详细路由频率、标签量、每偏移覆盖率／误差、前景／背景、有效数量及裁剪后的各专家梯度范数保存在本地 JSONL；不自动展开为 W&B 曲线。默认以相邻有效查询的命令变化统计转换位置，也可由来源提供更细的 `gripper_transition` 标签。

本地诊断另按真实技能汇总 `skill_<name>_accuracy`、`skill_<name>_motion_l2_px` 和 `skill_<name>_gripper_accuracy`，各有对应 `_count`。验证先累计分子／分母再求均值，计数为 0 的类别不作为有效表现；这些细分指标不自动上传 W&B。

W&B 配置读取本地 `wandb.json`，只由 rank 0 上传，保存配置清空密钥；正式运行与 smoke 检查按 [AGENTS.md](../../../AGENTS.md) 区分。验证使用固定抽样和噪声，并恢复训练 RNG。技术检查不能作为收敛或任务成功率。

## 在线加载

`create_trained_policy(checkpoint)` 加载 Stage 2 或联合训练 checkpoint，省略 VAE／视觉读出模块。Stage 1 尚未学习 Action 如何读取 Motion，拒绝作为完整策略启动测评。ESkill 在线只读当前 RGB 与任务指令，不读取机器人状态、Event、未来标签、二维起点或相机标定。服务端统一图像 resize。实际精度由 checkpoint 的 `precision` 或显式覆盖决定。

CPU 技术测试覆盖 SFT 转接、训练／恢复与连续运行一致、策略回读、硬路由梯度及两进程全局梯度等价。真实模型训练与闭环效果需按独立实验 recipe 验证。

最短监督窗口属于训练语义，恢复与 Stage 1→2 交接要求该字段一致。字段缺失的历史 checkpoint 解释为 0（严格 Event），不会静默按新默认 10 帧继续训练；新配方从原生 π0.5 开始重新训练。

## Motion 梯度诊断

`motion_grad_interval=100` 在第 1 步及其后每 100 步诊断实际配方加权后的梯度；阶段 1 没有 FM 时跳过，配置类默认 0 关闭。日志写入 `metrics.jsonl` 的 `motion_gradients`：

| 字段 | 含义 |
| --- | --- |
| `fm_norm` | `‖∇θM (w_action L_action)‖` |
| `aux_norm` | 五项辅助损失按本次 `loss_weights` 加权求和后，对 θM 的梯度范数 |
| `fm_aux_ratio` | `fm_norm / aux_norm`；辅助梯度为 0 时为 null |
| `cosine` | 两个梯度向量的余弦；任一向量为 0 时为 null |
| `loss_weights` | 本次诊断使用的实际权重，用于区分历史原始梯度记录 |

θM 为 `model.motion` 的共享参数（queries、交互层和 Skill FiLM），排除 `trace_head`／`gripper_head` 这两个辅助专用读出头；也不包含 Motion 以外的输入投影、Skill、Grounding 或 Action 参数。比较范围固定，避免把辅助专用参数加进分母。Skill CE 和两项 Grounding 损失不直接依赖下游 Motion 参数，其在这个范围的梯度为零；因此这里实际测到的辅助梯度来自路径与夹爪损失，但定义仍保留完整辅助损失和。

诊断复用本批次同一前向图、同一 FM 噪声／时间与同一 Action patch mask，在全局梯度裁剪及优化器更新之前计算。各分项 loss 日志继续保留未加权数值，总 loss 与梯度诊断使用实际配方权重。旧日志中不含 `motion_gradients.loss_weights` 的记录表示原始梯度，不能与新配方加权梯度直接混同。独立调用 `motion_gradient_balance` 不传权重时仍返回单位权重的原始梯度；训练入口显式传入当前配置。

DDP 下先对每组完整梯度向量跨 rank 取均值，再求范数与余弦；不平均各卡范数。诊断不累计 `.grad`，随后正常反传总损失。采样频率只控制观测，不改变权重、RNG 或采样位置；每次诊断增加两次定向反传及向量归约。零梯度不伪造方向相似度。这些详细指标不自动上传 W&B。

## 同一模型的阶段切换

`configure_stage` 支持反复切换。首次调用保存当时 joint 的可训练参数集合（标准入口先完成 LoRA 安装），返回 joint 时恢复该集合，保留原 VLM 冻结／LoRA 范围和冻结 VAE，恢复 Stage 1 曾关闭的 Action 参数。参数结构与权重不改变。

改变阶段会改变可训练参数集合，调用方须重建优化器与 DDP 包装；正式 Stage 1→2 仍按原入口重新构造模型和优化器。不要在首次配置之后替换网络模块或新安装 LoRA。

当前模型已删除 `state_projection`、`arm_state_dim`、`arm_state_mask` 和 `start_projection`，轨迹头直接输出绝对二维图像坐标。此前带状态或起点输入的 ESkill checkpoint 与当前结构不兼容，不能直接续训或加载后忽略多余权重；评测历史 checkpoint 使用其对应代码。新直接联合训练从官方基座开始，两阶段训练也需使用当前结构重新生成 Stage 1 checkpoint。
## Current Grounding 专项训练

`training_stage=grounding` 使用现有 v2/v3 Current decoder、每臂一组共享 Current Queries（正式配置 16 tokens）和冻结 VAE；`model.grounding_subject=true` 时增加一组夹爪 Subject Queries，不执行 Future、Skill、Motion、Action。每个观察每次只选一个目标，batch 不扩增。这里的专项阶段与历史 `stage1` 先验训练入口不同。

三类监督复用原有视觉编码器和 VAE：

- **Subject（夹爪）**：从融合文本前的主相机视觉 tokens 定位夹爪，使用小型 projection／Query decoder，不额外运行一遍视觉塔。固定输入图像时换目标文本，Subject 前向不依赖该文本。监督来自已有 `effector_regions.single.image` 完整框，沿用每边 4 源像素的 RGB 画布边距；框内的抓取物体自然保留，不增加持物实例标签。

- **Current 定位**：主相机 RGB＋当前目标短句。优先保留 `pick up the milk` 这类原动作模板；反事实替换完整目标名词短语，例如 `pick up the chocolate pudding`。空间关系复杂、动词与实体不兼容时退回 `Locate …`。同类实例用图像中的左右限定，无法消歧则不伪造实例编号。
- **完整任务→目标短句**：原始主图＋完整任务，以真实 Event 的当前交互目标短句做因果语言监督，复用原语言输出头。此路径不接收 Copy-Paste／Patch Masking，也不把原任务与反事实答案配对。反事实只增加定位监督，不是“抓错后自动改抓”的动作示范。

两条路径及验证／在线接口都只用主相机 `image`，不读取或编码腕图。在线 `create_current_grounding_policy` 先生成目标短句，再直接以该短句定位，不强行添加 `Locate` 前缀；生成未到 EOS／为空时返回 `incomplete_target_text`，不回退到真实标签。接口返回 `grounding`（Current tokens）与 `subject_grounding`（夹爪 tokens），尚未接管动作 rollout。没有开启 Subject 的历史配置后者为 None；新增示例配置已启用。推理不需要 Event、bbox、实例 ID、相机标定或本体状态。

增强按以下顺序在线执行：

1. **Object Copy-Paste**：与查询无关地抽一个可编辑实体和一个位移，失败即保留原图。使用离线原生可见 mask 搬像素、完整框更新标签。首版只编辑近乎全可见、远离夹爪的独立刚体，暂不搬容器、关节部件。源区（框外扩 2 px）／目的框不能覆盖其他实体或机器人；源区用邻近边界的中位颜色填充。没有在线分割、修复模型或三维场景合法性搜索；擦除区可能有可见色块，需人工验收。
2. **单目标采样与文本替换**：反事实候选只保留当前帧 `visible_fraction >= 0.8` 的实体（原生可见像素数／完整屏内 mask 像素数，`target_min_visible_fraction=0.8`）。遮挡或缺少可见比例的实体不作替换候选；没有合格候选就保留真实 Event 目标。抓着 Milk 时仍可查询可见的 Pudding；物体不可移动不意味着不可查询。仅读取已有比例，不增加在线分割或重渲染。移动后重新计算同类实例的左右描述；复制成功时，无可靠替换模板的原关系句也退回 `Locate`，避免移动参照物后继续使用失真的关系。
3. **构造 VAE 目标**：从编辑后、随机遮挡前的 RGB 及目标／夹爪各自的完整 bbox，沿用 `role_canvas` 的框外 4 源像素边距生成干净画布，再由冻结 VAE 编码；不平移旧 latent。
4. **随机 Patch Masking**：先排除当前文本目标和夹爪的完整 bbox＋各自每边 8 源像素边距，再加入夹爪／交互物局部保护区，再遮挡剩余 patch。相交的整个 patch 都受保护；不反复重采样，不保护所有其他实体。保护框使用 Copy-Paste 后的坐标，边界裁到图内。局部互动区使用原生框邻近关系的廉价代理，不宣称真实接触检测。

**遮挡不改变完整 bbox 的定义。** 真实 Event 及夹爪 Subject 原本部分／完全遮挡时继续保留完整屏内框与原有监督；上述可见阈值仅限制随机替换目标。VAE 画布仍是框内的真实 RGB（可能含遮挡者），不凭空还原被遮住的外观。当前完整框与边距在采样时已受保护，因此随机 Patch 不应再新增目标遮挡；`augmentation_invalidated` 保留为异常诊断，正常应为 0。文本生成监督不变。原生完全不可见另记 `native_invisible`，所选目标的可见比例记录为 `target_visible_fraction`。虚拟区域没有物理可见 mask，不参与随机替换。

| 配置 | Event 对照 | 在线增强 |
| --- | --- | --- |
| 文件 | `train_current_grounding_baseline.json` | `train_current_grounding.json` |
| 每个观察的定位目标数 | 1 | 1 |
| 反事实概率 | 0 | 0.5 |
| Copy-Paste 尝试概率 | 0 | 0.5，最大位移为图宽／高的 30% |
| Patch 应用概率／丢弃率 | 0 | 0.5／0.25 |
| Patch 尺寸 | — | 224 输入上 28 px，按源图缩放 |
| 当前目标／夹爪保护边距 | — | 完整框每边 8 源像素，`target_margin_px=8` |
| Current／Subject／目标文本 CE 权重 | 0.1／0.1／0.1 | 相同 |
| 相机／VLM | 仅主相机／解冻 | 相同 |

增强种子由全局 seed、轨迹、帧、epoch、抽样位置决定；DataLoader 将抽样键传给 worker，因此 persistent workers 与恢复时不会固定成永久的一张增强图。旧版未保护目标的 Patch 训练不可直接 resume 到此规则，应新开训练。验证完全关闭增强，仍是给定正确目标文本的定位验证，不能代替自主“生成→定位”的测评。

Subject loss 是夹爪 VAE latent 重建，与 Motion 的未来开合命令 `gripper_loss` 不同。数据 `role_images/weights/valid` 的 role 顺序为 `(current, subject)`；两项按各自有效数量归一化，共用 VAE 和 latent 读出头。缺失／出画的夹爪框只使 Subject 标签无效，原生遮挡不缩框、不抹除标签。

开启 Subject 会改变参数与优化器集合，应开始新的专项训练；模型配置默认关闭该分支以保留原动作 checkpoint 的参数结构，当前仅在 grounding 阶段启用。

逐样本 `metadata.augmentation` 记录尝试／成功搬动、实际搬动所监督目标、文本替换、Patch 比例、原生不可见、增强失效与有效监督数量；训练日志保留这些信息。W&B 上传 Current loss、`train/subject_grounding_loss`／`val/subject_grounding_loss`、目标文本 CE 与 teacher-forcing token accuracy，后者不是自主生成整句正确率。

本轮仍是 **32 步接口配置**，没有启动正式 π0.5 训练。样本范围为十条派生轨迹／2,198 帧／27 个 Event，继承原 split（7 train／3 val）。原 1,693 条数据没有全量迁移。真实图像／VAE 核验和增强开关吞吐结果见 [在线增强验收](../../../data/libero/atomic-vla/check/test/online_grounding/README.md)，操作入口见 [LIBERO 示例](../../../examples/libero/README.md#current-grounding-数据与训练接口)。
