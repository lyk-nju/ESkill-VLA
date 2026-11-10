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
| `model.future_bbox_expand_px` | 默认 8；Future 在现有范围上每边额外扩大 8 个 Grounding 画布像素，Current 不变；0 关闭额外外扩 |
| `model.skill_dim / skill_mlp / skill_depth` | 默认 512／2048／2，六类专门化＋共享专家 |
| `cameras` | 输入视角顺序；LIBERO 首视角必须为主相机 `image` |
| `model.action_patch_mask_prob` | 各视角 Action 视觉 K/V 的遮蔽概率，单值可广播 |
| `model.action_unmasked_prob` | 训练中保留完整视觉的样本比例 |
| `action_padding / norm_method` | Action 尾部规则与逐字段归一化，沿用 SFT 语义 |
| `skill_sampling / skill_sampling_max_weight` | 当前 ESkill 示例为 `true / 3.0`，启用有上限的平方根倒频率采样；配置类默认关闭 |
| `motion_grad_interval` | 0 关闭；Stage 2 示例为 100，在第 1 步及每 100 步测量 FM／辅助梯度的范数与余弦，仅本地日志 |

当前采用不带 Functional Reuse 的两阶段配方；本机仅运行技术测试与可视化，不启动正式训练。

| 阶段 | 初始化／预算 | 更新参数 | 数据与损失 |
| --- | --- | --- | --- |
| [Stage 1](../../../examples/libero/train_eskill_stage1.json) | 官方原生 π0.5；20k | Skill、Grounding、Motion、输入适配与辅助读出头 | 原图或水平镜像；五项辅助损失，无 Action loss |
| [Stage 2](../../../examples/libero/train_eskill_stage2.json) | Stage 1 的 20k 完整 checkpoint；另训练 40k | Skill、Grounding、Motion、辅助头及 Action Expert／Motion K/V | 原图；Action FM＋五项辅助损失 |

两阶段均冻结 VLM 和 VAE 编码器。Stage 1 冻结原生 Action；Stage 2 联合更新 Skill／Grounding／Motion、各输入适配、辅助读出头及 Action Expert／动作与时间投影／Motion K/V。FM 梯度经 Motion 回到 Grounding 和连续 Skill 分支，六项监督共同维持标签语义。Motion 按当前原图、任务、本体计算，不使用跨样本或镜像 latent 复用。Stage 1→2 只转移完整模型权重（含已学辅助头），重新建立优化器、步数、RNG 与 W&B run。

两份 recipe 均为 rank 0、每卡 batch 8、学习率 1e-5、warmup 1k，沿用稀有技能采样与固定 split。实际全局 batch=8×训练卡数；远端运行前按实际资源调整路径和 batch。每 5k 保存，每 1k 验证；不自动启动闭环测评。详见[实验记录](../../../experiments/eskill_libero40_mirror/README.md)。

`train_eskill.json` 保留为 32 步联合训练接口示例（`training_stage=joint`），不代表本次正式配方。联合模式可通过 `lora_rank` 配置 LoRA；当前两阶段仅支持 rank 0。

## 数据前置条件

ESkill 共用正式 Episode 索引、固定 split 和训练集归一化。当前 LIBERO 为 1,518 train／175 val；同一父轨迹及其视角、帧和派生样本不能跨集合。

当前本体输入为原始 EEF xyz、axis-angle 和双指位置，八维物理状态不做 quantile 归一化。离线主相机 EEF 起点与在线投影采用相同参考点和 resize／pad；state/actions 的 SFT 归一化仍保持。两组 Grounding 画布在 256×256，路径在 224×224，不能混用像素值。

Motion 直接按保存主相机帧序列查询，不要求精确时间戳。Dataset 在读取 RGB 前检查夹爪命令声明。`motion_stride=5` 控制未来点间隔；DataLoader 的 `frame_stride` 只选择当前训练样本，二者独立。

LIBERO 夹爪原命令已逐行核对并写入 `actions.gripper_convention`，不重复保存二值数组。按保存帧采样已可直接使用。Event 数量／帧长、开合命令及有效监督统计见 [ESkill 数据核验](../../../data/libero/atomic-vla/check/test/training/README.md)：

~~~bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --split data/libero/atomic-vla/split.json --subset train
PYTHONPATH=src:. python scripts/compute_norm_stats.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json
~~~

帧查询映射回真实来源索引，不插值缺失的 EEF 或命令。Motion 监督当前 Event＋最短 10 帧窗口中的真实路径与夹爪命令；+5、+10 可跨 Event，+15…40 仍按当前 Event 边界屏蔽。跨界点保留 `future_cross_event`，不因 `future_event_tail=true` 自动排除；Episode 尾部及终点填充排除主辅助损失。真实标签 mask 不控制在线 Motion 槽位。Action 使用原动作窗口，包含 recipe 规定的尾部重复，不按 Event 截断。

## 水平镜像

`training_stage=stage1`、`horizontal_flip_prob=0.5`：对每个抽到的训练样本以 50% 概率选择镜像；不是把每个 batch 扩成两倍。主相机和腕部同步翻转，在最终画布上同步翻转 Current／Future RGB 与前景权重，随后由冻结 VAE 编码。有效起点、未来二维点采用 `x'=1-x, y'=y`；像素中心归一化与训练一致，无效点及 mask 保持原值。

镜像样本将 `arm_state_mask[B,2]` 置为 false，Motion 屏蔽八维 `arm_state` 的整个投影贡献（包括 bias）；原图、验证、Stage 2 和推理默认对真实臂启用。二维起点继续按图像同步翻转，臂 presence、技能和夹爪命令监督仍有效。原始物理状态数值保留，文本状态输入必须关闭，避免从 tokens 泄漏该状态。不构造镜像 3D 或动作真值；Stage 1 不执行 Action 前向与损失。包含 left／right／leftmost／rightmost／clockwise 等方向词的任务只使用原图，不重写指令。验证与 Stage 2 均不翻转。

翻转抽样在主训练进程中、组 batch 之后执行，使用 checkpoint 保存的 rank RNG；恢复检查概率与阶段一致。核验视频右侧对符合条件的样本固定展示镜像，方便检查，并不代表训练时 100% 翻转。见[十条真实轨迹](../../../data/libero/atomic-vla/check/test/training/mirror_videos.md)。

## 技能采样

`skill_sampling=true` 仅对 train 生效。按固定父轨迹划分及 `frame_stride` 选出的实际训练动作行，查询它配对的当前主相机观察所属 Event；不按动作行时间误取下一 Event，也不按 Event 个数计算频次。只读索引与技能元数据，无需解码 RGB。

令 `N_s` 为技能 s 的有效样本臂数，`N_max` 为最大类计数，单臂样本权重为 `min(skill_sampling_max_weight, sqrt(N_max / N_s))`。只统计六类活跃且边界／技能有效的标签；缺失或未支持标签的样本保留权重 1。双臂样本整体抽取，权重取有效臂权重的最大值，不相乘。

按权重有放回抽样。每个 epoch 仍抽取 `ceil(N / world_size) × world_size` 个样本，不保证遍历所有训练帧；各卡从相同 `seed + epoch` 生成的全局序列中按 rank 切片，允许重复样本。batch、训练步数、动作窗口和 loss 权重保持 recipe 设置。恢复时重建相同序列并跳过已用 batch，采样开关或上限变化不能沿用 `--resume`。

启动后 `sampling.json` 保存各类原始计数、权重、预期占比与每 epoch 抽样量。验证集仍使用原来的固定随机子集；归一化仍只计算原始 train 数据，不按重采样频率重算。纯 π0.5 也可显式开启相同采样以进行公平对照，不需要实例化 ESkill 分支。

## 初始化、优化与恢复

本次 Stage 1 不使用 SFT：`pretrained_path` 指向官方 π0.5 转换权重，`init_checkpoint=null`，新增模块独立初始化。Stage 2 的 `init_checkpoint` 必须为完整 Stage 1 checkpoint，检查结构、数据／split、tokenizer 和归一化一致性，连同已学视觉读出头和冻结 VAE 一起严格加载；不继承优化器和步数。Stage 2 的 `vae_path` 与 Stage 1 保持相同内容。

联合训练接口仍可用 `init_checkpoint` 从本项目纯 π0.5 SFT 加载策略权重。ESkill 不构造文本技能原型。

当前 SFT 转接要求两阶段 `lora_rank`、`lora_alpha`、`backbone_dtype` 一致。完整 SFT 可接当前 rank 0 ESkill 模板；完整 SFT 直接切换为新的 LoRA 存储路线尚未支持，因为仅保存 adapter 无法恢复被 SFT 改变的冻结基座。

`--resume` 恢复同结构 checkpoint 的模型、Adam、各 rank RNG 和采样位置，检查训练语义与资源内容身份；ESkill checkpoint 独立保存，不重新读取初始化 SFT 文件。LoRA 恢复／推理仍需对应原始基座与 tokenizer。

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

`create_trained_policy(checkpoint)` 加载 Stage 2 或联合训练 checkpoint，省略 VAE／视觉读出模块。Stage 1 尚未学习 Action 如何读取 Motion，拒绝作为完整策略启动测评。ESkill 在线只读当前 RGB、任务、真实本体与相机信息，不读取 Event 或未来标签。服务握手要求真实主相机标定，并在服务端统一图像／起点 resize。实际精度由 checkpoint 的 `precision` 或显式覆盖决定。

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
