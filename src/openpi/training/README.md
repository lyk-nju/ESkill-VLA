# 训练接口

训练侧沿用 AtomicVLA／OpenPI 的职责：`data_loader.py` 创建 Dataset、组装采样、应用配置中的变换并组 batch；`config.py` 描述数据来源、平台适配、模型和训练参数。各平台适配归 `policies/libero_policy.py`、`policies/robotwin_policy.py`，可复用的变换归 `src/openpi/transforms.py`，具体分工见 [数据与策略接口](../policies/README.md)。

模型统一采用 PyTorch 并放在 `src/openpi/models/`：`pi0.py` 提供 π0.5 基座，`pi0_eskill.py` 组合技能、Grounding 和 Motion 分支，负责前向、损失与采样；`pi0_config.py` 定义模型结构参数，`lora.py` 对两个 Gemma 的 attention／MLP 线性层添加 LoRA。`training/config.py` 选择模型配置并创建模型，`scripts/train.py` 负责优化循环、DDP 和日志，`training/checkpoints.py` 保存恢复模型、优化器及各 rank 随机数状态；`scripts/compute_norm_stats.py` 计算训练集统计。

按 AtomicVLA 的加载顺序组织：Dataset 读取样本及事件监督 → 字段重排 → 平台输入适配 → 归一化 → 模型／监督目标变换 → batch。归一化统计由训练配置提供；推理复用平台输入适配及相应输入变换，Event 终点图像等未来信息只用于训练监督。

AtomicVLA 的 JSON＋NPZ 草案已接入 Dataset、平台适配、训练 batch、五项损失及策略回读。ESkill 已从完成 10k 步 SFT 的真实 π0.5 LoRA 权重开始，使用临时 SD VAE 和旧视觉去噪方案做四卡短程辅助联合训练，见 [检查记录](../../../data/libero/atomic-vla/check/test/sft_auxiliary_v1/README.md)；闭环效果尚待验证。RoboTwin 目前只有平台输入输出适配，既有 LeRobot 导出还未接入此 Dataset。

2026-11-02：冻结 FLUX VAE＋共享潜变量投影头＋区域加权 MSE 已接入训练，默认每角色 16 个 256 维 Query 与 256×256 原位画布。真实 π0.5 LoRA 已完成四卡 8 步训练、恢复至 10 步及无 VAE 策略回读，见 [grounding_mse_v1](../../../data/libero/atomic-vla/check/test/grounding_mse_v1/README.md)。三角色重建的人工质量验收、训练收敛与闭环效果仍待验证。

`model_type=pi05` 与 `model_type=eskill` 共用 `EpisodeDataset` 和统一 Episode 索引；前者只构造基础输入与动作，后者额外构造事件、区域和二维 Motion 监督。`cameras` 明确输入视角顺序，第一视角用于区域与 Motion。`action_padding` 为 `repeat_last` 或 `mask`；后者要求 `model.mask_action_loss=true`。`norm_method` 默认 quantile，统计文件逐字段保存方法。LoRA 与精度规则保持不变。

本体状态输入由 `model.use_arm_state` 控制，默认开启以兼容已有 checkpoint。设为 `false` 时不创建状态投影层，Motion／未来技能分支不接收状态 Token；训练变换不归一化或传递 `arm_state`，臂身份与存在标志保留。该开关与 `discrete_state_input` 独立，完全关闭数值状态需二者均为 `false`，见 `examples/libero/train_eskill_no_state.json`。模型结构改变后应从 SFT 开始新辅助实验，不修改旧 ESkill checkpoint 的配置后恢复；保存与回读沿用 checkpoint 中的开关。已有来源统计中的 arm_state 项可以保留，但此配方不使用它。

`torchrun` 启动时自动启用 NCCL／DistributedSampler；`batch_size` 按每卡解释。LoRA checkpoint 仅保存可训练参数与持久 buffer，回读时先加载配置中的原始基座，因此需保留 `pretrained_path` 及 tokenizer。恢复需使用原 world size，按 epoch 和步数恢复采样及各 rank 随机状态；默认非 LoRA 模式仍保存完整权重。

训练配置支持 JSON、YAML。正式训练统一读取项目根目录的本地 `wandb.json`，认证、entity 和 project 不在各训练配置中重复。run 默认按模型、微调方式和全局 batch 命名，例如 `pi05_lora_bs128`；可在训练配置中用 `run_name` 添加数据集、seed 等后缀。命名、上传与恢复约定见 [AGENTS.md](../../../AGENTS.md#训练记录与命名)。

只由 rank 0 按 `log_every` 上传 W&B，首步与末步也记录。命名分区如下：

| 分区 | 指标 |
| --- | --- |
| `train`、`val` | `loss`、`action_loss`、`grounding_loss`、`skill_loss`、`traj_loss` |
| `aux_train`、`aux_val` | `grounding_ee`、`grounding_obj`、`grounding_goal`、`current_skill_accuracy`、`future_skill_accuracy`、`motion_l2_px` |
| `chart` | `learning_rate`、`grad_norm`、`time_per_step`、`step` |

以 `chart/step` 为横轴；指标名不带开头的 `/`。每侧最多 11 项，含 chart 共最多 26 项自定义标量。`skill_loss` 是当前与未来技能原始损失之和，分支损失不乘总权重，总 `loss` 按配置权重计算。六项辅助摘要由 `training/metrics.py` 的固定清单选择，零有效数量时不上传该项。纯 π0.5 仅记录总／动作损失及通用指标。

详细诊断继续保存在本地 `metrics.jsonl`／`validation.jsonl`：当前／未来原始技能损失、类别频率、有效计数、Grounding 前景／背景、Motion 各偏移和真实／补齐点误差，不自动展开为 W&B 曲线。所有诊断按全局有效数量归一化，val 累计整个子集后再求均值。技能频率为有效标签位置的 argmax 频率；Motion 像素误差为归一化 UV 欧氏误差乘模型输入边长（正式配置 224）。W&B 的 Motion 摘要包含所有有效点，真实／补齐来源分项在本地查看。W&B config 与 tags 标明实际配方，在线恢复沿用同一 run；`smoke=true` 默认不创建 run，SDK 检查使用离线模式。新规则不会删除既有 run 的历史字段，已有面板可隐藏旧曲线，新 run 按精简清单记录。

`wandb.json` 不进入 Git；写入 `config.json`、checkpoint 和 W&B 的配置均清空 `key`。SDK 不自动上传代码或 Git diff。正常结束和异常退出都会关闭已创建的 run，本地 `metrics.jsonl` 继续逐步保存。

带有效性 mask 的损失按全局有效项数归一化：各卡反传 `world_size × local_sum / max(global_count, 1)`，与 DDP 梯度平均配合；日志继续对各卡返回值求平均。视觉监督对三个角色分别统计全局有效数量，本卡无标签也参加对应通信；全局无标签的项为零。此约定适用于当前每步一次 forward/backward 的 DDP 训练。

普通 π0.5 SFT 使用统一 Episode 中的双相机、源状态和动作，配置 `cameras=["image","wrist_image"]`、`action_padding="repeat_last"`、`norm_method="quantile"`。补齐动作不延长未来技能标签，Motion 使用独立有效性规则。

训练模块可以依赖 `openpi.policies.eskill_dataset`，数据读取模块不反向导入训练器或 `scripts.preprocess`。采样动作窗口、每臂 Event 和终点监督的具体张量组织以项目设计及后续实现为准。

Stage 1 采用[当前事件表征 → 未来技能序列 → 逐步条件化动作](../../../docs/design.md)。采样时按动作记录的公共索引查每臂 Event，派生 `future_skill_labels[2,H]` 与 `future_skill_valid[2,H]`；不扩充预保存格式。末尾 padding、缺标签及不存在臂排除分类监督。Goal 保留当前 Event 终点；Motion 已改为每臂 8 个 tokens、间隔 4 个来源步的二维点，允许跨 Event，末尾 `hold_last` 点按有效性参与轨迹监督。`motion_queries / motion_stride / motion_tail_policy` 随模型配置保存，Dataset 自动使用相同设置。

训练 batch 将未来技能、动作、终点图像和真实轨迹放在 targets；在线模型只接收当前／历史观察、指令、本体状态及机器人配置。当前／未来技能都按预测类别查同一张技能表，未来技能向量逐步注入动作 Token。第一版先接通完整新方案，完成小批前向／梯度检查、训练与闭环运行，不预建多组消融配置。

当前 ESkill 默认配置：主相机当前帧；LIBERO 单臂占左槽，右槽硬件有效性为 false；源 15 维状态中的后 8 维组成 π0.5 离散状态提示，完整 15 维进入本体条件；7 维源动作不额外做增量变换，填充至 32 维并屏蔽填充损失。输入缩放至 224；监督画布为 256；bbox 四周固定留边 4 像素，黑色背景，保持原布局。Motion 接受米制 world 末端位置与主相机标定，投影并按观察输入 resize／pad 归一化后监督八点；未提供标定时屏蔽该项。轨迹 head 为共享 LayerNorm → Linear(d,d) → GELU → Linear(d,2)，默认 d=256；SmoothL1 的 beta=0.05 按归一化坐标解释，224 画布每轴对应 11.2 像素，实际权重仍待正式训练核验。`trace_scale` 只为读取历史 SFT 配置保留，新二维损失不用该米制缩放。默认冻结 VLM，训练动作专家及 ESkill 分支。从 SFT 开始的配置另保留双相机、无离散状态提示及完整动作 MSE，冻结基座并更新 LoRA 和辅助分支，见下节。

新回归方案的第一版参数已确定：每角色 16 个 256 维 Token，共享两层 Query Decoder；画布改为 256×256，仍在源图中留边 4 像素，观察输入保持 224×224。16 个 Token 按 4×4 排列，共享 LayerNorm → 256→512→1024 MLP 为每个 Token 预测 16×8×8 块，拼成 16×32×32 潜变量；中间激活为 GELU。回归头与 256×256 画布已设为默认。旧十条子集检查配置已删除，新训练使用全量 `libero40/index.json`；旧 SD VAE 配置与结果仅供历史追溯。

训练与验证按**每个任务内的完整父轨迹**划分，默认 90%／10%；任务身份来自主任务指令，SFT 文本与 Episode `instructions[0]` 统一空白／下划线后对应。每个任务按父轨迹哈希排序取验证配额，视角副本只算一条；验证数量四舍五入，至少两条时两侧均保留，单条仅进入 train。固定清单由 `scripts/prepare_split.py` 生成，SFT、事件格式与派生子集通过相同 `split_path` 复用，后续更改训练 seed 不改变清单。当前 LIBERO-40 为 40 任务、1,518 train／175 val，见 [逐任务核对](../../../data/libero/atomic-vla/check/test/task_split_v1/README.md)。

`val_interval=1000`、`val_max_batches=32` 控制验证；末步也验证。子集和 Flow Matching 噪声／时间由固定清单 seed 决定，同一配置每次复用；多卡分配唯一、不重复的样本，每卡最多 32 batch。模型进入 eval/no_grad，验证后恢复训练 RNG 与 train 状态。按全局有效标签数汇总整个子集的损失，三角色分别归一化后相加；不平均各批次均值。W&B 记录 `val/*` 五项主损失及 `aux_val/*` 诊断，与同一步训练指标合并上传，详细记录写入 `validation.jsonl`。验证诊断累计整个子集的分子／有效数量后再求均值。`val_interval=0` 可关闭验证，仍保留原 train／val 归属。

统计只读 train 的原始状态／动作行，不重复计算动作窗口。`compute_norm_stats.py` 对各维有效有限值计算统计，只有使用 arm_state 的 LIBERO recipe 才计算该字段。统计方法逐字段保存；新实验默认 quantile。

## 从 SFT 开始辅助联合训练

Visual Grounding 已选择冻结 FLUX VAE 与潜变量直接 MSE；固定权重 revision 见 [Target Grounding](../../../docs/target_grounding.md)，编码使用 FP32 和 `latent_dist.mode()`，潜变量差值与 MSE 也使用 FP32。目标 VAE 编解码、回归头、四卡联合训练与保存恢复均已检查。`models.grounding.load_vae` 统一映射原始 FLUX 参数名并严格加载，也接受已转换的 Diffusers 权重；目标编码退出 BF16 autocast，VAE 保持 FP32、eval、无梯度。`sft_auxiliary_v1` 中的 `stabilityai/sd-vae-ft-mse` 是旧去噪链路检查临时选用的权重，历史配置与结果保持原义，不构成新方案实现或视觉表征质量验证。

设置 `init_checkpoint` 指向本项目纯 π0.5 SFT checkpoint，重新创建 ESkill 实验。`checkpoints.load_sft_weights` 校验基座／LoRA、结构及归一化后转移策略权重；新增分支独立初始化，`step_projection` 仍为零，技能文本原型在 SFT 权重加载后构造。优化器、步数、随机状态及 W&B run 从头建立。动作损失与四项辅助损失共同训练；LoRA 模式冻结原基座和 VAE，更新原 LoRA 与新增分支。

SFT 和辅助训练读取同一个自包含 Episode 索引，不再运行时拼接数据。相机、动作窗口及尾部规则按 recipe 显式指定，并随 checkpoint 保存。

`compute_norm_stats.py` 在配置包含 `init_checkpoint` 时继承其 state/actions 统计，要求逐字段方法与配置匹配。新增 arm_state 根据训练集按 norm_method 计算。2026-11-03 之前的 checkpoint 需要用当前代码重新训练。

这里的初始化与 `--resume` 分开：同结构 ESkill checkpoint 恢复优化器和进度，不再读取 SFT checkpoint；策略回读也只需 ESkill checkpoint、原始基座及 tokenizer。新潜变量回归方案可从纯 π0.5 SFT 权重初始化，旧去噪辅助 checkpoint 不作为新头的同配置断点恢复。示例和命令见 [LIBERO 两阶段检查](../../../examples/libero/README.md#从-sft-开始辅助联合训练)。

命令与正式权重要求见 [LIBERO 用法](../../../examples/libero/README.md)。已有 [ESkill CPU 核验](../../../data/libero/atomic-vla/check/test/training_interface_v1/README.md) 和 [原版 π0.5 四卡 LoRA 检查](../../../data/libero/atomic-vla/check/test/pi05_lora_v1/README.md)；完整训练配置示例中的路径与超参数需按实际资源设置。短程检查不提供任务成功率。
