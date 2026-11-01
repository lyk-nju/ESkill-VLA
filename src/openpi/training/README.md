# 训练接口

训练侧沿用 AtomicVLA／OpenPI 的职责：`data_loader.py` 创建 Dataset、组装采样、应用配置中的变换并组 batch；`config.py` 描述数据来源、平台适配、模型和训练参数。各平台适配归 `policies/libero_policy.py`、`policies/robotwin_policy.py`，可复用的变换归 `src/openpi/transforms.py`，具体分工见 [数据与策略接口](../policies/README.md)。

模型统一采用 PyTorch 并放在 `src/openpi/models/`：`pi0.py` 提供 π0.5 基座，`pi0_eskill.py` 组合技能、Grounding 和 Motion 分支，负责前向、损失与采样；`pi0_config.py` 定义模型结构参数，`lora.py` 对两个 Gemma 的 attention／MLP 线性层添加 LoRA。`training/config.py` 选择模型配置并创建模型，`scripts/train.py` 负责优化循环、DDP 和日志，`training/checkpoints.py` 保存恢复模型、优化器及各 rank 随机数状态；`scripts/compute_norm_stats.py` 计算训练集统计。

按 AtomicVLA 的加载顺序组织：Dataset 读取样本及事件监督 → 字段重排 → 平台输入适配 → 归一化 → 模型／监督目标变换 → batch。归一化统计由训练配置提供；推理复用平台输入适配及相应输入变换，Event 终点图像等未来信息只用于训练监督。

AtomicVLA 的 JSON＋NPZ 草案已接入 Dataset、平台适配、训练 batch、五项损失及策略回读。ESkill 分支的验证仍为 CPU 缩小配置和随机冻结 VAE；原版 π0.5 已用真实预训练权重执行两种数据读取的四卡 LoRA 短程训练。ESkill 预训练 VAE 和完整分支 CUDA 训练尚待验证。RoboTwin 目前只有平台输入输出适配，既有 LeRobot 导出还未接入此 Dataset。

`model_type=pi05` 选择原版 π0.5，只读 `LiberoDataset` 的观察与动作；`model_type=eskill` 是默认的新方案，读取 `ESkillDataset` 的全部监督。`data_format=episodes` 读取保存接口，`source_npz` 读取原始 LIBERO RLDS 摘取缓存。`lora_rank>0` 冻结基座并添加 LoRA；`backbone_dtype=bfloat16` 控制基座存储精度，`precision` 控制前向混合精度。配置示例与四卡命令见 [LIBERO 用法](../../../examples/libero/README.md)。

`torchrun` 启动时自动启用 NCCL／DistributedSampler；`batch_size` 按每卡解释。LoRA checkpoint 仅保存可训练参数与持久 buffer，回读时先加载配置中的原始基座，因此需保留 `pretrained_path` 及 tokenizer。恢复需使用原 world size，按 epoch 和步数恢复采样及各 rank 随机状态；默认非 LoRA 模式仍保存完整权重。

普通 π0.5 SFT 使用 `data_format=libero_sft`，由 `LiberoSFTDataset` 读取源主／腕相机和动作的 mmap 缓存；平台适配是8维源状态，模型可配置不把状态拼入提示词。`train_pi05_sft.json` 对齐上游 LIBERO 的10步动作窗口、分位数归一化和图像增强，使用完整32维动作 MSE。该模式在训练取窗时重复末条动作补齐尾部，不新增原始记录；ESkill 和此前接口检查仍按各自的有效性 mask 屏蔽 padding。`warmup_steps` 与 `adam_betas` 可配置，长程训练每次保存时另保留对应步数的 checkpoint。

训练模块可以依赖 `openpi.policies.eskill_dataset`，数据读取模块不反向导入训练器或 `scripts.preprocess`。采样动作窗口、每臂 Event 和终点监督的具体张量组织以项目设计及后续实现为准。

Stage 1 采用[当前事件表征 → 未来技能序列 → 逐步条件化动作](../../../docs/design.md)。采样时按动作记录的公共索引查每臂 Event，派生 `future_skill_labels[2,H]` 与 `future_skill_valid[2,H]`；不扩充预保存格式。末尾 padding、缺标签及不存在臂排除分类监督，当前 Goal／Motion 仍取当前 Event。

训练 batch 将未来技能、动作、终点图像和真实轨迹放在 targets；在线模型只接收当前／历史观察、指令、本体状态及机器人配置。当前／未来技能都按预测类别查同一张技能表，未来技能向量逐步注入动作 Token。第一版先接通完整新方案，完成小批前向／梯度检查、训练与闭环运行，不预建多组消融配置。

ESkill 当前落地范围：主相机当前帧；LIBERO 单臂占左槽，右槽硬件有效性为 false；源 15 维状态中的后 8 维组成 π0.5 离散状态提示，完整 15 维进入本体条件；7 维源动作不额外做增量变换，填充至 32 维并屏蔽填充损失。输入缩放至 224；监督画布为 128；bbox 四周固定留边 4 像素，黑色背景，保持原布局。路径用源 XYZ 差分，固定 scale=1；源坐标／单位的待核实状态不改写。默认冻结 VLM，训练动作专家及 ESkill 分支。上方纯 SFT 模式另用双相机输入与普通动作监督，不创建这些分支。

训练与验证按父轨迹哈希拆分，默认 seed=42、验证占比 0.1；统计只读训练轨迹的原始状态／动作行，避免动作窗口重复计数。当前训练入口保存五项损失、梯度范数、配置、统计、模型、优化器和随机数状态；同配置恢复会重建当前 epoch 的采样次序。验证集 loader 已提供，闭环测评和独立验证循环未自动运行。

命令与正式权重要求见 [LIBERO 用法](../../../examples/libero/README.md)。已有 [ESkill CPU 核验](../../../data/libero/atomic-vla/check/test/training_interface_v1/README.md) 和 [原版 π0.5 四卡 LoRA 检查](../../../data/libero/atomic-vla/check/test/pi05_lora_v1/README.md)；完整训练配置示例中的路径与超参数需按实际资源设置。短程检查不提供任务成功率。
