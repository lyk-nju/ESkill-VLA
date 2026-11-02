# ESkill-VLA

- 按 OpenPI 主结构组织：运行时接口在 `src/openpi/`，离线处理在 `scripts/preprocess/`，薄入口在 `scripts/`，用法在 `examples/`。`data/` 保存来源配置、标签与核验产物，不再放运行时代码。
- 沿用 AtomicVLA 的文件分工：`policies/eskill_dataset.py` 读取样本及监督，`policies/libero_policy.py`、`robotwin_policy.py` 做各平台输入输出适配；`training/data_loader.py` 组装采样、变换及 batch，`training/config.py` 保存训练配置，通用变换放 `src/openpi/transforms.py`。策略推理封装归 `policies/policy.py`、`policy_config.py`。这些模块随实际实现添加，职责表见 `src/openpi/policies/README.md`。
- 整体工程组织以 AtomicVLA 为主要参照，模型统一放在 `src/openpi/models/`，采用单一 PyTorch 实现，不建立 `models_pytorch/`。`models/pi0.py` 承载 π0.5 基座，`models/pi0_eskill.py` 组合本项目分支并实现前向、损失和采样，结构参数归 `models/pi0_config.py`；训练循环入口为 `scripts/train.py`。模型方法以 `docs/design.md` 的新设计为准，目录参照不意味着引入 AtomicVLA 的 Think／Act 或 MoE。
- 修改数据相关代码前，阅读 `docs/data.md` 和 `data/README.md`；其中时间、事件、区域、来源及人工验收约定同样适用于 `scripts/preprocess/`、`src/openpi/policies/` 和训练接入。
- 保持单向依赖：预处理可使用数据字段和读取器；数据读取器不导入预处理、仿真、视觉模型或训练器。训练加载与配置放 `src/openpi/training/`，实际实现时再新增相应模块。
- 当前 JSON＋NPZ 草案已接入 AtomicVLA Dataset、训练与策略接口。原版 π0.5 已完成四卡 LoRA SFT；ESkill 已从其 10k 步 checkpoint 接入真实冻结 VAE，采用旧视觉去噪方案完成四卡短程辅助联合训练，记录见 `data/libero/atomic-vla/check/test/sft_auxiliary_v1/README.md`。ESkill 闭环效果仍待验证。不要把结构检查、短程训练或渲染成功写成训练收敛／人工验收完成。新增模型代码统一放 models/，不修改全局 Transformers 安装；基座权重加载保持显式，正式配置不使用随机权重兜底。
- Visual Grounding 已实现 AffordanceVLA Which2Act 风格的冻结 FLUX VAE＋共享 MLP 投影／空间拼接＋潜变量加权 MSE，保留每臂三角色及原位画布。默认每角色 16 个 256 维 Token、两层共享 Query Decoder、256×256 画布和源图 4 像素留边；VAE 使用 FP32／`latent_dist.mode()`，版本及分块规格以 `docs/target_grounding.md` 为准。`models/grounding.py` 统一严格加载原始 FLUX／Diffusers 参数，提供回归头与目标编码；推理不创建该监督模块。真实 π0.5 LoRA 四卡 8 步训练及恢复到 10 步的记录见 `data/libero/atomic-vla/check/test/grounding_mse_v1/README.md`，不能视为收敛或闭环效果验证。旧 `flux_vae_v1` 仅是目标编解码检查；旧 SD VAE／去噪配置与历史结果保持原义，新实验采用 `examples/libero/train_grounding_check.json`。
- 历史 `check/` 文件保持原内容；用户要求精简时可核验后归档，当前导航同步更新，保留标签正在引用的来源文件。AtomicVLA 历史归档见 `data/libero/atomic-vla/check/archive/README.md`。代码迁移映射在 `scripts/preprocess/README.md`，迁移前代码快照在 `examples/archive/`。新增回归结果另存对应来源的 `check/test/`，视频放 `check/test/video/`。

## 训练记录与命名

- 每次正式训练默认读取项目根目录的 `wandb.json`，在线记录到 `laiyuankia-nanjing-university/EskillVLA`；除用户明确要求关闭或离线外，不静默禁用 W&B。训练配置不重复填写认证信息，也不要用旧的 `wandb_info.mode=disabled` 覆盖统一配置。
- W&B 使用 `train`、`train/aux`、`val`、`val/aux`、`chart` 五个命名分区。train／val 各放 `loss`、`action_loss`、`grounding_loss`、`skill_loss`、`traj_loss`；`skill_loss` 合并当前／未来技能原始损失，总损失沿用配置权重。chart 放 `learning_rate`、`grad_norm`、`time_per_step`、`step`，以 `chart/step` 为横轴。aux 放三角色及前景／背景 Grounding MSE、当前／未来技能损失、准确率、预测类别频率，以及 Motion 各偏移、真实点／补齐点误差与有效计数。诊断按全局有效数量归一化，val 累计整个验证子集后求均值；无有效标签时结合 count 阅读。Motion 像素误差基于模型输入图像分辨率。纯 π0.5 SFT 只记录已有的动作及通用指标，不伪造辅助损失。详细记录同步保留在本地 JSONL。
- 正式 train／val 必须在每个任务内部按完整父轨迹做 90%／10% 划分，同一轨迹的帧、双臂、视角和派生记录不得跨集合。LIBERO 固定清单为 `data/libero/atomic-vla/split.json`，SFT 与辅助训练通过同一 `split_path` 复用，不能按全体轨迹总体划分。任务内验证数取最接近 10% 的整数；至少两条时保留两侧，单条仅入 train。统计只使用 train；从 SFT 初始化辅助训练时必须沿用其固定清单。旧全量 SFT checkpoint 不作为新独立 val 方案的初始化。
- 正式配置默认每 1,000 步及末步执行固定子集 val，每卡最多 32 个 batch；验证不反传、不做训练图像增强，恢复训练 RNG。验证上传对应的 `val/*` 主损失及 `val/aux/*` 诊断，并写入 `validation.jsonl`；纯 SFT 仅有总／动作验证损失。训练与 val 在同一步合并为一次 W&B log。
- 根据实际训练 recipe 命名 run，包含模型、微调方式和全局 batch，例如 `pi05_lora_bs128`、`pi05_eskill_bs128`、`pi05_full_bs128`；ESkill 使用 LoRA 时为 `pi05_eskill_lora_bs128`。`bs` 表示每卡 `batch_size × world_size`，`full` 表示非 LoRA，实际冻结范围以配置为准。
- recipe 的数据集、seed、学习率、训练阶段等有区别时，用训练配置中的 `run_name` 添加对应后缀，例如 `pi05_lora_bs128_libero40_seed42`；名称必须与实际配置一致，不使用含糊的 `test`、`run1`。同一实验恢复时沿用 run ID 和名称，新实验使用新输出目录；训练报告给出 run 名称或链接。
- 只由 rank 0 上传汇总指标；`smoke=true` 的接口检查默认不上传正式项目，需要验证日志时使用离线模式。`wandb.json` 仅保存在本机并排除 Git，密钥不得打印、提交或写入训练产物。
