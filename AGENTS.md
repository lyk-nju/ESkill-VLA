# ESkill-VLA

- 按 OpenPI 主结构组织：运行时接口在 `src/openpi/`，离线处理在 `scripts/preprocess/`，薄入口在 `scripts/`，用法在 `examples/`。`data/` 保存来源配置、标签与核验产物，不再放运行时代码。
- 沿用 AtomicVLA 的文件分工：`policies/eskill_dataset.py` 读取样本及监督，`policies/libero_policy.py`、`robotwin_policy.py` 做各平台输入输出适配；`training/data_loader.py` 组装采样、变换及 batch，`training/config.py` 保存训练配置，通用变换放 `src/openpi/transforms.py`。策略推理封装归 `policies/policy.py`、`policy_config.py`。这些模块随实际实现添加，职责表见 `src/openpi/policies/README.md`。
- 整体工程组织以 AtomicVLA 为主要参照，模型统一放在 `src/openpi/models/`，采用单一 PyTorch 实现，不建立 `models_pytorch/`。`models/pi0.py` 承载 π0.5 基座，`models/pi0_eskill.py` 组合本项目分支并实现前向、损失和采样，结构参数归 `models/pi0_config.py`；训练循环入口为 `scripts/train.py`。模型方法以 `docs/design.md` 的新设计为准，目录参照不意味着引入 AtomicVLA 的 Think／Act 或 MoE。
- 修改数据相关代码前，阅读 `docs/data.md` 和 `data/AGENTS.md`；其中时间、事件、区域、来源及人工验收约定同样适用于 `scripts/preprocess/`、`src/openpi/policies/` 和训练接入。
- 保持单向依赖：预处理可使用数据字段和读取器；数据读取器不导入预处理、仿真、视觉模型或训练器。训练加载与配置放 `src/openpi/training/`，实际实现时再新增相应模块。
- 当前 JSON＋NPZ 草案已接入 AtomicVLA Dataset、训练与策略接口。ESkill 分支已做 CPU 缩小配置测试；原版 π0.5 已用真实预训练权重做两种数据读取的四卡 LoRA 短程测试（40 任务各一条、120 个窗口、每组 32＋8 步），记录见 `data/libero/atomic-vla/check/test/pi05_lora_v1/README.md`。ESkill 预训练 VAE、完整分支 CUDA 训练与闭环仍待验证。不要把结构检查、短程训练或渲染成功写成训练收敛／人工验收完成。新增模型代码统一放 models/，不修改全局 Transformers 安装；基座权重加载保持显式，正式配置不使用随机权重兜底。
- 历史 `check/` 文件保持原内容；用户要求精简时可核验后归档，当前导航同步更新，保留标签正在引用的来源文件。AtomicVLA 历史归档见 `data/libero/atomic-vla/check/archive/README.md`。代码迁移映射在 `scripts/preprocess/README.md`，迁移前代码快照在 `examples/archive/`。新增回归结果另存对应来源的 `check/test/`，视频放 `check/test/video/`。
