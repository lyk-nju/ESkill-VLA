# ESkill-VLA Agent 约定

先阅读本文件。模型方法见 [docs/design.md](docs/design.md)，数据协议见 [docs/data.md](docs/data.md) 和 [data/README.md](data/README.md)，训练接口见 [src/openpi/training/README.md](src/openpi/training/README.md)，测评协议见 [docs/evaluation.md](docs/evaluation.md)。具体实验配置和协议以该实验的记录为准。

## 当前研究入口

- 当前主线是多层原生注意力的 Visual Primitive，使用 `training_stage=visual_primitive`。四组短测模板、指标与历史 Grounding 对照见 [实验入口](experiments/visual_primitive/README.md)。
- 当前开发分支统一为 `visual-primitive`，使用当前 Current／Subject 架构，不增加历史信息流变体开关。运行与 checkpoint 说明见 [docs/versions.md](docs/versions.md)。
- 当前设计和操作放 `docs/`，联合动作说明放 `docs/history/`；实际运行配置、数据及结果记录保持原路径和原实验语义。

## 项目结构与代码

- 沿用 OpenPI／AtomicVLA 的职责划分：运行时代码放 `src/openpi/`，离线数据处理放 `scripts/preprocess/`，执行入口放 `scripts/`，示例配置和用法放 `examples/`。`data/` 保存数据来源、标签和核验产物。
- 模型统一使用 PyTorch，放在 `src/openpi/models/`；数据及策略适配放在 `src/openpi/policies/`，训练配置和数据加载放在 `src/openpi/training/`。模型设计以 `docs/design.md` 为准。
- 保持模块职责清晰、入口简洁；数据读取器不依赖预处理、仿真、视觉模型或训练器。复用已有实现，不修改全局依赖，不为历史结果改写其原有含义。
- 修改数据链路前阅读数据协议；使用真实权重时显式指定来源，结构检查不能冒充训练收敛或人工验收。

## 实验执行

- 用户明确要求执行实验后，完成必要检查并直接启动、跟踪和整理结果；方案的目标、数据、预算、对照或测评协议需要实质变化时，先与用户确认。仅讨论想法或可行性，不自动启动正式长训练。
- 启动前核对数据、划分、归一化、初始化权重、tokenizer、所需模型及运行环境；检查 GPU 占用、显存、进程和输出目录。不停止无关进程。
- 使用明确的配置文件；新实验使用独立配置、输出目录和 W&B run。记录模型分支、状态输入、相机、batch、学习率、训练步数及与对照实验的差异。
- 同一实验恢复沿用 checkpoint 和 W&B run；从 SFT 初始化新的辅助训练阶段使用 `init_checkpoint`，不继承 SFT 优化器和步数。
- 长任务记录启动命令、PID、开始时间、日志和产出路径。启动后确认进程运行、首批指标有限、checkpoint 能保存，且正式 W&B run 实际收到数据；异常如实报告。
- 启动和进度汇报写清 recipe、GPU、全局 batch、目标与当前步数、初始化来源、输出路径、W&B 链接以及下一里程碑。按已约定的频率验证、保存和测评，不擅自追加实验。

## W&B 配置与命名

- 正式训练默认读取项目根目录本机的 `wandb.json`，在线记录到其中配置的 entity 和 project。除用户明确指定关闭或离线外，不静默禁用 W&B；只由 rank 0 上传汇总指标。
- `wandb.json` 不进入 Git；密钥不得打印、提交或写入配置快照、checkpoint、日志和报告。W&B 上传的配置须脱敏。接口检查默认不上传正式项目；上传失败时保留本地记录并如实说明。
- run 名称使用“模型＋方法＋全局 batch＋数据集”；LoRA 必须标明 `lora`，默认 full 不写 `full`。例如 `pi05_lora_bs32_libero40`、`pi05_bs32_libero40`、`pi05_eskill_lora_bs32_libero40`、`pi05_eskill_bs32_robotwin`。
- `bs` 为每卡 batch size × 卡数 × 梯度累积次数。仅在区分实验确有必要时添加简短后缀；seed、学习率、步数等完整参数保存在配置和实验记录中，不堆进名称。名称与实际配置一致，实际冻结范围另行记录。
- 同一实验恢复沿用 run ID 和名称；新实验使用新 run ID。训练报告给出 run 名称或链接。

## W&B 指标

以 `chart/step` 为统一横轴，只上传实际存在且有效的指标；纯 π0.5 不伪造辅助指标。Grounding 专项仅记录 Current／Subject 与总损失及常规训练指标：Current 使用 `current_object_loss`，Subject 使用 `subject_grounding_loss`，不伪造文本、Motion 或 Action 指标。下表保留联合动作路线的指标约定。

| 分区 | 指标 |
| --- | --- |
| `train`、`val` | `loss`、`action_loss`、`skill_loss`、`current_object_loss`、`future_object_loss`、`traj_loss`、`gripper_loss` |
| `aux_train`、`aux_val` | `current_skill_accuracy`、`motion_l2_px`、`gripper_accuracy`、有效时的 `gripper_transition_accuracy` |
| `chart` | `learning_rate`、`grad_norm`、`time_per_step`、`step` |

- 上表用于 `eskill`；`skill_loss` 只表示当前技能 CE，两个对象损失及夹爪损失分别记录。分支损失不乘总权重，总损失按 recipe 权重计算；无有效标签的辅助摘要不上传。
- 不自动上传类别频率、各偏移、前景／背景及有效计数等详细曲线；完整诊断保存在本地 `metrics.jsonl`／`validation.jsonl`。同一步的 train／val 合并为一次 W&B log。
- 诊断按全局有效数量归一化，验证先累计整个验证子集再求均值。Motion 像素误差基于模型输入图像分辨率；具体定义以训练文档和实现为准。

## 数据与验证

- 避免训练与验证数据泄漏：同一父轨迹及其帧、视角、双臂和派生记录不得跨集合。数据划分、验证比例与频率以训练文档和实验配置为准。
- 归一化统计只使用相应训练数据；SFT 与后续辅助训练应保持数据划分和归一化语义一致。已用于训练的数据不能宣称为独立验证集；探索性全量训练须明确说明。
- 验证不反传或使用训练增强；报告验证数据范围，并区分损失验证、技术检查和闭环效果。

## 实验记录与产物

- 新实验说明放在项目内 `experiments/<实验名>/`，集中记录问题、recipe、对照、预算、结果和产出入口；权重、训练日志和其他大体积产物放在数据盘。历史实验保留在原有 `workspace/experiment/`，不自动迁移或改写。
- 每次运行记录 Git commit 和工作区状态、命令、环境、GPU、world size、seed、时间；记录数据及划分、归一化、初始化权重、模型和冻结范围、batch、学习率、训练步数、W&B run 与产物路径。配置快照须脱敏。
- 数据来源和预处理的检查放在对应数据集的 `check/test/`，视频放在 `check/test/video/`；检查产物不混入正式效果统计。

## 测评与汇报

- 使用实验约定的测评协议，记录 checkpoint、任务与初态、seed、回合数、控制和推理设置、视频要求。比较不同方法时检查协议与输入是否一致，差异须说明。
- 正式测评报告成功数／总数、成功率、异常及未完成数量，并保留逐任务结果；不同协议不混合统计。技术检查不当作模型效果，`technical` 成功率为 `null`。
- 完成后更新实验说明，简洁汇报实际步数、耗时、结果、W&B、checkpoint、报告位置和异常。未完成的训练、测评或人工验收不得写成已完成。
