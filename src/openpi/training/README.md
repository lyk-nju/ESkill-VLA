# 训练接口

运行时代码沿用 OpenPI 层次：Dataset／平台适配归 `policies/`，张量变换归 `transforms.py`，模型归 `models/`；`data_loader.py` 负责采样、固定父轨迹划分和组 batch，`config.py` 管结构与运行设置，`scripts/train.py` 组织优化、验证和保存。

## ESkill 入口

`model_type=eskill` 选择 `ESkillConfig` 和 `Pi0ESkill`，实现 Skill-MoE → Object-conditioned Motion → Action；`model_type=pi05` 选择纯动作基线。模型结构见 [Design](../../../docs/design.md)。

| 配置 | ESkill 含义 |
| --- | --- |
| `model.action_horizon` | LIBERO 为 10；与 Motion 两秒时域独立 |
| `model.motion_offsets_s` | 每臂八个查询，默认 +0.25s…+2s |
| `model.skill_dim / skill_mlp / skill_depth` | 默认 512／2048／2，六类专门化＋共享专家 |
| `cameras` | 输入视角顺序；LIBERO 首视角必须为主相机 `image` |
| `timing_path` | 可选来源时间／命令语义清单；已有可靠 Episode 字段时可省略 |
| `model.action_patch_mask_prob` | 各视角 Action 视觉 K/V 的遮蔽概率，单值可广播 |
| `model.action_unmasked_prob` | 训练中保留完整视觉的样本比例 |
| `action_padding / norm_method` | Action 尾部规则与逐字段归一化，沿用 SFT 语义 |

`examples/libero/train_eskill.json` 是 32 步接口示例。初始化路径、时间清单、损失权重、mask 比例与预算需按实验填写；示例超参数未经真实训练调优。当前模板为冻结 VLM、训练新增模块与 Action 的 rank 0 配方。LoRA 仍通过 `lora_rank` 控制两个 Gemma 流。

## 数据与时间前置条件

ESkill 共用正式 Episode 索引、固定 split 和训练集归一化。当前 LIBERO 为 1,518 train／175 val；同一父轨迹及其视角、帧和派生样本不能跨集合。

当前本体输入为原始 EEF xyz、axis-angle 和双指位置，八维物理状态不做 quantile 归一化。离线主相机 EEF 起点与在线投影采用相同参考点和 resize／pad；state/actions 的 SFT 归一化仍保持。两组 Grounding 画布在 256×256，路径在 224×224，不能混用像素值。

秒级 Motion 必须有真实时间或有证据的控制时钟＋逐帧映射。ESkill Dataset 在读取 RGB 前检查时间与夹爪命令声明；训练入口在分配完整模型前执行该检查。不能以播放 fps 或过滤后的行号冒充物理时间。

检查命令及清单 schema 见 [ESkill 数据核验](../../../data/libero/atomic-vla/check/test/training/README.md)：

~~~bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --timing /path/to/verified_libero40_timing.json \
  --split data/libero/atomic-vla/split.json --subset train
PYTHONPATH=src:. python scripts/compute_norm_stats.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json
~~~

时间查询首版精确匹配（1e-6 秒容差），不插值缺失帧。Motion 只监督当前 Event 内的路径与夹爪命令；终点填充排除主辅助损失。真实标签 mask 不控制在线 Motion 槽位。Action 使用原动作窗口，包含 recipe 规定的尾部重复，不按 Event 截断。

## 初始化、优化与恢复

`init_checkpoint` 从本项目纯 π0.5 SFT 加载策略权重；新增分支独立初始化，优化器／步数／RNG／W&B run 重新开始。原始基座、tokenizer、动作结构、split 与归一化要匹配；ESkill 不构造文本技能原型。

当前 SFT 转接要求两阶段 `lora_rank`、`lora_alpha`、`backbone_dtype` 一致。完整 SFT 可接当前 rank 0 ESkill 模板；完整 SFT 直接切换为新的 LoRA 存储路线尚未支持，因为仅保存 adapter 无法恢复被 SFT 改变的冻结基座。

`--resume` 恢复同结构 checkpoint 的模型、Adam、各 rank RNG 和采样位置，检查训练语义与资源内容身份；时间清单也参与内容校验。ESkill checkpoint 独立保存，不重新读取初始化 SFT 文件。LoRA 恢复／推理仍需对应原始基座与 tokenizer。

`torchrun` 自动启用 DDP／NCCL（CPU 检查用 Gloo），`batch_size` 为每卡大小。SwiGLU 专家只对路由样本执行非空计算，未选中专家保留零梯度连接以对齐分布式归约。FLUX VAE 固定 FP32、eval、无梯度；模型前向可用 BF16 autocast。

## 损失与指标

ESkill 返回六项原始损失，`loss_weights` 必须完整指定：

| 损失键 | 监督／汇总 |
| --- | --- |
| `action` | π0.5 Flow Matching，按 recipe 决定是否屏蔽动作维度 |
| `skill` | 六类当前技能 CE，仅支持且有效的标签 |
| `current` | Current Object 加权 VAE latent MSE |
| `future_object` | Future Object 加权 VAE latent MSE |
| `trace` | 同 Event 有效二维位置 SmoothL1 |
| `gripper` | 同时刻有效夹爪命令 BCEWithLogits |

各项分开按全局有效数量归一化；模型提供 `loss_counts`，验证以相同有效分母累计整个子集。空标签不伪造准确率，终点填充、出画和缺测分别处理。

| 分区 | ESkill 指标 |
| --- | --- |
| `train`、`val` | `loss`、`action_loss`、`skill_loss`、`current_object_loss`、`future_object_loss`、`traj_loss`、`gripper_loss` |
| `aux_train`、`aux_val` | `current_skill_accuracy`、`motion_l2_px`、`gripper_accuracy`、有效时的 `gripper_transition_accuracy` |
| `chart` | `step`、`learning_rate`、`grad_norm`、`time_per_step` |

详细路由频率、标签量、每偏移覆盖率／误差、前景／背景、有效数量及裁剪后的各专家梯度范数保存在本地 JSONL；不自动展开为 W&B 曲线。默认以相邻有效查询的命令变化统计转换位置，也可由来源提供更细的 `gripper_transition` 标签。

W&B 配置读取本地 `wandb.json`，只由 rank 0 上传，保存配置清空密钥；正式运行与 smoke 检查按 [AGENTS.md](../../../AGENTS.md) 区分。验证使用固定抽样和噪声，并恢复训练 RNG。技术检查不能作为收敛或任务成功率。

## 在线加载

`create_trained_policy(checkpoint)` 加载自身保存的模型与变换，省略 VAE／视觉读出模块。ESkill 在线只读当前 RGB、任务、真实本体与相机信息，不读取 Event 或未来标签。服务握手要求真实主相机标定，并在服务端统一图像／起点 resize。实际精度由 checkpoint 的 `precision` 或显式覆盖决定。

CPU 技术测试覆盖 SFT 转接、训练／恢复与连续运行一致、策略回读、硬路由梯度及两进程全局梯度等价。真实模型训练与闭环效果需按独立实验 recipe 验证。
