# ESkill-VLA

ESkill 中的 E 代表 Event。本分支的 **Motion Planner** 研究机器人能否从有限示范学到可复用的技能，并在目标改变时调整末端轨迹。例如，任务要求把牛奶放进篮子，Pick 时应关注牛奶，Place 时应转向篮子。当前实现与待验证的假设见 [模型设计](docs/design.md)。

## Motion Planner

模型读取一帧主相机 RGB 和完整任务指令。事件边界 Planner 在冻结的 PaliGemma 中编码 Start、End 和八个未来查询，用 Current 引导视觉读取。八个时间槽读取两个边界及其他时间槽，再由预测技能选择专家。共享轨迹头输出当前末端、本 Event 的末端终点和未来 +5 至 +40 帧的位置。

| 工作 | 入口 |
| --- | --- |
| 了解结构、标签和梯度 | [模型设计](docs/design.md) · [模型实现](src/openpi/models/motion_planner.py) |
| 训练和恢复 | [实验说明](experiments/motion_planner/README.md) · [gpu10 事件边界 20k 配方](examples/libero/train_motion_planner_boundary.json) |
| 检查轨迹与目标切换 | [逐帧诊断与同帧干预](scripts/analysis/motion_planner.py) · [A–E 定位评测](scripts/analysis/visual_primitive_eval.py) |
| 准备 LIBERO 数据 | [Grounding 数据生成配置](examples/libero/prepare_grounding_full.json) · [跨服务器同步数据](docs/remote_grounding_training.md#2-同步完整数据与权重) · [固定父轨迹划分](data/libero/atomic-vla/split.json) |

正式配方从非 LoRA KLAL 10k 初始化，八卡全局 batch 128，训练 20k。第一版保留两份 128 步开发配方用于对照。技术短测与评测口径见 [实验说明](experiments/motion_planner/README.md)。

## 数据与代码

Motion Planner 读取 `libero40_grounding/index.json` 及其引用的 Episode。固定划分覆盖 40 个任务、1,518 条训练轨迹和 175 条验证轨迹。模型代码在 `src/openpi/models/`，数据读取和推理接口在 `src/openpi/policies/`，离线标注与诊断在 `scripts/`。数据盘与环境遵循工作区存储约定。[数据协议](docs/data.md) · [数据来源与核验](data/README.md) · [预处理](scripts/preprocess/README.md) · [LIBERO](examples/libero/README.md) · [RoboTwin](examples/robotwin/README.md)

## 相关实验

Visual Primitive 单独评测 Current 和夹爪 Subject 的定位。历史联合动作模型让 π₀.₅ Action Expert 读取 Motion 条件，用于 LIBERO／Plus 闭环测评。各训练阶段与 checkpoint 入口见 [版本索引](docs/versions.md)。[Visual Primitive 实验](experiments/visual_primitive/README.md) · [联合动作直接训练](experiments/eskill_libero40_direct/README.md) · [两阶段镜像训练](experiments/eskill_libero40_mirror/README.md) · [全部实验](experiments/README.md)

论文笔记与代码许可见 [参考库](docs/ref_papers/README.md) 和 [第三方许可](THIRD_PARTY_NOTICES.md)。
