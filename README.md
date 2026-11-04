# ESkill-VLA

**E** 代表 Event。项目以一个 Event 对应一次 Skill 执行为核心，学习技能、交互目标与运动实现。

研究怎样的数据组织与学习目标，能够让机器人从有限示范中理解任务目的，形成可泛化、可依据实际效果持续推进的行为能力。

从 **[docs/idea.md](docs/idea.md)** 开始。文档按“研究主线 → 各方向方法 → 整体实现”组织：

| 文档 | 职责 |
| --- | --- |
| [Idea](docs/idea.md) | 稳定的研究问题、归纳偏好、能力目标与验证假设 |
| [Event Organization](docs/event_organization.md) | 事件怎样定义、为什么这样组织、参考来源 |
| [Atomic Skill](docs/atomic_skill.md) | 当前技能识别、未来技能序列预测与逐步动作条件 |
| [Target Grounding](docs/target_grounding.md) | 三视觉锚点的当前设计、监督理由与参考来源 |
| [Action Representation](docs/action_representation.md) | 事件级整段粗运动的当前设计、监督理由与参考来源 |
| [Design](docs/design.md) | Stage 1 架构图、张量接口、数据结构、梯度及训练／推理流程 |
| [Data](docs/data.md) | 各模块需要的外部数据 |
| [Evaluation](docs/evaluation.md) | LIBERO／LIBERO-Plus 的环境、策略服务与闭环测评接入路线 |

[ref_papers](docs/ref_papers/README.md) 保存当前方法来源、按需参考卡与压缩背景笔记。交给实现 agent 时，先读 [Design](docs/design.md)，再按模块查参考；论文卡说明原方法和借鉴边界，不增加实现或实验要求。改变方法选择时同步更新方向文档与 Design，研究主张变化才修改 Idea。当前模型与训练接口已有初版实现，方法效果仍待正式训练与实验验证。

当前 Stage 1 采用 **当前事件表征 → 每臂未来技能序列 → 逐步条件化联合动作**：预测当前技能、三锚点和剩余粗运动，再预测固定动作窗口内的技能安排，按动作位置注入同一个 Flow Matching Action Expert。先跑通完整训练与闭环评测，以平均任务成功率评价整体表现，不把旧方案复现或多组消融作为前置要求。

2026-11-02：视觉监督已接入冻结 FLUX VAE＋共享投影头＋潜变量加权 MSE，保留三角色与原位画布。默认每角色 16 个 256 维 Token、256×256 画布和 FP32／mode 编码；已完成真实 π0.5 LoRA 四卡 8 步训练及恢复到 10 步。这验证训练接口，收敛、人工细节验收与 ESkill 闭环效果仍待验证。固定权重版本、参数和采用边界见 [Target Grounding](docs/target_grounding.md)。

数据方案见 **[docs/data.md](docs/data.md)**；候选来源、局部检查记录及来源字段讨论保留在 **[data/README.md](data/README.md)**。LIBERO 新训练按每个任务内的完整父轨迹 90%／10% 划分，共 1,518 train／175 val；SFT 与全量带标定版共用 [固定清单](data/libero/atomic-vla/split.json)，逐任务核对与新训练入口见 [task_split_v1](data/libero/atomic-vla/check/test/task_split_v1/README.md)。

## 项目组织

整体工程组织以 AtomicVLA 为主要参照，沿用 OpenPI 的层次；模型采用 PyTorch，统一放在 `models/`。当前文件分工如下：

```text
src/openpi/
  models/         模型实现
    model.py          上下文与动作条件的公共类型
    gemma.py          PaliGemma／Gemma、AdaRMSNorm 与层级 KV
    pi0.py            π0.5 基座与 Flow Matching 动作专家
    pi0_config.py     基座与 ESkill 模型结构参数
    pi0_eskill.py     组合分支，组织前向、损失和动作采样
    lora.py           两个 Gemma 流的 LoRA，保留原始基座权重名
    skill.py          当前技能、未来技能 Decoder 与共享技能表
    grounding.py      三角色锚点与训练期视觉监督
    motion.py         剩余运动 Token、FiLM 与粗路径头
  policies/       Dataset、平台输入输出适配、推理策略与权重加载
  training/       data_loader.py 组 batch；config.py 管配置；checkpoints.py 保存恢复
  transforms.py   图像、状态、动作及监督目标的数据变换
scripts/
  train.py        训练循环、日志、保存与恢复
  compute_norm_stats.py  训练集归一化统计
  prepare_data.py 准备数据的薄入口
  libero_render.py  渲染 LIBERO 已保存标签的薄入口
  serve_policy.py   使用官方 OpenPi 运行时启动 baseline checkpoint 服务
  preprocess/    来源读取、事件整理、实例绑定、原生区域、导出和渲染
examples/         LIBERO／Plus 测评客户端、数据使用说明、历史代码快照
data/             各来源说明、配置、标签和 check／processed 产物
docs/             研究设计与数据需求
tests/            数据语义、规则、读写、模型梯度与训练恢复回归
docs/ref_papers/  参考工作
```

LIBERO 已接通保存数据 → batch → 五项损失 → 保存恢复 → Python 策略推理。完整预训练原版 π0.5 已完成基座转换与严格加载。LIBERO／LIBERO-Plus 客户端已接入；服务支持独立官方 OpenPi 运行时及本项目纯 π0.5 双相机 SFT checkpoint，后者已完成四套技术闭环。[SFT 训练与测评进度](../workspace/experiment/pi05_libero_sft/README.md)单独记录。`src/openpi` 是本项目的命名空间；π0.5 底层参考 OpenPI 并保留权重命名，使用 Transformers 模型组件与本地 AdaRMSNorm，不修改共享环境安装。来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 代码布局与数据分工

```text
src/openpi/                 训练与推理共用的运行时代码
  policies/eskill_dataset.py  EpisodeReader 读数组和图像；EpisodeDataset 组装训练样本
  policies/libero_policy.py   LIBERO 输入／输出适配；在线服务按 checkpoint 的 cameras 取视角
  policies/robotwin_policy.py RoboTwin 输入／输出适配
  policies/policy*.py         从 checkpoint 构建在线策略
  training/config.py          TrainConfig：cameras、action_padding、norm_method 等训练规则
  training/data_loader.py     所有配方共用 EpisodeDataset，再接平台适配与变换
  training/checkpoints.py     保存恢复；资源按内容摘要核对，换路径或换机器可续训
  transforms.py               归一化、图像缩放、三角色画布、Motion 点投影到模型画布
  models/                     π0.5 与 ESkill 模型
scripts/                    入口：train.py、compute_norm_stats.py、serve_policy.py、attention_viewer.py
  preprocess/export.py        EpisodeWriter 保存并计算派生字段；validate_episode 保存时校验一次
  preprocess/libero/          LIBERO 导出与标定
```

正式数据为 `/data1/yuankai/robot-research/data/atomicdata/libero40/`：`index.json` 与 `episodes/<id>/{episode.json, arrays.npz, images.npz}`。每条 Episode 只用目录内相对引用，π0.5 与 ESkill 读同一索引；整个目录复制到其他机器即可训练。

| 内容 | 位置 |
| --- | --- |
| 每臂逐帧 Event 行号 `event_index`、源像素末端坐标 `effector_pixels[arm][camera]` | 导出时由 `EpisodeWriter.save` 计算，`validate_episode` 核对与 Event 表、相机参数一致 |
| 技能名、Goal 帧、对象区域 | 保留在 Event 表，不逐帧重复 |
| 动作窗口、尾部补齐、未来技能、Motion 选点 | `EpisodeDataset` 按配置切片和查表 |
| 三角色画布、图像缩放、归一化 | `transforms.py` |
| bbox 与 mask | 原样保存；训练只读 bbox，mask 用于核验和渲染 |

结构检查只在保存时做一次，读取不重复校验。旧格式数据、旧格式 checkpoint 及转换代码均已删除。此前的 π0.5 SFT checkpoint 已转换为当前格式，保存在 `/data1/yuankai/robot-research/workspace/experiments/patched/`，可直接作为 `init_checkpoint`。

`models/pi0_config.py` 管模型结构，`training/config.py` 管数据、优化与运行配置；`scripts/train.py` 调用模型计算损失并组织训练，数据处理留在 Dataset、适配器与 transforms。数据和推理接口的具体职责见 [数据与策略接口](src/openpi/policies/README.md)。模型仍采用[当前事件表征 → 未来技能序列 → 逐步条件化动作](docs/design.md)。

在本项目的独立开发环境中安装包，再从项目根目录使用薄入口；`--no-deps` 适用于依赖已齐全的环境：

```bash
python -m pip install -e . --no-deps
python scripts/prepare_data.py atomic-vla --help
python scripts/prepare_data.py robotwin --help
python scripts/libero_render.py --help
python -m scripts.preprocess.fields
```

具体运行见 [LIBERO 示例](examples/libero/README.md)、[RoboTwin 示例](examples/robotwin/README.md) 和 [预处理模块](scripts/preprocess/README.md)。LIBERO 已使用统一 Episode 格式；RoboTwin 仍使用已有 LeRobot 导出器，源数据尚未接入该格式。基础安装只包含字段与读取所需依赖；训练使用 `.[train,test]`，离线工具使用 `.[preprocess,test]` 或现有专用环境，仿真和视觉模型另用原环境。

现有 RoboTwin π0.5 环境已经包含另一份 OpenPI。复用它做离线处理时，从本项目根目录临时使用 `PYTHONPATH=src:. /path/to/python scripts/prepare_data.py ...`，保留该环境原有的包安装。
