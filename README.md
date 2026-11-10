# ESkill-VLA

**E 代表 Event。** 项目研究如何从有限示范中学习可复用的视觉技能，将技能与对象目标结合，生成能够指导实际控制的交互运动。

当前设计为 **ESkill-VLA：Skill → Goal-conditioned Motion → Action**。每臂连续 Skill Query 经六类语义专家与 Shared Expert 形成先验，结合 Current／Future Object 和当前本体状态，预测未来 40 个保存帧的局部 Motion。只有高维 Motion 作为新增结构化条件进入 π0.5 Action Expert。

**ESkill 模型、数据采样、训练与在线接口已接入，并完成 CPU 技术回归；第一版每隔 5 个保存主相机帧采样，不强制物理时间戳。**

当前配方为 [Stage 1](examples/libero/train_eskill_stage1.json) 从官方原生 π0.5 初始化训练先验 20k，再由 [Stage 2](examples/libero/train_eskill_stage2.json) 联合训练 Skill／Grounding／Motion 与动作侧 40k；Stage 1 加入水平镜像，暂不加入 Functional Reuse。正式训练在远端进行，本机已完成技术检查与[十条镜像核验视频](data/libero/atomic-vla/check/test/training/mirror_videos.md)，待人工验收。运行方法见[训练接口](src/openpi/training/README.md)，新运行与历史记录见[实验入口](experiments/README.md)。

## 设计入口

从 [Idea](docs/idea.md) 开始；实现时以 [Design](docs/design.md) 为主。

| 文档 | 职责 |
| --- | --- |
| [Idea](docs/idea.md) | 研究问题与待验证假设 |
| [Event Organization](docs/event_organization.md) | 每臂事件、对象绑定及时间范围 |
| [Atomic Skill](docs/atomic_skill.md) | 连续 Query、六类预测路由与 6+1 小型专家 |
| [Target Grounding](docs/target_grounding.md) | 当前／Event 终点对象及冻结 VAE 目标监督 |
| [Action Representation](docs/action_representation.md) | 每臂八个固定时间 Motion tokens、二维路径与夹爪命令 |
| [Design](docs/design.md) | 架构、张量、梯度、Action-side Patch Masking 与实现计划 |
| [Data](docs/data.md) | 来源字段、时间与坐标对齐、ESkill 派生标签 |
| [Evaluation](docs/evaluation.md) | LIBERO／Plus、Object–Goal 重绑定与 Motion Donor 验收 |

[参考库](docs/ref_papers/README.md) 区分原论文机制与本项目的采用方式。方法修改同步更新模块文档与 Design；运行配置和实验结论保存在各自记录中。

## 项目组织

沿用 OpenPI／AtomicVLA 的层次，保持入口简洁、运行时与离线处理分开：

~~~text
src/openpi/
  models/          π0.5 与 ESkill 模型
  policies/        Episode 读取、平台适配、推理及权重加载
  training/        训练配置、data_loader、checkpoint
  transforms.py    图像、状态、动作与监督目标的变换
scripts/
  train.py
  run_libero.py    显式 recipe 的训练／测评编排
  compute_norm_stats.py
  prepare_data.py
  libero_render.py
  serve_policy.py
  preprocess/      来源读取、事件整理、实例绑定、区域、导出与渲染
examples/          LIBERO／Plus 客户端与平台用法
data/              数据来源说明、配置及人工核验入口
docs/              当前研究设计与参考工作
experiments/       实验 recipe、结果与产物入口
tests/             数据、模型与运行接口检查
~~~

模型结构参数属于 `models/pi0_config.py`，数据／优化／运行配置属于 `training/config.py`；训练入口只组织调用。具体接口见 [policies](src/openpi/policies/README.md) 和 [training](src/openpi/training/README.md)，它们说明现有实现，ESkill 接入时同步更新。

## 数据与运行入口

正式 LIBERO 数据位于 `/data1/yuankai/robot-research/data/atomicdata/libero40/`：`index.json` 与 `episodes/<id>/{episode.json, arrays.npz, images.npz}`。共 40 任务、1,693 条轨迹；[固定划分](data/libero/atomic-vla/split.json) 为 1,518 train／175 val，同一父轨迹及派生数据不跨集合，归一化只读 train。

已有逐帧对象／夹爪区域与相机标定。ESkill 采样与在线投影代码已接入；十条真实轨迹的起点投影与保存坐标一致。全量夹爪命令已核对并补齐语义声明；固定帧偏移监督已可构造，见 [ESkill 核验](data/libero/atomic-vla/check/test/training/README.md)。来源状态与保存格式见 [data/README.md](data/README.md)。

在依赖已齐全的独立开发环境中，从项目根目录使用薄入口：

~~~bash
python -m pip install -e . --no-deps
python scripts/prepare_data.py atomic-vla --help
python scripts/prepare_data.py robotwin --help
python scripts/libero_render.py --help
python -m scripts.preprocess.fields
~~~

具体用法见 [LIBERO](examples/libero/README.md)、[RoboTwin](examples/robotwin/README.md)、[预处理](scripts/preprocess/README.md)。LIBERO／Plus 服务与客户端已接入，ESkill 已通过 tiny 模型训练／恢复与在线回读；真实训练和闭环效果另行验收。

新环境、模型、数据与大体积实验产物放数据盘，遵循工作区存储约定。π0.5 底层参考 OpenPI；来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
