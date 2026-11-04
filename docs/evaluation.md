# LIBERO／LIBERO-Plus 测评

2026-11-01 已接入环境、官方资产、π0.5 策略服务入口和闭环客户端。`--runtime local` 可加载本项目纯 π0.5 SFT 或完整 ESkill checkpoint，复用相同客户端；服务根据训练配置公布相机及真实关节状态要求。[10k 步训练与测评进度](../../workspace/experiment/pi05_libero_sft/README.md) 另存，尚不能将服务接入视为已有模型成功率。

| 路线 | 用途 | 范围 |
| --- | --- | --- |
| LIBERO | 基础任务成功率 | Spatial、Object、Goal、LIBERO-10，共 40 个任务，默认每任务 50 回合 |
| LIBERO-Plus | 对扰动的鲁棒性 | 固定版本实际清单为 10,030 个变体，默认每变体 1 回合 |

Plus 七类扰动为对象布局、相机视角、机器人初态、语言、光照、背景、传感器噪声。分类从官方 `task_classification.json` 按任务名称关联，逐回合同时保留零起算任务 ID 与官方变体 ID。[官方说明](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md)

## 入口与依赖

```text
/home/yuankai/robot-research/
  LIBERO/                           官方基础任务源码入口
  LIBERO-plus/                      官方扰动环境源码入口
  openpi/                           固定版本的官方模型运行时
  workspace/envs/libero-evaluation/  独立环境配置、版本与资产清单
  ESkill-VLA/
    scripts/serve_policy.py          checkpoint 策略服务薄入口
    examples/libero/main.py          LIBERO 客户端薄入口
    examples/libero/evaluate.py      共用观测、动作执行与结果记录
    examples/libero_plus/main.py     Plus 客户端薄入口
```

两个基准都导入 `libero`，分别安装在独立 Python 3.8.20 环境中，各自使用 `LIBERO_CONFIG_PATH`。服务复用已有 π0.5 Python 3.11 解释器，加载工作区 `openpi/` 的官方模型、LIBERO 适配器与 checkpoint 归一化资产，通过 WebSocket 与客户端通信。启动命令见 [LIBERO 示例](../examples/libero/README.md#闭环测评) 与 [Plus 示例](../examples/libero_plus/README.md)。

策略进程独立选择运行时，避免两个 openpi 命名空间混用。`local` 读取本项目 checkpoint，其 `cameras` 决定输入视角和顺序。只有 ESkill 且 `model.use_arm_state=true` 时请求真实七维关节状态，与源八维状态组成 15 维输入；其他情况使用源八维状态。归一化使用 checkpoint 中各字段保存的方法。2026-11-03 之前的 checkpoint 需要用当前代码重新训练。

| 来源 | 固定 revision |
| --- | --- |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus | `4976dc30028e805ff8094b55501d532c48fec182` |
| OpenPi | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LeRobot（OpenPi 锁定的服务依赖） | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |

源码 checkout、Python、依赖快照、Plus 官方资产的固定版本与 SHA256 集中在 [环境说明](../../workspace/envs/libero-evaluation/README.md) 和 [manifest](../../workspace/envs/libero-evaluation/manifest.json)。它们与现有数据补框环境相互独立。

## 在线接口与协议

客户端根据服务握手中的 `cameras` 发送当前 RGB，默认保持官方双相机协议。源 `observation/state` 始终为 8 维（末端 xyz＋axis-angle＋双指位置）；当 `requires_joint_state=true` 时，额外发送同帧 `robot0_joint_pos`，字段为 `observation/joint_state`，服务按 joint[7]＋state[8] 拼接。缺少关节值、维度错误或非有限值均直接报错。相机按 OpenPi 约定旋转 180°，resize／pad 到默认 224；返回 `[horizon,7]` 的 OSC_POSE 增量与夹爪动作。记录数组 `states` 保持源 8 维，配置中的 `policy_state` 描述实际模型输入。

本项目固定协议采用上述 OpenPi 版本的默认值：seed=7，官方 `initial_states[trial]`，等待 10 步，每次执行预测块前 5 步，再用新观测重新预测。Spatial／Object／Goal／LIBERO-10 控制上限分别为 220／280／300／520。Plus 保留官方任务指令，语言扰动也沿用基准提供的原文。

这些设置随运行配置保存；更改配置或与其他工作比较时需逐项对齐。AtomicVLA、GuidedVLA 的默认执行长度和上限存在差异，不能仅凭模型名称比较论文数字。后续 baseline 与 ESkill 方法在同一基准上使用相同协议，并记录 checkpoint、训练数据版本和归一化配置。

预保存的真实 Event、bbox、终点图像和未来路径属于训练监督，不进入在线请求。ESkill 的技能、锚点与运动表征应由模型预测。主相机补框范围与策略使用腕部 RGB 是独立的配置，训练与推理应一致。

## 记录与验证状态

每次运行保存配置、源码 revision、服务元信息、逐回合结果、初态、实际执行动作、原预测块与动作块编号，按官方环境成功判定统计。LIBERO 按套件汇总，Plus 另按扰动类别汇总；总计采用逐回合微平均，并保留错误计数及分母。主相机视频 20 fps，动作块标记只添加到录像。

已完成四套基础任务各 1 例、Plus 七类各 1 例，共 11 例技术闭环；使用明确标识的零动作测试服务，每例 20 个控制步。它验证环境、通信、执行和记录，成功率统一为 `null`。详细证据见 [接入检查](../../workspace/experiment/libero_evaluation_integration/README.md)。实际权重加载与 baseline 指标在微调完成后验证。

正式实验放在 `workspace/experiment/<实验名>/`，按 [实验规范](../../workspace/experiment/README.md) 登记协议和预算后执行。训练数据为 `/data1/yuankai/robot-research/data/atomicdata/libero40/index.json`；后续 baseline 微调时需确定训练转换、数据版本和归一化统计，并与测评配置保持一致。

ESkill 服务已通过真实 LIBERO 环境的 6 步技术检查，使用已有 tiny CPU checkpoint、两次动作请求，错误数为 0。这是通信与执行链路验证，完整训练模型的任务成功率仍需正式测评。
