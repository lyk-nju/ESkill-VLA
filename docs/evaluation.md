# LIBERO／LIBERO-Plus 与 ESkill 行为验收

现有项目已接入 LIBERO／LIBERO-Plus 环境、策略服务与闭环客户端。ESkill 模型、当前 EEF／相机输入已接入，并通过 tiny checkpoint 回读；真实训练模型的闭环效果仍待实验验证。总体设计见 [Design](design.md)。

## 评测目标

| 路线 | 回答的问题 |
| --- | --- |
| LIBERO-40 | 基础任务能力是否保留；Spatial、Object、Goal、LIBERO-10 共 40 任务 |
| LIBERO-Plus | 对布局、相机、初态、语言、光照、背景、传感器噪声的鲁棒性 |
| Object–Goal 重绑定 | 熟悉对象与技能在改变目标关系后能否正确组合 |
| Motion Donor 干预 | 改变高维 Motion 是否产生符合目标的 EEF 运动与抓放时机变化 |

基础成功率、辅助损失与控制因果诊断分别报告，不互相替代。Plus 固定版本含 10,030 个变体，分类按官方任务名关联，逐回合保留任务 ID 和变体 ID。[官方说明](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md)

## 已有入口与环境

~~~text
/home/yuankai/robot-research/
  LIBERO/                           官方基础任务
  LIBERO-plus/                      官方扰动环境
  openpi/                           固定版本官方模型运行时
  workspace/envs/libero-evaluation/  环境与资产清单
  ESkill-VLA/
    scripts/serve_policy.py          策略服务薄入口
    examples/libero/main.py          LIBERO 客户端
    examples/libero/evaluate.py      共用观测、动作与结果记录
    examples/libero_plus/main.py     Plus 客户端
~~~

两个基准都导入 `libero`，分别使用独立 Python 3.8.20 环境与 `LIBERO_CONFIG_PATH`；策略服务使用独立 Python 3.11 环境，通过 WebSocket 通信。官方与本项目 `openpi` 运行时在独立进程选择，避免混用命名空间。运行命令见 [LIBERO](../examples/libero/README.md#闭环测评) 和 [Plus](../examples/libero_plus/README.md)。

上述标准服务／客户端已实现当前相机标定握手。工作区另一个 `libero-composition` 项目的部分历史诊断仍读取旧 `use_arm_state`、`future_skills` 及旧干预输出，不能只替换 checkpoint 路径来评测当前 Skill MoE→Motion→Action 架构。后续迁移需逐入口核对实时标定、当前技能与 Motion 接口及干预含义；不补造不存在的 Future Skill 输出来维持旧脚本表面可运行，也不改写旧测评结果。

| 来源 | 固定 revision |
| --- | --- |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus | `4976dc30028e805ff8094b55501d532c48fec182` |
| OpenPi | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LeRobot（官方服务依赖） | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |

版本、资产与 SHA256 见 [环境说明](../../workspace/envs/libero-evaluation/README.md) 和 [manifest](../../workspace/envs/libero-evaluation/manifest.json)。具体实验若采用其他 revision，必须独立记录。

## 在线输入与 ESkill 接入要求

客户端根据服务 `cameras` 发送当前 RGB。源 `observation/state` 为末端 xyz、axis-angle、双指位置共 8 维；ESkill 另请求当前主相机标定。在线输入与归一化按 checkpoint 配置处理。

ESkill 从这些源字段构造当前 EEF 位姿与实际夹爪状态，并通过真实主相机参数和图像变换得到二维起点。客户端按握手发送源分辨率 RGB 与每次实际相机标定，由服务端统一 resize；相机扰动不复用固定训练参数。十条真实轨迹的当前状态投影与离线保存坐标完全一致。

图像采用约定方向变换与 224×224 resize／pad。动作返回源 7 维控制命令。真实 Skill、Event 边界、对象 bbox、终点图像与未来路径均不进入在线请求。

## 基础协议与可比性

沿用项目固定协议作为起点：seed=7、官方 `initial_states[trial]`、初始等待 10 步、每次执行动作块前 5 步后重规划。Spatial／Object／Goal／LIBERO-10 控制上限为 220／280／300／520。LIBERO recipe 的动作窗口为 10，Flow Matching 推理步数为 10；实际值随运行配置保存。

每任务回合数按实验预算明确设置；标准全量 LIBERO 可采用每任务 50 回合，不能将少量试跑混入该分母。Plus 默认每变体 1 回合，保留基准提供的原始指令。

比较 SFT、ESkill 及其他方法时，统一任务／初态、训练数据与 split、归一化、观察视角、状态输入、推理精度、动作窗口、执行前缀和控制上限。ESkill 显式使用当前本体状态；若对照不使用，必须记录该差异，不能全归因于专家或 Motion。

`discrete_state_input=false` 只关闭 VLM 文本状态输入；当前 ESkill Motion 在原图、Stage 2 与推理时仍接收原始 8D EEF／双指状态，以及由相机投影获得的二维 EEF 起点。与仅图像／指令的旧 π0.5 60k 结果比较时，这两条状态信息通路均属于输入差异。旧结果可作系统配方参考，不能视为仅增减 ESkill 模块的严格消融。

严格对照另立匹配 recipe，核对状态信息、VLM 冻结范围、采样、镜像／视觉遮蔽、学习率、全局 batch、数据和验证协议。20k 先验＋40k 联合虽共 60k 更新，但 Action 只训练 40k；比较时分别报告总更新／计算预算和 Action 更新数，不能仅用“都是 60k”判定条件相同。当前不自动修改既有基线或启动新实验。

## Object–Goal 重绑定

在保持可执行初态的前提下改变对象与目标关系，逐项记录对象身份、目标位置、指令和成功判据。核查 SFT 初始化及 ESkill 训练的组合覆盖，只有两阶段均未训练的组合才称为未见组合。

同时观察当前对象绑定、预测终点与实际动作是否随任务关系改变。环境原生成功、物体到达目标区域、释放后稳定落位分别定义并报告；任务没有要求释放时，不额外改写其原生成功率。

## Motion Donor 干预

固定环境状态、当前观察、任务指令、采样噪声与原始 VLM K/V，仅替换 Motion 条件，比较原始与 donor 输出。候选 donor 应有明确运动／开合差异，并记录其来源、对应臂与适用性。

除动作张量差异外，检查短前缀执行后的 EEF 位移方向和大小、夹爪命令及实际开合变化，并保留接触与释放时机的回放。解码轨迹改变而实际行为几乎不变，不能记为 Motion 控制有效。

在同一 ESkill 结构和相同预算下比较有／无 Action-side Patch Masking；正常推理均使用完整视觉。辅助误差、attention 图或参数梯度存在，只说明局部机制工作，不能单独证明闭环收益。

## 记录与推进顺序

先完成保存帧／坐标／命令对齐及十条数据可视化核验，再做模型前向、梯度与服务技术检查，最后按另行确定的 recipe 执行训练和闭环测评。文档更新不自动启动正式实验。

新实验说明放 `experiments/<实验名>/`，权重、日志及大体积视频放数据盘；历史实验保留原位置。每次记录代码版本、配置、checkpoint、split、归一化、GPU／精度、seed、任务与初态、回合数、相机／状态、控制和采样参数。

结果保留逐回合动作块、实际执行动作、状态与视频，汇总成功数／总数、异常和未完成数量；LIBERO 按套件、Plus 按扰动类别汇总。技术闭环与模型效果分开，`technical` 成功率为 `null`。历史指标以原实验记录为准，不用 ESkill 设计改写旧结论。
