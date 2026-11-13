# LIBERO／LIBERO-Plus 与 ESkill 行为验收

现有项目已接入 LIBERO／LIBERO-Plus 环境、策略服务与闭环客户端。ESkill 模型与 RGB＋语言输入已接入，并通过 tiny checkpoint 回读；真实训练模型的闭环效果仍待实验验证。总体设计见 [Design](design.md)。

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

当前 ESkill 标准服务不再要求在线相机标定或二维起点。工作区另一个 `libero-composition` 项目的部分历史诊断仍读取旧 `use_arm_state`、`future_skills` 及旧干预输出，不能只替换 checkpoint 路径来评测当前 Skill MoE→Motion→Action 架构。后续迁移需逐入口核对当前输入、技能与 Motion 接口及干预含义；不补造不存在的 Future Skill 输出来维持旧脚本表面可运行，也不改写旧测评结果。

当前架构的离线预测面板使用 `scripts/prediction_viewer.py`，独立的 Action Attention 对比使用 `scripts/attention_viewer.py`，实现分别放在 `scripts/analysis/`。数据真值渲染仍使用 `scripts/libero_render.py`，与模型预测分开。

| 来源 | 固定 revision |
| --- | --- |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus | `4976dc30028e805ff8094b55501d532c48fec182` |
| OpenPi | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LeRobot（官方服务依赖） | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |

版本、资产与 SHA256 见 [环境说明](../../workspace/envs/libero-evaluation/README.md) 和 [manifest](../../workspace/envs/libero-evaluation/manifest.json)。具体实验若采用其他 revision，必须独立记录。

## 在线输入与 ESkill 接入要求

客户端根据服务 `cameras` 发送当前 RGB 与任务指令。ESkill 握手为 `requires_state=false`、`state_dim=0`、`requires_camera_calibration=false`；不读取或发送当前位姿、夹爪开度、二维起点或相机标定。旧服务没有 requires_state 字段时，客户端仍沿用原先的8维状态协议；纯 π0.5 是否启用状态文本由其 checkpoint 配置决定。

ESkill 客户端按握手发送源分辨率 RGB，由服务端统一 resize。Motion 直接预测未来图像坐标，不在推理时执行 EEF 投影。

图像采用约定方向变换与 224×224 resize／pad。动作返回源 7 维控制命令。真实 Skill、Event 边界、对象 bbox、终点图像与未来路径均不进入在线请求。

## 基础协议与可比性

沿用项目固定协议作为起点：seed=7、官方 `initial_states[trial]`、初始等待 10 步、每次执行动作块前 5 步后重规划。Spatial／Object／Goal／LIBERO-10 控制上限为 220／280／300／520。LIBERO recipe 的动作窗口为 10，Flow Matching 推理步数为 10；实际值随运行配置保存。

每任务回合数按实验预算明确设置；标准全量 LIBERO 可采用每任务 50 回合，不能将少量试跑混入该分母。Plus 默认每变体 1 回合，保留基准提供的原始指令。

比较 SFT、ESkill 及其他方法时，统一任务／初态、训练数据与 split、归一化、观察视角、推理精度、动作窗口、执行前缀和控制上限。当前 ESkill 与关闭状态文本的 π0.5 SFT 均只使用 RGB＋语言；额外的技能／对象／轨迹／夹爪监督和视觉遮蔽仍需作为实验差异记录。

ESkill 固定 `discrete_state_input=false`，且没有独立本体或起点输入分支。历史 ESkill 曾使用15D或8D状态，不能宣称所有旧版均为纯视觉。输入对齐只满足公平对照的一项条件；归因 Skill-MoE 或 Motion 增益仍需匹配配方并进行模块消融。

严格对照另立匹配 recipe，核对状态信息、VLM 冻结范围、采样、镜像／视觉遮蔽、学习率、全局 batch、数据和验证协议。20k 先验＋40k 联合虽共 60k 更新，但 Action 只训练 40k；比较时分别报告总更新／计算预算和 Action 更新数，不能仅用“都是 60k”判定条件相同。当前不自动修改既有基线或启动新实验。

## Object–Goal 重绑定

在保持可执行初态的前提下改变对象与目标关系，逐项记录对象身份、目标位置、指令和成功判据。核查 SFT 初始化及 ESkill 训练的组合覆盖，只有两阶段均未训练的组合才称为未见组合。

同时观察当前对象绑定、预测终点与实际动作是否随任务关系改变。环境原生成功、物体到达目标区域、释放后稳定落位分别定义并报告；任务没有要求释放时，不额外改写其原生成功率。

## Motion Donor 干预

固定环境状态、当前观察、任务指令、采样噪声与原始 VLM K/V，仅替换 Motion 条件，比较原始与 donor 输出。候选 donor 应有明确运动／开合差异，并记录其来源、对应臂与适用性。

除动作张量差异外，检查短前缀执行后的 EEF 位移方向和大小、夹爪命令及实际开合变化，并保留接触与释放时机的回放。解码轨迹改变而实际行为几乎不变，不能记为 Motion 控制有效。

在同一 ESkill 结构和相同预算下比较有／无 Action-side Patch Masking；正常推理均使用完整视觉。辅助误差、attention 图或参数梯度存在，只说明局部机制工作，不能单独证明闭环收益。

## 模型预测与 Attention 面板

```bash
PYTHONPATH=src:. python scripts/prediction_viewer.py \
  --checkpoint /path/to/checkpoint.pt --episode libero_object_no_noops_s009_r002 \
  --data-index /path/to/libero40/index.json \
  --tokenizer-path /path/to/paligemma_tokenizer.model --vae-path /path/to/vae \
  --object-word milk --target-word basket --frames 40 --max-frames 3 \
  --check-action-parity --video /path/to/check/test/video/predictions.mp4 \
  --attention-video /path/to/check/test/video/action_attention.mp4
```

面板显示主相机预测／真值八点轨迹、六类路由概率、Action→Motion token 权重，以及 Current／Future Object 的真值画布、保存的读出头预测和真值 VAE 回环。只有 checkpoint 实际保存了视觉读出头，且本机 VAE 内容身份匹配时才解码；缺失时明确显示不可用。GT 在动作推理完成后单独读取，不参与策略输入；GT 缺失不隐藏预测点。完整轨迹、夹爪概率与有效掩码保存在同名 NPZ。

右下角为三张 **VLM 词 Query→主相机图像 Key** 注意力图：Target 对应指令中指定的目标词，Object 对应操作物体词，Instruction 对应整句指令。它们不是 Grounding token 或 Action Query 的注意力。词片段按实际 SentencePiece 输入对齐，排除 BOS、换行和 padding；Object 使用指定短语第一次出现，Target 使用最后一次。短语未指定、不在指令中或被截断时显示不可用，不从未来标签推断身份。Target 在这里表示语言中的目标（如 basket），与 Future Object 的 Event 终点画布含义不同。

注意力均由原始前向的 Q/K 和有效 Key 掩码重建 FP32 softmax；保留原 attention 运算与输出。词图平均指定 VLM 层、全部头及该短语 token；Action 图平均指定动作专家层、全部头和整个动作窗口，只保留最后一个 FM 步。默认层为深度的第三个四分位区间，18 层模型为 9–12（从 0 计）。`--layers` 同时指定两条分支的层。

热图在各自相机内归一化，颜色强度不能跨图解释为绝对权重。面板同时显示该相机占全部 Key 的权重；Action 的 image／text／motion／action 四组权重之和为 1。NPZ 保存原始词注意力概率、相机内热图、相机绝对权重及实际输入图像；JSON 保存词片段与 token 位置、层、精度、checkpoint 和采样设置。`--check-action-parity` 对首个选中观察进行同噪声原始推理对照，要求动作输出完全一致。

这是保存示范观察的离线回放，未执行预测动作；播放 FPS 与源帧时间分开记录。注意力与辅助读出不代表因果归因或闭环成功率。首轮真实 30k 核验入口见[预测面板检查](../data/libero/atomic-vla/check/test/model_predictions/README.md)。

## 记录与推进顺序

先完成保存帧／坐标／命令对齐及十条数据可视化核验，再做模型前向、梯度与服务技术检查，最后按另行确定的 recipe 执行训练和闭环测评。文档更新不自动启动正式实验。

新实验说明放 `experiments/<实验名>/`，权重、日志及大体积视频放数据盘；历史实验保留原位置。每次记录代码版本、配置、checkpoint、split、归一化、GPU／精度、seed、任务与初态、回合数、相机／状态、控制和采样参数。

结果保留逐回合动作块、实际执行动作、状态与视频，汇总成功数／总数、异常和未完成数量；LIBERO 按套件、Plus 按扰动类别汇总。技术闭环与模型效果分开，`technical` 成功率为 `null`。历史指标以原实验记录为准，不用 ESkill 设计改写旧结论。
