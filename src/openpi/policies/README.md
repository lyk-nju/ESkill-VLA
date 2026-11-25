# 数据与策略接口

数据读取与平台适配集中在 `policies/`，组 batch 在 `training/`，可学习网络在 `models/`。离线来源转换、实例绑定和原生区域生成位于 `scripts/preprocess/`；读取器不依赖仿真或视觉标注模型。

## Motion Planner 推理

对 `training_stage=motion_planner` 的 checkpoint，使用单主相机接口读取技能概率、Current 图和未来二维轨迹。以标准配方为例，单张图推理保留 batch 维度，Current 的 256 项按行排列为 16×16。输出坐标相对缩放并补边后的模型画布归一化。[推理实现](motion_planner.py) · [逐帧与干预诊断](../../../experiments/motion_planner/README.md)

```python
from openpi.policies.motion_planner import create_motion_planner_policy

planner = create_motion_planner_policy("/path/to/planner/checkpoint.pt", device="cuda")
prediction = planner.infer(main_rgb_uint8, task_instruction)
skill = prediction["skill_probabilities"][0]   # [6]
current = prediction["current_map"][0].reshape(16, 16)
trace = prediction["trace"][0]                 # [8, 2]
```

## 动作策略文件

| 文件 | 职责 |
| --- | --- |
| `eskill_dataset.py` | Episode 数组／RGB 读取、动作窗口、Event 与 ESkill 时间查询 |
| `libero_policy.py` | LIBERO 单臂状态、当前主相机投影与源动作适配 |
| `robotwin_policy.py` | RoboTwin 双臂输入与动作输出适配 |
| `policy.py` | 推理与输入输出变换；ESkill 返回当前全局技能 ID |
| `policy_config.py` | 按 checkpoint 配置与归一化构建模型，推理不加载 VAE |

## 联合动作模型的离线输入与目标

`EpisodeDataset(..., auxiliary=True)` 构造 ESkill 标签，`motion_stride=5` 默认查询未来 +5、10…40 个已保存主相机帧。无需精确时间戳；夹爪声明提供列、符号和执行顺序。`event_motion_targets` 同时供 Dataset、统计和标签核验使用，独立维护路径／命令有效性。来源检查见 [训练数据核验](../../../data/libero/atomic-vla/check/test/training/README.md)。

输入只包括当前图像、任务及固定平台臂存在性。ESkill 不读取本体状态、二维起点或相机标定；真实 Skill、Event、终点图像与未来状态只进入 targets／metadata。

目标为当前技能、Current／Future Object 两组原位画布、同 Event 内未来八时刻的二维 EEF 与开合命令，以及标准动作窗口。xy／grip 各有有效 mask，分别保留真实记录、Event 尾部、Episode 尾部与缺测。终点填充不进入主 Motion 损失，标签 mask 不用于屏蔽 Action 槽位。

Dataset 传递原始 bbox；`ModelTransforms` 保持 Current 范围，对 Future 在现有画布范围上每边额外外扩 `model.future_bbox_expand_px`（默认 8）个画布像素。终点 RGB 与区域监督权重一起扩大，裁到有效图像范围，源标签不改写。

Dataset 读已保存 EEF 源像素坐标，按主图方向与 resize／pad 生成归一化未来路径监督。训练采样不读取当前数值状态；ModelTransforms 也不生成状态张量或状态 tokens。动作继续使用训练集归一化统计；共享统计文件保留的 state 项不参与 ESkill 前向。

~~~python
from openpi.policies.eskill_dataset import load_episode

episode = load_episode("path/to/episode.json")
event = episode.event_at("single", frame=0)
goal = episode.goal(event["event_id"], "image")
~~~

纯 π0.5 通过 `auxiliary=False` 只读取当前观察和动作目标。

## 在线请求

LIBERO 联合动作策略使用主／腕 RGB 与任务指令。服务握手为 `requires_state=false`、`requires_camera_calibration=false`、`image_resize_on_server=true`；客户端不读取真实状态或投影，发送当前源分辨率 RGB，由服务端统一 resize／pad。固定单臂身份由适配器提供。ESkill 返回源动作与当前技能，轨迹／开合头只在需要诊断时读取。

RoboTwin ESkill 同样只输入 RGB＋任务，适配器提供固定双臂身份。纯 π0.5 若配置启用状态文本，仍保留原始关节状态适配；其动作输出保持原控制语义。

训练与保存恢复见 [training](../training/README.md)，在线客户端见 [LIBERO](../../../examples/libero/README.md)。
