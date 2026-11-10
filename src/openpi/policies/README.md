# 数据与策略接口

数据读取与平台适配集中在 `policies/`，组 batch 在 `training/`，可学习网络在 `models/`。离线来源转换、实例绑定和原生区域生成位于 `scripts/preprocess/`；读取器不依赖仿真或视觉标注模型。

| 文件 | 职责 |
| --- | --- |
| `eskill_dataset.py` | Episode 数组／RGB 读取、动作窗口、Event 与 ESkill 时间查询 |
| `libero_policy.py` | LIBERO 单臂状态、当前主相机投影与源动作适配 |
| `robotwin_policy.py` | RoboTwin 双臂适配；ESkill 显式接收真实 EEF 状态 |
| `policy.py` | 推理与输入输出变换；ESkill 返回当前全局技能 ID |
| `policy_config.py` | 按 checkpoint 配置与归一化构建模型，推理不加载 VAE |

## ESkill 离线输入与目标

`EpisodeDataset(..., auxiliary=True)` 构造 ESkill 标签，`motion_offsets_s` 默认查询 +0.25s…+2s。Episode 的 `timestamps_s` 或 `timing_path` 清单建立物理时间，夹爪命令声明提供列、符号和执行时序；缺少依据时在读取 RGB 前报错。来源 schema 见 [训练数据核验](../../../data/libero/atomic-vla/check/test/training/README.md)。

输入包括当前图像、任务、臂存在性、原始八维 EEF／实际夹爪状态和主相机二维起点。真实 Skill、Event、终点图像与未来状态只进入 targets／metadata。

目标为当前技能、Current／Future Object 两组原位画布、同 Event 内未来八时刻的二维 EEF 与开合命令，以及标准动作窗口。xy／grip 各有有效 mask，分别保留真实记录、Event 尾部、Episode 尾部与缺测。终点填充不进入主 Motion 损失，标签 mask 不用于屏蔽 Action 槽位。

Dataset 读已保存 EEF 源像素坐标，ModelTransforms 按当前主图方向与 resize／pad 生成归一化起点／路径；本体状态保留物理单位。状态／动作 SFT 归一化继续使用训练集统计。

~~~python
from openpi.policies.eskill_dataset import load_episode

episode = load_episode("path/to/episode.json")
event = episode.event_at("single", frame=0)
goal = episode.goal(event["event_id"], "image")
~~~

纯 π0.5 通过 `auxiliary=False` 只读取当前观察和动作目标。

## 在线请求

LIBERO ESkill 使用当前源八维状态：EEF xyz＋axis-angle＋双指位置。服务握手包含 `requires_camera_calibration=true` 和 `image_resize_on_server=true`；客户端发送旋转后的源分辨率图像及该次真实 MuJoCo 主相机参数，包含相机扰动后的参数。

`observation/main_camera` 提供 `intrinsic`、`world_to_camera`、`image_transform`、`image_shape`。平台适配器使用与离线相同的投影函数，服务端再统一 resize／pad。不存在臂被屏蔽，出画起点有显式有效性。

直接 Python 推理可提供 `effector_start[2,2]` 与 `effector_start_valid[2]`（已经按模型主相机画布归一化），或提供上述 `main_camera` 由适配器投影；这两条路径都只使用当前信息。ESkill 返回源动作与当前技能，轨迹／开合头只在需要诊断时读取。

RoboTwin 的关节值不能充当 ESkill EEF 位姿。其 ESkill 适配要求另给真实 `arm_state[2,8]` 及相同几何定义的起点；原始关节／动作适配继续服务既有数据路线。

训练与保存恢复见 [training](../training/README.md)，在线客户端见 [LIBERO](../../../examples/libero/README.md)。
