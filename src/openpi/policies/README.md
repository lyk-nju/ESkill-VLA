# 数据与策略接口

整体组织以 AtomicVLA 为主要参照：数据读取与各平台输入输出适配集中在 `policies/`，训练组装放在 `training/`，PyTorch 模型统一放在 `models/`。完整目录规划见[项目说明](../../../README.md)。

| 文件（相对 `src/openpi/`） | 职责 | 当前进度 |
| --- | --- | --- |
| `policies/eskill_dataset.py` | `EpisodeReader` 与 `EpisodeDataset`：按配置视角读取图像、状态和动作；ESkill 另加每臂 Event／区域／Motion 监督 | π0.5 与 ESkill 共用；动作尾部按 `action_padding` |
| `policies/libero_policy.py` | LIBERO 样本到模型输入的字段映射，在线服务按 checkpoint `cameras` 取视角，以及推理动作输出适配 | 已接入单臂状态、主／腕相机和 7 维源动作 |
| `policies/robotwin_policy.py` | RoboTwin 样本到模型输入的字段映射，以及双臂推理动作输出适配 | 已有 14 维状态／动作适配；源数据接入待完成 |
| `policies/policy.py` | 封装模型推理与输入输出变换 | 已实现；无需未来标签／图像 |
| `policies/policy_config.py` | 根据训练配置、权重及归一化统计创建推理策略 | 已实现；推理不创建 VAE 和视觉训练监督模块 |
| `training/data_loader.py` | 创建 Dataset，组装采样、变换、shuffle 与 batch | 已实现父轨迹拆分及 batch |
| `training/config.py` | 数据来源、平台适配、模型与训练参数配置 | 已实现 JSON 配置和模型构建 |
| `training/checkpoints.py` | 模型、优化器及各 rank 随机数状态的保存恢复 | 支持完整权重和仅 LoRA 权重；LoRA 回读另加载原始基座 |
| `transforms.py` | 通用归一化、图像处理、文本及监督目标变换 | 已实现；训练／推理复用输入变换 |

Dataset 负责关联 Event、采样动作窗口和派生逐步技能标签；平台适配器负责图像槽位、状态与动作映射。可学习的技能、锚点和运动网络，以及冻结 VAE 的模型计算归 `models/`，训练循环归项目根目录的 `scripts/train.py`。

来源转换、实例绑定、技能／边界整理、补框、保存及核验在 `scripts/preprocess/` 完成。`fields.py` 描述预保存字段及对照 `docs/data.md` 的 11 项数据需求；`export.py` 负责保存与结构校验。训练侧读取导出的标签与来源证据，推理侧只使用在线可获得的输入。

`EpisodeReader` 按帧索引读取各视角 RGB、数组和区域 bbox（默认不读 mask）。`EpisodeDataset` 按配置视角读取所有相机；π0.5 只返回基础输入和动作，ESkill 通过 `event_index` 查技能、从 `effector_pixels[arm][camera]` 读源像素坐标并按 `future_targets` 组装 Motion、以 `role_sources`（RGB + bbox）交给 `ModelTransforms` 构造画布。未来数据仅进入 targets／metadata。

Motion 的几何确认和投影在 `EpisodeWriter.save` 时完成（`export.py` 中的 `effector_pixels`）：要求米制 world 末端位置、明确的参考点及静态 `world_to_camera_cv` 相机参数。移动相机不生成投影。Dataset 按 `self.camera`（第一个配置相机）读取 `effector_pixels[arm][camera]`，以配对观察为锚点，默认取 `+4…+32` 的八个来源步，允许跨 Event；只在真实 Episode 末帧有效且在画面内时保持终点。内部缺帧不插值，缺少可核验几何／标定时仅屏蔽 Motion。`trace[2,8,2]`、有效性及真实／补齐标识在 targets，请求／实际取值帧在 metadata，不进入在线输入。

```python
from openpi.policies.eskill_dataset import load_episode

episode = load_episode("path/to/episode.json")
event = episode.event_at("single", frame=0)
goal = episode.goal(event["event_id"], "image")
```

策略通过 `create_trained_policy(checkpoint)` 创建，`policy.infer({"images": [rgb, ...], "state": state15, "prompt": task})`（视角顺序与 checkpoint 的 `cameras` 相同） 返回源动作空间的 `[H,7]` 数组；ESkill 模型同时返回预测技能，原版 π0.5 只返回动作。状态排列与训练相同。`model.use_arm_state=false` 的 LIBERO 策略可直接接收源八维状态，不创建本体状态输入；此时还需 `discrete_state_input=false` 才完全关闭数值状态条件。LIBERO 的 `--runtime local` 服务根据 checkpoint 的状态开关和相机配置适配在线观测，技术核验已通过自动测试。

纯 π0.5 配置通常不开启本体状态，`LiberoInputs` 直接使用源 8 维状态。`scripts/serve_policy.py --runtime local` 把客户端的 `observation/<camera>` 按 checkpoint 的 `cameras` 排成图像列表，并加载 checkpoint 内统计；默认 `official` 路径保持原用途。

代码参考：AtomicVLA 的 [`atomic_dataset.py`](https://github.com/zhanglk9/AtomicVLA/blob/33d2f98574a52838b9db34b334bd5cc6cbe57f50/src/openpi/policies/atomic_dataset.py) 继承 `LeRobotDataset`，在 `__getitem__` 中关联事件标注；[`libero_policy.py`](https://github.com/zhanglk9/AtomicVLA/blob/33d2f98574a52838b9db34b334bd5cc6cbe57f50/src/openpi/policies/libero_policy.py) 提供 `LiberoInputs`／`LiberoOutputs`；[`training/data_loader.py`](https://github.com/zhanglk9/AtomicVLA/blob/33d2f98574a52838b9db34b334bd5cc6cbe57f50/src/openpi/training/data_loader.py) 创建 Dataset 并应用配置中的变换。此处采用其代码组织，预保存格式仍按本项目草案解释。
