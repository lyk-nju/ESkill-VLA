# GuidedVLA 的 RoboTwin 数据管线

核对日期：2026-10-30。范围：论文、主仓库和其固定 RoboTwin 子模块的基础采集／转换代码。作者训练集尚未发布，不能用本地 clean50 代替其数据做字段实读。这里只读代码与现有证据。

主仓库版本：`04be059e0d6bd448be5cb45fdbafc775f7eb5e38`；固定 RoboTwin 子模块版本：`958a6d2910a0262f5531fcdeb7fffae4184bb586`。以下将作者论文设置、上游参考入口及已实现加载器分开记录。

## 逻辑链路与公开程度

```text
随机化场景的脚本专家示范
  → 原始图像／双臂关节与夹爪记录／任务指令
  → 基础轨迹转换 + 阶段技能／对象区域标注
  → LeRobot v3
  → RoboTwin 三视角／14 维状态动作适配
  → 关节相对量变换、归一化、动作窗口技能分布
```

上图不表示已有公开的一键程序；完整标签构建与 v2→v3 的具体执行入口未在主仓库找到。

| 环节 | 已核对内容 | 公开边界 |
| --- | --- | --- |
| 采集设置 | 论文 8 个任务，每任务 1,000 条随机化示范 | 具体训练轨迹与场景 seed 清单未发布 |
| 上游采集器 | 先搜索规划和任务成功的 seed／关节路径，再用保存的 seed／路径重放采集图像和状态，并生成指令 | 固定子模块的标准采集逻辑；不能当作作者另行公开的完整采集配置 |
| 上游基础处理 | process_data.py 拼接 `[L6,Lgrip,R6,Rgrip]`，qpos 取原序列 `[:-1]`、action 取 `[1:]`，图像保留前 T 帧；读取 seen 指令 | 动作是下一采样关节记录，不是末端 delta；处理器不导出 Event、mask 或三维路径 |
| 上游 LeRobot 转换 | 读取 `/observations/qpos`、`/action` 和三视角，逐帧写入；每条轨迹随机取一条任务描述 | 旧转换器写死 fps=50、图像 480×640；不能认定这是作者发布数据的真实采样频率或本地数据频率 |
| 辅助标注 | 论文给出共用的阶段对象／技能半自动标注方法 | 主仓库未提供完整阶段切分、Qwen3-VL 提示、SAM2 批处理及人工修订工具 |
| 容器 | 论文称自采 RoboTwin 数据也从 LeRobot 2.0 转为 3.0 | 主仓库 README 假定已有 LeRobot 数据；RoboTwin 训练集仍未发布 |
| 模型加载 | 已有 LeRobotRobotwinDataConfig、三视角区域映射、技能窗口和 14 维动作适配 | 可直接学习这些接口，但不是完整数据引擎 |

## 加载器实际要求的字段

| 字段 | 含义与用途 |
| --- | --- |
| observation.images.cam_high | 当前主视角 RGB → base_0 |
| observation.images.cam_left_wrist／cam_right_wrist | 当前双腕 RGB → left_wrist_0／right_wrist_0 |
| observation.state[14] | `[左臂6关节,左夹爪,右臂6关节,右夹爪]` |
| action[14] | 原容器语义为绝对关节目标；训练对 12 个关节减当前状态，两个夹爪维度保留绝对值；推理再回转 |
| observation.skill_id | 按动作窗口读取并生成 skill_soft；接口没有左右臂轴，不能直接表示两臂不同技能 |
| cam_high_attention_object | 主视角 256 个 patch 的区域监督 |
| cam_left_wrist_attention_object／cam_right_wrist_attention_object | 双腕各自的区域监督；视角不同不等于按臂事件标签不同 |
| task 元数据 | 生成 prompt；具体指令来源随数据构建保存 |

模型适配使用 `adapt_to_pi=False`，不套用真实 ALOHA 的符号／夹爪映射；关节 delta 与绝对夹爪分开处理。完整配置默认 `skill_num_classes=8` 只是可修改参数，不能据此宣布作者公开了八类技能的名称和映射。

## 对我们的构建直接有用的结论

- 原生采集和模型输入适配可以复用其思路；逐臂语义 Event、并发、hold、夹爪区域、同一对象终态与路径监督需要我们补建。不能把三视角 object_maps 当作双臂三角色标签。
- 目前本地文件是已修改的 `state/action/vision/events` HDF5，旧上游 process_data.py 读取的是 `joint_action/observation`。应单独适配，避免重复执行 `[:-1]/[1:]` 而再次丢帧或错移动作。
- 本地读取来源为 15 Hz，转换时使用经核验的源时间和帧映射；旧脚本的 50 fps 常数不适用。三维末端、相机和可重放证据应保留在原始记录，不能因模型只读关节而丢弃。
- 对象标签若改用仿真原始实例 ID，这是我们的待验证补充方案，不是 GuidedVLA 已公开完成的原生分割管线。具体实施与人工验收接 [construction_plan.md](construction_plan.md)。

## 代码证据

- [主仓库 RoboTwin 说明](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/examples/robotwin/README.md)、[数据适配](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/config.py#L316)、[对象字段映射](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/shared/attention_map.py#L12)。
- 固定子模块：[标准采集](https://github.com/RoboTwin-Platform/RoboTwin/blob/958a6d2910a0262f5531fcdeb7fffae4184bb586/script/collect_data.py)、[基础轨迹处理](https://github.com/RoboTwin-Platform/RoboTwin/blob/958a6d2910a0262f5531fcdeb7fffae4184bb586/policy/pi05/scripts/process_data.py)、[基础 LeRobot 转换](https://github.com/RoboTwin-Platform/RoboTwin/blob/958a6d2910a0262f5531fcdeb7fffae4184bb586/policy/pi05/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py)。
- [论文 III-D、附录 H／I／J](https://arxiv.org/html/2605.12369v2)。RoboTwin 实际标签覆盖、质量、技能类别及采样频率仍不能通过未发布数据验证。
