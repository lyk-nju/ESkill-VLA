# 自研 RoboTwin 数据管线核对

2026-10-30。范围：当前采集／标注／转换代码，已有 50 个活动任务的 2,500 份事件 JSON，前序 10 条 HDF5 字段检查，以及 place_object_stand、lift_pot、handover_block 各 episode 0 的原视频／HDF5 对照。只读检查，未重新采集、渲染、补标或训练，也未修改 RoboTwin／XPolicyLab 代码。

当前代码版本：RoboTwin `b80a05e9a144464ec63b099af861bf3ae0ccacec`，XPolicyLab `7b0069a76ed51500eba5e85def30c4aeef5d1f89`。旧采集汇总记录完成于 2026-09-02；完成标记没有精确生成代码版本，不能仅凭当前 getter 推断历史文件的全部物理语义。当前文件指纹、全量事件统计及三条图像对照见 [证据](check/pipeline_review_evidence.json)。

## 已有链路与复用范围

```text
collect_event_dataset.py：按任务独占调度、重试和完成检查
  → collect_data.py：搜索成功 seed／规划轨迹，再重放采集
  → _base_task + EventRecorder：同采样时刻记录 RGB／状态／事件
  → 原视频 + events JSON + XPolicyLab HDF5 + 指令 + 关键帧可视化
  → transform_lerobot_v30_format.py：普通动作训练数据
  → π0.5 ALOHA 风格适配：关节 delta、夹爪绝对量
```

| 内容 | 当前已有 | 对本项目的适用边界 |
| --- | --- | --- |
| 示范采集 | 成功 seed、规划路径、任务独占调度、失败重试、完成标记 | 采集基础值得复用；当前批次为 clean50，非随机化 1,000 条设置 |
| 同步原始记录 | 主视角／双腕、双臂关节字段、末端位姿、15 Hz、相机参数、任务文本 | 需明确命令状态、实际状态、坐标和末端参考点 |
| 按臂事件来源 | subtasks 的 primitive_id、arm、起止帧、部分 actor，另有 phase 和交互事件 | 可以组织分臂事件；仍需共享技能映射、完整区间及 hold 关系 |
| 既有可视化 | 原视频、video_labeled、关键帧与 manifest | 适合检查稀疏边界；尚不能验收缺失的区域或完整语义 Event |
| 训练容器 | 共享 v3 转换器读取原 action，按当前环境配置使用 15 Hz | 只导出图像／关节状态动作／任务，未带上事件和三维监督 |

采集汇总中的 50 个任务完成属于文件与运行完整性记录，不等于已验收语义标签。

## 需要优先处理的内容

### 1. 关节“状态”当前读取驱动目标

当前 [robot.py](../../../RoboTwin/envs/robot/robot.py) 的 get_left/right_arm_jointState 使用 `joint.get_drive_target()`，夹爪值也是保存的命令；另有 get_left/right_arm_real_jointState 才读取实际 qpos。[采集 get_obs](../../../RoboTwin/envs/_base_task.py) 将前者保存到 joint_action，再由转换器写为 state／action。

因此当前字段不能直接称为实测关节。下一步要明确 π0.5 的状态契约：保留命令状态时保证训练／在线一致；需要真实本体反馈时另存实际 qpos 与夹爪观测。末端 XYZ 则来自机器人 link 的 global_pose，是另一种来源，不能将两者统称为同一种测量。

三条实读样本均满足 `action[:-1] == state[1:]`（双臂关节最大差为 0）。这证明了存储上的一步移位，不单独证明命令执行或接触成功。

### 2. 接触事件存在跨臂归属风险

[EventRecorder.observe](../../../RoboTwin/envs/utils/event_recorder.py) 按臂循环，却调用不接受 arm 的 get_gripper_actor_contact_position(actor_name)。该函数在 [_base_task.py](../../../RoboTwin/envs/_base_task.py) 中查询左右夹爪名称的合集。若两臂跟踪同一对象，另一臂的接触可能被本臂使用，再结合本臂“非张开”的命令状态产生 grasp。

这是当前代码可见的归属风险；尚未通过新物理试验测定旧数据中的错误比例。`source=physics` 仍是接触＋命令／阈值的启发式事件，不能等同于可靠的逐臂稳定抓持真值；hold 需要进一步核验。

### 3. 稀疏边界不能直接当完整 Event 流

[optimize_event_boundaries](../../../RoboTwin/envs/utils/event_recorder.py) 对全局事件时刻做默认 1 秒最小间隔筛选，没有分臂处理。2,500 份 JSON 中有 29,198 条 boundary_candidate，9,797 条被抑制；其中 920 条候选记录的保留对手仅来自另一臂。这些是记录数，不是独立错误事件数。

例如 handover_block/episode_0000000 的左臂 target_entry（帧 101）被右臂帧 115 的保留边界压制。原始候选、phase 和 subtasks 仍在，并未被删除。新标签应从这些原始记录组织，不能只用筛选后的 boundary／frame_labels。primitive 起止还对应脚本调用／同步等待，不自动等于每臂交互目的开始和结束。

### 4. 持物关系与区域尚未补齐

已有 16,055 个 primitive 区间；其中全部 3,804 个 lift 区间的 actor 为空，不能靠该字段直接找到被搬运对象。需结合前序抓持、释放／交接和画面核验持续关系；不能向前无条件填充，也不能把无动作区间自动标 idle。现有记录没有显式 hold 区间。

采集配置关闭 actor／mesh segmentation，所读 10 条 HDF5 均没有 bbox／mask。因此夹爪区域、当前对象区域尚缺。当前 [分割接口](../../../RoboTwin/envs/camera/camera.py) 返回 uint8／伪彩色；若采用原生分割补充，需保留原始 ID 与 actor／link 映射，并让 [HDF5 导出](../../../RoboTwin/envs/utils/pkl2hdf5.py) 实际保存所需标签，单开配置开关还不够。

### 5. 原始时间线比训练 HDF5 多一帧

原序列含 N+1 帧，训练 HDF5 保存前 N 帧 RGB／state 和后 N 帧 action；事件及原始主视频保持 N+1 帧。三条实读为：place_object_stand 135／136、lift_pot 113／114、handover_block 286／287。

按共享 decode_image_bit 解码 HDF5，原视频用 RGB 解码，对照首／中／最后公共帧，尺寸相同、平均像素绝对差约 1.25–1.63／255，与两种有损压缩的近似一致性相符；并非无损像素相等证明。原主视频可提供最后一帧，action 侧末端位姿可提供末端终点。旧双腕终点 RGB 未在所查 HDF5 中保留，不能从主视频恢复。后续应明确终点图像存储和区间闭开约定，不能将终点夹到上一帧。

### 6. 训练导出需要保留辅助监督的关联

当前 [共享 v3 转换器](../../../RoboTwin/XPolicyLab/scripts/transform_lerobot_v30_format.py) 正确读取 HDF5 原 action；当前 aloha_agilex.yml 配置 collect_freq=15。其 features／写帧部分只有图像、关节状态动作和任务，events、subtasks、EE 三维位置、相机参数及源 episode 映射没有进入输出。普通动作训练可使用这条基础路径，本项目需扩展字段或保留可追溯的旁路索引。

另一条 [Pi_05 旧 process_data.py](../../../RoboTwin/XPolicyLab/policy/Pi_05/openpi/scripts/process_data.py) 会从 state 重造 action、最后一步重复末 state，且写死 50 fps；对当前 `(N,1)` 夹爪字段又加 `[:,None]`，拼接维度也不兼容。这是旧入口的问题，不应套到当前共享转换器或既有所有训练资产上。

三维与投影适配还要注明：get_ee_pose 与 get_tcp_pose 使用不同参考点（当前实现相差沿工具轴的 0.12 偏置），HDF5 外参优先保存 cam2world_gl，不能默认当作 world-to-camera 的 CV 矩阵。

## 当前建议

先复用采集、原始证据和调度校验，明确命令／测量状态与逐臂交互语义，再补按臂技能区间、对象关系和区域，最后扩展训练关联。不要先把 2,500 条重新采一遍；现有原视频和候选事件仍有可利用的信息。是否需要重放／新采集，应在缺口核验和具体方案确定后决定。

原有 10 条抽样范围见 [check/README.md](check/README.md)，尚未完成本项目约定的 10 条原始核验视频与人工验收；已有历史 video_labeled 不自动视为本项目完成验收。
