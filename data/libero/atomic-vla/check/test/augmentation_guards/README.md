# 反事实启动检查与附属 Region 保护

2026-11-18，基于 `8b519ea` 的工作区修复；技术核验，无模型训练。

- 当训练启用 `target_swap_prob>0`，Dataset 初始化检查所有主相机物体的 `interaction_locked` 数组引用、布尔类型及逐帧形状。缺失或无效直接报错，包含 Episode、实体和 `interaction_locks` 补齐命令。验证或关闭反事实时不要求此标记。
- 当前监督目标若为 `region`，Copy 候选中排除其 `instance` 父物体，避免移动物体后留下旧位置的 Region 标签。其他物体仍可编辑；当前没有监督到的附属 Region 不阻止父物体移动。不实现额外的父子区域联动变换。

全量检查 1,693 条轨迹、8,615 个物体实例记录，`interaction_locked` 引用及数组全部通过。77 个作为 Event 目标的 Region 中，36 个属于背景表面、41 个属于固定装置，没有附属于可移动物体的目标 Region，因此本机当前数据没有发现这条 Copy 路径的受影响样本；后续数据由新增规则覆盖。

实际 `create_dataset` 已通过：train 1,518 条／245,553 帧（增强开启），val 175 条／27,912 帧（增强关闭）。相关回归 111 项通过、1 项因沙箱 IPC 限制跳过，覆盖缺失／坏引用／错形状／错类型的启动报错、关闭增强时可读取、Region 父物体排除与未使用 Region 不阻止 Copy。Ruff 与 diff 检查通过。

[机器可读检查结果](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_guards_20261118_001/checks.json)。本轮仅修改读取与在线候选规则，无需重新渲染或改写已保存数据。
