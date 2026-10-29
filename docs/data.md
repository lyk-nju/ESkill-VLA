# Data：外部数据需要提供什么

下表用于核对外部数据集已有的内容与缺失项。

| 数据内容 | 对应模块 | 具体需要什么 |
| --- | --- | --- |
| RGB 图像 | VLM、Visual Grounding | 当前、历史及 Event 终点画面，保留帧号和视角对应 |
| 任务指令 | VLM | 完整 Episode 的自然语言任务描述 |
| 机器人状态 | Latent Motion、Action Expert | 与图像同步的关节、夹爪及所需本体状态，明确左右臂、单位和字段含义 |
| 机器人动作与运动记录 | Latent Motion、Action Expert | 与图像同步的动作序列，能够读取或恢复各臂逐帧末端三维位置；明确动作含义、坐标系与单位 |
| 每臂 Event 边界 | Atomic Skill、Visual Grounding、Latent Motion | 每只臂的子事件起始帧、结束帧；允许两臂事件并发 |
| 子事件描述与技能标签 | Atomic Skill | 描述该 Event 执行什么操作，并对应 Pick、Place、Hold 等技能类型 |
| 执行器视觉区域 | Visual Grounding：当前执行器 | 当前时刻本臂夹爪／手的位置与 bbox 或 mask |
| 当前对象视觉区域 | Visual Grounding：当前对象 | 各帧中本臂被操作物体的 bbox 或 mask，能够对应同一物体；不单列对象身份字段 |
| Event 终点对象区域（非必要独立字段） | Visual Grounding：事件目标 | 直接根据 Event 结束帧，读取上述对象视觉区域及对应 RGB 图像，无需额外标注 |
| 时间与同步信息 | 所有模块 | 图像、状态、动作、事件标注的时间戳／帧号映射与采样频率 |
| 坐标与相机信息 | Visual Grounding、Latent Motion | 三维位置的坐标系、单位、末端参考点；需要投影或跨视角对齐时，再提供相机内外参 |

## 数据如何形成各模块监督

- **Event Organization**：每臂起始帧、结束帧 → 确定子事件范围，将技能描述、视觉区域与运动记录对应起来。
- **Atomic Skill**：子事件描述 → 共享技能类别 → 监督当前技能选择；模型按预测类别查找可训练技能向量。
- **Visual Grounding**：RGB 图像 + 对应执行器／对象区域 → 保留原图布局的局部画布 → 冻结 VAE 编码得到视觉 latent。当前帧构造当前执行器与当前对象目标；Event 结束帧构造同一对象的终点目标。模型预测的三角色视觉 Token 条件化去噪器，通过这些 latent 的去噪监督学习。
- **Latent Motion**：运动记录 + 当前帧至 Event 结束帧的区间 → 提取真实末端三维路径 → 按弧长采样并减去当前末端位置 → 监督运动 Token 解码出的相对三维粗路径。
- **Action Expert**：当前／历史 RGB、任务指令、机器人状态 + 模型预测的技能向量、视觉 Token 与运动 Token → 生成动作块，由同步的真实动作序列监督。

VAE 编码的 latent 是训练监督目标，不是在线预测的 Visual Grounding Token；推理时从当前／历史输入预测这些 Token，无需未来图像或运行 VAE 目标编码器、去噪器。

数据来源与检查记录见 [data/](../data/README.md)。
