# RoboTwin 原生采集字段自查

2026-10-30，对照 [docs/data.md](../../../docs/data.md) 及 [data/data.py](../../data.py)。范围：新原生区域试采、六任务候选 Event 及当前代码；不代表历史 2,500 条数据已补齐。用户已选择 RoboTwin 原生区域路线，不接外部模型做 bbox 提取；模型内部的 Target Grounding 监督仍保留。**数据来源／可生成能力基本满足，当前预览文件尚未完整输出 EpisodeData，语义标签仍待人工验收。**

2026-10-31 更新：用户授权 clean 50×50；正式采集入口已接入主相机原生区域／完整夹爪框、任务指令及语言变体、真实时间、逐物理步动作／状态与条件图像行关联，并补原语实例身份。八任务验证通过，全量采集中，见 [采集协议和运行入口](../collection_clean/README.md)。下表保留 10-30 预览检查的范围，已完成的采集端收尾以本段及采集记录为准；统一 EpisodeData 导出仍未完成。

| 我们需要什么 | 有没有 | 备注 |
| --- | --- | --- |
| RGB 图像 | 有 | 主相机、双腕、完整保存的源帧；当前／历史／终点按索引读取。 |
| 任务指令 | 部分 | 原生模板与 runner 生成功能已有；本轮 native 试采文件没有 instructions，需接入落盘。 |
| 机器人状态 | 有 | 双臂实测关节 qpos、夹爪命令分开记录；夹爪命令不能冒充实测开度。 |
| 机器人动作与运动记录 | 部分 | 有 14 维关节／夹爪驱动目标及双臂末端位姿；需落实 π0.5 的动作定义、频率和观测→动作对应。 |
| 每臂 Event 边界 | 部分 | 六任务已生成 25 个分臂候选 Event，保留并发与共同终点；需人工核验并转换为正式 end_frame。 |
| 子事件描述与技能标签 | 部分 | 已有原语映射、合并、Hold／Idle 及对象标签；逐事件描述、最终词表和 skill_valid／evidence 仍待整理。 |
| 执行器视觉区域 | 有 | 原生可见区域及忽略外部遮挡的屏内完整夹爪框已试采验证；全量流程尚未统一启用。 |
| 当前对象视觉区域 | 部分 | 六任务已有关联对象的候选 Event 与原生区域；需投影到 object_regions／object_present，区分无对象与未知，并补逐项有效性。 |
| Event 终点对象区域 | 部分 | 不需独立补标；Event 与对象关联确定后，读取同一对象的结束帧区域和 RGB，无效区域单独屏蔽。 |
| 时间与同步信息 | 部分 | 新 native 保存真实物理步与时间；需建严格递增的公共时间轴、保留同刻源行映射并明确动作条件图像。不能把标称 15 fps 或 frame/15 当真实时刻。 |
| 坐标与相机信息 | 部分 | 有内外参；源码确认末端 XYZ 为世界坐标，当前参考点沿工具轴比夹爪中心后退 0.12 m，需统一定义并写入元数据。 |

**语义进展**：每臂 Event／技能与操作对象绑定、Hold／Idle／Return 已有六任务候选规则和视频；共享锅的共同 goal 已对齐。仍需人工核验，不将技术通过当作 accepted；未知与无对象分开。当前 data.py 的 goal 可用性依据边界、终点 RGB 和同一对象区域表示的可用性派生，不新增独立事件成功标签要求。对应 [接口 EventData](../../data.py) 与 [原语预览](../test/events/README.md)。

**接口收尾**：合并区域试采与正式原始采集入口，保存完整任务指令；统一真实公共时间轴、动作表示／通道单位／时距与 conditioning_image_indices、状态及末端参考点；将来源数组／侧车投影为 DataRef、逐项 valid、AnnotationEvidence。visible 与 full_in_frame 分别生成 RegionSequence，bbox／mask 的有效性独立记录。当前 source.py 仍只读取历史格式／raw v2，新 native 数据和候选事件尚未统一输出 EpisodeData。

候选预览使用 `[start_frame,stop_frame)` 且 goal_frame 为最后源状态，data.py 则要求 `[start_frame,end_frame)` 且真实终点为 end_frame；必须结合真实时间与源行映射转换，不机械重命名或丢弃动作／终点观察。pot 的协作组也需转换为共同公共终点；这项映射尚未实现。

**自动派生监督**：标注就绪后，构造保留原图布局的 ee／obj／goal 画布及区域有效性、剩余末端弧长路径、动作块／有效性，再按完整轨迹拆分并仅用训练集计算统计。现有剩余路径辅助函数已有；当前视频居中放大的 crop 是核验展示，不是 Design 要求的训练画布。这里不增加新的人工标注类别。

历史字段表见 [field_check.md](field_check.md)；原生证据见 [区域试点](../test/README.md)、[随机化试采](../test/randomized/README.md)、[完整夹爪框](../test/occlusion/README.md)、[六任务 Event 视频](../test/events/README.md)。六任务共 909 帧、25 个候选 Event，技术核对通过。adjust_bottle 缺成功 random 试采，dump_bin_bigbin 的 Pour 源阶段／等待采样仍缺，不能据六例外推所有任务。

本轮为接口核对及文档更新，未修改 data.py、转换代码或启动新采集／训练。
