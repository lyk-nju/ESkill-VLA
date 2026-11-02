# Data：外部数据需要提供什么

下表用于核对外部数据集已有的内容与缺失项。

正式 train／val 在**每个任务内部**按完整父轨迹划分 90%／10%，帧、左右臂、视角及派生记录跟随父轨迹；不按全体轨迹总体划分，也不按帧随机切分。SFT 与辅助训练共用固定清单，归一化统计只读取 train。LIBERO 当前为 1,518／175 条，清单与逐任务核对见 [task_split_v1](../data/libero/atomic-vla/check/test/task_split_v1/README.md)。

当前 Visual Grounding 只要求主相机中的各臂夹爪、操作物体及同一对象的 Event 终点区域；腕部相机暂不做区域标注。此范围不改变原始 RGB 的保留要求。

区域范围统一为 **忽略外部遮挡的完整屏内区域**（`full_in_frame`）：夹爪取指定手掌／手指几何，物体取所绑定实例／部件的完整几何，在同一相机中分别独立渲染并裁掉屏外部分，再由 mask 求 bbox。部分或完全被遮挡不使框缩小或失效；完全出画则没有屏内框，不等于没有操作对象。当前区域与 Event 终点区域使用同一约定。原 RGB 保留真实遮挡内容；历史可见区域标签保留原义，不能直接改名充当完整区域。

| 数据内容 | 对应模块 | 具体需要什么 |
| --- | --- | --- |
| RGB 图像 | VLM、Visual Grounding | 当前、历史及 Event 终点画面，保留帧号和视角对应 |
| 任务指令 | VLM | 完整 Episode 的自然语言任务描述 |
| 机器人状态 | Latent Motion、未来技能预测、Action Expert | 与图像同步的关节、夹爪及所需本体状态，明确左右臂、单位和字段含义 |
| 机器人动作与运动记录 | Latent Motion、Action Expert | 与图像同步的动作序列，能够读取或恢复各臂逐帧末端三维位置；明确动作含义、坐标系与单位 |
| 每臂 Event 边界 | 当前／未来技能、Visual Grounding、Latent Motion | 每只臂的子事件起始帧、结束帧；允许两臂事件并发，能关联动作窗口内各步 |
| 子事件描述与技能标签 | 当前／未来技能 | 描述该 Event 执行什么操作，并对应 Pick、Place、Hold 等技能类型 |
| 执行器视觉区域 | Visual Grounding：当前执行器 | 当前时刻本臂夹爪／手的完整屏内 bbox 或 mask，忽略外部遮挡 |
| 当前对象视觉区域 | Visual Grounding：当前对象 | 各帧中本臂被操作物体的完整屏内 bbox 或 mask，忽略外部遮挡，能够对应同一物体；不单列对象身份字段 |
| Event 终点对象区域（非必要独立字段） | Visual Grounding：事件目标 | 直接根据 Event 结束帧，读取上述对象视觉区域及对应 RGB 图像，无需额外标注 |
| 时间与同步信息 | 所有模块 | 图像、状态、动作、事件标注的时间戳／帧号映射与采样频率 |
| 坐标与相机信息 | Visual Grounding、Latent Motion | 三维位置的坐标系、单位、末端参考点；二维 Motion 投影需要主相机内外参及图像方向／缩放映射，或已核验的等价二维末端坐标 |

## 数据如何形成各模块监督

- **Event Organization**：每臂起始帧、结束帧 → 确定子事件范围，将技能描述、视觉区域与运动记录对应起来。
- **Atomic Skill**：子事件描述 → 共享技能类别 → 监督当前技能选择；模型按预测类别查找可训练技能向量。
- **未来技能序列**：动作窗口第 j 条记录 → 该动作时刻所属的每臂 Event → 技能标签，形成 `future_skill_labels[2,H]` 与有效性；无需额外原始标注。当前与未来技能共用词表与可训练技能表。
- **Visual Grounding**：RGB 图像 + 对应执行器／对象区域 → 保留原图布局的局部画布 → 冻结 FLUX VAE 编码得到目标 latent。当前帧构造当前执行器与当前对象目标；Event 结束帧构造同一对象的终点目标。模型预测的三角色视觉 Token 经共享 MLP 投影与空间拼接，直接以区域加权潜变量 MSE 监督；现有 RGB、区域和事件标注可复用，无需新增原始字段。
- **Latent Motion**：逐帧末端位置 + 相机／图像变换 → 主相机归一化二维坐标 → 按 `[4,8,12,16,20,24,28,32]` 个来源步取八个未来点 → 逐 token 监督；允许跨 Event，Episode 末尾采用 `hold_last`。
- **Action Expert**：当前／历史 RGB、任务指令、机器人状态及当前事件表征作为共享条件；模型预测的未来左右臂技能向量逐动作步注入 → 固定长度联合动作块，由同步的真实动作序列监督。

未来技能标签必须按动作记录及其对齐规则生成，不能直接按 RGB 行号切片。Episode 末尾 padding、缺失标签和不存在臂排除对应分类监督，不补成 idle。Stage 1 使用动作步位置编码；秒数与采样频率未知时仍保留源步索引。Goal 由当前 Event 终点构造；Motion 使用独立的固定八点未来时域，不按 Event 边界截断。

**二维 Motion 新设计**固定每臂 8 个 tokens、间隔 4 个来源步；LIBERO 对应 `[4,8,12,16,20,24,28,32]` 个来源帧，不再额外乘四。八点 Dataset、模型与损失已接入；全量 1,693 条／273,465 帧已保存主相机内外参、图像方向变换及八点标签，训练索引为 `/data1/yuankai/robot-research/data/atomicdata/libero40/motion_2d/index.json`，见 [全量相机核验](../data/libero/atomic-vla/check/test/calibration_full_v1/README.md)。按用户确认的 `hold_last` 规则，超出 Episode 末尾的目标取最后一帧末端位置，等价于终点后的坐标保持。只在末帧位置可靠且位于画面内时补齐，内部缺帧仍无效。

这些补齐点参与 Motion 监督，但保留 `future_observed`／`future_padded` 区分，以及请求未来帧 `future_frame_indices` 和实际取值帧 `future_source_frame_indices`；属于训练标签约定，不记为额外观察到的未来。当前／未来技能的有效性仍按真实 Event 计算。当前八点结果见 [八点／四步核验](../data/libero/atomic-vla/check/test/motion_2d_8x4_v1/README.md)。已有六点试验使用 `[8,16,20,28,36,40]`，保持原标签与视频；历史核验见 [六点四帧单位、终点保持版](../data/libero/atomic-vla/check/test/motion_2d_unit4_hold_v1/README.md)。

VAE 编码的 latent 是训练监督目标，不是在线预测的 Visual Grounding Token；二者通过潜变量投影头联系。推理时从当前／历史输入预测这些 Token，无需未来图像，也不运行 VAE 或潜变量投影头。目标画布、区域权重和角色有效性只进入监督侧。新回归设计的参数与实现状态见 [Target Grounding](target_grounding.md)。真实未来技能、边界和动作也只用于监督；训练与推理均以预测未来技能条件化动作。

这些内容的 Python 字段接口与逐项预处理清单见 [scripts/preprocess/fields.py](../scripts/preprocess/fields.py)。2026-11-01 的预保存草案采用 [JSON＋NPZ 及公共读写接口](../data/README.md)：Event 的 `[start_frame, stop_frame)` 与实际 `goal_frame` 分开；时间未知时保留源步索引，不伪造时间戳。AtomicVLA 的动作窗口与模型 batch 已由 [Dataset 与训练接口](../src/openpi/training/README.md) 适配，源格式不因此改写。数据来源与检查记录见 [data/](../data/README.md)。
