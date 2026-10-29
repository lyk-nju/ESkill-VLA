# AgiBot World 2026 数据来源调研

核实日期：2026-10-29。**状态：初步调研，未确定采用、训练配比或转换接口。** 本文区分官方声明、发布文件检查与研究推断。有限样本按可访问性及文件大小选择，不是随机抽样，统计不能外推整个数据集。

当前判断：**有理由把它纳入重点候选。** 已发布数据同时包含真实双臂动作、末端位姿、分层操作描述，以及部分视觉和执行反馈标注，与 ESkill-VLA 的研究问题有较强关联。但不同主题、批次的监督差别很大；“作为较大规模动作来源”和“提供完整的分臂 K/G/Z 监督”仍是两个需要分别验证的问题。

**续查入口：§9。** 在首轮检查基础上，已扩展至第四期两个批次、18 个任务的 29 个分片，覆盖 2,864 个 episode 条目的元数据；另核对 Dense Prompt 合并记录。逐分片数字、来源哈希及统计口径见 [机器可读清单](metadata_inventory.json)。下文 §1–8 保留首轮检查范围，避免混淆样本口径。

## 1. 来源与首轮检查范围

- [官方 HF 仓库](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)、[项目页](https://agibot-world.com/)。这是 2026 版；不能沿用旧版 AgiBot World Alpha/Beta 的轨迹数、任务数、本体和字段结论。
- 固定 revision：`95b159a10305a0385f369aba5d4b16242526d17b`，API 返回最后更新于 `2026-10-24T13:41:37Z`。
- API 的 `gated=false`，本次直接读取公开文件成功；数据卡许可为 **CC BY-NC-SA 4.0**。后续使用与衍生数据发布需要保留署名、非商业和相同方式共享条件，模型权重等具体产物的条款适用范围需另行确认。
- API `usedStorage=17,887,306,165,670` 字节，约 17.9 TB。这是仓库存储口径，不代表去重后的训练数据大小。HF Viewer 显示的约 73.7 万行是图像条目，不能当成独立机器人轨迹数。

原始检查材料存于：

```text
/data1/yuankai/robot-research/data/external/agibot_world_2026/raw/95b159a10305/
├── repository.json                  # API 快照；siblings 在 100,000 项截断
├── root_tree.json
├── tree_*.json                      # 分主题目录清单，含请求 URL
├── README.upstream.md
├── files/                           # 保留上游相对路径的完整小文件
│   ├── PhysicalReasoning/...
│   ├── ReinforcementLearning/README_THEME_3.md
│   └── split_episodes_tool.zip
└── archive_prefixes/                # 4 个归档各前 16 MiB，以及完整读出的成员
```

完整下载的小文件旁有 `.source.json`，记录固定版本 URL、字节数与 SHA256；归档前缀另记录成员名和成员哈希。前缀不是完整压缩包，读到截断位置是预期行为，不表示上游文件损坏。首轮材料约 126 MB，没有全量下载、启动训练或生成新的实验数据；续查材料沿用同一固定版本目录。

| 检查对象 | 已读取元数据范围 | 已读取时序数据 |
| --- | --- | --- |
| PhysicalReasoning / `opensourcev4-20261001/lerobot/task_100347/02` | 170 条 episode、75,828 帧的元数据 | episode 0，391 行；另读取头部视频，解码查看 3 帧 |
| 同批次 `task_87262/05` | 1 条、1,069 帧 | episode 0，1,069 行 |
| 同批次 `task_104212/02` | 1 条、1,733 帧 | episode 0，1,733 行 |
| ImitationLearning / `CommercialSpaces/task_4542/553049_553079.tar.gz` | 4 条、8,073 帧 | episode 0，1,858 行 |
| RichInteraction / `CommercialSpaces/task_4439/458713_458713.tar.gz` | 1 条、2,089 帧 | episode 0，2,089 行 |
| ReinforcementLearning / `Home/task_12192/HG-DAgger/20857245_20857259.tar.gz` | 主题说明；本次前缀未读到 info.json | episode 0，2,313 行 |
| ReinforcementLearning / `Home/task_12228/Rollout/20838754_20838881.tar.gz` | 主题说明；本次前缀未读到 info.json | episode 0，1,295 行 |

合计检查了覆盖 **177 条 episode 的 5 份 info.json**、**7 条 episode 的 Parquet（10,748 行）**。元数据覆盖数不等于看过视频或审计过动作的轨迹数。此外读取了 task_100347/02 的 ShareGPT 框、点、轨迹文件、发布清单及官方切分工具源码。

## 2. 四个主题应分别看待

固定版本的根目录有下列四个真实数据主题，另有 `simulation/`。模拟数据未在本次检查。

| 主题 | 官方定位 | 文件清单与本次认识 |
| --- | --- | --- |
| ImitationLearning | 专家遥操作、长任务及分层标注 | 完整目录清单：146 个 tar.gz、13 个 task 目录，约 4.77 TB；样本的任务层、步骤层与空的分臂占位同时存在 |
| RichInteraction | 探索性物理交互，保留抓取失误、掉落、碰撞等过程 | 89 个 tar.gz、10 个 task 目录，约 2.51 TB；抽查的一条是整段探索指令，不能直接视为细粒度技能示范 |
| ReinforcementLearning | 专家、策略 rollout 与人工接管纠正 | 99 个 tar.gz 和主题说明、14 个 task 目录，约 4.20 TB；实际 HG-DAgger 样本含非平凡接管及终止奖励 |
| PhysicalReasoning | 同目标、不同状态／约束下的决策与执行 | 2026-10-22 发布；有 `opensourcev4-20261001` 和 `opensourcev4-20261003` 两批，含独立 LeRobot 文件和 ShareGPT 标注；本次实际样本只来自前一批 |

前三行数量是对分页目录清单的计算，**task 目录数、压缩包数均不是独立语义任务或轨迹数**。第四期发布清单列出 98 个 product、8,146 个 completedEpisodes、126 个 preservedEpisodes，但这些是构建／标注发布口径，尚未逐项对齐最终 Parquet，不据此给出全库去重轨迹总数。全库 `siblings` 截断于第一批 ShareGPT 路径，不能由其中可见的 77 份 info.json 或 7,470 个 Parquet 推断第四期总量。

官方对 Theme 3 报告了 11,430 条轨迹、14 个任务，其中 1,024 条成功策略 rollout、1,369 条失败 rollout；另有任务区间、干扰、错误和接管标注。这些总量来自发布方，本次没有全量复算。[Theme 3 发布说明](https://www.agibot.com/article/231/detail/95.html)

Theme 4 的条件变化和任务顺序分支与技能组合研究相关；官方还描述了高层问答和细化操作描述。当前检查已确认框／点／轨迹的 ShareGPT 文件，**尚未完成高层决策问答、Dense Prompt 与全部动作轨迹之间的回链检查**。[Theme 4 发布说明](https://agibot.com/article/231/detail/129.html)

Theme 1 的层次设计见[初次发布说明](https://www.agibot.com/article/231/detail/54.html)；Theme 2 的探索性采集见[官方介绍](https://www.agibot.com/article/231/detail/72.html)。这些定位不能替代逐批次字段与质量检查。

## 3. 对 Event 与 K 的支持：有区间，不等于已有完整分臂事件

PhysicalReasoning 三份元数据共有 **956 个 instruction segment**，全部 `track="default"`。样本中可见 Push、Pick、Clip、Place、Rotate 五类技能。文本经常明确左右臂，部分 Rotate 指的是腰部或底盘，不能全部映射到某只手臂的技能。

例如 `task_100347/02` 的 episode 0：

| 起止帧（保留原值） | 原始技能 | 操作含义 |
| --- | --- | --- |
| 4–112 | Push | 右臂推开挡住钥匙的纸巾盒 |
| 112–233 | Pick | 右臂取钥匙 |
| 233–308 | Clip | 左臂夹住包 |
| 308–351 | Place | 右臂将钥匙放入包中 |
| 351–384 | Place | 左臂放下包 |

这提供了研究“同目标下前置技能与对象状态”的具体材料。但右臂放钥匙时左臂继续持包的区间，没有单独的 hold 标签；不能把没有被描述的一只臂一律标成 idle。文字中的臂身份可以作为转换线索，仍需结合动作与图像检查并发事件、持物和恢复行为。

同三份元数据的 affordance segment 中，`active_arm` 全为 `{"arm":"unknown","source":"disabled"}`。其内部部分 `target_point`、`trace_2d_gt` 又有 left/right；这说明臂信息分散在多个层次，不能只读取一个字段后判断完整性。

ImitationLearning 的 4 条元数据共有 81 条 segment：23 条 Sub-task、50 条 default、左右臂各 4 条。技能字段 71 条为 Other、10 条为空。episode 0 的左右臂轨道覆盖整段，但 instruction/skill 为空。因此**出现 Left arm / Right arm 轨道本身不能证明具备可用的分臂技能监督**。步骤文本仍可能有价值，需比较重标、语言映射或仅使用部分监督的成本。

RichInteraction 抽查的 2,089 帧只有一条整体探索指令，技能为 Other、key_frame 为空。这是一个样本的情况，不据此排除整个主题。

## 4. 对 G 的支持：有空间标签，但角色与时刻不等同于当前设想

PhysicalReasoning 的 `affordance_annotations` 实际包含：

- `bbox_xyxy_norm`：归一化对象／交互参照物框。
- `point_xy_norm`：接触或交互点。
- `trace_2d_gt`：带 waypoint 帧索引的二维夹爪路径；样本记载来源为 hand camera extrinsic、投影参考为 gripper center，使用 `gripper_offset_m=0.15`。
- 相机、图像尺寸、原始技能、原子动作类型、生成模型与审核记录。

三份元数据均仅在 `observation.images.top_head` 上提供这套标注：

| 元数据子集 | 全部帧数 | 含 affordance 记录的不同帧 | bbox found=true | point found=true |
| --- | ---: | ---: | ---: | ---: |
| task_100347/02 | 75,828 | 5,145 | 3,385 | 3,384 |
| task_87262/05 | 1,069 | 78 | 62 | 45 |
| task_104212/02 | 1,733 | 87 | 66 | 55 |
| 合计 | 78,630 | 5,310（约 6.75%） | 3,513 | 3,484 |

`found=true` 只表示记录存在，不表示人工确认准确。5,310 条记录的 `audit.status` 均为 pending；3,950 条注明 `qwen3.7-plus`，其余生成来源字段缺失。发布清单另有 completedEpisodes，不能用这个总数字证明每个空间标签已经审核，也不能只凭 pending 断言所有标签都从未人工修改。

与当前 G 设想的差距：

1. **单个 target 的角色会变化。** Pick 的框可能是被抓对象；Place 的框可能是垃圾桶或容器。不能把所有 bbox 统一当成被操作对象区域。
2. **交互点不等于事件终点目标区域。** ShareGPT 的 point 提问明确要求接触／交互点；不是最终对象与参照物的并集框。
3. **二维夹爪轨迹不等于执行器区域，也不等于三维 Z。** 轨迹字段名里的 gt 不替代投影、参考点与时间对齐验证。
4. **标注是稀疏的。** 示例轨迹通常按 30 帧步长取点，再包含段末点，也有补充标注帧；不能把缺帧默认当成有效标签。
5. 本次样本未发现完整的逐帧执行器 mask、对象 mask、终点对象／参照物角色链。是否补框、跟踪、投影或采用部分监督，仍是研究选择。

同一 task_100347/02 的 ShareGPT 版包含 bbox 3,384 条、point 3,383 条、trace 2,895 条；它们是这 170 条轨迹的派生问答，不是新增轨迹。其坐标范围是 **0–1000**，info.json 中则是 **0–1**。问答中的 `episode_uid` 仅含 task 与局部 episode 编号，跨批次／分片连接时需要额外保留来源路径。

## 5. 对 Z 与动作的支持：基础字段较好，物理语义仍需核对

实测状态维度在三个主题样本中分别为 **193、169、216**；七份 Parquet 的 action 均为 **44 维**。因此即便都叫 LeRobot v2.1，也不能共用写死的 state 切片。

PhysicalReasoning 的 216 维状态中：

| 字段 | 维数／索引 | 本次认识 |
| --- | --- | --- |
| state/end/arm_position | 6，58–63 | 两组逐帧三维位置，具备构造双臂路径的基础 |
| state/end/arm_orientation | 8，50–57 | 两组四元数长度；所读三条的范数约为 1，但分量顺序仍待确认 |
| state/joint/position | 14，65–78 | 双臂关节状态 |
| state/left_effector/position、right_effector/position | 各 1，0／1 | 夹爪相关标量，不能误当成三维末端位置 |
| state/robot/position、orientation | 3 + 4 | 底盘相关状态，需确定参考系 |
| extrinsic_end_T_* | 每相机旋转矩阵与平移 | 六路相机的逐帧外参线索，变换方向与共同参考系尚未验证 |

RichInteraction 样本中的 arm_position 位于 **34–39**，不是 58–63。各文件应从 `field_descriptions.indices` 解释字段。

44 维 action 含左右夹爪 2、末端位置 6、末端姿态 8、关节 14、头部 3、腰部 5、底盘速度 6。同一个动作向量同时记录关节和末端表示，**不是 44 个独立控制自由度**。它也不能不加解释地截断／补齐成现有策略动作接口。选关节还是末端、是否保留腰部／底盘，取决于任务和训练目标；样本确实有腰部、底盘转动的步骤。

本次三条 PhysicalReasoning 时序的 state/action 均为有限值，末端位置有非零变化。仍未确认：左右臂排列的权威定义、位置单位和轴方向、共同参考系、腕部／TCP／夹爪中心的对应关系、动作是命令还是后续目标状态、时间偏移、跨主题字段一致性。特别是移动头部和底盘的数据，不能直接跨帧比较像素位置，或把局部坐标中的差值当作固定世界系位移。

视频方面，样本有头部 RGB、左右腕部、三个鱼眼及 head_depth 共七路；头部 RGB 为 640×400、30 fps。读取的头部视频为 AV1，391 帧与 Parquet 行数一致，已查看推盒、取钥匙、放入包附近的三帧。当前环境 OpenCV 解码失败，PyAV/libdav1d 成功；这属于读取依赖问题，不能据此判断视频损坏。`head_depth` 的 shape 虽写为三通道，但续查确认其 video_info 声明 PNG、gray16be、is_depth_map=true；因此不能据 shape 认作 RGB 深度可视化。尚未实际解码深度文件，也未确认尺度与米制转换。

## 6. 时间、成功和接管语义不能混用

- `task_104212/02` 的 episode 0 有 1,733 行，最大 frame_index=1,732，最后一段 end_frame_index=1,733。这支持右开区间的可能解释，但不能将端点直接当作可读取的图像索引；不同标注层的约定还需核查。
- 三条 PhysicalReasoning Parquet 的 succeed、done、intervened 全为 false，reward、action_prob 全为 0；info.json 却有 Success Frame。**不能仅凭 succeed=false 把这些轨迹认作任务失败**，也不能用列名推断它提供了有效奖励。
- 同样，PhysicalReasoning 的 take_over 存在很长区间，但 Parquet intervened 全 false。遥操作区间与失败后的人工纠正不是同一语义。
- 实际 HG-DAgger 样本的 2,313 帧中，intervened=true 为 1,057 帧；succeed 全 true，reward 只有末行 1，done 只有末行 true。这确认该主题至少有可用的非平凡执行反馈字段。
- Rollout 样本的 1,295 帧中 succeed/reward/done/truncated 均为 false 或 0，不能假设每条文件最后都有 done=true。该样本未读到 info.json，结果与终止语义尚不能交叉确认。

这些差别影响失败段、恢复段、人工动作和策略动作的使用方式。若后续采用行为克隆，不宜把整个主题内所有动作都默认为同质量专家标签。

## 7. 数据链路方面已经明确的注意点

这里是依据文件提出的接口讨论，不是已批准的实现方案。

1. **保留原生分层标注再考虑训练格式。** 官方 split_episodes_tool.zip 的 main.py 按 Task Frame 切分，随后删除 key_frame、instruction_segments、take_over、h5_path、camera_parameters。对研究 Event/G/Z 有用的信息会丢失；本次仅阅读源码，没有执行。样本里还出现 Sub-task frame 命名，工具的匹配条件不能直接外推适用于全部批次。
2. **源身份要包含批次和分片。** 可以讨论使用 revision／theme／batch／task／shard／local episode 组合，再保留原始 episode ID。发布 manifest 含部分 localEpisode 到原始 episode_id 的映射，可进一步验证。多相机、ShareGPT 派生记录应跟随原轨迹分组，避免划分泄漏。
3. **标注层保留原义。** task、步骤、按臂事件、错误、接管、成功分别保存来源和有效性；Other、空值和未标注不能强行变成已知技能或 idle。
4. **数值字段按文件 schema 读取。** 状态维度已经发现变化；单位、坐标、参考点、动作时序在核实前保留未知状态。
5. **视频与稀疏空间标签分开索引。** 留存原始帧号、视角、图像尺寸和坐标归一化方式；补标结果与原始标注区分。
6. **下载可以按问题选择子集。** 第四期独立文件便于先看元数据、少量 Parquet 与需要的 RGB；前三期的大 tar.gz 更不便随机读取。本次前缀读取能取得部分元数据，不保证适用于全部归档。

## 8. 它可能承担什么角色，还缺什么证据

| 可能用途 | 当前支持证据 | 尚不能确定之处 |
| --- | --- | --- |
| 真实双臂动作及三维运动来源 | 已读到实际动作、关节、双臂末端和视频 | 坐标与动作语义、跨本体迁移收益、有效规模 |
| 条件变化下的技能组合研究 | 第四期条件化任务；实际取钥匙样本有前置推盒操作 | 变化覆盖、去重、任务分布及泛化评测划分 |
| Event／K 的直接或弱监督 | 时间区间、原始技能与步骤文本 | 分臂并发、hold/idle、Other／空值及标注质量 |
| G 的部分监督或补标种子 | 对象框、交互点、标定和二维轨迹 | 角色一致性、审核质量、执行器区域与终点区域 |
| 失败、恢复与接管研究 | Theme 3 真实接管列、奖励和发布的过程标签 | 全部字段覆盖率、策略来源、质量筛选和目标函数 |

在现有证据下，**继续重点调查 PhysicalReasoning 与 ImitationLearning 的适用子集是有依据的**；前者适合了解结构化标签与条件变化，后者适合了解较大规模长任务示范。RL 主题可以单独评估执行反馈用途；RichInteraction 的探索性数据是否有利于当前训练目标仍待讨论。这个划分是用途假设，不是接入顺序或排除决定。

进一步只读调查最有价值的几个问题：

- 按主题、批次、任务分层后，分臂事件、有效 K、各角色 G 和末端路径究竟能覆盖多少独立轨迹？当前取样偏向小文件且第四期统计主要来自一个任务。
- 原始采集／转换定义能否解释坐标、参考点、动作时间，以及投影时使用的相机参考系？
- 第四期另一批次、高层问答与 Dense Prompt 是否提供当前样本之外的分臂角色和目标状态信息？
- 用同样口径检查 RoboInter、RoboCOIN、ABC 后，哪种组合在监督质量、覆盖和接入成本上更合适？

能否承担主要训练量、需不需要补标、是否调整现有 G/Event 表示，都暂不定论。新的补标、诊断或训练试点仍按 [实验管理规则](../../../workspace/experiment/README.md) 另写具体方案并确认；本文记录的是已完成的资料与文件检查。

## 9. 续查：跨任务、跨批次的元数据覆盖

2026-10-29，在用户同意继续调研后完成。仍为只读文件检查，没有启动补标、投影试验或训练。固定上游 revision 不变。

### 9.1 范围与分母

从公开 export_manifest 的每个 `(batch, task)` 组合中，选择路径排序最前的分片。共 18 个任务、29 个分片；这比首轮覆盖更广，但仍不是随机或全量样本。第一批许多首分片约有 20–21 万帧，任务条数因时长不同而差异很大，不能据此推断全库任务分布。

| 批次 | 分片 | episode 条目 | 元数据帧数 | 步骤区间 | 空间标注帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| opensourcev4-20261001 | 18 | 2,351 | 3,655,921 | 27,742 | 186,720 |
| opensourcev4-20261003 | 11 | 513 | 993,360 | 7,373 | 48,647 |
| 合计 | 29 | 2,864 | 4,649,281 | 35,115 | 235,367 |

各分片 info.json 的 episode 数与帧数均与 episodes.jsonl 对得上。按 30 fps 折合约 43.05 小时，仅为元数据时长，不表示已读取全部动作或视频。

manifest 的 updatedEpisodes 映射覆盖其中 2,845 条，`(source, task_id, episode_id)` 未发现重复；其余 19 条对应 preservedEpisodeCount，没有用该映射完成去重。尚未全量核对实际 Parquet 清单，也未以视频／状态内容去重。

本轮另读取第二批次的 `task_111349/01`（收纳箱搬运）和 `task_75540/01`（番茄炒蛋）各一条 Parquet：分别 3,454 与 4,162 行，实际均为 216 维 state、44 维 action，数值有限、行数与元数据相符。累计两轮已读取 9 条 Parquet、18,364 行；两轮所有 info.json 去除重复路径后为 33 份，覆盖 2,871 个 episode 条目。

### 9.2 对 Event/K/G 的认识如何变化

**时间区间和技能字段覆盖广，完整分臂事件仍缺。** 2,864 条均有 instruction_segments，35,115 个步骤的技能名都非空且不为 Other；但是全部仍在 default 轨道，没有可直接读取的独立左右臂时间线。35 种原始技能写法包含同义变体（Pick／Pick up、Place／Put）、全身运动（移动、转底盘、弯腰）和组合操作（Pick-and-place）。这些数字不能直接当成 ESkill-VLA 的 35 类共享技能。

任务 `task_111349` 已看到文字明确描述双臂合作搬收纳箱；`task_75540` 有左臂夹锅柄、右臂取锅铲及炒菜步骤。这确认数据确有协调行为，但共用时间线没有同时标注另一只臂的持续持物关系。如何构造分臂事件仍需讨论。

**空间标注在轨迹层面覆盖广，在时间上稀疏。** 2,863／2,864 条有至少一个空间标注帧；不同帧共 235,367，占全部帧 **5.06%**，包含 found=false 记录。其中 bbox found=true 为 151,799，point found=true 为 121,235，分别占全部帧 3.26% 和 2.61%。有效存在的框均通过数值范围与正面积检查，这不代表框住了正确对象。

235,367 条空间记录的 audit.status 全部 pending。35,092 个 affordance segment 中，35,089 个 active_arm 为 unknown/disabled，另 3 个缺失。不能依据这一个字段恢复双臂分工，亦不能把审核状态当作视觉质量评测结果。

这一轮没有支持“换个任务／批次就能直接取得完整 K/G/Z”的推断；它支持的更具体认识是：**许多轨迹有可用的结构化标注入口，而监督粒度和角色仍需适配与核实。**

### 9.3 Dense Prompt：历史接受与当前采纳是不同状态

已读取固定版本下的：

- `PhysicalReasoning/.denseprompt_merge/previous_export/export_report.json`；
- 一个 input_manifest（15 条步骤记录）；
- `PhysicalReasoning/.denseprompt_merge/decisions.jsonl` 全部 135,421 条合并记录。

历史导出报告只覆盖 `task_75089` 的 5 个分片、560 个 episode、7,911 个步骤，其中 4,314 段 accepted；该报告同时声明 release_ready=false。它不是当前全库的 Dense Prompt 覆盖率或标签质量报告。

当前合并记录显示：

| 合并状态 | 记录数 | 文件记录的含义 |
| --- | ---: | --- |
| source_preserved | 131,107 | 没有既有的已接受候选，保留原步骤 |
| requires_review | 4,233 | 历史候选接受后，关联 affordance 内容发生变化，需要重新审核 |
| adopted | 81 | 本次合并采纳了候选描述；步骤内容哈希发生变化 |

81 条 adopted 均来自 `task_75089`，其中 6 条位于本轮已读取的 `/01` 分片。逐条核对后，6 条 candidate_instruction 都与当前 info.json 的 instruction 一致。细化描述是**写回 instruction**，并不一定出现在新的 dense 字段或 instruction_augmentation 中；其余 75 条仅核实合并记录，未读取对应的最终 info.json。

因此，不能直接用历史 4,314 段 accepted 声称当前已发布同等规模的可用 Dense Prompt，更不能认为整个第四期已有丰富的分臂描述。合并记录使后续追溯具备基础；若使用，需保存采纳状态和来源，不把 requires_review 自动当成有效标签。

29 份 info.json 都未提供 high_level_instruction，instruction_augmentation 也均为空。task_75089、task_87252 的任务名分别指向天气条件、饮食条件，但本轮还未找到逐 episode 的具体天气／饮食条件如何作为模型输入回链。仅用通用任务名训练条件化决策可能缺少必要上下文，这仍是待查问题；不能据此认定整个发布库没有高层问答。

### 9.4 坐标、深度与边界的进一步线索

- [RoboChallenge 的 G2 原始数据说明](https://huggingface.co/datasets/RoboChallenge/icra_wbc/blob/main/README.md) 区分 state/end/position 与 state/end/arm_position，后者标为 arm-frame positions；动作位置写明米、左右排列和 xyzw 姿态。这是另一套 G2 数据的发布方定义，**不能直接升级为 AgiBot World 2026 的已验证语义**，但加强了“arm_position 不应直接当作世界系位置”的疑问。
- [智元 G2 用户手册](https://www.agibot.com.cn/filepage/362.html) 的附录提到《G02离线采集数据格式说明-rb235.pdf》。本次查看页面 HTML，只有文件名文字，没有该附件的可访问链接；尚未取得这份格式说明。已查看的官方数据卡也没有解决参考系、轴方向、TCP 与动作时序问题。
- 所读 task_75089 input_manifest 明确声明边界为 `[start,end)`，帧号为 episode_local，30 fps。这为右开区间提供了发布文件内的证据，但目前仅核实这一份准备清单，不替代其他主题的边界检查。
- head_depth 元数据声明 PNG／gray16be，不能只凭 shape 的三通道维度判断内容。实际深度读取与尺度仍未核实。

### 9.5 后续调查可以按研究问题选子集

下列是有具体内容依据的候选观察对象，不是训练清单、排序或实验方案：

| 想了解的问题 | 可继续看的任务 | 已有线索 |
| --- | --- | --- |
| 前置操作如何影响技能选择 | task_100347：取被遮挡的钥匙 | 推盒、取钥匙、左右臂协同放入包 |
| 搜索历史与下一步事件 | task_87265：抽屉物品搜索；task_100448：补充洗漱用品 | 搜索／取放序列，需要继续检查历史条件与目标信息 |
| 双臂共同搬运、持物与全身运动 | task_111349：收纳箱；task_75540：番茄炒蛋 | 明确的双臂协作文本和持续持物需求 |
| 对象角色与目标区域 | task_87261：餐盒与盖子形状匹配 | 取放、闭合与容器关系，适合继续核对角色标签 |

现阶段仍保留两种可能：从中筛出能承担较大训练量的子集，或主要利用它的部分结构化监督与特定任务。跨本体训练收益、补标投入与最合适的数据组合，不能由本轮元数据统计确定。
