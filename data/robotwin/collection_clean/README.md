# RoboTwin clean：原语补齐与 50×50 采集

状态：采集完成；50 任务各 50 条，共 2,500 条。十条核验视频已生成，人工验收待完成。创建日期：2026-10-31（Asia/Shanghai）。

授权范围：用户本轮明确要求“给各个任务补充原语……并接着进入数据收集”，且“这一轮我们只收集 clean 环境下的”；规模沿用本对话的 50 个任务、每任务 50 条成功专家示范，共 2,500 条。包含必要的采集验证与既定十条渲染核验；不运行策略训练或新增对照实验。

## 目标与协议

任务清单沿用 `RoboTwin/env_cfg/eval/all_tasks.yml` 的 50 个任务，本体为 aloha-agilex。仅用 `demo_clean` 的场景参数：不随机背景、杂物、灯光、相机距离和桌高；原生任务的位置／模型抽样保留。专家仍执行现有动作，不为标注改变目标、轨迹、规划器和物理参数。

先检查各任务直接构造的 Action 及通用原语；补缺失的原语、阶段、执行臂、对象身份与真实起止时间。倾倒、摇晃、关节物体操作等专用片段保留任务语义；重复尝试保留，不假定每个原语都成功。候选 Event 的细化、Hold／Idle 合并及 EpisodeData／π0.5 适配后续离线进行。

采集保存三相机 RGB、主相机无损整数实例 ID／对象区域／可见与屏内完整夹爪区域、相机参数、机器人测量状态与驱动命令、世界系末端位姿、真实物理步和时间、任务指令、原语与按臂接触证据。保存实际控制时序，明确采样驱动目标与实测关节量；保留同刻源记录和终点快照。任务执行期间的 delay 连续采样，初始化等待不并入示范。

每任务从 seed 0 顺序尝试，收满 50 条通过原生 `plan_success && check_success()` 且采集结构核对通过的轨迹；这是专家示范筛选，不是策略成功率评测。失败／不稳定／重放失败单独登记，不计入 50 条，也不冒充人工验收。保留尝试 seed 与成功 seed 对应，不复用旧 clean 数据冒充本轮。

## 预算与验证

- 最多四张空闲 GPU，每卡一个采集进程；GPU 仅在调度日志中标识。
- 沿用每任务最多 6 小时；每任务最多考察 500 个 seed。本轮总运行上限 48 小时、总新增数据上限 400 GB；触及上限记录未完成任务，不自动扩大预算。
- 修改后先做文件级测试，必要的短片验证置于 `workspace/.experiment_tmp/robotwin_clean_primitive_collection/`，不计正式分母；成功后清理可重建临时内容。
- 正式采集逐轨迹核对 RGB／状态／区域帧数、时间单调非递减、原语起止与对象身份、文件完整性及原生成功结果；每任务第一条验证通过再继续该任务。
- 汇总分别列目标数、成功落盘数、规划失败、不稳定、重放／采集异常与剩余数，最终是否齐全以 50 个任务各 50 条为准。

## 输出与视频

大文件使用独立新根目录 `/data1/yuankai/robot-research/data/robotwin/clean_native/`，不覆盖历史 `event_data/`。任务与本体两级目录沿用当前原生批量入口。运行配置、源码身份、调度与 seed／尝试记录随数据保存。

本目录保存任务原语覆盖表、采集索引和简短汇总。采集后按固定 seed 42、不同任务优先抽十条独立轨迹，核验 MP4 放 `data/robotwin/processed/video/clean/`，对应身份与人工表保存在 `processed/`。只输出带标注的渲染画面，不额外生成 2,500 个原始 MP4。

视频沿用已核验的三视角布局，主视图显示原生对象框与双夹爪完整框，下方显示左右臂原语／阶段时间轴、任务／seed／源帧／真实仿真时间；未做正式 Event 映射的片段明确标为源原语。腕部只展示 RGB，不新增腕部区域训练要求。按保存顺序、标称 15 fps 核验播放，明确不等间隔源采样，不称为实时播放。视频可解码属于技术核对，人工验收保持待核验。

```text
ESkill-VLA/data/robotwin/
├── collection_clean/
│   ├── README.md
│   ├── primitive_coverage.csv
│   └── summary.json
└── processed/
    ├── clean_manifest.json
    ├── clean_manual_review.csv
    └── video/clean/*.mp4

/data1/yuankai/robot-research/data/robotwin/clean_native/
├── collection.json
├── collection_report.json
├── logs/
└── <task>/aloha_agilex/
    ├── raw/episode_XXXXXXX.hdf5
    ├── seed.txt
    └── _traj_data/
```

## 执行记录

2026-10-31：完成 50 任务静态检查，覆盖表见 [primitive_coverage.csv](primitive_coverage.csv)。直接 Action 出现在 click_alarmclock、dump_bin_bigbin、put_bottles_dustbin；闹钟已有 click 标记，另两任务补 pour／transport_to_bin。其余通用原语增加场景实体 ID 与真实起止；调姿、工具接触、摇晃、开笔记本／微波炉／抽屉、扫描、盖章补专用来源标签。唯一直接调用 together_open_gripper 的任务位于加载阶段，不作为执行期漏标。未覆盖的运行路径会保存为 unclassified_action 并拒绝计入正式数据，不默认猜成技能。

原生区域已接入正式 runner，原始 HDF5 沿用 raw v2 容器并明确 `clock=physics`；同刻 RGB 行完整保存。主相机区域单独保存 visible 与 full_in_frame 的有效性，场景身份跨臂共享且不以重名字符串代替实例 ID。按臂接触查询同时过滤对象实例 ID，避免多只同名瓶子混淆。三相机图像均保留。另保存逐物理步驱动命令、实测手臂状态／夹爪命令及其最近已保存 RGB 行索引，明确命令在该时间起执行一个物理步。

新入口一次执行专家并保存成功记录，不额外重复执行同一条已规划轨迹；成功轨迹的规划缓存仍保留。失败 seed 按顺序登记并继续，不引入策略推理或外部标注模型。

文件级回归 **34 项通过**。临时验证第一轮七任务成功；倾倒路径因给 ArmTag 深拷贝失败而中止，修正为只附加动作元数据后第二轮通过，未改变原有运动命令。八条结果如下，不计入正式 2,500 条分母：

| 任务 | 成功 seed | 源帧 | 原语区间 | 实际控制步 |
| --- | --- | --- | --- | --- |
| adjust_bottle | 0 | 144 | 3 | 1,923 |
| dump_bin_bigbin | 5 | 306 | 7 | 4,017 |
| lift_pot | 0 | 116 | 6 | 1,544 |
| open_microwave | 5 | 617 | 18 | 8,011 |
| place_can_basket | 0 | 247 | 9 | 3,305 |
| place_object_stand | 0 | 140 | 3 | 1,818 |
| put_bottles_dustbin | 0 | 665 | 22 | 8,934 |
| shake_bottle | 3 | 254 | 9 | 3,371 |

倾倒记录含三个 Pour 周期，随后保留双臂 wait 来源区间；7.2 秒等待已连续采样，整条记录最大相邻时间差约 0.06 秒，同刻的 17 对源行保留。锅的渲染预览已完整解码并检查三视角、完整夹爪框、物体框与双臂来源时间轴；这属于采集与呈现检查，不填写人工通过。

倾倒等待画面也已检查：该例终段夹爪和小桶离开主相机视野，区域按屏外无效保存，不生成假框；后续相关区域监督应使用其有效性。临时验证的可重建视频、原始记录和缓存已清理，八条结论保留在本表。正式四任务首条另已重新核对，原语／对象身份／控制时序均通过，来源代码与运行快照一致；新增原始格式也能被现有 source reader 读取。

全量入口为 `python -m collection.collect --config env_cfg/collection_clean_native.yml --save-root /data1/yuankai/robot-research/data/robotwin/clean_native`。后台启动信息在数据根目录 `launcher.json`，总日志 `collection.log`，逐任务进度为 `<task>/aloha_agilex/progress.json`，每次尝试为 `attempts.jsonl`；完成后自动执行十条抽样与标注渲染，汇总见 [summary.json](summary.json)。

手工刷新汇总可从 ESkill-VLA 根目录运行：`python data/robotwin/scripts/check.py collection --source-root /data1/yuankai/robot-research/data/robotwin/clean_native`；加 `--render` 会复用或创建固定十条清单并渲染。正式数据未齐时会明确标记 `all_complete=false`，不把部分完成写成 50×50 已完成。

### 异常修复与缺额续采

2026-10-31 上午查询时，原批次已完成 39 个任务，4 个任务因程序异常退出，其余仍在运行。已定位并修复：

| 任务 | 当时已采集 | 原因与修复 |
| --- | --- | --- |
| click_alarmclock | 31/50 | 无可用抓取位姿时直接切片 None；检测后按专家规划失败登记，不进入动作执行。 |
| put_bottles_dustbin | 1/50 | 抓取失败后仍索引空动作列表；右臂与左臂两条分支均在失败后结束本次尝试。 |
| stack_blocks_three | 2/50 | 同一任务实例的 last_actor 残留上一场景引用；采集器改为每次尝试创建独立任务实例。 |
| stack_blocks_two | 2/50 | 与三方块相同的对象生命周期问题，同步修复。 |

修复先在隔离目录 `/home/yuankai/robot-research/workspace/checkouts/robotwin_clean_recovery/` 实施，原批次源码保持不变。具体改动见 [failed_tasks_fix.patch](failed_tasks_fix.patch)。续采与来源合并共 40 项相关回归通过；覆盖无解位姿、双臂抓取失败、逐轨迹实例隔离、父批次文件只读、样本编号／seed 接续及跨来源重复编号拒绝。

自动续采已登记在 `clean_native/supplement/launcher.json`，其 `launcher_status.json` 标明等待／运行状态。它等待原批次采集进程退出，再以最多四卡补齐所有未满 50 的任务；样本编号和 seed 从父批次接续，不重新采集已有成功轨迹。仍仅 clean，使用原始启动时刻起算的 48 小时总期限、合计 400 GB 和每任务 500 个 seed 的边界。原批次日志、错误与 manifest 保持原样，补采使用独立的 `supplement/collection.json` 与指纹，不混写来源身份。

后续查询时原批次已达 45 个任务完成，只有 put_object_cabinet（40/50）仍占 GPU 2，另外三卡空闲。为并行完成缺额，已将上述四个修复任务提前调度到 GPU 0／1／3；原等待进程只负责排队，已被新的续采调度接替，其启动信息保留在 `launcher.json` 的 `previous_waiting_launcher`。原柜子采集继续，始终最多四个 GPU worker；样本、clean 条件、来源版本与总预算不变。主源码补丁仍在原批次退出后检查应用，最终视频仍等待合计齐全。

原批次退出后会检查并应用补丁到主源码；若主源码上下文已被其他修改改变，则保留现场并使用隔离源码执行，不强制覆盖。完整采集入口新增 `--resume-from`，逐轨迹沿用原成功编号，补采文件仅写在 `supplement/<task>/aloha_agilex/raw/`。合并汇总分别验证来源指纹，任何任务的重复 episode 编号均报错；历史原始文件不改写。

已登记续采时，十条最终抽样等待合计 2,500 条全部齐全再冻结；此前不将部分池子误写成完整采集验收。合并进度仍使用本目录 [summary.json](summary.json)，其中 `source_versions` 记录两批来源。

## 最终结果

两批合计 **2,500／2,500 条，50／50 个任务均各 50 条**。首批保留 2,336 条，修复版本续采 164 条；首批 report 中四个异常任务保留历史失败记录，续采四任务全部完成。合并时逐文件验证来源身份并检查任务内 episode 编号唯一，`all_complete=true`。修复补丁已应用到主源码，续采与后处理退出码均为 0。

固定 seed 42 从完整 2,500 条池子抽取不同任务的十条轨迹，清单见 [clean_manifest.json](../processed/clean_manifest.json)，`collection_complete_at_sampling=true`。十条 MP4 已完整渲染，渲染器逐视频完整解码核对；本轮另读十份视频报告与对应 HDF5 确认帧数均一致。视频目录为 [processed/video/clean/](../processed/video/clean/)，人工表 [clean_manual_review.csv](../processed/clean_manual_review.csv) 仍为 pending。

完成的是原始数据采集、技术核对与十条可视化。正式 Event／技能语义、逐 Event 的 obj／goal 监督及 EpisodeData／π0.5 训练导出仍属后续离线工作。

### 堆三个碗的对象身份核对

只读核对 `stack_bowls_three/aloha_agilex/raw/episode_0000000.hdf5`（seed 1，443 源帧）。三个碗名称均为 `002_bowl`，场景实例映射分别为 bowl1→68、bowl2→69、bowl3→70；这些 ID 只对该轨迹有效。源 grasp／place 原语通过 `metadata.actor_entity_ids` 记录本次操作的具体实例，区域通过同一 ID 提取。

| 源 Place 操作 | 操作实例 | 放置参照（来自专家脚本） | 源 Place 结束行 | 同一对象的终点 bbox |
| --- | --- | --- | --- | --- |
| 放置第一只碗 | bowl1，ID 68 | 桌面指定位置 | 138 | [141, 121, 213, 187] |
| 第二只碗叠到第一只碗 | bowl2，ID 69 | bowl1 | 271 | [139, 109, 215, 179] |
| 第三只碗叠到第二只碗 | bowl3，ID 70 | bowl2 | 426 | [139, 98, 217, 172] |

三个源终点均有同刻 RGB 和有效可见区域。最终 goal 由本 Event 对象在本 Event 结束时的区域派生：第二只碗的 Place 取 bowl2 的终点区域，承托它的 bowl1 是放置参照；每段分别使用自己的结束时刻。同一只碗的 Pick 与 Place 绑定相同实例，但各取自己的终点。上述源行号仅用于核对，不冒充尚未转换的公共时间轴 EventData.end_frame；本次未新增正式 Event 标签。撤离来源中 last_tracked_target 的引用也不自动证明仍在持物，后续须按无对象／未知的实际语义处理有效性。
