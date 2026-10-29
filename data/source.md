# 外部数据来源

核实日期：2026-10-29。每项都对照 HF 数据卡与 API、官方仓库、arXiv 或项目页核实，链接附在条目中。规模和字段以核实时的官方发布为准，可能随上游更新变化。本地 RoboTwin `event_data` 不在本页范围内。

## 接入评估口径

以 [design.md](../docs/design.md) §3 的样本接口为准，看数据能提供哪些项：

| 需求 | 说明 |
| --- | --- |
| 事件区间与技能 | 带起止时间的事件，技能能映射到共享词表；双臂数据要能分到各臂事件流 |
| 对象角色 | 被操作对象与参照对象身份 |
| 视觉区域 | 执行器、当前对象、事件终点对象的区域（mask 或框） |
| 三维末端路径 | 各臂末端在统一参考系中的逐帧位置 |
| 动作 | 与图像同步的机器人动作 |

评估档位：**可转换接入**（有动作，主要字段齐全，写适配器即可）、**部分接入**（缺某类标签，需补标或屏蔽对应损失）、**仅作评测**、**仅作参考**（只能借鉴方法、格式或分段粒度）。

## 1. 带机器人动作的事件标注数据

### RoboInter-Data
[HF](https://huggingface.co/datasets/InternRobotics/RoboInter-Data) · [arXiv 2602.09973](https://arxiv.org/abs/2602.09973) · [GitHub](https://github.com/InternRobotics/RoboInter)

- 规模：235,920 条（DROID 152,986，RH20T 82,894），仓库 408 GB。
- 许可：CC BY-NC-SA 4.0；数据卡有申请表单。
- 本体：DROID 部分为 Franka + Robotiq，单臂。
- 格式：推荐 `Annotation_with_action_lerobotv21`（LeRobot v2.1，10 fps，按 chunk 打成 tar）。DROID 175.7 GB，RH20T 93.6 GB。另有 `Annotation_pure`（LMDB，37.3 GB）、`Annotation_raw`（pkl 与 `segmentation_npz`，87.1 GB）、`Annotation_demo_app`（20 条，0.97 GB）。
- 状态与动作：`state` 是 7 维末端位姿 `[x,y,z,rx,ry,rz,gripper]`，`action` 是 7 维 delta EEF；`other_information.*` 还有 tcp 位姿和关节。相机 primary 与 wrist，180×320。
- 标注字段（JSON 字符串）：`time_clip`、`instruction_add`、`substask`、`primitive_skill`、`segmentation`、`object_box`、`placement_proposal`、`trace`（未来 10 步二维夹爪像素轨迹）、`gripper_box`、`contact_frame`、`state_affordance`（接触时 6D 末端位姿）、`affordance_box`、`contact_points`、`origin_shape`。每项另有 `Q_annotation.*` 质量等级（Primary／Secondary）。框为像素坐标，mask 单独存于 `segmentation_npz`。
- 未公开完整的 `primitive_skill` 词表，数据卡只举了 pick、place、push、reach 等例子。

**评估：可转换接入**，监督字段最全。限制：单臂，只能形成一条臂的事件流；许可仅限非商用；三维路径需从 `state` 重算；词表需映射。

### ABC-130K
[XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) · [lerobot/abc_130k_v3_train](https://huggingface.co/datasets/lerobot/abc_130k_v3_train) · [lerobot/abc_130k_v3_val](https://huggingface.co/datasets/lerobot/abc_130k_v3_val)

- 规模：130,703 条、3,590.7 小时，其中 42,980 条带子任务标注；原版 23.3 TB。
- 本体：两条 6 自由度 YAM 臂，平行夹爪。
- 原版：需申请（自动批准），Apache-2.0。格式为 MCAP，目录 `data/{train,val,sim,eval,dagger}/<task>/episode_<uuid>/`。`episode.mcap` 含每臂关节、展平的 4×4 末端位姿、夹爪状态、各相机视频流和标定；`annotation.mcap` 含 `/instruction`、`/subtask-annotation`。子任务消息 schema 在申请后才能查看的文档里。
- LeRobot 转换版：不需申请，LeRobot v3.0，30 fps。train 129,225 条、3.43 TB；val 1,592 条。只有 14 维关节，没有末端位姿；三路相机 224×224。是否保留子任务未核实。

**评估：部分接入**。它是唯一同时具备双臂和逐帧末端位姿的大规模真机数据（MCAP 原版）。没有对象区域；子任务是否分臂，要拿到文档后确认。

### FineART
[HF](https://huggingface.co/datasets/lerobot/FineART) · [arXiv 2609.36416](https://arxiv.org/abs/2609.36416)

- 规模：40,543 条、1,718 小时、151 个任务、533,913 个子任务，约 1.86 TB。
- 许可：CC-BY-4.0，不需申请。
- 本体：Trossen Stationary AI（ALOHA 式双臂）。
- 格式：LeRobot v3.0，30 fps。`observation.state` 与 `action` 均为 14 维关节（左、右各 7）。四路相机 360×640。
- 子任务：存在 `language_persistent` 列，只记开始时间，结束时间取下一条的开始；每个 episode 只有一条子任务流，不分臂。人工撰写。

**评估：部分接入**。双臂真机，人工子任务。缺分臂事件、末端位姿（需按本体模型做正运动学）和对象区域。

### AtomicVLA 标注
[arXiv 2603.07648](https://arxiv.org/abs/2603.07648) · [GitHub](https://github.com/zhanglk9/AtomicVLA)（MIT）

- 文件 `data_split_json/libero_lerobot.json`，2.67 MB，覆盖 LIBERO 1,693 条 episode，共 3,873 段。
- 字段：`segments[{action_name, chain_of_thought, update_chain_of_thought, primary_action_verb, start_frame, end_frame, timestamp_formatted}]`。`action_name` 实际是整条任务指令；逐步子任务文本只出现在 `chain_of_thought` 字符串里。动词分布：pick 1,834、place 1,800、turn 91、open 79、close 69。
- 只有标注，需配合仓库的 LIBERO 转换脚本，episode 编号依赖转换顺序。

**评估：部分接入**，单臂仿真，只有边界和动词。

### LIBERO-SARM
[HF](https://huggingface.co/datasets/k1000dai/libero_subtask_sarm)（社区上传，Apache-2.0）

- 1,693 条、40 个任务，34.9 GB，LeRobot v3.0，10 fps，Panda 单臂。
- 每帧带 `subtask_index`；`meta/episodes` 有 `dense_subtask_names/start_frames/end_frames` 等列；共 52 种子任务。
- 边界由 Qwen3-VL-8B 生成，约 48 条退化为按时长均分。

**评估：部分接入**，单臂仿真，边界质量需抽检。

### robotwin_segmented_2500（ViGAR）
[HF 模型仓库](https://huggingface.co/DAGroup-PKU/ViGAR) · [下载说明](https://github.com/DAGroup-PKU/ViGAR/blob/main/docs/download.md) · [arXiv 2610.02368](https://arxiv.org/abs/2610.02368)

- 位置：模型仓库下的 `data/robotwin_segmented_2500/`，2.50 GB；数据卡 MIT。
- 内容：RoboTwin 2.0，aloha-agilex，50 任务 × 50 条 clean 示范，共 552,287 帧。LeRobot v3.0，30 fps，三路相机 320×240。`state` 与 `action` 是 49 维规范化向量，带 valid mask。
- 阶段：`meta/annotations.json` 中的 `action_steps[{start_frame, end_frame, action_text, stage_index, semantic_stage_index}]`。阶段是全局的，不分臂。阶段结束取首个满足阶段谓词的帧，最终阶段等于任务成功。19 个任务按子目标分段，31 个任务整条只有一个目标；共 3,691 个阶段。
- 目标图像：`goal_cache/` 下 3,691 张 PNG，每个阶段一张，由规划器生成，不是示范帧。
- 不含对象区域和 seed。约 900 小时 AgiBot A2 真机数据（85,043 条）标为待发布。

**评估：仅作参考**。任务集和本体与本地数据一致，但采集批次不同，无法逐帧对齐；阶段不分臂；目标是整图。适合对照分段粒度和阶段谓词。

### Cortex
[HF 数据集](https://huggingface.co/datasets/Steinate/Cortex)（Apache-2.0）· [arXiv 2607.05377](https://arxiv.org/abs/2607.05377) · [项目页](https://steinate.github.io/cortex.github.io)

- JSONL 标注共 425.6 MB，来源包括 AgiBot、BEHAVIOR、Galaxea、RoboCerebra、RoboTwin；只有标注，靠 `video_path` 回链视频。
- `robotwin_norm_mem_train.jsonl`：13.36 MB，13,000 条记录，字段 `{task_id, episode_index, video_path, tasks, length, action_config[{start_frame, end_frame, action_text, skill, language_memory}]}`。逐帧、连续的段。只覆盖 20 个任务，每个任务有 `clean_50`、`collect_100`、`randomized_500` 三个变体；`collect_100` 的最后一段都在 `length` 之前结束。用到 7 种技能：Pick、Place、Handover、Stack、Pour、Open、Rotate。
- 论文附录列出 32 种技能原语；发布的 JSONL 并未统一到这套词表，全部文件共出现 60 种写法（如 HandOver 与 Handover 并存）。
- 没有 seed；`task_id` 沿用 RoboTwin 2.0 官方数据集命名。

**评估：仅作参考**。可对照技能粒度（RoboTwin 上 Pick、Place、Handover 等），以及事件均衡采样的做法；它标注的是官方发布数据，不是本地数据。

### RoboTwin-MeM（EventVLA）
[HF](https://huggingface.co/datasets/ganlinyang/RoboTwin-MeM)（Apache-2.0）· [arXiv 2606.20092](https://arxiv.org/abs/2606.20092) · [GitHub](https://github.com/InternRobotics/EventVLA)

- 45.26 GB；8 个新的记忆类任务，每个 50 条 demo_clean（论文报告 17 个任务）。
- 格式：LeRobot v2.1 + HDF5，15 fps，14 维关节，三路相机 640×480。
- 标注：每条 episode 的 `keyframe_steps`，是记忆证据的关键帧点，由 Qwen3-VL-235B 生成。

**评估：仅作参考**。关键帧是记忆证据点，不是技能边界；任务也不在标准 50 任务内。

### GiT 数据集与 GiTBench
[arXiv 2610.04255](https://arxiv.org/html/2610.04255) · [基准仓库](https://github.com/xgq2005/GiTBench) · [仿真训练集](https://huggingface.co/datasets/XGQ12345/GiTBench_sim_train_lerobotv3)

- 全数据集：18 个任务、4,680 条、30.99 小时，含真机、仿真、UMI 三部分；180 类子任务，带反事实 A/B 任务对。
- 公开的仿真训练集：9 个任务 × 160 条 = 1,440 条，9.64 GB，Apache-2.0。LeRobot v3.0，`bimanual_7dof`，14 维关节，每帧带 `subtask_index`。
- 仿真平台是 ManiSkill-3，不是 RoboTwin。测试集 540 条，分 ID、CF、OOD。仓库提供 π0.5、RDT、GR00T 的训练与评测代码。

**评估：仅作评测**，考察历史事件是否改变下一步决策。

### RoboCombine
[arXiv 2610.15425](https://arxiv.org/abs/2610.15425) · [GitHub](https://github.com/AutoLab-SAI-SJTU/RoboCombine)（MIT）· [HF](https://huggingface.co/datasets/AutoLab-SJTU/RoboCombine)（CC-BY-4.0）

- 基于 RoboTwin 2.0 的 50 个原子任务训练，评测 19 个两段组合任务，每个 20 次 rollout；提出"起始状态差异"（start state gap）。
- 数据包 13.14 GB（未拆开核实本体和场景设置），评测代码含 π0.5 适配。

**评估：仅作评测**。可用来检验技能衔接，平台与本地相同。

### Argus 标注
[Pantheon 研究页](https://pantheon.inc/research/argus) · [pipeline 代码](https://github.com/Pantheon-Industries-Inc/argus) · [独立发布 nkalyanv/argus-robot-annotations](https://huggingface.co/datasets/nkalyanv/argus-robot-annotations)

- Pantheon 官方：审计了 9 个数据集的 3,546 条 episode（66.5 小时），在 data-board 上以 JSON／JSONL 浏览导出。
- nkalyanv 独立发布（与 Pantheon 无关）：数据卡称 290,026 条、约 5,146 小时，来源为 ABC-130k、DROID、MolmoAct2 YAM、ABC-Sim、GM-100；CC-BY-4.0，标签由 VLM 生成。JSONL 每行一条 episode，字段包括：
  - `event_labels`：分臂记录，`t_s, end_s, verb_class, object, carry_phase, progress`；
  - `completion`：`task_completed` 取 success／failure／unclear／success_then_undone，另有目标达成时间；
  - 以及 `key_events`、`operator_mistakes`、`recovery`、`state_changes`、`scene_graph`。
- HF 尺寸标签为 10K–100K，与数据卡的条数不一致，以实际下载为准。

**评估：部分接入（仅标注）**。它是唯一现成的"分臂事件 + 结果标签"格式，可配合 ABC-130k 使用；标签由 VLM 生成，需抽检。

### SPARC DROID Annotations
[HF](https://huggingface.co/datasets/irl-kit/sparc-droid-annotations)（未设许可字段）

- 128,136 行，每行是一个视角下的一个子任务；覆盖 53,282 条 DROID 源 episode 中约 5.3 万条。`data/` 7.29 GB。
- 格式：每个视角一个 `annotations.jsonl`（schema `sparc.annotation` v2.0），附 HDF5 数组：`object_masks`、`robot_masks`、`centroid_traces` 等。阶段类型有 grasp、interact、release。坐标对应原始未裁剪帧；只标注两路外部相机。
- 不含图像、动作、状态；`selection.score` 是样本内排序分，不是校准过的置信度。

**评估：部分接入（仅标注）**，提供对象和夹爪 mask，需按 `source` 字段回链 DROID。许可未声明，使用前需确认。

## 2. 事件边界与技能评测集

### Video2Skill-Bench
[HF](https://huggingface.co/datasets/Sterzhang/video2skill-bench)（CC-BY-NC-4.0）· [arXiv 2609.36691](https://arxiv.org/abs/2609.36691) · [评测代码](https://github.com/andyzworks/video2skill/tree/main/eval)

- 172 段视频、1,671 个事件，视频随数据提供：
  - HD-EPIC 人类视频 40 段、1,137 个事件、46 种技能；
  - RoboInter 机器人视频 132 段（98 段 DROID、34 段 RH20T）、534 个事件、15 种技能。
- 格式：`test/{hdepic,robointer}.jsonl`，`{clip_id, video, duration, fps, events:[{start, end, skill, call}]}`。`call` 如 `grasp(object=lid, source=pot)`，参数角色有 object、source、destination、instrument、direction、substance、state。
- 机器人部分的 15 种技能：grasp、place、transfer、fold、throw、pull、wave、idle、press、insert、manipulate、push、wipe、pour、close。

**评估：仅作评测**。技能与对象参数分开标注，可检验技能能否跨对象复用。

### WGO-Bench
[原版](https://huggingface.co/datasets/macrodata/WGO-Bench) · [VideoDB 修订版](https://huggingface.co/datasets/VideoDB/WGO-Bench) · [修订说明](https://www.videodb.io/blog/wgo-bench-action-annotations)（均为 CC-BY-NC-SA-4.0）

- 原版：100 条 episode、743 个片段。来源为 50 条 RoboInter DROID、25 条 RoboCOIN Galaxea R1 Lite、25 条 HomER 第一视角人类视频。
- 格式：单个 parquet，约 1.40 GB，视频内嵌；`segments[{start_sec, end_sec, subtask}]`，`perception_state` 带末端位姿或关节（HomER 为空）。不含动作。
- 边界原则：片段描述完成的操作事件；新片段一般对应可见的状态变化（拿起、放下、打开等），不为每次小的位姿调整单独切分。
- 修订版：534 个修正片段，固定上游版本 9cfb7bb。

**评估：仅作评测**。边界原则与我们按操作结果定义的 Event 一致。

## 3. 人类视频

### Open-AoE-2000h
[HF](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) · [arXiv 2607.14183](https://arxiv.org/abs/2607.14183) · [工具链](https://github.com/ant-research/Open-AoE)

- 约 2,000 小时第一视角手机视频，约 23 TB，至少 10,723 个样本目录；2026-10-17 全量上传。
- 许可：Open-AoE Dataset License v1.0，允许商用，需署名。手部与相机重建文件另受 MANO 许可约束。不需申请。
- 样本目录：`raw_video.mp4`（30 fps，1920×1080）、`video_info.json`、`ego_annotation/ego_action_annotation.json`、`ego_process/ego_hands_reconstruction/{hands.npz, camera_traj.npz}`、去畸变视频。
- 标注：片段列表，`{id, start_ts, end_ts, start_frame, end_frame, scene, atomic_action[{verb, object, hand, description, bbox, confidence}]}`。片段首尾相接、覆盖全片；`hand` 取 left／right／both／none；bbox 每片段一个（像素），不是逐帧。自动生成。
- `hands.npz`：左右手（索引 0 为左、1 为右）逐帧腕部平移与旋转（世界坐标与相机坐标）、MANO 姿态与形状参数、有效性 mask、逐帧相机位姿，单位米。`camera_traj.npz`：`cam_c2w` 与内参。

**评估：部分接入**，没有机器人动作。`hand` 字段可直接拆成左右手事件流；三维腕部轨迹可构造各手的剩余粗路径；bbox 可作粗区域。

### VITRA-1M
[HF](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M)（MIT）· [arXiv 2510.21571](https://arxiv.org/abs/2510.21571) · [数据说明](https://github.com/microsoft/VITRA/blob/main/data/data.md)

- 1,222,918 段：Ego4D 948,683、EPIC 154,464、EgoExo4D 67,053、SSv2 52,718。元数据约 126 GB；原视频需按各来源条款自行获取。
- 每段一个 `.npy`：相机内外参，左右手的 MANO 参数和关节（世界坐标与相机坐标），左右手各自的语言子片段 `(description, (start, end))`。数据卡称标注准确率约 90%。
- 同组的 `microsoft/VITRA-TeleData` 是机器人遥操作数据（Realman 臂 + XHand，只发布了右臂），不是人类视频。

**评估：部分接入**。有左右手分开的子片段和三维手部轨迹，没有对象区域。

### Assembly101
[项目页](https://assembly-101.github.io/) · [arXiv 2203.14712](https://arxiv.org/abs/2203.14712) · [标注](https://github.com/assembly-101/assembly101-annotations)（CC BY-NC 4.0）

- 362 段装配过程 × 12 个视角 = 4,321 段视频，513 小时。
- 细粒度片段 1,013,523 个（1,380 类），粗粒度超过 10 万（202 类）；1,800 万个三维手部姿态，带相机外参；有错误标注。

**评估：仅作参考**，领域集中在玩具装配；可用于粗细两级边界和错误检测的对照。

### Ego4D Goal-Step 与 EgoExo4D keystep
[Goal-Step](https://github.com/facebookresearch/ego4d-goalstep) · [EgoExo4D keystep](https://docs.ego-exo4d-data.org/annotations/keystep/)

- Goal-Step：goal → step → substep 三级嵌套 JSON，字段 `start_time, end_time, step_category, step_description, is_relevant` 等。标注 MIT；视频需签署 Ego4D 许可。
- EgoExo4D keystep：带 taxonomy 树的步骤片段（`step_id, step_name, start_time, end_time, is_essential`）。需 EgoExo4D 许可；数据集另有对象 mask 轨迹、双手关键点、相机轨迹。

**评估：仅作参考**，可借鉴事件粒度的层级设计。

### Action100M
[arXiv 2601.10592](https://arxiv.org/abs/2601.10592) · [GitHub](https://github.com/facebookresearch/Action100M) · [预览](https://huggingface.co/datasets/facebook/action100m-preview)

- 120 万条教学视频，约 1 亿个片段。用 V-JEPA 2 做层级切分，再用 VLM 和 LLM 生成多级描述。
- 节点字段：`start, end, node_id, parent_id, level` 加各模型生成的描述与动作文本。公开 10% 预览（约 39 GB），FAIR 非商用研究许可；只给 YouTube id。

**评估：仅作参考**，没有手部或空间信息。它的切分流程可与 `vjepa2-event-analysis` 的边界研究对照。

### Something-Something V2
[arXiv 1706.04261](https://arxiv.org/abs/1706.04261) · [HF](https://huggingface.co/datasets/HuggingFaceM4/something_something_v2)

- 220,847 段短视频，174 类，Qualcomm 研究用途许可。只有片段级类别标签。

**评估：仅作参考**。

### EgoDex
[GitHub](https://github.com/apple-aiml-research/ml-egodex) · [arXiv 2505.11709](https://arxiv.org/abs/2505.11709)（CC BY-NC-ND）

- 829 小时，Apple Vision Pro 采集，30 Hz 1080p，约 1.7 TB。
- 每段 `hdf5 + mp4`：相机内参、逐帧相机位姿、左右手及手指关节的 SE(3) 变换；只有整段语言描述，没有时间分段。

**评估：部分接入**。三维手部轨迹质量高，但没有事件边界，需自行切分。

### UniSkill 与 XSkill
[UniSkill 数据](https://huggingface.co/datasets/HanjungKim/UniSkill)（MIT）· [UniSkill 论文](https://arxiv.org/abs/2505.08787) · [XSkill](https://github.com/real-stanford/xskill)

- UniSkill：`Tabletop.tar`（149.6 GB）与 `kitchen.tar`（51.7 GB），数据卡没有内容说明。
- XSkill：仿真与真实厨房数据压缩包，规模和许可未在页面说明。

**评估：仅作参考**，没有事件边界标注。

## 4. 在自有数据上构建事件的工具

| 工具 | 来源 | 公开内容 | 对我们的适用性 |
| --- | --- | --- | --- |
| StructVLA | [arXiv 2603.12553](https://arxiv.org/abs/2603.12553) · [GitHub](https://github.com/wm-planner/structvla)（Apache-2.0） | `tools/structured_frames/` 下的 `structured_frames_extract.py`、`extract_real_robot.py`、`VLM_filter.py`（调用 OpenAI API），输出 `triplets_manifest.csv`。输入 pickle，每条 episode 需 `action [T,≥7]`（delta 位置、delta 旋转、夹爪）和主相机图像 | 关键帧只由动作增量和夹爪翻转决定，不读绝对末端位姿。RoboTwin 是绝对关节动作，需先把末端位姿差分成增量。可作为运动学边界基线 |
| Flora-TempSeg | [HF](https://huggingface.co/nvidia/Flora-TempSeg) · [Captioner](https://huggingface.co/nvidia/Flora-Captioner) | Qwen3.5-9B 上的 LoRA 适配器加推理代码。输入每窗口 64 帧（4 Hz），输出只有片段起止，标签需另用 27B Captioner 生成。NVIDIA 非商用许可；训练数据未公开 | 可生成视频语义边界候选 |
| NILS | [arXiv 2410.17772](https://arxiv.org/abs/2410.17772) · [GitHub](https://github.com/intuitive-robots/NILS) · [标签集合](https://huggingface.co/collections/holgerson/nils-datasets-671b4cc18750e04558d7f98f) | 零样本标注了超过 11.5 万条、430 小时轨迹；已发布 `bridge_NILS`、`fractal_NILS`，HF 上无许可标签 | 依据物体运动与状态变化标注，可参考其规则；代码许可未确认 |

## 5. 构建了事件数据但未公开

| 工作 | 数据来源与规模 | 事件构建方式 | 公开内容 |
| --- | --- | --- | --- |
| AtomSkill（[arXiv 2512.18368](https://arxiv.org/abs/2512.18368)） | RLBench 仿真与真实任务 | 夹爪关键帧分段 + 技能语义 | 项目页代码链接仍是占位，没有代码和数据 |
| VISTA（[arXiv 2602.10983](https://arxiv.org/abs/2602.10983)） | OXE、AgiBot World Beta、自采 Mobile ALOHA；120 万条轨迹、14 种本体 | 按运动和夹爪变化切分，Qwen2.5-VL-72B 合并与描述，转成子任务文本与目标图像交替的序列 | [代码](https://github.com/vista-wm/Vista-WM)与权重；转换后的数据未公开 |
| VPP2（[arXiv 2610.10270](https://arxiv.org/abs/2610.10270)） | 机器人与人类视频 | 按速度极小值和夹爪开合切成 1–12 秒片段，VLM 描述 | [代码](https://github.com/roboterax/video-prediction-policy-2)与权重；事件数据未公开 |
| WALL-WM（[arXiv 2606.01955](https://arxiv.org/abs/2606.01955)） | 机器人与人类视频（规模未核实） | 以变长 Event 对齐语言、视频与动作 | [代码](https://github.com/X-Square-Robot/wall-wm)（Apache-2.0），权重待发布，数据未公开 |
| WorldScape Policy 2.0（[arXiv 2607.18840](https://arxiv.org/abs/2607.18840)） | ManipEvent-5M：74.4 万条、489 万段、4,982.81 小时；其中 RoboTwin 2.0 为 42.16 小时、3.57 万条、14.5 万段 | 运动与夹爪信号定边界，VLM 分层标注 | [代码](https://github.com/manifoldai-research/WorldScape-Policy)与模型；数据集未公开 |
| ForeAct（[arXiv 2602.12322](https://arxiv.org/abs/2602.12322)） | 约 116 万子任务、约 1,000 万帧对 | 直接使用源数据已有的子任务边界 | [代码](https://github.com/mit-han-lab/foreact)、模型，以及一份小规模真机数据集；预训练数据未公开 |
| FineVLA（[arXiv 2605.27284](https://arxiv.org/abs/2605.27284)） | 从 10 个数据集共 972,247 条轨迹中选出 47,159 条，220,606 个步骤标注 | VLM 生成细粒度步骤，人工核验 | [RoboFine-Bench](https://huggingface.co/datasets/xlangai/RoboFine-bench)（500 段视频）、[代码](https://github.com/xlang-ai/FineVLA)与标注模型；训练标注的下载位置未找到 |

## 6. 跨数据集接入的注意点

- **统一事件定义。** 外部数据大多按运动阶段或全局子任务切分，同名 grasp 可能表示夹爪闭合，也可能表示物体被抓起。接入前先映射到我们的技能词表和按结果完成的约定。
- **补出分臂结构。** 现成数据中只有 Argus（nkalyanv）的 `event_labels` 和 Open-AoE、VITRA 的 `hand` 字段区分左右；FineART、ViGAR、Cortex 都是全局一条事件流，需要按动作或末端运动推断各臂的活动区间。
- **显式转换监督形式。** 框与 mask、二维 trace 与三维路径、关节与末端位姿、delta 与绝对动作，在适配器中显式转换，并记录监督有效性；缺失项只屏蔽对应损失。
- **只给标注的数据要回源对齐。** SPARC、Cortex、Argus、AtomicVLA 按源 ID 和时间戳核对，不假设 episode 编号一致。
- **许可。** RoboInter、WGO-Bench、Video2Skill、Assembly101、EgoDex、Flora、Action100M 限非商用；ABC-130K 原版、Ego4D、EgoExo4D 需申请；SPARC 未声明许可。

## 核实中排除的条目

- KiRoSe：在 arXiv、HF 和网页搜索中均未找到。
- `nurlanm/UniSkill_dataset`、`knot-ai/gitbench`、`riceboy26/gitbench`：同名但无关的数据集。
