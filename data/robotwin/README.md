# RoboTwin：Stage 1 仿真数据候选

更新：2026-10-30。基座已确定为 π0.5；本目录是 RoboTwin 专项核对及构建草案，尚未确定训练任务、规模或发布新的训练集。

2026-10-31：用户授权的 **clean 50 任务 × 50 条成功专家示范已全部采齐（2,500 条）**，十条独立核验视频已生成，人工验收待完成。原始数据使用独立的 `/data1/yuankai/robot-research/data/robotwin/clean_native/`，修复续采保存在其 `supplement/` 并保留独立来源身份；本轮不启用场景随机化。协议、原语覆盖与结果见 [collection_clean/README.md](collection_clean/README.md)，视频在 [processed/video/clean/](processed/video/clean)。训练接口与正式 Event 细化仍按后续离线任务处理。

**最新决定：RoboTwin 使用原生相机实例区域，不接入外部 Qwen／SAM 进行视觉 bbox 提取；夹爪支持忽略遮挡、只保留屏内范围的完整框。** 这不移除模型内部的 Target Grounding 学习。当前主要语义缺口是每臂 Event／技能与对象关联，另有指令、时间／动作契约及采集→监督导出等工程收尾，见 [原生字段自查](check/native_field_check.md)。

| 入口 | 内容 |
| --- | --- |
| [collection_clean/README.md](collection_clean/README.md) | clean 50×50 已采齐；十条核验 MP4 已生成，人工验收待完成 |
| [event_plan.md](event_plan.md) | 已确认以原语为骨架：轻量映射、合并与 Hold／Idle 补段；八任务细则待核验 |
| [construction_plan.md](construction_plan.md) | 数据构建草案及新版文件规划：五个离线处理文件、采集职责、依赖与迁移顺序 |
| [guidedvla_pipeline.md](guidedvla_pipeline.md) | 作者论文管线、固定子模块的基础转换入口及待补建部分 |
| [pipeline_review.md](pipeline_review.md) | 自研旧管线核对：采集、分臂事件、状态语义、时间线及训练导出 |
| [architecture_review.md](architecture_review.md) | 软件架构审查：职责边界、完成／恢复契约、版本身份与训练导出 |
| [check/native_field_check.md](check/native_field_check.md) | 当前原生采集对照 docs/data.md 的 11 项字段与剩余工作 |
| [check/field_check.md](check/field_check.md) | 历史 event_data 的 11 项字段初查，保持原范围 |
| [check/README.md](check/README.md) | 10 条已有轨迹的读取范围、抽样规则和待核验项 |
| [check/source_inspection.json](check/source_inspection.json) | 逐条来源、原字段、primitive、文件指纹及数值检查 |
| [scripts/check.py](../../scripts/preprocess/robotwin/check.py) | 新核验入口：字段、固定抽样、验证、渲染 |
| [test/README.md](test/README.md) | 原生相机区域试点：两个任务、4 条新采轨迹的 mask／bbox／crop 核验 MP4 |
| [test/randomized/README.md](test/randomized/README.md) | 八任务 randomized 固定 seed 试采、MP4 与失败记录 |
| [test/occlusion/README.md](test/occlusion/README.md) | 遮挡时完整夹爪框试点，两个仅含渲染后画面的 MP4 |
| [test/events/README.md](test/events/README.md) | 六任务原语候选事件＋原生框＋左右臂时间轴视频，按任务 random 目录保存 |

现有来源：`/data1/yuankai/robot-research/data/robotwin/event_data/`。文件清单为 50 个任务、2,500 个事件 JSON 和 2,500 个 HDF5；文件数不代表全部内容已核验。本次只读取 10 条轨迹的数据字段和视频容器元数据，未渲染、运行策略或补标。

10 条中都有按臂状态／动作、末端位姿、任务文本、primitive 记录和多视角 RGB；所读 HDF5 均没有 bbox／mask。事件记录和视频元数据帧数均比 HDF5 RGB 行数多一帧，需要继续验证终点画面与时间对齐。现有 clean50 不等同于 GuidedVLA 的每任务 1,000 条随机化示范。

后续自研管线核对读取了全部 2,500 份事件 JSON，并对 3 条既有主视频和 HDF5 作只读对照；证据见 [pipeline_review_evidence.json](check/pipeline_review_evidence.json)。当前采集关节字段使用驱动目标，接触查询未按臂过滤，稀疏边界有全局抑制；这些语义需要明确，不能据文件齐全推定全部标签合格。未修改采集实现或旧数据。

## 重构后的使用入口

视觉区域采用原生采集，语义标注可复用 [text_annotation](../../scripts/preprocess/text_annotation/README.md) 的源证据整理；不经过外部视觉 locate／track。下述来源专用 annotate／render／export 保留已有接口，原生试采格式与正式 Event／区域标签仍需桥接。文本模型若参与语义 annotation，不等于重新启用视觉 bbox 提取。

以上调查记录描述重构前的代码／历史数据。2026-10-30 已实施代码重构，旧数据和既有检查记录未改写。随后按确认方案新采了 4 条原生区域试点轨迹，见 [test/](test/README.md)；尚未全量重采或运行模型补标。

```text
RoboTwin/collection/                 ESkill-VLA/data/robotwin/
  collect.py   批量调度                config.yaml
  runner.py    单任务采集              scripts/
  recorder.py  源证据记录                source.py    读取
  storage.py   完整原始存储              annotate.py  标注装配与校验
  regions.py   原生实例区域              render.py    离线 RGB／区域视频
  __init__.py  空
                                        check.py     核验与固定抽样
                                        export.py    显式动作映射与导出
```

**采集环境**沿用 RoboTwin；**离线处理／导出**可使用现有 Pi_05 虚拟环境。离线模块不导入 SAPIEN；只有实际执行导出才导入 LeRobot／Torch。旧 `scripts/collect_data.py`、`scripts/collect_event_dataset.py` 仅转发新命令；重复样本采集、视频命令、汇总脚本已退役。历史 HDF5 写入和事件渲染工具暂保留用于旧格式调用，不进入新采集链路。

从 RoboTwin 根目录执行（以下为使用说明，本次未启动采集）：

```bash
python -m collection.collect --config env_cfg/collection_all50_50eps.yml \
  --save-root /data1/yuankai/robot-research/data/robotwin/NEW_DATASET --dry-run
# 实际采集去掉 --dry-run；新仿真试点仍先确认具体实验方案。
# 单任务：python -m collection.runner TASK CONFIG --save-root NEW_ROOT
```

新原始布局为 `NEW_ROOT/<task>/<embodiment>/raw/episode_XXXXXXX.hdf5`，保留原生任务／本体两级路径，避免为目录外观改变机器人路径约定。运行级 `collection.json` 保存配置、代码与未提交差异；HDF5 自身保存完成状态和来源指纹，不要求额外视频／done 文件。`seed.txt`、`_traj_data/` 仍服务原生规划／重放。旧目录缺少新来源身份时拒绝混写。

`robotwin.raw.v2` 保存完整 `observations`、帧号／时间、指令和源交互证据。`joint_action` 仍指采样驱动目标；新增 `joint_state` 的手臂关节为实测 qpos，夹爪仍明确记作命令。原始记录不移位、不省略终点。新 recorder 的 `evidence_only` 路径按臂查询接触，不推断 grasp／hold、不执行全局 NMS；legacy 模式保留旧事件 API 的行为。

从 ESkill-VLA 根目录执行：

```bash
# 沿用原先十条样本身份，生成独立清单，不覆盖旧 source_inspection.json。
python -m scripts.preprocess.robotwin.check sample \
  --from-report data/robotwin/check/source_inspection.json
python -m scripts.preprocess.robotwin.check validate --manifest data/robotwin/check/raw/manifest.json
python -m scripts.preprocess.robotwin.check fields --manifest data/robotwin/check/raw/manifest.json
python -m scripts.preprocess.robotwin.check render --manifest data/robotwin/check/raw/manifest.json
```

默认读取 [config.yaml](config.yaml) 中的历史 `event_data`，输出 `check/raw/`。不传 `--from-report` 时按 seed 固定抽取不同任务的 episode；已存在的样本清单不会自动重抽。渲染保留源帧率，末帧缺图显示 MISSING，不伪造或截到上一帧。视频蓝框为夹爪、绿框为当前对象、黄框为 Event 终点同对象；没有补标时不出现这些框。原 primitive 明确作为来源展示，技术检查不填写人工通过。

### 标注与导出

适配公共 [数据接口](../../scripts/preprocess/fields.py) 时，RoboTwin 沿用原生完整夹爪区域：夹爪框使用
`RegionSequence.boxes_xyxy`，`region_extent="full_in_frame"`，可用性写入
`boxes_valid`。对象可见区域按 `region_extent="visible"` 记录；若提供 mask，
单独填写 `masks_valid`。这是 RoboTwin 的标签选择，公共接口不限定其他来源采用
可见区域还是完整区域；现有导出器尚未统一输出该接口。

`annotate.py --manifest ... --output NEW_ANNOTATION_DIR` 只装配源证据，`events/regions` 保持空和缺失；提供 `--labels LABEL_DIR` 才载入逐任务／episode 的显式标签。没有默认技能映射、Hold 推断或模型补标，不把“脚本能运行”写成“监督已齐全”。

当前项目侧标注支持 `events` 与像素 bbox `regions`：Event 包含 `arm/start_frame/end_frame/skill` 及用于关联的 `object`；区域包含 `arm/frame_index/camera/role/object/bbox`，`role` 为 gripper 或 object，bbox 为像素 xyxy。Event 当前帧范围为 `[start_frame,end_frame)`，目标画面取 `end_frame`；源 primitive 的原始闭区间另存，不能未经判断直接复制。对象关联只用于追踪标签，不作为新增模型输入。像素 mask／自动补标后端尚未实现。

```bash
python -m scripts.preprocess.robotwin.annotate --manifest CHECK_MANIFEST \
  --labels LABEL_DIR --output NEW_ANNOTATION_DIR
python -m scripts.preprocess.robotwin.check render --manifest CHECK_MANIFEST \
  --annotations NEW_ANNOTATION_DIR --output data/robotwin/check/annotated
python -m scripts.preprocess.robotwin.export --manifest TRAIN_MANIFEST \
  --annotations NEW_ANNOTATION_DIR --output NEW_EXPORT_DIR \
  --repo-id local/eskill_robotwin --state-source stored --action-source stored
```

`TRAIN_MANIFEST` 按 episode 明确指定 `split: train/val/test`，不是自动把 10 条核验样本全部当训练集。历史格式使用 `stored/stored`，不再次移动动作；raw v2 必须显式选 `--action-source next-command`，状态选 `command` 或 `measured-arm-command-gripper`。这只是可选的采样命令映射，不声称它已被确认为最终专家动作契约。

导出生成 `lerobot/`、辅助标注／末端记录及来源帧映射。先写独立 `.building` 目录，成功后发布到新输出目录；已有结果不覆盖。不同 fps 不隐式拼成同一时间轴。辅助 RGB 仍引用固定哈希的原始 HDF5，保留原始数据是读取终点画面的前提。`export.auxiliary_sample` 提供同对象终点区域与按弧长采样的剩余路径；π0.5 模型的去噪／技能／路径损失接入属于后续训练实现。这里未计算训练归一化统计，也没有将辅助标签喂入在线输入。

### 当前验证范围

文件级回归覆盖完整终点保存、历史动作不重复移位、RGB 通道、分臂证据、并发 Event、同对象终点、MP4 编解码、LeRobot 写入后重载及离线依赖隔离。新读取器已只读检查原有十条样本。原生区域试点另完成 4 条真实 SAPIEN 专家采集；模型区域／技能补标、人工标签验收和 π0.5 训练尚未执行。

本轮 24 项回归通过；原有十条样本的新报告见 [check/raw/report.json](check/raw/report.json)。测试使用既有 Pi_05 环境，缓存设置在 `/tmp`，未安装或合并依赖环境。采集 requirements 已去掉重复 OpenCV 提供包和重复 imageio；共享导出器区分编码 codec 与解码 backend，并使用 LeRobot 的完整 finalize 接口。

原生区域试点使用 `collection/regions.py` 读取同帧原始整数分割 ID，生成可见 mask、紧致 bbox 和 RGB crop；4 条、842 帧技术核对通过，人工验收待完成。该能力尚未默认接入 `collection.runner`／raw v2 或语义 annotation。历史数据没有因此获得区域标签。另发现 `save_freq=15` 是控制步采样间隔，含动作边界附加帧；试点保存真实物理步号和时间，后续全量链路须落实时间语义后再用于训练。

八任务 randomized 扩展使用 `collection/native.py` 作为固定 seed 的原生区域预览入口，支持独立多对象、参考物和关节模型 link 合并；另存同步状态／命令、末端、相机内外参及源交互记录。这里的 `robotwin.native_regions.v2` 为可追溯的试采格式，尚未接入离线训练导出，未替代 `robotwin.raw.v2`。源交互记录中的标称 frequency／时间保留来源含义，准确仿真时刻以记录的物理步号与 timestep 为准。

完整夹爪框可通过 `collection.native --gripper-full --config clean|randomized` 选用：独立辅助场景仅渲染当前姿态的本臂手指，忽略外部遮挡、保留相机视野内的部分；真实 RGB 与 visible 标签继续保留。两条／530 帧技术核对通过，真实完全遮挡样例尚未覆盖；对象区域仍为可见定义，训练导出尚未切换。新视频仅含已叠加标注的画面与真实 RGB crop。

原语候选预览已完成六任务各一条 random：五条新采、汉堡一条复用，909 帧、25 个候选 Event。`annotate.py --native-record SOURCE --output EVENTS.json` 可生成显式任务规则下的候选映射／合并及有证据的 Hold／Idle；`render.py --native-record SOURCE --events EVENTS.json --output VIDEO.mp4` 在完整夹爪框画面下叠加两臂时间轴。输出含源原语回链与独立 goal_frame，所有人工结论 pending；当前是预览格式，尚未桥接为正式 EventData／训练导出。视频见 [test/events/](test/events/README.md)。
