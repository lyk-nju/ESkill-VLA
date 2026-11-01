# 数据调研目录规则

- 2026-11-01 用户指定处理后的 AtomicVLA LIBERO-40 数据统一放在 `/data1/yuankai/robot-research/data/atomicdata/libero40/episodes/`，批次索引为同级 `index.json`。项目内 `check/test/full_frame_v1/episodes/` 与 `index.json` 链接到此位置；后续保存与读取沿用该目录。迁移记录在 `libero/atomic-vla/check/archive/location_20261101/migration.json`。

- 2026-11-01 用户要求精简 AtomicVLA 重复检查内容后，旧模型补框、诊断、100 条／v2 和保存／迁移试点已集中归档，入口为 `libero/atomic-vla/check/archive/README.md`。历史记录按原 `check/` 相对路径在 `archive/history_20261101/` 查询，原内容与逐文件 SHA256 保留。当前验收入口为 `check/README.md`；来源、人工实例、技能确认及官方 LIBERO-40 仍依赖的两份固定 100 条清单保留原运行路径。下方历史讨论中的旧检查路径应按此映射查询，不为查看历史重新铺回当前目录。

- 2026-11-01 用户确认按 OpenPI 主结构组织项目，并取消独立的 `datasets/`：数据读取沿用 AtomicVLA 的 `policies/xxx_dataset.py` 方式，放在 `src/openpi/policies/eskill_dataset.py`；训练加载与配置归 `src/openpi/training/`，预保存字段说明和离线处理归 `scripts/preprocess/`，薄入口归 `scripts/`。`data/` 保留数据、来源配置与核验证据；该约定更新下方历史讨论中代码位于 data 的旧路径。迁移映射见 `scripts/preprocess/README.md`，旧核验报告和原始数据不改写，原代码 SHA256 可用 `examples/archive/data_code_20261101.tar.gz` 核对。
- LIBERO 主来源已选定 AtomicVLA 的事件标注＋作者指定的基础数据；其他候选未自动采用。每个数据集的调研说明、字段核对、统计清单放在 `data/<数据集>/`，以该目录的 `README.md` 为入口；专用代码在对应 `scripts/preprocess/<数据集>/`。
- 当前仿真数据引擎已明确分为 LIBERO 和 RoboTwin 两条链路，统一 π0.5；来源版本、任务、规模和补标方案仍需核对，外部真实数据候选也未定案。每条链路沿用字段自查、10 条渲染及人工验收。
- `data/` 根目录保留公共入口、跨数据集来源索引和共性格式讨论。来源索引可保留比较摘要，详细检查证据回链各数据集目录。
- 用户已确认共享模型预处理分为 `scripts/preprocess/visual_grounding/` 与 `scripts/preprocess/text_annotation/`；视觉补标组合为 Qwen3-VL＋SAM3。产物仍归各数据集目录，不在共享代码目录堆放样本和视频。
- 2026-10-31 用户后续明确：供人工审核的补标测试产物放各来源 `check/test/`，视频统一 `check/test/video/`；当前 AtomicVLA 使用 `data/libero/atomic-vla/check/test/video/`。这更新了测试直接放 processed 的旧约定，正式产物路径仍可使用 processed；已有历史结果不因路径更新而搬动或改写。
- 视觉区域当前只要求主相机中的各臂夹爪与操作物体，终点对象区域也从同一主相机读取；LIBERO 使用 `image`，RoboTwin 使用 `cam_head`。暂不处理腕部相机区域，原始腕部图像及历史多视角结果保留。
- 2026-10-31 用户要求 LIBERO 各来源共用 `scripts/preprocess/libero/render.py` 渲染器，来源在 `view.py` 适配已有字段；缺字段显示 MISSING，不由渲染器猜标签、修端点或补 bbox。共用输入及入口见 `data/libero/README.md`。新版人工核验视频用 `check/test/video/*_review.mp4`，与旧结果按各自 report 区分。
- 用户随后要求补三维 XYZ 轨迹图；共用渲染器复用 `eef_xyz`，同时保留三维空间轨迹及 X／Y／Z 时间曲线。固定视角、等比例轴，标当前点、当前 Event 路径和实际终点，缺失点不连接。此次用 `--tag 3d` 另存 `*_review_3d.mp4` 与对应 render_3d 报告，原布局保留。
- 2026-10-30 后续明确：RoboTwin 使用已验证的原生实例区域与完整夹爪框，不接外部视觉模型提取 bbox；Qwen3-VL＋SAM3 视觉补标继续用于 LIBERO 等需要它的来源。RoboTwin 的语义 annotation 与模型内部 Target Grounding 监督仍保留，不混同于外部框提取。
- 同一数据集的主题、批次与格式差异在其目录内部记录；需要新增文件时按内容组织，不将单一候选的接口默认为全项目接口。
- 2026-10-31：LIBERO 按来源分为 `data/libero/guidedvla/` 和 `data/libero/libero-90/`。前者保存既有发布版调研、脚本与核验结果；后者用于拟自建链路，尚未执行。两者分别保存 check／processed，不混用来源身份或验收结论。
- 2026-11-01 用户要求补官方 Spatial／Object／Goal／LIBERO-10 的 HDF5 来源，代码与对齐产物放 `data/libero/libero-40/`。与现有 AtomicVLA／RLDS 的动作对应、状态差异和图像核验分别记录；找到官方状态不等于原图像素对齐，也不自动替换历史补框结果。
- 用户随后新增 `data/libero/lilo-vla/`、`atomic-vla/`、`cyclevla/`，分别按 docs/data.md 自查并将原版核验视频放在各自 `check/raw/video/`。LiLo 的再生成技能轨迹、Atomic 的完整任务轨迹、Cycle 的子任务片段分别记清单位，不把增强副本或同父轨迹的多个片段算成独立完整任务。
- 2026-10-31 用户抽看后认可 AtomicVLA 大体可用并确定为后续 LIBERO 主来源。沿用源事件和技能标签，优先用 Qwen3-VL＋SAM3 补主相机夹爪／操作物体 bbox；终点对象 bbox 从实际 Event 结束帧读取。端点约定及动作／坐标／同步在接入时核实；处理后视频放 `data/libero/atomic-vla/processed/video/`。其他来源保留参考，LIBERO-90 为扩展备选；来源选择不自动填写逐条人工验收结论。
- 2026-10-31 后续用户查看 AtomicVLA 原生夹爪三帧试点后，明确夹爪采用相机原生渲染路线；以源关节状态恢复手掌／手指几何，生成忽略外部遮挡并裁掉屏外部分的完整框。用户同时要求物体尝试原生分割，先核对完整场景状态与源 RGB 的对应；物体路线尚待试点，不将随机场景或未匹配镜像视为原轨迹。更新了此前夹爪优先 Qwen3-VL＋SAM3 的约定，旧模型结果保留。
- 用户随后明确夹爪、操作物体及同一 Event 的终点对象均采用忽略外部遮挡的完整屏内区域（`full_in_frame`）。目标分别独立渲染，彼此不遮挡；屏外部分裁掉，完全出画无屏内框。完全遮挡不等于标签无效或无对象，原 RGB 仍保留遮挡内容。历史 visible mask／bbox 保留原义，不改名冒充完整区域；既有批次是否补齐需分别核验。规则见 docs/data.md，LIBERO 三帧对照见 atomic-vla/check/test/native_object_full/。
- 用户明确要求实例绑定规则尽可能通用。核心逻辑依据文本语义、场景实体／区域／部件关系、事件上下文及可核验的运动证据，不按任务名、轨迹号、Event 序号或实例编号指定答案。类别别名及模拟器几何名称属于来源适配，空间关系优先查询区域归属和场景关系，避免逐任务追加完整句子的查表特例。保留显式属性约束和歧义，不用候选顺序、最近物体或最大位移强行兜底。现有 100 条已用于调研和规则分析，只能作为开发／回归样本；泛化结论需另用未参与规则制定的任务或组合验证。具体约定见 `libero/atomic-vla/README.md`。
- 2026-11-01 用户批准 AtomicVLA 同臂共享边界帧归后一个 Event，前一 Event 的 stop／goal 相应更新；由来源适配器转换并保留原始边界，不由渲染器修复。核验材料见 `libero/atomic-vla/check/test/annotation_review/`，视频在对应 `check/test/video/`。历史批次仍按各自记录的边界约定解释。
- 用户观看 33 条 Pick／Push 冲突视频后确认是 Push：“压住碗然后用摩擦力往外移动”。当前固定 AtomicVLA 发布版中，事件描述明确为 Push 而原标签为 Pick 的 33 个事件，派生技能统一修正为 Push，来源 Pick 和用户依据保留；不按任务或实例编号改标签。此确认针对本组技能类别，不扩展为对象类别、bbox、场景对齐或全数据验收。新版事件与视频见 `annotation_review/push_reviewed/`。
- 用户随后要求全量测试当前通用规则。固定 1,693 条／3,873 Event 的来源、动作、逐帧运动和关键帧原生区域检查在 `libero/atomic-vla/check/test/all_rules_v1/`；1,681 条具备完整唯一候选，12 条仍有 34 个绑定缺口，740 条有独立的状态对齐警示。全量 bbox 检查仅限 Event 关键帧，另有 10 条完整视频，不将候选覆盖或技术检查说成全数据人工验收通过。本轮不调规则或放松运动质量阈值。
- 用户明确授权手工确定上述 12 条的实例。助手逐条检查原始主相机事件图、原生实例锚点和运动／接触记录后，将 34 个缺口另存 `libero/atomic-vla/check/test/manual_binding12/annotations.json`，全量合并实例标签为该目录 `merged_bindings.json`。这些是来源专用的 `manual_bound`，不是通用规则中的任务／轨迹查表；不计入自动覆盖率。标注者为 assistant_visual_review，用户验收保持 pending；实例补充不消除 740 条状态对齐警示。
- 2026-11-01 用户明确要求完成全量逐帧夹爪／物体 bbox 并准备人工验收。当前批次为 `libero/atomic-vla/check/test/full_frame_v1/`，覆盖固定 1,693 条／273,465 帧；代码入口 `scripts/preprocess/libero/atomic_vla/batch.py`。大体积标签与视频在数据盘，通过本批次 `episodes/` 和 `check/test/video/full_frame_v1/` 链接访问。预选 55 条验收视频覆盖全部 40 任务、12 条助手实例补充和各套件最大对齐偏差，具体 ID 与理由以本批次 manifest 为准；全部人工结论保持 pending。技术完成范围以 report.json 为准。
- 大体积原始数据与缓存继续存放于 `/data1/yuankai/robot-research/` 下对应数据集目录，项目内保存来源路径、版本及核对依据。
- 移动文档时同步修正相对链接；保留原始数据版本、样本身份、统计口径和历史检查范围，不把目录整理写成新的数据核验。

## 统一核对流程（用户于 2026-10-30 明确要求）

- 2026-11-01 用户要求先设计预保存格式／接口，再批量产出并保存标签，最后可视化。当前字段在 `scripts/preprocess/fields.py`，读取在 `src/openpi/policies/eskill_dataset.py`，保存与结构校验在 `scripts/preprocess/export.py`；不保留独立 `io.py`。每条轨迹保存 JSON＋NPZ、原 RGB 引用及来源证据；Event 使用显式 `start_frame/stop_frame/goal_frame`，源频率未知时使用 source_steps，不能伪造秒数。后续可视化从保存接口回读，不在渲染阶段重绑对象或重算框。现有旧格式仍按原版本解释，来源适配须显式转换。
- 2026-10-30 用户本轮明确要求在 `scripts/preprocess/fields.py` 设计数据接口，便于后续 agent 判断预处理缺口，此要求更新此前暂不建立 schema 的约定。逐项语义需求以 `docs/data.md` 为准，字段结构与预处理清单见 `scripts/preprocess/fields.py`；两者应保持一致。该接口描述示范数据，不自行固定存储容器、训练超参数或新增模型输入。
- 每个来源的 `field_check.md` 简约使用三列：**我们需要什么／有没有／备注**。对照 `docs/data.md` 逐项填写有／部分／无／待核实；备注只写主要缺口、待确认语义及补充途径。详细字段路径、统计与证据保留在对应来源记录中。
- 有字段、数值合法、语义明确和标签正确分别核对。查不到记待核实，不将 unknown／默认值当作否或有效标签。
- 缺口写出可评估的补充途径，例如检测／分割／跟踪补 bbox、运动学恢复末端、人工或模型辅助整理事件；建议方法不等于已补齐。实际补标与原始标签分开保存并保留来源。
- 字段自查后，抽取 10 条独立原始轨迹，渲染供人工核验。相机、问答或子片段副本不重复计数；记录来源版本、抽样范围、规则、seed、轨迹身份和异常，避免只挑正常样本。
- 原版字段核查及已有原始可视化保留在 `data/<数据集>/check/`，例如 LIBERO 的 `check/raw/video/`。用户后续明确：经过我们预处理后的核验视频放在 `data/<数据集>/processed/video/`（video 单数），例如 `data/libero/guidedvla/processed/video/`；对应说明、样本／标注版本清单与人工核验表放在 `processed/`。先生成可视化，再由人工检查；目录约定本身不代表已运行补标或通过验收。原始大文件继续放 `/data1/yuankai/robot-research/`。
- 渲染保留原时间含义、视角与帧号；展示已有事件、技能、bbox／mask 和可解释的状态。缺标、推断臂、未确认坐标及终点时刻明确标注，不把插值或模型生成结果冒充原始标注。
- 每轮开始先记录具体来源范围、10 条样本的抽法、下载／计算预算、视频布局与输出路径。用户已授权本流程的字段检查、抽样和原始数据可视化；按该范围例行执行，不逐步重复询问。新增模型补标、策略诊断或训练实验仍遵循工作区实验规则。
- 自动检查与人工验收分开：视频可解码和数值有限只属于技术检查；对象、事件、并发、hold 和数据是否适合由人工核验，未完成时保持待核验，不自行填写通过。
- 后续所有候选沿用此流程。既有调研不自动算作完成 10 条可视化；没有选定的数据集不写成已采用。
