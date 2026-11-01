"""ESkill-VLA 预保存数据接口与预处理需求（2026-11-01，v2 草案）。

给数据接入 agent 的阅读顺序：
    1. EpisodeData：一条示范最终需要承载什么。
    2. EventData / RegionSequence：事件怎样关联同一对象及其终点区域。
    3. REQUIREMENTS：对照源数据，判断读取、转换、恢复或补标的缺口。

语义依据是 docs/data.md 和 docs/design.md。本文件定义来源无关的预处理
交付接口。落盘采用 episode.json + arrays.npz，读取见
src/openpi/policies/eskill_dataset.py，
保存与校验见 scripts/preprocess/export.py；RGB 可以
引用原始缓存。不固定训练容器、技能词表、H、M、模型 padding 或归一化。
TypedDict 是字段说明与静态类型提示，不是运行期数据或标注质量验证器。

共同约定：
* frame_index 指向 Episode 的公共索引轴；0 <= frame_index < frame_count。
  有时钟时，公共时间轴是各数据流对齐后的全部采样时刻与 Event 边界时刻的
  有序并集，相同时刻共用一个索引；它不是某路相机的帧轴，也不是等频网格。
  每路图像、状态、动作、末端、区域都有自己的 frame_indices，行数可以不同。
  索引表示本行的实际采样时刻，不表示最近图像；不舍弃动作、不默认重采样，
  也不通过时间取整合并不同采样时刻。源时钟转换及原始行对应需保留依据。
  保留终点观察，即使该时刻没有动作；缺帧不复制邻帧，不重新编号掩盖缺口。
  只有源步序对应时可用 source_steps，timestamps_s=None；不伪造秒数或 fps。
* 实际存在的手臂由 arm_ids 指定，源臂名通过 source_arm_mapping 映射。
  静止／hold 臂仍然存在；缺标签只屏蔽对应监督，不改变在线臂槽位。
* None 表示缺失／尚不明确，不能当作数值零、idle、没有物体或标签正确。
  valid 数组描述本项数据的可用性，不等于人工验收通过。未知语义要继续核实。
* Event 当前采样范围为 [start_frame, stop_frame)，实际终点另存 goal_frame。
  stop_frame 可等于 frame_count；goal_frame 必须是实际记录索引或 None。
  goal 可以是区间末条记录，也可以是 stop 时刻的额外终点观察；由来源适配器
  明确指定。读取器不默认 stop-1、不夹断越界、不借用下一 Event 的对象。
* 每个 Event 的 object_regions 始终对应该 Event 的同一个物理对象，覆盖
  当前区间及可用终点。无需额外必填全局 object_id；源 track ID 可保留在引用中。
  边界切换对象时，goal 读取旧 Event 的区域，不能读取新 Event 的当前对象。

训练时派生（不是额外原始标注字段）：
    Atomic Skill: Event.skill -> skill_labels[2] + skill_validity[2]。
    Future Skill: 每条动作记录的公共索引 -> 各臂所属 Event.skill
        -> future_skill_labels[2,H] + future_skill_validity[2,H]。
        与 actions[H,D] 逐行对齐，padding、缺标签和不存在臂排除监督。
        不新增原始标注或保存字段；在线以预测序列查技能表、逐步条件化动作。
    Target Grounding: 当前 RGB + ee / obj 区域；终点 RGB + 同一 Event 的 obj 区域
        -> 保留原图布局的三角色画布 -> 冻结 VAE latent 与区域权重。
        每臂每角色单独判断有效性；目标区域不要求参照物联合框。
        goal 可用性仅由 Event 边界有效、结束时刻 RGB 可用、同一对象在该图像
        中所用区域表示可用派生，不要求独立的事件成功或期望结果标签。
    Latent Motion: 本臂当前位置至 goal_frame 的真实 XYZ
        -> 弧长采样 -> 减去当前位置 -> trace[2,M,3] + trace_validity[2,M]。
        双臂共用固定参考系；只归一化采样进度，保留位移物理尺度。
        Goal 与 Motion 仍只覆盖当前 Event，不按未来技能窗口扩展终点。
    Action Expert: 同步动作序列 -> actions[H,D] + 时间／通道有效性。
        H 按动作序列行计数，不是公共时间轴的 H 个索引；采样时距须保留，
        如训练需要固定频率，应在另行明确的派生协议中处理。
        姿态、夹爪等继续由真实动作监督，粗路径不替代动作。
    VAE latent、未来区域、事件真值与有效性只服务监督，不进入在线预测器。

运行 ``python -m scripts.preprocess.fields`` 可打印覆盖 docs/data.md 的三列字段自查模板。
模块仅使用标准库，导入或运行不会读取数据、加载模型或启动预处理。
"""

from dataclasses import dataclass
from typing import Dict, Literal, NotRequired, Optional, Tuple, TypedDict


Arm = Literal["left", "right", "single"]


class DataRef(TypedDict):
    """定位已有文件／数组；大数据继续保留在数据盘。

    例：uri='/data1/.../episode.hdf5', key='state/left_ee_poses'。
    key 也可指 Parquet 列、NPZ 数组或 JSON 路径；整个文件使用 None。
    shape / dtype 描述解码后的逻辑数组。相对 uri 相对 episode.json 所在目录。
    当前读写器支持 NPZ 普通数组、逐行 packbits mask、JPEG bytes + offsets。
    后两者通过 encoding 声明；旧来源中的其他容器须由来源适配器先读取。
    """

    uri: str
    key: Optional[str]
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[str]
    encoding: NotRequired[Literal["array", "packbits", "jpeg_sequence"]]
    offsets_key: NotRequired[str]  # jpeg_sequence 在同一 NPZ 中的 [N+1] 字节偏移。


class SourceIdentity(TypedDict):
    """原始轨迹身份用于去重与 split；视角／裁剪副本共享父轨迹身份。

    parent_trajectory_id 查不到时保留 None，不能将视角条目 ID 冒充已去重身份。
    revision 与 checksums 保存已核实的版本信息，未知项不编造。
    """

    dataset: str
    revision: Optional[str]
    episode_id: str
    parent_trajectory_id: Optional[str]
    artifacts: Tuple[DataRef, ...]
    checksums: Dict[str, str]  # URI -> 已核实的 sha256；没有记录则留空。


class AnnotationEvidence(TypedDict):
    """来源标签与新增标注可追溯；自动技术检查不能写成人工 accepted。"""

    method: Literal["source", "rule", "native", "model", "assistant", "human", "unknown"]
    description: str  # 原生实例区域、规则版本、模型／提示词版本或人工记录。
    references: Tuple[str, ...]
    review_status: Literal["pending", "accepted", "rejected"]


class Channel(TypedDict):
    """通道顺序即数值最后一维顺序；unit=None 表示仍需核实。"""

    name: str
    arm: Optional[Arm]  # 机体等非臂通道使用 None。
    unit: Optional[str]


class NumericSequence(TypedDict):
    """N 行数值各自映射到公共时间轴；各序列 N 可不同。

    frame_indices 非递减；同一时刻若有多条源记录，保留各行及源顺序，允许
    重复索引，不去重丢数据。valid 必须与 values 同形状。
    各来源原始时间如与公共时间轴不同，适配器须核实对齐并保留转换证据。
    """

    values: DataRef  # [N,D]，保留通道的物理数值，尚未模型归一化。
    frame_indices: DataRef  # [N] int，本序列采样时刻的公共索引；不是图像行号。
    valid: DataRef  # [N,D] bool；无效填充值不能成为监督。


class RobotState(NumericSequence):
    """原生状态允许包含关节、夹爪、末端等，具体通道由来源说明。"""

    channels: Tuple[Channel, ...]
    semantics: Optional[str]  # 区分实测状态、控制命令、混合字段及夹爪含义。


class RobotAction(NumericSequence):
    """保留本体动作格式；相同维数不代表相同控制语义。

    frame_indices 指向动作记录自身的采样／命令时刻；alignment 说明源时间含义、
    动作目标时刻或执行区间，以及条件图像的配对规则，例如来源原有配对，或
    显式选择最近的历史图像。已经配对的 state/action 不再移动一帧。

    conditioning_image_indices[camera] 每行指向该相机 CameraStream.rgb 的行号
    （不是公共时间轴索引），长度与动作行数相同；允许重复，-1 表示缺失或尚未
    确定。字典缺少某相机也表示该相机尚无配对，不默认按行号或最近时间配对。
    作为在线条件的图像不得晚于该动作对应的决策时刻，且需检查相机 valid。

    例：相机时刻 [0,1/30,2/30]，动作时刻 [0,.01,.02,.03,.04,.05,.06]。
    公共轴为 [0,.01,.02,.03,1/30,.04,.05,.06,2/30]；相机索引 [0,4,8]，
    动作索引 [0,1,2,3,5,6,7]。若明确采用最近历史图像作为条件，其图像行号为
    [0,0,0,0,1,1,1]，七条动作全部保留；此例的配对规则不是接口默认策略。
    """

    channels: Tuple[Channel, ...]
    representation: Optional[str]  # joint_position / joint_delta / eef_delta / ...
    coordinate_frame: Optional[str]  # 末端动作需声明 base / world / tool；关节可为 None。
    rotation_convention: Optional[str]  # Euler 顺序、quat 顺序、增量组合等；不猜测。
    gripper_convention: Optional[str]  # 开闭方向、范围、位置／速度／二值命令。
    alignment: Optional[str]
    conditioning_image_indices: Dict[str, DataRef]  # 相机名 -> [N] int，图像行号或 -1。
    step_duration_s: Optional[DataRef]  # [N]：已核实的每步时距，未知时 None。


class EndEffectorPosition(NumericSequence):
    """values 为 [N,3] XYZ，valid 为 [N,3]；有真实静止记录时零路径有效。

    位置来自测量、记录或 FK，不能直接将动作增量累加当成已验证轨迹。
    进入路径监督前须落实米制、双臂共同固定参考系和一致的末端参考点。
    None 语义可以在待处理记录中保留，但不能据此构造有效 Latent Motion 目标。
    """

    unit: Optional[str]
    coordinate_frame: Optional[str]
    reference_point: Optional[str]  # TCP、夹爪中心等；偏移转换须有依据。
    evidence: AnnotationEvidence


class CameraCalibration(TypedDict):
    """只有投影、相机运动补偿或跨视角对齐需要时才要求。

    标定应与该 CameraStream 的图像坐标及帧映射对应；静态矩阵允许不带 N 维。
    """

    intrinsics: Optional[DataRef]  # [3,3] 或 [N,3,3]。
    extrinsics: Optional[DataRef]  # [4,4] 或 [N,4,4]。
    extrinsics_convention: Optional[str]  # 例如 camera_to_world_gl，必须核实方向与轴。
    world_frame: Optional[str]


class CameraStream(TypedDict):
    """读取后为 RGB uint8 [N,height,width,3]；腕部／外部视角按来源角色对应。"""

    rgb: DataRef
    frame_indices: DataRef  # [N]，每行图像采样时刻的公共索引，保留源行对应。
    valid: DataRef  # [N] bool，图像是否实际可用。
    height: int
    width: int
    role: str  # external / left_wrist / right_wrist / ...，不固定相机数量。
    calibration: Optional[CameraCalibration]


class RegionSequence(TypedDict):
    """一个视觉主体在一个相机中的区域序列；相机由外层字典 key 指定。

    boxes_xyxy 为源 RGB 像素坐标 [x1,y1,x2,y2]，上界 exclusive，屏内裁剪。
    masks 解码为 [N,height,width] bool；bbox 与 mask 分别由 boxes_valid 和
    masks_valid 标识逐行可用性。某种表示为 None 时，其有效性数组也必须为
    None；提供该表示时必须同时提供其 [N] bool 数组。消费者检查所用表示的
    有效性，不能因为 mask 可用就读取无效 bbox，反之亦然。
    两种表示共用 frame_indices 与 region_extent；若区域含义不同须分别记录。
    resize / crop 后要同步转换图像和区域；固定留边在构造画布时统一处理。
    项目当前监督采用 full_in_frame：夹爪／对象独立渲染，忽略外部遮挡并裁到
    屏内，当前与 Event 终点采用同一定义。原 RGB 仍保留真实遮挡内容。
    region_extent 区分可见区域与忽略遮挡的屏内完整区域，不能混为一种标签；
    历史 visible 来源需如实保留，不能改名充当已补齐的完整区域。
    完整区域在完全遮挡时仍可能有效；可见性与标注可用性分别记录。
    """

    frame_indices: DataRef  # [N]，严格递增；可以包含 Event.goal_frame。
    boxes_xyxy: Optional[DataRef]  # [N,4]；bbox 与 mask 允许只提供其中之一。
    boxes_valid: Optional[DataRef]  # [N] bool；不存在的框不是全图框或零框真值。
    masks: Optional[DataRef]  # 文件／压缩 mask 引用，解码形状见上。
    masks_valid: Optional[DataRef]  # [N] bool；与 bbox 独立，缺失不等于空 mask。
    visibility: Optional[DataRef]  # [N]：1 可见，0 不可见，-1 未知。
    region_extent: Literal["visible", "full_in_frame", "unknown"]
    evidence: AnnotationEvidence


class EventData(TypedDict):
    """本臂一次语义 Skill；同臂区间不重叠，跨臂允许并发。

    0 <= start_frame < stop_frame <= episode.frame_count。
    goal_frame 若已知，start_frame <= goal_frame <= stop_frame 且 < frame_count。
    当前采样用 [start_frame,stop_frame)，goal 图像与末端路径终点取 goal_frame。
    源 primitive 是候选证据，转换成语义 Event 的合并与边界规则需要单独记录。

    object_regions[camera] 只跟踪本 Event 对应的同一对象，含可用终点区域。
    多 Event 可引用同一原始 track；此处不要求另做全局对象 ID 标注。
    object_present=False 表示确实无对象；None 表示未知，不将二者混用。
    终点监督按 boundary_valid、goal_frame 对应的 RGB 及同一对象所用区域表示
    的可用性派生，不额外标注 Event 是否成功。
    """

    event_id: str
    arm: Arm
    start_frame: int
    stop_frame: int
    goal_frame: Optional[int]
    boundary_valid: bool
    description: Optional[str]
    skill: Optional[str]  # 映射后的共享类别；具体词表在数据配置确定。
    skill_valid: bool
    object_present: Optional[bool]
    object_regions: Dict[str, RegionSequence]
    cooperative_group_id: Optional[str]
    evidence: Dict[str, AnnotationEvidence]  # 如 boundary / skill / object_binding。
    source_annotation: Dict[str, object]  # 原边界／技能、实例及部件、修订依据；不作模型输入。


class CooperativeGoalGroup(TypedDict):
    """只在共同对象、共同效果被核实时填写；普通并发／handover 不强行成组。

    成员 Event.goal_frame 均等于 shared_goal_frame；每臂仍取自己的真实末端轨迹。
    """

    group_id: str
    member_event_ids: Tuple[str, ...]
    shared_goal_frame: int
    evidence: AnnotationEvidence


class EpisodeData(TypedDict):
    """来源读取／预处理交付的逻辑示范；数值和视频通过 DataRef 引用。

    在检查阶段允许缺项：None / 空集合表示待补齐，不能因此宣称训练就绪。
    arm_ids 仅列物理存在的臂；单臂保留 single，训练时再显式映射到模型槽位。
    timestamps_s 保留所有流的采样时刻并集及 Event 边界时刻；不能以相机行数
    限制动作数，也不能以动作行数裁掉最后一帧。frame_count 是公共时间点数。
    原生格式、视觉／文本侧车文件仍可分别保存，由来源适配器投影到这个接口。
    """

    schema_version: Literal["eskill.episode.v2-draft"]
    episode_id: str
    source: SourceIdentity
    instructions: Tuple[str, ...]  # 完整任务指令；空集合表示缺失，不能偷换成当前子任务。
    embodiment: str
    arm_ids: Tuple[Arm, ...]
    source_arm_mapping: Dict[str, Arm]
    frame_count: int
    timeline: Literal["timestamps", "source_steps"]
    timestamps_s: Optional[DataRef]  # [T] 严格递增秒数；源未发布时 None。
    alignment: str  # 公共索引如何建立；不能把展示 fps 当成数据采样率。
    nominal_fps: Optional[float]  # 仅同频来源保留标称频率；异频填 None，不用于对齐。
    cameras: Dict[str, CameraStream]
    state: Optional[RobotState]
    actions: Optional[RobotAction]
    effector_positions: Dict[Arm, EndEffectorPosition]
    effector_regions: Dict[Arm, Dict[str, RegionSequence]]
    events: Tuple[EventData, ...]
    cooperative_goal_groups: Tuple[CooperativeGoalGroup, ...]
    quality: Dict[str, object]  # warnings、技术测量与 manual_review 分开；不等于训练就绪。


@dataclass(frozen=True)
class Requirement:
    """对应 docs/data.md 一行；预处理建议不表示已完成、已授权运行或已验收。"""

    key: str
    title: str
    fields: Tuple[str, ...]
    modules: Tuple[str, ...]
    preprocessing: str
    checks: str
    derived: bool = False


# agent 应逐项填写有／部分／无／待核实，并回链来源字段与检查证据。
# 原生数据完整时直接映射，不强制跑模型补标。下列清单不自动选定数据集或工具版本。
REQUIREMENTS = (
    Requirement(
        "rgb", "RGB 图像", ("cameras", "timestamps_s"), ("VLM", "Target Grounding"),
        "解码／索引当前、历史和终点 RGB，建立相机与源帧映射；记录缺帧。",
        "RGB 通道、分辨率、视角、时间及末帧真实可用性；不能以邻帧替代终点。",
    ),
    Requirement(
        "instruction", "任务指令", ("instructions",), ("VLM",),
        "读取／补齐完整 Episode 指令，保留多条等价描述的来源。",
        "不是逐阶段离线标签；标签描述不能自动作为部署输入。",
    ),
    Requirement(
        "state", "机器人状态", ("state", "arm_ids", "source_arm_mapping"),
        ("Latent Motion", "Action Expert"),
        "整理各臂关节、夹爪及所需本体状态的通道、单位与图像对齐。",
        "实测状态与命令分清；静止臂、缺失臂、缺失状态分别处理。",
    ),
    Requirement(
        "action_motion", "机器人动作与运动记录", ("actions", "effector_positions"),
        ("Latent Motion", "Action Expert"),
        "明确动作类型／时距／夹爪定义；读取末端 XYZ，缺失时评估经本体模型 FK 恢复。",
        "动作和下一状态是否已配对；坐标、单位、TCP；不盲目积分增量或重复移动一帧。",
    ),
    Requirement(
        "event_boundary", "每臂 Event 边界", ("events.*.arm", "events.*.start_frame", "events.*.stop_frame", "events.*.goal_frame"),
        ("Atomic Skill", "Target Grounding", "Latent Motion"),
        "从脚本／阶段／状态或人工／模型候选整理语义 Event，核实归臂、合并和区间端点。",
        "同臂无重叠，跨臂可并发；未知间隙不补 idle；保留原边界证据。",
    ),
    Requirement(
        "skill", "子事件描述与技能标签", ("events.*.description", "events.*.skill", "events.*.skill_valid"),
        ("Atomic Skill",),
        "按交互目的整理子事件描述并映射共享词表；未确定类别保持无效。",
        "技能粒度与边界一致；闭合不等于抓住，静止不等于 idle，左右臂共用词表。",
    ),
    Requirement(
        "effector_region", "执行器视觉区域", ("effector_regions",), ("Target Grounding: ee",),
        "原生区域可用则直接映射；缺失时评估检测／分割／跟踪，并统一像素坐标。",
        "执行器与手臂对应；区分可见区域和屏内完整区域；bbox／mask 各自记录有效性。",
    ),
    Requirement(
        "object_region", "当前对象视觉区域", ("events.*.object_regions", "events.*.object_present"),
        ("Target Grounding: obj",),
        "将逐帧 bbox／mask 绑定到具体 Event 的同一对象；缺失时评估原生提取或检测跟踪。",
        "无对象与未知分开；身份不随换臂、遮挡或下一 Event 的对象切换而错误改变。",
    ),
    Requirement(
        "goal_region", "Event 终点对象区域（非必要独立字段）",
        ("events.*.boundary_valid", "events.*.goal_frame", "events.*.object_regions", "cameras"),
        ("Target Grounding: goal",),
        "读取当前 Event 对象在 goal_frame 的区域及 RGB；保留原图布局，不额外标一套 goal。",
        "由边界、终点 RGB 及同一对象所用区域表示的可用性派生；缺项仅屏蔽对应角色，不借用新对象。",
        derived=True,
    ),
    Requirement(
        "time", "时间与同步信息",
        ("timeline", "timestamps_s", "nominal_fps", "*.frame_indices", "actions.alignment", "actions.conditioning_image_indices"),
        ("所有模块",),
        "建立所有流采样时刻与 Event 边界的公共并集轴，保存各流映射及动作条件图像行号。",
        "动作保留自身时间及全部行，可多对一关联图像；不默认重采样，不去重丢动作；时钟及配对规则须明确。",
    ),
    Requirement(
        "geometry", "坐标与相机信息", ("effector_positions", "cameras.*.calibration"),
        ("Target Grounding", "Latent Motion"),
        "核对位置单位／参考系／末端参考点；需要投影或对齐时读取标定并验证变换方向。",
        "双臂位置处于共同固定参考系；旋转／偏移有依据；相机标定不是所有来源的必填项。",
    ),
)


def requirements_markdown() -> str:
    """生成字段自查空表；默认待核实，不自动判断某个数据集已有／缺失。"""
    lines = ["| 我们需要什么 | 有没有 | 备注 |", "| --- | --- | --- |"]
    for item in REQUIREMENTS:
        fields = "、".join("`{}`".format(path) for path in item.fields)
        note = "字段：{}；预处理：{} 核对：{}".format(fields, item.preprocessing, item.checks)
        lines.append("| {} | 待核实 | {} |".format(item.title, note.replace("|", "\\|")))
    return "\n".join(lines)


if __name__ == "__main__":
    print(requirements_markdown())
