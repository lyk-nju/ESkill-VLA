# Event 边界修正与技能冲突核验

2026-11-01，用户批准：同臂共享边界帧归后一个 Event，前一 Event 的终点相应前移；技能冲突先提供视频，由用户判断是否改为 Push。

**45 处共享边界已修正。** 全量 1,693 条／3,873 个 Event 共 273,465 帧，每帧恰好归属一个 Event，剩余重叠和缺口均为 0。本页原视频保存修改前的 Pick 标签。

**用户随后确认这组 33 个事件属于 Push**：压住物体，利用摩擦力平移。来源适配器和最新版派生标注已修正为 Push，原始标签保留。[新版事件、统计与视频](push_reviewed/README.md) · [用户确认记录](skill_review.json)。本目录旧 `events.json`、`boundary_report.json` 和原视频报告保留边界修正阶段的快照；当前使用 `push_reviewed/events.json`。

核验入口：[合并视频和逐条跳转](index.html) · [33 条技能冲突合并视频](../../archive/history_20261101/test/video/atomicvla_skill_conflicts_33.mp4) · [人工记录表](manual_review.csv)。视频源帧共 4,990 帧，10 fps 展示时长 8 分 19 秒；全部为完整单事件轨迹，没有只截取看起来像 Push 的片段。

边界检查：[45 处逐项变更](boundary_changes.csv) · [规范化事件](events.json) · [全量报告](boundary_report.json) · [014 修改前后对照](boundary_014_before_after.jpg)。例如 014：Pick 原 `[0,57)`／goal=56 改为 `[0,56)`／goal=55，Place 保持 `[56,172)`／goal=171。

规则在 [source.py](../../../scripts/source.py) 的 `source_intervals` 中执行，现有读取／补标入口默认生效；源起止、技能和 JSON 均保留，改动写入 `boundary_correction`。只处理相邻 Event 共用的一帧；多帧重叠会报错，不按这个规则直接截断。`resolve_shared_boundaries=False` 可读取旧的区间转换，用于历史比较。

终点区域和三维路径应随更新后的 `goal_frame` 查询；未来重新生成区域／视频时使用新批次目录。旧 v1／v2／官方状态检查的产物保留旧约定，不把旧视频或旧缓存标成已经更新。

范围：固定 AtomicVLA `33d2f98574a52838b9db34b334bd5cc6cbe57f50` 的全部 1,693 条／3,873 个事件，检查 45 处共享边界，并逐条展示全部 33 个 Pick／Push 文本冲突，不另抽样替代。RGB／状态／动作来自作者指定的 `openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`，每条另核对任务、帧数和分片哈希。

预算：补齐所需 Goal 分片，新增下载不超过 1.5 GiB；缓存不超过 2 GiB；最多 4 个工作进程，CPU 不超过一小时，无训练、模型或 GPU 渲染。沿用用户确认的共用渲染器，1920×1340，只显示原版主相机、Pick 标签、Push 描述、夹爪／动作和三维末端；不为此次技能判断新增 bbox。10 fps 是展示帧率，源帧全部保留。

本目录保存规范化事件、45 处变更、33 条来源清单及人工表；逐条视频放 `../video/skill_conflicts/`，合并视频放 `../video/atomicvla_skill_conflicts_33.mp4`。已有固定 100 条中的 035 对应标注 ID 647，046 对应 ID 584；本轮视频按标注 ID 升序排列，顺序号不是原 100 条编号。

来源清单见 [manifest.json](manifest.json)。本轮新增下载 1.184 GiB，共读取所需的 14 个 Goal 分片，33 条主相机流均独立；核对作者转换顺序的候选 ID、任务文本和精确帧数，未将此关联说成作者提供了独立的逐条原始哈希映射。视频展示的区域缺失不属于此次技能判定结论。

相关测试 **60 passed，6 subtests passed**，见 [tests.log](tests.log)，覆盖共享边界归属、终点同步更新、源标注保留和较宽重叠拒绝。视频的解码、源帧顺序、合并章节和人工状态见 [video_report.json](video_report.json)；技术检查不替代用户判断技能。

复跑入口为 [review_annotations.py](../../../scripts/review_annotations.py)，从项目根目录依次执行 `prepare`、`fetch`、`extract`、`render`、`merge`；`fetch` 用有 requests 的 visual-grounding 环境，其余用 libero-native 环境。大缓存留在 `/data1/yuankai/robot-research/data/external/atomic_vla/annotation_review/source/`。已有人工表不覆盖，视频输出目录需要为空。

```bash
FETCH_PY=/data1/yuankai/robot-research/workspace/envs/visual-grounding/bin/python
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export MPLCONFIGDIR=/tmp/eskill-review100-mpl
export OMP_NUM_THREADS=1
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations prepare
"$FETCH_PY" -m data.libero.atomic-vla.scripts.review_annotations fetch --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations extract
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations render --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations merge
```
