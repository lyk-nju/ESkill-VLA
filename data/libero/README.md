# LIBERO 数据来源

2026-10-31：按来源管理各工作发布的 LIBERO 衍生数据，以及拟自建的 LIBERO-90 数据。

**当前选择：用户抽看视频后认可 AtomicVLA 大体可用，确定后续 LIBERO 主来源为 [AtomicVLA](atomic-vla/README.md) 的事件标注＋作者指定的基础 RGB／动作数据。** 当前优先验证原生相机渲染夹爪／操作物体区域，统一忽略外部遮挡、裁到屏幕内；见 [完整区域三帧试点](atomic-vla/check/archive/history_20261101/test/native_object_full/README.md)。事件终点对象区域从同一对象在实际终点帧的完整屏内 bbox 提取。现有事件与技能保留，既有 Qwen3-VL＋SAM3 结果保留参考，历史可见区域不会自动变成完整区域。

接入时核实 end_frame 的闭／右开含义及动作、坐标和时间约定；具体训练任务与预算后续确定。其余发布来源保留为参考，LIBERO-90 作为后续扩展备选。

| 目录 | 内容 | 状态 |
| --- | --- | --- |
| [guidedvla/](guidedvla/README.md) | GuidedVLA 发布的 40 任务数据、字段自查、读取脚本与既有核验结果 | 已完成 10 条原始及预处理视频，待人工验收 |
| [libero-40/](libero-40/README.md) | 官方 Spatial／Object／Goal／LIBERO-10 的 HDF5 场景状态，与当前 AtomicVLA／RLDS 的对应检查 | 固定 100 条全部找到唯一动作匹配；图像与区域另行核验，见来源内报告 |
| [libero-90/](libero-90/README.md) | 拟从 LIBERO-90 原始示范构建的自有数据链路 | 后续扩展备选；当前优先接入 AtomicVLA |
| LiLo-VLA（历史候选） | LIBERO-90 衍生原子技能数据 | 本工作区对应数据目录已移除 |
| [atomic-vla/](atomic-vla/README.md) | 已选定的 LIBERO 主来源：事件 JSON＋基础 RGB／动作数据 | [全量检查](atomic-vla/check/test/all_rules_v1/README.md)中自动绑定完整 1,681 条，其余 12 条已由助手 [补充实例](atomic-vla/check/test/manual_binding12/README.md)；区域对齐与用户验收另记 |
| CycleVLA（历史候选） | 子任务片段及 stop／progress | 本工作区对应数据目录已移除 |

代码集中在 [scripts/preprocess/libero/](../../scripts/preprocess/libero)，模型补标备选位于 [visual_grounding/](../../scripts/preprocess/visual_grounding/README.md) 与 [text_annotation/](../../scripts/preprocess/text_annotation/README.md)。各来源的字段核对及产物仍放在自己的数据目录；测试视频使用 `<来源>/check/test/video/`，正式结果使用 `<来源>/processed/video/`。

2026-10-31 已将原 `data/libero/` 下的 `check/`、`processed/`、`scripts/` 和专项说明整体移入 `guidedvla/`。历史 JSON、CSV、视频和 mask 内容保持原样；旧 manifest、日志及统计中的 `data/libero/check/`、`data/libero/processed/` 路径分别对应当前 `data/libero/guidedvla/check/`、`data/libero/guidedvla/processed/`。原始大文件路径不变。

当前共享 CLI 的 `--dataset libero` 仍专指 GuidedVLA 发布版，默认输出为 `data/libero/guidedvla/processed/`；LIBERO-90 的读取和处理入口待接入。

后三个来源的原版核验入口为 [scripts/check_raw.py](../../scripts/preprocess/libero/check_raw.py)，共用轻量 TFRecord 读取器 [rlds.py](../../scripts/preprocess/libero/rlds.py)。使用现有 Pi_05 Python 3.11 环境，依赖 NumPy、Pillow、PyAV、protobuf、google-crc32c；不加载 TensorFlow、仿真或模型。命令示例：`python -m scripts.preprocess.libero.check_raw prepare --source atomic-vla`，随后 `render`、`verify`；已有固定 manifest 或人工表不覆盖。数据盘下载清单为各来源 `raw_root/candidate_shards.json`，本轮下载脚本仅做公开分片传输，后续按 manifest 中路径及官方 SHA256 可复取。

## 共用人工核验渲染器

按用户提供的参考布局统一展示任务及当前技能、主相机、当前 Event、对象、终点帧、夹爪状态、动作及终点图。补充三维轨迹后画布为 1920×1340：下半部左侧为三维末端轨迹，右侧保留 X／Y／Z 时间曲线，底部为整段进度。只显示主相机，有字段就展示，缺失显示 **MISSING**，框是否存在与人工是否验收分开显示。

| 文件 | 职责 |
| --- | --- |
| [render.py](../../scripts/preprocess/libero/render.py) | `Episode/Event/Region` 展示输入、统一布局和视频编码；不读取具体来源、不推断标签 |
| [view.py](../../scripts/preprocess/libero/view.py) | 把原版／补标记录适配为展示输入，保持来源帧及端点语义 |
| [scripts/render_review.py](../../scripts/preprocess/libero/render_review.py) | GuidedVLA／AtomicVLA／LiLo-VLA／CycleVLA 的共用入口；可选择加载已有区域 |

2026-11-01 新增 `view.from_saved()`：读取 [公共预保存格式](../README.md) 的 `episode.json`／NPZ 及原 RGB 引用，再使用本渲染器。AtomicVLA 接入见 [保存接口试点](atomic-vla/check/archive/history_20261101/test/saved_format/README.md)；不在可视化时补框、推断实例或重算目标帧。

### 展示输入

这是渲染接口，不替代 [fields.py](../../scripts/preprocess/fields.py) 中的训练数据接口。数组按展示的源图像行对应；时间映射、源边界转换与对象关联由适配层提供。

| 输入 | 格式／含义 | 缺失时 |
| --- | --- | --- |
| `key / frame_count` | episode 身份、需要展示的完整帧数；必填 | 无法定义视频 |
| `rgb(frame)` | 主相机 RGB uint8 `[H,W,3]`，不裁剪、不翻转；可返回 None | 图像面板 MISSING |
| `task` | 完整任务指令 | MISSING |
| `events` | Event 列表；`id/start_frame/stop_frame/skill`，当前区间为 `[start,stop)` | 当前事件／技能 MISSING；同帧冲突也不擅自选一个 |
| `event.goal_frame` | 已明确的实际终点图像行，不默认 stop−1、不自动截断越界索引 | 终点图及终点区域 MISSING |
| `event.object_name` | 当前操作物体名称／已有实例标识 | MISSING，不从整句任务猜对象 |
| `gripper_state / actions` | `[N,D]`，动作维度按输入显示，原数值保留 | MISSING；NaN 按项显示 MISSING |
| `eef_xyz / xyz_note` | `[N,3]` 同时绘制三维轨迹与三条时间曲线，附来源坐标／单位说明；不跨缺失值连线 | 三维图缺完整 XYZ 时 MISSING，各时间曲线分别判断可用性 |
| `gripper_region(frame)` | 当前夹爪 `Region` | 夹爪区域 MISSING |
| `event.object_region(frame)` | 该 Event 同一物体的区域，能查询当前及 goal 帧；不在切换时改查下一 Event 对象 | 当前／终点对象区域独立显示 MISSING |
| `Region.bbox / mask` | 源 RGB 像素 xyxy（上界不含）／同分辨率 bool mask，可只给其中一种 | bbox 与 mask 独立；有 mask 无 bbox 时明确提示 |
| `source_fps / playback_fps` | 已知源频率／展示帧率；RLDS 当前 10 fps 仅为展示 | 不推定物理时间 |
| `annotation_label / review_status` | 来源标签或模型候选；默认人工 pending | 不根据有框自动判正确 |

配色统一为蓝色夹爪、绿色当前对象、黄色同一对象在实际终点的区域。渲染器仅画现有标注，不运行模型、补框或重新切分事件。GuidedVLA 原始相关性 patch mask 尚未确认实例身份，不能直接充当同一操作物体的 current／goal 区域。

三维轨迹使用同一 `eef_xyz`：灰色为整段源路径，绿色为当前 Event 路径，白点为当前位置，黄圈为实际事件终点。固定视角、等比例三轴，坐标数值保持原样；这是独立坐标图，不投影到相机画面。缺失点断线，不插值；真实静止轨迹仍有效。Matplotlib Agg 缓存坐标轴及整段轨迹，再绘制动态高亮。

```bash
# 使用已有 AtomicVLA 测试框，输出新布局视频；已有模型结果直接读取。
python -m scripts.preprocess.libero.render_review --source atomic-vla \
  --annotations data/libero/atomic-vla/check/test --tag 3d

# 原版字段：省略 annotations；缺框、缺对象名称的区域显示 MISSING。
python -m scripts.preprocess.libero.render_review --source atomic-vla \
  --sample 1 --preview-only --output /tmp/atomic-raw-preview
```

默认输出为来源的 `check/test/`。`--tag 3d` 使用 `video/*_review_3d.mp4`、`render_3d_preview.png`、`render_3d_report.json` 和 `render_3d_index.html`，与先前仅有时间曲线的布局分开保存。无 tag 时沿用 `*_review.mp4`、`render_preview.png`、`render_report.json`、`render_index.html`。历史原版视频和补标测试、模型标注、manifest 与人工表保持原样；各布局按自己的 report 核验，已有结果不覆盖。绘图依赖 Matplotlib（本次验证 3.10.3），其他依赖沿用 NumPy、Pillow、PyAV。
