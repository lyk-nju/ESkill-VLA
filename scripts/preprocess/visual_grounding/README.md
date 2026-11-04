# Qwen3-VL＋SAM3 视觉预处理

LIBERO 的视频绘制与编码现统一委托 [scripts/preprocess/libero/render.py](../libero/render.py)，来源适配使用 [view.py](../libero/view.py)。固定展示字段、MISSING 规则及独立重渲入口见 [渲染说明](../../../data/libero/README.md#共用人工核验渲染器)；RoboTwin 暂沿用原有布局。

2026-10-31 当前来源选择：LIBERO 使用 [AtomicVLA](../../../data/libero/atomic-vla/README.md)，优先补主相机夹爪／操作物体 bbox。用户后续指定测试产物放 `data/libero/atomic-vla/check/test/`，视频为 `check/test/video/`，入口见 [当前测试](../../../data/libero/atomic-vla/check/test/README.md)。下面的历史模型测试和 `--dataset libero` CLI 仍对应 GuidedVLA；AtomicVLA 使用来源专用入口和读取器。

当前只处理主相机里的各臂夹爪和操作物体：LIBERO 为 `image`，RoboTwin 为 `cam_head`；同一对象的候选终点也取主相机。默认配置已更新，腕部图像保留但暂不补区域。渲染分别为每条臂保留终点面板，不要求每臂一个相机。历史运行配置及双视角视频保持原样，本次范围调整未重新推理或渲染。

状态：2026-10-30 已完成 LIBERO 单条烟测及固定 10 条真实模型测试，最终视频见 [LIBERO processed](../../../data/libero/guidedvla/processed/README.md)。1,592 帧均完成渲染与技术核对，但有事件漏标、夹爪漏检和对象选错，所有人工验收保持 pending。本次没有运行 RoboTwin 模型补标。

2026-10-30 路线更新：用户决定 RoboTwin 直接使用原生 bbox／mask，后续不运行这里的视觉 locate／track；本模块主要服务 LIBERO 等缺少原生区域的来源。RoboTwin 的历史读取／CLI 接口保留兼容，不表示仍需执行外部视觉补标。文本 annotation 可继续复用 Qwen 加载模块。

最初文件级验证：新增 8 项预处理回归与既有 8 项 RoboTwin 回归共 16 项通过；离线 import 不加载模型或仿真依赖。读取器核验 LIBERO 10 条／1,592 帧／131 个技能连续段、RoboTwin 10 条／1,806 帧／54 个 primitive 区间。本次真实测试修复两处问题后，在 Python 3.12 环境再次通过 8 项预处理回归。模型试点及版本记录见 [实验记录](../../../../workspace/experiment/libero_visual_grounding/README.md)。

## 文件与依赖

| 文件 | 职责 |
| --- | --- |
| `annotate.py` | `locate`／`track` 两阶段入口、来源分派、提示和区域记录 |
| `qwen.py` | 本地 Qwen3-VL 的唯一加载与调用实现；文本标注也复用它 |
| `sam.py` | SAM3 指定实例的框提示与双向视频传播 |
| `render.py` | 原始 RGB、当前区域与候选终点画面的统一渲染及人工表 |
| [../artifacts.py](../artifacts.py) | 两条处理链和渲染共用的来源／标注文件助手 |
| `config.yaml` | 本地模型路径、设备、视角与关键帧数量 |

入口统一从 ESkill-VLA 根目录用 `python -m scripts.preprocess...` 执行，不依赖当前脚本目录或 import fallback。GuidedVLA 读取器在 `scripts/preprocess/libero/guidedvla/source.py`，RoboTwin 在 `scripts/preprocess/robotwin/source.py`，无 SAPIEN 导入。当前 `--dataset libero` 专指 GuidedVLA 发布版，默认输出 `data/libero/guidedvla/processed/`；LIBERO-90 尚未接入。检查／渲染需要 NumPy、Pillow、PyYAML、h5py、PyArrow、PyAV；RoboTwin 图像仍调用 XPolicyLab 的 RGB 解码器。Torch／Transformers／SAM3 只在实际调用模型时加载。

运行依赖见 [requirements.txt](requirements.txt)。本次独立环境为 `/data1/yuankai/robot-research/workspace/envs/visual-grounding`，Python 3.12.13／Torch 2.10.0+cu128／Transformers 4.57.6。SAM3 使用 NumPy 1.26，故固定 SciPy 1.15.3、PyArrow 24，另补其实际导入需要的 pycocotools，并保留 `setuptools<81`。完整安装版本、权重校验和实际配置保存在试点 `runs/run_001/metadata/`，仓库默认配置仍使用空路径，避免绑定本机绝对路径。

Qwen 当前实现使用 Transformers 的 `Qwen3VLForConditionalGeneration`，支持本地 dense Qwen3-VL 权重；不是 MoE 或量化模型的通用加载器。SAM 使用官方 `build_sam3_video_model(..., load_from_HF=False)` 的 tracker，模型为 SAM3，不自动替换为 SAM3.1。模型可从 ModelScope 下载后填本地路径，代码不自动联网下载权重或安装依赖。Qwen 与 SAM 分进程执行，不必同时驻留显存。现有 Python 3.11 的 Pi_05 环境仅用于本轮文件级测试；SAM3 官方环境要求 Python 3.12+，不能据这些测试声称现有环境已支持实际模型推理。

AtomicVLA 新测试增加 SAM3 原生文本候选区域，再由 Qwen 选实例；夹爪候选可能含整段机械臂，允许 Qwen 提出更紧的框但仍须人工核验。`SamDetector` 显式调用 `.to(device)`，因为本版本官方 image builder 只对字符串 `cuda` 自动迁移，不处理 `cuda:0`。该问题在三帧接口检查中发现并修复；跟踪器继续保留 GPU 状态。

SAM 接口依据：[官方实例视频示例](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_for_sam2_video_task_example.ipynb)。使用“框指定这个实例”的接口，不把概念示例框当作唯一实例。源图像上的 pixel xyxy 框转换为 0..1 坐标后传给 tracker；Qwen 输出约定为 0..1000，先显式转换到源像素。

## 执行顺序

以下是正式模型试点获准并配置环境／权重后的命令，不表示已经执行。先阅读 [文本标注入口](../text_annotation/README.md)，在独立的 `processed/text/` 中准备文本记录。LIBERO 例子：

```bash
python -m scripts.preprocess.text_annotation.annotate run --dataset libero \
  --manifest data/libero/guidedvla/check/raw/manifest.json
python -m scripts.preprocess.visual_grounding.annotate locate --dataset libero \
  --manifest data/libero/guidedvla/check/raw/manifest.json
python -m scripts.preprocess.visual_grounding.annotate track --dataset libero \
  --manifest data/libero/guidedvla/check/raw/manifest.json
python -m scripts.preprocess.visual_grounding.render --dataset libero \
  --manifest data/libero/guidedvla/check/raw/manifest.json
```

两个模型路径分别填入两份配置中的 Qwen 与 SAM 项；采用同一 Qwen checkpoint。也可用 `--config` 指向具体试点的配置快照。已有输出不覆盖，修改协议或重跑选新的 `--output` 并保留旧结果。RoboTwin 的兼容入口使用 `--dataset robotwin`，默认输出 `data/robotwin/processed/`；当前路线采用原生区域。

```text
processed/
├── manifest.json            # 来源清单、视觉配置与代码版本
├── text/<episode-key>.json  # 源区间、候选技能与物体关联、原模型回复
├── prompts/<episode-key>.json
├── regions/<episode-key>.json
├── masks/<episode-key>/     # 全分辨率二值 mask，packbits 压缩存储
├── video/                  # 一条源 episode 对应一个核验 MP4
├── report.json             # 视频帧数／帧率等技术结果
└── manual_review.csv       # 人工结论，初始 pending
```

这里只约定预处理产物，不重新建立全项目训练 schema。视频和 mask 不提交 Git。原始数据和 `check/raw/` 保持原位。本轮最终结果在 `processed/`，修复前 SAM 输出在 `processed/history/run_002/`，两批统计不混合。

## 区域与时间规则

- 每个夹爪一个 track，每个物理对象一个 track；对象跨 Event、跨臂交接时 ID 保持不变。不同相机分别定位／跟踪，物理身份关联由文本记录提出并由人工核验，不将每视角的 tracker ID 当物理真值。
- Qwen 从相关区间首／中／末等候选帧中选择一个可见的种子帧；夹爪默认使用整条轨迹首／中／末候选。无法确定实例时记 unlocalized。首版不做自动反复重试或失踪补框，纠错需要明确的新提示版本。
- SAM 在整条可用相机时间线上前后传播，保留原帧对应；不按每个 Event 重建对象身份。临时 JPEG 仅为官方 tracker 输入，RGB 顺序不变；数值 mask 保持原分辨率。临时帧在进程正常退出时删除。
- 视频帧可放 CPU，但 tracker 状态保留在 GPU。本次实测发现 CPU 状态异步卸载会在连续调用中混入前一次 mask；修复证据见实验 debug 记录。
- 文本回复未完整对齐源事件时保存 annotation_error 和原回复，语义字段保持空值；可继续处理夹爪，视频明确显示文本失败，不据此制造对象标签。
- 每帧保存 `target_id/camera/frame_index/bbox/mask/slot/status`。空 mask 为 empty，旧 RoboTwin 尾帧缺图为 missing_source_image；不插值成有效区域，内部缺图直接报告。
- 文本区间是 `[start_frame,stop_frame)`，终点图像单独由 `goal_frame` 指定，默认源区间最后一帧。它与旧 RoboTwin 导出器的 end_frame 约定不同，未经显式适配不能直接送进旧训练接口。
- 渲染按源 fps、完整 episode 输出。顶部原图，中部蓝色夹爪／绿色当前对象，下部黄色同对象候选终点。未核验阶段明确标为候选；同臂源区间重叠时不擅自选择一个 goal。
- 模型结果始终为 proposed/predicted，人工状态始终 pending，技术通过不等于标签正确。缺少 object 关联时仍能标夹爪，但不得声称对象监督齐全。

当前范围为双来源读取→候选文本→定位→分割／跟踪→核验视频；完整 Event 重整、自动提示纠错、与既有 π0.5 辅助监督导出的桥接留待标签规则验收后实现。
