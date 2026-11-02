# 官方 LIBERO-40：场景状态与对齐检查

2026-11-01：**官方四套任务的 HDF5 来源已补齐；固定 100 条全部找到唯一动作对应，但不代表 100 条物体框均已与原图对齐通过。** 本轮独立保存官方状态的区域、偏差和核验材料，既有 AtomicVLA v1／v2 产物保留。

入口：**[10 条视频＋100 组三帧对照](check/index.html)** · [字段自查](check/field_check.md) · [逐轨迹数值](check/alignment.csv) · [汇总报告](check/report.json)。人工状态均为 `pending`。

随后用户已批准修正 AtomicVLA 的单帧共享边界，最新规范化事件见 [annotation_review](../atomic-vla/check/test/annotation_review/README.md)。本目录既有 100 条产物保留生成时的边界约定；下面关于 014 重叠的记录属于该历史批次。后续生成器已增加事件适配代码的版本核对，新的边界不能直接混入旧区域缓存。

## 来源和范围

读取 [LIBERO 官方下载代码](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/utils/download_utils.py) 指定的 [yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a)，固定 revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。`clip-rt`、`dinyu` 只用于历史结果比较，不作为本轮状态输入。

| 官方套件 | HDF5 任务文件 | 本轮轨迹 | 唯一动作匹配 |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 10 | 25 | 25 |
| LIBERO-Object | 10 | 25 | 25 |
| LIBERO-Goal | 10 | 25 | 25 |
| LIBERO-10 | 10 | 25 | 25 |

沿用 [AtomicVLA 固定样本清单](../atomic-vla/check/test/review100_v1/manifest.json)：100 条、16,662 帧、238 个 Event。RGB／机器人状态／动作仍来自作者指定的 `openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`，事件来自 AtomicVLA `33d2f98574a52838b9db34b334bd5cc6cbe57f50`。官方 HDF5 本身没有这些 Event 标签。

本轮扫描每个任务的 50 个示范的动作；**91 条完整原动作直接相等，9 条按 OpenVLA no-op 规则过滤后完整相等**。保存 RLDS 行号到官方原始行号的映射，不按 demo 编号猜测，不接受近似动作或任意子序列。100 条状态尺寸、关节名称顺序及 XML 中显式的主相机参数均通过兼容检查；这不是整份仿真模型完全一致的证明。

40 个完整 HDF5 共 33,784,856,577 字节。本轮仅用 HTTP Range 摘取动作、匹配状态、模型 XML 和关键帧；成功任务记录的传输合计 2,763,779,385 字节（2.574 GiB），不含一次 SSL 读取失败的重试开销。没有下载全套文件，也没有声称验证完整远端文件的 SHA256；摘取缓存逐份记录 SHA256。缓存位于 `/data1/yuankai/robot-research/data/external/libero_40/f13aa24a3da8/`，约 50.1 MiB。

## 对齐结果与限制

011／044／067 均已恢复场景、生成物体框及完整视频；官方 `states` 和行映射与之前 `dinyu` 候选逐项相同。这三条此前尚未完成状态来源核对及管线接入，不能据此判断官方 LIBERO 缺失轨迹；本轮找到官方对应也不代表物体框已通过图像对齐验收。

| 编号 | 官方示范 | 末端最大偏差 | 三帧恢复图 MAE / 255 | 核验视频 |
| --- | --- | ---: | ---: | --- |
| 011 | `demo_37` | 6.62 mm | 2.48 | [视频](check/test/video/011_libero_10_no_noops_s007_r002.mp4) |
| 044 | `demo_26` | 7.40 mm | 6.76 | [视频](check/test/video/044_libero_goal_no_noops_s001_r020.mp4) |
| 067 | `demo_10` | 14.76 mm | 3.12 | [视频](check/test/video/067_libero_object_no_noops_s007_r009.mp4) |

末端偏差指“官方场景状态恢复的机器人末端”与“同一动作行的 RLDS 末端记录”的三维欧氏距离，**不是 bbox 误差，也不是已确认的固定时间偏移**。记录了前后各一行的诊断，实际未挪动任何源帧。夹爪区域仍直接使用 RLDS 关节和手指状态恢复，不使用存在偏差的官方机器人姿态。

全批末端最大偏差为 65.01 mm（036）；97 条可比较的历史 `clip-rt` 状态在各套件的末端误差中位数均更低。因此官方来源补齐身份和状态，并不意味着直接替换旧状态会更贴合当前原图。[OpenVLA 再生成代码](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py) 会设置初态、稳定场景、过滤 no-op 并重新执行动作、记录图像；这是两版状态可能不同的机制依据，本轮未进一步拆解每条轨迹偏差的具体成因。

100 条各抽首帧、中间帧、末帧。每组三帧平均 RGB MAE 的中位数为 4.65/255，范围 1.78–8.67/255；MAE 包含机器人、物体、纹理及背景差异，不能作为 bbox 正确率。助手抽看 011／044／067／036／088；044 中段框与奶油奶酪有可见偏移，仍需人工逐段判断。没有据这些数值设“自动通过”阈值。

## 区域与人工核验

沿用未修改的 v2 语义／运动绑定规则，得到 **237/238 个 Event 唯一候选，99/100 条全部事件具备区域**；237 个终点对象框可用。015 仍有一个歧义 Event；它的末段在官方状态下取得运动证据，仅记录结果，没有调规则或针对该轨迹加特例。其余绑定决策与原 v2 相同。014 的事件重叠、035／046 的源技能冲突保留；若再要求每帧事件唯一，则区域完整的轨迹为 98 条。这些都是覆盖率，不是人工正确率。

所有框取主相机的 `full_in_frame`：夹爪及每个对象／部件分别独立渲染，忽略外部遮挡，屏外裁掉，再由 mask 求 bbox。终点框查询同一 Event 的对象和适配后的 `goal_frame`，原始事件标签／端点保留在源清单中。

10 条视频在渲染前固定为 **011、021、022、044、053、058、067、085、088、092**：强制含三条缺场景样本，其余 seed=42 分层抽取且任务不重复；Spatial／Object／Goal／10 分别 3／2／2／3 条。015 不进入本轮人工重点。复用用户确认的 [render.py](../../../scripts/preprocess/libero/render.py)，1920×1340，共 1,866 帧，10 fps 仅为展示；视频逐帧解码和尺寸检查完成。蓝色夹爪、绿色当前对象、黄色同一对象的事件终点框。

对照图四列分别是 RLDS 原图、官方存储观测、官方状态恢复图、原图＋完整框。官方观测为 128×128，按来源约定旋转 180° 后放大到 256×256 便于观看；它与 256×256 的 RLDS 重放图不应混作同一图像版本。

自动核对 33,223 组有效 mask／bbox、区域与实例关联、终点查询及视频哈希；相关规则、来源和渲染测试 **58 passed，6 subtests passed**，见 [tests.log](check/tests.log)。自动检查和助手抽看均不替代用户验收。当前固定 100 条已参与规则开发，不能用来宣称未见任务泛化。

## 文件与复跑

| 文件 | 功能 |
| --- | --- |
| [source.py](../../../scripts/preprocess/libero/libero_40/source.py) | 固定官方文件清单、动作匹配、原始行映射及状态／XML 摘取；复用已有 Range 读取器。 |
| [align.py](../../../scripts/preprocess/libero/libero_40/align.py) | 状态与旧结果比较、原规则绑定、原生区域、三帧对照、10 条视频和汇总。 |
| [check/manifest.json](check/manifest.json) | 来源 revision、文件信息、固定 100 条身份和预先选定的 10 条视频。 |
| `check/matches/`、`check/test/alignments/` | 每任务读取记录和每轨迹状态／绑定证据。 |
| `check/test/regions/`、`comparisons/`、`video/` | 区域元数据、100 组对照、10 条完整核验视频；原始数组在数据盘。 |

从项目根目录执行；`prepare` 读取已保存的官方仓库元数据，并核验原 100 条清单。读取重试会复用已完成的任务。

```bash
FETCH_PY=/data1/yuankai/robot-research/workspace/envs/visual-grounding/bin/python
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export LIBERO_CONFIG_PATH=/data1/yuankai/robot-research/workspace/envs/libero-native/config
export NUMBA_CACHE_DIR=/tmp/eskill-review100-numba
export MPLCONFIGDIR=/tmp/eskill-review100-mpl
export OMP_NUM_THREADS=1

"$FETCH_PY" -m scripts.preprocess.libero.libero_40.source prepare
"$FETCH_PY" -m scripts.preprocess.libero.libero_40.source fetch --workers 4
"$NATIVE_PY" -m scripts.preprocess.libero.libero_40.align audit --workers 4

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$LIBERO_CONFIG_PATH/nvidia_egl.json"
"$NATIVE_PY" -m scripts.preprocess.libero.libero_40.align regions --workers 4
"$NATIVE_PY" -m scripts.preprocess.libero.libero_40.align compare
"$NATIVE_PY" -m scripts.preprocess.libero.libero_40.align render --workers 4
"$NATIVE_PY" -m scripts.preprocess.libero.libero_40.align report
```

任一阶段可用 `--sample 11` 查看单条；`report` 汇总已有记录，`render` 默认只渲染预选 10 条。绑定、状态适配代码有改动时应使用新的实验目录，避免覆盖本轮证据。

预先记录的预算：Range 新增传输上限 6 GiB、缓存上限 8 GiB，每阶段最多 4 个工作进程，CPU 与 GPU 阶段各不超过 1 小时，无训练和视觉模型。100 条状态检查累计工作进程计时约 226 秒，区域生成约 309 秒；本轮在预算内完成。
