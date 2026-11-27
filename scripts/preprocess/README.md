# 离线数据处理

当前 Grounding 使用 `libero/grounding_batch.py` 补标、`libero/interaction_locks.py` 刷新标记，`grounding_check.py`／`grounding_coverage.py`／`grounding_review.py` 核验数据与增强，命令见 [LIBERO](../../examples/libero/README.md)。

来源基础导出入口是 `scripts/prepare_data.py`，`scripts/libero_render.py` 负责 LIBERO 保存结果的回读渲染。来源读取、实例绑定、原生区域生成和渲染的实现集中在这里；`openpi.policies.eskill_dataset` 只读取结果，不反向导入本目录。

原顶层 `preprocess/` 已并入 `scripts/preprocess/`；原 `python -m preprocess...` 命令对应 `python -m scripts.preprocess...`。原 `scripts/render_data.py` 已改名为 `scripts/libero_render.py`，参数保持一致。

| 模块 | 内容 |
| --- | --- |
| `fields.py` | 预保存字段说明与 `docs/data.md` 的 11 项需求自查清单 |
| `export.py` | Episode 保存：`EpisodeWriter.save` 写入数组和图像后自动计算 `event_index` 和 `effector_pixels`，`validate_episode` 校验结构及派生字段一致性 |
| `artifacts.py` | 来源清单分派、文件摘要与标注文件定位；由文本和视觉预处理共同使用 |
| `libero/` | 共用 RLDS 读取与渲染；`atomic_vla/`、`guidedvla/`、`libero_40/` 保存来源专用处理 |
| `robotwin/` | `episodes.py` 导出统一 Episode；`check.py` 核对来源与回读内容，`render.py` 渲染保存结果；保留历史标注／LeRobot 复查入口 |
| `agibot_world/` | 已有 PhysicalReasoning 抽样与核验脚本 |
| `visual_grounding/`、`text_annotation/` | Qwen3-VL＋SAM3 的模型补标路径；当前仿真框主线使用原生渲染 |

`visual_grounding/check_vae.py` 回读保存的 Episode，复用训练侧原位画布变换，检查冻结 FLUX VAE 的三角色重建。它只做编解码与核验渲染；用法与结果见 [AtomicVLA 十条轨迹检查](../../data/libero/atomic-vla/check/test/flux_vae_v1/README.md)。

代码目录使用可正常 import 的下划线，例如 `atomic_vla`；数据目录仍使用原来的 `data/libero/atomic-vla/`，已有文件与来源身份保持原路径。

## 交互目标与全场景区域核验

入口为 `python -m scripts.preprocess.libero.interaction_review`，配置见 [review_interaction_targets.json](../../examples/libero/review_interaction_targets.json)。三个阶段依次为 `--stage inventory`、`--stage extract`、`--stage render`；后两阶段可用 `--number 1` 单独检查选样。先核对配置中的源索引、旧场景匹配记录、LIBERO checkout 和数据盘输出位置。

- `atomic_vla/interaction_targets.py`：复用实例／部件词典生成 Event 交互目标；Place 解析目的地，歧义抽屉用该操作对象的任务目标约束消歧。规则不按轨迹编号猜答案。所有候选仍需人工核验。
- `atomic_vla/scene_regions.py`：恢复已匹配仿真状态，隔离每个实例／部件渲染完整屏内 mask 与 bbox；空间区域投影原生 box site。保存独立的 `scene_entities` 和 Event 引用，保留原字段。
- `libero/interaction_review.py`：全量清单、固定十条选样、回读标签渲染。左侧原图，右侧目标红框／其他实例绿框；输出逐帧视频和 Event 预览。

使用已有 `libero-native` 环境。运行 `inventory`／`extract` 前，`LIBERO_CONFIG_PATH/config.yaml` 应指向配置中 checkout 的 `bddl_files`、`assets`、`init_files`；设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`。本机 NVIDIA vendor 文件为 `/home/yuankai/robot-research/workspace/envs/libero-evaluation/nvidia_egl.json`，通过 `__EGL_VENDOR_LIBRARY_FILENAMES` 指定，不修改系统驱动。`render` 仅需 NumPy、Pillow、PyAV，不加载仿真。例：

```bash
PYTHONPATH=src:. /data1/yuankai/robot-research/workspace/envs/libero-native/bin/python \
  -m scripts.preprocess.libero.interaction_review \
  --config examples/libero/review_interaction_targets.json --stage render
```

十条初始核验结果通过相对引用读取原资产。全量入口 `libero/grounding_batch.py` 将原数组／图像以只读硬链接放入各 Episode，同时补齐完整区域、可见 mask 和在线增强元数据，训练所需引用均为目录内文件。`libero/grounding_check.py` 检查保存回读、训练／验证加载和可选小模型反传，生成全部任务的 Event 中点预览；用法见 [LIBERO 示例](../../examples/libero/README.md#current-grounding-数据与训练接口)。

主相机物品位置分布使用 `libero/object_distribution.py`，配置见 [audit_object_distribution.json](../../examples/libero/audit_object_distribution.json)。`--stage extract` 对全部轨迹等距抽五帧，以同一套原生隔离渲染提取所有有限实例的完整屏内 bbox；`--stage summarize` 回读逐任务 CSV，输出28类物品的中心分布、框覆盖热图和位置／大小统计。不包含部件或背景表面，五帧采样的范围不代表逐帧极值。运行环境与 EGL 配置沿用上面的区域核验入口。

`libero/atomic_vla/review_*`、`manual_binding12.py` 等保留了之前的研究批次入口及其内部依赖，供已有核验结果追溯。这些脚本没有接入训练加载器，也不代表通用正式入口。当前导出器直接调用来源读取函数，不再依赖固定 100 条的 `review_source.py`。

`libero/atomic_vla/batch.py` 负责本次全量逐帧区域生成、保存回读和验收索引，调用已有原生提取器与导出器；使用方法见 [LIBERO 示例](../../docs/history/libero_joint.md#数据准备与可视化)。实例决策复用已保存的合并标签，视频从落盘的 Episode 回读。

`libero/atomic_vla/calibrate.py` 从导出的 Episode 补齐主相机参数，输出自包含的训练 Episode。EEF 源像素坐标由 `EpisodeWriter.save` 从标定参数自动计算并存入 `effector_pixels`，不再保存固定窗口 Motion 数组。逐条检查场景模型、逐帧检查末端运动学；训练时无需仿真。见 [八点二维 Motion 用法](../../docs/history/libero_joint.md#帧采样与标签核验)。

完整帧标注仍须先生成，再用来源导出器保存；`prepare_data.py` 当前只导出已有标签，不自动启动全量仿真或模型补标。数据格式及时间、对象、有效性约定见 [数据说明](../../data/README.md)。旧命令的模块前缀对应如下：

| 原位置 | 当前位置 |
| --- | --- |
| `data/data.py` | `scripts/preprocess/fields.py` |
| `data/io.py` | 读取归 `src/openpi/policies/eskill_dataset.py`；保存与校验归 `scripts/preprocess/export.py` |
| `data/libero/atomic-vla/scripts/` | `scripts/preprocess/libero/atomic_vla/` |
| `data/libero/guidedvla/scripts/` | `scripts/preprocess/libero/guidedvla/` |
| `data/libero/libero-40/*.py` | `scripts/preprocess/libero/libero_40/` |
| `data/libero/scripts/`、`render.py`、`view.py` | `scripts/preprocess/libero/` |
| `data/robotwin/scripts/`、`data/agibot_world/scripts/` | `scripts/preprocess/robotwin/`、`scripts/preprocess/agibot_world/` |
| `data/visual_grounding/`、`data/text_annotation/` | `scripts/preprocess/` 下的同名目录；原视觉 `io.py` 已移为公共 `artifacts.py` |

历史 `check/` 中的命令、代码路径和 SHA256 保留原记录；对照上表使用新入口。

`libero/atomic_vla/gripper.py` 核对源动作及配对后补齐夹爪语义；`libero/audit_training.py` 统计 Event 帧长、技能／任务／划分和实际 Motion 有效性，输出 JSON、CSV 与 Markdown；`libero/review_motion.py` 保存并回读训练目标，复用 LIBERO 渲染器生成人工核验视频。用法见 [训练数据检查](../../data/libero/atomic-vla/check/test/training/README.md)。

`libero/review_mirror.py` 使用保存的 Episode 与训练侧空间变换，抽取十条覆盖六类技能的轨迹，将原图／镜像并排渲染。标签先保存为 NPZ 后回读，RGB／目标画布使用同一确定性变换；不需要模型、VAE 或 GPU。视频为人工核验，不能视为预测或模型效果。用法与产物见[镜像检查](../../data/libero/atomic-vla/check/test/training/mirror_videos.md)。

`libero/grounding_review.py` 回读单目标 `GroundingDataset`，展示 Copy-Paste、随机 Patch Masking、干净 VAE 目标及可选真实 VAE 重建；`--benchmark` 检查数据加载与小模型训练吞吐。原生完整区域由 `interaction_review.py` 的 `protect_robot=true` 提取，`--stage augment` 再补可见 mask 与轻量编辑元数据；训练读取不依赖仿真。操作入口见 [LIBERO 示例](../../examples/libero/README.md#current-grounding-数据与训练接口)。
