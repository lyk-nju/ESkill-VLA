# 离线数据处理

正式使用入口在 `scripts/prepare_data.py` 与 `scripts/libero_render.py`，后者负责 LIBERO 保存结果的回读渲染。来源读取、实例绑定、原生区域生成和渲染的实现集中在这里；`openpi.policies.eskill_dataset` 只读取结果，不反向导入本目录。

原顶层 `preprocess/` 已并入 `scripts/preprocess/`；原 `python -m preprocess...` 命令对应 `python -m scripts.preprocess...`。原 `scripts/render_data.py` 已改名为 `scripts/libero_render.py`，参数保持一致。

| 模块 | 内容 |
| --- | --- |
| `fields.py` | 预保存字段说明与 `docs/data.md` 的 11 项需求自查清单 |
| `export.py` | Episode 保存：`EpisodeWriter.save` 写入数组和图像后自动计算 `event_index` 和 `effector_pixels`，`validate_episode` 校验结构及派生字段一致性 |
| `artifacts.py` | 来源清单分派、文件摘要与标注文件定位；由文本和视觉预处理共同使用 |
| `libero/` | 共用 RLDS 读取与渲染；`atomic_vla/`、`guidedvla/`、`libero_40/` 保存来源专用处理 |
| `robotwin/` | 原始示范读取、事件整理、现有 LeRobot 导出与核验 |
| `agibot_world/` | 已有 PhysicalReasoning 抽样与核验脚本 |
| `visual_grounding/`、`text_annotation/` | Qwen3-VL＋SAM3 的模型补标路径；当前仿真框主线使用原生渲染 |

`visual_grounding/check_vae.py` 回读保存的 Episode，复用训练侧原位画布变换，检查冻结 FLUX VAE 的三角色重建。它只做编解码与核验渲染；用法与结果见 [AtomicVLA 十条轨迹检查](../../data/libero/atomic-vla/check/test/flux_vae_v1/README.md)。

代码目录使用可正常 import 的下划线，例如 `atomic_vla`；数据目录仍使用原来的 `data/libero/atomic-vla/`，已有文件与来源身份保持原路径。

`libero/atomic_vla/review_*`、`manual_binding12.py` 等保留了之前的研究批次入口及其内部依赖，供已有核验结果追溯。这些脚本没有接入训练加载器，也不代表通用正式入口。当前导出器直接调用来源读取函数，不再依赖固定 100 条的 `review_source.py`。

`libero/atomic_vla/batch.py` 负责本次全量逐帧区域生成、保存回读和验收索引，调用已有原生提取器与导出器；使用方法见 [LIBERO 示例](../../examples/libero/README.md#全量逐帧区域与验收)。实例决策复用已保存的合并标签，视频从落盘的 Episode 回读。

`libero/atomic_vla/calibrate.py` 从导出的 Episode 补齐主相机参数，输出自包含的训练 Episode。EEF 源像素坐标由 `EpisodeWriter.save` 从标定参数自动计算并存入 `effector_pixels`，不再保存固定窗口 Motion 数组。逐条检查场景模型、逐帧检查末端运动学；训练时无需仿真。见 [八点二维 Motion 用法](../../examples/libero/README.md#八点二维-motion)。

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
