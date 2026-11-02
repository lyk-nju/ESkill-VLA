# AtomicVLA 全量通用规则检查

**全量已完成：1,693 条／3,873 个 Event／273,465 帧。1,681 条（99.29%）的全部事件获得唯一实例候选；12 条仍有 34 个事件未完成绑定。** 候选覆盖率不代表人工正确率。

后续用户授权手工确定这 12 条，助手已另存 [34 个实例补充及全量合并标签](../manual_binding12/README.md)。本目录保留纯自动规则结果，不因人工补充将自动覆盖率改为 100%；740 条对齐警示仍保留。

入口：**[全部统计、三帧对照与 10 条完整视频](index.html)** · [绑定缺口 12 条](binding_issues.csv) · [对齐风险 740 条](alignment_risks.csv) · [完整事件表](events.csv) · [汇总 JSON](report.json)。表格使用作者从 0 开始的标注 ID，视频序号从 1 开始，均不同于此前固定 100 条的编号。

| 检查 | 结果 | 含义 |
| --- | ---: | --- |
| RLDS 任务／帧数／场景文件对应 | 1,693 / 1,693 | 96 个分片均核对完整 SHA256，主图／状态／动作指纹独立。 |
| 官方 HDF5 完整动作唯一匹配 | 1,693 / 1,693 | 1,488 条直接相等，205 条按既有 no-op 规则过滤后相等。 |
| 事件帧归属／技能文本一致 | 全部 | 45 处共享边界和 33 个用户确认的 Push 修正已应用，无重叠、缺口或剩余文本技能冲突。 |
| 静态实例唯一候选 | 3,757 / 3,873 Event | 其余 96 个多候选、20 个未找到一致候选。 |
| 加入逐帧接触／运动后的唯一候选 | 3,839 / 3,873 Event | 消除 82 个多候选；剩余 14 个歧义、20 个未解析。 |
| 全部 Event 有唯一候选的轨迹 | 1,681 / 1,693 | 其余 12 条都属于双摩卡壶任务。 |
| 关键帧原生区域检查 | 26,175 组 mask／bbox | 首／中／终点及全局三帧，3,839 个已绑定 Event 的终点框均非空；无渲染执行错误。 |
| 官方状态末端与 RLDS 相差超过 1 cm | 740 / 1,693 条 | 对齐风险，不能据此认定 740 条 bbox 都错误。 |

关节顺序及 XML 中显式相机参数的兼容检查全部通过，但没有据此证明整份模型、材质及状态／RGB 时间关系完全相同。3,147 个事件有唯一运动支持，726 个证据不足；规则允许已由类别／属性／空间关系唯一确定的对象继续保留候选，不把证据不足强制改成新实例。当前 12 条绑定缺口与 740 条对齐风险重叠 11 条，不能将两项直接相加；共 741 条至少触发一种检查标记，其余 952 条未触发这些标记，也仍需人工确认。

## 后续优化位置

| 类型 | 轨迹／Event 数 | 具体原因与后续方向 |
| --- | --- | --- |
| 序数称呼 | 5 条／20 Event | `first moka pot`、`second moka pot` 被当作物体属性，导致没有一致候选。可将序数解析为事件内的对象称呼，再用接触和持物上下文绑定；不能把 first 直接查表映射到实例 1。 |
| 双壶实例歧义 | 7 条／14 Event | 通称 `moka pot` 有两只候选，现有接触、相对位姿和 1 cm 状态质量条件不足以消歧。需补充证据、检查重放对齐，不用候选顺序强行填答案。 |
| 状态与原图对齐风险 | 740 条 | 最大末端差 276.91 mm，三帧平均 RGB MAE 的轨迹中位数为 4.58/255；数值包含机器人、材质等差异，不能当成对象框误差。下一步应核对原始状态与 RLDS 重放版本，结合对象区域对照判断。 |

序数称呼的标注 ID：**51、73、143、288、341**。双壶歧义的标注 ID：**23、46、67、70、204、283、306**。旧固定 100 条中的 015 对应标注 ID 46，仍有一个歧义 Event；014 的边界和 035／046 的 Push 修正已生效。

夹爪独立采用源 RLDS 关节恢复，关键帧中其末端与源状态的最大差为 0.702 mm；这与上述官方场景状态的最大差分开记录。对齐警示不是放松运动质量阈值的理由，本轮未调阈值或修改绑定规则。

## 验证范围和复跑

10 条完整视频已输出，共 2,023 帧，1920×1340；自动逐帧解码、帧数和时间连续性检查通过。另为全部 1,693 条保存三帧恢复图对照。每个 Event 的对象／夹爪区域只在指定关键帧全量核对，不能称为全部 273,465 帧的 bbox 已生成或验收。10 条视频才生成了整段逐帧区域。

固定 100 条回归：原 237 个唯一候选全部保留，对象／部件身份变化 0、丢失绑定 0；仅应用用户批准的 014 边界和 035／046 技能更新。相关检查 **63 passed，6 subtests passed**，见 [tests.log](tests.log)。用户确认过的 Push 类别保留其确认依据；本轮其余实例、框和整体数据适合性均为人工 pending。

| 文件 | 职责 |
| --- | --- |
| [review_all.py](../../../scripts/review_all.py) | 固定全量来源、逐分片核对、官方动作／状态匹配、原规则运动检查和汇总。 |
| [review_all_regions.py](../../../scripts/review_all_regions.py) | 按任务复用原生环境，检查完整屏内关键帧区域，并生成 10 条完整核验视频。 |
| [manifest.json](manifest.json) | 版本、规则哈希、全部轨迹身份和事先选定的 10 条人工样本。 |
| `audits/`、`keyframe_regions/`、`comparisons/` | 每条运动／绑定证据、区域检查记录和源图对照；大数组在数据盘。 |
| [trajectories.csv](trajectories.csv)、[input_issues.csv](input_issues.csv) | 全部逐轨迹状态；来源／执行问题表当前为 0 条。 |

本轮新增 RLDS 数据约 7.119 GiB；一次 SSL 中断未写入分片数据，重试后完成。官方 HDF5 新读取 2,583,424,313 字节（2.406 GiB），只摘取动作和匹配状态，不读取全套 RGB，也未声称校验完整远端 HDF5 的哈希。缓存根目录为 `/data1/yuankai/robot-research/data/external/atomic_vla/all_rules_v1/`，复用的旧样本路径和哈希另记在逐轨迹记录。

从项目根目录执行；全量下载文件清单为 [source_files.json](source_files.json)，用既有 `review_source.download` 下载固定分片。随后：

```bash
FETCH_PY=/data1/yuankai/robot-research/workspace/envs/visual-grounding/bin/python
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export LIBERO_CONFIG_PATH=/data1/yuankai/robot-research/workspace/envs/libero-native/config
export NUMBA_CACHE_DIR=/tmp/eskill-review100-numba
export MPLCONFIGDIR=/tmp/eskill-review100-mpl
export OMP_NUM_THREADS=1
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_all prepare
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_all extract --workers 4
"$FETCH_PY" -m data.libero.atomic-vla.scripts.review_all states --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_all audit --workers 4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$LIBERO_CONFIG_PATH/nvidia_egl.json"
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_all regions --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_all render --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_all report
```

规则哈希在阶段间保持一致；修改规则应使用新批次目录。本轮全部任务均已用于开发讨论，不能将同任务其余轨迹称为未见任务泛化测试。

---

以下为执行前记录的范围与预算。

2026-11-01，用户要求对全部数据执行当前通用规则，统计需要后续优化的范围。本轮固定 AtomicVLA `33d2f98574a52838b9db34b334bd5cc6cbe57f50` 的全部 1,693 条／3,873 个 Event，基础数据固定 `openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`。沿用用户批准的 45 处共享边界整理和 33 个 Push 技能修正；本轮只检查，不调规则、阈值或按任务增加特例。

检查分层：全量核对来源身份、完整动作对应、事件覆盖、技能／描述、实例／部件绑定和几何；匹配场景后逐帧读取运动与接触证据，记录场景和源末端偏差。主相机完整屏内 bbox 在每个 Event 的首／中／终点帧做原生渲染检查，范围和实际检查数量独立统计，不将关键帧检查说成全量逐帧 bbox 已验收。

人工视频：清单生成时按 seed=42 在四套任务中分层选择 10 条独立完整轨迹，包含自动发现的问题类型，再补足不同任务。静态问题类型目前都来自双摩卡壶任务，因此该任务选两条不同问题的轨迹，其余任务不重复。使用共用 `data/libero/render.py`，主相机、三维轨迹和源帧序保留，10 fps 仅为展示，视频放 `check/test/video/all_rules_v1/`。全量数值检查与这 10 条人工样本分开，人工默认 pending。

预算：新增 RLDS 分片约 7.12 GiB，上限 7.3 GiB；官方 HDF5 按 HTTP Range 读取动作和匹配状态，新传输上限 12 GiB（含失败重试），不全量下载其 RGB。新增缓存上限 15 GiB，放数据盘；项目内保存报告和核验视频。每阶段最多 4 个 CPU／GPU 工作进程，CPU／GPU 各不超过一小时，无训练或视觉模型调用。已有 100 条仅作为开发／回归子集；本轮不把同任务其他轨迹叫作未见任务泛化。
