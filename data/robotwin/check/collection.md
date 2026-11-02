# RoboTwin clean 数据

**50 个任务各 50 条，共 2,500 条，采集已完成。** 原始批次 2,336 条，修复版本续采 164 条；合并时核对来源指纹、任务内 episode 编号唯一及任务配额，结果见 [summary.json](summary.json)。十条核验视频已生成，人工验收待完成。

## 采集约定

- 任务清单为 `RoboTwin/env_cfg/eval/all_tasks.yml`，本体 aloha-agilex，场景使用 `demo_clean`；保留任务原生的位置／模型抽样，不新增背景、杂物、灯光、相机或桌高随机化。
- 每任务从 seed 0 顺序尝试，保留通过原生 `plan_success && check_success()` 及采集结构检查的 50 条。规划失败、不稳定及异常分别登记；这是专家示范筛选，不是策略成功率评测。
- 保存三相机 RGB、主相机整数实例 ID／对象可见区域／夹爪可见与完整屏内区域、相机参数、实测手臂状态与驱动命令、世界系末端位姿、真实物理步／时间、指令、原语与按臂接触证据。
- 原语记录具体实例 ID 和实际起止，命令与实测状态分开；逐物理步控制记录关联已保存 RGB 行。保留终点快照和任务期间的 delay 采样，不将初始化等待并入示范。
- 运行协议最多四个 GPU worker、每任务最多 6 小时／500 个 seed，总预算 48 小时／400 GB。详细执行过程与临时验证已收入 [文档快照](../../../examples/archive/README.md#数据调研文档)。

## 来源与核验

| 内容 | 位置 |
| --- | --- |
| 主批次 | `/data1/yuankai/robot-research/data/robotwin/clean_native/` |
| 修复续采 | 主批次的 `supplement/`，保存独立 `collection.json` 与来源身份 |
| 原始 HDF5 | 各来源的 `<task>/aloha_agilex/raw/episode_XXXXXXX.hdf5` |
| 原语覆盖与修复依据 | [primitive_coverage.csv](primitive_coverage.csv) · [修复补丁归档](../../../examples/archive/README.md#robotwin-历史材料) |
| 合并来源版本、配额及结果 | [summary.json](summary.json) 中的 `source_versions` 与任务统计 |
| 十条固定样本与人工表 | [clean_manifest.json](../processed/clean_manifest.json) · [clean_manual_review.csv](../processed/clean_manual_review.csv) |
| 十条核验视频 | [processed/video/clean/](../processed/video/clean/) |

修复涉及无解抓取、双臂空动作列表和跨 episode 的旧实体引用，主源码已应用补丁；父批次文件与失败记录保留，补采只写入独立来源。

十条样本按 seed 42 从完整 2,500 条池子抽取，优先不同任务。视频展示三视角 RGB、主相机区域、左右臂源原语／阶段及真实仿真时间，完整解码帧数与源记录一致。15 fps 仅为标称核验播放速度，不表示源采样等间隔；展示的是源原语，正式 Event／技能、逐 Event 对象／goal 监督及统一训练导出仍待完成。

从 RoboTwin 根目录采集到新的输出位置：

```bash
python -m collection.collect --config env_cfg/collection_clean_native.yml --save-root /path/to/new/dataset
```

从 ESkill-VLA 根目录刷新已有结果汇总；加 `--render` 可复用固定十条清单渲染：

```bash
python -m scripts.preprocess.robotwin.check collection \
  --source-root /data1/yuankai/robot-research/data/robotwin/clean_native
```

## 堆三个碗的对象身份核对

只读核对 `stack_bowls_three/aloha_agilex/raw/episode_0000000.hdf5`（seed 1，443 源帧）。三个碗名称均为 `002_bowl`，场景实例映射分别为 bowl1→68、bowl2→69、bowl3→70；这些 ID 只对该轨迹有效。源 grasp／place 原语通过 `metadata.actor_entity_ids` 记录本次操作的具体实例，区域通过同一 ID 提取。

| 源 Place 操作 | 操作实例 | 放置参照（来自专家脚本） | 源 Place 结束行 | 同一对象的终点 bbox |
| --- | --- | --- | --- | --- |
| 放置第一只碗 | bowl1，ID 68 | 桌面指定位置 | 138 | [141, 121, 213, 187] |
| 第二只碗叠到第一只碗 | bowl2，ID 69 | bowl1 | 271 | [139, 109, 215, 179] |
| 第三只碗叠到第二只碗 | bowl3，ID 70 | bowl2 | 426 | [139, 98, 217, 172] |

三个源终点均有同刻 RGB 和有效可见区域。最终 goal 由本 Event 对象在本 Event 结束时的区域派生：第二只碗的 Place 取 bowl2 的终点区域，承托它的 bowl1 是放置参照；每段分别使用自己的结束时刻。同一只碗的 Pick 与 Place 绑定相同实例，但各取自己的终点。上述源行号仅用于核对，不冒充尚未转换的公共时间轴 EventData.end_frame；本次未新增正式 Event 标签。撤离来源中 last_tracked_target 的引用也不自动证明仍在持物，后续须按无对象／未知的实际语义处理有效性。
