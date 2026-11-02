# RoboTwin 数据准备

采集由相邻 RoboTwin 仓库的 `collection/` 负责；本项目离线处理在 `scripts/preprocess/robotwin/`。在项目根目录、已有 RoboTwin／LeRobot 专用环境中运行。数据位置与验收入口见 [data/robotwin/](../../data/robotwin/README.md)。

## 当前 clean 批次

刷新已有 2,500 条的来源与配额汇总；加 `--render` 可复用固定十条样本并渲染：

```bash
python -m scripts.preprocess.robotwin.check collection \
  --source-root /data1/yuankai/robot-research/data/robotwin/clean_native
```

汇总保存到 `data/robotwin/check/summary.json`，原语覆盖表在同目录。原十条清单、人工表和视频继续使用 `data/robotwin/processed/`，不重抽或自动填写人工通过。

新采集仍从 RoboTwin 根目录运行，输出到独立位置：

```bash
python -m collection.collect --config env_cfg/collection_clean_native.yml --save-root /path/to/new/dataset
```

raw v2 保留完整源时间线与终点；`joint_action` 为驱动目标，`joint_state` 的手臂部分为实测 qpos、夹爪为命令。真实时间来自物理步记录，不能用播放 fps 换算。离线模块不导入 SAPIEN，执行导出时才加载 LeRobot／Torch。

## Event 与训练导出

整理以源原语为骨架，轻量映射、相邻合并和有证据的 Hold／Idle 补段，保留双臂并发和源记录回链；六任务候选预览尚未冻结为全量语义标签。早期细则和试点材料可从 [历史归档](../archive/README.md#robotwin-历史材料)查回。

`annotate.py --manifest ...` 只装配源证据；提供 `--labels` 才载入显式标签，不默认补齐技能或区域。来源专用标签目前用 `[start_frame,end_frame)`，终点取 `end_frame`；公共保存接口分别记录 `stop_frame/goal_frame`，迁移时必须显式转换。

```bash
python -m scripts.preprocess.robotwin.annotate --manifest CHECK_MANIFEST \
  --labels LABEL_DIR --output NEW_ANNOTATION_DIR
python -m scripts.preprocess.robotwin.check render --manifest CHECK_MANIFEST \
  --annotations NEW_ANNOTATION_DIR --output data/robotwin/check/annotated
python -m scripts.preprocess.robotwin.export --manifest TRAIN_MANIFEST \
  --annotations NEW_ANNOTATION_DIR --output NEW_EXPORT_DIR \
  --repo-id local/eskill_robotwin --state-source stored --action-source stored
```

`TRAIN_MANIFEST` 显式指定 episode 的 train／val／test。历史格式使用 `stored/stored`，不重复移位；raw v2 必须选 `--action-source next-command`，状态选 `command` 或 `measured-arm-command-gripper`。最终训练动作契约仍待确定。

当前 `prepare_data.py robotwin` 调用既有 LeRobot 导出器，保存基础训练数据、辅助记录及源帧映射，RGB 引用原始 HDF5；尚未改成 AtomicVLA 的公共 JSON＋NPZ。夹爪 `full_in_frame` 与当前对象 `visible` 的定义及各项有效性须分别保留，不能将可见框直接改名为完整框。

## 历史读取配置

`data/robotwin/config.yaml` 默认指向旧 `event_data`／`xpolicylab-v1`，不是当前 clean 批次。`check sample/fields/validate/render` 使用各自显式清单；旧十条清单与报告已归档。命令参数可用以下入口查看：

```bash
python -m scripts.preprocess.robotwin.check --help
python -m scripts.preprocess.robotwin.annotate --help
python scripts/prepare_data.py robotwin --help
```
