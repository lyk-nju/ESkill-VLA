# RoboTwin 数据准备

采集由相邻 RoboTwin 仓库的 `collection/` 负责；离线处理在 `scripts/preprocess/robotwin/`。以下命令从本项目根目录运行，使用现有 Python 数据处理环境，无需 SAPIEN 或 LeRobot。

## 统一 Episode 入口

`prepare_data.py robotwin` 调用 `episodes.py`，与 LIBERO 共用 EpisodeWriter／EpisodeDataset。每条保存 `episode.json`、`arrays.npz`、`images.npz`，读取与渲染不访问源 HDF5。固定十条已完成导出、逐像素回读核验和视频渲染，见 [核验说明](../../data/robotwin/check/test/episode_v1/README.md)。全量 2,500 条尚未转换。

```bash
export PYTHONPATH=src:.
python scripts/prepare_data.py robotwin \
  --manifest data/robotwin/processed/clean_manifest.json \
  --output /path/to/new/robotwin_episode_check --workers 2
python -m scripts.preprocess.robotwin.check episodes \
  --manifest /path/to/new/robotwin_episode_check/index.json \
  --output data/robotwin/check/test/episode_v1/readback.json
python -m scripts.preprocess.robotwin.render \
  --episode-index /path/to/new/robotwin_episode_check/index.json \
  --output data/robotwin/check/test/video
```

输出目录须新建；已有十条视频不自动覆盖。全量转换时将 `--manifest ...` 换为 `--source-root /data1/yuankai/robot-research/data/robotwin/clean_native`，自动包含 `supplement/`。

- 同一物理时刻保留最后一条观测，保存原始行号映射；时间戳保留真实间隔。
- 当前观测配下一次独立观测的 **14 维绝对关节目标**，逐条核对它等于该时刻前最后执行的控制目标；末帧只有观测。这是观测间隔上的目标点，不是 250 Hz 电机命令流，也不是固定 15 Hz 策略。
- 状态为每臂 6 个实测关节位置＋夹爪命令；动作每臂 6 个绝对关节目标＋夹爪命令。关节单位 rad，夹爪 0 关／1 开。
- 保存主相机、双腕及源额外前视图。源 RGB 为解码像素，使用无损 PNG 编码装入不压缩 NPZ，回读与源像素完全一致。
- 源原语和可见物体框只作证据；正式 Event／技能和物体完整框待补。当前只生成可靠的夹爪区域与末端二维位置，缺失辅助标签保持无效。
- Motion 的未来偏移以去重后的观测步计数，不能当作固定毫秒数。正式训练前仍需冻结控制执行协议、按任务划分 train／val 并计算 train-only 统计。

## 来源核验

当前 clean 50 任务 × 50 条已采齐。全量配额盘点、每任务一条数值检查：

```bash
python -m scripts.preprocess.robotwin.check audit \
  --source-root /data1/yuankai/robot-research/data/robotwin/clean_native
```

原始数据与 `supplement/` 保留独立来源；不从视频播放 fps 推算物理时间。新采集从 RoboTwin 仓库运行 `python -m collection.collect --config env_cfg/collection_clean_native.yml --save-root /path/to/new/dataset`。

## 尚待完成与历史入口

源原语有手臂与实例证据，但不直接等同于正式技能。每臂 Event、Hold／Idle、实例绑定和物体完整框需要继续整理；旧六任务候选规则不能直接覆盖全部 50 任务。源边界迁移时显式记录公共 `stop_frame/goal_frame`。

`annotate.py`、`export.py` 保留历史标注／LeRobot 结果的复查入口，后者不再是 `prepare_data.py robotwin` 的实现。`data/robotwin/config.yaml` 也仅用于历史 `event_data` 读取；新 clean 数据用显式源路径或固定清单。
