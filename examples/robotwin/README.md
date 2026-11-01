# RoboTwin 数据准备

在项目根目录使用已安装本项目的 RoboTwin／LeRobot 专用环境运行。采集仍由相邻 RoboTwin 仓库负责，ESkill-VLA 的离线处理集中在 `scripts/preprocess/robotwin/`。

```bash
python -m scripts.preprocess.robotwin.check --help
python -m scripts.preprocess.robotwin.annotate --help
python scripts/prepare_data.py robotwin --help
```

当前 `prepare_data.py robotwin` 调用已有 LeRobot 导出器；原有 state/action 语义与导出参数继续显式指定，尚未改成 AtomicVLA 的预保存格式。源数据、配置、核验视频和具体操作见 [RoboTwin 数据说明](../../data/robotwin/README.md)。
