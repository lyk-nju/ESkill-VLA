# 历史快照

## 迁移前代码

`data_code_20261101.tar.gz` 保存目录调整前的数据处理代码、规则、模型配置和回归测试，保留原相对路径及文件内容。既有 `data/**/check/` 报告中的代码 SHA256 对应当时的文件，不能用迁移后修改了 import 和路径的新文件替代核对。

这个快照不参与包安装或运行时 import。当前代码与命令见 [scripts/preprocess/README.md](../../scripts/preprocess/README.md)。如需查看某个原文件，可直接读取归档而无需恢复旧目录：

```bash
tar -xOf examples/archive/data_code_20261101.tar.gz data/io.py
```

## 数据调研文档

`data_notes_20261102.tar.gz` 保存三份已被实现取代的旧方案：RoboTwin `architecture_review.md`、`construction_plan.md`，以及 AtomicVLA `binding_v2_plan.md`。同时收拢 AtomicVLA 的十份 `navigation_before/` 重复副本，并保存本轮精简前的九份入口文档。

归档保留原项目相对路径及原始字节，`manifest.json` 列出每份文件的 SHA256 和归档类别。当前数据位置、接口与验收入口以 [data/README.md](../../data/README.md) 为准。

```bash
tar -xOf examples/archive/data_notes_20261102.tar.gz data/robotwin/construction_plan.md
tar -xOf examples/archive/data_notes_20261102.tar.gz manifest.json
```

## RoboTwin 历史材料

RoboTwin 的旧试采、历史字段检查、早期 Event／管线审计、采集修复补丁，连同整理前完整目录和相关汇总代码，保存在一个本机归档中：[robotwin_history_20261102.tar.gz](../../../workspace/reports/robotwin-data-cleanup-20261102/robotwin_history_20261102.tar.gz)。大体积历史视频不随 Git 发布，当前 clean 数据与十条验收仍在 [data/robotwin/](../../data/robotwin/README.md)。

归档按 `robot-research/` 根目录保留原相对路径和文件字节，内含 `manifest.json` 与 SHA256。查旧规则或报告时可只提取单个文件：

```bash
tar -xOf ../workspace/reports/robotwin-data-cleanup-20261102/robotwin_history_20261102.tar.gz \
  ESkill-VLA/data/robotwin/event_plan.md
```
