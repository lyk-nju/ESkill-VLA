# 现有 RoboTwin 示范核对

2026-10-30。状态：只读字段初查完成；尚未渲染视频、补标或完成人工核验。

重构验证补充：新读取器已重读同一批十条样本，技术检查无错误；新清单补充 HDF5 全文件哈希，见 [raw/manifest.json](raw/manifest.json) 和 [raw/report.json](raw/report.json)。原始 `source_inspection.json` 保留不变；人工表见 [raw/manual_review.csv](raw/manual_review.csv)，仍全部待核验。本次仅单元测试生成过临时小视频，未生成正式十条核验视频。

来源：`/data1/yuankai/robot-research/data/robotwin/event_data/`。先统计活动任务目录的事件 JSON／HDF5 文件数，不读取 `_archive`；再读取 10 条轨迹的事件、状态／动作数值与 HDF5 字段，视频只读容器元数据。结果见 [source_inspection.json](source_inspection.json)，字段摘要见 [field_check.md](field_check.md)。

抽样规则：GuidedVLA 的 8 个任务，加 handover_block 和 place_object_stand；按脚本固定任务顺序，各从排序后的事件文件列表用 `random.Random(42)` 选 1 条。目的是覆盖普通操作、双臂协作、交接及已知终点问题，属于目的性抽样，不代表全部任务分布；这 10 个任务也不是已确定的训练集。

本次只使用 CPU、本地已有文件、无下载／仿真／模型运行。10 条状态／动作均为有限值，未发现 bbox／mask；视频元数据帧数与事件 frame_count 一致，均比 HDF5 RGB 行数多 1。尚未解码核对画面或证明帧级对齐。事件 JSON 保存 SHA-256；HDF5 保存路径、大小和修改时间，未做全文件哈希，不称作不可变快照。

旧检查结果及抽样规则保留不变。重构后可用新入口重读相同样本，生成独立清单（从 ESkill-VLA 根目录执行）：

```bash
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  data/robotwin/scripts/check.py sample \
  --from-report data/robotwin/check/source_inspection.json
# 然后 check.py validate --manifest data/robotwin/check/raw/manifest.json
```

下一步原始数据可视化沿用这 10 条身份，保持源采样时间；显示已有视角、源 primitive、左右臂信息，缺少的对象区域和未确认的 Event 明确标注。具体预算在正式渲染前补录；新入口视频输出 `check/raw/video/`，不生成新标签。人工核验重点为事件语义、同一对象、并发、hold 和终点对齐，当前全部待核验。
