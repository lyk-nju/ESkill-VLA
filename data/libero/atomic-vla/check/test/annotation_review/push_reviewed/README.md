# 用户确认的 Push 技能修正

2026-11-01，用户观看 33 条冲突视频后确认：“我看了确实是push技能，他是压住碗然后用摩擦力往外移动”。本轮将这组 33 个事件的派生技能由 Pick 修正为 Push，原始技能仍保留为 source_skill；结论范围是这组技能类别，不扩展为 bbox／场景对齐验收。

入口：[新版合并视频与逐条跳转](index.html) · [最新事件](events.json) · [33 处技能修改](skill_changes.csv) · [用户确认依据](../skill_review.json)。视频仍为 4,990 帧、8 分 19 秒，主相机、动作、状态和帧顺序与用户所看的版本一致。

规则位于 [source.py](../../../../scripts/source.py)：当前来源中，原技能为 Pick、该 Event 自身描述以 Push 开头时，派生技能使用 Push，并写入 `skill_correction`。任务整体包含 push 不会改掉另一个 Pick Event；规则不依赖任务名、轨迹号或对象实例编号。夹爪下压属于这组 Push 的接触方式，不改变物体类别或源文本。`apply_reviewed_skills=False` 可读取原技能，用于历史比较。

全量派生技能统计：**Pick 1,801／Place 1,800／Open 79／Close 69／Turn 91／Push 33**，共 3,873 个 Event。本轮恰好修改用户确认的 33 个事件；共享边界修正仍为 45 处，技能修正没有再移动边界。原 100 条的 035／046 分别对应标注 ID 647／584，在新输出中均为 Push。

源 JSON 的 SHA256 仍为 `2d97c7d6b4d7ae9cc43ea0bca73e13d5ed6c8126836b4c92f5444b2379e03d11`，原始 `primary_action_verb=pick` 和旧视频保留。父目录的人工表已记录本组 Push 结论；不把此记录解释为用户对每条 bbox 或其他技能类别都逐项验收。

沿用上一轮的 33 条固定来源及 4,990 个源帧，不重新抽样、不下载、不调用模型，不生成新的 bbox。使用已确认的共用主相机渲染器更新标签，最多 4 个 CPU 工作进程、一小时内；原视频保留，新版视频放 `check/test/video/push_reviewed/`，合并为 `atomicvla_push_reviewed_33.mp4`。

相关测试 **63 passed，6 subtests passed**，见 [tests.log](tests.log)。增加了源 Pick 保留、Push 修正规则、任务上下文中的 push 不误改 Pick、Push 不建立 Pick 持物继承的检查。

从项目根目录用 `libero-native` 环境执行：

```bash
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export MPLCONFIGDIR=/tmp/eskill-review100-mpl
export OMP_NUM_THREADS=1
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations approve
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations render --reviewed --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.review_annotations merge --reviewed
```

复跑渲染须使用新的空视频目录，避免覆盖本轮证据；`approve` 记录的是用户在本次对话已给出的确认，不是自动模型验收。
