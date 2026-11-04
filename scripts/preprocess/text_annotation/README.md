# 子事件文本标注

2026-10-31：LIBERO 主来源已选 [AtomicVLA](../../../data/libero/atomic-vla/README.md)，优先沿用它的事件区间和 `primary_action_verb`，接入时整理边界约定及项目技能映射；不再默认从 GuidedVLA 的 skill 连续段重新生成 LIBERO 事件。下方现有 CLI 及测试结果仍属于 GuidedVLA，AtomicVLA 适配待完成。

2026-11-01 用户已批准 AtomicVLA 单帧共享边界归后一个 Event。[来源适配器](../libero/atomic_vla/source.py) 已统一 `[start,stop)` 及 `goal=stop-1`，保留原始边界和修改记录。用户随后观看视频并确认 33 个冲突事件是通过下压和摩擦力平移物体的 Push，来源适配器已修正派生技能，原始 Pick 保留。最新规范化事件见 [push_reviewed](../../../data/libero/atomic-vla/check/test/annotation_review/push_reviewed/README.md)；本步骤不调用 Qwen，也不等于通用模型 CLI 已接入 AtomicVLA。

AtomicVLA 已另建来源专用的 [实例绑定规则](../libero/atomic_vla/bind_instances.py)：源 Event → BDDL 实例／部件 → 原生几何。这一步保留原事件与技能，不生成新事件，不调用 VLM；通用 CLI 与正式训练数据导出尚未因此完成接入。

`annotate.py` 复用已有源分段，整理候选技能、子事件描述和物理对象关联。不是自动事件发现，也不会把未核验 primitive 强制改成 Pick／Place。2026-10-30 已完成 LIBERO 固定 10 条真实测试：4 条回复完整对齐，6 条事件有缺漏；完整不代表语义正确，所有人工验收保持 pending。结果见 [processed](../../../data/libero/guidedvla/processed/README.md)。使用实验目录的本地配置，仓库默认模型路径仍留空。

RoboTwin 后续使用原生区域，不通过外部视觉模型提取 bbox；文本标注仍需完成 Event／技能与原生实体的每臂关联。新 `native_regions` 试采格式尚未接入此处来源适配；本模块当前也未完成 Hold／Idle／Return 区间重整、共享协作终态与正式训练标签桥接，不能认为运行候选文本就完成全部 annotation。

后续用户确认：RoboTwin 以已有原语区间为 Event 骨架，采用轻量映射、相邻合并与有证据的 Hold／Idle 补段，默认保留边界；具体规则集中在 RoboTwin 来源 annotation 层，见 [RoboTwin 用法](../../../examples/robotwin/README.md#event-与训练导出)。本文件下面保持 source_id 的 Qwen 候选标注流程仍有既有用途，但尚未实现 RoboTwin 这层合并／补段，不是原语整理的必需前置步骤。

- LIBERO：从逐帧 skill_id 提取连续区间，保留源类别；相邻同类阶段不可恢复的问题仍存在。
- 当前 `--dataset libero` 只读取 GuidedVLA 发布版，默认输出 `data/libero/guidedvla/processed/text/`；LIBERO-90 的原始示范读取和事件整理待实现。
- RoboTwin：保留 primitive 的 arm 和源起止，将闭区间明确转为 `[start_frame,stop_frame)`；goal_frame 保留原末帧。同臂重叠记录为待复核，不自动合并。
- `prepare` 仅整理源证据，skill／object 保持 null；可供人工填写。不把作者阶段词表自动映射到我们技能。
- `run` 用 Qwen3-VL 查看任务、完整源区间列表和有限关键帧，提出技能、描述和对象表。物理对象可跨 Event／臂复用 ID；框和 mask 由 visual_grounding 构造。
- 结构不合格的事件回复保留 annotation_error／原回复，语义字段留空，并继续下一条；不重试生成更好的标签。源 other 被模型提议为 Idle 的现象仍需人工核验。

```bash
# 只整理源证据到指定草稿目录；不运行模型。
python -m scripts.preprocess.text_annotation.annotate prepare --dataset libero \
  --manifest data/libero/guidedvla/check/raw/manifest.json --output /tmp/eskill-text-draft

# 配置权重且模型补标试点获准后再执行：
python -m scripts.preprocess.text_annotation.annotate run --dataset libero \
  --manifest data/libero/guidedvla/check/raw/manifest.json
```

配置中的技能表目前是候选集合，需与 Event 粒度一起核验，不是已冻结的训练分类表。模型不能改源区间或臂身份，不能漏掉／添加 source_id；不确定输出 null。人工调整边界时显式修改源关联记录并保存新版本，不能悄悄覆盖作者标签。

输出位于 `data/<来源>/processed/text/<episode-key>.json`，包括对象表、原区间及候选标签、模型原始回复和实际输入帧号、配置／代码依据。所有人工结论为 pending。`prepare` 与 `run` 使用不同输出目录，避免覆盖已整理的草稿；视觉主链默认消费 `processed/text/`。

Qwen 加载只在 [visual_grounding/qwen.py](../visual_grounding/qwen.py) 实现一次；本入口复用公共 [artifacts.py](../artifacts.py) 来源／标注文件助手，不反向导入视觉处理入口，模型和 SAM 均不在 import 阶段启动。训练读取器仍需在正式 Event 粒度确认后接入这些标签。
