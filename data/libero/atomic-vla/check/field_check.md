# 字段自查

依据 [docs/data.md](../../../../docs/data.md)。当前已全量检查 1,693 条事件 JSON 和作者指定基础 RLDS 的 96 个分片／1,693 条轨迹，任务、帧数、场景文件及官方 HDF5 动作对应均已核对，见 [全量报告](test/all_rules_v1/README.md)。早期 4 分片／80 条记录保留在 [source_evidence.json](source_evidence.json)、[raw](raw/README.md)。实例语义和区域质量仍待人工核验。

2026-10-31：用户认可原版大体可用并选定为 LIBERO 主来源；后续优先补主相机夹爪与对象 bbox。下表保留已知接入事项，来源选择不等于缺项已补齐。

下表的“无”指原始发布内容。

| 我们需要什么 | 有没有 | 备注 |
| --- | --- | --- |
| RGB 图像 | 有 | 基础 RLDS 的 `steps/observation/image`，256×256；本轮只渲染主相机 |
| 任务指令 | 有 | `language_instruction`，完整轨迹的任务文本 |
| 机器人状态 | 有 | `state[8]`、`joint_state[7]`，包含夹爪状态；Panda 单臂 |
| 机器人动作与运动记录 | 部分 | `action[7]`、末端 `state[:3]`；绝对状态与动作增量的坐标／单位需随上游生成器再核对 |
| 每臂 Event 边界 | 部分 | 有 `start_frame/end_frame`；适配器已明确实际终点并按用户规则处理全量 45 处共享边界，当前无重叠／空缺，源边界保留 |
| 子事件描述与技能标签 | 有 | 原词表为 pick／place／open／close／turn；用户核验后已将 33 个事件修正为 Push，见 [记录](test/annotation_review/push_reviewed/README.md)；Hold 等仍需补充 |
| 执行器视觉区域 | 无 | 原版未发布；用户确认采用关节恢复＋原生相机渲染，全量结果见 [full_frame_v1](test/full_frame_v1/README.md) |
| 当前对象视觉区域 | 无 | 原版未发布；已全量做原生关键帧检查。纯自动的 12 条缺口已由助手另补 34 个实例及完整视频，见 [补充记录](test/manual_binding12/README.md)；位置对齐和用户验收另行保留 |
| Event 终点对象区域 | 无 | 明确实际终点图像索引并补齐同一对象 bbox 后，按事件末帧提取 |
| 时间与同步信息 | 部分 | RLDS 内图像／状态／动作逐步对应，JSON 按整数 episode ID 关联；本轮任务和帧数对应，缺少跨容器源哈希及显式时间戳／fps |
| 坐标与相机信息 | 部分 | 原 RLDS 未提供完整场景；已从官方 HDF5 补充状态／XML，核对关节顺序及显式相机参数；原图对齐另行核验 |
