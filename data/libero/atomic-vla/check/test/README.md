# AtomicVLA 数据核验索引

当前主线是 Current／Subject 与在线增强。核验说明随 Git 保存，视频、数组和完整日志位于数据盘；部分链接只在源服务器可用。每份报告保留当时配置，历史增强比例不代表当前配方。

## 当前 Grounding 链路

| 记录 | 用途 |
| --- | --- |
| [full_grounding](full_grounding/README.md) | 全量场景框／mask、输入与划分检查；内含后续修订记录 |
| [interaction_targets](interaction_targets/README.md) | 交互目标及全场景区域定义 |
| [task_grounding](task_grounding/README.md) | 完整任务直接 Grounding 的早期核验 |
| [online_grounding](online_grounding/README.md) | 在线编辑与真实 VAE 标签的历史检查 |
| [augmentation_coverage](augmentation_coverage/README.md) | 定位／反事实／Copy 三类候选与覆盖率 |
| [copy_placement](copy_placement/README.md) | 先计算可行落点再采样 |
| [copy_target_bias](copy_target_bias/README.md) | 原 Event 目标优先 80%、文本替换 20% |
| [copy_no_blacklist](copy_no_blacklist/README.md) | 取消容器黑名单、全量标记刷新与容器移动视频 |
| [augmentation_guards](augmentation_guards/README.md) | 锁定标记启动检查与当前 Region 父物体保护 |

真实模型启动／冻结与 checkpoint 检查在 [current_grounding_fused](../../../../../experiments/current_grounding_fused/README.md)。当前参数与流程以 [增强说明](../../../../../docs/augmentation.md) 为准。

## 来源依据与历史动作检查

| 记录 | 用途 |
| --- | --- |
| [full_frame_v1](full_frame_v1/README.md) | 原全量逐帧操作对象／夹爪区域，55 条覆盖全部任务的验收视频 |
| [all_rules_v1](all_rules_v1/README.md)／[manual_binding12](manual_binding12/README.md) | 批处理依赖、实例绑定及助手审核依据 |
| [annotation_review](annotation_review/README.md) | Event 边界修订与用户确认的 Push 技能 |
| [review100_v1](review100_v1/README.md) | 固定 100 条来源与规则回归 |
| [task_split_v1](task_split_v1/README.md) | 固定父轨迹划分 |
| [training](training/README.md) | 历史动作、夹爪、Motion 与镜像检查，操作命令不作为当前训练说明 |
| [原始样本](../raw/README.md) | 发布数据原始抽样 |

原有 740 条状态对齐警示和绑定审核状态继续保留；自动生成完成与人工质量验收分开记录。不删除来源证据来简化目录。
