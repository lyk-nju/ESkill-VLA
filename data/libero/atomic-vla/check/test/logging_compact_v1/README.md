# 精简 W&B 指标

原记录中 train／val 各有 88 项：5 项主损失、18 条技能类别频率、24 条逐偏移 Motion 误差、23 个有效数量及其他诊断。现在每侧固定选择最多 11 项，chart 保留 4 项，合计最多 26 项自定义标量。

| 分区 | 上传字段 |
| --- | --- |
| train／val | loss、action_loss、grounding_loss、skill_loss、traj_loss |
| train/aux／val/aux | grounding_ee、grounding_obj、grounding_goal、current_skill_accuracy、future_skill_accuracy、motion_l2_px |
| chart | learning_rate、grad_norm、time_per_step、step |

有显式计数且计数为零的辅助摘要不上传。完整诊断继续保存在本地 JSONL；模型、损失和全局有效数量归一化不受上传筛选影响。已上传的历史字段不会自动删除；新进程使用精简规则，旧面板可隐藏历史曲线。

26 项相关测试通过。另将历史真实数据诊断探针 `logging_diagnostics_v1/payload.json` 输入当前筛选，确认 train／val 各从 88 项降到 11 项、总计 26 项，预览见 [payload.json](payload.json)。这是日志筛选核验，没有启动训练或上传 W&B；历史探针使用 tiny CPU 模型，chart 中部分值仅为命名示例。
