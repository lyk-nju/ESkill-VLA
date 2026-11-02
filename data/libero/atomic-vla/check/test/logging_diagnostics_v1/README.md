# W&B 分区与诊断检查

W&B 指标使用 `train`、`train/aux`、`val`、`val/aux`、`chart` 五个命名分区，以 `chart/step` 为横轴。实际输出示例见 [payload.json](payload.json)，核验概要见 [report.json](report.json)。这里只核验命名及日志链路，不代表模型效果或完整验证集成绩。

| 分区 | 内容 |
| --- | --- |
| `train`、`val` | loss、action_loss、grounding_loss、skill_loss、traj_loss |
| `train/aux`、`val/aux` | 三角色及前景／背景 Grounding MSE；当前／未来技能损失、准确率与各预测类别频率；Motion 总误差、各偏移误差、真实／补齐点统计及有效数量 |
| `chart` | learning_rate、grad_norm、time_per_step、step |

技能频率是有效标签位置的 argmax 类别频率，未来技能按动作步统计，均排除不存在的手臂。Motion `l2_px` 使用模型输入图像的像素坐标，正式配置边长 224；本次 tiny 检查边长 32。真实点为 `observed`，hold-last 补齐点为 `padded`，缺少来源标志时记入 `unclassified`。无有效标签时均值显示为零，结合对应 count 阅读。验证先累计整个子集的分子和有效数量，再求均值。

相关单元／回归检查 44 项通过，另有 1 项跳过；两进程 Gloo 全局计数、DDP 验证均值及真实 W&B SDK 离线训练检查另有 3 项通过。真实数据探针使用 `task_split_v1/smoke/auxiliary_full/checkpoint.pt` 的已有 tiny CPU 模型，读取一个训练 batch 和两个验证 batch，共生成 180 项标量；所有值有限、有效类别频率和为一、Motion 来源计数可合并还原总数、诊断张量不携带梯度。模型和 VAE 均为测试规模，不是预训练 π0.5。本次未启动正式训练或向线上项目上传数据。

示例中 `chart/grad_norm` 和 `chart/time_per_step` 置零，仅作为命名预览；它们的真实训练采集由训练循环及回归测试核验。历史训练产物保持原样。
