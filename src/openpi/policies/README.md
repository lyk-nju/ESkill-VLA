# 数据与推理

`eskill_dataset.py` 读取 Episode、控制时钟、相机配对与事件标签。`grounding_dataset.py` 对同一观测生成阶段需要的空间监督。动作窗口使用记录中的 conditioning image index；同一观察可对应多个控制行。

`policy_config.create_trained_policy` 加载动作 checkpoint 与输出归一化。`current_grounding.create_current_grounding_policy` 用于视觉预测。具体可用的阶段与入口见 [分支设计](../../../docs/design.md)。
