# 数据与策略接口

运行时接口集中在 `policies/`，组 batch 在 `training/`，模型在 `models/`，离线来源处理在 `scripts/preprocess/`。Dataset 不依赖仿真或标注模型。

| 文件 | 职责 |
| --- | --- |
| `eskill_dataset.py` | 公共 Episode 数组／RGB 读取；历史动作目标与时间查询 |
| `grounding_dataset.py` | 当前专项：Event 目标、Subject 标签、在线反事实／Copy／Patch |
| `current_grounding.py` | 当前 RGB＋完整任务 → Current／Subject tokens |
| `libero_policy.py`／`robotwin_policy.py` | 各平台图像／源动作适配；ESkill 不输入真实状态或投影 |
| `policy.py`／`policy_config.py` | 历史动作策略、checkpoint 加载与归一化 |

## 当前 Grounding 接口

`GroundingDataset` 读取自包含的 `libero40_grounding`。输入为主相机与完整任务，目标为 Current／Subject 的干净原位画布和有效性；Event、框、mask、实体和编辑信息只参与 targets／metadata，不供模型作为预测条件。

两路共享标签布局 `(current, subject)`，保留两臂轴但只启用单臂。普通训练组 batch 按最长有效任务长度裁掉 padding，保留 EOS；增强采样键带 epoch／draw，可跨 worker 和恢复复现。

```python
from openpi.policies.current_grounding import create_current_grounding_policy

policy = create_current_grounding_policy('/path/to/current/checkpoint.pt', device='cuda')
result = policy.infer([main_rgb_uint8], 'put the milk in the basket')
current = result['grounding']
subject = result['subject_grounding']
```

返回 tokens，不返回动作或已定义的存在置信度。推理不读取真实 Event、bbox、状态或相机标定，也不加载 VAE。模型 feature／监督定义见 [Design](../../../docs/design.md) 和 [Grounding](../../../docs/target_grounding.md)。

## 保留的动作接口

`EpisodeDataset(..., auxiliary=True)`、`create_trained_policy()` 和原动作服务用于历史 Skill／Current／Future／Motion／Action 路线；纯 π0.5 使用相应基线配置。它们没有自动接入新的 Current／Subject tokens，不能只换专项 checkpoint 就当动作策略运行。

历史 Motion 保持点由 `motion_terminal_hold` 控制：开启时二维终点保持可进入 Trace loss，夹爪不补造命令；关闭时越界监督排除。当前 Grounding 不读取这些未来标签。历史读取、在线动作和参数范围见 [历史训练](../../../docs/history/joint_training.md) 与 [历史测评](../../../docs/history/joint_evaluation.md)。
