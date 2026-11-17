# 三类候选与增强覆盖率

2026-11-18，`feat/visual-primitive-v3`。本轮只修改数据候选／编辑规则，模型沿用融合 Current＋未融合 Subject。此页记录可行落点改造前的结果；后续更新见[先计算可行落点](../copy_placement/README.md)。

| 集合 | 条件 |
| --- | --- |
| `locatable_objects` | 当前完整框有效、描述可消歧、原生可见比例至少 80% |
| `swap_candidates` | 可定位，并适合简单完整任务的源物体替换；排除原源／目的、空间消歧描述、交互锁定或锁定状态缺失的物体 |
| `copy_candidates` | 单独沿用 `copy_eligible`；至少 98% 可见、独立刚体、远离夹爪等原约定 |

新增 `interaction_locked` 直接保存原生预处理中已有的 bbox 邻近代理：物体完整框与夹爪框外扩 4 源像素相交。它不是接触／持物真值。已有 1,693 条轨迹已补齐，共 1,360,425 个物体帧；布尔数组追加到现有 NPZ，元数据备份在报告目录 `interaction_locks/episode_metadata_before`，原图像、框与 mask 不改写。新增预处理自动保存该标记，其他机器可使用[补齐命令](../../../../../../src/openpi/training/README.md#current-grounding-专项训练)。

`simple_transfer()` 仅负责文本改写，支持 put／place、in／into／on／onto 及 pick up … and place／put it …。Copy 独立判断；leftmost／beside／between、源位置关系等任务首版跳过，其他任务最多一次随机编辑。Copy 和目标选择保持独立，未增加重采样或复杂场景判断。

## 相同样本对比

旧规则取自 `0e2017b`。训练集 40 任务各随机抽 200 帧，共 8,000 帧；修改前后逐条 Episode／frame 相同，均为 seed 42、epoch 0、相同抽样 draw。统计只包含 Dataset 第一阶段，不含 batch 图像增强或模型。这是分层抽样，不是全量逐帧比例。

| 指标 | 旧规则 | 最终规则 |
| --- | --- | --- |
| 反事实替换 `target_swapped` | 1,116／8,000，13.95% | 1,210／8,000，15.13% |
| Copy 尝试 | 1,205／8,000，15.06% | 2,129／8,000，26.61% |
| Copy 成功 `copy_applied` | 398／8,000，4.98% | 551／8,000，6.89% |
| 监督目标移动 `target_moved` | 110／8,000，1.38% | 62／8,000，0.78% |

最终有 552 次替换选择了 `copy_eligible=false` 的物体。逐条验证 1,210 个替换目标可见比例达标且未锁定；无缺失锁定标记，增强新造成的完全遮挡计数为 0。3,800 个空间保护样本没有执行 Copy。文本替换覆盖 12 类简单任务；21 类任务允许 Copy，18 类实际有成功搬运，其余三类本批次没有合格的成功编辑。

监督目标移动率仍低且低于旧规则。候选扩展与编辑／查询独立后，Copy 成功不能代表所监督目标位置发生变化；本轮按约定只统计，不增加优先监督粘贴物体的偏置。这里也不证明完全排除了视觉捷径。

[逐任务对比表](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_coverage_20261118_001/by_task.md) · [最终逐任务 CSV](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_coverage_20261118_001/after_final/by_task.csv) · [完整报告](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_coverage_20261118_001/final_comparison.json)

报告还给出按训练帧数加权的估计，保留逐样本记录、失败原因、源码／配置快照和随机键。最终 8,000 帧检查总耗时约 67.6 秒，包含 Dataset 创建与 I/O；旧检查有不同缓存状态，不将两者耗时当作训练吞吐对比。

CPU 相关回归 111 项通过、1 项因沙箱 IPC 限制跳过。测试包括：可见但不可 Copy 的目标、已交互锁定目标、into／place 句式、非 Transfer 的 Copy、空间指代保护、编辑后的像素／框／可见 mask／VAE 画布同步，以及元数据补齐的数组保留和幂等性。本轮没有新增模型训练。

覆盖率入口：

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.grounding_coverage \
  --config examples/libero/train_current_grounding.json --samples-per-task 200 \
  --output /data1/yuankai/robot-research/workspace/reports/grounding_coverage_new_run
```
