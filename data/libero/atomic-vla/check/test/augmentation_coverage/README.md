# LIBERO 在线位置增强与文本替换统计

从 LIBERO 的四个套件各随机选 5 个任务，使用种子 42。统计选中任务的全部训练轨迹，共 20 个任务、772 条轨迹、125,951 帧。使用单进程训练 DataLoader 的首轮抽样键，每帧回放一次 Dataset 在线增强。

配置为位置增强触发概率 50%，每帧最多移动 3 个物体，合格文本候选的替换概率 30%。下表的帧比例均以各套件统计帧数为分母。

| 套件 | 训练轨迹 | 帧数 | 位置增强尝试 | 至少移动一个物体 | Current 目标被移动 | 文本替换 | 两种增强都成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LIBERO-Object | 205 | 30,839 | 15,387 | 15,093（48.9%） | 2,918 | 9,228（29.9%） | 4,503 |
| LIBERO-Goal | 196 | 22,939 | 11,534 | 8,193（35.7%） | 866 | 3,500（15.3%） | 1,479 |
| LIBERO-Spatial | 195 | 24,425 | 12,240 | 8,875（36.3%） | 0 | 0 | 0 |
| LIBERO-10（Long） | 176 | 47,748 | 23,719 | 18,382（38.5%） | 1,532 | 0 | 0 |
| **合计** | **772** | **125,951** | **62,880** | **50,543（40.1%）** | **5,316** | **12,728（10.1%）** | **5,982** |

位置增强尝试后有 **80.4%** 的帧成功移动物体。成功帧中，移动 1、2、3 个物体的帧分别为 **32,352、13,161、5,030**，共发生 **73,764 次物体移位**。失败的 12,337 次尝试中，按记录的最终状态，无可移动候选 10,223 次、原位与其他实体重叠 1,408 次、原位与机器人重叠 706 次。

Current 移动以文本替换后的最终监督目标计算，合计占 **4.2%**。这次抽到的 Spatial 任务保留了影响空间指代的实体位置，仍可移动其他物体。42,656 帧有合格文本候选，其中 12,728 帧抽中了替换（29.8%），每条替换都得到有效的 Current 标签。

[完整统计](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_train_20tasks_20261118_001/coverage.json) · [逐任务 CSV](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_train_20tasks_20261118_001/by_task.csv) · [逐轨迹 CSV](/data1/yuankai/robot-research/workspace/reports/atomicvla_augmentation_train_20tasks_20261118_001/by_episode.csv)

## 复现

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.grounding_coverage \
  --config examples/libero/train_current_grounding.json --tasks-per-suite 5 --workers 10 \
  --output /data1/yuankai/robot-research/workspace/reports/grounding_coverage_new_run
```

该入口统计 Dataset 阶段的 Copy-Paste 和反事实替换，采样次数与实际替换结果会随训练轮次和分布式采样键变化。
