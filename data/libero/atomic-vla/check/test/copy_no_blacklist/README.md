# 取消容器 Copy 类别黑名单

2026-11-18。代码与本机 1,693 条轨迹的预保存候选均已更新；不训练模型。

`scene_regions.copy_eligibility()` 统一要求可移动场景物体 `kind=object`、不在夹爪邻近区、可见比例至少 98%、完整区域非空，不再按 basket／bowl／mug／ramekin／pot／frypan 类别排除。固定装置、部件、虚拟区域仍不移动；源区、机器人、落点和空间指代保护继续执行。含有其他实例的容器仍需通过源区检查，不允许擦掉其中的其他物体。

已有数据通过 `interaction_locks --refresh-copy-eligibility` 更新：1,693 条全部完成，共 1,360,425 个物体帧，新增 221,639 个可 Copy 的物体帧候选。使用原可见／完整 packed mask 的精确像素数重算阈值；所有原先有效候选均保留。追加布尔数组、更新引用并备份原元数据，未改写 RGB、bbox 或 mask。其他机器需运行相同命令，单独更新 Git 代码不会刷新数据盘里的旧标记。

## 相同 8,000 帧检查

40 任务各 200 个训练帧，修改前后 Episode／frame 逐条一致。保持原目标优先 80%、反事实条件触发 20%、Copy 条件触发 50% 和既有保护规则。

| 指标 | 取消前 | 取消后 |
| --- | ---: | ---: |
| Copy 尝试 | 2,129 | 2,129 |
| 原目标确有安全移动预案 | 150 | 400 |
| Copy 成功 | 1,427 | 1,649 |
| 容器实际 Copy | 0 | 574 |
| 原任务目标作为监督且移动 | 117 | 284 |
| 反事实目标作为监督且移动 | 50 | 30 |
| 移动监督合计 | 167（2.09%） | 314（3.93%） |
| 反事实文本替换 | 454 | 454 |

实际移动的容器为 Basket 400 次、Moka pot 71 次、Porcelain mug 76 次、Black bowl 18 次、Frypan 9 次。这是固定抽样和数据链路检查，不是全量逐帧比例或训练效果。

[容器核验视频](../video/copy_no_blacklist_20261118.mp4) 展示 12 个保持原任务、移动当前容器目标的样例：红框是 Current，蓝框是夹爪；下方为干净 Current／Subject RGB 画布。仅覆盖 Dataset 阶段，未做后续经典增强、VAE 编码或模型预测。

[完整比较](/data1/yuankai/robot-research/workspace/reports/atomicvla_copy_no_blacklist_20261118_001/comparison.json) · [逐任务 CSV](/data1/yuankai/robot-research/workspace/reports/atomicvla_copy_no_blacklist_20261118_001/coverage/by_task.csv)

元数据备份与刷新记录在报告目录 `eligibility_refresh/`，来源代码、配置、原始抽样、执行脚本和核验图片保存在同一报告目录。相关 CPU 回归 103 通过、1 项因沙箱 IPC 限制跳过，覆盖容器资格、交互区／可见度／固定部件保护、数组保留和重复刷新一致性。数据资源身份已经改变，训练使用新运行。

刷新命令：

```bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.interaction_locks \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40_grounding/index.json \
  --output /data1/yuankai/robot-research/workspace/reports/copy_eligibility_new_run \
  --refresh-copy-eligibility
```
