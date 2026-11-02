# 逐任务 train／val 划分

2026-11-03：40 个任务**分别**按完整父轨迹划分 90% train／10% val。合计 **1,518 train／175 val，共 1,693 条**。每个任务均有验证轨迹，两侧没有父轨迹交集。

[固定清单](../../../split.json) · [跨格式核对](verification.json) · [真实数据 CPU 链路检查](smoke/report.json)

## 规则

- 任务按主任务指令识别，统一空白／下划线，未使用 shard／轨迹编号前缀当作任务。
- 每个任务单独将父轨迹按 seed=42 的哈希排序，验证数量取最接近 10% 的整数，0.5 向上取整。至少两条时保留两侧；单条只能进入 train。全量任务为 29–50 条，因此全部具有 val。
- 计数单位为完整父轨迹；帧、左右臂、主／腕相机和派生记录跟随父轨迹。重复视角不增加配额。
- SFT 缓存、基础 Episode 和全量相机版各自独立计算的逐任务配额与归属完全一致。训练统一通过同一个 `split_path` 读取固定清单；切换训练 seed 不改变清单及固定验证子集。
- 不改写或复制原 Episode、RGB、动作或标签。归一化只用 train；SFT train-only 分位数已生成，覆盖 1,518 条／245,553 个原始来源帧，排除 175 条／27,912 帧。

逐任务的整数配额使总体比例约为 89.66%／10.34%；没有用总体比例反向调节各任务。

## 每个任务的数量

| 任务 | 总轨迹 | train | val |
| --- | ---: | ---: | ---: |
| open the middle drawer of the cabinet | 43 | 39 | 4 |
| open the top drawer and put the bowl inside | 36 | 32 | 4 |
| pick up the alphabet soup and place it in the basket | 44 | 40 | 4 |
| pick up the bbq sauce and place it in the basket | 46 | 41 | 5 |
| pick up the black bowl between the plate and the ramekin and place it on the plate | 45 | 40 | 5 |
| pick up the black bowl from table center and place it on the plate | 46 | 41 | 5 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 42 | 38 | 4 |
| pick up the black bowl next to the cookie box and place it on the plate | 46 | 41 | 5 |
| pick up the black bowl next to the plate and place it on the plate | 47 | 42 | 5 |
| pick up the black bowl next to the ramekin and place it on the plate | 45 | 40 | 5 |
| pick up the black bowl on the cookie box and place it on the plate | 43 | 39 | 4 |
| pick up the black bowl on the ramekin and place it on the plate | 39 | 35 | 4 |
| pick up the black bowl on the stove and place it on the plate | 35 | 31 | 4 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 44 | 40 | 4 |
| pick up the book and place it in the back compartment of the caddy | 41 | 37 | 4 |
| pick up the butter and place it in the basket | 45 | 40 | 5 |
| pick up the chocolate pudding and place it in the basket | 50 | 45 | 5 |
| pick up the cream cheese and place it in the basket | 45 | 40 | 5 |
| pick up the ketchup and place it in the basket | 45 | 40 | 5 |
| pick up the milk and place it in the basket | 45 | 40 | 5 |
| pick up the orange juice and place it in the basket | 45 | 40 | 5 |
| pick up the salad dressing and place it in the basket | 47 | 42 | 5 |
| pick up the tomato sauce and place it in the basket | 42 | 38 | 4 |
| push the plate to the front of the stove | 33 | 30 | 3 |
| put both moka pots on the stove | 29 | 26 | 3 |
| put both the alphabet soup and the cream cheese box in the basket | 43 | 39 | 4 |
| put both the alphabet soup and the tomato sauce in the basket | 33 | 30 | 3 |
| put both the cream cheese box and the butter in the basket | 49 | 44 | 5 |
| put the black bowl in the bottom drawer of the cabinet and close it | 35 | 31 | 4 |
| put the bowl on the plate | 49 | 44 | 5 |
| put the bowl on the stove | 48 | 43 | 5 |
| put the bowl on top of the cabinet | 46 | 41 | 5 |
| put the cream cheese in the bowl | 40 | 36 | 4 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 38 | 34 | 4 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 36 | 32 | 4 |
| put the wine bottle on the rack | 36 | 32 | 4 |
| put the wine bottle on top of the cabinet | 47 | 42 | 5 |
| put the yellow and white mug in the microwave and close it | 34 | 31 | 3 |
| turn on the stove | 50 | 45 | 5 |
| turn on the stove and put the moka pot on it | 41 | 37 | 4 |

## 新训练入口与验证

- [SFT 配置](../../../../../../examples/libero/train_pi05_sft_val.json)：原始 π0.5 基座＋四卡 LoRA，使用固定清单和 train-only 统计。
- [辅助训练配置](../../../../../../examples/libero/train_eskill_val.json)：新 SFT checkpoint＋全量带相机标定的 Episode，继续使用同一清单。
- 每 1,000 步及末步验证，每卡最多 32 个固定 batch；eval/no_grad，不进行训练图像增强。Flow Matching 噪声／时间固定，验证后恢复训练 RNG。
- 整个验证子集按全局有效标签数平均，三角色分开归一化后相加；多卡分配不重复样本。W&B 只加五项对应 `val_*loss`，SFT 只有总／动作两项；训练和验证同一步合并上传。
- 旧全量 SFT checkpoint 已见过全部数据，不用于这份独立 val 方案。初始化入口会核对相同的固定清单和验证比例。

已通过逐任务配额、父轨迹／视角不泄漏、跨格式及派生子集清单复用、train-only 分位数、验证 RNG 隔离、跨批次有效项平均、两进程全局平均和带验证的恢复测试。

真实数据 CPU smoke 已完成 SFT 的 2 步＋恢复第 3 步，以及全量辅助数据的同样检查；两阶段各三次 val 均为有限值。固定验证重复结果逐值相同，回读 checkpoint 的验证结果也相同。这里使用随机小模型／随机冻结小 VAE，仅验证划分、加载、统计与训练流程；新的正式 SFT 和辅助训练尚未启动。

清单生成入口为 [prepare_split.py](../../../../../../scripts/prepare_split.py)，实现见 [训练配置](../../../../../../src/openpi/training/config.py) 与 [验证循环](../../../../../../src/openpi/training/validation.py)。清单已经存在，直接复用；另建划分时使用新的输出文件。
