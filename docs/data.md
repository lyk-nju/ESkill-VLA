# Current／Subject 数据需求与监督接口

当前专项读取已有 LIBERO40 的完整场景标注，按 Event 构造 Current，按执行器区域构造 Subject。模型只输入当前主相机 RGB＋完整任务。原始状态、动作、标定与历史终点字段可继续保存，但不作为这条路线的模型条件或训练目标。

## 当前资产

入口为数据盘 `atomicdata/libero40_grounding/index.json`，40 任务、1,693 条／273,465 帧／3,873 Events。固定父轨迹划分见 [split.json](../data/libero/atomic-vla/split.json)：1,518 train／175 val。所有帧、视角和派生样本跟随父轨迹，验证不使用训练增强。

整个 Grounding 数据目录可自包含迁移，训练不依赖原始 HDF5、LIBERO 仿真或 provenance 中的原机器路径。运行时数组使用相对引用，需复制整个目录。旧 `libero40` 数据没有全部在线编辑字段，不能直接替代。迁移命令见 [远端使用说明](remote_grounding_training.md)。

## 字段清单

| 需求 | 保存字段／来源 | 用途 |
| --- | --- | --- |
| 当前图像 | `cameras.image` 的 RGB、frame_indices、valid | 主相机模型输入；当前不使用腕相机 |
| 完整任务 | `instructions` | Current 的文本条件，不生成中间目标短句 |
| Event 时间和技能 | `events` 的 start_frame／stop_frame／skill | 离线阶段绑定与统计；不输入模型 |
| 当前交互目标 | `events[*].interaction_target` | 引用具体实例、部件或原生目标区 |
| 全场景实体 | `scene_entities` 的 kind／instance／category | 目标绑定、反事实和编辑候选 |
| 物体完整区域 | `scene_entities[*].regions.image` 的 bbox／mask | Current 原位标签；忽略外部遮挡，保留屏内部分 |
| 夹爪完整区域 | `effector_regions.single.image` | Subject 标签与保护区 |
| 原生可见区域 | `visible_masks`、`visible_fraction` | 实际可见像素编辑，反事实可见阈值 |
| 交互锁定 | `interaction_locked` | 阻止将夹爪附近物体用于移动或新的待抓目标 |
| 可移动标记 | `copy_eligible` | 预处理时保存的 Copy 候选统计 |
| 机器人与交互保护 | `robot::single` mask、`online_grounding.interaction_boxes` | Copy 用机器人 mask 和逐实体保护框，Patch 用保存的交互区域 |

`interaction_locked` 保存物体 bbox 与夹爪 bbox 每边外扩 4 源像素后的相交结果，供文本反事实和离线核对使用。在线 Copy 使用完整物体框、至少 **90%** 的原生可见率，以及按 `copy_gripper_margin_px` 重新计算的夹爪保护框，默认每边外扩 4 源像素。离线 `copy_eligible` 记录原预处理口径。

反事实开启时，Dataset 初始化要求主相机物体的逐帧布尔锁定标记齐全；Copy 开启时要求逐帧机器人 mask 有效。缺失会启动报错，避免训练中途才发现增强未生效。缺失框或 mask 需重新同步完整标注；补齐脚本不能凭空恢复原生区域。

## 保存布局

```text
libero40_grounding/
  index.json
  episodes/<episode_id>/
    episode.json
    arrays.npz
    images.npz
    scene_regions.npz
    online_grounding.npz
    grounding_check.json
```

Episode 保存来源事实、实体、数组引用和绑定依据；训练增强、原位画布、VAE latent 和 batch 在读取时生成，不另存一份重复训练格式。底层保存约定、NPZ／JPEG 引用、mask packbits 和区间见 [data/README.md](../data/README.md)。

## 监督语义与有效性

Current 在当前 Event 边界切换：Pick milk → milk，Place milk in basket → basket，关节操作 → 相应部件，明确位置目标 → 原生 Region。原有 `object_regions` 仍表示被操作物体，不能直接用来替代新的交互目标。

Subject 是夹爪完整框周围的局部 RGB，允许包含被抓物体像素，不新增持物身份标签。完整屏内框不因遮挡缩小；RGB 保留真实遮挡。缺失或完全出画时只关闭对应标签。

Visual Primitive 的 Current 用可见 mask 监督多层注意力。Dataset 初始化时检查采样帧中所有有效物理框的可见 mask。有效的全零 mask 表示完全不可见，该实体不参与 Current KL 和 GAC。投影 Region 的几何标签和 Subject 的夹爪框回退见 [Visual Primitive](../experiments/visual_primitive/README.md)。

反事实候选需唯一描述且至少 80% 可见；复制资格与查询资格独立。Copy 与标签同步、保护区及几何变换规则集中在 [增强流程](augmentation.md)。真实 Event、bbox 和 mask 仅用于标签、编辑及诊断，不提供给模型作为预测条件。

## 补齐与核验

已有原生框／mask 时用 `python -m scripts.preprocess.libero.interaction_locks --index ... --output ... --refresh-copy-eligibility` 补锁定并刷新无黑名单 Copy 候选。先备份再追加布尔数组，不改 RGB／bbox／mask；训练期间不要改数据。

`grounding_check` 检查全量元数据、划分和真实 DataLoader；`grounding_coverage` 按任务抽样统计替换／Copy／监督目标移动；`grounding_review` 展示增强后图像与真实 VAE 重建。结果保存在数据盘，项目内 [check/test/](../data/libero/atomic-vla/check/test/README.md) 保留入口，视频放该目录的 `video/`。

已完成自动补标与数据链路检查，不等于全量人工质量验收；实例绑定人工状态和来源对齐记录继续保留。历史动作窗口、未来二维轨迹、夹爪命令、标定及旧镜像监督见 [联合动作数据说明](history/joint_data.md)。
