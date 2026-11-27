# Current／Subject：区域与 VAE 监督

当前两路是任务条件 Current 和视觉 Subject。Current 表示下一步交互目标，Subject 表示夹爪所在区域；这里不训练 Future。结构见 [Design](design.md)。

## 标签语义

| 观察／任务 | Current | Subject |
| --- | --- | --- |
| milk 在桌上；put milk in basket | milk | 夹爪 |
| 已抓起 milk；put milk in basket | basket | 夹爪局部，可含 milk |
| 抓着 milk；反事实 put pudding in basket | 可见且未锁定的 pudding | 相同真实夹爪局部 |

真实 Current 由当前 Event 的 `interaction_target.entity_id` 绑定到场景实体。Pick／Push 通常指操作对象，Place 指目的实例或明确目标区，Open／Close／Turn 指相关部件。不能把这些目标统一退化成原操作物体或整张桌子。事件与实例规则见 [Event Organization](event_organization.md)。

Subject 读取已保存 `effector_regions.single.image`，沿用夹爪 bbox 的局部 RGB；当前不额外绑定持物实例。任务替换不改变物理交互状态，也不改变 Subject 标签。

## 完整框与原位画布

所有区域均为主相机 `full_in_frame`：独立渲染指定几何、忽略外部遮挡，裁掉屏外部分，由 mask 提取紧致 bbox。完全出画时框无效；被遮挡不缩小完整框。可见 mask 只用于增强资格与编辑。

Current 和 Subject 都使用原始完整 bbox，每边留量为 0。先在源图上保留框内实际 RGB、将框外置黑，再等比例缩放并补边到 256×256。前景权重使用同一范围，目标保留原位置、比例和实际遮挡。标签在读取 Episode 时生成，已有轨迹可直接使用。使用过每边 4 像素留量的实验应单独比较，紧框配方从新 run 开始。

Copy 成功时更新被移动实体框，并从编辑后的 RGB 重新构造画布。当前目标是附属 Region 时，不移动其父物体。干净画布在随机 Patch 前生成；后续裁切／旋转同步变换输入、标签和权重，仅输入做颜色扰动。

## Current 紧框监督

定位实验以最终主相机坐标系中的紧框监督 Current。BBox 使用 L1＋GIoU，BBox＋GAC 用同图实体 ROI 加一项对比交叉熵；两组 Subject 都用独立的 L1＋GIoU 夹爪框监督。在线策略直接从 RGB 和完整任务生成预测框。详见[实现、配方和验收](../experiments/current_grounding_localization/README.md)。

## VAE 画布配方的 latent 监督

冻结 FLUX VAE 用 FP32、posterior mode、固定 scaling／shift 编码标签。16×256 预测 tokens 经共享可训练读出头生成 `16×32×32` latent，使用区域加权 MSE，前景／背景默认 1／0.1。Current／Subject 分别使用有效 mask，损失总权重均为 0.1。

原生遮挡保留原始监督。反事实新目标要求至少 80% 可见。经典几何变换使目标完全出画时，只关闭该路标签，其他有效监督继续使用；背景边距不当作目标面积。缺失／出画与完全不可见分别记录。

## 能力边界

VAE 重建损失和画布亮度不等于目标存在概率。Subject 区域可能包含持物，但不保证识别物体身份；反事实目标定位不等于学会恢复动作。先验证目标与位置变化下的稳定读出，再研究接回动作模型。

视觉监督参考 [AffordanceVLA](ref_papers/target_grounding/affordancevla.md)，操作区域动机参考 [ReconVLA](ref_papers/target_grounding/reconvla.md)。历史 Current／Future 定义保留在 [联合动作设计](history/joint_design.md#4-currentfuture-object-grounding)。
