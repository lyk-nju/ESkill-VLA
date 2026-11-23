# Grounding 在线增强

每个训练样本依次经过以下两个处理阶段。它们是同一次 Grounding 训练中的数据处理顺序，与历史模型 Stage 1／Stage 2 训练配方无关。验证关闭全部增强。

## 第一阶段：局部编辑与反事实

`GroundingDataset` 读取当前 RGB、真实 Event 目标、场景实体及保存的可见／完整框，使用独立 RNG 处理 Copy、文本选择和 Patch。

1. **Copy-Paste**。Current Grounding 主配方按帧以 70% 概率进入位置增强。候选须有完整 bbox，原生可见率至少 **90%**，且未被交互锁定。夹爪与相关实体各自保留独立保护框。当剩余候选数为 `N>0` 时，在 `1～min(3,N)` 之间均匀抽取计划移动数。每次移动先检查擦除范围，再用最新的其他实体 bbox、机器人 mask 和各保护框计算可行落点。约束为不越界、不重叠、位移至少 4 源像素且每轴不超过图像对应尺寸的 30%。从可行位置中均匀选择一次。成功后立刻更新图像和 bbox，下一次规划可以使用已腾空的位置。失败则跳过该物体，同一物体每帧只处理一次。
2. **完整任务反事实**：Pick 使用原指令和原目标。Place 对能唯一绑定源／目的的简单搬运任务尝试替换源物体名，保留原句式和标点。同图依然只有一个 Current Query。候选要求有有效 bbox、描述唯一、至少 80% 可见、未被交互锁定，排除原源／目的和依赖图像左右消歧的对象。物体能否放进目的地不参与筛选。反事实只提供 Grounding 监督；没有合格候选时沿用原指令与标签。
   LIBERO-Goal 的「开顶层抽屉再把碗放进去」沿用 `Open → Pick → Place` 的实体绑定，只在 Place 时尝试替换待放物体，顶层抽屉保持原绑定。配对诊断也从 Place 帧构造反事实任务。
3. **生成干净标签**。从编辑后的 RGB 按 Current 和 Subject 各自的完整 bbox 生成原位画布。框内保留 RGB，框外置黑，前景权重取同一范围。
4. **Patch Masking**：先保护 Current 与夹爪完整 bbox 各外扩 8 源像素，以及夹爪／交互物局部区，再遮挡剩余 patch。与保护区相交的整块 patch 都保留；标签仍使用第 3 步干净画布。

Copy 先尝试矩形擦除。矩形触及邻物、机器人或保护框时，才尝试只擦除目标可见 mask 外扩 1 源像素的范围。该范围仍须与邻物可见像素、机器人 mask 和各保护框分离。邻物已移动时用其更新后的 mask 位置检查。粘贴与落点仍按原有规则执行。

LIBERO Pick Event 可保存 `copy_precontact_stop_frame`，表示允许例外的区间结束帧，右开。在此区间内，原 Current 若满足可见率和任务约束，可以通过自身与夹爪外扩框重叠的候选检查。擦除原位置时保护机器人实际像素、其他交互实体和任务区域，落点仍避开夹爪外扩框。首次接触前两帧起恢复常规锁定。缺少该字段时使用常规规则。

在能运行 LIBERO 且能读取原始匹配状态的环境中，用 `python -m scripts.preprocess.libero.pick_precontact --index DATA/index.json --output pick_windows.json --write` 生成接触区间并写入 Episode 元数据。省略 `--write` 时仅生成报告。训练时读取已保存的区间。

候选有三个不同概念：可定位对象、可用于当前完整任务替换的对象、可 Copy 的对象。清晰可见但不能移动的物体仍可被查询，实际替换还需要符合该任务和交互状态。

解析任务时接受 LIBERO 指令中的简称，例如 `stove` 对应 `flat stove`，`rack` 对应 `wine rack`，`cabinet` 对应 `wooden cabinet` 或 `white cabinet`。简称仍需唯一匹配。`on top of` 是独立的放置关系，替换时保留原文。训练使用已保存的任务文本及 Event 绑定。

遇到 leftmost、between、next to 等空间指代时，组合使用原任务中的实体名称和保存的 Event 绑定。锁定相关操作物体、参照物、可能参与消歧的同类实例以及有标注的目标区域。它们的框与夹爪保护区一起参与擦除和落点检查。未涉及的物体按常规规则尝试移动。简单柜顶放置只锁定目的地，任务源物体仍可移动。当前监督目标是附属 Region 时，保护区域和父物体。

可明确拆开的指令按当前 Event 释放独立的 Current。例如「拿起炉子上的碗，放到盘子上」在 Place 可搬唯一的盘子。如果 Pick 描述使用盘子作参照，则保留保护。对「把白杯放左盘、把黄白杯放右盘」这类指令，Pick 可搬仅靠名称就能唯一指代、且没有用作目的地参照的杯子。释放后仍要满足现有 Copy 资格、夹爪保护、擦除和落点检查。

## 增强前的 Current 优先

取原指令在当前 Event 的 Grounding 目标。对 `put milk in basket`，Pick 优先 Milk，Place 优先 Basket。该目标通过现有 Copy 资格和保护规则后，以 75% 概率成为本轮第一个尝试移动的物体。未抽中时，本轮从其他候选中按随机顺序尝试。

整轮位置增强结束后再筛选文本替换候选。若新文本选中了被移动的物体，Current 使用其更新后的框和可见掩码。Subject 和 Patch 保护区也从最终图像和标签构造。

| 参数 | 增强配置 | 无增强对照 |
| --- | ---: | ---: |
| `copy_prob` | 0.7 | 0 |
| `copy_target_prob` | 0.75 | 不使用 |
| `copy_max_objects` | 3 | 不使用 |
| `copy_max_shift` | 0.3 | 不使用 |
| `copy_gripper_margin_px` | 每边 4 源像素 | 不使用 |
| `target_swap_prob` | 0.3 | 0 |
| `target_min_visible_fraction` | 0.8 | 不使用 |
| `patch_prob`／`patch_drop_prob` | 0.5／0.25 | 0 |
| `patch_size` | 224 输入上 28 px，按源图缩放 | 不使用 |
| `target_margin_px` | 每边 8 源像素 | 不使用 |

Copy 需通过资格与落点检查。Place 帧的反事实先确认任务和候选都有效，再按 30% 概率抽样。

## 第二阶段：经典图像增强

batch 层使用共享归一化仿射参数变换主图、两路干净标签、权重及几何有效性。之后只对输入 RGB 做颜色扰动，冻结 VAE 在几何变换后编码干净标签。

| 参数 | 当前增强配置 |
| --- | --- |
| 水平翻转 | 主配方以 50% 概率翻转。含明确左右方向或顺／逆时针的指令跳过翻转 |
| 随机裁切 | 每次，边长固定 0.95；224→212→224 |
| 中心旋转 | 每次，[-5°, 5°] |
| 亮度／对比度／饱和度 | 每次，幅度 0.3／0.4／0.5 |

强度参数来自官方 [OpenPI PyTorch 预处理](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models_pytorch/preprocessing_pytorch.py)。沿用项目同步仿射／颜色实现，不宣称与上游逐像素或随机序列一致。含 left／right 等方向词的文本跳过几何增强，保留颜色扰动。

目标完全被裁出／旋出画面时，仅关闭对应角色 loss；部分出画仍训练。检查使用未外扩的完整目标框，背景上下文不计为目标。原生遮挡不通过可见 mask 缩框或删除标签。

## 启动、诊断与复现

反事实开启时，初始化检查每个主相机物体的逐帧布尔 `interaction_locked`；Copy 开启时检查逐帧 `robot::single` mask。缺失标记会报错并提示补齐，不静默降低增强覆盖率。锁定是“物体框与外扩夹爪框相交”的二维代理，不等于真实接触。

记录 `target_swapped`、`copy_applied`、`target_moved`、`copy_priority_target_moved`、每个物体的位移 `moved_offsets`、计划及实际移动数量、任务保护实体、失败原因、可行落点数、原生不可见与标签有效性。Copy 成功不等于最终监督目标移动；反事实可能选到其他对象。

Dataset 的 seed、episode、frame、epoch、draw 决定三条 RNG；batch 使用可恢复 Torch RNG。中断恢复沿用相同数据、配置与卡数。修改候选规则或刷新数据后新开实验，不混用旧增强序列。

用户确认保留目标专属保护。遮挡分布会随当前目标改变，存在潜在统计捷径，但没有证据说明模型已利用；像素／标签一致性也不能排除它。必要时用关闭 Patch 的对照判断实际影响。

核验入口见 [LIBERO](../examples/libero/README.md)，当前记录见 [数据验收索引](../data/libero/atomic-vla/check/test/README.md)。
