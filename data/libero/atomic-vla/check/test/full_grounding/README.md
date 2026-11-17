# LIBERO40 全量 Grounding 与输入核验

2026-11-17：自动补标和输入链路检查完成；全量人工质量验收待进行。原数据保持不变，未启动正式 π0.5 训练。

数据入口：[libero40_grounding/index.json](/data1/yuankai/robot-research/data/atomicdata/libero40_grounding/index.json)。增强与无增强对照配置均已切到此索引，仍为 32 步接口模板。生成和检查命令见 [LIBERO 示例](../../../../../../examples/libero/README.md#current-grounding-数据与训练接口)。

| 保存项目 | 结果 |
| --- | --- |
| 范围 | 40 任务、1,693 条、273,465 帧、3,873 Events |
| 场景实体 | 物体、有限固定装置、关节部件、原生目标区域；不包含桌面／地面／墙面等背景 |
| 完整屏内 bbox／mask | 2,448,505 个实体帧，含机器人保护 mask；忽略外部遮挡，屏外裁掉 |
| 原生可见 mask | 2,158,386 个实体帧；虚拟区域没有物理可见 mask |
| Event 当前目标框 | 273,465／273,465 帧有效 |
| 夹爪框 | 273,431／273,465 帧有效；34 帧完整出画 |
| 在线增强元数据 | 可见比例、可移动资格、局部交互保护框、目标文本模板 |
| 保存核验 | 全部 mask／bbox 一致；54,890 个历史抽样框一致；15,654 个十条试验实体帧的框与 mask 一致 |

每条 `episodes/<episode_id>/` 保存 `episode.json`、`arrays.npz`、`images.npz`、`scene_regions.npz`、`online_grounding.npz` 和核验记录。训练用引用均在本目录内。原数组／图像在本机以硬链接复用，按只读使用；跨机器复制整个数据目录即可，provenance 中的原始 HDF5 路径不参与训练读取。

本次在 GPU 3／0／2 分任务执行，原生批处理约 30 分 39 秒。新增区域／在线增强 NPZ 约 18.14 GiB；图像与原数组约 9.57 GiB 在本机复用。可见 mask 按 bit 打包，以未压缩 NPZ 保存供逐帧内存映射读取。相同实体与相机姿态仅在逐值相等时复用隔离 mask。

| 输入检查 | 结果 |
| --- | --- |
| 固定划分 | 1,518 train／175 val，245,553／27,912 帧；无父轨迹交叉 |
| Event 首／中／末帧 | 训练 10,416、验证 1,203 个样本，覆盖全部轨迹和 Events |
| 多 worker | 两 worker 与单进程首批张量及元数据逐元素一致 |
| 训练增强 | 5,224 次文本替换均满足 ≥80% 可见；1,203 次成功 Copy-Paste；增强造成的监督失效为 0 |
| 验证增强 | 文本替换、Copy-Paste、随机 Patch 全部关闭 |
| 正式 DataLoader 入口 | 随机 64 batch／128 样本，覆盖 125 条轨迹；预取 worker 正常结束 |
| 反传 | 随机小骨干＋真实冻结 FLUX VAE，batch 2、8 步、六类技能来源；Loss 与梯度有限 |
| 回归 | 37 项相关测试通过；单独 worker 回归本轮未重跑，已执行真实两 worker 输入检查 |

短程随机读取约 93.6 样本／秒（两 worker、batch 2，去除前八批）。这是共享机器上的 CPU 输入测量，不包含模型计算、GPU 搬运和启动；不是完整 π0.5 或独占机器吞吐。反传使用 FP32、小模型随机初始化、AdamW 5e-5、betas=(0.9,0.95)、weight decay=1e-10、梯度裁剪 1；Current／Subject／目标文本 CE 各 0.1，W&B 关闭。

首轮检查提前中断大数据集的预取 worker 时出现退出警告，已把检查轮次限制为 128 个随机样本并正常读完；同一批样本复核通过，日志无退出警告。此修改仅涉及检查脚本，不改变训练采样。

两类有效性例外明确保留：34 帧夹爪出画均来自 `libero_10_no_noops_s024_r011`，只关闭对应 Subject loss；另外 5 条训练轨迹合计 12 帧中，同类碗的左右描述无法唯一消歧，跳过原 Event 文本监督及相应原目标查询，可用的反事实目标不受影响。其 bbox 仍完整保存；逐帧名单在 `source_label_coverage.json`，不把它们算成缺框。

查看视频：

- [全场景框／目标切换，3 分 06 秒](../video/full_grounding_all_tasks.mp4)：每个任务一条轨迹，93 个 Event 中点；目标红、其他物体绿、夹爪蓝。
- [在线增强／Protect／真实 VAE 重建，3 分 18 秒](../video/full_grounding_augmentation_all_tasks.mp4)：相同 40 条轨迹，共检查 6,632 帧，展示 99 张面板；保护区像素检查全部通过，3,358 次反事实查询均满足 ≥80% 可见。

两份视频均每张面板停留两秒，属于标签与增强检查，不是连续 rollout，也不是模型预测。VAE 重建使用真实 FLUX 权重；人工抽看了目标切换、部件框和移动目标样例，尚未替代用户的全任务验收。

详细结果：

- [原生生成汇总](/data1/yuankai/robot-research/data/atomicdata/libero40_grounding/summary.json)；同目录 `run.json`、逐任务日志和逐轨迹检查记录保留命令、GPU、代码哈希及来源依据。
- [输入与反传](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/input_full/input_check.json)、[全帧文本覆盖](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/input_full/source_label_coverage.json)、[随机加载复核](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/loader_check/checks.json)。
- [40 任务增强检查](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/review40/checks.json)。每次技术运行的 `run.json` 均保留当次源码状态，初次退出警告日志也保留，不改写历史结果。

相机恢复 RGB 的抽样 MAE 最大约 9.15／255；框与已有标签一致不等于逐像素 RGB 完全复现。原有对齐质量记录及目标绑定人工审核状态继续保留，不宣称全量人工通过。

## 同日代码复查修正

Copy-Paste 增加机器人 mask 启动检查，预处理示例明确开启 `protect_robot`；全量 1,693 条已通过新检查，无需重新渲染。EpisodeReader 改用公开 NPY header API，保留 C／Fortran 顺序和只读 mmap，压缩成员／NPY 3.0 用 `np.load`。部件及目标区域继承父实例的消歧描述，唯一背景表面上的合法目标区域保留；全量目标文本覆盖仍为 273,453 帧有效、12 帧歧义，无新增缺失。

目标文本在 collate 时按最长有效长度裁剪，保留 EOS；全无效 batch 留一个 masked 位置。合成测试中 logits 从 `[2,64,vocab]` 降为 `[2,3,vocab]`，有效 Loss 和梯度一致；真实 32 样本 batch 的最长文本为 11，实际裁为 `[32,11]`。

相关回归共 88 项通过，包含多 worker／中途恢复一致性；存储测试覆盖 NPY 1.0／2.0／3.0、压缩／未压缩、C／Fortran 排列。执行环境 NumPy 1.26.4，未在评审所述的 NumPy 2.5.5 环境复测。详细真实数据检查见 [review_fixes/checks.json](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/review_fixes/checks.json)。

讨论后用户确认保留目标专属 Patch 保护和随机遮挡，潜在目标位置捷径不作为当前训练阻断项。本页此前的保护区、画布和标签一致性检查不代表排除了统计捷径，也没有证据说明模型已利用它。VLM 解冻与目标文本生成配置、夹爪 Subject 的定义均保持原约定，未新增持物身份或纠错动作监督。

## 组合水平翻转和常规图像增强

增强配置已统一接入训练 batch 入口：在线 Copy-Paste → 单目标文本替换／干净画布 → 目标及夹爪保护下的 Patch Mask → 水平翻转 → 同步轻裁切／旋转 → 输入颜色扰动 → 冻结 VAE 编码。水平翻转概率 0.5、颜色 0.8、裁切 0.5（边长 0.95～1）、旋转 1.0（±5°）。含左右方向词的查询跳过几何变换；真实任务文本生成仍用原图。验证和 Event 基线不增强。

[组合增强核验视频，3 分 06 秒](../video/combined_grounding_augmentation.mp4) 覆盖 40 任务、93 个 Event 中点：44 次翻转、41 次裁切、81 次旋转、71 次颜色扰动，含 6 次 Copy-Paste 和 39 次反事实查询。186 项 Current／Subject 标签变换前后均有效；标签使用真实 FLUX VAE 重建。抽看了左右指令保护以及复制＋翻转＋旋转案例，图像与标签位置对应；尚待用户验收。视频是逐样本面板，不是 rollout。

68 项增强／Grounding／阶段回归通过，含 CPU 两进程 DDP 和 worker 重放。另通过正式 train 入口运行 4 步随机小骨干＋真实冻结 VAE 的 GPU FP32 检查，损失及梯度有限，两次验证完成并保存 checkpoint；不是原 π0.5 正式训练或 BF16／多卡吞吐验证。

新增 batch 增强在 CPU 单样本下平均 19.09 ms、P95 22.93 ms，排除 Dataset、VAE、渲染，不能作为完整训练吞吐。详见 [组合预览记录](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/combined_review/checks.json) 和 [技术训练输出](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/combined_smoke/train/metrics.jsonl)。原数据和历史报告未改写，无需重新生成标签。

## 第二阶段参数改为官方 π0.5 配置

当前 `train_current_grounding.json` 的第二阶段改用官方 OpenPI PyTorch 预处理参数：关闭水平翻转；主图固定裁切 95% 边长（224→212→224）并随机旋转 ±5°；颜色每次采样，亮度／对比度／饱和度幅度为 0.3／0.4／0.5。第一阶段配置沿用既定方案。Grounding 的方向词保护、几何与标签同步、输入独立颜色扰动继续生效；对齐的是参数，底层同步采样及颜色实现未改为上游算子。

本次增强回归 15 项通过，包含整数裁切起点／尺寸、颜色范围、监督对齐与原始任务图隔离。上面的组合视频、19.09 ms 耗时和 4 步技术训练对应此前 50% 翻转配方，保留作历史检查，不代表此次官方参数配置的实测结果。

## 当前参数的数据复查

2026-11-17 复查：71 项相关测试通过（NPY 3.0 兼容性样例有预期的 NumPy 格式提示）。全量元数据中的 3,873 个 Event 边界／技能标记均有效，交互目标绑定状态均为 resolved；这不替代人工语义审核。按当前官方强度配置检查 40 任务、763 条轨迹的 1,301 个样本（每任务一条轨迹的 Event 首／中／末帧，加固定随机 1,024 帧，去重）：148 次 Copy-Paste、637 次反事实查询，可见比例筛选均通过，无重复 caption、非有限输入或原始任务图被改写。2,602 项区域监督变换后有效 2,601 项；本轮真实抽样未发现物体已完全裁掉却仍有效的案例。

额外构造边缘案例发现一处未修复问题：256 图中目标 bbox `[0,100,12,120]`，对应 224 输入从 x=12 开始裁取 212 像素，物体完全出画，但 4 源像素边距仍留下背景。`affine_eskill_batch` 以带边距的 `role_weights` 是否非零判断有效性，导致 Current 仍参与 loss。Subject 复用同一判断，也有这一边界风险。后续修复应以目标本体区域是否保留判断，不能把背景边距当作物体；原生遮挡保留标签的既定语义不变。本轮仅检查，未修改该规则。

另有 20 个样本裁切后保留的原可见像素不足 80%，其中 9 个为反事实查询。这是几何裁切的像素保留率，与第一阶段原生遮挡可见比例是两个指标；当前 80% 阈值仅用于文本替换前。部分出画与完整屏内 bbox 约定兼容，不应误报成原生可见比例筛选失败。

证据：[真实抽样检查](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/final_data_review/checks.json)、[边缘问题复现参数](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/final_data_review/edge_case.json)，同目录保存检查脚本和运行配置。这些是 CPU 数据检查，无新模型训练或闭环效果验证。

## 裁切／旋转有效性修复

上述背景边距误判已修复。GroundingDataset 从已有 bbox 生成未外扩的 `role_core_boxes`，每项仅四个归一化坐标，作为标签附属数据；不进入模型输入。水平翻转同步更新这些框，裁切／旋转复用主图逆向采样网格检查框内区域是否仍存在。仅剩 4 px 背景边距时关闭对应 role，部分出画与原生遮挡继续保留；Current 和 Subject 分别判断。日志有效标记同步为第二阶段后的结果。无需重新生成数据或增加原生渲染。

本轮共 82 项相关回归通过（81 项数据／增强回归，加新增 1 项 loss／梯度及日志检查；最终增强文件 26 项全部通过）。覆盖左右裁切、±5°旋转、角落旋出、镜像后裁切、原生遮挡、部分出画及单项失效；被完全裁掉的 Current loss 和对应 latent 梯度为 0，Subject 梯度仍有效。

复跑相同 40 任务／763 条轨迹／1,301 个样本：148 次 Copy-Paste、637 次反事实，变换后 2,601／2,602 项区域监督有效，与修复前一致；原始任务图未被编辑，无新增异常或误删正常监督。详细 [复核记录](/data1/yuankai/robot-research/workspace/reports/atomicvla_grounding_full_20261117_001/crop_validity_fix/checks.json)。此前边缘复现报告保留原样作为修复依据。
