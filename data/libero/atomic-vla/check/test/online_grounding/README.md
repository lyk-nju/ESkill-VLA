# 单目标在线增强验收 · 2026-11-17

最新批量导出：[200 条增强样本与耗时](export200/README.md) · [40 秒预览](../video/online_grounding_export200.mp4)。数据准备开销约 4.84 → 5.21 ms／条；文件保存与 VAE 编码分别计时，未启动模型训练。

已实现单 Current Query、在线 Copy-Paste、随机 Patch Masking、单目标反事实文本监督。夹爪 Subject 分支已接入；随机文本替换只从可见比例 ≥80% 的实体中选择，没有合格候选就保留原 Event。Protect 同时覆盖当前目标、夹爪各自完整框＋每边 8 源像素，以及夹爪／交互物局部区域。保留原来的 Current decoder；专项入口只使用主相机，反事实不产生动作标签。

[Subject＋Current 验收视频](../video/subject_current_grounding.mp4) · [34 张面板](previews/) · [逐帧统计与索引](checks.json) · [吞吐原始记录](benchmark.json)

视频为九格标签检查：原图／Copy-Paste／Mask 输入、Subject 画布／Target 画布／Protect、两者的真实 FLUX VAE 标签重建及 Target 可见 mask。**蓝框为夹爪、红框为文本目标、绿框为其他实体；Protect 面板白色 patch 永不遮挡。** 共 34 张面板，每张两秒，合计 68 秒。VAE 重建是标签重建，不是训练后模型预测。人工验收待用户查看。

| 检查项 | 结果 |
| --- | --- |
| 样本 | 10 条轨迹，2,198 帧，27 个 Event；原 split 为 7 train／3 val |
| 增强 RGB 与 VAE 画布对应 | Subject／Current 均为 2,198／2,198；均从编辑后、Patch 遮挡前 RGB 构造 |
| 实际 FLUX VAE 编码／重建 | 34 个展示样本 × 两个角色，latent 和重建均有限 |
| 文本目标替换 | 1,079／2,198（49.1%）；逐个回读可见／完整 mask 核验，全部 ≥80% |
| Copy-Paste 尝试／成功 | 1,083／121；成功占全部观察 5.5% |
| 真正移动当次监督目标 | 23／2,198（1.0%） |
| 原生完全不可见目标 | 0 个；真实 Event／Subject 的原生遮挡标签仍保留 |
| 增强造成完全不可见 | 0 个；逐帧确认当前完整框＋8 px 内像素未被 Patch 改动 |
| 有效 Current 监督 | 2,198／2,198（100%） |
| 有效夹爪 Subject 监督 | 2,198／2,198（100%） |
| Protect 像素检查 | 当前框＋8 px、夹爪框＋8 px，全部 2,198 帧保持未遮挡 |

原生部分／完全遮挡都不删除完整 bbox；VAE 画布仍保留实际遮挡者的外观，不还原不可见部分。Copy-Paste 仅使用近乎全可见的可编辑实体。Patch 保护当前文本目标和夹爪各自完整框＋每边 8 源像素，以及夹爪与相邻交互物体／部件的联合局部区域；其他物体不统一保护。移动后无可靠替换模板的关系句退回 `Locate …`，避免参照物移动后文本失真。

Copy-Paste 的主要限制是 726 次尝试没有可编辑实体；另有 136 次实体重叠、91 次出画／位移过小、9 次机器人重叠。失败直接使用原图，没有不断寻找合法位置。源区用邻近中位颜色填充，仍有合成色块；已修复可见 mask 外抗锯齿轮廓残留。当前移动监督覆盖率偏低，不代表已消除位置捷径。

以下吞吐数值来自**此前未保护目标、尚未加入 Subject**的版本，保留为历史记录，本次未重复测吞吐。

| 历史实测项目 | 关闭增强 | 开启增强 |
| --- | ---: | ---: |
| 单进程加载 | 217.4 样本/s | 203.6 样本/s |
| 两 worker 加载 | 218.2 样本/s | 218.8 样本/s |
| 小模型训练平均步耗时 | 111.2 ms | 113.6 ms |
| 其中平均取批／搬运等待 | 4.46 ms | 4.24 ms |

吞吐比较保持主相机 224、VAE 画布 256、单目标、batch 2 和任务文本生成分支一致。每项两次交错开关测量；加载每次预热 64 个样本后计时 256 个样本；训练每次预热 8 步后计时 20 步，每组累计 40 步／80 个有效 Current 目标。训练用随机小骨干、真实 tokenizer 和冻结 FLUX VAE，FP32、AdamW 5e-5、betas=(0.9,0.95)、weight decay=1e-10、梯度裁剪 1、Current／文本 CE 权重各 0.1，`smoke=true`、W&B 关闭，没有加载原 π0.5 权重。

训练步耗时增加约 2.2%；RTX 3090 GPU 1 同时有其他任务，不能将小差异解读为确定性能结论，也不能外推为完整 π0.5 吞吐。数值短测均有限。主回归 53 项通过，另有局部交互保护和增强多 worker／中途恢复两项新增测试通过；增强重放图像与标签逐元素一致。

资产：`/data1/yuankai/robot-research/workspace/reports/atomicvla_online_grounding_20261117_001/`。最新图像检查在 `visible80_target_review/`；历史吞吐在 `accepted_throughput/`，各有包含命令、PID、时间、代码哈希及环境的 `run.json`。`subject_review/` 是未加替换目标可见门槛的 Subject 检查，`visible_target_review/` 是中途 98% 门槛检查，均已由当前 80% 规则取代。`accepted_review/`、`target_protected_review/` 是此前 Current-only 的图像记录。较早试跑保留原结果，不作为新版图像验收。原始 1,693 条 Episode 未修改，也未全量迁移；正式训练和完整 π0.5 吞吐尚未执行。

实现与操作入口见 [训练说明](../../../../../../src/openpi/training/README.md#current-grounding-专项训练) 和 [LIBERO 示例](../../../../../../examples/libero/README.md#current-grounding-数据与训练接口)。

此前 Subject 接入核验（本次未重复训练）：

- 31 项针对性测试通过：图像固定时换文本 Subject 输出逐元素一致、Current 会变化；Subject 梯度进入视觉塔，不进入文本 embedding；Future／Motion／Action 未执行。
- 三步随机小骨干＋真实冻结 FLUX VAE 训练、保存、从第 2 步恢复到第 3 步及策略回读通过，W&B 关闭，非正式 π0.5 训练。模型只接收图像和文本，不输入真实夹爪位置。
- 两进程 CPU DDP：一个 rank 缺 Subject 标签时，所有可训练参数梯度仍与整体 batch 一致，最大误差约 7.45e-9。
- 新 Subject loss 与原未来开合命令 Gripper loss 独立；示例配方 Current／Subject／目标文本 CE 权重各 0.1。
- 夹爪画布仅用现有完整框＋每边 4 源像素，保留框内真实抓取物体上下文，不新增持物身份绑定，也不强求包含完整持物。

技术运行目录：`/data1/yuankai/robot-research/workspace/experiment/current_grounding/subject_technical_20261117_001/`，含配置、checkpoint、日志、`verification.json` 与 `ddp_check.json`。第 148 帧见 [Subject／Current／Protect 对照](previews/008_libero_object_no_noops_s000_r007_f148.jpg)。

本次可见目标筛选核验：25 项针对性测试通过（1 项 worker 测试本次未运行），包含 80% 临界值、部分／完全遮挡及缺少可见比例时不替换、可见候选不要求可搬动、原 Event 标签保留。筛选只读取已有比例，不新增在线分割或重渲染。视频新增 `query=counterfactual / real Event` 与 `target visible`，区分随机替换与真实标签。

用户指出的 `libero_goal_no_noops_s000_r003` 第 57 帧：原先选中的 wine bottle 可见比例为 0%，现被排除；相同随机种子改为查询 cream cheese（94.3%）。见 [修正后第 57 帧](previews/012_libero_goal_no_noops_s000_r003_f057.jpg) 和 [80% 规则验收视频](../video/subject_current_grounding_visible80.mp4)。
