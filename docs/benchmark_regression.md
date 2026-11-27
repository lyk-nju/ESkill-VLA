# 历史测评回归 · 2026-11-27

复核对象是 11 月 27 日交接包，以及本机发布的 CRAFT Appendix A 逐回合表。以下成功数均带原分母。技术样本用于检查推理签名和输出文件。

## 可以逐对或逐回合复算的结果

| Motion 20k，四类目标干预各 48 对 | 轨迹规划（视觉主干冻结） | 轨迹规划（视觉主干参与训练） | 边界优先（视觉主干冻结） |
| --- | ---: | ---: | ---: |
| 轨迹沿目标变化方向移动 | 183/192 | 168/192 | 180/192 |
| 两侧轨迹均更接近各自目标区域 | 140/192 | 118/192 | 134/192 |
| 两侧 Current 均在 10 源像素内 | 158/192 | 124/192 | 158/192 |
| 仅移动干扰物，轨迹平均位移的中位数，10 对 | 1.900 px | 1.194 px | 2.709 px |

每版 202 条逐对记录齐全，顺序和封存清单完全一致。轨迹看名义槽的最后四点，区域距离为点到闭矩形的最近距离；224×224 中的轨迹位移和 256×256 中的 Current 误差分别计算。全表五个条件的计数、中位数和历史 `comparison.json` 逐项核对通过。Boundary 的 Start、End 漂移由逐对坐标重算。End swap 对 B／C／D 的正向响应分别为 44/48、39/48、45/48，由历史逐对 swap 投影值重汇总；包中没有替换后的逐槽原始 trace。推理适配器按旧代码固定 A 侧技能，仅把 B 的 End latent 换到 A。

| CRAFT，seed 7 | 原生三任务 | 新组合五任务 |
| --- | ---: | ---: |
| 公开 π₀.₅-LIBERO | 148/150 | 118/250 |
| 自训 π₀.₅ 60k | 149/150 | 158/250 |
| SLA | 148/150 | 178/250 |
| 旧 ESkill 30k | 141/150 | 147/250 |

来源为本机 `libero-composition/results/2026-11-12-craft-appendix-a/per_episode.csv`。四模型共 1,600 条，按 `model + uid` 检查唯一性、初态指纹、任务组、seed 和官方 success 字段，结果与发布表一致。CRAFT 的 5 个新组合及其 250 个任务－初态条件已包含在九组合的 450 个主回合中。九组合单列 40 个对照回合。上表保留历史结果，250 个重合条件只计一次。

## 有历史清单或发布表的结果

| 协议 | 历史覆盖或成绩 | 本次核验层级 |
| --- | --- | --- |
| Stage 1 A/B/C/D/E | 102 / 522 / 525 / 158 / 0 条。A–D 共 351 张图、1,307 条查询 | 原始 records、Native test、KLAL 转换清单按 ID、任务、Current／Subject 框及 RGB 摘要对齐。E 为无合格样本，命中率未定义。 |
| Stage 1 20k Current | 旧 VAE 512、BBox 405、BBox+GAC 479、VAE+GAC 412，分母均为 1,307 | 原 JSON 按任务计数重新求和。四路 Subject 均为 1,307/1,307。包中没有原版逐预测 CSV。 |
| Stage 1 10k Current | 同序 299、295、260、234，分母均为 1,307 | 这是同一清单上的独立 10k 对照。Subject 同序为 1,307、1,307、1,302、1,302。 |
| 非 LoRA KLAL 10k，完整增强 | Current 1,220/1,307，Subject 976/1,307 | 历史运行报告。 |
| Native Grounding E0 | Check 95/272，正式 509/1,307 | 锁文件标明 sentence、L12/H3、mass、local3、10 源像素，由开发数据选出。 |
| E1 路由 | Oracle Phrase 8/136、Oracle Stage 8/136、预测阶段 4/136 | 训练来源的 68 个 Check 父轨迹上的开发诊断。 |
| Frozen representation Probe | Check 候选 196/197、中心 199/204。正式候选 1,175/1,306、中心 1,025/1,307 | 锁定 `projector_linear_bilinear`。一条查询缺少可核验的候选几何，保留其自由定位结果。LIBERO-90 另有独立类别和配对清单。 |
| Planner 原生 20k | 175 条验证示范、4,695 观察、1,203 锚点。6,195 个真实点、3,319 个派生保持点、61 个跨 Event 真实点；真实点误差 7.05 px | 本机 175 份 Episode 元数据与封存 SHA 一致。逐帧预测未随包，保留旧报告。 |
| LIBERO 200／Plus 280／Plus 全量 | 200 / 280 / 10,030 个预定回合 | 记录为三个闭环协议。本次未启动正式回合。 |

## 本机资产

| 所需资产 | 本机核验 | 影响及原资产定位 |
| --- | --- | --- |
| LIBERO40 数据与 split | 数据 index、固定 split 的 SHA 一致，1,518 train / 175 val。开发 Fit 136 和 Check 68 个父轨迹互斥，均在 train | Native 开发集的 408 个不同源帧均已从本机 Episode 恢复，逐张像素 SHA 与原清单一致，覆盖 816 查询。 |
| A–D 正式 RGB、202 对反事实 RGB | 正式 A–D 的 351 张、Planner 配对的 203 个独特输入在映射目录均缺失，两组可能共享图像 | 原机 `experiments/stage1_visual_localization/run_001/{pick,place}` 和 `current_subject_spatial_retest/run_003/inputs`；KLAL 转换 PNG 按原始像素哈希命名。文件清单可用 `relocate` 导出。 |
| 三版正式 Planner 20k | 三份均缺失 | 依次为 `experiments/motion_planner/run_{001,002}/train`、`experiments/motion_planner_boundary/run_001/train`。每个 checkpoint 指纹在对应 `reference/motion/.../run.json`。 |
| BBox、BBox+GAC、旧 VAE、VAE+GAC 正式权重 | 10k 和 20k 共八份均未在映射路径找到 | 从 `reference/visual/stage1_visual_localization/run_{001,002,003}/results/summary.json → models → identity` 找 checkpoint、源码和 SHA。VAE+GAC 还需要 `experiments/current_grounding_vae_gac/run_001/source`，交接包内的快照无法满足其全部源文件指纹。 |
| 非 LoRA KLAL 10k | 本机有权重，`c482a25b…` 为本机记录值。checkpoint 中的基座要求为 `bbb94502…`，本机基座为 `ce648b1f…`；tokenizer 一致 | 需原服务器与该训练匹配的 `ckpt/pi05/base/pytorch_bf16/model.safetensors`。正式加载已实际触发内容检查。 |
| Native／Probe | 已恢复开发 RGB；原始捕获、Probe 特征、候选几何、拟合权重和原机配方未随包提供 | 查 `experiments/native_grounding/run_001/{config.json,captures}` 和 `experiments/vlm_capability_diagnosis/frozen_representation_probe/run_001/{config.json,features,models}`；用 `selection.json` 中的 Fit/Check ID 与模型 SHA 验收。独立 Pick 的 `plan.json`、`manifest_lock.json` 与相应 scene / features 必须取自同一运行。 |
| Plus 280 抽样名单 | 包内有每回合重置 RNG 的协议、原计划 SHA，未附完整初始名单 | 找原机 `experiments/eskill_libero40_mirror/run_001/stage2/evaluation_sample_10k_20k/run_001/{job-plan.json,selected-variants.json}`，摘要分别以 `6651646f`、`513d6d2d` 开头。 |

本机结果位于数据盘 `workspace/reports/handoff/`。`motion202-regression.json` 和 `visual-regression.json` 是复算表，`assets-20261127-all-steps.json` 是机器可读资产清单，`native-dev-001/mapping.json` 记录恢复路径。

## 本机技术推理

| 版本与设备 | 完成与输出 |
| --- | --- |
| BBox 32 步、KLAL 32 步，GPU 3 | 各完成 1/1 条开发查询，分别输出双框和注意力预测。结果目录为 `bbox-32-native-dev-001`、`klal-32-native-dev-001`。KLAL 技术权重另在 CPU 跑通。 |
| Native E0，GPU 3 | 完成 2/2 条开发查询。注意力图形状为 `[2,6,18,8,256]`，图和 mass 均为有限值。12 个方法－查询组合中 11 个有效，第 2 条的 destination 短语无效。捕获与分母在 `native-technical-001/technical/dev/shard_000.{json,npz}`。 |
| 旧 VAE 8 步，CPU | 按原源码和 FLUX 解码完成 1/1 条。阈值 32 下 Current、Subject 均不可读，两路各 0/1 个有效中心。`vae-8-cpu-summary.json` 保存原报告。GPU 3 当时可用显存不足，GPU 尝试失败。 |
| 冻结／联合 Planner 4 步，CPU | 两个独立历史加载器在同一张有哈希的开发图上各输出一条有限的 `[1,8,2]` 轨迹。小模型画布为 32×32，正式画布为 224×224。Boundary 的 End swap 使用单元测试和历史逐对输出核对，缺少正式权重做现场推理。 |

以上开发查询均排除在正式分母外。闭环本次新增 0 回合。与该迁移相关的 60 项本仓测试、24 项历史源码测试均通过。另有两项 RoboTwin 导出测试在本机缺少匹配的 LeRobot／Accelerate 环境。
