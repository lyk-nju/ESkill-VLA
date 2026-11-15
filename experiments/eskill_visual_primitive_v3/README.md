# ESkill V3：Visual Primitive for Reusable Skill Learning

状态：**仅架构实现，未正式训练、未闭环测评**。实验数值留待真实 checkpoint。当前分支中的 `visual_primitive_version` 为 `v0/v1/v2/v3`，默认 `v0`，旧 checkpoint 不因默认路径发生结构变化。

## 问题和证据

目标不是让 decoder 在训练场景中更漂亮，而是让视觉目标跨样本、场景和 Object–Goal 重绑定保持可靠，并由目标状态决定视觉运动。详见 [旧模型诊断](../grounding_motion_diagnostic/README.md) 与 [130 条新组合](../composition_eskill_step030000/README.md)。

- Milk 在原训练场景 Milk→Basket 能正确 Grounding，但在 LIBERO-GOAL 新场景可能混淆为 Bowl；Chocolate Pudding 在新场景更稳定。这是已有观察，非已量化的整体对象识别率。
- 原有 Current/Future 并行地从 VLM context 读取信息，Current query 没有额外语言条件；Motion 拼接完整融合 VLM context 与 C/F，有强信息旁路。
- Goal-only 注入的两个状态，Motion 只改 0.370 / 0.187 px；两条闭环仍 0/2 到 Plate，继续到 Basket。样本极少，Oracle 是临时反求 token。
- 自然 Motion donor 更换导致读出轨迹改 57.46–65.34 px，5 步 EEF 末端差仅 0.00727–0.03048 mm。说明 Motion→Action 也可能有旁路；V3 **尚未**解决这个问题，后续仍需单独试验。
- ESkill 30k 130 条新组合 48/130（36.9%）对 π0.5 SFT60k 54/130（41.5%），模型和训练配方有差异，不能直接把差值归因于 Grounding。

## 版本和前向

| 版本 | Current | Future | Motion 直接可读的信息 |
| --- | --- | --- | --- |
| v0 | 与 Future 联合解码 | 与 Current 联合解码 | 融合 VLM context + Current/Future + Skill |
| v1 | Current query cross-attend VLM 文本 token，再联合解码 | 旧联合解码 | 同 v0 |
| v2 | Text-conditioned Current 独立先解码 | 文本条件 + Current tokens，再读当前观测，顺序解码 | 同 v0 |
| v3 | 同 v2 | 同 v2 | 先从每臂 Current/Future/Skill 解码 Goal-conditioned Query，再只读 **pre-fusion 主相机视觉 patches** |

v3 的视觉 patches 取自同一 π0.5 视觉塔投影输出，在语言主干融合前截取，已含视觉塔的位置编码；没有增加第二个视觉塔。Current/Future 的位置 token 仍由原 VAE latent 网格监督，Future 仍是**同一个被操作对象在 Event 边界的未来状态**，不是容器检测框或任意时间点。

推理仅需要图像、文本和臂存在性；不输入真实 bbox、未来帧、机器人本体状态、EEF 起点或相机参数。

所有版本保留：Skill-MoE、FLUX VAE 监督、8 个 fixed-frame EEF 2D 点、gripper 辅助监督、Motion→Action per-layer KV、原 π0.5 Flow Matching。V3 **不启用** Flow Matching source bias、coarse proposal、CRAFT 对比学习或额外数据标注。Action-side Patch Masking 可用，但正式配对对照配置中关闭。

## 代码位置

- `src/openpi/models/pi0_config.py`：`visual_primitive_version` 版本开关，默认 v0。
- `src/openpi/models/pi0.py` 与 `model.py`：保留预语言融合的视觉 patch tokens。
- `src/openpi/models/grounding.py`：v1 文本条件 Current，v2/v3 Current→Future。
- `src/openpi/models/motion.py`：v3 先从 C/F/Skill 得到 query，再看主相机视觉空间 patches；禁止直接进入完整融合上下文。
- `src/openpi/models/pi0_eskill.py`：将各版本条件连接起来，不修改 Action/Flow 的计算路径。
- `tests/test_visual_primitive_v3.py`：确认形状、V0 无新增参数、文本影响 Current、Current 影响 Future、C/F 改变 Motion、Action 梯度链路。

## 第一轮公平训练和验收

主配置：[train.json](train.json)。与 [A0 直接联合对照](../eskill_libero40_direct/train.json) 相比，模型结构为 `visual_primitive_version=v3`；2026-11-15 另启用 `motion_terminal_hold=true` 的二维终点保持监督。独立输出目录与 W&B run 名保留其余配方：相同官方 π0.5 基座、双相机、split、归一化、40k 联合训练、bs128、lr5e-5、六项 loss；关闭 augmentation、Masking、重采样。该配置同时改变架构和 Trace 标签，不能将差异全部归因于架构；纯架构对照需设 `motion_terminal_hold=false`。初始随机新增模块与旧 V0 的额外参数容量不同，属于新架构的一部分。

二维终点保持保留最短 +10 保存帧（前两点可跨 Event），此后在当前 Event 最后观察或最短窗口终点中较晚处保持；Episode 末尾也保持最后观察。例如终点在 +20，标签来源为 `5,10,15,20,20,20,20,20`。有效重复点明确标记 `trace_terminal_hold` 并以同等点权重计入 Trace SmoothL1；不补造夹爪命令、不改变 Future Object 目标。规则与缺测处理见 [Motion 监督](../../docs/action_representation.md#event-边界与缺失标签)。这是停止位置的监督假设，是否帮助模型识别 Event 边界及实际 Action 仍需训练、干预与闭环验证。

2026-11-15 技术核验：抽取覆盖六类技能的 10 条真实轨迹，共 1,739 个采样帧；二维零有效点样本从 50 降为 0，新增部分全部标为派生停止目标，原有真实点数量与夹爪标签逐值不变。这不是全量覆盖率或训练效果。另回读一条轨迹保存三个边界预览，最后一帧投影无效的例子仍正确排除全部保持点。CPU 单元与两进程 DDP 检查包含终点梯度、有效分母、旧配方兼容及恢复配置检查；未启动正式训练。

本机检查产物位于 `data/libero/atomic-vla/check/test/motion_terminal_hold/`，软链接到数据盘 `workspace/reports/eskill_motion_terminal_hold_20261115_001/`，包含 `report.json`、保存的 targets NPZ 和三个边界预览 JPG。`audit_training.py --terminal-hold`、`review_motion.py --terminal-hold` 可按相同规则重新核验，图片不是模型预测。

1. **技术检查**：CPU tiny forward/backward、DDP、已有模型测试；需验证视觉 patch 脱离融合上下文的梯度路径。静态可编译不等于通过测试。
2. **Grounding**：相同观察改操作对象文本（Milk/Bowl/Pudding），统计 Current 对应实例的定位准确率与预测图；新场景与原场景分开统计；同一对象改 Place destination，核对 Future 的 Event-end 状态是否变化。
3. **Motion**：同一 RGB、不同 Object/Goal 文本的配对轨迹方向；在同一场景分别替换 Current/Future 预测 tokens，记录 Z 范数、解码 EEF 2D 差、相对目标方向和固定噪声 Action 差。不同对象与目标只作为已预测 latent 的干预，不用真实标签直接替换在线预测。
4. **闭环**：标准 LIBERO 及新组合 benchmark，复用完全相同 13×10 初态/seed、20Hz、320 步上限、10步预测/5步执行。报告总体成功数、Milk/Pudding 单独成功数、Goal/Object 组、旧目标偏好与正确抓取；对照需有同训练配方 v0 的真实训练结果。
5. **进入 v4 的条件**：如果 Grounding 和 Motion 明显按目标变化、但 Action 几乎不变，则才去处理 Motion→Action（例如 biased FM）；若 Grounding 本身仍错，先解决跨场景定位，而不是叠加控制模块。

注意：以上分析、超参数和可行性均是**待验证假设**，不是已取得 V3 效果。正式训练必须在实际服务器上核对资源、GPU 和 W&B 后按仓库 AGENTS.md 启动。
