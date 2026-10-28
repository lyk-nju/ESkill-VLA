# target grounding：来源与核查证据

> **历史证据页。** 下文保留 2026-10-27 等调研记录，旧“核心／支持”标签和方法建议属于当时判断。2026-10-28 的当前优先级以 [本方向索引](README.md) 和 [catalog.json](../catalog.json) 为准。

本页合并此前的机制溯源、近期筛选和相关原文复核，保留来源链接、版本、阅读范围与未闭合问题。记录截止 2026-10-27；本次仅整理结构，没有新增论文阅读或复现。

方法取舍见 [docs/target_grounding.md](../../docs/target_grounding.md)，选论文从 [本方向索引](README.md) 开始。以下“本轮”和实验建议指对应调研记录，不自动构成当前架构决定。

[机制溯源](#trace) · [近期筛选](#recent) · [报告原文复核](#review)

<a id="trace"></a>

## 机制溯源

核查日期：2026-10-27。焦点是项目当前定义的目标对象、执行器与局部交互关系；atomic skill 与 target grounding 保持并行。本文是定向机制溯源，不是完整引文史。新作日期与选择性清单见 [近期筛选证据](evidence.md#recent)，原有 12 张卡的保留建议见 [分层与收录理由](../catalog.json)。

最值得改变当前研究设计的发现有三点：ROI 重建已有更近的多头几何接口竞争者 AffordanceVLA；事件对应对象的训练监督已经出现在 Mind-VLA 与 SAM3D-VLA；“表征中能读出目标”必须与“更换目标后动作转向正确对象”分别验证。

### 1. 重建监督的祖先解决什么，ROI 又增加什么

| 关系 | 从哪里发现；双端原文 | 核实后的机制变化 |
| --- | --- | --- |
| MAE → ROSS：明确启发 | ROSS §1 明确以 MAE 为动机；[MAE §3](https://arxiv.org/html/2111.06377v3)、[ROSS §4.1–4.2](https://arxiv.org/html/2410.09575v2) | MAE 随机遮挡后重建像素；ROSS 让 VLM 表征支持视觉 latent 去噪。继承的是重建监督动机，不能说照搬遮挡结构。 |
| ROSS → ReconVLA：明确沿用/改进 | ReconVLA §1 明确写受 reconstructive visual instruction tuning 启发，参考文献对应 Wang 等；[ROSS §4](https://arxiv.org/html/2410.09575v2)、[ReconVLA §3.2](https://arxiv.org/html/2508.10333v1) | 从一般视觉表征保真，转向被操作 gaze 区域的视觉 latent 重建。当前区域监督不等于未来事件完成图。 |
| ROSS → AffordVLA：明确引用、提供替代 | AffordVLA §II-C 引用 [47]，参考文献核对为 ROSS；[ROSS §4](https://arxiv.org/html/2410.09575v2)、[AffordVLA §III-B/C](https://arxiv.org/html/2605.17517v1) | 从重建目标转为任务相关 affordance teacher 特征对齐。其对重建冗余的批评是设计动机，不是证明所有 ROI 重建无效。 |
| ReconVLA → AffordanceVLA：明确引用、相似机制扩展 | AffordanceVLA §2.1 的 reconstructive/gaze-region targets 对应 Song 等 2025；[ReconVLA §3.2](https://arxiv.org/html/2508.10333v1)、[AffordanceVLA §3.2](https://arxiv.org/html/2606.06155v2) | 两者都有对象视觉 latent 目标；后者还区分交互热图和 shape/layout。没有查到“直接以 ReconVLA 实现为基础”的声明，故不画成确定代码继承。 |

**项目判断。** 这里至少有三个可独立操纵的因素：重建还是教师对齐；整图还是任务区域；监督是否保留全图位置及对象—执行器关系。先固定同一 mask、backbone、数据、训练预算，再改变目标，否则“区域更准”与“监督类型更好”会混在一起。

ROI 被裁剪、缩放后，形状好看不保证当前相对位姿准确。建议增加同外观不同位置、同对象不同交互部位的对照。需要局部几何的证据应落到接触点、接近方向和动作误差，不能只用图像重建分数。

### 2. 语义—位置分工：CLIPort → MOO → RoboGround

| 关系 | 从哪里发现；双端原文 | 核实后的机制变化 |
| --- | --- | --- |
| CLIPort → MOO：明确启发 | MOO §2 把 CLIPort [38]、PerAct [12] 列为最接近并称受其启发；[CLIPort §3.1–3.2](https://arxiv.org/html/2109.12098)、[MOO §3.1–3.4](https://arxiv.org/html/2303.00905v2) | CLIPort 组合语义与空间分支；MOO 使用检测得到的对象中心标记，让策略语言保留行为模板、去掉对象名称。 |
| MOO → RoboGround：明确引用、改进对象接口 | RoboGround §2 将 MOO [35] 列为 closely related；[MOO §3](https://arxiv.org/html/2303.00905v2)、[RoboGround §4、附录 A](https://arxiv.org/html/2504.21530v1) | 中心标记扩展为对象/放置区域 mask 与 Grounded Perceiver。原 MOO 实际给策略的是中心点通道，不能沿用后作“粗框”简称而说它直接输入完整框。 |
| RoboGround → ReconVLA：明确引用、替代方案 | ReconVLA §2 的 visual grounding 比较；[RoboGround §4.2、附录 A](https://arxiv.org/html/2504.21530v1)、[ReconVLA §2–3](https://arxiv.org/html/2508.10333v1) | 部署显式 grounding 条件与训练期重建监督是不同接口选择，不构成直接沿用。 |

**项目判断。** MOO 是 atomic skill 与 target grounding 解耦值得补读的早期代表：行为模板与对象绑定分别提供。这里的可迁移设计不是“有一个离散编号”，而是改变对象描述时行为模板如何保持、位置接口如何变化。项目不必照搬它的初始帧标记，但应做等价消融：仅行为文本、行为文本+角色点、共享视觉+隐式 queries。

RoboGround 的 object/placement 角色还提示：place 中被移动对象与接收容器不能挤进同一个“target”字段。若本项目只用 object+effector 两个 ROI，需要单独测试放置目的地关系。

**原文交叉纠正。** ReconVLA related work 把 RoboGround 的 grounding 模型写作 LISA；RoboGround 自己的方法与附录使用 GLaMM。整理实现来源时以后者为准。RoboGround 的 GR-1 实现也去掉未来图像预测头，不能把原 GR-1 所有机制一起归入该策略。

### 3. 目标关系和未来路径应分开比较

| 关系 | 从哪里发现；双端原文 | 核实后的机制变化 |
| --- | --- | --- |
| VoxPoser ↔ 3D-LOTUS++：明确引用、不同方案/批评 | 3D-LOTUS++ §II 引 VoxPoser [3]；[VoxPoser §3.2–3.3](https://arxiv.org/html/2307.05973)、[3D-LOTUS++ §IV-C](https://arxiv.org/html/2410.01345v2) | VoxPoser 以语言生成三维 value maps 后闭环规划；3D-LOTUS++ 以对象角色点云条件化学习的运动策略。后作的精度批评不能推广为所有场景的结论。 |
| RT-Trajectory → HAMSTER：明确引用、扩展训练接口 | HAMSTER §2 credits Gu 等 2023 的轨迹任务接口；[RT-Trajectory §3.1–3.4](https://arxiv.org/html/2311.01977v2)、[HAMSTER §2、§4](https://arxiv.org/html/2502.05485v4) | 从轨迹草图条件执行，扩展到高层 VLM 预测路径与低层策略分工。 |
| HAMSTER → 3D HAMSTER：明确改进 | 3D HAMSTER §II–III 的 2D path 批评与层级架构；[HAMSTER §4](https://arxiv.org/html/2502.05485v4)、[3D HAMSTER §III](https://arxiv.org/html/2606.31329v2) | 由 2D 路径变成预测 metric depth 的三维轨迹接口，避免直接用轨迹像素下方表面深度造成错误路径。 |

3D-LOTUS++ 的 role-aware 表征包含 manipulated object、target、robot、obstacle；它适合检验“角色关系比外观更易组合”的归纳偏置。这里不意味着原子技能一定先于 target grounding：它只是一个可比较的层级系统。

**必要修正。** RT-Trajectory 本来就有编码高度的 2.5D 草图变体。不能写成“HAMSTER 系列首次让轨迹含第三维”。同样，未来路径带入的运动意图比当前 ROI 更多，3D HAMSTER 是信息接口对照，不是纯 current target grounding 的等信息量替换。

### 4. 从场景几何教师到对象几何：两条分支

| 关系 | 从哪里发现；双端原文 | 核实后的机制变化 |
| --- | --- | --- |
| Spatial Forcing → Mind-VLA：明确引用、替代监督目标 | Mind-VLA §II 点名 Spatial Forcing；[Spatial Forcing §2.3](https://arxiv.org/html/2510.12276v2)、[Mind-VLA §III-B–D](https://arxiv.org/html/2608.04633v2) | 场景 VGGT 特征监督与按指令对象构建监督的差异，不是简单“加入 3D”之别。 |
| Spatial Forcing → SAM3D-VLA：明确引用、对象化扩展 | SAM3D-VLA §2 引 [9]；[Spatial Forcing §2.3](https://arxiv.org/html/2510.12276v2)、[SAM3D-VLA §3.1–3.2](https://arxiv.org/html/2607.25912v1) | 后者只在子任务相关对象 tokens 上对齐 SAM3D 教师；部署去除教师处理链。 |
| RKD → RIGID：明确引用、关系蒸馏扩展 | RIGID §2 引 Relational Knowledge Distillation [39]；[RKD §3.2](https://arxiv.org/html/1904.05068v2)、[RIGID §3.1–3.2](https://arxiv.org/html/2610.24705v1) | 从样本间距离/角度关系，转向视图内与跨视图的空间关系分布。不是把 RKD 原损失直接照搬进机器人。 |
| Spatial Forcing ↔ RIGID：明确比较/批评 | RIGID §2–3 明确分析直接对齐；[Spatial Forcing §2.3](https://arxiv.org/html/2510.12276v2)、[RIGID §3、附录 B](https://arxiv.org/html/2610.24705v1) | 点对点教师对齐与关系分布对齐的取舍。RIGID 的“旋转不变”指特征基变换，不保证物理场景旋转泛化。 |
| Mind-VLA ↔ SAM3D-VLA：机制相似、未证实传承 | 独立按关键词检索后读到；[Mind-VLA §III](https://arxiv.org/html/2608.04633v2)、[SAM3D-VLA §3](https://arxiv.org/html/2607.25912v1) | 两者都让训练目标随子任务对象变化；表示目标与动作读取接口不同。不能凭先后日期连成继承链。 |

**项目判断。** “使用事件为每段挑当前对象”已经不是空白。更可区分的问题是：同一对象在不同事件中应保留什么不同的交互信息；事件边界附近的监督是否预先对应下一个目标；对象形状、场景位姿、目标—执行器关系应怎样分配给不同监督。

Mind-VLA 的 canonical mesh 经过中心与尺度归一化。其形状监督可有价值，但不能自动等同当前 metric object–effector relation。SAM3D-VLA 的按视图对象教师则提供较轻的结构对照。二者都值得与项目同数据、同事件分配下比较，避免把外部教师质量当作事件结构的贡献。

**原文交叉纠正。** RIGID §2 将 ReconVLA 归入“显式 3D feature reconstruction”。ReconVLA 原方法的 gaze RGB latent 重建不支持这一概括；本项目笔记不采纳此表述。

### 5. VISTA：视觉敏感性与正确目标控制之间还有一步

| 关系 | 从哪里发现；双端原文 | 核实后的机制变化 |
| --- | --- | --- |
| VCD → VISTA：明确启发、改变用途 | VISTA §3.2 明确受 Leng 等 VCD 启发；[VCD §3.2–3.3](https://arxiv.org/html/2311.16922v1)、[VISTA §3.2](https://arxiv.org/html/2602.05049v1) | 图像加噪后的输出差异，从文本生成的对比解码迁为动作视觉依赖诊断。不是同一个训练算法。 |
| TraceVLA → VISTA：明确沿用局部处理 | VISTA §4.1.1 明确 follows TraceVLA 的活跃轨迹筛选；[TraceVLA §3](https://arxiv.org/html/2412.10345v3)、[VISTA §4](https://arxiv.org/html/2602.05049v1) | TraceVLA 用历史轨迹输入；VISTA 用未来轨迹构建训练期教师/偏好信号，再蒸馏学生。时间可用性不同。 |
| LangGap → Encoded but Not in Control：明确承接诊断 | 后者 §2、§4 明确引用 Hou & Zhao；[LangGap §III-A](https://arxiv.org/html/2603.00592v1)、[Encoded §3–6](https://arxiv.org/html/2610.06235v1) | 同场景不同有效指令的行为评测进一步连接到内部表征探针。 |
| VISTA ↔ Encoded：机制互补，未证实引用 | 本轮把两者放在同一假设下比较；上述原文 | 前者关心视觉条件是否影响动作，后者区分这种敏感性、目标可解码性与正确行为选择。不是已证实的作者承接关系。 |

**项目判断。** target grounding 的最小有效性标准应包括：同场景不动、保留 atomic skill 和目的地，仅换成另一个可见可达目标后，动作是否改选正确对象。另加位置交换、同类多实例、接触部位替换和目的地替换，分开报告目标选择与控制失败。探针与 attention 可定位问题，但单独不足以证明动作因果使用。

FocusVLA 提供相邻的动作端视觉路由思路：其意图是未来各步的局部交互视觉需求，不能直接充当 atomic skill。GroundingPI 则是“感知预训练更好”这条竞争解释。两者的选择性记录见本页近期筛选部分，不在本轮强行追成同一主线。

### 6. 收敛后的阅读与实验顺序

1. 先读 [AffordanceVLA](affordancevla.md) 与 [Mind-VLA](mind_vla.md)，确定现有结构已覆盖的部分。
2. 用 [AffordVLA](affordvla.md)、Spatial Forcing、SAM3D-VLA 作为较简单的监督对照；用 [RIGID](rigid.md) 检查直接对齐是否损害其他能力。
3. 用 [Encoded but Not in Control](encoded_not_control.md) 与 LangGap 设计有效目标替换；这一步应先于堆更多模块。
4. 需要显式角色接口时回到 MOO、RoboGround、3D-LOTUS++；需要未来路径条件时再读 HAMSTER 链。

建议的公平实验不是一次实现全部论文。先固定一个 backbone 和 RoboTwin 示范集，比较 action-only、ROI 重建、对象 teacher 对齐、ROI+交互关系；每个条件都比较固定时间窗与事件分配。先确认收益来自哪类信息，再增加显式 action representation。

### 7. 阅读登记与断点

本轮实读的是上述所列方法/related-work 小节和用于核对引用身份的文献表；没有完成代码复现、全论文逐页精读或重新测量结果。版本明确的节点已固定到链接；CLIPort、MOO、VoxPoser 等原文节点是机制史参照，不声称最早发明。

没有继续深挖的分支：Transporter 的更早空间等变历史、PerAct→C2FARM、GLaMM/SAM3 教师基础模型史、3D HAMSTER 引用的 RoboTracer/RoboBrain 2.5。它们暂不改变当前“对象身份—局部关系—未来路径”的区分。

GraspVLA、RoboPoint、RT-Affordance 沿用上一轮实读笔记作为支持级比较，本轮未伪称新增深溯源。Gaze2Act、Object-Centric Residual RL 等只筛读摘要，不能据此作核心继承结论。更多具体未深读项见本页近期筛选部分。

停止依据：当前 ROI、对象教师、关系蒸馏、显式角色与行为诊断的主要差异已经有双端证据；继续扩大通用 3D/VLA 名录目前不如做公平消融有价值。

### 两报告增量核查

相关原文、阅读范围与纠错见 [分支证据](evidence.md#review)，综合取舍见 [target grounding 方法总结](../../docs/target_grounding.md)。

[SG-VLA：当前空间监督与分阶段梯度介入](sg_vla.md)。

<a id="recent"></a>

## 近期筛选与阅读边界

核查日：2026-10-27。纳入“新作”的标准是 arXiv **v1 首次公开日期在 2026-04-27—2026-10-27（含首尾）**；不是搜索引擎抓取时间、会议年份或最近修订时间。日期按 arXiv abs 的 submission history 核对。这里记录定向检索结果，不声称覆盖全部近期工作。

### 已核实新作

| 工作 | v1 日期；本次阅读版本 | 阅读深度与取舍 |
| --- | --- | --- |
| [AffordVLA](https://arxiv.org/abs/2605.17517) | 2026-05-17；v1 | §II-C、III、V。**核心**：功能区域教师替代重建监督；[卡片](affordvla.md)。 |
| [AffordanceVLA](https://arxiv.org/abs/2606.06155) | 2026-06-04；v2 2026-10-18 | §2、3.1–3.3、附录 C/D/H。**核心**：ROI、交互位置、shape/layout 与动作接口的直接近邻；[卡片](affordancevla.md)。与上一行是不同论文。 |
| [3D HAMSTER](https://arxiv.org/abs/2606.31329) | 2026-06-30；v2 2026-07-01 | §II–III。**支持**：当前区域与未来三维路径的信息量对照，原目录已有卡，不重复新增。 |
| [SAM3D-VLA](https://arxiv.org/abs/2607.25912) | 2026-07-28；v1 | §2、3、4 实际数据处理、6。**支持**：子任务对象 teacher 的简单对照；见 trace 第 4 节。 |
| [Mind-VLA](https://arxiv.org/abs/2608.04633) | 2026-08-05；v2 2026-08-27 | §II、III-B–D、IV-A。**核心**：对象随事件变化的监督与 object queries；[卡片](mind_vla.md)。 |
| [GroundingPI](https://arxiv.org/abs/2609.39601) | 2026-09-30；v2 2026-10-16 | §4.2、5.2.3、5.3。**支持**：专用 grounding 预训练骨干接入同类动作专家；用于排查“数据/预训练收益”。 |
| [Encoded but Not in Control](https://arxiv.org/abs/2610.06235) | 2026-10-05；v1 | §2–6。**核心诊断**：同场景目标替换与可解码信息是否控制行为；[卡片](encoded_not_control.md)。 |
| [RIGID](https://arxiv.org/abs/2610.24705) | 2026-10-17；v1 | §2、3.1–3.2、附录 B。**核心方法对照**：关系对齐与直接特征对齐；[卡片](rigid.md)。 |
| [FocusVLA](https://arxiv.org/abs/2610.31423) | 2026-10-22；v1 | §2.2、3、附录 A。**支持/跨事件方向**：未来逐步局部交互需求控制 global/local 视觉路由，不是对象 grounding 标签。 |

新作中优先级最高的是 AffordanceVLA、Mind-VLA 与行为诊断；AffordVLA/RIGID 用于选择监督方式。不要把全部新作同时列成第一轮实现清单。

### 三个需要单独保留的旁支结论

**SAM3D-VLA。** 它把子任务对象 mask 用于训练时的对象 token 对齐，单独处理不同视图，部署去掉教师链。与 Mind-VLA 的关系目前只能记为机制相近。由此建议：事件监督先和“相同对象标签、不同时间分配”的控制组比，而不只与没有对象监督的模型比。[§3–4](https://arxiv.org/html/2607.25912v1)

**GroundingPI。** 同类动作适配下替换感知骨干，并检查预训练规模/数据配比；但跨骨干排行仍受基础架构与预训练影响。OCR 组的作用在论文里也只是机制假设，不能写成已证实 OCR 导致机器人空间推理。[§4.2、5.3](https://arxiv.org/html/2609.39601v2)

**FocusVLA。** 由夹爪与运动启发式构造弱标签，部署预测连续路由概率。它提示事件结构可帮助动作决定使用哪一路视觉；本项目若声称“事件改善细粒度视觉”应比较这种直接作用于动作端的轻量路线。[§3.2、附录 A](https://arxiv.org/html/2610.31423v1)

### 时间窗口之外与旧文修订

本轮重点溯源的 ROSS、ReconVLA、MOO、HAMSTER、3D-LOTUS++、VCD、TraceVLA、Spatial Forcing、RKD 都是窗口前工作；不会因为近来被引用就标成“2026 新作”。

VISTA visual conditioning 的 v1 是 2026-02-04；[LangGap](https://arxiv.org/abs/2603.00592) 的 v1 是 **2026-02-28**，不能从编号 2603 猜成 3 月。GLaD 的 v1 是 2025-12-10。本轮核查的这些旧文中，没有另列已验证在本窗口内更新的条目；这不代表所有旧文都没有新版。

Mind-VLA、GroundingPI、3D HAMSTER、AffordanceVLA 虽有窗口内 v2，但 v1 也在窗口内，仍属于新作。本文事实按实际阅读版本归属，未把 v2 细节追溯声称为 v1 已有。

### 仅筛选、暂不进主清单

| 候选 | 本轮状态 | 不作为当前核心的理由 |
| --- | --- | --- |
| [Gaze2Act 2605.30282](https://arxiv.org/abs/2605.30282) | 搜索返回原论文摘要，未深读方法/独立核日期 | 人类 gaze 是额外部署输入；适合交互纠错旁支，不是自主 current grounding 的公平替换。 |
| [Object-Centric Residual RL 2606.18953](https://arxiv.org/abs/2606.18953) | 原论文摘要筛选 | 以对象 pose 作残差控制输入，解决 sim-to-real correction；当前阶段不扩到 RL。 |
| [GTA-VLA 2605.13632](https://arxiv.org/abs/2605.13632) | 摘要筛选 | 点/框/轨迹交互是额外人工条件，留作部署纠正接口。 |
| [GeoMem-VLA 2610.29866](https://arxiv.org/abs/2610.29866) | 目录/摘要；整合后仍待核方法 | temporal geometry memory 与本分支相邻，不重复建立本方向核心卡。 |
| [Grounded in Time 2610.04255](https://arxiv.org/abs/2610.04255) | 摘要/候选转交事件方向 | 历史与时间接地优先归事件方向。 |
| [机制解释研究 2610.10178](https://arxiv.org/abs/2610.10178) | 搜索发现，未读方法 | 后续可补 activation patching；当前已有有效目标替换主诊断。 |
| [VLGA 2606.12396](https://arxiv.org/abs/2606.12396) | 摘要筛选 | 自动驾驶与 LiDAR 专家，偏离本项目操作场景。 |
| [硬件约束规划 2610.14106](https://arxiv.org/abs/2610.14106) | 摘要筛选 | 系统规划/IK 路线，尚非所需学习表征核心对照。 |

### 检索可复查性

实际检索覆盖 vision-language-action/robot manipulation 与 grounding、affordance、spatial、reconstruction、object-centric、contact、gaze、visual routing 的组合；加入 after:2026-04-26 before:2026-10-28，并补做 before:2026-08-01 的查询，降低只命中十月论文的偏差。再沿种子 related work、方法引用和参考文献表打开关键原文。

检索时间过滤只是候选筛选工具。核心条目回到 abs 核 v1 日期，再回 HTML 核实际方法；仅摘要条目明确降级。未搜索全语言、全数据库或穷尽全部后续引用。

### 两报告复核补充（2026-10-27）

下列是报告带来的增量，不是重新执行整轮穷尽检索。首发与近期修订分开。

| 工作 | 首发 | 本轮版本 | 时间归类 |
| --- | --- | --- | --- |
| [SG-VLA：当前空间监督与分阶段梯度介入](sg_vla.md) | 2026-03-24 | 2603.22760v1 | 更早工作 |

<a id="review"></a>

## 报告原文复核

核查日期：2026-10-27。对象为 Gemini 与 GPT 的原始调研报告（原稿已移出 docs 备份） 及现有 `ref_papers/target_grounding`。本稿不修改两份报告。按 idea-trace 核对原始方法与版本；本轮仅新增精读 AffordanceWAM、AtVLA、SG-VLA 三篇，其他新名字只做身份与摘要级筛选。

### 结论与纳入范围

两份报告有用的增量主要是 AffordanceWAM 的条件接口消融，以及 SG-VLA 的分阶段空间监督与梯度安排。AffordanceVLA、ReconVLA、FocusVLA 已在本项目上一轮研究中覆盖，不应再以“新发现”重复入库。GPT 报告对关键分数和比较边界较谨慎；Gemini 报告存在把前向注意力限制推成梯度保护、把特定模型结果推成一般设计规律的问题。

| 工作 | 与现有库关系、阅读程度 | 合并建议 |
| --- | --- | --- |
| AffordanceWAM，2609.22332 | 新；读 v3 方法、相关工作及接口消融 | **核心**；主卡放 `action_representation/affordancewam.md`，target grounding 交叉引用。重点为联合 World–Action 学习与可见条件，不应等同于当前场景的 grounding 模块。见[新卡](../action_representation/affordancewam.md)。 |
| SG-VLA，2603.22760 | 新；读 v1 §4–5 的方法、训练及相关分析 | **支持**；放 `target_grounding/sg_vla.md`，为当前空间目标和梯度课程提供对照。见[新卡](sg_vla.md)。 |
| AtVLA，2608.02197 | 新；读固定 v1 方法与附录 A–D | **候选支持**；先保留下面的核查记录，不增加主卡。 |
| AVA-VLA，2511.18960 | 新候选；原始摘要、作者页、CVPR 页面 | **候选支持**；历史视觉状态和注意力重加权支线，尚未完成方法核查。 |
| DeepVision-VLA，2603.15618 | 新候选；原始摘要与作者页 | **候选支持**；深层视觉特征注入支线，尚未核验完整实验。 |
| AffordanceVLA、ReconVLA、FocusVLA | 已有卡或已有筛选/溯源记录 | 只补明确纠错，不重复建卡。 |
| ReWAM | 报告中的 World 表征线索 | 保留待核候选；本次没有完成其方法核查，不据报告直接提升为核心。 |

### 1. 前向注意力不是反向梯度隔离

Gemini 报告从 AffordanceVLA 的单向信息可见性进一步推论其能避免动作回归侵蚀语义能力。报告没有逐字写“stop-gradient”，因此不应把这个词当作它的原话；但若把该推论解释成动作损失不能更新理解表征，机制上不成立。

这是两个独立问题。令理解特征为 `h = fθ(x)`，动作为 `a = gφ(h)`。即使注意力禁止理解 token 读取动作 token，动作损失对 `θ` 仍可沿 `a → h → θ` 求导。只有明确 detach、冻结参数或其他梯度路由，才能据实现判断该路径是否被切断。方向性 attention mask 能限制前向输入泄漏，不能单独证明语义能力被保留；后者还需要训练配置和语义能力评测。

AffordanceVLA v2 的附录 G 明确区分三阶段；后两个阶段联合端到端训练。这与“单向 mask 自然阻断动作梯度”的读法不符。[原文：§3.1.4、附录 G](https://arxiv.org/html/2606.06155v2)。AffordanceWAM 则在任务后训练阶段明确冻结 World model 并停止接口梯度；它是可以讨论实际梯度隔离的例子，不能与前向 mask 混为一项。

建议项目文档对每个方案分开记录四列：**监督从哪里来、前向哪些信息可见、各损失更新哪些模块、部署时哪些量实际可用**。这比统一标记“解耦”更有可操作性。

### 2. AffordanceVLA：已有近邻，报告需要校正版本与单位

本轮核对的是 [v2，2026-10-18](https://arxiv.org/html/2606.06155v2)，不对未检查的旧版本作统一结论。

- Gemini 的“两阶段、100k+ 标注、固定 0.5 权重”把不同阶段与计数单位压成了一套配方。v2 是三阶段；第一阶段 grounding/pointing 记录为 8.8M，第二阶段 A1 为 149K trajectories；affordance:action 权重从第二阶段的 0.5:1 改为后续的 0.15:1（附录 F/G）。记录数与轨迹数不能互换。
- GPT 报告的 LIBERO-Plus **45.5 → 62.9**，以及同表 π0.5 **84.2**，与 v2 Table 5 相符。该结果应按表中基线与设置解释，不能写成对更强 π0.5 的全面优势。

项目已有此文主卡。值得补的是准确训练条件与对应比较边界，不必以更新的版本号重复加一篇。报告若采用旧版本，需要把旧版来源标出来，而不是把不同版本的数据混合引用。

### 3. AffordanceWAM：正确消融，较窄解释

GPT 报告给出的 heatmap 接入动作后的 RoboCasa **69.4 → 63.5** 与 v3 Table 11 一致。该论文区分 scalar affordance 与含 RGB 信息的可视化 heatmap；后者作为辅助 World 监督时有用，但作者设计中不向动作提供这一路径。[原文：§3.2–3.4、§4.3](https://arxiv.org/html/2609.22332v3)。

由此可以提出“条件容量可能放大外观捷径”的项目假设，不能直接证明该机制，更不能推成“高维 grounding 对动作有害”。模型仍使用 RGB，实验改变的是指定 heatmap 通路；同时它利用未来 World 表征，训练和推理信息条件都与仅预测当前 target grounding 不同。

对本项目最有价值的迁移是接口消融：在相同预测器、相同数据下，比较只作辅助监督、动作读取紧凑交互信息、动作读取带 RGB 的重建特征。若目标替换能改变动作而外观替换不能，再结合梯度/计算匹配对照，才能更接近“利用了交互语义”的判断。

### 4. SG-VLA：不能把二元抓取状态改写成精细 affordance

GPT 报告把其监督概括为“grasp affordance”，容易让读者误以为接触点、功能区域或未来抓取位姿。原文对应的是 **binary grasp success/state**；其他辅助目标包括对象 pose、mask、机器人状态。其梯度安排也分阶段：早期阻断辅助 decoder 对 VLM 的梯度，动作预测仍可更新 VLM；之后联合训练；可选 flow action head 阶段完全冻结 VLM。[原文：v1 §4.3、§5.2–5.3](https://arxiv.org/html/2603.22760v1)。

因此“第一阶段冻结整个骨干”与“所有空间监督都需要端到端回传”都不是准确总结。对本项目可迁移的是明确当前空间目标，以及检验不同损失介入时机；不能把其移动操作、RGB-depth 设置的训练经验当作所有 VLA 骨干的固定规则。

### 5. AtVLA、AVA-VLA、DeepVision-VLA：同属视觉改进，不是同一因果机制

**AtVLA** 的报告概括不完整。原文将 registers 保留为可被下游利用的视觉上下文，并以动作不确定性选择局部高分辨率再编码；其完整训练还使用标注 ROI 和 attention grounding 目标，部署有额外 crop 计算。不能仅从摘要“registers 使用原动作目标”推成整套方法无额外标签、无辅助目标、无推理代价。固定 v1 主文与附录对完整损失配方的写法还有差异，复现前需查代码；本轮仅采纳两处一致支持的结论。[原文：Method、附录 A–D](https://arxiv.org/html/2608.02197v1)。

**AVA-VLA** 正式题名为 *AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention*；v1 为 2025-11-24，v4 为 2026-06-14。它不是最近半年首发的新论文。原始摘要与官方页支持历史状态参与视觉重加权，但未在本轮方法级核验其所有 SOTA 表述。[原始记录](https://arxiv.org/abs/2511.18960) · [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_AVA-VLA_Improving_Vision-Language-Action_models_with_Active_Visual_Attention_CVPR_2026_paper.html)。

**DeepVision-VLA** 正式题名为 *Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models*，v1 为 2026-03-16。报告的 9.0% 与 7.5% 可在原始摘要找到，但本轮没有逐任务核验其口径；因此可说“作者报告”，不能据此推出任意设定中的优势。[原始记录](https://arxiv.org/abs/2603.15618)。

这些工作分别涉及输入/特征利用、时间上下文、局部计算分配、监督目标及训练接口。共同支持“视觉接口值得做针对性实验”，不共同证明“先做聚焦再回归动作”这一普遍最优顺序。ReconVLA 的局部重建同样应作为具体方案比较；少量消融不能建立所有几何 token 或其他视觉接口必然干扰动作分布的规律。

### 合并与实验顺序

1. 新增 AffordanceWAM 核心卡，放 action representation 方向，target grounding 交叉引用；新增 SG-VLA 支持卡。两个名字回答不同的接口问题，不合并成一个“更强 grounding”方案。
2. 对 AffordanceVLA 已有卡只补版本、阶段和 mask/gradient 边界；AtVLA、AVA-VLA、DeepVision-VLA 留在候选表。现阶段不扩大核心清单。
3. 先在同骨干和同标签预算下比较当前目标监督与前向条件接口，再考虑未来 World 条件。目标/对象替换、相同目标的外观扰动、辅助头可解码性与实际动作变化应同时观察。
4. 梯度路由另作消融：辅助头单独训练、联合更新、动作分支对相关特征 detach。冻结策略必须控制可训练参数和阶段长度，避免将训练预算变化解释成表示分解收益。

本轮没有运行作者代码，也没有验证标签流水线、所有基准或复现结果。“核心”是本项目阅读优先级，非对可复现性或适用范围的背书。

当前综合取舍见 [target grounding](../../docs/target_grounding.md) 与 [action representation](../../docs/action_representation.md) 方法总结。
