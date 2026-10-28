# Event organization：来源与核查证据

> **历史证据页。** 下文保留 2026-10-27 等调研记录，旧“核心／支持”标签和方法建议属于当时判断。2026-10-28 的当前优先级以 [本方向索引](README.md) 和 [catalog.json](../catalog.json) 为准。

本页合并此前的机制溯源、近期筛选和相关原文复核，保留来源链接、版本、阅读范围与未闭合问题。记录截止 2026-10-27；本次仅整理结构，没有新增论文阅读或复现。

方法取舍见 [docs/event_organization.md](../../docs/event_organization.md)，选论文从 [本方向索引](README.md) 开始。以下“本轮”和实验建议指对应调研记录，不自动构成当前架构决定。

[机制溯源](#trace) · [近期筛选](#recent) · [报告原文复核](#review)

<a id="trace"></a>

## 机制溯源

核查日期：2026-10-27。按 idea-trace 工作流执行。研究对象是**边界依据、区间粒度、监督用途和在线切换**；沿用项目术语 atomic skill、target grounding、action representation。本文件不把未来子目标改称当前 target grounding，也不把可变执行长度自动解释成语义 Event。

本轮的主发现不是一条统一的“Event 方法发展史”，而是四条目标不同、部分交叉的路线：机器人关键状态压缩；语义技能的进入/结束条件；视觉子目标及进度；通用视频事件分段。新增在线自适应 chunk 路线用于检验语义边界是否必要。

### 如何阅读证据

下表的箭头表示**从后作查到前作**。关系类型分为：明确引用；明确沿用/改进；替代/批评；机制相似。仅有引用不升级为继承。小节号优先于网页行号，链接指向实际读取的原文版本。以下概括均为短释义，不是长段引文。未复现代码。

### A. 机器人关键状态 → 可组合技能：AtomSkill 的边界不是从零提出

| 追踪边 | 关系与后作证据 | 前作原文核查 | 改变了什么、没有证明什么 |
| --- | --- | --- | --- |
| AtomSkill → DeCo | **明确沿用**。[AtomSkill v2 §III-A](https://arxiv.org/html/2512.18368v2) 在夹爪变化分段处写 following [30]，参考条目为 DeCo | [DeCo v2 §III-A/B](https://arxiv.org/html/2505.00527v2) 区分 half/full interaction，围绕完整交互构造任务 | AtomSkill 增加语义标注和表示对齐；不能因此说其边界算法本身新颖，也不能把两篇的粒度视作完全相同 |
| DeCo → Q-attention | **明确沿用**。DeCo §III-B 指明 keyframe discovery [43]，依据夹爪变化或近零关节速度 | [Q-attention v2 §IV-D](https://arxiv.org/html/2105.14829v2) 明确给出两条件的 OR，并把中间状态到下一关键帧加入示范数据 | 早期作用是增加示范利用率、提供关键位姿监督；DeCo 将这些状态组织为可组合 atomic tasks，不是只压缩控制频率 |
| DeCo → VoxPoser | **明确沿用/改造，跨方向支线**。DeCo §III-D 说明 cost map adapted from VoxPoser [13] | 本轮未独立重读 VoxPoser 方法；只确认后作引用上下文 | 这是技能衔接空间可达性的来源，不能算作边界来源；交给 target grounding 主线继续核验，当前不画成已完成双端证据边 |

**由此修正的研究判断。** “在夹爪变化处分段”是一种已有工程策略。项目的关键问题应是：完整交互、单次抓放转换、语义动词三种粒度，哪个最有助于跨任务共享？例如接近—抓取—搬运可拥有多个物理里程碑，却仍对应较粗的语义动作。将每个里程碑都变成 atomic skill 类别，可能把轨迹细节重新混入语义表示。

**不能略去的接口。** DeCo §III-C/D 有目标位姿监测与技能间空间衔接；AtomSkill §III-C 有预测 keypose 的切换。我们可只借鉴它们的数据组织，但实验不能把完整方法的闭环成绩全部归于离线分段。

### B. 同样是 waypoint，优化目标却不同：StaKe → AWE → PerAct

| 追踪边 | 关系与后作证据 | 前作原文核查 | 关键差异 |
| --- | --- | --- | --- |
| StaKe → AWE | **明确引用与方案对比**。[StaKe §II-C](https://arxiv.org/html/2606.26801v1) 将 AWE [32] 列为缩短决策时域的 waypoint 路线；没有写沿用其 DP 算法 | [AWE §4](https://arxiv.org/html/2307.14326v1) 以轨迹重构误差约束最少 waypoint 数 | StaKe 使用夹爪派生标签训练辅助头；AWE 改变策略要预测的 waypoint。两者监督位置和运行接口不同 |
| AWE → PerAct | **明确批评/替代**。AWE §2 引 PerAct [4]，讨论零速度/夹爪启发式的适用限制 | [PerAct v2 §3.2](https://arxiv.org/html/2209.05451v2) 将每个观测转成下一关键帧预测，§3.4 给出监督学习接口 | AWE 不依赖这些事件触发点，而按运动可线性重构性选择；几何压缩良好不等于语义一致 |
| StaKe ↔ AtomSkill | **机制相似，无本轮核实的直接继承边**。StaKe §III-B/C 与 AtomSkill §III-A 均利用夹爪线索 | 两端上述方法均已读 | StaKe 二元阶段与未来关节目标不等于 AtomSkill 的语义动作表示；不能因共享传感器线索写成同一方法 |

PerAct 的 §3.2 对静止点的表述包含夹爪状态稳定约束；Q-attention §IV-D 则明确说明两类候选的 OR。这里只比较论文文本，不在没有代码核查时强行统一精确实现。DeCo §II 对 PerAct 的任务特性概述也不能取代 PerAct 自己的多任务设置。

**项目含义。** 建议将 AWE 类几何压缩作为廉价强对照：等量 waypoint 下，若语义事件没有更好的重组执行与跨对象稳定性，便不能只因标签更可读而宣称更好的知识组织。StaKe 是另一维度的对照——相同原始数据上，简单阶段辅助损失已经能带来多少收益。

### C. 稀疏视觉目标有两条分叉，不能都称作事件检测

| 追踪边 | 关系与后作证据 | 前作原文核查 | 机制变化 |
| --- | --- | --- | --- |
| StructVLA → ForeAct | **明确引用与差异化**。[StructVLA §2](https://arxiv.org/html/2603.12553v1) 引 ForeAct [44]，对比 subtask-anchored 与 kinematic anchors | [ForeAct §3.2–3.4](https://arxiv.org/html/2602.12322v1)：在子任务内采样未来图像，VLM 监控/重规划，VLA 接受未来图像输入 | StructVLA 的稀疏结构预训练再转动作预测，不能等同部署时的独立视觉规划器 |
| ForeAct → SuSIE | **明确引用，未声称直接移植**。ForeAct §2 Visual Prediction for Robot Control 引 SuSIE [7] | [SuSIE §4.1/4.2](https://arxiv.org/html/2310.10639v1)：未来区间内随机采目标，定期刷新图像目标 | 同为图像子目标，目标时间采样、语言接口与刷新规则不同；ForeAct 是可连接的旁支，不强写成算法继承 |
| StageWAM → TaKSIE | **明确引用与替代方案比较**。[StageWAM Related Work / Hierarchical and Stage-Guided Manipulation](https://arxiv.org/html/2608.10780v3) 将 TaKSIE 置于显式子目标路线 | [TaKSIE §3.3](https://arxiv.org/html/2410.11013v2)：进度关键点、递归记忆、相似度/超时更新 | StageWAM 预测下一阶段 latent，并内部条件化 WAM；不是将 TaKSIE 的目标图直接编码后复用其检测器 |
| TaKSIE → SuSIE | **明确改进/批评**。TaKSIE §2.2 称 SuSIE 为最接近方法，并指出固定间隔难适应进度 | SuSIE §4.1/4.2 的训练未来范围和固定刷新均实际核对 | 改变的是选目标、保持模式与在线刷新三个环节，不能把收益只记到边界选择 |

**有证据的两层链**是 StructVLA→ForeAct→SuSIE，以及 StageWAM→TaKSIE→SuSIE。它们共同指向“只给下一个有用状态”的归纳偏置，但不共同证明一个唯一的语义边界定义。

StageWAM 的 Method / Stage-pair construction 使用自身的 V-JEPA2 特征变化启发式。本轮没有核实它从 UVD 或 TaKSIE 直接继承边界算法；“同属视觉变化”只能记为机制相似。它与 StaKe 的关系也是 Related Work 引用，不是辅助头路线的直接延长。StageWAM Related Work 对多篇方法的概括较粗，必须回到各原文接口判断。

### D. WALL-WM 引到的是通用事件感知，不是机器人技能分段器

| 追踪边 | 类型及后作位置 | 前作核查位置 | 证据边界 |
| --- | --- | --- | --- |
| WALL-WM → GEBD | **明确概念引用**。[WALL-WM v2 §4.3](https://arxiv.org/html/2606.01955v2) 在 atomic manipulation actions 的时间分段描述后引 [66] | [GEBD v5 §3.1/3.2](https://arxiv.org/html/2101.10511v5)：taxonomy-free、人类感知的边界、相对视频整体“下一层”粒度、多标注者 | 引文确实存在，但未发现 WALL-WM 声称部署 GEBD 检测器；不能把引用改写为算法沿用 |
| GEBD → Tversky & Zacks, Event Perception | **明确概念/标注原则引用，双端深度不足**。GEBD §1、§3.1 引 [51] 作为认知实验标注依据 | [Oxford Handbook 第6章，2013，pp.83–94](https://doi.org/10.1093/oxfordhb/9780195376746.013.0006) 本轮核对出版方摘要；全文未读 | 可确认章节身份和事件分块主题，尚不能逐项验证 GEBD 的标注指南究竟继承哪些具体实验 |

GEBD 的边界还可来自主体、环境或镜头变化；这与“机器人正在执行的行为发生变化”并不相同。它最有价值的贡献是提醒：粒度存在合法分歧，应保存多个标注者或多层解释，不能把一个时间点当成普遍真值。

WALL-WM §4.3 的 task/subtask/action/segment 是嵌套数据组织，§4.4 的平衡采样又是另一因素。当前文献证据不足以说明其边界自动化算法与 GEBD 的具体实现相同。对项目而言，应分别消融边界来源、描述粒度和采样权重；不把三者当一个 Event 开关。

认知文献的更早候选是 Zacks 等 2007 的 *Event Perception: A Mind–Brain Perspective*。已核对 [PubMed 摘要和 DOI](https://pubmed.ncbi.nlm.nih.gov/17338600/)，但 PMC 全文入口返回验证页；未绕过访问限制。本轮不宣称完整追溯到心理学最早起源。

### E. 新近旁支：可变执行时长也可以来自预测可信度

[Knowing When to Stop v3 §II-B](https://arxiv.org/html/2609.00908v3) 明确引 AAC [18]；[AAC v2 §4](https://arxiv.org/html/2604.04161v2) 通过多次动作采样估计熵。前者 §III-B/C 则利用动作到 VLM 的 cross-attention 熵及高位平台来截短执行，减少额外采样。这是**明确引用与替代设计**，不是语义 Event 分段的后继。

与 AtomSkill 的 keypose 到达判断、TaKSIE 的目标相似度、Hi Robot 的周期更新形成一组有用控制：边界可由语义完成、可达状态、预测可信度或外部时钟决定。若我们把它们混成一个“事件结束”，就无法解释增益来自语义还是闭环频率。

### 阅读账本与收敛范围

| 原始论文标识 / 阅读版本 | 实际阅读范围 | 本轮用途 |
| --- | --- | --- |
| AtomSkill 2512.18368v2；2026-07-02 | §II、§III-A–C、参考[30] | 发现 DeCo 沿用边；其余表示细节由 atomic skill 方向处理 |
| DeCo 2505.00527v2；2026-02-15 | §II、§III-A–D、参考[43] | 物理交互、关键帧、切换与衔接分开 |
| Q-attention 2105.14829v2；2022-02-04 | §II、§IV-D | 本轮找到的早期机器人关键帧代表，不主张首创 |
| AWE 2307.14326v1；2023-07-26 | §2、§4 | 几何重构准则与启发式分叉 |
| PerAct 2209.05451v2；2022-11-11 | §3.2–3.4、§4.1 | 核验 AWE 所讨论的被引方法 |
| StructVLA 2603.12553v1；2026-03-13 | §2、§3.1–3.3、相关参考 | 结构目标与部署接口 |
| ForeAct 2602.12322v1；2026-02-12 | §2、§3.1–3.5 | 子任务内图像目标、在线监控 |
| SuSIE 2310.10639v1；2023-10-16 | §2、§4.1/4.2 | 稀疏目标路线的早期代表 |
| TaKSIE 2410.11013v2；2024-12-17 | §2.2、§3.3 | 自适应进度与固定刷新对照 |
| StaKe 2606.26801v1；2026-06-25 | §II-C、§III、§V 限制 | 训练辅助监督路线 |
| StageWAM 2608.10780v3；2026-08-14 | Related Work、Method、Inference | latent 阶段与局部控制的两种时域 |
| WALL-WM 2606.01955v2；2026-09-06 | §2、§4.3/4.4、参考[66] | 数据组织；其动作世界模型交由主方向 |
| GEBD 2101.10511v5；2021-08-19 | §1–3、相关参考[51]/[54] | 粒度歧义与控制边界的区别 |
| AAC 2604.04161v2；2026-04-10 | §2.2、§4 | 执行长度来自不确定性的对照 |
| Knowing When to Stop 2609.00908v3；2026-09-16 | §II-B、§III、§IV 对照 | 新近无需训练的执行长度对照 |

本轮已解释主要机制分叉，停止继续堆积通用 diffusion、VLM 或 RL 背景引用。尚未闭合的点是：GEBD→认知标注原则的全文双端核验；WALL-WM 自动时间边界的可复现实现；DeCo→VoxPoser 的空间衔接接口；StageWAM 边界规则的代码与窗口重叠审计。对这些点，当前结论是待查而非没有前驱。

### 对研究计划的收缩建议

主阅读入口保留六篇现有核心：AtomSkill、UVD、RDD、StructVLA、StaKe、StageWAM；加 DeCo 与 AWE 补全机制来源，加 Knowing When to Stop 提供执行时长对照。其余按统一收录清单 保留支持或历史卡，避免每篇都占 Phase 1 的优先级。

最先实施的证伪实验应固定 backbone、数据与监督预算：**同一边界不同监督用途；同一监督不同边界；同一训练不同执行长度**。三组分别回答 Event 数据组织、语义因素分工和闭环更新的问题。近期候选与日期核查见 [近期筛选证据](evidence.md#recent)。

### 两报告增量核查

相关原文、阅读范围与纠错见 [分支证据](evidence.md#review)，综合取舍见 [Event 方法总结](../../docs/event_organization.md)。

[ATOM-Bench：单项能力与组合表现的评测对照](atom_bench.md)、[依赖覆盖：正确 Event 阶段不等于充分任务信息](diagnosing_compositional_generalization.md)、[LOTUS：视觉分段与持续扩展技能库](lotus_continual.md)。

<a id="recent"></a>

## 近期筛选与阅读边界

检索及核验日期：2026-10-27。时间窗按 **arXiv 首次提交 v1 日期**判断，包含两个端点；近期修订另列。搜索引擎的 Published、抓取日期、会议年份与标题页当前版本日期均不替代 v1。

### 检索范围与取舍

已执行公开网络检索并打开候选 arXiv 原页。查询覆盖 `event manipulation`、`stage manipulation`、`subgoal robot`、`temporal segmentation robot`、`hierarchical world model`、`adaptive action chunking`、`VLA stage-aware` 等，同义词查询包含 `after:2026-04-27 before:2026-10-28`，以及不带日期约束的相关引用追踪；后者用于避免漏掉被错误排序的记录。另沿 StageWAM、StaKe、WALL-WM、StructVLA 的参考文献和新候选 related work 反查。

主要保留标准是改变事件边界、阶段目标、技能切换或历史依赖评价的机制。仅共享 stage/event 关键词、只改扩散求解阶段或只做一般软件 agent 的条目不进入机器人 Event 主入口。以下是已核验候选的选择记录，不是对 arXiv 全站的穷尽扫描。

### 半年内首次发布：核心与支持候选

| 候选与 arXiv 原始记录 | v1 日期 | 查阅版本 / 更新日期 | 阅读深度与取舍 |
| --- | --- | --- | --- |
| [WALL-WM: Carving World Action Modeling at the Event Joints](https://arxiv.org/abs/2606.01955) | 2026-06-01 | v2 / 2026-09-06 | **核心，已有跨方向主卡**。本轮读 §4.3/4.4 与引用上下文；事件区间、层次描述、采样分别考虑，不另建重复卡 |
| [Improving Vision-Language-Action Model Fine-Tuning with Structured Stage and Keyframe Supervision（StaKe）](https://arxiv.org/abs/2606.26801) | 2026-06-25 | v1 / 同日 | **核心，已有卡**。§II-C、§III：最贴近当前轻量辅助监督的对照 |
| [StageWAM: Joint-Embedding Stage Prediction for World-Action Models in Robot Manipulation](https://arxiv.org/abs/2608.10780) | 2026-08-11 | v3 / 2026-08-14 | **核心，已有卡**。已读方法与推理；下一阶段 latent 是独立目标，不能改称当前 target grounding |
| [Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs](https://arxiv.org/abs/2609.00908) | 2026-09-01 | v3 / 2026-09-16 | **新增核心对照卡**。已读 §II-B、§III、§IV：检验可变执行时长是否需要语义 Event，而不是将它收作语义分段方法 |
| [τ0-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation](https://arxiv.org/abs/2608.16885) | 2026-08-17 | v1 / 同日 | **支持，偏后续上下文/高层决策**。读 §II、§III 及机制概述；世界模型比较候选 subtask 的后果，新增的是选择过程与计算预算，不是离线边界算法 |
| [PILOT: Guiding Long-Horizon World-Model Planning for Robot Manipulation with Visual Subgoals and Policy Priors](https://arxiv.org/abs/2610.14335) | 2026-10-09 | v1 / 同日 | **支持，阶段目标的反例对照**。已读 §II–III：从预测视频均匀选子目标，再用 policy prior 初始化 CEM；并不需要语义 Event 检测 |
| [Grounded in Time: A Multi-Source Dataset and Benchmark for Temporal Grounding in Robotic Manipulation（GiT）](https://arxiv.org/abs/2610.04255) | 2026-10-03 | v1 / 同日 | **支持，Phase 3 诊断优先**。已读 §III-A/C/D、§IV 指标：相同当前场景下由不同过去事件决定正确目标；区别于仅延长观察窗口 |
| [Adaptive Action Chunking via Multi-Chunk Q Value Estimation（ACH）](https://arxiv.org/abs/2605.10044) | 2026-05-11 | v1 / 同日 | **支持，方法待深读**。原始摘要核查：离线到在线 RL 按状态选 chunk 长度；训练条件与 BC/VLA 方案不同，暂不新增主卡 |
| [ValueFormer: Stage-Aware Value Labels for Online Failure Detection and VLA Post-Training](https://arxiv.org/abs/2608.02958) | 2026-08-03 | v2 / 2026-10-14 | **支持，摘要筛查**。阶段化失败/进度标签更接近 critic 与恢复评价，尚未完成标签构造原文核对，不作为 Event 已验证机制 |
| [FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models](https://arxiv.org/abs/2610.31423) | 2026-10-22 | v1 / 同日 | **跨方向支持**。本分支核摘要；target grounding 分支已读方法，见 [其筛选记录](../target_grounding/evidence.md#recent)。不重复建卡 |

ValueFormer 搜索结果仍可显示旧题名 *A Causal Transformer Value Function with Stage-Aware Labels for Semi-Autonomous Vision-Language-Action Policies*；本表使用 arXiv 当前 v2 标题。基于 arXiv 基础 ID 去重，不当两篇论文。

### 为什么这些新工作值得比较，而不全部进入核心清单

**Knowing When to Stop 与 StaKe 解的是不同问题。** 前者仅改变执行前缀长度，后者改变训练监督。在同一模型上单独开关两者，能检验收益是否分别来自表征与闭环频率。前者使用模型内部统计，不能把注意力熵当成真实接触或成功判据，也不能把多模态可行动作与错误混为一谈。

**PILOT 是对“必须语义划分才能做长期控制”的有用反例候选。** 它在 §III-B 明确采用均匀帧索引抽取生成子目标。这里的时间层次仍可能有效，但没有证明边界语义。其策略先验只是搜索初始化，不能将 CEM 带来的收益记作纯子目标监督收益。该论文也明确区分长动作规划时域和多任务组合，不能把四个局部任务成绩直接扩展为组合泛化。

**τ0-VLA 改变的是决定下一子任务的过程。** 它比较候选语言子任务的预测后果，并维护执行记忆；§III 的每步重推子任务允许保持或变更当前命令。因而不能根据“hierarchical”就假定依赖一个独立 Event 结束检测器。其大规模异构训练与搜索预算也决定了它更适合作为后续系统参照，而非 Phase 1 的最小实现。

**GiT 提供更有针对性的可证伪评价。** 当前图像相似、历史不同的反事实配对，可以检测模型是否真正利用过去交互来决定目标。§III-D 的标签流水线是机器人状态规则断点加 VLM 识别和边界细化；不是完全无监督发现。可借鉴配对测试原则，先在 RoboTwin 构建小规模诊断，不必立即迁移全套数据。

### 关键词命中但不纳入 Event 主清单

| 候选 | v1 / 当前版本 | 原始核查与排除理由 |
| --- | --- | --- |
| [StairVLA: Stage-Aware Hierarchical Action Generation for Vision-Language-Action Models](https://arxiv.org/abs/2610.07756) | 2026-10-06 / v2，2026-10-07 | 摘要中 stage 指去噪阶段；高层部分去噪轨迹加高频局部细化。对时序系统有参考价值，但不是语义事件组织 |
| [Toward Real-Time VLAs: Stage-Aware Two-Step Flow Denoising and System-Level Evaluation](https://arxiv.org/abs/2609.39822) | 2026-09-30 / v2，2026-10-03 | 摘要中 stage 同样是 flow 积分阶段，主要研究延迟与执行系统；留在部署效率方向 |

### 旧作近期修订，以及恰好落在窗口之前的工作

| 论文 | 原始 v1 | 查阅版本日期 | 本轮处理 |
| --- | --- | --- | --- |
| [AtomSkill / Learning Semantic Atomic Skills for Multi-Task Robotic Manipulation](https://arxiv.org/abs/2512.18368) | **2025-12-20** | v2，2026-07-02 | 核心种子，但不是半年内新作 |
| [H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model](https://arxiv.org/abs/2602.11291) | **2026-02-11** | v3，2026-09-29 | 近期修订；本轮仅元数据/摘要核查，不能因 9 月版本当新路线 |
| [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208) | **2026-04-03** | v3，2026-10-12 | 多时间尺度 latent planning 支持候选，非时间窗内首次发布 |
| [Event-Centric World Modeling with Memory-Augmented Retrieval for Embodied Decision-Making](https://arxiv.org/abs/2604.07392) | **2026-04-08** | v3，2026-07-08 | 时间不满足；仅元数据与摘要筛查，标题相似不等于 WALL-WM 同类事件算法 |
| [Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](https://arxiv.org/abs/2604.04161) | **2026-04-05** | v2，2026-04-10 | 窗口之前，但属于 Knowing When to Stop 的明确被引前驱，仍纳入 trace |
| [StructVLA / Beyond Dense Futures](https://arxiv.org/abs/2603.12553) | **2026-03-13** | v1，同日 | 继续作为核心种子；不计近期新作 |
| [ForeAct](https://arxiv.org/abs/2602.12322) | **2026-02-12** | v1，同日 | 为 StructVLA 的被引旁支；不计近期新作 |
| [DeCo](https://arxiv.org/abs/2505.00527) | **2025-05-01** | v2，2026-02-15 | 必须补读的边界来源，价值来自机制关系而非新近日期 |

对旧作修订，本轮只核实版本和日期，没有做 v1→新版逐段 diff，因此不推断某个新实验或模块恰好在哪个版本加入。

### 下一步优先级

Phase 1 先用现有 StaKe/StageWAM 与新增 Knowing When to Stop 做训练监督和执行长度的正交对照；DeCo/AWE 用于明确分段粒度与目标。GiT 的历史反事实配对及 τ0-VLA 的候选后果评估留给上下文阶段。PILOT、ACH、ValueFormer 保留支持记录，只有当项目要引入在线规划、RL 或失败评价器时再升级精读优先级。

机制证据链见 [机制溯源证据](evidence.md#trace)；逐卡裁剪建议见 [分层与收录理由](../catalog.json)。

### 两报告复核补充（2026-10-27）

下列是报告带来的增量，不是重新执行整轮穷尽检索。首发与近期修订分开。

| 工作 | 首发 | 本轮版本 | 时间归类 |
| --- | --- | --- | --- |
| [ATOM-Bench：单项能力与组合表现的评测对照](atom_bench.md) | 2026-06-15 | 2606.16826v1 | 半年内首发 |
| [依赖覆盖：正确 Event 阶段不等于充分任务信息](diagnosing_compositional_generalization.md) | 2026-07-31 | 2607.29687v2 | 半年内首发 |
| [LOTUS：视觉分段与持续扩展技能库](lotus_continual.md) | 2023-11-03 | 2311.02058v4 | 更早工作 |

<a id="review"></a>

## 报告原文复核

复核日期：2026-10-27。范围：GPT 原报告的 Event、实验和整体建议，以及 Gemini 原报告中的相应主张（两份原稿已移出 docs 备份）；与现有 `ref_papers` 卡片及思想溯源交叉检查。本轮只新增阅读三篇原始论文，未检查代码或复现实验。

结论：两份报告不要求改变 Phase 1/2/3。最值得加入的是 **ATOM-Bench 的分项能力评测**和 **Diagnosing Compositional Generalization 的依赖覆盖与阶段信息反证**。二者应作为 `event_organization` 下的评测对照支线，不能列为 Event 分段算法。LIBERO-PRO 仅保留为辅助评测参照，不新增核心卡片。

### 已经覆盖，不必再开一轮文献扩张

| 报告建议 | 现有覆盖 | 本轮判断 |
| --- | --- | --- |
| 物理/脚本 Event；夹爪变化只是候选边界 | AtomSkill、DeCo、UVD、RDD；Event 溯源 | 保留。先核对实际数据字段与可复现的标签生成规则，不默认 HDF5 含阶段、对象状态和掩码 |
| TACO 弱顺序监督、CompILE 潜在分段 | atomic skill 溯源已有两者 | 后备机制；不把 Phase 1 改成无监督分段项目 |
| 固定窗口、关键帧、语义事件的公平比较 | 历史综合记录的边界比较提议；Event 溯源末节 | 保留，并同时固定执行更新频率、真实时间跨度及监督预算 |
| 未见技能—对象组合、未见事件组合、外观鲁棒性分开 | 总溯源与 atomic skill 溯源已有拆分 | 合理；双臂并行、序列重排、实体重组也应分别报告 |
| 表征交换、目标敏感性、成功路径多样性 | 总溯源实验 2–5 | 已覆盖；不能用探针/聚类结果代替闭环控制证据 |
| oracle teacher 与部署预测之间的差距 | 总溯源实验 4 | 已覆盖；新增评测不要把未来信息作为在线输入 |
| 组合时前一事件终态与下一事件初态不匹配 | 总溯源实验 6 | 已覆盖，但需要转成明确的配对诊断 |

报告以 AtomSkill 为 Event 主入口可以理解，但知识框架应继续保留 StructVLA、StaKe、StageWAM，以及 WALL-WM 相关溯源；无需把现有结构收缩回一篇代表作。已有关系中，AtomSkill 沿用 DeCo 分段；WALL-WM 对 GEBD 是概念引用，不能写成采用该检测器。

### 有用的新证据及应降级的表述

**ATOM-Bench。** 报告所述“30 原子任务、24 组合任务”与原文一致；Atomic Score 和 Compositional Failure Share 的精确定义见[卡片](atom_bench.md)。它增加的是一套可执行的评测语言。需要把“识别到底是哪种失败”改成“帮助定位需要进一步检查的差距”：评分点、不同初态、轨迹长度和依赖约束都会影响差值，不能由一个比值完成因果归因。

**Diagnosing Compositional Generalization。** 报告关于三类分布变化的方向正确。更重要的增量是：正确 Event 阶段与充分任务信息是两件事，见[卡片](diagnosing_compositional_generalization.md)。本项目可以让 atomic skill 表示保持语义粒度，同时保留全局指令或相关约束；不应把“可复用”解释成“动作只能依赖当前技能名”。该论文的覆盖规律尚不能直接作为 π0.5/RoboTwin 的保证。

**LIBERO-PRO。** 两份报告均未直接提及该名字，本轮将其作为补充参照。原文 §4 按对象属性、初态、指令/任务及环境构造可执行扰动，§5 评测 OpenVLA、π0、π0.5。适合借鉴单因素配对扰动；不能把它当作纯粹的组合泛化拆分，也不能把摘要中的极端退化概括成所有模型、所有扰动均为零。只记为候选，不增加主实验套件。

**Gemini 的 Event 因果措辞。** “夹爪变化是接触发生或解除的强确定性信号，天然对应技能起止”过强。空夹、滑移、闭夹推物以及保持闭夹的连续操作都使这两种事件不等价；应写成候选边界启发式，并用对象/接触状态及人工抽样验证。“固定窗口必然产生无语义碎片”也应改成待比较的假设。LOTUS 的技能发现描述不能因与 QueST 同段讨论就被归为相同的固定窗口机制。

**总体取舍。** GPT 报告优先验证机制、保留 Phase 2 人类视频与 Phase 3 固定参数示范 Context 的划分是合理的。B0–B7 可作消融目录，但不必一次训练全部组合；先完成数据和配对评测，再决定哪个表示接口值得加。Gemini 的“首次”“彻底”“工程定律”不宜用于本项目的新颖性或可行性判断；这些表述需逐项证据，不能由跨论文头条数字推出。

### 可加入 Phase 1 的小规模诊断

以下是本项目的实验建议，未运行；不是对原论文实验的复述，也不要求迁移到新 benchmark。

1. **先建立拆分清单。** 从现有 RoboTwin 任务挑选一个可重组任务族，列出技能、对象、目标、顺序、约束及初态范围；仅纳入物理可行组合。训练/测试共享单因素取值，记录每个取值的频率与关键因素对覆盖。测试组合在训练、验证和模板生成中均不得泄漏。任务数量由可用脚本决定，不预设完整笛卡尔积。
2. **记录原子能力和完整组合。** 对同一模型、相同对象与目标，评测单事件执行和保留组合；预先冻结完成谓词及少量过程检查点。主报告仍是整任务成功率，附逐事件成功、过程进度、约束违反和样本分母。必要时才计算 ATOM 风格差距；它是 RoboTwin 自定义诊断，不能标为 ATOM-Bench 官方成绩。
3. **做配对交接。** 从示范事件起点与前一事件真实执行终态分别启动同一后续控制器。先只纳入满足前置条件的真实终态；前置条件不满足单独计数。若重置为示范起点后恢复成功，优先检查状态分布与接口，而不是直接声称没有组合知识。此实验用于定位，不扩展为失败恢复主线。
4. **加一个跨事件依赖反例。** 在技能和当前场景相同的两个可行任务中，仅改变会约束当前动作的后续目标。比较完整指令与当前子指令；先用 oracle 阶段做诊断，再检查在线预测。若当前子指令本身不足以唯一确定合法动作，不能把失败全部记到 atomic skill 上。
5. **只加两个轻量扰动对照。** 同一意图的语言改写应保持结果；同一场景的合法目标切换应改变对象绑定。必要时再补目标小幅移位。背景变化、目标变化、序列变化分列，避免一个综合 OOD 分数遮蔽原因。

预算控制：首先固定训练示范数、优化更新数、动作标签、辅助标签来源、采样权重、执行频率与每个任务的最大物理时长。在同一保留集合和初态种子上比较基线与单一改动；报告每个任务的试验数和不确定性，不因少量 rollout 得到百分比就声称稳定规律。若后续比较数据覆盖方案，应共享保留测试集合，并匹配边际频率及总数据量；不能把不同方案各自的不同 OOD 集合直接当成公平对照。

### 阅读记录与来源关系

| 原始来源 | 本轮实际阅读范围 | 关系与边界 |
| --- | --- | --- |
| [ATOM-Bench，2606.16826v1](https://arxiv.org/html/2606.16826v1) | §2、§3.1–3.3、§4.1–4.4、§6 | 明确引用 LIBERO-PRO 作为相关评测；新增卡片是评测支线，未查官方评分代码 |
| [Diagnosing Compositional Generalization，2607.29687v2](https://arxiv.org/html/2607.29687v2) | §3–6，附录 B.2/B.3、C、D、E | 数据覆盖/阶段条件的对照机制；不主张它与 ATOM-Bench 有继承关系；未复核全部证明、图表或代码 |
| [LIBERO-PRO，2510.03827v2](https://arxiv.org/html/2510.03827v2) | §2.2–2.4、§4、§5；[官方仓库说明](https://github.com/Zxy-MLlab/LIBERO-PRO) | 补充扰动设计参照；未核验每个扰动实现与官方 checkpoint |

现有语料对照：历史综合记录、Event 与 atomic skill 的机制溯源、AtomSkill 与 QueST 卡片、总目录。本轮没有重新打开这些已核验论文，也没有改动用户报告。


当前综合取舍见 [Event 方法总结](../../docs/event_organization.md)；研究目标与路线见 [idea.md](../../docs/idea.md)。
