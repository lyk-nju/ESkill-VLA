# atomic skill：来源与核查证据

> **历史证据页。** 下文保留 2026-10-27 等调研记录，旧“核心／支持”标签和方法建议属于当时判断。2026-10-28 的当前优先级以 [本方向索引](README.md) 和 [catalog.json](../catalog.json) 为准。

本轮补齐当前方法已引用工作的独立卡片；新卡各自记录来源与核查范围，未改写以下历史记录。

本页合并此前的机制溯源、近期筛选和相关原文复核，保留来源链接、版本、阅读范围与未闭合问题。记录截止 2026-10-27；本次仅整理结构，没有新增论文阅读或复现。

方法取舍见 [docs/atomic_skill.md](../../docs/atomic_skill.md)，选论文从 [本方向索引](README.md) 开始。以下“本轮”和实验建议指对应调研记录，不自动构成当前架构决定。

[机制溯源](#trace) · [近期筛选](#recent) · [报告原文复核](#review)

<a id="trace"></a>

## 机制溯源

调研日期：2026-10-27。沿已有 LISA、LCSD、SkillDiffuser、PRISE、QueST、CompILE 和跨方向 AtomSkill 继续阅读引用上下文与被引原文。本轮使用 idea-trace；只核对相关章节，未复现代码，也不声称覆盖完整引用史。近期工作见 [近期筛选证据](evidence.md#recent)，保留/降级决策见 [分层与收录理由](../catalog.json)。

### 当前问题与新增判断

本项目希望 atomic skill 表达可跨对象、场景和运动实现复用的行为类别，target grounding 表达当前交互对象和空间约束，action representation 保留合理运动差异。文献中的“skill”却可能是离散动作片段、状态可区分的 option、完整操作—实体组合，或一个可执行子策略；必须先比较接口，再讨论是否支持这三种信息分工。

本轮最重要的新增对照是 [CRAFT](craft.md)：它直接检验跨执行交换表示后，动作专家是否仍能完成对应行为。不过它将 skill 定义为 operation–entity，而非本项目期望的对象无关行为类别。因此它是必须比较的竞争解释，不能作为本项目语义拆分已经得到证明的依据。

另外两个修正影响方案选择：[LADS](lads.md) 通过梯度和预测接口区分 latent plan 与高层预测，但仍联合优化；[DASL](dasl.md) 的 diffusion 只在训练期规整表示。不能把“解耦”“双过程”直接读成先训练技能、再冻结技能，或默认部署阶段必须额外生成轨迹。

### 关系标记与阅读范围

- **明确引用**：后作确实提到前作，但不足以认定实现继承。
- **明确沿用或改进**：后作明确说明采用/建立在前作之上，并核对两端对应机制。
- **替代或批评**：后作明确比较另一方案或提出不足；批评归属作者，不扩大为独立验证的普遍结论。
- **相似机制**：本次比较发现类似接口，未据此建立引用或历史影响关系。

表中箭头表示阅读方向或被比较关系，不统一代表技术传承。“已读”指下列章节，不等于逐页精读全文。版本日期是实际阅读版本；近期新作的判据另以 arXiv v1 日期计算。

### 分支一：可区分行为、语言语义与技能预测并非同一个目标

| 关系与类型 | 从后作哪处发现 | 两端原文及机制核查 | 对本项目的意义 |
| --- | --- | --- | --- |
| VIC → DIAYN；明确引用、替代或改进 | DIAYN §2、§4.1 Question 4 直接比较 Gregor 等的 intrinsic control | [VIC v1 §2–3](https://arxiv.org/html/1611.07507v1) 用 option 与终态的互信息组织可控行为；[DIAYN v6 §2–3、§4.1](https://arxiv.org/html/1802.06070v6) 比较固定先验、逐状态辨别与最大熵策略。VIC §3 自身也允许固定先验，因此不是“VIC 不能固定先验” | 状态可分性提供行为覆盖，不自动提供语言类别，也不保证对象不变性 |
| DIAYN → LISA；明确引用、替代或批评 | LISA §2.3 将 DIAYN/DADS 与语言条件、多技能轨迹进行比较 | [DIAYN §3 Algorithm 1](https://arxiv.org/html/1802.06070v6) 给出无任务奖励训练；[LISA v3 §2.3、§3.1–3.2](https://arxiv.org/html/2203.00054v3) 用语言和历史预测 VQ 技能，低层依据状态和技能模仿动作 | 不能把无监督技能发现的“无奖励”搬成 VLA 的“无需动作示范”；此处不声称 LISA 直接沿用了 DIAYN 算法 |
| LISA → SkillDiffuser；明确沿用或改进 | SkillDiffuser §2.2 明确说跟随 LISA 提取 subskills | [LISA §3](https://arxiv.org/html/2203.00054v3) 的离散技能接口，对照 [SkillDiffuser v3 §2.2、§4.2–4.4](https://arxiv.org/html/2312.11598v3) 的技能条件状态 diffusion 和逆动力学 | 离散技能条件化多模态运动生成已有先例；创新点不能只写此接口 |
| LISA → LCSD；明确引用、替代或批评 | LCSD Related Work 的 “Skill Discovery via mutual information” 批评单一编码器的语言联系及稳定性 | [LISA §3.2 的 “Avoiding language reconstruction”](https://arxiv.org/html/2203.00054v3) 对照 [LCSD v1 Approach / Mutual Information Skill Learning](https://arxiv.org/html/2402.17511v1) 的轨迹技能集合恢复语言嵌入 | 这是监督目标的实质分歧。恢复完整语言可能鼓励对象信息，是否有害需以控制和组合测试判断 |
| LISA → LADS；明确沿用或改进 | LADS §2 明确 “builds on LISA”，随后解释分离 latent plan 学习和高层预测 | [LISA §3.1–3.2](https://arxiv.org/html/2203.00054v3) 对照 [LADS §4.1–4.5，PDF pp.4–6](https://proceedings.iclr.cc/paper_files/paper/2025/file/e7c786024ca718f2487712bfe9f51030-Paper-Conference.pdf)：未来轨迹技能编码、高层分类预测、stop-gradient 与联合训练 | 预测接口/梯度解耦与训练阶段拆分应分别设计、分别做消融 |
| SPiRL → LADS；明确引用，机制启发 | LADS 引言引用 task-agnostic latent skill learning，并列举 Pertsch 等 | [SPiRL v1 §3.2–3.3](https://arxiv.org/html/2010.11944v1) 区分动作片段编码与状态条件 prior；[LADS §1、§4.1–4.3](https://proceedings.iclr.cc/paper_files/paper/2025/file/e7c786024ca718f2487712bfe9f51030-Paper-Conference.pdf) 分开示范技能编码与在线语言条件预测 | 这是接口启发证据，不把 LADS 写成 SPiRL 的离散化复刻 |
| LCSD → LADS；明确引用 | LADS 引言用 LCSD 讨论 codebook collapse | [LCSD Approach / Mutual Information Skill Learning](https://arxiv.org/html/2402.17511v1) 的语言恢复与重初始化，对照 [LADS §1、§4.3–4.4](https://proceedings.iclr.cc/paper_files/paper/2025/file/e7c786024ca718f2487712bfe9f51030-Paper-Conference.pdf) 的离散预测及轨迹级语言对齐 | 两者都关注稳定性，不等于用了同一个损失或训练路径 |
| LISA / LCSD / SkillDiffuser / LADS → DASL；明确引用 | DASL §2.2 对四篇工作逐项定位；§2.3 另讨论生成模型 | 前端机制见以上章节；[DASL v2 §2.2–2.3、§3.1–3.5](https://arxiv.org/html/2607.10625v2) 使用慢速技能和快速动作接口，训练期 latent diffusion 正则 | 原文引用足以说明研究邻接，不足以证明每个模块直接继承自其中一篇 |
| SkillDiffuser ↔ DASL；相似机制、不同部署接口 | 此行为本轮方法对读，不增补未核实的具体继承主张 | [SkillDiffuser §4.3–4.4](https://arxiv.org/html/2312.11598v3) 在规划/执行中使用 diffusion；[DASL §3.3–3.5](https://arxiv.org/html/2607.10625v2) 部署移除训练 diffusion | 将“辅助表示学习”与“推理必经生成模块”列为直接对照，控制延迟与模型容量 |

**项目推论。** 可以把不变性、可预测性和可执行性分开检验：同一个行为身份在不同对象上是否稳定；仅凭部署可见信息能否选中它；与当前 target grounding 组合后是否产生可行控制。某个对象探针准确率低，既不能单独证明不变性成功，也不能解释动作失败。

### 分支二：分段监督的多跳来源

| 关系与类型 | 从后作哪处发现 | 两端原文及机制核查 | 对本项目的意义 |
| --- | --- | --- | --- |
| CTC → TACO；明确沿用或改进 | TACO §2 说明扩展 CTC，§3.3 / §4 分别写对齐和控制 | [Graves 等 2006 §3.1，PDF pp.2–3](https://www.cs.toronto.edu/~graves/icml_2006.pdf) 对多种路径求和；[TACO v2 §3.3、§4.1–4.2](https://arxiv.org/html/1803.01840v2) 将动作/stop 概率纳入 sketch 对齐 | 已知顺序、未知边界是独立于精确事件标签的可行监督等级 |
| TACO → CompILE；明确引用、替代或批评 | CompILE §3 将 TACO 的给定 task sketch 与无监督分段作对比 | [TACO §3.3、§4.2](https://arxiv.org/html/1803.01840v2) 需要有序标签；[CompILE v2 §2.2、§3、§4.4](https://arxiv.org/html/1812.01483v2) 联合推断分段和身份，但分段数有超参数且存在位置绑定例子 | 学到了边界仍不等于学到了跨位置的行为语义；应分开测量边界与身份质量 |
| DeCo → AtomSkill；**仅后端引用已核，前端未由本分支复核** | AtomSkill §III-A “Following [30]”；参考文献 [30] 是 DeCo 2505.00527 | [AtomSkill v2 §III-A / reference 30](https://arxiv.org/html/2512.18368v2) 与 [DeCo 原文入口](https://arxiv.org/abs/2505.00527)；已交 Event 分支核对前端 | 这是保留的证据断点，不计作已完成的两端传承边 |

**跨方向整合更新：** Event 分支已完成 DeCo 两端方法核验，并继续追到 Q-attention；见 [Event 证据链 A](../event_organization/evidence.md#trace)。上表保留 atomic skill 分支自己的阅读范围，整合后的总库不再把该关系视为未闭合。

**项目推论。** 先保留 exact boundary、ordered sketch、learned segmentation 三档。特别增加“连续两次相同技能、交互对象不同”的测试：重复技能码、重复标签折叠与真实 Event 身份并不天然一一对应。TACO 原文的对齐规则不应未经修改就当作所有此类事件的 oracle。

### 分支三：动作压缩与语义对齐的交界

| 关系与类型 | 从后作哪处发现 | 两端原文及机制核查 | 对本项目的意义 |
| --- | --- | --- | --- |
| Subword BPE → PRISE；明确沿用或改进 | PRISE §2 明确引用 Gage 1994、Sennrich 等 2016；§3.1 将 BPE 用于动作码序列 | [Sennrich 等 v5 §3.2](https://arxiv.org/html/1508.07909v5) 按频繁相邻符号合并；[PRISE v3 §2–3.1](https://arxiv.org/html/2402.10450v3) 先学状态/动作单步码，再压缩序列 | 可变时长来源是序列频率，不等于交互语义边界。Gage 原件未读，停止上溯而不宣称最早源头已核 |
| PRISE → QueST；明确引用、替代或批评 | QueST §2.3 对 PRISE 的动态词表与 BPE 路线作比较 | [PRISE §3.1–3.4](https://arxiv.org/html/2402.10450v3) 对照 [QueST v3 §2.3、§3.2、§4.1–4.3](https://arxiv.org/html/2407.15840v3) 的 actions-only 多 token 抽象 | 两者都叫 skill，不代表一个 token 对应一个稳定语义；QueST 不是 BPE 的自然下一代 |
| LISA / LCSD / SkillDiffuser → AtomSkill；明确引用、替代或批评 | AtomSkill 引言与 §II-B 将这些列为语言引导分支 | 各前作方法见上；[AtomSkill §II-B、§III-B](https://arxiv.org/html/2512.18368v2) 引入语义和时序对比 | 批评是 AtomSkill 作者的分类，不代表本轮独立证明三者在所有数据上都失败 |
| QueST / STAR → AtomSkill；明确引用、替代或批评 | AtomSkill 引言与 §II-B 将它们列为 action-only 抽象 | [QueST §4](https://arxiv.org/html/2407.15840v3)、[STAR v3 §3.2](https://arxiv.org/html/2506.03863v3) 与 [AtomSkill §III-B](https://arxiv.org/html/2512.18368v2) 对读；AtomSkill 语义正对还要求同一量化码，不只是同一 skill 标签 | 不能把 AtomSkill 总结为“将全部同动词轨迹压到一个点”；它仍保留动作子结构 |

**项目推论。** 语义技能、动作 token 和语义约束的动作 token 都应留在主对照中。父分支已核的 X-Tokenizer 将语义约束施加到分层量化首层，进一步提醒我们：显式分开三个表示是否必要，需要优于共享且结构化的 tokenizer 才能成立，而非靠变量名称成立。

### 分支四：组合可以通过检索、生成数据或表示监督实现

| 关系与类型 | 从后作哪处发现 | 两端原文及机制核查 | 对本项目的意义 |
| --- | --- | --- | --- |
| X-ICM → Decompose and Recompose；明确引用、替代或批评 | 后者 §2 / §3 将动态示范检索与其 skill coverage 方案比较 | [X-ICM v3 §4.1–4.3](https://arxiv.org/html/2505.15660v3) 用动力学/语言相似度选例、场景和关键动作进行上下文预测；[后者 v1 §3.1–3.3](https://arxiv.org/html/2605.01448v1) 用显式技能与覆盖库补足动态检索 | 不要将 X-ICM 写成“只看关节、不含对象”；公平比较检索量与上下文预算 |
| TLI → ACT-VLA；明确沿用或改进 | ACT-VLA §II-B、§III-A 明确采用 Li 2025 的 text latent interpolation | [TLI v5 §3.2–3.3](https://arxiv.org/html/2505.03500v5) 插入/插值隐藏文本状态；[ACT-VLA §III-A–B](https://arxiv.org/html/2607.00351v1) 将其用于离线合成动作数据并训练 | latent 插值的组合能力与可解释原子类别不同；ACT-VLA 包含环境 rollout 和成功筛选 |
| ACT-VLA → CRAFT；明确引用、替代方案 | CRAFT §2.1 比较基于合成演示的数据路径 | [ACT-VLA §III](https://arxiv.org/html/2607.00351v1) 与 [CRAFT v1 §2.1、§4.1–4.4](https://arxiv.org/html/2610.00524v1) 对照 | 一个增加合成轨迹，一个将已有执行的监督转给反事实技能表示；应显式报告额外交互和计算 |
| SkillVLA → CRAFT；明确引用，不是串行组合继承 | CRAFT §2.1 列举 structured skill composition | [SkillVLA v1 §III、§IV-A–C](https://arxiv.org/html/2603.03836v1) 的双臂技能分解/协作，对照 [CRAFT §3–4](https://arxiv.org/html/2610.00524v1) 的固定操作顺序实体组合 | 双臂并行、子任务串行、实体重组是不同泛化维度；不宜汇总为一个“组合泛化”分数 |

**项目推论。** 对同一批训练数据至少区分三类增益来源：数据补齐、检索覆盖、表示监督。CRAFT 尤其提示一个比 embedding 可视化更直接的检验：固定可行状态及动作专家，只替换行为表示，然后测动作误差与闭环行为。若表示交换无效，即使语义聚类漂亮，也不足以支持可组合接口。

### 核心卡与建议阅读顺序

1. 先将 [CRAFT](craft.md) 与 LCSD、AtomSkill 对读，明确“语义等价”究竟是动词、操作—实体还是语义标签内的动作子类。
2. 再将 LISA、[LADS](lads.md)、[DASL](dasl.md) 对读，决定在线预测、训练梯度及部署路径；不要因论文都写 decoupling 而合并方案。
3. 用 PRISE、QueST 与主分支 X-Tokenizer 设置强动作抽象对照；原子语义是否提供额外价值由实验回答。
4. 将 [Decompose and Recompose](decompose_recompose.md)、[ACT-VLA](act_vla.md) 作为组合能力的外部竞争路线，控制检索和新增数据成本。
5. Event 监督问题由 CompILE、TACO、AtomSkill/DeCo 分支联合决定。

### 证据台账、停止点与未读部分

| 论文 | 实读版本 / 出版源 | 本轮相关原文范围 |
| --- | --- | --- |
| VIC | 1611.07507v1，2016-11-22 | §2–3 |
| DIAYN | 1802.06070v6，2018-10-09 | §2–3，§4.1 Question 4 |
| LISA | 2203.00054v3，2022-12-06 | §2.2–2.3，§3.1–3.2 |
| LCSD | 2402.17511v1，2024-02-27 | Related Work，Approach，MI Skill Learning |
| SkillDiffuser | 2312.11598v3，2024-03-28 | §2.2–2.3，§4.2–4.4 |
| LADS | ICLR 2025 proceedings PDF | §1–2，§4.1–4.5；未核 arXiv 标识 |
| DASL | 2607.10625v2，2026-09-11 | §2.2–2.3，§3.1–3.5 |
| CTC | ICML 2006 原论文 | §3.1，PDF pp.2–3 |
| TACO | 1803.01840v2，2018-08-10 | §2，§3.3，§4.1–4.2 |
| CompILE | 1812.01483v2，2019-05-14 | §2.2、§3；§4.4 承接首轮 |
| Subword BPE | 1508.07909v5，2016-06-10 | §3.2 |
| PRISE | 2402.10450v3，2024-06-06 | §2、§3.1–3.4 |
| QueST | 2407.15840v3，2024-09-11 | §2.3、§3.2、§4 |
| AtomSkill | 2512.18368v2，2026-07-02 | 引言，§II-B，§III-A–C，参考文献 [30] |
| X-ICM | 2505.15660v3，2025-10-19 | §4.1–4.3 |
| Decompose and Recompose | 2605.01448v1，2026-05-02 | §2、§3.1–3.3 |
| TLI | 2505.03500v5，2026-05-01 | §3.2–3.3，§4 设定 |
| ACT-VLA | 2607.00351v1，2026-07-01 | §II-B，§III-A–B |
| CRAFT | 2610.00524v1，2026-09-30 | §2–4.4，§5.1，Appendix B.1 |
| SkillVLA | 2603.03836v1，2026-03-04 | §III、§IV-A–C |
| Semantic Handoff | 2607.06256v2，2026-07-15 | §II–III 开头；诊断支持材料 |
| SkillNet | ICML 2026 官方摘要与作者方法页 | 非全文方法核对；论文 PDF 获取失败 |
| SPiRL / STAR / VQ-BeT / SAILOR 2022 / LGSD | 首轮原卡所列版本 | 本轮继承已读记录；不重复计为新精读 |

停止继续上溯 CTC 的语音识别历史、Gage 的通用压缩历史，以及 VQ/Transformer 等通用组件：当前足以解释本项目需要的机制差别。VIC 是本轮核到的早期代表，不声称最早发明。

以下仍是具体断点：SkillDiffuser 引用的 Decision Diffuser 原文未在本轮追完；QueST 的 FSQ 原件未本轮打开；AtomSkill 的 supervised contrastive learning 原件；DeCo 前端已由 Event 分支补齐；CRAFT 的 CAST、ReSteer 等原文未核，不能画成已验证承接链。AtomSkill v2 新引用的 ARP 2606.22480 与 2605.23477 仅在引用表发现，未晋升方法级条目。SkillNet 不能因会议年份就被计为半年内 arXiv 新作。

主要机制覆盖已具备多跳原文证据。本轮在这些明确断点处收敛；未通过继续堆候选数量稀释阅读深度。

### 两报告增量核查

相关原文、阅读范围与纠错见 [分支证据](evidence.md#review)，综合取舍见 [atomic skill 方法总结](../../docs/atomic_skill.md)。

[AtomicVLA：语义技能路由的 π 系直接近邻](atomic_vla.md)。

<a id="recent"></a>

## 近期筛选与阅读边界

检索截止：**2026-10-27**。新作窗口：**2026-04-27 至 2026-10-27（含端点）**，按 arXiv **v1 首次提交日期**，不按最新修订、会议年份或编号前缀。方法和传承证据见 [机制溯源证据](evidence.md#trace)。

### 已核方法、符合 v1 窗口的主要工作

| 工作 | 首次提交与实读版本 | 为什么进入本分支 | 决策 |
| --- | --- | --- | --- |
| [Decompose and Recompose](https://arxiv.org/abs/2605.01448) | v1 2026-05-02；读 v1 §2、§3.1–3.3 | 显式技能标注、动态检索和静态技能覆盖库；是参数化语义结构以外的组合路线 | 核心；[卡片](decompose_recompose.md) |
| [ACT-VLA](https://arxiv.org/abs/2607.00351) | v1 2026-07-01；读 v1 §II-B、§III-A–B | TLI 插值用于合成演示；组合能力可能来自数据补齐，应与表示分解区分 | 核心竞争路线；[卡片](act_vla.md) |
| [DASL](https://arxiv.org/abs/2607.10625) | v1 2026-07-12；读 v2 2026-09-11 §2.2–2.3、§3.1–3.5 | 低频技能、高频控制、仅训练期 diffusion；直接影响部署结构选择 | 核心；[卡片](dasl.md) |
| [CRAFT](https://arxiv.org/abs/2610.00524) | **v1 2026-09-30**；读 v1 §2–4.4、§5.1、Appendix B.1 | 用动作监督对齐可交换的技能表示；skill 的 operation–entity 定义与本项目不同 | 核心；[卡片](craft.md) |
| [Semantic Handoff](https://arxiv.org/abs/2607.06256) | v1 2026-07-07；读 v2 2026-07-15 的 §II–III 开头 | 比较干净起始状态与前一技能实际终态；支持组合失败的接口诊断 | 支持，RSS SemRob 2026 workshop；不由此要求采用外部 verifier |

CRAFT 的 arXiv 编号是 2610.00524，但提交历史明确为 9 月 30 日。DASL 的日期窗口资格取自 v1；方法判断取自 v2，两个事实分别记录。

### 旧论文，近期修订不计新作

| 工作 | 原始发表与本轮所读 | 处理 |
| --- | --- | --- |
| [AtomSkill](https://arxiv.org/abs/2512.18368) | v1 **2025-12-20**；v2 **2026-07-02** | 核心交叉工作，主卡在 Event；v2 的语义正对条件、DeCo 引用和补充文献均应按此版本理解 |
| [TLI / VLAs are Confined yet Capable of Generalizing to Novel Instructions](https://arxiv.org/abs/2505.03500) | arXiv 2025 年旧作；本轮 HTML v5 **2026-05-01** | ACT-VLA 的已核前驱，不计半年新作 |
| [STAR](https://arxiv.org/abs/2506.03863) | 2025 年旧作；本轮承接 v3 **2026-04-07** | 连最近修订也早于窗口；作为已有量化器分支，不重复列“新” |
| [SkillVLA](https://arxiv.org/abs/2603.03836) | v1 **2026-03-04** | 窗口之前，已读 §III、§IV；支持双臂并行组合维度，与 CRAFT 实体重组不同 |
| [LADS](https://openreview.net/forum?id=L66G39JrM4) | ICLR **2025** 正式论文；未核 arXiv ID | 本轮新补读、并非新发表；[核心卡](lads.md) |
| [X-ICM](https://arxiv.org/abs/2505.15660) | 本轮读 v3 **2025-10-19** | Decompose and Recompose 的已核比较对象，保留于引用链 |

### 值得知道但未晋升为本分支核心的方法线索

下面的阅读深度较浅，不计入“已完成方法核对”的新增工作数。日期标记来自摘要页或搜索条目，不用于伪装完整时间窗覆盖。

| 候选 | 实际读到什么 | 暂停/转交理由 |
| --- | --- | --- |
| [SkillNet](https://proceedings.mlr.press/v306/xie26ac.html) | ICML 2026 正式摘要和[作者方法页](https://vipl-vsu.github.io/skillnet-website/)；原始 PDF 获取失败 | 层级技能与 skill-context 专家路由很相关；无已核 arXiv v1，亦无全文方法证据，不计近半年 arXiv 新作。优先补 PDF |
| [Fast Plans, Faithful Actions](https://arxiv.org/abs/2609.30833) | 摘要页，v1 2026-09-25；未核方法 | 计划使用与动作忠实性；整合后仍为待查，未晋升方法级证据 |
| [H-VLA](https://arxiv.org/abs/2609.22895) | 搜索/摘要线索，未独立复核 v1 及方法 | 动作层级与规划接口；保留给 action representation 后续待查，本轮未读方法 |
| [EmbodiedSkills](https://arxiv.org/abs/2609.01281) | 搜索/摘要线索 | 技能前后置条件与运行时编排，更接近 Event 接口；未作为对象无关潜技能证据 |
| [OrthoSkillVLA](https://arxiv.org/abs/2608.19589) | 搜索/摘要线索 | 主要关注持续学习参数子空间和遗忘，skill 词义不同，暂不进入 atomic skill 主线 |
| [Weights or Skills](https://arxiv.org/abs/2608.01851) | 摘要级综述线索 | 可用于补检索词；不替代原方法证据 |
| [ARP](https://arxiv.org/abs/2606.22480) / [Semantically Structured MoE](https://arxiv.org/abs/2605.23477) | 仅 AtomSkill v2 参考文献条目 | 尚未打开两篇方法，不推断它们与本项目表示分工的相容性 |

主分支另外核对 X-Tokenizer、UniLAP 等 action representation 工作；本分支不重复记为独立精读。CRAFT、SkillNet 与这些新 tokenization 工作应在总综述中相互交叉链接，而非把所有带 skill 的文章都搬入同一目录。

### 检索与收敛记录

检索沿三个入口推进：

1. 种子的相关工作与方法引用：LISA → DIAYN → VIC；CompILE → TACO → CTC；QueST → PRISE → BPE；AtomSkill 对语言技能和动作抽象的比较。
2. 前驱/后续线索：LISA、LCSD、SkillDiffuser 的语义技能与 codebook 稳定性问题，扩到 LADS、DASL；近期 CRAFT 的参考文献进一步找到 ACT-VLA、SkillVLA 和 SkillNet。
3. 近期概念查询：atomic skill learning、semantic skill abstraction、compositional VLA、skill alignment、text latent interpolation、skill retrieval / coverage、skill handoff，并结合 2026 / arXiv 标识、各原始摘要页逐项核对日期。

本记录是**有焦点的引用追踪与近期检索**，不是 arXiv 全库导出、完整引文计量或系统综述。搜索结果中出现的日期未自动被视为 v1；核心表都回到了原始论文/摘要页。为避免扩张清单代替理解，本轮在已覆盖主要竞争机制后停止，新线索保留可查入口与读取深度。

最应优先补的不是更多同名 skill 条目，而是 SkillNet 全文，以及在同一实验预算下区分表示分解、技能覆盖检索和合成数据三条路线。

### 两报告复核补充（2026-10-27）

下列是报告带来的增量，不是重新执行整轮穷尽检索。首发与近期修订分开。

| 工作 | 首发 | 本轮版本 | 时间归类 |
| --- | --- | --- | --- |
| [AtomicVLA：语义技能路由的 π 系直接近邻](atomic_vla.md) | 2026-03-08 | 2603.07648v2 | 旧作近期修订 |

<a id="review"></a>

## 报告原文复核

评审日期：2026-10-27。输入是原项目的 Gemini 与 GPT 调研报告（原稿已移出 docs 备份），与当前 `ref_papers` 比较。报告作为待审资料读取；没有修改报告或源项目。本分支补读 LOTUS、AtomicVLA 和前驱 BUDS 的相关原文，另复核 QueST / AtomSkill 的关键表述；不审计整份报告所有成绩。

### 值得吸收的增量

**最值得补的是 AtomicVLA 与 LOTUS，但用途不同。** AtomicVLA 直接涉及语义技能如何调度 π 系动作专家，足以改变我们的架构对照；LOTUS 补齐的是技能库如何增长、复用与保留旧知识。后者不能直接证明对象无关 atomic skill，也不应把持续学习变成 Phase 1 的强制要求。

GPT 报告在本分支更有价值的是研究组织：明确离散不等于语义不变、Event 标签与运动编码分开、CRAFT 的 skill 含实体、真实部署条件与 teacher 条件区分，以及保持数据/计算预算的消融。大部分已被现有 ref_papers 覆盖，应合并为实验设计与阅读注释，不重复建论文卡。

Gemini 报告提供了两个确实遗漏的近邻，同时有几处把实验现象提升为普遍机制或确定保证的表述，不能原样并入知识库。

### 需要修正的关键表述

| 报告定位 / 原表述要点 | 核查结论 | 建议表述与原文依据 |
| --- | --- | --- |
| Gemini，方向三 LOTUS：silhouette 高于阈值表示新技能 | **方向写反** | 高值归入匹配的旧分区，低值开新分区。[LOTUS v4 §IV-A](https://arxiv.org/html/2311.02058v4) |
| Gemini，AtomSkill 小节开头：把 QueST 与 LOTUS 都归入固定滑窗的共同局限 | **LOTUS 分类错误** | LOTUS 由层次聚类取得 temporal segments；应分别讨论视觉分段与固定动作窗口的限制。[LOTUS §IV-A](https://arxiv.org/html/2311.02058v4) |
| Gemini，QueST：去状态输入证明得到纯粹 grasp/place 语义、观察输入导致捷径是确定原因 | **将假说写成因果事实** | QueST 的 Appendix C 将观察条件损害量化写为 hypothesis；actions-only 输入也不能证明无对象/几何信息，轨迹本身可携带这些信息。[QueST v3 §4、Appendix C](https://arxiv.org/html/2407.15840v3) |
| Gemini，QueST：FSQ 全面替代 VQ、彻底解决塌缩 | **超出该比较支持范围** | 可保留其在本设定下的优势；不要变成所有 VLA 的量化器定论。原文同一消融说明架构也有贡献。[QueST Appendix C](https://arxiv.org/html/2407.15840v3) |
| Gemini，AtomSkill：夹爪变化是接触变化的强确定性信号，天然就是技能边界 | **工程启发被写成保证** | 夹爪事件是候选边界；连续行为等存在明确局限，空合夹爪也不能据命令本身证明接触成功。[AtomSkill v2 §III-A、§IX](https://arxiv.org/html/2512.18368v2) |
| Gemini，AtomSkill：相同 VLM 标签片段都聚拢 | **漏了关键正对条件** | 语义正对还要求同量化 token。不能据此写成“所有同动词轨迹收缩到同一点”。[AtomSkill §III-B，公式 (2)–(4) 后的定义](https://arxiv.org/html/2512.18368v2) |
| Gemini，AtomSkill：keypose 使空间和长时程行为普遍改善 | **需保留分组结果和不确定性** | Table V 总体 73.3→74.2（+0.9pp），定位组 61.1→66.1（+5.0pp），运动模式组 85.5→82.2（−3.3pp）；不能写成所有子类都改善。表未给该差值显著性证据。[§IV-D / Table V](https://arxiv.org/html/2512.18368v2) |
| Gemini，AtomicVLA：首次完全统一规划与执行 | **本轮未验证“首次”** | 限定到“以技能引导专家路由实现统一接口”；其 §2.1 自己列出已有统一推理/动作工作。不能从一篇摘要确认完整优先权。[AtomicVLA v2 §2.1、§3.3](https://arxiv.org/html/2603.07648v2) |
| Gemini，AtomicVLA：遇到新技能自动分配专家，实现零干扰终身扩容 | **方法支持与验证范围混淆** | 新技能扩展需要示范及训练；持续学习实验是四个旧短任务后加入一个 Open 任务。固定旧专家不自动保证整个规划/路由系统永不退化。[§3.4、§A.2、§A.3.4](https://arxiv.org/html/2603.07648v2) |

**AtomicVLA 表格另有内部不一致，引用时要保留。** Table 4 的旧任务变化列印为 −1.3pp；但列出的旧任务成功率从 (95, 80, 70, 100) 到 (90, 80, 80, 100)，按表注的四旧任务均值计算，应是 86.25→87.5，即 +1.25pp。HTML 与 [v2 PDF 第 8 页 Table 4](https://arxiv.org/pdf/2603.07648v2) 一致存在此问题。本评审不替作者猜测哪一格应改，也不以印刷的负号断言平均遗忘。可靠结论是目前仅验证有限扩展，尚不能宣称长期零干扰。

GPT 报告的两处合并建议：

- §2.1、§3.1 对 AtomSkill 的解释大体正确，但加入“同标签且同量化 token”的正对条件；“离散瓶颈鼓励复用”作为设计动机，不作语义保证。
- §3.2 的 CRAFT 反事实监督描述应明确：换指令产生的 skill 表示结合匹配演示的 state 表示与动作监督，不是把另一状态的原始动作当固定当前观测的真值。继续保留其 operation–entity 定义及固定操作顺序的范围。既有 `craft.md` 已记录，无需再建新卡。
- §3.3 的“同 pick 表示相近”应以控制相关差异为边界：夹持方式、工具及接触规律改变时，是否仍合并由受控实验决定，不能把删除全部运动信息当作先验正确答案。

### 新补读的两条路线应如何影响方案

#### AtomicVLA：技能可以改变参数调用，而不仅是输入 token

详见 [新卡](atomic_vla.md)。其最有价值的对照是**语义条件究竟放在哪里**：给共享动作专家增加条件，还是让语义决定哪些专家处理当前阶段。这里共享专家和选中的技能专家共同参与，不等于一个技能拥有一个完全孤立的控制器。

对本项目，先在相同 atomic skill 监督下比较共享专家条件化、技能路由，以及非语义路由；保持总容量、激活容量、训练预算可比。只有这样才能区分语义分解、参数扩容与负迁移减轻分别带来多少收益。无需立刻采用完整 think/act 循环，也不能凭专家分工就推断 target grounding 或 action representation 已解耦。

另一个应补入 Event 标签方案的线索是其运动轴/夹爪粗分解再由视频模型修正语义。它覆盖夹爪不变化但旋转/平移模式改变的部分情形；与脚本状态、接触事件及视觉分段比较后再选，不把一个阈值规则升级为通用事件定义。

#### LOTUS：边界、技能身份和技能库增长是三个不同问题

详见 [新卡](../event_organization/lotus_continual.md)。LOTUS 将轨迹分段、跨任务分区、选择技能与子目标、持续更新连在一起，但它的低层子目标是未来图像 latent，不能直接改名为当前 target grounding。

对本项目真正可吸收的是技能目录的生命周期：新演示复用已有类别，还是需要新子类；边界变化后旧技能统计是否稳定；扩展数据时旧任务是否受损。Phase 1 可以先固定技能字典，仅记录这些诊断；若后续正式研究持续学习，再引入多阶段顺序、回放预算和旧任务保留指标。

这也补充了一个与“纯动词技能”不同的竞争解释：由视觉/运动相似性形成的可执行技能可能跨不同动词复用。评测应同时问“能否命名”和“是否改善控制”，不以标签一致率作为唯一尺度。

### 已核关系链与命名消歧

| 关系 | 类型 | 两端原文与变化 |
| --- | --- | --- |
| BUDS → LOTUS | **明确沿用或改进** | [BUDS v2 §III-B–C](https://arxiv.org/html/2109.13841v2) 通过层次分段、跨示范聚类和目标条件技能建库；[LOTUS v4 §IV-A 引用 [63]，§IV-B](https://arxiv.org/html/2311.02058v4) 明确承接分段，增加 DINOv2 表征、增量分区与回放。不是仅据时间先后连线 |
| LOTUS → AtomicVLA | **明确引用；不同方案** | [AtomicVLA §2.3 / reference [49]](https://arxiv.org/html/2603.07648v2) 引用 LOTUS 的持续技能发现；对照 [LOTUS §IV](https://arxiv.org/html/2311.02058v4) 与 AtomicVLA §3.3–3.4，不能说后者继承了前者的聚类算法 |
| AtomSkill ↔ AtomicVLA | **本轮机制比较** | 两者分别是语义约束动作码、技能标识引导专家。未在本轮核出直接引用，不写成前后继承 |
| LOTUS ↔ 3D-LOTUS++ | **仅名称相似，不合并** | LOTUS 是 2311.02058 的持续模仿/技能发现；现有 `target_grounding/lotus_3d.md` 是 2410.01345 的 GemBench 与 LLM-guided 3D policy。不同作者线和问题，保留独立条目 |

BUDS 上游还明确提到多模态表征和 Mid-level Action Elements，但本轮不继续扩展这些通用/远端前驱：三篇已解释报告漏掉的视觉层次分段→持续建库→语义专家路由变化。其余不是本轮完成的精读。

### 建议怎样合并进现有知识库

| 条目/位置 | 分层建议 | 合并理由 |
| --- | --- | --- |
| `atomic_skill/atomic_vla.md` | **支持，重要近邻**；跨链 Event | 与 π 系底座、语义标签和动作专家接口直接相邻；填补“条件化 vs 参数路由”对照，但 Phase 1 不必采用 MoE |
| `event_organization/lotus_continual.md` | **支持**；跨链 atomic skill | 视觉层次分段补齐 Event 路线，持续技能库属于后续扩展；勿与 3D-LOTUS++ 合并 |
| BUDS | 溯源中的**支持节点**，本轮不新增第三张主卡 | 已核两端，是 LOTUS 的直接前驱；当前不需要扩大核心阅读列表 |
| 既有 QueST、AtomSkill 卡 | 加证据限定和表格备注 | 不新建重复卡；补“动作码不等于纯语义”、正对条件及 keypose 分组收益 |
| 既有 CRAFT 卡 / 实验综述 | 合并 GPT 报告中的接口实验建议 | 已有原文与卡片；重点补合法反事实、同预算及 teacher–部署条件一致性 |
| 新近论文清单 | AtomicVLA 记为**旧作近期修订** | v1 2026-03-08，v2 2026-09-06；不能加入 2026-04-27 起的 v1 新作统计 |
| 总体研究结论 | 限定持续学习、长时程与未见组合三个评价轴 | 三者互相关联，但一种成功率不证明另外两种能力 |

### 阅读与校验边界

- LOTUS：2311.02058v4，2024-11-23；ICRA 2024；读 §III、§IV-A–B、§V-A 及相关比较。
- AtomicVLA：2603.07648v2，2026-09-06；CVPR 2026；读 §2.1、§2.3、§3.2–3.5、§4.3–4.4、§A.2、§A.3.1–A.3.5；Table 4 另核 PDF 第 8 页。
- BUDS：2109.13841v2，2022-01-22；RA-L 2022；读 §III-A–C。
- QueST：2407.15840v3，复核 §4 与 Appendix B.5/C；不是本轮新收录。
- AtomSkill：2512.18368v2，复核 §III-B–C、§IV-D、§IX；不是本轮新收录。
- CRAFT：沿用已完成的 v1 §3–4、Appendix B.1 原文记录；本轮不重复计作新精读。

未执行代码、复现实验或审计全部成绩。结论区分原文机制、作者解释与本项目推论；特别是“首次”“彻底”“零干扰”等强断言，均没有因为报告写得肯定就视为已核事实。



当前综合取舍见 [atomic skill](../../docs/atomic_skill.md) 与 [Event](../../docs/event_organization.md) 方法总结。
