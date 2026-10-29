# Atomic Skill：语义选择与可学习操作先验

本页维护当前技巧定义、方法和来源，服务于 [idea.md](idea.md)。网络接口和训练路径见 [design.md](design.md)。Stage 1 采用 **冻结文本原型识别技能 + 可训练技能向量条件化目标、运动与动作**。

左右臂各有一个带臂嵌入的 Skill Query，共享分类器、文本原型与技能表。两臂可以同时选不同技能；同一个 Pick 在左右臂上仍是同一技能，不采用双臂组合词表。

## 1. 技能表示什么？

**Atomic Skill 是可作用于不同对象、在不同状态下通过不同运动实现的交互操作，其语义由预期建立或改变的交互关系界定。** 它既提供当前操作语义，也通过与 Grounding 及共享网络联合学习，形成可复用的操作先验。

例如 Pick 描述使对象进入受控抓持状态，Place 描述使对象在目标支撑区域稳定放置。对象身份、具体位置和实例几何交给 Target Grounding；当前如何运动由 Action Representation 结合场景与本体状态生成。

Atomic 不等于最小控制阶段。技能词表与 Event 边界一起确定：若 Pick 包含接近、抓持和抬升，它可以作为一个技能；若使用 Approach / Grasp 等细分事件，标签和监督范围必须同步细分。不能将瞬时夹爪闭合阶段标为 Grasp，却期待它独自覆盖整段抓取过程。

词表同时覆盖 idle（无主动交互）、hold（保持已建立的抓持／支撑关系）、return（返回规定姿态）。不能把静止自动归为 idle；hold 仍关联对象，无对象 return 则允许对象角色监督无效。两臂共同操作的关联记录在事件标签中，不新增协作技能分类。

## 2. 两套表示，一条离散选择接口

| 表示 | 职责 | 如何学习 |
| --- | --- | --- |
| 文本原型 \(e_k\) | 定义语义类别，供在线选择器匹配 | 冻结文本编码器离线生成，部署缓存 |
| 技能向量 \(\mathrm{SkillEmbedding}_k\) | 条件化事件目标、粗运动和实际动作 | 独立可训练查找表，由下游监督更新 |

为每类技能编写不含具体对象名称的交互目的描述 \(d_k\)。共享基础 Skill Query 加上臂嵌入 \(e_r^{arm}\)，通过同一 Decoder 关注当前／历史视觉语言上下文；\(r\in\{L,R\}\)：

\[
e_k=\operatorname{Normalize}(E_{text}(d_k)),\qquad
v_t^r=\operatorname{SkillDecoder}(Q^{skill}+e_r^{arm},C_t),\qquad
q_t^r=\operatorname{Normalize}(W_s v_t^r).
\]

\[
p_t^r(k)=\frac{\exp((q_t^r)^\top e_k/\tau)}{\sum_j\exp((q_t^r)^\top e_j/\tau)},
\qquad \ell_{skill}^{r}=-\log p_t^r(\mathrm{SkillID}_t^{r*}),\quad\tau>0.
\]

\[
\mathrm{SkillTable}\in\mathbb R^{N_{skill}\times d},\qquad
\widehat{\mathrm{SkillID}}_t^r=\arg\max_k p_t^r(k),\qquad \mathrm{SkillEmbedding}_t^r=\mathrm{SkillTable}[\widehat{\mathrm{SkillID}}_t^r].
\]

\(\mathrm{SkillTable}\) 可由文本原型投影后的向量初始化，但随后是独立参数表，不在每次前向中重新计算 \(W_{adapt}e_k\)。同一类别跨臂共享同一向量；选择器特征 \(v_t^r\) 不直接绕过查表成为下游技巧条件。\(\mathcal L_{skill}\) 在有效批次／臂标签上求平均，缺失标签不补成 idle，也不因双臂而直接翻倍损失。

分类交叉熵训练选择器；硬 argmax 不传递下游梯度。被选中的技能向量仍通过目标、轨迹和动作损失学习。联合训练与部署均使用预测类别，不将真值技能作为默认动作条件。

误分类会使下游梯度更新到错误技能行；初始化不保证误选无害或零样本执行能力。收敛诊断与数值配置统一见 [design.md](design.md) 第 4.1 节。

## 3. 如何参与实际运动？

每臂技能向量进入三处计算：作为本臂事件目标 Query 的上下文、通过 FiLM 调制本臂运动 Token、直接作为联合 Action Expert 的条件。

FiLM 按所属臂应用，运动网络和动作专家的完整接口见 [design.md](design.md) 第 2 节。同一技能向量可结合不同对象与目标生成不同粗路径；两臂选中同一行时共同更新该行。直接输入动作专家，使技能语义也能帮助区分 Grasp / Release 等位置路径相近、夹爪行为不同的操作。

**操作先验由技能向量与共享网络共同承载。** 此设计明确了语义识别和执行条件的分工，不保证模型必然使用技能，也不证明跨本体不变性。

## 4. 参考来源与当前范围

| 来源 | 借鉴内容 | 本项目边界 |
| --- | --- | --- |
| [AtomSkill](../ref_papers/event_organization/atomskill.md) | Event 语义与具有动作解码用途的技能表示 | 技能应参与执行学习；本轮不复制其动作量化码本 |
| [w²VLA](../ref_papers/atomic_skill/w2vla.md) | 冻结文本技巧嵌入与下游接口 | 文本原型只负责识别，下游技能向量独立学习 |
| [AR-WAM](../ref_papers/atomic_skill/ar_wam.md) | 显式操作条件参与模型计算 | 借鉴条件接口，不等同于在线原型分类 |
| [CRAFT](../ref_papers/atomic_skill/craft.md) | 技能应能跨执行复用并影响动作 | 其技能包含 operation–entity；我们将操作与对象 Grounding 分工，不直接采用反事实训练 |

当前保留离散选择，不新增 VQ、对比或反事实损失。Stage 1 完成标准见 [design.md](design.md) 第 6 节；更多参考见 [技巧论文库](../ref_papers/atomic_skill/README.md)。
