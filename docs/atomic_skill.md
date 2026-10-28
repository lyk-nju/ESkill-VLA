# Atomic Skill：语义选择与可学习操作先验

本页维护当前技巧定义、方法和来源，服务于 [idea.md](idea.md)。网络接口和训练路径见 [design.md](design.md)。Stage 1 采用 **冻结文本原型识别技能 + 可训练技能向量参与目标与运动解码**。

## 1. 技能表示什么？

**Atomic Skill 是可作用于不同对象、在不同状态下通过不同运动实现的交互操作，其语义由预期建立或改变的交互关系界定。** 它既提供当前操作语义，也通过与 Grounding 及共享网络联合学习，形成可复用的操作先验。

例如 Pick 描述使对象进入受控抓持状态，Place 描述使对象在目标支撑区域稳定放置。对象身份、具体位置和实例几何交给 Target Grounding；当前如何运动由 Action Representation 结合场景与本体状态生成。

Atomic 不等于最小控制阶段。技能词表与 Event 边界一起确定：若 Pick 包含接近、抓持和抬升，它可以作为一个技能；若使用 Approach / Grasp 等细分事件，标签和监督范围必须同步细分。不能将瞬时夹爪闭合阶段标为 Grasp，却期待它独自覆盖整段抓取过程。

## 2. 两套表示，一条离散选择接口

| 表示 | 职责 | 如何学习 |
| --- | --- | --- |
| 文本原型 \(e_k\) | 定义语义类别，供在线选择器匹配 | 冻结文本编码器离线生成，部署缓存 |
| 技能向量 \(u_k\) | 条件化事件目标、粗运动和实际动作 | 独立可训练查找表，由下游监督更新 |

为每类技能编写不含具体对象名称的交互目的描述 \(d_k\)。一个可学习 Skill Query 交叉关注当前／历史视觉语言上下文，产生 \(v_t\)，再投影到文本原型空间：

\[
e_k=\operatorname{Normalize}(E_{text}(d_k)),\qquad
v_t=\operatorname{SkillDecoder}(Q^{skill},C_t),\qquad
q_t=\operatorname{Normalize}(W_s v_t).
\]

\[
p_t(k)=\frac{\exp(q_t^\top e_k/\tau)}{\sum_j\exp(q_t^\top e_j/\tau)},
\qquad \mathcal L_{skill}=-\log p_t(k_t^*),\quad\tau>0.
\]

\[
U\in\mathbb R^{N_{skill}\times d},\qquad
\hat k_t=\arg\max_k p_t(k),\qquad u_t=U[\hat k_t].
\]

\(U\) 可由文本原型投影后的向量初始化，但随后是独立参数表，不在每次前向中重新计算 \(W_{adapt}e_k\)。同一类别共享同一个向量；选择器特征 \(v_t\) 不直接绕过查表成为下游技巧条件。

分类交叉熵训练选择器；硬 argmax 不传递下游梯度。被选中的技能向量仍通过目标、轨迹和动作损失学习。联合训练与部署均使用预测类别，不将真值技能作为默认动作条件。

训练初期保留预测类别查表，关注总体／逐类准确率、预测类别占比及技能表各行的梯度或更新范数。误分类会让下游梯度更新到错误技能行，需结合这些记录设置 \(\lambda_s\) 与分类头学习率，避免少数行长期吸收不同技能的更新；不默认把权重调得越大越好，也不增加教师标签切换流程。文本原型初始化不保证误选无害或零样本执行能力。

## 3. 如何参与实际运动？

技能向量进入三处计算：作为事件目标 Query 的上下文、通过 FiLM 调制共享运动网络、直接作为 Action Expert 的条件。

\[
g_t^{goal}=P_\theta(Q^{goal},[C_t,g_t^{ee},g_t^{obj},u_t]),
\]

\[
Z_t=\operatorname{MotionTransformer}_{\mathrm{FiLM}(u_t)}
(Q^{motion},G_t,F_{\le t}^{spatial},A_{body}(s_t)),
\qquad \Delta\hat p_{t,j}=D_{trace}(z_{t,j}),
\]

\[
A_{t:t+H}\sim\pi_{FM}(\cdot\mid C_t,s_t,u_t,G_t,Z_t).
\]

其中 \(Q^{motion}\) 已包含固定弧长进度编码。同一个技能向量可以结合不同对象和目标生成不同粗路径；所有技能共用运动网络及轨迹头。真实动作监督同时覆盖姿态和夹爪控制，避免只靠近静止的位置轨迹区分 Grasp / Release。

**操作先验由技能向量与共享网络共同承载。** 此设计明确了语义识别和执行条件的分工，不保证模型必然使用技能，也不证明跨本体不变性。旧方案的文本适配器本来就可接收下游梯度；本次变化是独立可训练技能表，不是首次建立技巧到运动的训练路径。

## 4. 参考来源与当前范围

| 来源 | 借鉴内容 | 本项目边界 |
| --- | --- | --- |
| [AtomSkill](../ref_papers/event_organization/atomskill.md) | Event 语义与具有动作解码用途的技能表示 | 技能应参与执行学习；本轮不复制其动作量化码本 |
| [w²VLA](../ref_papers/atomic_skill/w2vla.md) | 冻结文本技巧嵌入与下游接口 | 文本原型只负责识别，下游技能向量独立学习 |
| [AR-WAM](../ref_papers/atomic_skill/ar_wam.md) | 显式操作条件参与模型计算 | 借鉴条件接口，不等同于在线原型分类 |
| [CRAFT](../ref_papers/atomic_skill/craft.md) | 技能应能跨执行复用并影响动作 | 其技能包含 operation–entity；我们将操作与对象 Grounding 分工，不直接采用反事实训练 |

Stage 1 先落实语义标签、预测选择和三条下游通路，联合训练完整模型，以平均任务成功率为主。技能分类准确率用于定位问题，不新增 VQ、对比或反事实损失，也不将各技巧接口的逐项消融列为前置条件。更多证据见 [技巧论文库](../ref_papers/atomic_skill/README.md)。
