# Atomic Skill：语义选择与可学习操作先验

本页维护技巧定义、当前识别与未来预测，服务于 [idea.md](idea.md)。网络接口和训练路径见 [design.md](design.md)。Stage 1 采用 **当前事件表征 → 每臂未来技能序列 → 与动作步对齐的技能条件**；当前技能通过冻结文本原型分类，当前与未来类别查同一张可训练技能表。

左右臂各有一个当前 Skill Query 和 \(H\) 个未来技能 Query，使用臂嵌入区分。当前识别器、未来预测器分别跨臂共享；两者使用同一词表与技能表。同一个 Pick 在左右臂上仍是同一技能，两臂可以预测不同的技能序列与切换位置。

## 1. 技能表示什么？

**Atomic Skill 是可作用于不同对象、在不同状态下通过不同运动实现的交互操作，其语义由预期建立或改变的交互关系界定。** 它既提供当前操作语义，也通过与 Grounding 及共享网络联合学习，形成可复用的操作先验。

例如 Pick 描述使对象进入受控抓持状态，Place 描述使对象在目标支撑区域稳定放置。对象身份、具体位置和实例几何交给 Target Grounding；当前如何运动由 Action Representation 结合场景与本体状态生成。

Atomic 不等于最小控制阶段。技能词表与 Event 边界一起确定：若 Pick 包含接近、抓持和抬升，它可以作为一个技能；若使用 Approach / Grasp 等细分事件，标签和监督范围必须同步细分。不能将瞬时夹爪闭合阶段标为 Grasp，却期待它独自覆盖整段抓取过程。

词表同时覆盖 idle（无主动交互）、hold（保持已建立的抓持／支撑关系）、return（返回规定姿态）。不能把静止自动归为 idle；hold 仍关联对象，无对象 return 则允许对象角色监督无效。两臂共同操作的关联记录在事件标签中，不新增协作技能分类。

## 2. 两套表示，一条离散选择接口

| 表示 | 职责 | 如何学习 |
| --- | --- | --- |
| 文本原型 \(e_k\) | 定义语义类别，供当前技能选择器匹配 | 冻结文本编码器离线生成，部署缓存 |
| 技能向量 \(\mathrm{SkillEmbedding}_k\) | 当前向量条件化事件目标、粗运动和未来技能预测；未来向量逐步条件化动作 | 独立可训练查找表，当前／未来分支共用，由各可达的下游损失更新 |

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

当前技能交叉熵训练选择器；硬 argmax 不传递下游梯度。被选中的当前技能向量通过目标、轨迹、未来技能分类和动作损失学习。联合训练与部署均使用预测类别，不将真值技能作为动作条件。

误分类会使下游梯度更新到错误技能行；初始化不保证误选无害或零样本执行能力。收敛诊断与数值配置统一见 [design.md](design.md) 第 4.1 节。

## 3. 如何预测未来技能安排？

当前技能与当前 ee／obj 锚点并行预测，再生成当前 Event 的 Goal 和固定未来时域的运动 Token。在此基础上，每臂建立 \(H\) 个带动作步位置编码和臂嵌入的查询；共享未来技能 Decoder 读取当前／历史上下文、本体状态、当前技能、三角色视觉 Token 和未来运动 Token。

未来查询跨臂、跨动作位置联合自注意力，再交叉读取这些条件；共享线性分类头并行输出 \([B,2,H,N_{skill}]\) 的分布。逐位置 argmax 后查同一 \(\mathrm{SkillTable}\)，得到 \([B,2,H,d]\) 的技能向量。Stage 1 使用动作步编码；物理时间须在采样／控制周期核实后解释，不能把来源帧偏移或展示 fps 当成未来秒数。

监督直接由各动作记录所属的每臂 Event 技能派生，采用有效批次／臂／动作步上的交叉熵。未来分类损失通过连续输入回传到预测锚点和运动分支，不需要额外标注切换时间。padding、缺标签和不存在臂不计该损失；已知 idle／hold 正常监督。具体查询、标签与梯度接口见 [design.md](design.md) 第 2.4、3、4 节。

技能序列保留操作顺序和预期切换位置；它不确认事件成功，也不能区分连续同技能、不同对象的 Event。Goal 对应当前 Event，Motion 每臂八点／间隔四个来源步的窗口允许跨 Event；执行短前缀后用新观察重新估计整个技能窗口。

## 4. 如何参与实际运动？

当前技能向量条件化本臂 Goal、运动 FiLM 和未来技能 Decoder，也保留为动作专家的共享事件上下文。未来第 \(j\) 步的左右臂技能向量按固定顺序拼接、投影，并加入第 \(j\) 个 Action Token；动作专家统一生成联合动作块。所有位置共用投影，不增加逐技能动作专家。

训练与推理都使用预测未来类别。硬选择下，动作损失可更新选中的技能表行及动作投影，但不能直接穿过 argmax 优化未来分类器或切换位置；未来分类器由序列交叉熵训练。每次新观察只预测一次条件，在本次 Flow Matching 采样中缓存复用。

**操作先验由技能向量与共享网络共同承载。** 此设计明确了语义识别和执行条件的分工，不保证模型必然使用技能，也不证明跨本体不变性。

## 5. 参考来源与当前范围

| 来源 | 借鉴内容 | 本项目边界 |
| --- | --- | --- |
| [AtomSkill](../ref_papers/event_organization/atomskill.md) | Event 语义与具有动作解码用途的技能表示 | 技能应参与执行学习；本轮不复制其动作量化码本 |
| [w²VLA](../ref_papers/atomic_skill/w2vla.md) | 冻结文本技巧嵌入与下游接口 | 文本原型只负责识别，下游技能向量独立学习 |
| [AR-WAM](../ref_papers/atomic_skill/ar_wam.md) | 显式操作条件参与模型计算 | 借鉴条件接口，不等同于在线原型分类 |
| [CRAFT](../ref_papers/atomic_skill/craft.md) | 技能应能跨执行复用并影响动作 | 其技能包含 operation–entity；我们将操作与对象 Grounding 分工，不直接采用反事实训练 |

当前保留离散选择，不新增 VQ、对比或反事实损失。Stage 1 完成标准见 [design.md](design.md) 第 6 节；更多参考见 [技巧论文库](../ref_papers/atomic_skill/README.md)。
