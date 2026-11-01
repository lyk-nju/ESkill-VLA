# Action Representation：事件条件下的整段粗运动表征

本页维护 Insight 3 的设计、理由与来源，服务于 [idea.md](idea.md)。当前 Stage 1 采用 **Event-conditioned Spatial Trace**：每臂用固定数量的运动 Token 表达从当前末端到本臂 Event 结束的有序粗路径，双臂 Token 在共享网络中联合交互。它既支持未来技能安排预测，也作为动作专家的共享条件。整体接口见 [design.md](design.md)，模型尚待实现和验证。

## 1. 学习什么，为什么这样设计？

Atomic Skill 描述操作技巧，Target Grounding 描述当前交互状态与事件目标，Action Representation 补充：**在这些条件下，末端应大致经过怎样的空间路径到达事件终点。**

只用抽象“运动意图”缺少可操作的监督；直接预测密集控制轨迹又承担了大量速度与本体控制细节。因此借鉴 RoboTracer 的 Spatial Trace 思路，保留整个剩余 Event 的空间路径，用少量有序点表达其形状。覆盖完整事件范围，不等于复现每个控制时刻的动作。

路径按归一化弧长采样，运动 Token 对应空间进展而非固定未来时间。同一条路径执行快慢不同，可以具有相同的粗空间标签。固定数量 Token 让训练和动作条件接口保持简单，无需先预测剩余时长，也无需为不同事件分配不同数量的 Token。

## 2. 当前表示与监督

每臂使用 \(M\ge2\) 个 Motion Queries，共享基础查询并加入显式归一化弧长进度编码与臂嵌入。令 \(r\in\{L,R\}\)：

\[
s_j=\frac{j}{M-1},\qquad
q_{r,j}=q_j^{\mathrm{learn}}+\operatorname{PE}(s_j)+e_r^{arm},\qquad
Q^{motion}=[Q^L_{0:M-1},Q^R_{0:M-1}].
\]

\(\operatorname{PE}\) 使用与查询同维度的固定正弦／余弦编码，在训练与推理中保持一致。可学习查询承载槽位特征，进度编码显式标记剩余路径的起点、中段和终点，输出槽位 \(j\) 始终对应同一 \(s_j\) 的标签。

这里的进度由固定采样网格给出，不需要未来轨迹或真实事件长度；它表示沿本臂预测剩余路径的空间比例，不是未来时间或机器人当前任务完成率。左右臂各自归一化，相同槽位 \(j\) 不表示同一未来时刻。编码有助于学习顺序，但不硬性保证路径无折返、自交。

将双臂共 \(2M\) 个查询展平，经同一个轻量 Transformer 联合自注意力，再交叉关注双臂预测锚点、当前／历史空间视觉特征及带臂标识的状态 Token。各臂所选技能向量 \(\mathrm{SkillEmbedding}_t^r=\mathrm{SkillTable}[\widehat{\mathrm{SkillID}}_t^r]\) 经共享 FiLM 网络产生参数，只在所属臂的 \(M\) 个查询上广播：

\[
\mathrm{LatentMotion}_t=\operatorname{MotionTransformer}_{\mathrm{FiLM}(\mathrm{SkillEmbedding}_t^L,\mathrm{SkillEmbedding}_t^R)}
\left(Q^{motion},\mathrm{TargetGrounding}_t,F_{\le t}^{spatial},\text{双臂状态 Token}\right),
\qquad \mathrm{LatentMotion}_t\in\mathbb R^{2\times M\times d}.
\]

联合自注意力允许两臂交换运动条件，不要求两条路径按时间逐点对齐，也不保证自动学会协调。共享轻量解码头将每个 Token 转为相对本臂当前末端的三维位移：

\[
\Delta\hat p_{t,j}^r=D_{trace}(z_{t,j}^r),\qquad
\hat p_{t,j}^r=p_t^r+\Delta\hat p_{t,j}^r,\qquad j=0,\ldots,M-1.
\]

每臂训练标签来自当前时刻到其事件终点 \(t_e^r\) 的真实末端路径。分别以累计弧长归一化得到 \(\gamma_t^{r*}(s)\)，取 \(s_j=j/(M-1)\)：

\[
\Delta p_{t,j}^{r*}=\gamma_t^{r*}(s_j)-p_t^r,\qquad
\mathcal L_{trace}=
\frac{\sum_{r,j}v_{t,r,j}\operatorname{SmoothL1}_{\beta_{trace}}(\Delta\hat p_{t,j}^r/s_{xyz},\Delta p_{t,j}^{r*}/s_{xyz})}
{\max(1,\sum_{r,j}v_{t,r,j})}.
\]

三维位置在双臂共用的固定参考系中计算，保留各自实际起点和末端终点，不将终点替换为对象中心。只有实际记录的路径近零才使用零位移，避免弧长归一化除零；不能按 idle／hold 标签直接置零。\(v_{t,r,j}\) 标记有效臂／点标签；批量训练时分子分母一起累加批次维度。\(M\) 作为配置选择，需保证关键转弯不会被过度稀疏化。归一化只作用于采样进度，不用每条路径的总长度缩放三维位移，因此保留空间尺度。具体参考系、单位与执行器参考点在数据接入时确定。

**损失约定：** \(s_{xyz}>0\) 是统一固定空间尺度，轨迹头仍输出物理位移；仅计算损失时除以该尺度。SmoothL1 对 XYZ 逐元素取均值，再等权平均有效批次／臂／点，不另加端点权重。尺度来源、阈值与诊断统一见 [design.md](design.md) 第 4.1 节。

## 3. 怎样服务动作执行？

**双臂预测的运动 Token** 首先与当前技能、三角色视觉 Token、上下文和本体状态共同输入未来技能 Decoder，预测每个动作步的技能安排；同时保留为动作专家的共享条件。未来技能向量逐步注入动作 Token，由同一 Flow Matching Action Expert 输出固定长度联合动作块。

运动分支接受粗轨迹、未来技能分类和动作损失，并沿 FiLM 路径更新被选中的当前技能向量。所有臂和技能共享运动网络和轨迹头；完整真实未来路径只用于监督，不进入在线分支。

运动 Token 的 \(M\) 个位置表示弧长进度，未来技能的 \(H\) 个位置表示动作步，两者通过注意力联系，不要求数量相同或逐点对应。弧长路径不直接给出速度，未来技能预测仍读取当前状态和可用历史。即使技能序列跨越 Event，运动监督仍只到当前 Event 的终点。

推理移除轨迹解码头，缓存运动 Token 及预测的逐步技能条件；执行动作块的短前缀后根据新观察重新预测。独立并发事件保留各自剩余路径，共同追求同一物体终态的协作事件按数据约定对齐语义终点，但两条真实末端路径仍分开监督。运动表征提供空间参考，不要求底层逐点跟踪，也不一次开环执行整段事件。

Grasp / Release 等近静止阶段可以具有相近的位置标签，其区别由技巧、观测和真实动作监督承担。位置轨迹不完整表达朝向、速度、接触与夹爪行为，因此不能替代动作专家。

当前不加入时长头、变长运动 Token、热图、布朗桥、椭球容忍损失或独立轨迹生成器。联合预测保证 Token 可以交换信息，但确定性 SmoothL1 仍是示范路径拟合，**不保证多模态路径保持，也不等于已学到与本体无关的意图**。相对位移消除了平移原点，不自动消除旋转、尺度和身体差异。

## 4. 方法来源与我们的改造

| 来源 | 借鉴内容 | 本项目的选择与边界 |
| --- | --- | --- |
| [RoboTracer](../ref_papers/action_representation/robotracer.md) | 用有序、稀疏三维 Spatial Trace 连接任务理解与执行 | 主要表征参考；原文输出 \((u,v,d)\) 空间点。本项目改为每臂固定 Motion Queries、双臂联合注意力、相对 XYZ 辅助回归和动作专家条件；这不是原文架构 |
| [WALL-WM](../ref_papers/action_representation/wall_wm.md) | Event 对齐运动监督与数据组织 | 保留事件组织依据；其时长预测、变长视频／动作塔不作为当前 Stage 1 实现 |
| [LAPA](../ref_papers/action_representation/lapa.md) / [ConLA](../ref_papers/action_representation/conla.md) | 从视频学习动作相关知识、减少外观捷径 | 保留跨数据来源的设计动机；当前不直接采用其量化教师或对比目标 |

RoboTracer 的规划与控制接口不能直接证明本项目的辅助 Token 会提升 VLA。我们借鉴的是空间路径表示，再将其接入当前联合训练框架。更广候选留在 [论文库](../ref_papers/action_representation/README.md)。

Stage 1 的数据检查、训练与评测标准统一见 [design.md](design.md) 第 6 节。未来接入人类视频仍需解决手部三维重建、相机补偿、尺度与参考点对齐。
