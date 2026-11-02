# Action Representation：固定未来时刻的二维末端轨迹

本页维护 Insight 3 的设计、理由与来源，服务于 [idea.md](idea.md)。2026-11-02 确认：**每臂 8 个 Motion tokens，每隔 4 个来源步预测一个二维末端轨迹点**。运动 Token 同时服务未来技能预测与动作专家；整体接口见 [design.md](design.md)。八点二维方案已接入 Dataset、逐 token MLP、损失与推理，并完成十条真实轨迹的标签／渲染核验及小模型链路检查；正式训练与闭环效果尚未验证。

## 1. 学习什么，为什么这样设计？

Atomic Skill 描述操作技巧，Target Grounding 描述当前交互状态与事件目标，Motion 补充当前条件下未来一段时间的末端运动。第一版采用固定时刻的二维辅助监督：每个 token 绑定一个未来时刻，解码一个主相机中的末端位置。

固定参数为每臂 \(M=8\)、特征宽度 \(d=256\)、间隔 \(\Delta=4\)，偏移为 `[4,8,12,16,20,24,28,32]`。LIBERO 的一步指一个来源帧索引增量，不再额外乘“四帧单位”；观察与动作序列保持原采样粒度。来源步不直接等于物理秒数，跨数据集使用时须核对采样与同步关系。

八点描述一条未来轨迹，不是八条轨迹。固定时域允许跨 Event；当前 Event 的 Goal 仍监督该事件终点的对象状态，不随 Motion 窗口移动。M=8 是当前首版选择，不能据此认定其效果优于 16 点或其他间隔。

## 2. Motion tokens 与逐点解码

左右臂共享基础查询与网络参数，加入臂嵌入。令 \(r\in\{L,R\}\)、\(j=0,\ldots,7\)：

\[
\delta_j=4(j+1),\qquad q_{r,j}=q_j^{\mathrm{learn}}+e_r^{arm}.
\]

查询槽位与未来偏移一一对应，时间语义由固定监督建立。第一版不额外向轨迹 head 输入时间编码；旧的归一化弧长进度不再作为时间含义。训练和推理使用同一偏移配置，不依赖真实事件长度或未来轨迹生成在线查询。

双臂共 16 个查询经共享轻量 Transformer 联合自注意力，再交叉读取预测 Grounding、当前／历史空间特征与本体状态。各臂当前预测技能通过共享 FiLM 调制本臂查询：

\[
Z_t=\operatorname{MotionTransformer}_{\mathrm{FiLM}(\mathrm{SkillEmbedding}_t^L,\mathrm{SkillEmbedding}_t^R)}
(Q^{motion},\mathrm{TargetGrounding}_t,F_{\le t}^{spatial},\text{双臂状态}),
\qquad Z_t\in\mathbb R^{B\times2\times8\times256}.
\]

跨臂、跨时刻共享一个逐 token MLP 轨迹头：

\[
\hat u_{t,r,j}=D_{trace}(z_{t,r,j})\in\mathbb R^2,\qquad
\widehat U_t\in\mathbb R^{B\times2\times8\times2}.
\]

每个 token 预测对应时刻的一点；所有 tokens 已通过注意力交换信息，逐点解码不意味着各点独立建模。共享 head 已实现为 LayerNorm(256) → Linear(256,256) → GELU → Linear(256,2)，输出归一化 xy；推理动作路径跳过该 head。

**Motion 与 Grounding 的 token 数量独立。** Grounding 每臂每角色 16 个 tokens 对应 4×4 个空间块，三个角色共 48 个；Motion 每臂 8 个 tokens 对应八个未来时刻。两者通过交叉注意力连接，统一辅助维度 256，不做逐 token 空间—时间配对。未来技能的 H 个动作位置也无需与 M 相同。

## 3. 标签、有效性与损失

主相机二维目标来自真实末端参考点，LIBERO 采用已核验的 `grip_site` 投影。三维位置经相机投影、来源图像方向变换和观察输入相同的 resize／pad，得到模型观察画布上的归一化二维坐标。保留原图位置；不改成物体中心、bbox 中心或相对当前末端的位移。图像变换必须同步作用于坐标与有效性。

设 Episode 最后一个来源帧索引为 \(T_{last}\)：

\[
\tau_{t,j}=t+\delta_j,\qquad
\bar\tau_{t,j}=\min(\tau_{t,j},T_{last}),\qquad
u_{t,r,j}^{*}=\operatorname{ProjectToObservation}(p^r_{\bar\tau_{t,j}}).
\]

采用 `hold_last`：超过 Episode 末尾时保持最后一帧末端位置，参与监督；不是在当前 Event 结束处截断。仅当末帧位置可靠且在画面内时补齐。内部缺帧、无效投影、相机后方或屏外点不填造坐标；遮挡本身不使几何投影失效。

保留 `future_observed`／`future_padded`，分别标识实际未来记录与终点保持点；`future_frame_indices` 保存请求帧，`future_source_frame_indices` 保存实际取值帧。末尾补齐不产生额外真实观察，也不改变未来技能或动作的标签规则。

沿用逐元素 SmoothL1，先平均二维坐标，再平均有效批次／臂／点：

\[
\mathcal L_{trace}=
\frac{\sum_{b,r,j}v_{b,r,j}\operatorname{Mean}_{xy}
[\operatorname{SmoothL1}_{\beta_{trace}}(\hat u_{b,r,j}-u^*_{b,r,j})]}
{\max(1,\sum_{b,r,j}v_{b,r,j})}.
\]

有效性排除不存在的臂与无效目标；各点等权，记录真实未来点和终点保持点的误差与数量。二维归一化下重新核验阈值、损失权重与像素误差，旧米制三维损失的数值不直接沿用。当前不自动引入参考论文中的 validity 预测头。

## 4. 怎样服务动作执行？

预测运动 tokens 与当前技能、三角色视觉 tokens、上下文和本体状态共同输入未来技能 Decoder，同时作为动作专家共享条件。运动分支接受轨迹、未来技能分类和动作损失；真实未来坐标只出现在监督侧。

推理可以省去轨迹解码，动作专家直接读取 latent。每次新观察重新预测条件，在本次 Flow Matching 采样内缓存复用；执行动作块短前缀后再观察。八点窗口不规定动作块长度或执行步数，也不要求控制器逐点跟踪。

二维末端位置不完整表达深度、朝向、接触力和夹爪行为；这些仍由观测、本体状态、技能与真实动作监督承担。确定性辅助回归不保证多模态路径或跨本体迁移效果。

## 5. 方法来源与实现状态

[SeeTraceAct §3 与附录 B](https://arxiv.org/html/2606.02745v2) 的可学习规划查询逐个对应未来二维末端点；轨迹监督训练 latent，动作专家读取 latent，推理可移除轨迹头。其报告设置为 5 点、间隔 5 步，并有 in-frame validity 预测头。我们借鉴 token 与时刻绑定及辅助监督链路，使用 π0.5、双臂与技能条件、8 点／4 步及终点保持规则，不将论文效果视为本方案验证。

[RoboTracer](ref_papers/action_representation/robotracer.md) 保留为有序空间轨迹的背景参考；事件内弧长采样、相对三维位移属于旧方案。其他来源见 [论文库](ref_papers/action_representation/README.md)。

现有 [六点终点保持试验](../data/libero/atomic-vla/check/test/motion_2d_unit4_hold_v1/README.md) 已保存逐帧投影、六点标签与视频；保持原参数和记录。新 [八点／四步核验](../data/libero/atomic-vla/check/test/motion_2d_8x4_v1/README.md) 已完成十条轨迹导出、Dataset 回读比较和渲染，附带可读取的派生训练索引；其余轨迹仍需导出对应相机标定。
