# Atomic Skill：背景机制笔记

本页将 14 项较远路线压缩为机制、参考价值与适用边界。这里没有待实施方案、实验清单或排期；当前方法见 [方向文档](../../docs/atomic_skill.md)，代码实现以 [design.md](../../docs/design.md) 为准。

来源沿用既有阅读记录，本次没有重新联网核验或复现。原文的模块、训练流程和结果只描述原工作。

<a id="lisa"></a>

## LISA 语言条件下的离散技能

[论文来源](https://arxiv.org/abs/2203.00054) · [出版／作者入口](https://papers.nips.cc/paper_files/paper/2022/hash/883105b282fe15275991b411e6b200c5-Abstract-Conference.html)

阅读范围：摘要级记录，未完成方法全文核查。

**原方法：** 语言条件层次模仿学习用向量量化形成离散技能码。

**保留价值与边界：** 保留语言与技能抽象的早期路线。原卡只核摘要，不能据此认定对象信息被消除或确定完整训练接口。

<a id="lcsd"></a>

## LCSD：语言重建能建立技能，也可能混入对象参数

[论文来源](https://arxiv.org/html/2402.17511v1) · [出版／作者入口](https://ojs.aaai.org/index.php/ICAPS/article/view/31488)

阅读范围：v1 Approach、Mutual Information Skill Learning、Diffusion Policy。

**原方法：** 状态与语言产生量化技能，压缩连续重复码，以技能恢复指令 CLIP 语义，并训练状态／技能条件 diffusion policy。

**保留价值与边界：** 保留语义监督目标会决定表示保留哪些信息的观点。完整指令重建可能同时要求对象信息；语言 decoder 与动作 decoder 不同，当前不引入语言重建目标。

<a id="skilldiffuser"></a>

## SkillDiffuser：语义技能与多模态运动生成已有直接先例

[论文来源](https://arxiv.org/html/2312.11598v3) · [出版／作者入口](https://openaccess.thecvf.com/content/CVPR2024/html/Liang_SkillDiffuser_Interpretable_Hierarchical_Planning_via_Skill_Abstractions_in_Diffusion-Based_Task_CVPR_2024_paper.html)

阅读范围：v3 §4.2–4.4。

**原方法：** 从观测与语言预测离散技能，技能条件的 state diffusion 生成未来表示，再用逆动力学恢复动作。

**保留价值与边界：** 保留技能身份与运动生成分层的先例。固定时窗内重复技能码不等于语义 Event；当前不引入该扩散规划与逆动力学链路。

<a id="quest"></a>

## QueST：动作压缩码不自动等于语义技能

[论文来源](https://arxiv.org/html/2407.15840v3) · [出版／作者入口](https://papers.nips.cc/paper_files/paper/2024/hash/076c3e48fa502c660902105965fdd9f6-Abstract-Conference.html)

阅读范围：v3 §3.2、§4.1–4.3、Appendix C。

**原方法：** 动作序列经因果编码器和 FSQ 压缩重建；观测条件自回归 prior 预测码序列，再解码执行动作前缀。

**保留价值与边界：** 保留动作压缩码与语义类别的区别。单个码不必有独立语义；少示范微调不是无更新 in-context 学习，FSQ 也非普遍优于 VQ 的结论。

<a id="prise"></a>

## PRISE 动作序列压缩与变长技能

[论文来源](https://arxiv.org/abs/2402.10450) · [出版／作者入口](https://proceedings.mlr.press/v235/zheng24b.html)

阅读范围：作者方法页；机制链另核 v3 §2–3.1。

**原方法：** 状态条件动作量化后，用 BPE 压缩为变长技能 token，下游预测并解码动作。

**保留价值与边界：** 保留频率压缩产生时间抽象的机制。压缩片段不自动具备语义事件身份；当前不采用 BPE 技能。

<a id="compile"></a>

## CompILE：学到事件分段，不代表学到预期的技能语义

[论文来源](https://arxiv.org/html/1812.01483v2) · [出版／作者入口](https://proceedings.mlr.press/v97/kipf19a.html)

阅读范围：v2 §2.2、§4.3–4.4。

**原方法：** 条件 VAE 联合学习软边界与子任务 latent，状态条件策略解码动作，分段数量由超参数控制。

**保留价值与边界：** 保留分段发现和技能语义可辨识性的区别。无监督码可能主要表达位置；原实验包含 grid-world、reacher 和分层 RL，不直接证明操作 VLA 的跨对象复用。

<a id="taco"></a>

## TACO：知道子任务顺序，也许不需要精确逐帧边界

[论文来源](https://arxiv.org/html/1803.01840v2) · [出版／作者入口](https://proceedings.mlr.press/v80/shiarlis18a.html)

阅读范围：v2 §3.3、§4.1–4.2。

**原方法：** 以有序 task sketch 监督子策略，动态规划边缘化对齐，动作与停止概率参与联合似然。

**保留价值与边界：** 保留有顺序而无逐帧边界的弱监督机制。测试给定新 sketch 与从自由语言自主发现程序不同；当前仍使用可核验事件标签。

<a id="vq_bet"></a>

## VQ-BeT 动作量化与多模态控制

[论文来源](https://arxiv.org/html/2403.03181v2) · [出版／作者入口](https://proceedings.mlr.press/v235/lee24y.html)

阅读范围：v2 §2.3、§3。

**原方法：** 以真实动作训练残差 VQ，从观测预测动作码，用连续 offset 修正量化误差。

**保留价值与边界：** 保留压缩与细节残差协作的思路。需要动作标签，粗细码并非自然对应 K 与 Z；当前不采用动作量化。

<a id="spirl"></a>

## SPiRL：技能身份稳定，不等于技能选择不依赖状态

[论文来源](https://arxiv.org/html/2010.11944v1) · [出版／作者入口](https://proceedings.mlr.press/v155/pertsch21a.html)

阅读范围：v1 §3.2–3.3。

**原方法：** 固定动作片段训练连续 VAE 技能及状态条件 prior；下游高层 RL 选择 latent 并受先验正则，低层解码执行。

**保留价值与边界：** 保留技能身份可复用但选择仍依赖状态这一概念。下游依赖奖励和交互；当前不引入 RL 或连续技能选择。

<a id="sailor_2022"></a>

## SAILOR 2022：技能检索与在线可预测性也是重要对照

[论文来源](https://arxiv.org/html/2210.11435v2) · [出版／作者入口](https://proceedings.mlr.press/v205/nasiriany23a.html)

阅读范围：v2 §3–4.2。

**原方法：** 学习 VAE 技能与起止观测 prior，按目标示范检索技能相近数据，再训练目标策略；部署由历史预测 latent。

**保留价值与边界：** 保留离线技能表示与在线可预测性的区别。新任务需要示范和参数更新；与 NeurIPS 2025 同名 SAILOR 不同。

<a id="star"></a>

## STAR：码本利用率与语义多样性应分别验证

[论文来源](https://arxiv.org/html/2506.03863v3) · [出版／作者入口](https://proceedings.mlr.press/v267/li25bo.html)

阅读范围：v3 §3.2.1–3.2.4；版本晚于会议版。

**原方法：** 动作序列经旋转增强残差量化，再训练技能 Transformer 自回归预测粗细码，并有连续 offset。

**保留价值与边界：** 保留码本利用率与几何细节覆盖的技术背景。causal 指预测依赖，不表示识别操作因果；码本多样性不等于语义多样性。

<a id="lgsd"></a>

## LGSD：语言首先定义了什么样的差异值得学习

[论文来源](https://arxiv.org/html/2406.06615v2) · [出版／作者入口](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dae30ba63ca043588cdf804bbba295ed-Abstract-Conference.html)

阅读范围：v2 §4.1–4.4。

**原方法：** 连续技能 latent、语言描述距离和内在奖励联合学习；规则先把模拟状态转为文本。

**保留价值与边界：** 保留语言描述如何改变技能区分尺度的机制。依赖状态文本与在线 RL；reward-free 不代表没有奖励优化，也不是原始视频无监督学习。

<a id="decompose_recompose"></a>

## Decompose and Recompose：技能覆盖也能通过示范检索实现

[论文来源](https://arxiv.org/html/2605.01448v1)

阅读范围：v1 §2、§3.1–3.3。

**原方法：** 关键帧配合 VLM／规则形成 Verb[obj,target] 描述，动态检索和静态覆盖库提供技能—动作示范 context。

**保留价值与边界：** 保留未来基于示范 context 重新组合任务的路线。无目标任务参数更新仍依赖已有动作示范；本项目 Stage 1 不增加检索上下文模块。

<a id="act_vla"></a>

## ACT-VLA：通过 latent 组合补齐训练数据

[论文来源](https://arxiv.org/html/2607.00351v1)

阅读范围：v1 §II-B、§III-A–B；沿用 TLI 的关系见原卡记录。

**原方法：** 通过文本 latent 插值合成候选行为，在模拟器 rollout 并按成功筛选，再训练 VLA。

**保留价值与边界：** 保留任务组合也可以通过增加有效训练覆盖实现这一解释。含环境交互与筛选成本；当前不增加合成数据流水线。
