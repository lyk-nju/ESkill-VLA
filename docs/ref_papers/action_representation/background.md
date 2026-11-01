# Action Representation：背景机制笔记

本页将 24 项较远路线压缩为机制、参考价值与适用边界。这里没有待实施方案、实验清单或排期；当前方法见 [方向文档](../../action_representation.md)，代码实现以 [design.md](../../design.md) 为准。

来源沿用既有阅读记录，本次没有重新联网核验或复现。原文的模块、训练流程和结果只描述原工作。

<a id="univla"></a>

## UniVLA 任务相关潜动作与显式解码

[论文来源](https://arxiv.org/html/2505.06111v3)

阅读范围：v3 §III-A–C。

**原方法：** 在 DINOv2 特征上用语言和两阶段码本学习任务相关潜动作，部署保留 latent 预测并用视觉条件 decoder 适配机器人。

**保留价值与边界：** 保留中间动作表征显式连接本体控制的先例。不证明本项目 Atomic Skill、Target Grounding 与 Latent Motion 因子化或 Event 收益；当前不采用其码本和 decoder。

<a id="xskill"></a>

## XSkill 跨身体技能与示范对齐

[论文来源](https://arxiv.org/abs/2307.09955) · [出版／作者入口](https://xskill.cs.columbia.edu/)

阅读范围：作者方法页 Abstract、One-shot learning。

**原方法：** 人和机器人视频发现技能 prototypes，条件扩散策略执行，Skill Alignment Transformer 将当前状态与提示视频技能对齐。

**保留价值与边界：** 保留后续人类视频提示和技能序列对齐的方向。共享表征与机器人策略训练是不同阶段，不支持任意互联网视频零样本控制的推论。

<a id="unipi"></a>

## UniPi 生成视频作为运动计划接口

[论文来源](https://arxiv.org/abs/2302.00111) · [出版／作者入口](https://research.google/blog/unipi-learning-universal-policies-via-text-guided-video-generation/)

阅读范围：作者官方方法介绍，未完成全文实现审计。

**原方法：** 当前图像和语言生成未来视频计划，再由逆动力学模型提取动作。

**保留价值与边界：** 保留显式视频计划连接执行的路线。视频视觉合理不等于物理可执行；当前不引入视频生成规划。

<a id="rynnworld_latent"></a>

## RynnWorld-Latent：把同一潜动作接到世界模型

[来源入口](https://github.com/alibaba-damo-academy/RynnWorld-Latent)

阅读范围：官方 README；未核实正式论文 ID。

**原方法：** 以 RynnLAM 连续潜动作条件化视频世界模型，官方示例由初始帧和潜动作序列预测后续视频。

**保留价值与边界：** 保留同一运动接口用于视频生成的可能用途。官方项目记录，不冒充论文全文；含相机分量，跨外观生成不等于机器人执行迁移。

<a id="clam"></a>

## CLAM：连续潜动作与真实动作联合接地

[论文来源](https://arxiv.org/abs/2505.04999v2) · [出版／作者入口](https://clamrobot.github.io/)

阅读范围：v2 官方摘要与作者方法页；未完整核查正文。

**原方法：** 观测转移的连续 IDM/FDM 与少量带动作 play 联合训练 decoder；策略部署产生 latent 再解码控制。

**保留价值与边界：** 保留连续运动表示和少量真实动作接地的机制。专家示范可无动作，但 play 仍有动作监督，不能描述为全流程无动作标签。

<a id="latent_wm_in_the_wild"></a>

## 野外潜动作世界模型：迁移的可能是相机相对运动

[论文来源](https://arxiv.org/html/2601.05230v2) · [出版／作者入口](https://proceedings.mlr.press/v306/garrido26a.html)

阅读范围：v2 问题设定、信息约束、局部性分析与规划接口。

**原方法：** IDM/FDM 以信息约束学习视觉转移，分析野外数据中的相机相对与空间局部含义；真实动作另映射到 latent。

**保留价值与边界：** 保留相机运动和操作运动可能混合的解释边界。视频迁移与循环一致性不等于语义技能复用，离散基线结果只覆盖该设定。

<a id="lawam"></a>

## LaWAM：action representation 经未来视觉特征连接动作专家

[论文来源](https://arxiv.org/html/2606.15768v2) · [出版／作者入口](https://rlinf.github.io/LaWAM/)

阅读范围：v2 §3.2–3.3。

**原方法：** IDM 从当前／未来特征学习 latent，decoder 预测未来特征；第二阶段当前条件 prior 预测 latent 并通过 decoder 条件化动作，部分梯度隔离。

**保留价值与边界：** 保留运动 latent、预测未来特征和动作条件不是同一变量的区别。确定性 prior 不保证多峰分布；当前不用其两阶段训练、未来特征 decoder 或 stop-gradient。

<a id="lac_wm"></a>

## LAC-WM：真实候选动作进入潜空间，再预测后果

[论文来源](https://arxiv.org/html/2610.10846v1) · [出版／作者入口](https://lacwm.github.io/)

阅读范围：v1 §3–4。

**原方法：** IDM/FDM 配合增强与手、末端、相机运动标签学习 latent；真实候选动作经 projector 进入 latent，由模型预测后果。

**保留价值与边界：** 保留跨本体运动监督和世界模型动作投影的方向。A→latent→未来与本项目当前观测→Latent Motion→动作方向不同；当前不增加候选动作评估器。

<a id="moto"></a>

## Moto：运动 token 有训练价值，也可以在推理时省略

[论文来源](https://arxiv.org/html/2412.04445v4) · [出版／作者入口](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Moto_Latent_Motion_Token_as_the_Bridging_Language_for_Learning_ICCV_2025_paper.html)

阅读范围：v4 §3.2–3.4。

**原方法：** 帧对经查询式 M-Former 与 VQ 得到运动 token，预训练运动序列；真实动作微调插入 action query，并随机屏蔽部分运动条件。

**保留价值与边界：** 保留运动学习与部署是否显式生成运动 token 的区别。运动码不等于语义 Skill；该文推理可省略运动生成，当前则保留预测 Latent Motion 条件。

<a id="dynamo"></a>

## DynaMo：动力学预训练也可以只改善视觉表征

[论文来源](https://arxiv.org/abs/2409.12192) · [出版／作者入口](https://dynamo-ssl.github.io/)

阅读范围：作者方法页，未完成全文与代码审计。

**原方法：** 领域内观测用视觉编码、latent IDM/FDM 和特征前向一致性预训练，表征供下游模仿策略使用。

**保留价值与边界：** 保留动力学监督改善视觉表示的路线。主要对象是视觉预训练，不是部署显式运动接口；下游仍需动作监督。

<a id="rla_wm"></a>

## RLA-WM：把状态转移残差压缩后建模

[论文来源](https://arxiv.org/html/2605.07079) · [出版／作者入口](https://mlzxy.github.io/rla-wm/)

阅读范围：当前 HTML §3、§4 概述；原卡未固定修订号。

**原方法：** 压缩未来与当前 DINO 特征残差，再以真实动作块和当前状态通过 Flow Matching 预测该残差 latent。

**保留价值与边界：** 保留视觉转移残差作为紧凑预测目标的机制。残差不是先验纯 motor code；动作条件世界模型仍需动作数据。

<a id="fast_wam"></a>

## Fast-WAM：训练预测未来，测试直接控制

[论文来源](https://arxiv.org/html/2603.16666v2)

阅读范围：v2 §3.1–3.2。

**原方法：** 训练联合视频和动作，推理只对当前干净帧做一次视频骨干前向，直接提供动作条件。

**保留价值与边界：** 保留联合学习未来与部署生成未来的区别。当前条件特征不是帧对潜动作；结果不能推广为所有长任务都无需规划。

<a id="dreamdojo"></a>

## DreamDojo：人类视频迁移到机器人世界模型

[论文来源](https://arxiv.org/html/2602.06949v1) · [出版／作者入口](https://dreamdojo-world.github.io/)

阅读范围：v1 §3.1–3.3。

**原方法：** 连续 latent 为人类视频提供动作代理，目标机器人阶段重置动作条件层并适配真实动作，可再蒸馏。

**保留价值与边界：** 保留人类视频预训练与本体动作适配分开完成的路线。这是动作条件世界模型，非直接控制 VLA；适配仍依赖机器人数据。

<a id="adaworld"></a>

## AdaWorld：预训练世界模型的可迁移动作接口

[论文来源](https://arxiv.org/html/2503.18938v4) · [出版／作者入口](https://proceedings.mlr.press/v267/gao25u.html)

阅读范围：v4 §2.1–2.3。

**原方法：** beta-VAE 从帧对获得连续 latent，条件化自回归视频模型；真实控制通过离散动作映射或连续 MLP 适配。

**保留价值与边界：** 保留视频迁移与真实控制接地的区别。零训练视频动作迁移不等于零适配机器人执行。

<a id="igor"></a>

## IGOR：图像目标转移同时服务策略和世界模型

[论文来源](https://arxiv.org/html/2411.00785) · [出版／作者入口](https://www.microsoft.com/en-us/research/project/igor-image-goal-representations/)

阅读范围：§2.1–2.3；原卡未固定修订号。

**原方法：** IDM/VQ 压缩视觉转移，IDM 与 FDM 用不同 crop；latent 供世界模型和高层策略使用，控制微调加入低层动作模块。

**保留价值与边界：** 保留视觉转移瓶颈和数据增强对捷径的约束。atomic control unit 不等于 pick/place 类别；当前不增加其量化与冻结训练流程。

<a id="what_lam_learn"></a>

## What Do LAMs Actually Learn：低重建误差不能保证动作语义

[论文来源](https://arxiv.org/html/2506.15691v3) · [出版／作者入口](https://papers.neurips.cc/paper_files/paper/2025/hash/d7b3cef7c31b94a4a533db83d01a8882-Abstract-Conference.html)

阅读范围：v3 §3–4.1、§5；未逐条重推全部证明。

**原方法：** 在线性及不相关假设下，重建目标对应可控变化和外生噪声混合上的 PCA；受控非线性实验研究类似趋势。

**保留价值与边界：** 保留重建精度本身不能唯一识别控制语义的理论边界。不等于所有深度 LAM 都是 PCA，也不是本项目在 RoboTwin 的效果预测。

<a id="fast"></a>

## FAST：动作序列压缩

[论文来源](https://arxiv.org/html/2501.09747v1)

阅读范围：v1 §V。

**原方法：** 对动作序列做 DCT、量化和 BPE，得到可逆解码的压缩 token。

**保留价值与边界：** 保留动作压缩与语义抽象的区别。一个压缩 token 不等于一个 Event；当前保留基座 Flow Matching 动作格式。

<a id="pixels_to_tokens"></a>

## From Pixels to Tokens：把 teacher 来源与监督接口分开比较

[论文来源](https://arxiv.org/html/2605.04678v1)

阅读范围：v1 §2–4、附录 A.1/B.2/C。

**原方法：** 在统一 VLA 中研究视觉／动作来源表征及不同监督接口，保留动作损失；image teacher 另使用少量动作标签。

**保留价值与边界：** 保留教师来源、标签预算与消费接口是不同因素这一认识。并非所有排行榜方法都同预算重训；当前不将该研究展开为项目实验矩阵。

<a id="unilap"></a>

## UniLAP：从连续视频后验到可采样策略

[论文来源](https://arxiv.org/html/2610.16969v1)

阅读范围：v1 §2–3.3、§4.2、附录 A.2–A.3。

**原方法：** 冻结 DreamDojo LAM teacher，当前观测／语言条件的连续 flow policy 学 latent，以 DINOv2 跨帧对应残差辅助运动；新本体训练 decoder。

**保留价值与边界：** 保留未来需要采样多种运动时，后验教师与在线分布建模分开的路线。当前确定性粗路径不引入 flow latent policy；运动残差不是严格相机不变量，跨语料单位也未自然统一。

<a id="faster_wam_shallow"></a>

## Faster-WAM（Ma）：当前帧世界模型与浅动作专家

[论文来源](https://arxiv.org/html/2608.02365v2)

阅读范围：v2 §2.1–2.5 及训练／推理附录。

**原方法：** 继承 Fast-WAM 当前帧 prefill，用 DoT/Lite KV-Fusion 让浅动作专家读取多层视频 KV，推理不生成未来视频。

**保留价值与边界：** 保留不同层特征接入动作专家的接口背景。与 2608.04404 的 Faster-WAM 是不同论文，速度数字不能跨设置直接比较。

<a id="mv_lam"></a>

## MV-LAM：多视角运动可观测性与动作监督上界

[论文来源](https://arxiv.org/html/2610.27794v1)

阅读范围：v1 §II–III、§V-C–E。

**原方法：** 同步多视角先空间交互再时序建模，压缩前后特征差形成运动标签；串行 Flow Matching 专家预测运动和动作。

**保留价值与边界：** 保留三维运动可观测性与多视角信息的关系。与 MVP-LAM 是不同工作；当前不新增多视角教师或串行动作生成阶段。

<a id="affordancewam"></a>

## AffordanceWAM：哪些未来 World 信息应该传给动作

[论文来源](https://arxiv.org/html/2609.22332v3)

阅读范围：v3 §2、§3.1–3.4、§4.3。

**原方法：** 未来 RGB 与不同 affordance 信号参与 World 学习，动作只读指定通路；任务后训练冻结 World 并停止部分接口梯度。

**保留价值与边界：** 保留学习信息、向动作开放的信息和梯度通路是不同设计选择。其通路分析不证明 RGB 无用或重建必然有害；当前仍按 design.md 联合训练并使用预测三锚点。

<a id="vh_diffuser"></a>

## VH-Diffuser：预测时域后生成变长轨迹

[论文来源](https://arxiv.org/html/2509.11930v1)

阅读范围：v1 §V；未复现。

**原方法：** 估计起止状态的最短步数，再设置扩散噪声序列长度；训练使用随机长度片段。

**保留价值与边界：** 保留时域与轨迹内容分开预测这一研究背景。属于独立的变时域规划路线；当前每臂固定 M 个空间进度 Query，双臂联合自注意力，不使用时长头或变长生成。

<a id="adaptive_action_chunking"></a>

## Adaptive Action Chunking：轻量长度头与动作联合学习

[来源入口](https://www.mdpi.com/2313-7673/11/5/316) · [补充来源](https://pmc.ncbi.nlm.nih.gov/articles/PMC13204913/) · [补充来源](https://pubmed.ncbi.nlm.nih.gov/42187383/)

阅读范围：既有方法记录；出版入口核对时正文 429、PMC 验证页，未重新完整核读。

**原方法：** 既有记录描述共享视觉特征联合预测动作与 chunk 长度类别，以最大长度输出后选择前缀，长度标签来自难度启发式。

**保留价值与边界：** 保留动作块长度与语义事件长度的区别。最近一次正文访问未完成，机制来自更早笔记；不将其视为已核验代码配方，当前不采用其长度分类头。
