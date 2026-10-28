# action representation：来源与核查证据

> **历史证据页。** 下文保留 2026-10-27 等调研记录，旧“核心／支持”标签和方法建议属于当时判断。2026-10-28 的当前优先级以 [本方向索引](README.md) 和 [catalog.json](../catalog.json) 为准。

本轮补齐当前方法已引用工作的独立卡片；新卡各自记录来源与核查范围，未改写以下历史记录。

本页合并此前的机制溯源、近期筛选和相关原文复核，保留来源链接、版本、阅读范围与未闭合问题。记录截止 2026-10-27；本次仅整理结构，没有新增论文阅读或复现。

方法取舍见 [docs/action_representation.md](../../docs/action_representation.md)，选论文从 [本方向索引](README.md) 开始。以下“本轮”和实验建议指对应调研记录，不自动构成当前架构决定。

[机制溯源](#trace) · [近期筛选](#recent)

<a id="trace"></a>

## 机制溯源

核查日期：2026-10-27。本文按机制追踪，不将 action representation 等同于 latent action。相关小节核读不等于完整通读或复现；旧卡未重读的部分仍沿用其阅读状态。

### 1. 三条多层证据链

以下箭头表示后作到前作的追踪关系，具体关系类型与核验状态见表。不是领域完整历史。

| 关系（后作 → 前作） | 从哪里发现；两端原文 | 可确认的变化与关系强度 |
| --- | --- | --- |
| LAC-WM → LAOM | [LAC-WM §1、§3.1](https://arxiv.org/html/2610.10846v1) 的 Nikulin et al. (2025)；[LAOM §4–5](https://arxiv.org/html/2502.00379v5) | 明确沿用：在 latent 学习期预测显式运动/动作标签；LAC-WM 再用于跨本体世界建模。监督对象和数据设定不同。 |
| LAOM → LAPO | LAOM §4 明说复现和修改 LAPO；[LAPO §4.1–4.3](https://arxiv.org/html/2312.10812v2) | 明确修改：从量化转移瓶颈到连续、高容量、特征预测与增强，再在学习期间加入少量真实动作监督。 |
| From Pixels to Tokens → LAOM | [From Pixels to Tokens 附录 A.1](https://arxiv.org/html/2605.04678v1) 的 Nikulin et al.；LAOM §5 | 明确借鉴监督时机：其 image-based teacher 也使用 5% 动作标签，不能称为纯无动作对照。 |
| UniLAP → DreamDojo | [UniLAP §3、附录 A.2](https://arxiv.org/html/2610.16969v1) 指定 frozen DreamDojo LAM；[DreamDojo §3.3.2](https://arxiv.org/html/2602.06949v1) | 直接采用 teacher；把后验转成由当前观测/语言条件化的连续 flow policy。不是仅在 related work 提到。 |
| DreamDojo → AdaWorld | DreamDojo §3.3.2 的 Gao et al. (2025)；[AdaWorld §2.1–2.3](https://arxiv.org/html/2503.18938v4) | 明确借鉴连续 VAE latent；应用转向人类视频预训练及目标机器人世界模型适配。 |
| AdaWorld → Genie / LAOM | AdaWorld §2.1、附录 F.1–F.2；LAOM §4–5；Genie 本轮仅确认引用身份 | Genie 是时空架构/离散控制比较来源；LAOM 是量化局限的引用。Genie 两端方法未补读，此分支停在待核，不能画成已完成的继承链。 |
| X-Tokenizer → ActionCodec | [X-Tokenizer §1–2、§3](https://arxiv.org/html/2606.14752v2) 的 [25]；[ActionCodec v1 §4–5](https://arxiv.org/html/2602.15397v1)、[v2 §3.3–4](https://arxiv.org/html/2602.15397v2) | 明确比较与差异化，不认定代码继承：前作先对齐第一层再加残差；后作联合非对称 RVQ、冻结 VLM 对齐与未来特征监督。引用发生时是 v1，不能用 10 月 v2 倒推 6 月影响。 |
| ActionCodec → FAST | ActionCodec v1 §2、v2 §5.1；[FAST §V-A–V-C](https://arxiv.org/html/2501.09747v1) | 明确比较/替代：从 DCT+BPE 压缩进入学习型码本的优化研究；不是在 FAST 算法上加模块。 |

这三条链分别回答：怎样让视频变化包含控制信息、怎样把离线后验变成在线策略、怎样让动作编码兼顾可学习性与精度。它们不应合并为“统一的技能发现史”。

### 2. 未来表征是否必须留在部署阶段

| 工作 | 已读位置 | 部署接口与关系 |
| --- | --- | --- |
| [Fast-WAM](fast_wam.md) | 上轮 §3.1–3.2；本轮以后作两端对照 | 训练预测未来，测试只预填当前帧，是省略未来条件的基线。 |
| [Faster-WAM / future conditioning](faster_wam_future.md) | Introduction、Method、Future Conditioning 消融 | 明确批评 Fast-WAM 的 OOD 泛化；单次视频前向保留带噪 future slots，再缓存和稀疏融合。并不生成完整未来视频。 |
| [Faster-WAM / shallow action expert](faster_wam_shallow.md) | §2.1–2.5 | 明确继承 Fast-WAM 的 observed-frame prefill；轻量动作专家读跨层 KV。v2 §2.5 明确将另一 Faster-WAM 列为并行路线。 |
| [LaWAM](lawam.md) | §1–3、参考文献映射 | 将 LAM forward decoder 保留为未来特征接口。§1 的 [5] 是 Genie，不能误认成 UniVLA；“保留 decoder”是相对一组 prior approaches 的设计差异，不等于从某一篇直接派生。 |

**项目判断：** Zhao 一文还有保留相同 SparseMoT/KV-Fusion 的 current-only 对照，支持其设置中的未来条件贡献；这比只比 Fast-WAM 更有力。两篇 Faster-WAM 之间仍改变了动作容量和输入路径，跨论文分数不能推导普遍的“future slots 必要性”。在我们的相同 backbone、数据和动作头下开关未来条件，并报告新增延迟，才是可迁移的比较。

### 3. 对当前 Insight 的修订

1. **压缩、语义对齐、控制可用性分开测。** FAST 给压缩基线；X-Tokenizer 提供语义约束机制；ActionPiece 检查重建后的动作关系。三者均不能直接充当对象无关 atomic skill 的证明。
2. **表征中存在动作信息，不等于在线 prior 能预测或控制器会使用。** LAOM 的线性探针与闭环差异，要求报告 teacher→动作和 predicted representation→动作两种性能。
3. **保持类内运动多样性。** UniLAP 是近期直接候选，但 teacher、时间尺度和适配成本与本项目不同。我们需固定同一 atomic skill / target grounding 后判断多样性，排除换目标、背景变化和碰撞轨迹。
4. **真实动作监督应作为显式实验轴。** 不能拿使用动作标签的视觉 teacher 与纯视频方法比较后，把收益全部归于表示结构。
5. **跨本体迁移区分策略与世界模型。** LAC-WM 的显式动作→latent→未来，与 UniLAP 的观测→latent→真实动作方向相反；RynnWorld-Latent 的可控视频生成也不是闭环策略证据。

### 4. 阅读账本与停止点

| 节点 | 本轮实际读取 | 状态 |
| --- | --- | --- |
| LAPO 2312.10812v2 | §2、§4.1–4.3 | 历史支撑；不新增主卡 |
| LAOM 2502.00379v5 | §3–5 | 新增诊断卡 |
| AdaWorld 2503.18938v4 | §2.1、附录 F；§2.2–2.3沿用上轮 | 迁移支撑 |
| DreamDojo 2602.06949v1 | §3.1、§3.3.1–3.3.2 | UniLAP teacher 来源；窗口外 |
| LAC-WM 2610.10846v1 | §1–3.1、参考条目；其余沿用旧卡 | 核心跨本体对照 |
| LaWAM 2606.15768v2 | §1–2、参考条目；方法沿用上一轮 §3.2–3.3 | 核心接口对照 |
| FAST 2501.09747v1 | §V | 新增非 latent 基线卡 |
| ActionCodec 2602.15397v1/v2 | v1 §2–4；v2 §3.3–4、§5.1 | 支撑节点，10月修订不算新作 |
| X-Tokenizer 2606.14752v2 | §1–3.4 | 核心新卡 |
| From Pixels to Tokens 2605.04678v1 | §2–4、附录 A.1/B.2/C | 核心新卡，含监督预算纠正 |
| ActionPiece 2609.18487v1 | §3–4、§5.5、附录配置 | 核心诊断卡 |
| UniLAP 2610.16969v1 | §2–3.3、§4.2、附录 A.2–A.3 | 核心新卡 |
| 两篇 Faster-WAM | 前者 Introduction/Method，后者 §2；消融与附录部分定位 | 方法已核读，尚无统一复现 |
| M²Tok 2609.18259 | arXiv 摘要 | 容量/精度支线，待核；不新增主卡 |
| ILPO / Rybkin 早期方法 | 仅从 LAPO/诊断文献定位引用 | 暂停，继续追溯不会直接改变当前实现选择；不声称已追到起源 |
| RynnVLA-Latent / RynnWorld-Latent | 本轮未重审代码；沿用官方项目卡 | 保留实现参考，未确认 arXiv ID，不列半年论文 |

检索还使用了 action tokenizer、semantic interface、cross-embodiment、distractors、world action model 等概念组合；部分加月份查询为空。近期清单是定向筛选，不是 arXiv 全量普查。

### 两报告增量核查

CoMo、CLAP、MV-LAM 和 AffordanceWAM 的相关小节已补入各自卡片，综合取舍见 [action representation 方法总结](../../docs/action_representation.md)。

[CoMo：连续运动教师与时序方向诊断](como.md)、[CLAP：机器人动作锚定与人类视频伪正例](clap.md)、[MV-LAM：多视角运动可观测性与动作监督上界](mv_lam.md)、[AffordanceWAM：哪些未来 World 信息应该传给动作](affordancewam.md)。

<a id="recent"></a>

## 近期筛选与阅读边界

窗口：2026-04-27 至 2026-10-27，按 arXiv v1 首发，不按最后修订。核查：2026-10-27。

| 工作 | v1 日期 / 已读版本 | 定位与决定 |
| --- | --- | --- |
| [From Pixels to Tokens](https://arxiv.org/abs/2605.04678) | 2026-05-06 / v1 | 核心新增：统一接口比较；image teacher 有 5% 动作监督 |
| [RLA-WM](https://arxiv.org/abs/2605.07079) | 2026-05-08 / v2 10-05 | 原有支持：随机特征转移，保留为 feature world model 分支 |
| [WALL-WM](https://arxiv.org/abs/2606.01955) | 2026-06-01 / v2 09-06 | 原有核心，Event agent 本轮进一步追来源 |
| [X-Tokenizer](https://arxiv.org/abs/2606.14752) | 2026-06-07 / v2 06-28 | 核心新增：语义层与运动残差的分工 |
| [LaWAM](https://arxiv.org/abs/2606.15768) | 2026-06-14 / v2 10-25 | 原有核心：未来特征部署接口 |
| [Faster-WAM / shallow](https://arxiv.org/abs/2608.02365) | 2026-08-03 / v2 10-06 | 新增支持：仅当前帧缓存+浅动作专家 |
| [Faster-WAM / future](https://arxiv.org/abs/2608.04404) | 2026-08-05 / v1 | 核心新增：保留未来条件的反方对照 |
| [ActionPiece](https://arxiv.org/abs/2609.18487) | 2026-09-16 / v1 | 核心新增：精细动作关系不能只用 MSE 评价 |
| [LAC-WM](https://arxiv.org/abs/2610.10846) | 2026-10-07 / v1 | 原有核心：跨本体监督与动作→latent 接口 |
| [UniLAP](https://arxiv.org/abs/2610.16969) | 2026-10-12 / v1 | 核心新增：连续多模态 policy、teacher 蒸馏、运动监督 |

### 旧作与近期修订，单独处理

- [CLAM](https://arxiv.org/abs/2505.04999)：v1 2025-05-08，v2 2026-07-30；不算半年新作。全文细节仍按旧卡待查状态。
- [ActionCodec](https://arxiv.org/abs/2602.15397)：v1 2026-02-17，v2 2026-10-08。新补方法阅读，作为 X-Tokenizer 前驱比较；不单开主卡。
- [Fast-WAM](https://arxiv.org/abs/2603.16666)、[DreamDojo](https://arxiv.org/abs/2602.06949)、[ConLA](https://arxiv.org/abs/2602.00557) 在窗口外，仍可能是必要基线。
- Rynn 两项按官方技术项目处理；本轮没有核实到 arXiv 身份与 v1 日期，不能算入半年新论文。

### 本轮未晋升

[M²Tok](https://arxiv.org/abs/2609.18259) 为 09-16 的摘要级候选，关注多头多码本容量。与本项目语义分工的直接关联弱于 X-Tokenizer，与控制精度诊断的区分度弱于 ActionPiece，先留待查。

FASTer 2512.04952 只作为压缩路线候选；Genie/ILPO 为历史断点。此次不把每条引用扩成主卡。

### 两报告复核补充（2026-10-27）

下列是报告带来的增量，不是重新执行整轮穷尽检索。首发与近期修订分开。

| 工作 | 首发 | 本轮版本 | 时间归类 |
| --- | --- | --- | --- |
| [CoMo：连续运动教师与时序方向诊断](como.md) | 2025-05-22 | 2505.17006v3 | 旧作近期修订 |
| [CLAP：机器人动作锚定与人类视频伪正例](clap.md) | 2026-01-07 | 2601.04061v2 | 旧作近期修订 |
| [MV-LAM：多视角运动可观测性与动作监督上界](mv_lam.md) | 2026-10-20 | 2610.27794v1 | 半年内首发 |
| [AffordanceWAM：哪些未来 World 信息应该传给动作](affordancewam.md) | 2026-09-16 | 2609.22332v3 | 半年内首发 |

World–Action 条件与梯度的逐项复核保存在 [target grounding 证据记录](../target_grounding/evidence.md#review)；CoMo、CLAP、MV-LAM 的版本与结论边界见各自论文卡。
