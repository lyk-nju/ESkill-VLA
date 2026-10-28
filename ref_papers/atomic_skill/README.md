# Atomic Skill：按当前方法组织的论文索引

当前方法见 [docs/atomic_skill.md](../../docs/atomic_skill.md)，整体接口见 [design.md](../../docs/design.md)。

先读 w²VLA 与 AR-WAM 的条件接口，再结合 AtomSkill / StaKe 的语义与辅助监督；CRAFT 用于检验表示是否真正控制行为。

本方向主卡：3 核心、15 补充、2 归档。跨方向工作只链接同一主卡，不重复计数。优先级更新于 2026-10-28；“核心”按当前设计用途选择，不代表已复现或整篇方法都会采用。

## 当前核心

| 工作 | 当前角色 | 直接服务什么设计 |
| --- | --- | --- |
| [w²VLA：文本技巧与 FiLM 条件化](w2vla.md) | 直接方法来源 | 文本技巧嵌入、适配与 FiLM 条件化；对象无关原型分类器是我们的改造。 |
| [AR-WAM：离散操作与视觉接地条件](ar_wam.md) | 直接接口参照 | 离散操作嵌入与对象接地分别条件化策略；外部技能条件不等于自主技能预测。 |
| [CRAFT：用动作监督检验技能表示能否跨执行复用](craft.md) | 复用验证依据 | 用控制及合法条件交换检验技巧复用；operation–entity 与本项目对象无关技巧定义不同。 |

## 跨方向核心接口

| 工作 | 当前角色 | 对本方向的作用 |
| --- | --- | --- |
| [AtomSkill 语义事件与技能对齐](../event_organization/atomskill.md) | 直接方法来源 | Event 分段与技巧语义；借鉴组织方式，不直接沿用动作量化码本。 |
| [StaKe：π0.5 的阶段与关键帧辅助监督](../event_organization/stake.md) | 轻量监督参照 | 阶段辅助头与关键帧监督；二元阶段预测不等于在线多类别技巧选择。 |

<details>
<summary>按需补充与替代路线（15 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [LISA 语言条件下的离散技能](lisa.md) | 按需补充与替代路线 |
| [LCSD：语言重建能建立技能，也可能混入对象参数](lcsd.md) | 按需补充与替代路线 |
| [SkillDiffuser：语义技能与多模态运动生成已有直接先例](skilldiffuser.md) | 按需补充与替代路线 |
| [QueST：动作压缩码不自动等于语义技能](quest.md) | 按需补充与替代路线 |
| [PRISE 动作序列压缩与变长技能](prise.md) | 按需补充与替代路线 |
| [CompILE：学到事件分段，不代表学到预期的技能语义](compile.md) | 按需补充与替代路线 |
| [TACO：知道子任务顺序，也许不需要精确逐帧边界](taco.md) | 按需补充与替代路线 |
| [VQ-BeT 动作量化与多模态控制](vq_bet.md) | 按需补充与替代路线 |
| [SPiRL：技能身份稳定，不等于技能选择不依赖状态](spirl.md) | 按需补充与替代路线 |
| [SAILOR 2022：技能检索与在线可预测性也是重要对照](sailor_2022.md) | 按需补充与替代路线 |
| [DASL：慢速技能与快速动作，不要求部署时生成轨迹](dasl.md) | 按需补充与替代路线 |
| [LADS：分开学习什么技能与如何在线预测它](lads.md) | 按需补充与替代路线 |
| [Decompose and Recompose：技能覆盖也能通过示范检索实现](decompose_recompose.md) | 按需补充与替代路线 |
| [ACT-VLA：通过 latent 组合补齐训练数据](act_vla.md) | 按需补充与替代路线 |
| [AtomicVLA：语义技能路由的 π 系直接近邻](atomic_vla.md) | 按需补充与替代路线 |

</details>

<details>
<summary>历史归档（2 项）</summary>

这些工作保留原文机制与既有阅读积累，当前不作为首轮实现的核心依赖。

| 工作 | 原文阅读入口 |
| --- | --- |
| [STAR：码本利用率与语义多样性应分别验证](../_archive/atomic_skill/star.md) | 历史归档 |
| [LGSD：语言首先定义了什么样的差异值得学习](../_archive/atomic_skill/lgsd.md) | 历史归档 |

</details>

[历史溯源与核查](evidence.md)保留各轮证据；其中旧分层不代表当前阅读顺序。当前级别与调整前记录统一在 [catalog.json](../catalog.json)。

返回 [论文库入口](../README.md)。
