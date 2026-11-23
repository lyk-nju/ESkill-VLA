# Event Organization：背景机制笔记

本页将 11 项较远路线压缩为机制、参考价值与适用边界。这里没有待实施方案、实验清单或排期；当前方法见 [方向文档](../../event_organization.md)，代码实现以 [design.md](../../design.md) 为准。

来源沿用既有阅读记录，本次没有重新联网核验或复现。原文的模块、训练流程和结果只描述原工作。

<a id="buds"></a>

## BUDS 从重复结构发现可复用技能

[论文来源](https://arxiv.org/abs/2109.13841) · [出版／作者入口](https://ut-austin-rpl.github.io/BUDS-website/)

阅读范围：原卡以作者方法页为主；前驱对读涉及 v2 §III-B–C。

**原方法：** 对未分段示范构建凝聚聚类层次，从跨任务重复片段发现技能，训练目标条件低层与高层控制器。

**保留价值与边界：** 保留跨任务重复性与技能粒度的关系，作为视觉分段背景。新组合实验可重新训练 meta-controller，不等于冻结策略零更新重组。

<a id="uvd"></a>

## UVD 通过视觉进展发现子目标

[论文来源](https://arxiv.org/abs/2310.08581) · [出版／作者入口](https://zcczhang.github.io/UVD/)

阅读范围：作者方法页、伪代码与实验说明。

**原方法：** 冻结视觉表征，以到目标帧的距离变化递归提取子目标，再由目标条件策略推进。

**保留价值与边界：** 保留仅视觉数据的离线子目标发现机制。离线分解读取完整视频；视觉变化本身没有给出行为类别或物理接触真值。

<a id="stagewam"></a>

## StageWAM：预测下一阶段的潜在状态

[论文来源](https://arxiv.org/html/2608.10780v3)

阅读范围：v3 Method、stage-pair 构造与两阶段训练接口。

**原方法：** 用 V-JEPA2 特征变化和时间 NMS 找边界；Stage-JEPA 预测下一阶段 latent，经有界门控条件化视频分支。部署使用历史缓冲。

**保留价值与边界：** 保留阶段未来状态与事件数据组织的区别。其未来状态预测不等于本项目当前／终点对象视觉 latent；当前不引入 Stage-JEPA 或视频生成分支。

<a id="taksie"></a>

## TaKSIE：子目标选择、进度记忆和切换条件

[论文来源](https://arxiv.org/html/2410.11013v2) · [出版／作者入口](https://openaccess.thecvf.com/content/WACESkill025/papers/Kang_Incorporating_Task_Progress_Knowledge_for_Subgoal_Generation_in_Robotic_Manipulation_WACV_2025_paper.pdf)

阅读范围：v2 §3.1–3.4。

**原方法：** 视觉距离曲线选择子目标，递归进度记忆条件化图像编辑模型；部署用目标相似度或超时刷新目标。

**保留价值与边界：** 保留目标生成、进度记忆与切换条件的不同职责。相似度不保证物理完成，生成目标与在线切换模块不属于当前方案。

<a id="hulc"></a>

## HULC：没有显式事件边界也能学习层次计划

[论文来源](https://arxiv.org/html/2204.06252v2) · [出版／作者入口](https://github.com/lukashermann/hulc)

阅读范围：v2 §III、§IV-A/B。

**原方法：** 从 play 窗口以事后目标和少量语言学习变分 latent plan；结合局部腕部视觉控制，视觉与语言用对比目标连接。

**保留价值与边界：** 保留未来轨迹后验与当前条件先验的区别。类别随机变量不是 VQ 码本，也不是已命名的语义 Event；仍使用机器人动作数据。

<a id="hi_robot"></a>

## Hi Robot：从开放任务到可执行子指令

[论文来源](https://arxiv.org/html/2502.19417v2) · [出版／作者入口](https://proceedings.mlr.press/v267/shi25d.html)

阅读范围：v2 §4.1–4.4。

**原方法：** 高层 VLM 从复杂任务与当前画面产生简单指令，低层 π0 执行；高层定期或收到反馈后更新。

**保留价值与边界：** 保留全局指令、局部操作与更新频率之间的关系。子指令可含对象信息；原工作不能直接证明对象无关技能，也没有要求成功判定后才切换。

<a id="rt_h"></a>

## RT-H：语言动作层级与技能粒度

[论文来源](https://arxiv.org/html/2403.01823v2) · [出版／作者入口](https://www.roboticsproceedings.org/rss20/p049.html)

阅读范围：v2 §III–IV。

**原方法：** 动作主要变化轴派生语言动作标签；同一 VLM 先预测语言动作，再条件化机器人控制，并可异步刷新。

**保留价值与边界：** 保留语言动作的粒度与延迟接口背景。“向上移动”等运动短语与一次完整 Skill 执行不同；本项目未采用异步语言动作层。

<a id="openhouse"></a>

## OpenHOUSE：人类视频中的在线、多粒度事件理解

[论文来源](https://arxiv.org/html/2509.12145v1) · [出版／作者入口](https://openaccess.thecvf.com/content/ICCESkill025/html/Kang_Open-ended_Hierarchical_Streaming_Video_Understanding_with_Vision_Language_Models_ICCV_2025_paper.html)

阅读范围：v1 §3、§4.1。

**原方法：** 流式 RNN 检测多层边界，稀疏调用冻结 VLM，结合记忆描述事件；训练使用层次边界标注。

**保留价值与边界：** 保留未来人类长视频中候选边界与语义命名的分工。仅评估视频理解，没有机器人控制迁移证据；当前不增加在线事件检测器。

<a id="knowing_when_to_stop"></a>

## Knowing When to Stop：动作执行前缀

[论文来源](https://arxiv.org/html/2609.00908v3)

阅读范围：v3 §II-B、§III、§IV。

**原方法：** 以动作对 VLM 的 cross-attention 熵及持续高熵平台选择本次动作执行前缀，不额外训练或采样。

**保留价值与边界：** 保留训练 Event、预测动作块和实际执行前缀是不同时间单位这一概念。熵是经验代理，不是成功标签；当前仍采用基座配置的执行前缀。

<a id="atom_bench"></a>

## ATOM-Bench：原子与组合能力背景

[论文来源](https://arxiv.org/html/2606.16826v1) · [出版／作者入口](https://flageval-baai.github.io/AtomBenchPage/)

阅读范围：v1 §2–4、§6；未核查评分代码。

**原方法：** 在两个平台组织原子任务与保留组合任务，使用过程得分及原子／组合差距描述表现。

**保留价值与边界：** 保留单项能力与任务组合的评价语义。差距指标不是组合导致失败的因果比例；当前主指标仍为平均任务成功率，不增加该基准或派生指标。

<a id="lotus_continual"></a>

## LOTUS：视觉分段与持续扩展技能库

[论文来源](https://arxiv.org/html/2311.02058v4)

阅读范围：第 4 版 §III–V。

**原方法：** DINOv2 层次分段后增量划分技能，按 silhouette 判断复用或新建，配合子目标、高层选择与回放。

**保留价值与边界：** 保留边界发现、技能身份和技能库增长的区别；§IV-A 明确承接 BUDS。与 3D-LOTUS++ 是不同工作；当前固定技能词表，不引入增量分区或回放。
