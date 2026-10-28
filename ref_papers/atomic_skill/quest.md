# QueST：动作压缩码不自动等于语义技能

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为技能发现、编码或条件接口的扩展比较；当前先采用轻量文本原型与离散技巧条件。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/atomic_skill.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

**论文：** *QueST: Self-Supervised Skill Abstractions for Learning Continuous Control*，Atharva Mete 等，NeurIPS 2024。arXiv:2407.15840；本轮方法核对版本 v3（2024-09-11），调研日期 2026-10-27。

**原始来源与阅读范围：** [论文 §3.2、§4.1–4.3](https://arxiv.org/html/2407.15840v3)、[正式会议页面](https://papers.nips.cc/paper_files/paper/2024/hash/076c3e48fa502c660902105965fdd9f6-Abstract-Conference.html)、[作者项目页](https://quest-model.github.io/)。已核对方法与部分实验设定，未检查代码或复现。

## 原工作做了什么

第一阶段只编码动作序列，用因果时序网络和有限标量量化 FSQ 得到多个离散 token，再重建动作。第二阶段以观测历史、本体信息及任务条件训练自回归 skill prior。推理时预测 token、解码动作序列，执行前缀后重规划。论文明确指出单个码未必有独立含义，码序列共同表示运动。新任务实验包含 5 条示范微调，不能算作无更新的 in-context 学习。

## 对 atomic skill / target grounding / action representation 的意义与边界

**归档到 atomic skill，是因为它提供了必须正面对比的“skill”定义。** QueST 的 state-independent action abstraction 值得借鉴，但更接近本项目的运动表示 action representation；“编码器不输入图像”无法保证码本具有跨对象语义不变性。动作轨迹中的位置、幅度、接近方向仍可能间接暴露对象或场景。

我们没有理由要求一个动作码等价于 pick 或 place；同样也不能直接把一串 FSQ token 宣称为 atomic skill。合理比较应包含三个接口：纯动作 token、显式语义 atomic skill、atomic skill 条件下的动作 token。atomic skill 与当前 target grounding 仍可并行预测。

## 可以推翻本项目假设的实验

在同一动作坐标系、相同时间窗和相近 token/参数预算下，比较上述三个接口。训练集交叉覆盖 skill 与 object，测试保留未见组合及轨迹方向变化；同时报告动作成功率、同技能跨对象编码一致性、同对象跨技能区分度。

如果纯动作 token 已达到相同组合成功率，而显式 atomic skill 没有额外收益，就应缩小“独立语义接口必要性”的主张。新任务实验分别报告冻结 decoder 与更新 decoder，避免把适配收益归因于已有技能可组合。

## 下一步精读

确认预训练与新任务阶段各模块冻结范围、动作归一化和执行前缀；检查码序列长度与控制收益的关系。优先判断多 token 的容量在承担语义、运动细节还是两者，而非先增加一个语义 loss。

相关：[STAR](../_archive/atomic_skill/star.md)、[VQ-BeT](vq_bet.md)、[方向论文索引](README.md)。

## 两报告复核补充（2026-10-27）

[Appendix C](https://arxiv.org/html/2407.15840v3) 对观察条件损害量化的解释是作者假说；actions-only 动作码仍可能携带物体几何与轨迹信息。其 FSQ 消融不证明 FSQ 在所有 VLA 上全面替代 VQ，也不证明码本就是纯语义 grasp/place。详见 [复核](evidence.md#review)。
