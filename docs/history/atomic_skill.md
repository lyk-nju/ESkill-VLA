# Atomic Skill：连续视觉技能与语义专家

> 历史联合动作路线说明，整理于 2026-11-18。仅用于对应动作模型与实验的追溯；当前 Current／Subject 视觉专项见 [当前设计](../design.md)。本文中的训练阶段与 Future 不属于当前 Grounding 入口。

本文规定 ESkill 的 Skill 分支；总体信息流与实现状态见 [Design](joint_design.md)。

## 职责

Skill 表示可跨对象与目标复用的操作知识。每臂一个连续 Query 从当前图像与指令中提取上下文，通过显式技能类别组织专家参数，再输出随观察变化的 Skill Prior。该 prior 调制 Motion，由 Motion 将技能实例化为具体交互行为。

“一类一个专家”是结构归纳偏好，不保证专家自动剥离对象外观或获得组合泛化；这些能力需要独立验证。

## 标签与路由

| 全局 ID | 技能 | 首版用途 |
| --- | --- | --- |
| 0 | Pick | 活跃类别与专门化专家 |
| 1 | Place | 活跃类别与专门化专家 |
| 2 | Open | 活跃类别与专门化专家 |
| 3 | Close | 活跃类别与专门化专家 |
| 4 | Turn | 活跃类别与专门化专家 |
| 5 | Push | 活跃类别与专门化专家 |
| 6 | Hold | 保留编号，当前不启用 |
| 7 | Idle | 保留编号，当前不启用 |
| 8 | Return | 保留编号，当前不启用 |

九类是数据词表，六类是当前可训练路由空间，两者显式映射。Shared Expert 始终启用，不是第七个语义类别。未知、缺失与未支持标签保持原义，不补成 Idle，不映射成最接近的已支持类别。

当前 AtomicVLA 全量 Event 核对为 Pick 1,801、Place 1,800、Open 79、Close 69、Turn 91、Push 33；这些是全量 Event 数，不是 train 独立统计，也不是帧采样频率。训练前按固定 split 另统计专家实际获得的标签量。

## 小型专家默认配置

- VLM 上下文适配到 512 维，每臂一个连续 Skill Query；左右臂共享参数。
- 可训练分类器输出六类 logits，以真实当前技能计算 CE。
- 训练与推理都按预测 Hard Top-1 路由，避免训练时依赖真值选择。
- 六个专门化专家加一个 Shared Expert；每个专家宽度 512、中间宽度 2048、两层残差 SwiGLU FFN。
- Query hidden 与两个专家输出残差组合、归一化，得到 `z_skill[B,2,512]`。
- 显式 `Linear(512,256)` 适配到 Motion，用于每臂 FiLM。

七个专家的 FFN 权重合计约 4,404 万，不包含 Query Decoder、分类器与适配器。先用这个小规模验证机制，容量扩展另做配置，不在首版堆叠大专家。

## 梯度与监督边界

Skill CE 更新分类器和连续 Query。Motion／Action 损失经过选中专家及 Shared Expert 更新连续特征；Hard Top-1 的离散选择不可微，因此不能声称 FM loss 直接训练分类器的选择。

技能标签只在损失侧使用。辅助标签缺失不关闭模型的预测分支，缺失臂则由在线可知的 presence mask 排除。六类 softmax 无法识别未训练的 Hold／Idle／Return；未来引入这些数据时再扩展活跃类别及专家。

记录当前技能准确率、每类样本数、预测路由频率及专家梯度。类别不均衡是真实语义分布，不默认用强制均匀负载损失改写它。

## 与其他模块的关系

Skill Prior 只进入 Motion；对象的当前状态与完整 Event 目标由独立 Grounding 分支预测。二者共同决定局部路径与开合行为，Action 通过高维 Motion 接受这些条件。

[AtomicVLA](../ref_papers/atomic_skill/atomic_vla.md) 提供语义路由与共享／专门化专家的参考；本项目将专家放在 VLM 输出端。它与原论文修改 Action Expert 内部 FFN 的位置不同。[w²VLA](../ref_papers/atomic_skill/w2vla.md) 的条件调制机制可供参考，但不代表本项目的具体 FiLM 与遮蔽接口已得到验证。

更多原方法见 [参考索引](../ref_papers/atomic_skill/README.md)。
