# Target Grounding：方法来源索引

当前设计：[docs/target_grounding.md](../../target_grounding.md)。整体实现：[design.md](../../design.md)。本方向关注每臂 Current／Future Object 两组视觉 tokens，以冻结 FLUX VAE 提供目标，双臂共享 Decoder 与可训练潜变量投影头，采用区域加权 MSE。ESkill 的接入与验收状态见当前设计。

论文描述与项目实现分开维护。以下卡片提供机制依据；只有当前设计文档确定的模块、损失和接口才进入 ESkill。

## 当前方法参考

| 工作 | 参考价值 |
| --- | --- |
| [AffordanceVLA：角色 Query 与潜变量直接回归](affordancevla.md) | 当前主要实现参考：Which2Act 的冻结 FLUX VAE、Token 到潜变量网格的投影和直接 MSE。本项目采用当前／终点对象、原位画布与区域加权目标，预测 tokens 经 Motion 参与动作生成。 |
| [ReconVLA：局部视觉 latent 去噪](reconvla.md) | 提供操作区域视觉监督的动机；本项目采用直接 latent 回归，与原文的去噪机制分开。 |

## 按需参考

| 工作 | 保留价值 |
| --- | --- |
| [RoboGround：对象与目的地区分](roboground.md) | 借鉴对象区域为动作策略提供空间条件；其双 mask 接口只作为原论文机制保留。当前事件目标仅监督被操作对象在事件终点的视觉区域。 |
| [3D HAMSTER：末端路径与深度尺度](hamster_3d.md) | 保留悬空末端轨迹与可见表面深度的区别。二维路径像素的背景深度不能直接当末端深度；三维标签需要一致参考系和尺度。 |
| [RIGID：关系信息与特征对齐](rigid.md) | 保留几何信息不必依赖教师特征逐维相等这一认识；与当前画布保留原图布局和空间位置信息的动机相邻。 |
| [SG-VLA：空间监督与梯度路径](sg_vla.md) | 保留前向信息流、辅助梯度和骨干冻结范围的区别，有助于准确阅读当前联合训练接口。 |

## 跨方向关联

[AR-WAM：可学习操作条件](../atomic_skill/ar_wam.md)（操作与空间条件分工）

[背景机制笔记](background.md) 收录其余 12 项工作，供后续调研检索。

返回 [论文库入口](../README.md)。
