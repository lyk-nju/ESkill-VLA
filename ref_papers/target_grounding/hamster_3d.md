# 3D HAMSTER 的度量轨迹与坐标一致性

> **当前定位（2026-10-28）：补充 · 按需补充与替代路线。** 作为几何、功能区域或行为诊断的补充；当前主线是三视觉锚点的局部去噪监督。 以 [当前清单](../catalog.json) 与 [方法文档](../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

3D HAMSTER 将 target grounding / action representation 接口的一个具体问题讲清楚：图像上正确的路径，不一定在控制坐标系中正确。重建监督也可以用于保留几何，而不必只重建对象 RGB 外观。

## 论文与阅读范围

- 正式题名：3D HAMSTER: Bridging Planning and Control in Hierarchical Vision Language Action Models through 3D Trajectory Guidance。
- 版本：arXiv 2606.31329v2，2026-07-01；arXiv 记录注明 IROS 2026。
- 来源：[原文](https://arxiv.org/html/2606.31329v2)、[版本记录](https://arxiv.org/abs/2606.31329)、[作者项目页](https://davian-robotics.github.io/3D_HAMSTER/)。
- 阅读深度：主文 §III-A–B 的模型、数据及坐标接口，摘要与评估概览；未复现。
- 关联：Insight 2 与 Insight 3。

## 原工作的接口与证据

| 项目 | 原文方法 |
| --- | --- |
| 高层输入 | 标定的 RGB-D 观测、语言；Qwen3-VL 与专用 depth encoder |
| 中间输出 | u、v、度量深度 d 组成的 waypoint 序列及夹爪状态 |
| 训练监督 | 轨迹与空间数据，部分使用估计深度；额外 dense depth reconstruction |
| 控制接口 | 经相机参数反投影到世界坐标；轨迹加入场景点云，供低层 rectified-flow 策略使用 |

论文指出：直接取二维 waypoint 下方可见表面的深度，会把空中的末端轨迹错误贴到场景表面。所提三维接口针对这一错配，并在轨迹预测、仿真和真实操作上评估。[§III 与摘要](https://arxiv.org/html/2606.31329v2)

## 对本项目的推论

target grounding 的要求应包含“对控制保留足够坐标信息”。只看 ROI 重建质量，可能奖励物体外观，无法检验位置、深度和朝向是否仍能恢复。即使使用 RGB-only 部署，也可以在训练时比较深度或相对位姿辅助监督；这种方案与部署必须配备 RGB-D 的三维策略要单列。

这项工作也提示一种替代解释：三维轨迹的收益可能来自感知与动作坐标匹配，而不是显式事件结构。若本项目在 RGB-only 设置中对比 RGB-D 强基线，应明确新增传感信息，不能把差异全归于 latent 表征。

## 可区分机制的实验

采用两个公平比较组：相同 RGB-only 部署下比较 RGB ROI 与训练期 depth/relative-pose 监督；相同 RGB-D 部署下比较隐式 target grounding、显式三维目标与三维轨迹。另设二维轨迹加表面深度的对照，专门检验悬空末端路径错误。

在同一技能下改变对象距离、相机视角和越障高度，分别测三维端点误差、碰撞率与任务成功。对相机外参扰动做独立测试，可以区分模型几何理解与对标定准确性的依赖。

后续需核验各数据源的真实深度与估计深度占比、消融中深度编码和重建损失的独立作用，以及轨迹刷新频率。与 [HAMSTER](hamster.md) 联读。

返回 [方向论文索引](README.md)。
