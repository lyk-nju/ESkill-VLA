# SuSIE 的未来子目标图像与控制

> **当前定位（2026-10-28）：归档 · 历史归档。** 未来视觉子目标与当前 target grounding 不同；保留历史卡，放世界模型/未来目标旁支。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

SuSIE 是“未来效果图像能否代替显式语义事件”的重要对照。它的 subgoal 是未来状态，本项目的 target grounding 是当前目标与执行器的空间接地，二者应保持独立定义。

## 论文与阅读范围

- 正式题名：Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models。
- 方法名：SuSIE，Subgoal Synthesis via Image Editing。
- 版本：arXiv 2310.10639v1，2023-10-16；ICLR 2024。
- 来源：[原文](https://arxiv.org/html/2310.10639v1)、[版本记录](https://arxiv.org/abs/2310.10639)、[作者项目页](https://rail-berkeley.github.io/susie/)。
- 阅读深度：主文 §4.1–4.3 与问题定义；未复现。
- 关联：Insight 0、2、3 的相邻路线。

## 原工作的接口与证据

| 项目 | 原文方法 |
| --- | --- |
| 高层 | 微调 InstructPix2Pix，由当前 RGB 和语言生成未来 subgoal RGB |
| 高层监督 | 有语言的视频与机器人轨迹；在未来时间范围内均匀采样目标帧 |
| 低层 | 机器人数据训练 language-agnostic goal-conditioned BC |
| 部署 | 当前图像与生成目标图像驱动控制；执行固定步数后重新生成目标 |
| 坐标接口 | 图像空间；没有以度量几何作为显式中间接口 |

采样未来帧的时间范围和部署刷新步数是超参数，不能直接称为语义事件分段或成功检测触发。论文的零样本任务设置仍建立在预训练和机器人低层数据之上。[§3–4](https://arxiv.org/html/2310.10639v1)

## 对本项目的推论

如果未来目标图像足以传达行为、对象和结果，atomic skill / target grounding / action representation 分解的收益应体现在数据效率、可控性、组合泛化或计算效率中的可测差异。更细的分解可能减少生成图像的代价，也可能丢失未来状态中有用的空间约束。

可用 R 表示 event postcondition 或 future goal，与当前 target grounding 分开。用同一个名称 target grounding 同时表示二者，会掩盖模型到底从哪里获得未来信息，并让训练时 teacher forcing 与部署时预测目标的差异难以追踪。

## 可区分机制的实验

对相同轨迹比较固定时间未来帧、可变长度事件末帧与显式事件语义。统一最大控制时长和重规划预算，分别测可达子目标比例、低层达到子目标比例和完整任务成功。这样能区分生成器失真、子目标太远和低层控制失败。

进一步比较训练期 ROI 重建、部署期未来图像，以及仅保留目标 latent 三种接口。真值未来图像只作上界；主比较应使用生成目标，纳入延迟和生成错误。

后续需检查新对象外观变化时子目标是否保留对象身份，以及目标刷新频率与鲁棒性的关系。与 [CoT-VLA](cot_vla.md) 联读。

返回 [方向论文索引](../../target_grounding/README.md)。
