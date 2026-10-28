# MimicPlay 的三维手轨迹与潜在计划

> **当前定位（2026-10-28）：归档 · 历史归档。** 从 target grounding 主要阅读队列移除；保留历史卡，跨链到人类视频/运动迁移。 以 [当前清单](../../catalog.json) 与 [方法文档](../../../docs/target_grounding.md) 为准。

> 以下保留此前原文阅读与推论；其中“核心／优先实现／本轮”等表述属于记录时的判断，不替代上述当前定位。

MimicPlay 为 Phase 2 和 Phase 3 提供了具体先例：人类演示可以通过具有三维运动约束的 latent plan 进入机器人控制，但需要明确数据采集和跨域条件。

## 论文与阅读范围

- 正式题名：MimicPlay: Long-Horizon Imitation Learning by Watching Human Play。
- 版本：arXiv 2302.12422v2，2023-10-13；CoRL 2023。
- 来源：[原文](https://arxiv.org/html/2302.12422v2)、[版本记录](https://arxiv.org/abs/2302.12422)、[作者项目页](https://mimic-play.github.io/)、[官方代码](https://github.com/j96w/MimicPlay)。
- 阅读深度：主文 §3.1–3.3 与 §4 任务设置；代码仅核对使用范围，未复现。
- 关联：Insight 2、3 与后续 Context；latent plan 不是已解耦的原子技能。

## 原工作的接口与证据

| 项目 | 原文方法 |
| --- | --- |
| 人类数据 | 同环境 human play；两台标定相机检测并重建三维手轨迹 |
| 高层训练 | 当前图像与目标图像编码为 plan，以 GMM 预测手轨迹；跨域视觉分布对齐 |
| 低层训练 | 冻结 planner，使用机器人遥操作训练；输入含 plan、腕部图像和 proprioception |
| 部署提示 | 人或机器人的视频帧序列提供目标图像，生成 latent plan 后输出机器人动作 |

真实实验涉及同环境人类数据与少量机器人数据；官方仿真复现使用 robot play，不能将两者合并成任意互联网 RGB 视频迁移的证据。[§3–4](https://arxiv.org/html/2302.12422v2)、[官方实现范围](https://github.com/j96w/MimicPlay)

## 对本项目的推论

“无机器人动作标签”不等于“没有运动几何监督”。三维手轨迹比纯帧间重建更明确地约束动作信息，适合作为 action representation 的替代 teacher。代价是多视角标定、检测误差和数据域限制，不能只按视频分钟数比较数据成本。

如果目标是跨身体形态，手轨迹中哪些成分可转给夹爪、哪些依赖手部接触方式，需要单独测试。物体相对运动、末端路径和接触时序可能具有不同迁移性。三维计划也同时含目标和运动，尚不足以证明 atomic skill 与 target grounding 已分离。

## 可区分机制的实验

将 Phase 2 数据分成机器人视频、同环境人类 play、跨环境人类视频三个层次；在固定机器人微调量下比较帧变化 teacher、二维手轨迹 teacher、三维手轨迹 teacher。分别报告跟踪覆盖率、几何误差和任务成功，避免只看最终平均值。

对 Phase 3，可将视频按帧推进作为基线，再比较根据当前状态检索相关事件的 Context 对齐。人为改变演示速度并加入中途停顿，检验模型是否依赖演示时间表。这里的事件对齐是项目实验提案，不是 MimicPlay 已有结论。

后续需追问：跨域视觉对齐是否隐藏行为差异、手轨迹噪声容忍度、latent plan 对未见事件转移的有效性。

返回 [方向论文索引](../../target_grounding/README.md)。
