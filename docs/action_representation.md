# Action Representation：最长两秒的 Event 内交互计划

本文规定 ESkill 的 Motion 表征与监督，整体结构见 [Design](design.md)。

## 表征与时间尺度

Motion 将 Skill Prior、Current／Future Object、VLM 上下文和当前 EEF／夹爪状态结合，预测朝当前 Event 目标推进的局部行为。每臂使用 **8 个连续 256 维 tokens**，对应固定未来时间：

| 槽位 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 偏移（秒） | 0.25 | 0.50 | 0.75 | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 |

槽位有固定时间编码，不按 Event 长度缩放。Future Object 是完整 Event 终点；如果 Event 尚需四秒，第八个 Motion 点仍只表示两秒内的位置。

上述秒级定义以原始时间轴核验为前提。当前 AtomicVLA 导出保存 source steps，时间戳／频率尚不完备；去除 no-op 后必须恢复来源索引与控制时钟关系，不能把训练行号或视频 fps 当物理时间。

## 输入与联合解码

当前输入是 EEF 三维位置、姿态及实际夹爪状态，通过本体适配进入 Motion。当前 EEF 位置经主相机标定、方向变换和 resize／pad 得到归一化二维起点 `p0`。

Skill 从 512 投影到 256 维，对每臂 Motion 做 FiLM。双臂 Query 联合 self-attend，允许协作；单臂任务只保留真实臂。Current／Future Object tokens 及 VLM 上下文提供对象状态与场景条件。

同一个未来 token 使用两个共享读出头：

| 读出 | 输出 | 标签与损失 |
| --- | --- | --- |
| `trace_head` | `delta_xy[B,2,8,2]`，预测点为 `p0 + delta_xy` | 主相机归一化二维 EEF 坐标，SmoothL1 |
| `gripper_head` | `logits[B,2,8,1]` | 同时刻的未来 Open／Close 控制命令，BCEWithLogits |

当前起点不占未来八个 token。当前真实夹爪开度是输入，未来命令是标签，不能混为物理状态或抓取成功标签。LIBERO 的控制符号约定为负值打开、正值闭合；实际保存值与命令相对观测的执行时序仍须逐来源核验。

EEF 与 Gripper 查询同一物理时间，各自保留有效性。离散开合命令不能线性插值；连续坐标若需插值，只能使用时间已核验、同一 Event 内且连续有效的记录，不能跨缺测或事件边界填值。具体匹配／容差策略需在时间轴核验后固定并保存来源索引。

## Event 边界与缺失标签

每臂独立判断查询是否位于当前 Event。只监督当前 Event 内的真实有效位置与命令，不借用后续 Event 的标签。Event 记录不保证包括释放或稳定落位，因此不能仅凭 Place 标签补造一次开爪动作。

分别保留：

- `observed_mask`：能与真实有效记录对齐的查询。
- `terminal_padded_mask`：为显示或可选弱监督重复终点的查询。
- Event 尾部、Episode 尾部、内部缺测与出画等原因。
- 独立的 `xy_valid` 与 `gripper_valid`；坐标出画不自动使命令无效。

默认终点填充不参与主路径／夹爪损失，避免短 Event 被大量静止目标主导。先统计没有任何有效未来查询的样本比例、各偏移覆盖率和短 Event 分布，再决定是否对首个越界槽位使用弱终点监督。

真实边界与标签有效性只控制监督，不能用于隐藏 Action 中的未来 Motion token。模型推理时不知道 Event 终止真值。无辅助监督的槽位仍通过 Action FM 学习，其解码结果在核验图中需标明未监督。

## 坐标与损失

当前起点和未来点使用相同的末端参考点、三维坐标系、相机投影、翻转及 resize／pad。物理投影使用未被训练归一化破坏的状态。VLM／路径画布为 224×224，Grounding 为 256×256；禁止直接混用两者像素坐标。

路径损失先按每点二维坐标求均值，再按有效点数归一化；像素误差在 224×224 模型画布上计算。夹爪损失按其独立有效命令数归一化。分布式训练与验证分别累计全局分子、分母，无有效标签不输出伪造的准确率。

诊断保留路径像素误差、开合准确率／BCE、开合转换附近表现，以及各偏移的有效样本数。整体辅助误差下降不等于动作策略已利用这些信息。

## Motion 到 Action

Action 的唯一新增结构化条件是 `Z_motion[B,2,8,256]` 经逐层投影得到的 Auxiliary K/V。两个读出头只提供辅助监督与可视化；高维 latent 可保留未被二维坐标与开合标签表达的操作信息。

条件不 detach，Action FM 可沿 Motion 更新 Skill 与 Grounding 的连续路径。纯动作推理可跳过两个读出头。原始视觉语言 K/V 仍保留，训练时通过 Action-side Patch Masking 鼓励使用 Motion；这不构成强制的信息瓶颈。

LIBERO 保持动作窗口 10 步、测评执行前缀 5 步。Motion 的两秒跨度、Event 的实际长度与动作块长度分别管理，不为凑齐尺度修改动作配方，也不默认按 Event 裁掉 FM 标签。

## 参考与验证

[SeeTraceAct](https://arxiv.org/html/2606.02745v2) 提供按未来时刻监督规划 token、以 latent 条件化动作的参考。本项目增加每臂 Event 范围、八个固定时间查询及夹爪命令监督；不将其原方法等同于本项目实现。

验收除轨迹与开合标签外，还要做同状态、同噪声的 Motion Donor 干预：检查执行 EEF 是否按目标方向变化、抓放时机是否改变，再测新 Object–Goal 组合闭环成功率。协议见 [Evaluation](evaluation.md)，其他方法见 [参考索引](ref_papers/action_representation/README.md)。
