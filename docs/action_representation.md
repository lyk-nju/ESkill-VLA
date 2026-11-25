# Action Representation：固定帧间隔的局部交互计划

Motion Planner 的八个槽位各预测一个主相机画布上的二维末端点，使用下文的固定帧偏移和 Event 边界规则。文中的双臂 FiLM、夹爪命令头及 Motion 到 Action 接口属于历史联合动作模型。[Motion Planner 设计](design.md)

## 表征与时间尺度

Motion Planner 用预测技能选择专家，用 Current 注意力引导视觉读取。输出的 **8 个 256 维 tokens** 分别对应以下保存帧偏移。

| 槽位 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 偏移（保存帧） | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |

槽位编码固定的帧偏移，不按 Event 长度缩放。第八个 Motion 点查询未来第 40 个保存帧。Motion Planner 的两份配方都启用终点保持，监督窗口提前结束时，后续二维点保持窗口终点。

第一版直接按保存主相机的帧序列采样，不要求精确物理时间戳。LIBERO 默认控制频率为 20 Hz：按连续控制步估算，5 帧约 0.25 秒，40 帧约 2 秒；这只是量级参考，不写入时间戳或声称全部来源等频。去除 no-op 的索引关系仍保留，渲染 fps 只控制播放速度。

## 输入与联合解码

历史联合动作模型的 Motion 输入来自 VLM、Skill Prior 与预测对象 tokens。固定 arm embedding 表示臂身份，不携带真实机器人状态。不输入当前 EEF 位姿、实际夹爪开度、二维投影起点或相机标定。

Skill 从 512 投影到 256 维，对每臂 Motion 做 FiLM。双臂 Query 联合 self-attend，允许协作；单臂任务只保留真实臂。Current／Future Object tokens 及 VLM 上下文提供对象状态与场景条件。

同一个未来 token 使用两个共享读出头：

| 读出 | 输出 | 标签与损失 |
| --- | --- | --- |
| `trace_head` | `xy[B,2,8,2]`，直接预测绝对二维图像坐标 | 主相机归一化二维 EEF 坐标，SmoothL1 |
| `gripper_head` | `logits[B,2,8,1]` | 同时刻的未来 Open／Close 控制命令，BCEWithLogits |

八个 token 均对应未来点。当前真实夹爪开度不作为输入，未来命令仅作为标签，不能混为物理状态或抓取成功标签。LIBERO 的控制符号约定为负值打开、正值闭合；实际保存值与命令相对观测的执行时序仍须逐来源核验。

EEF 与 Gripper 使用同一组查询偏移，各自保留有效性；终点保持只改变二维点的标签来源。以主相机帧的序列位置计算偏移，再映射到 Episode 公共帧索引；不按过滤前跳号累加，也不受 DataLoader 的 `frame_stride` 影响。不插值或借用附近命令，缺失的 EEF／命令分别屏蔽。

## Event 边界与缺失标签

每臂独立使用当前 Event＋最短窗口：`matched & (same_event | offset <= motion_min_horizon_frames)`，默认最短窗口为 10 个保存帧。首个查询仍是 +5 帧；Event 结束太近时，+5、+10 两点允许使用边界之后的真实轨迹与命令。真实观测查询不越过 Episode。

Motion Planner 配方开启 `motion_terminal_hold=true`。设当前相机行为 `r`、当前 Event 内最后保存相机行为 `e`、最短窗口为 `m`，则二维轨迹使用的终点行为 `h=min(max(e,r+m),last_episode_row)`，第 `k` 个点取 `min(r+offset[k],h)` 的 EEF 投影。

| 情况 | 八个二维点的来源（相对当前保存帧） |
| --- | --- |
| Event 最后观察在 +20 | `5,10,15,20,20,20,20,20` |
| Event 最后观察在 +18 | `5,10,15,18,18,18,18,18`，使用真实终点，不回退到 +15 |
| Event 在 +3 结束，最短窗口为 10 | `5,10,10,10,10,10,10,10`，先保留最少两点真实未来 |
| Episode 仅剩 3 帧 | `3,3,3,3,3,3,3,3`，全部标为派生停止目标 |

终点遵循 `[start,stop)` 内最后保存观察；不改变 Future Object 独立的 `goal_frame`。Event 过短时保持最短窗口的终点，不能在跨界预测之后又跳回旧 Event 终点。Episode 最后一帧也可作为保持位置。终点投影缺失或出画则不产生有效保持点；未知 Event 不补造边界。

按名义 20 Hz，10 帧约为 0.5 秒；过滤 no-op 后不声称严格的物理时间。最短窗口是允许监督的范围，不保证图像内投影及记录有效。设 `motion_min_horizon_frames=0` 可复现严格 Event 边界对照。

跨界点保留 `future_cross_event`；`future_event_tail=true` 表示真实越过当前 Event，不再等价于该点被排除。当前 Skill 与 Future Object 仍取当前 Event，局部 Motion 因而可包含下一 Event 的过渡。夹爪使用未来真实命令，不能凭 Place 或其他技能名称补造开爪动作。

分别保留：

- `observed_mask`：位于允许监督窗口内的已有相机序列查询；具体 EEF／命令有效性另行判断。
- `future_cross_event`：在最短窗口内获准越过当前 Event 的查询。
- `terminal_padded_mask`／`future_padded`：重复终点的派生查询。
- `trace_terminal_hold`：明确参与二维轨迹 loss 的派生停止目标，与真实 `observed_mask` 互斥。
- `trace_source_frame_indices`：二维标签的来源帧；`future_source_frame_indices` 仍只标真实未来查询（用于命令等）。
- Event 尾部、Episode 尾部、内部缺测与出画等原因。
- 独立的 `xy_valid` 与 `gripper_valid`；坐标出画不自动使命令无效。

有效保持点与真实点以相同权重计入 SmoothL1，并共同参与有效点数归一化；没有额外的停止分类 loss。夹爪仍只学习允许窗口中的真实命令，不重复或猜测开合。关闭 `motion_terminal_hold` 可复现旧配方：越界二维点不监督；旧 checkpoint 缺少该字段时默认为 false。内部缺测始终不作终点保持。

真实边界与标签有效性只控制监督，不能用于隐藏 Action 中的未来 Motion token。模型推理时不知道 Event 终止真值。无辅助监督的槽位仍通过 Action FM 学习，其解码结果在核验图中需标明未监督。

## 坐标与损失

未来点监督由离线末端参考位置经过相机投影、翻转及 resize／pad 得到；这些坐标不进入模型输入。每个未来点按自身投影有效性监督，不依赖当前帧投影是否有效。VLM／路径画布为 224×224，Grounding 为 256×256；禁止直接混用两者像素坐标。

路径损失先按每点二维坐标求均值，再按有效点数归一化；像素误差在 224×224 模型画布上计算。夹爪损失按其独立有效命令数归一化。分布式训练与验证分别累计全局分子、分母，无有效标签不输出伪造的准确率。

诊断保留路径像素误差、开合准确率／BCE、开合转换附近表现，以及各偏移的有效样本数。启用保持时整体路径误差包含真实点与派生点，本地另记 `motion_observed_l2_px` 和 `motion_terminal_hold_l2_px`；审计分别统计两类覆盖率。整体辅助误差下降不等于动作策略已利用这些信息，终点保持也不直接等价于 Action 停止指令。

## Motion 到 Action

Action 的唯一新增结构化条件是 `Z_motion[B,2,8,256]` 经逐层投影得到的 Auxiliary K/V。两个读出头只提供辅助监督与可视化；高维 latent 可保留未被二维坐标与开合标签表达的操作信息。

条件不 detach，Action FM 可沿 Motion 更新 Skill 与 Grounding 的连续路径。纯动作推理可跳过两个读出头。原始视觉语言 K/V 仍保留，训练时通过 Action-side Patch Masking 鼓励使用 Motion；这不构成强制的信息瓶颈。

LIBERO 保持动作窗口 10 步、测评执行前缀 5 步。Motion 的 40 保存帧跨度、Event 的实际长度与动作块长度分别管理，不为凑齐尺度修改动作配方，也不默认按 Event 裁掉 FM 标签。

## 参考与验证

[SeeTraceAct](https://arxiv.org/html/2606.02745v2) 提供按未来时刻监督规划 token、以 latent 条件化动作的参考。本项目增加每臂 Event 范围、八个固定帧偏移查询及夹爪命令监督；不将其原方法等同于本项目实现。

验收除轨迹与开合标签外，还要做同状态、同噪声的 Motion Donor 干预：检查执行 EEF 是否按目标方向变化、抓放时机是否改变，再测新 Object–Goal 组合闭环成功率。协议见 [Evaluation](evaluation.md)，其他方法见 [参考索引](ref_papers/action_representation/README.md)。
