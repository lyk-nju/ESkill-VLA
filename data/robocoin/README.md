# RoboCOIN Agilex 子集 数据调研

状态：初步调研，尚未确定采用。以下内容迁自公共来源索引，核实日期沿用 2026-10-29；本次仅整理目录。

[样例官方数据卡](https://huggingface.co/datasets/RoboCOIN/Agilex_Cobot_Magic_heat_burger) · [项目页](https://flagopen.github.io/RoboCOIN/)

- 本次核对样例 `Agilex_Cobot_Magic_heat_burger`：819 条、30 fps、LeRobot v2.1，Apache-2.0。该样例信息不外推到整个组织。
- 数据卡明确 state/action 的 26 维按左右臂排列，每臂为 6 维关节、3 维末端位置、3 维 Euler 姿态、1 维夹爪。另有 `eef_sim_pose_state/action[12]`；参考点、坐标系与两套位姿的关系仍待核实。
- `subtask_annotation[5]` 与独立 JSONL 的字段解释尚待样本审计。子任务名称包含 Left hand / Right hand，但不能据此确认并发的分臂事件流。
- 数据卡的 split 区间存在重叠且超出所报条目范围，接入时须核对实际元数据，不直接采用展示值。

**初步评估：待样本审计的双臂候选。** 有末端字段线索；有效事件覆盖、动作语义、对象区域来源及是否适合接入，都有待进一步调查。
