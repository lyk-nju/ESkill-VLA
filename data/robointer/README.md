# RoboInter-Data 数据调研

状态：初步调研，尚未确定采用。以下内容迁自公共来源索引，核实日期沿用 2026-10-29；本次仅整理目录。

[HF](https://huggingface.co/datasets/InternRobotics/RoboInter-Data) · [arXiv 2602.09973](https://arxiv.org/abs/2602.09973) · [GitHub](https://github.com/InternRobotics/RoboInter)

- 发布条目数：235,920（DROID 152,986，RH20T 82,894），仓库 408 GB。条目数不能直接当作独立轨迹数；本地 DROID chunk-000 的 1,000 个条目对应 500 条轨迹的两个外部视角。
- 许可：CC BY-NC-SA 4.0；数据卡有申请表单。
- 本体：DROID 部分为 Franka + Robotiq，单臂。
- 格式：推荐 `Annotation_with_action_lerobotv21`（LeRobot v2.1，10 fps，按 chunk 打成 tar）。DROID 175.7 GB，RH20T 93.6 GB。另有 `Annotation_pure`（LMDB，37.3 GB）、`Annotation_raw`（pkl 与 `segmentation_npz`，87.1 GB）、`Annotation_demo_app`（20 条，0.97 GB）。
- 状态与动作：`state` 是 7 维末端位姿 `[x,y,z,rx,ry,rz,gripper]`，`action` 是 7 维 delta EEF；`other_information.*` 还有 tcp 位姿和关节。相机 primary 与 wrist，180×320。
- 标注字段（JSON 字符串）：`time_clip`、`instruction_add`、`substask`、`primitive_skill`、`segmentation`、`object_box`、`placement_proposal`、`trace`（未来 10 步二维夹爪像素轨迹）、`gripper_box`、`contact_frame`、`state_affordance`（接触时 6D 末端位姿）、`affordance_box`、`contact_points`、`origin_shape`。每项另有 `Q_annotation.*` 质量等级（Primary／Secondary）。框为像素坐标，mask 单独存于 `segmentation_npz`。
- 未公开完整的 `primitive_skill` 词表，数据卡只举了 pick、place、push、reach 等例子。

**初步评估：已有局部审计的候选来源，尚未确定接入方案。** 单臂；`state` 可作为三维路径来源进行研究。已审计的 LeRobot 分片没有实际 mask，`placement_proposal` 等于终点对象框，可在身份、时刻及区域质量核实后构造事件目标；夹爪框存在背景误标。动作增量的参考系与旋转语义仍待核实；主 state/action 与本地 OpenPI 关节 DROID 适配器的接口不同。如何映射技能、使用部分监督或补标仍待比较，许可仅限非商用。
