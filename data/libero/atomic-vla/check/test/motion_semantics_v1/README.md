# Motion 几何语义核验

2026-11-02：AtomicVLA 固定 RLDS 发布版的末端位置采用 `unit=m`、`coordinate_frame=world`、`reference_point=gripper0_grip_site`。依据是 OpenVLA 采集、四套 RLDS 转换及 robosuite Panda 的参考点定义；固定源码链接随每条位置记录的 `evidence` 保存。

导出器填写以上语义；Dataset 在原端点、连续性、数值检查前增加三个字段检查。不满足时只屏蔽该臂的 Motion 目标。其他单位或坐标系的转换、末端参考点定义的统一由来源适配负责，训练入口不猜测或转换。

现有 **1,693 条／273,465 帧**已回填元数据。XYZ、动作、bbox／mask 的 NPZ 哈希全部不变，Event、RGB 引用、索引及人工验收状态不变；740 条场景状态对齐警示保留。原 `action_and_eef_semantics_pending` 改为 `action_semantics_pending`，不表示动作语义或物理时间已确认。

验证：30 项相关回归通过；真实样本 0145 重新导出后所有数组一致；10 条轨迹的 30 个窗口与修复前 Dataset 的输入、监督数值和样本元数据逐项一致，实际手臂 16 个 Motion 点均有效。

[核验记录与修改前后 SHA256](report.json) · [修改前 JSON 备份](/data1/yuankai/robot-research/data/atomicdata/libero40/metadata_revisions/motion_semantics_v1/episodes_before.tar.gz)。备份中的路径相对 `libero40/`。`full_frame_v1/checks/` 的历史 JSON 哈希与此备份全部匹配；历史核验文件保持原样，当前 JSON 哈希以本记录为准。
