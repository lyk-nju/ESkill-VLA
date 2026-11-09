# RoboTwin 统一 Episode 接入核验

2026-11-04：已有 clean 50×50 原始轨迹；本轮完成固定十条的统一导出、内容回读与视频渲染。未启动训练，未进行 2,500 条全量转换，人工验收 pending。

| 产物 | 入口 |
| --- | --- |
| 十条保存数据 | `/data1/yuankai/robot-research/data/robotwin/check/episode_v1/index.json` |
| 十条回读视频 | [video/](../video/) · [视频清单](../video/episodes.json) |
| 人工核验 | [manual_review.csv](manual_review.csv)，所有结论待填写 |
| 来源盘点 | [source_audit.json](source_audit.json)：2,500 条配额；每任务一条，共 50 条数值检查 |
| 内容逐项比对 | [readback.json](readback.json)：十条、2,018 观测、2,008 动作、8,072 张 RGB 全部通过 |
| 数据加载检查 | [interface_report.json](interface_report.json) · [配置](interface_config.json) |

样本沿用 `data/robotwin/processed/clean_manifest.json`，没有重抽。来源为 `/data1/yuankai/robot-research/data/robotwin/clean_native/`，包含 `supplement/`。逐项检查覆盖所选源行的 RGB 像素、状态、动作、时间戳、EEF、区域与相机参数；渲染仅回读保存 Episode，不再读源 HDF5。57 项相关回归测试通过。

## 本轮数据约定

同一物理时刻只保留最后一条观测，记录源行号与物理步。当前观测配下一次独立观测的 14 维绝对关节目标，并与对应物理时刻前最后执行的控制目标核对。N 个观测产生 N−1 个动作；末帧保留为真实终点。间隔使用真实时间，不把播放 15 fps 当作控制频率。

状态是左右各 6 个实测关节位置＋夹爪命令；动作是左右各 6 个关节驱动目标＋夹爪命令，分别为 rad 和归一化 0 关／1 开。四相机图像使用无损 PNG 字节保存在不压缩 `images.npz`。EEF 保留米制世界坐标及源 `get_ee_pose(is_endpose=False)` 参考点定义；不能改名当作 LIBERO 的 grip_site。

纯 π0.5 和 ESkill 使用同一 EpisodeDataset，均检查到 2,008 个动作样本；三相机输入为 `[B,3,3,224,224]`，H=10 的动作目标补至 `[B,10,32]`。ESkill 启用状态分支时为 `[B,2,15]`，每臂只有前 7 个来源维度有效，其余填充不冒充实测状态。当前八点 Motion 使用去重观测步，间隔不是固定秒数；正式闭环控制协议仍需另外验证。

本目录 `norm_stats.json` 仅供这十条接口检查，**不是正式 train-only 统计**。正式训练之前按每任务父轨迹 90%／10% 划分并重新计算统计。本轮没有加载基座做训练，也没有验证闭环成功率。

## 字段与视频解释

| 我们需要什么 | 有没有 | 备注 |
| --- | --- | --- |
| RGB／指令 | 有 | 主相机、双腕、额外前视图；源语言变体保留 |
| 状态／动作 | 有 | 14 维；下一独立观测目标，保留实际时间间隔 |
| 每臂 Event／技能 | 部分 | 源原语保留作证据；正式 events 为空，当前与未来技能标签无效 |
| 夹爪完整屏内框 | 有 | 主相机左右臂分别读取原生 full_in_frame 框 |
| 当前物体完整区域 | 部分 | 只有源实例可见框；完整框和每臂操作实例关系待补 |
| Event 终点对象区域 | 待补 | 依赖正式 Event 和同一实例完整框，不用可见框代替 |
| EEF／相机标定 | 有 | 保存位姿、参考点和标定；共用导出器生成主相机二维坐标 |

视频蓝色实线是左夹爪完整框，黄色实线是右夹爪完整框。绿色虚线是**所有源候选物体的可见框**，不表示当前 manipulated object，也不用于正式 Grounding 监督。下方显示源原语、真实时间、状态和下一动作目标，Event／skill 明确标为 pending。视频仅显示主相机及双腕图像，腕部不画 Grounding 框。

下一步先人工核验这些保存结果，再整理每臂 Event／技能、实例关系与物体完整区域。全量导出入口已实现，命令见 [examples/robotwin](../../../../../examples/robotwin/README.md)。
