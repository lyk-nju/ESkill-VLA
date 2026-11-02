# 全量相机参数与八点 Motion

2026-11-02：40 任务、1,693 条／273,465 帧全部完成保存和训练接口回读核验。

训练索引：[/data1/yuankai/robot-research/data/atomicdata/libero40/motion_2d/index.json](/data1/yuankai/robot-research/data/atomicdata/libero40/motion_2d/index.json)。[train.json](../../../../../../examples/libero/train.json) 已使用这个索引。

| 文件 | 内容 |
| --- | --- |
| `motion_2d/episodes/<id>/episode.json` | 引用原 RGB、状态、动作、Event 和区域；补充 `cameras.image.calibration` |
| `motion.npz` | 主相机内参、world→CV 相机外参、图像方向变换；逐帧二维末端及八点目标、有效性、真实／补齐标识 |
| `motion.json` | 来源哈希、相机约定、逐帧运动学核验、场景栅格检查依据 |

每臂八点取未来 `[4,8,12,16,20,24,28,32]` 个来源帧，归一化到模型图像坐标；允许跨 Event。超出 Episode 的位置使用有效的末帧位置保持，原轨迹不降采样。

| 检查 | 结果 |
| --- | --- |
| 相机覆盖 | 1,693／1,693 条，四种固定相机配置 |
| 源关节→末端，逐帧核对 | 全部通过；最大偏差 0.728 mm，投影偏差 0.317 px |
| 相机矩阵随轨迹变化 | 0 |
| 原生栅格投影 | 40 任务各三帧，共 120 帧；最大误差 0.385 px |
| Dataset 逐帧回读 | 273,465 帧，坐标／有效性／取值帧与保存结果完全相同 |
| 独立投影及未来索引核对 | 全量通过；40 任务各取一条完整 Dataset 样本通过 |
| 历史十条八点结果 | 所有数值数组完全一致 |
| 原数据 | 原 Episode 未改写；除新增相机与来源说明外，字段内容相同 |

271,712 帧末端位于画面内，1,753 帧出画，涉及 59 条轨迹；其中 49 条轨迹的末帧出画。屏外点保留原坐标与无效标识，不裁到屏幕边缘，不使用较早位置代替无效终点。

总共 2,166,640 个有效八点监督：1,929,904 个来自真实未来帧，236,736 个为终点保持；21,080 个目标点无效。所有轨迹均保留，缺少某个二维目标不会删除其动作、技能或视觉监督。

逐条核对来源模型的关节顺序和相机参数；每帧从源关节恢复 `gripper0_grip_site`。同一 BDDL 的栅格检查仅在相机矩阵完全一致时共享，关联文件及哈希写入各条记录。栅格标记选择完整位于画面内的三个时刻，避免出画或裁切造成质心偏差。

复跑入口：`python -m scripts.preprocess.libero.atomic_vla.calibrate`，环境和参数见 [LIBERO 用法](../../../../../../examples/libero/README.md#八点二维-motion)。体积约 92 MB，复用原图像及区域数组。原 `libero40/index.json` 和历史检查保持原内容。

[汇总](report.json) · [逐帧回读](verification.json) · [独立核对与屏外统计](coverage.json) · [来源与代码版本](manifest.json)。原有 740 条官方场景状态对齐警示，以及时间／动作语义未确认记录继续保留；本次相机检查不消除这些原有标记。人工验收仍为 pending，本轮未启动模型训练。
