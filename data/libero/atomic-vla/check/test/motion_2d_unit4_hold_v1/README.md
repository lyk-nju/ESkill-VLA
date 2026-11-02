# 四帧单位，末尾保持最后位置

2026-11-02：按用户确认的 `hold_last` 规则，未来帧超出 Episode 时持续取最后一帧的末端位置，补齐点参与 Motion 监督。

`取值帧 = min(当前帧 + 未来偏移, Episode 最后一帧)`；六个偏移仍为 `[8,16,20,28,36,40]` 个来源帧。跨 Event 时按真实轨迹继续，直到整个 Episode 的末尾。

[十条合并视频](../video/motion_2d_unit4_hold_v1/merged10_motion_2d.mp4) · [单条视频与末尾预览](index.html) · [末尾补齐示例](episodes/0051/tail_preview.jpg) · [完整报告](report.json)

视频中 `END` 标记末尾补齐，白色圆圈标记所保持的位置。例如第 0051 条共有 404 帧，在第 383 帧，+8／+16／+20 分别取第 391／399／403 帧；+28／+36／+40 均取第 403 帧，并标 `END`。

| 检查 | 结果 |
| --- | --- |
| 样本 | 同批十条、四套任务，2,023 帧 |
| 真实未来点 | 10,658，坐标与此前版本逐点一致 |
| 末尾补齐点 | 1,480，全部等于对应 Episode 末帧位置 |
| 可用监督点 | 12,138 / 12,138，包含上述补齐点 |
| 最后一帧 | 六个目标全部等于末帧位置，且有效 |
| 跨 Event 的真实点 | 2,345，与此前版本一致 |
| 视频验证 | 1920×1340，202.3 秒；全帧解码、连续时间和十个章节通过 |
| 回归测试 | 46 项通过，含末尾保持、内部缺帧、无效末帧、单帧轨迹 |

补齐是训练约定，不是额外观察到的未来。`future_valid` 包括可靠的真实点和补齐点；`future_observed` 与 `future_padded` 区分两者。`future_frame_indices` 保留请求时刻，`future_source_frame_indices` 保存实际取值帧。最后一帧自身无效或出画时不生成有效补齐；内部缺帧仍无效，不回退到更早位置。

每条 `episodes/<id>/motion.json` 与 `motion.npz` 使用 `eskill.motion2d.v2-pilot`，显式记录 `tail_policy=hold_last`。末帧坐标与相机投影复用 [首轮核验](../motion_2d_v1/README.md)，来源及数组 SHA256 保留；本轮无仿真重跑。此前 [末尾 mask 版本](../motion_2d_unit4_v1/README.md)保持原内容。

当前仅修改二维标签试点，正式模型尚未从三维 Motion 迁移。未来 Skill 标签仍依据真实 Event；末尾坐标保持不自动产生新的 Hold 技能标注。人工验收 pending。

执行方法见 [LIBERO 示例](../../../../../../examples/libero/README.md#二维未来末端监督试点)：`--time-unit-frames 4 --tail-policy hold_last --reuse-projection .../motion_2d_v1`。
