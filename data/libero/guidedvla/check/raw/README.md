# GuidedVLA LIBERO 原始发布数据核验

2026-10-31 渲染代码已委托 LIBERO 共用渲染器，原有 1440×1120 双视角视频及其 manifest 保留为历史核验。本页以下布局、颜色与统计描述该历史批次。新布局使用 [render_review](../../../scripts/render_review.py)，见 [统一说明](../../../README.md#共用人工核验渲染器)，不直接覆盖本目录视频。

状态：10 条完整视频与技术检查已完成；人工验收待完成。用户已明确要求本轮原始数据可视化，输出 `check/raw/video/`。这里 raw 指 GuidedVLA 发布版未经本项目补标／改写的数据，包含作者已有技能和对象 patch mask；不指移除这些标签后的原版 LIBERO。

直接打开 [播放索引](index.html)，字段情况见 [field_check.md](../field_check.md)，人工结论填入 [manual_review.csv](manual_review.csv)。

## 本轮协议

- 来源：`ybwowen/libero`，固定 revision `477f79595e4bc55829706fc655419523ae1da3b9`。大文件保存在 `/data1/yuankai/robot-research/data/external/guidedvla_libero/raw/477f79595e4b/`，本目录保存来源清单、核验结果及视频。
- 需求依据：[docs/data.md](../../../../../docs/data.md)。脚本将其 SHA-256 写入 manifest。
- 抽样：按元数据中的任务文本排序，用 Python `random.Random(42)` 无放回抽 10 个任务，再按相同随机流从各任务按 episode_index 排序的轨迹中选 1 条。共 10 条独立完整轨迹；不根据标签质量挑选或替换。
- 固定来源清单：`manifest.json`，记录任务、episode、帧范围、数据文件和官方 LFS SHA-256。若读取或渲染异常，记录异常并先排查；不静默换样本。
- 预算：只下载命中的 Parquet，下载前核对合计不超过 1.2 GiB；CPU 下载并发最多 3，渲染并发最多 2，不启动仿真、模型补标或训练。输出视频目标不超过 300 MiB。
- 视频：完整轨迹，1440×1120、H.264、10 fps；每个源帧对应一个输出帧，按发布版时间播放，不额外插值或慢放。
- 布局：上排原始主视角、主视角区域叠加、腕部区域叠加；下排状态／动作数值、末端三维记录的独立坐标图、当前技能连续段末帧预览。底部显示原始逐帧技能时间线。
- 区域：绿色半透明块只来自作者 16×16 patch mask，最近邻放大；不生成 bbox、不补细分割。空 mask 明确标注。原图不翻转、不重排颜色通道。
- 技能：展示原 skill_id／skill_text；连续段由原始技能变化读出，标作“技能连续段”，不冒充原 event_id。other 原样显示，不推断 idle／hold。
- 末帧预览：取当前技能连续段最后一个有该标签的源帧，并明确它是连续段末帧，不是已核验的语义 Event 终点；对象区域也不冒充同一对象的 goal 真值。
- 路径：state 前三维作为已核对构造器中的末端 XYZ，独立坐标图保留数值尺度、标出轴；全轨迹灰色、当前技能段蓝色、当前点橙色。坐标系／单位仍待确认，不投影回相机，也不预测未来。
- 缺失项：夹爪区域、已核验 Event／同一对象 goal、坐标单位及相机约定在画面标明未提供或待核验。Panda 单臂；不显示虚构的另一臂。

## 产物与验收

- `video/`：10 条供人工核验的视频。
- `manifest.json`：固定样本、来源指纹与渲染协议。
- `render_summary.json`：源字段数值、时间、标签段和编码记录。
- `verification.json`：完整解码、帧数、时间戳、尺寸和格式检查。
- `manual_review.csv`：逐条人工核验记录，初始均为待核验，不自动填通过。
- `index.html`：本地浏览入口，直接播放视频；不发布到网络。

技术检查通过仅表示文件可读取和视频与源时间一致。事件语义、区域正确性、同一对象关联、路径含义及数据是否适用，由人工验收填写。

## 本轮执行结果

2026-10-30 完成。10 个任务、10 条独立轨迹，共 1,592 帧／159.2 秒；视频合计 25,057,512 字节（约 23.9 MiB）。下载原始分片 951,295,634 字节，全部匹配官方 LFS SHA-256。全部源帧的两路 RGB 已解码，数值有限、帧号和时间戳与元数据对应；全部输出视频完整解码、H.264／10 fps／1440×1120 和逐帧时间检查通过，详见 [verification.json](verification.json)。

原始主视角有 1 个空区域帧，腕部有 13 个空区域帧，均保留并显示 EMPTY SOURCE MASK，未插值补齐。技能连续段及缺标统计见 [render_summary.json](render_summary.json)。这些计数不代表语义标签已人工验收。

| 视频 | Episode | 任务 | 时长 |
| --- | --- | --- | --- |
| [01](video/01_episode_001285_raw.mp4) | 1285 | pick up the black bowl next to the cookie box and place it on the plate | 14.6 s |
| [02](video/02_episode_000428_raw.mp4) | 428 | open the top drawer and put the bowl inside | 18.4 s |
| [03](video/03_episode_000919_raw.mp4) | 919 | pick up the cream cheese and place it in the basket | 16.7 s |
| [04](video/04_episode_000942_raw.mp4) | 942 | pick up the butter and place it in the basket | 16.5 s |
| [05](video/05_episode_000312_raw.mp4) | 312 | pick up the book and place it in the back compartment of the caddy | 17.4 s |
| [06](video/06_episode_001656_raw.mp4) | 1656 | pick up the black bowl next to the plate and place it on the plate | 14.0 s |
| [07](video/07_episode_001303_raw.mp4) | 1303 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 15.6 s |
| [08](video/08_episode_001620_raw.mp4) | 1620 | pick up the black bowl from table center and place it on the plate | 12.2 s |
| [09](video/09_episode_000072_raw.mp4) | 72 | put both the cream cheese box and the butter in the basket | 24.6 s |
| [10](video/10_episode_000779_raw.mp4) | 779 | turn on the stove | 9.2 s |

脚本入口：[render_raw.py](../../scripts/render_raw.py)，当前使用环境：

```bash
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m data.libero.guidedvla.scripts.render_raw prepare
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m data.libero.guidedvla.scripts.render_raw download
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m data.libero.guidedvla.scripts.render_raw render
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m data.libero.guidedvla.scripts.render_raw verify
```

已有 manifest 时不重新抽样。执行记录：首轮画面抽查发现字体缺少拉丁字母／数字，改用 NotoSansCJK 并重新渲染；重渲染进程以 SIGTERM 中止时已完成 9 条，使用 `render --resume` 复核来源与既有成片后补完第 9 号样本，最终全部通过完整解码检查。可重建的中间成片与未完成文件已替换；未更换样本、补标签或填写人工通过。
