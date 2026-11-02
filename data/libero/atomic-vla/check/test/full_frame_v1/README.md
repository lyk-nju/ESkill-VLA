# LIBERO-40 全量逐帧区域与人工验收

范围：1,693 条、273,465 帧、3,873 个 Event、40 个任务。

按用户指定，处理后的数据已迁至 `/data1/yuankai/robot-research/data/atomicdata/libero40/episodes/`，同级 `index.json` 为批次索引。本目录对应入口已同步；迁移核对见 [记录](../../archive/location_20261101/migration.json)。

复用固定 RLDS RGB／关节／动作、官方动作匹配场景、已规范化的事件与合并实例标签。夹爪使用 RLDS 关节，操作实例／部件使用对应官方场景状态；每个目标独立渲染，忽略外部遮挡并裁掉屏外部分，从 mask 求紧致 bbox。完全出画保留有效空 mask，bbox 无效。

方法：逐条生成区域后保存 JSON＋NPZ，回读核对源状态、动作、XYZ、所有 mask／bbox、逐帧事件归属与各 Event 的同对象终点。已有区域仅在身份、事件、来源和像素检查通过后复用。绑定 rules 指向既有标签发布版；当前执行代码的真实 SHA256 单列于 manifest.runtime_sha256。

人工视频预选 55 条：seed=42 每任务一条、全部 12 条助手实例补充、各套件末端偏差最大的一条，并保证覆盖 Push 任务；去重后的作者 ID 在 manifest 中固定。这些是验收样本，不是未见任务的泛化评测。

预算：不下载，不运行 VLM／SAM 或策略；4 个工作进程，预计 2 小时内、数据盘新增低于 10 GiB。视频沿用原 1920×1340 布局，每个源步对应一视频帧，10 fps 仅为展示。

路径：`episodes/` 指向数据盘标签，`regions/` 为逐条原生区域元数据，`checks/` 为保存回读检查，`index.json` 为全量标签索引；视频在 `../video/full_frame_v1/`，从保存标签回读生成。结束后查看 `index.html`、`manual_review.csv` 和 `report.json`。

状态：全量逐帧生成、保存回读和预选视频已完成，用户人工验收 pending。740 条原状态对齐警示继续保留，逐帧生成成功不表示区域已与原 RGB 人工对齐通过。

| 技术结果 | 数量 |
| --- | ---: |
| 保存轨迹／帧 | 1,693／273,465 |
| 逐帧 mask／bbox 对 | 546,930 |
| 可读取的同 Event 终点对象框 | 3,873／3,873 |
| 原始缓存、官方状态及保存文件哈希核验 | 全部 1,693 条通过 |
| 验收视频／源帧 | 55／12,137 |
| 验收任务覆盖 | 40／40 |
| 回归测试 | 103 passed |

所有验收视频均完整解码，尺寸 1920×1340、展示时间戳与帧数通过；合计 20 分 13.7 秒。标签、区域缓存和视频合计约 231 MiB，其中视频约 64 MiB。结果见 [report.json](report.json)、[verification.json](verification.json) 与 [tests.log](tests.log)。已保存的原始数值与原生提取数组一致；本批次无有效 Event 帧缺对象 mask，所有目标框均非空。助手只抽看少量中帧预览，不据此填写用户验收通过。

## 人工验收

打开 [index.html](index.html) 选择视频，结论填写 [manual_review.csv](manual_review.csv)。表中有全部 1,693 条轨迹，预选 55 条提供完整视频；其他轨迹可以按作者 ID 单独补渲染。

| 核验项 | 视频对应位置 | 重点 |
| --- | --- | --- |
| 实例绑定 | 当前 Event 的对象名称 | 是否为当前操作的具体实例／部件；同类物体切换是否正确 |
| 夹爪框 | 主相机蓝框 | 是否覆盖完整屏内手掌／手指；遮挡期间仍应保留完整区域 |
| 当前对象框 | 主相机绿框 | 是否跟随本 Event 的操作对象；不应框到参照物或另一同类实例 |
| 终点对象框 | Goal reference 黄框 | 是否为同一 Event、同一对象，在标注 goal_frame 的完整屏内区域 |

各项可填 `accepted`、`rejected` 或 `needs_review`，`decision` 记录整条结论；问题在 `notes` 中写明 Event、帧号和现象。对齐警示需结合画面判断，它本身不是 bbox 错误的判定。完全出画时没有屏内框；被其他物体遮挡时仍应有完整框。

单条回读渲染使用 [scripts/libero_render.py](../../../../../../scripts/libero_render.py)，完整复跑步骤见 [LIBERO 示例](../../../../../../examples/libero/README.md#全量逐帧区域与验收)。

## 55 条合并验收视频

用户随后要求将预选视频整理成一条。[直接打开合并 MP4](../video/full_frame_v1/atomicvla_55_full_frame_review.mp4)，或使用 [章节跳转页面](merged55/index.html)。视频按套件、任务、作者 ID 排序，含全部 55 条／40 任务／12,137 帧，时长 20 分 13.7 秒，1920×1340、10 fps 展示；55 个章节已嵌入 MP4。

合并直接复用压缩视频包，原帧不重新编码。逐帧解码、连续时间戳、完整帧数及拼接前后视频包内容哈希均通过，见 [合并核验](merged55/verification.json)。原 55 条视频和既有全量检查报告保留。

播放序号为 1–55，作者 ID 是原标注的零起算 ID；对应时间及身份见 [chapters.csv](merged55/chapters.csv)。反馈时给出播放序号与画面帧号，或合并视频时间，即可定位到原轨迹；人工结论仍填写原 [manual_review.csv](manual_review.csv)。
