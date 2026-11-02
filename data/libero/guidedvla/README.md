# LIBERO：GuidedVLA 发布版初查

2026-10-31 目录调整：本来源从原 `data/libero/` 移至 `data/libero/guidedvla/`，与拟自建的 [LIBERO-90](../libero-90/README.md) 分开。既有清单和运行产物保持原内容，历史路径映射见 [LIBERO 入口](../README.md)。

核对日期：2026-10-30。状态：候选调研；尚未安装 LIBERO 仿真环境。字段初查、10 条原始视频和 10 条 Qwen3-VL＋SAM3 预处理视频已完成，人工验收待完成。入口：[原版](check/raw/README.md)／[原版播放](check/raw/index.html)／[预处理核验](processed/README.md)。

[GuidedVLA 代码](https://github.com/GuidedVLA/GuidedVLA) · [作者发布数据 ybwowen/libero](https://huggingface.co/datasets/ybwowen/libero) · [方法与对照说明](../../../ref_papers/guidedvla.md) · [字段自查](check/field_check.md)

当前工作范围：与 RoboTwin 分别构建我们的仿真数据引擎，基座统一 π0.5；具体来源版本、任务及规模尚待确定。新增 [GuidedVLA LIBERO 数据管线核对](guidedvla_pipeline.md)，区分作者发布包、上游转换示例与未公开的标注步骤。

- 数据 revision：`477f79595e4bc55829706fc655419523ae1da3b9`；代码 revision：`04be059e0d6bd448be5cb45fdbafc775f7eb5e38`。
- 元数据：LeRobot v3.0、Panda 单臂、40 个任务、1,722 条轨迹、277,947 帧、10 fps。episode 元数据的轨迹数与长度总和均与 info.json 一致。HF Viewer 的行数展示有不同口径，不替代这一核对。
- API 存储约 32.3 GB，gated=false。数据仓库未提供数据卡和明确的许可字段；代码 Apache-2.0 不自动等同于此衍生数据的许可。
- 实际读取 `data/chunk-000/file-000.parquet`：670 行，含完整 episode 0／1／2，各 224／257／189 帧。state[8]、joint_state[7]、actions[7] 均为有限值。RGB 以 bytes 内嵌在 Parquet 中，原图 224×224。
- 两路对象标签为 16×16 二值 patch mask：该文件中主视角 670 帧非空、腕部 667 帧非空；存在不代表已确认目标角色及精度。未读取到独立的夹爪区域。已有逐帧阶段技能时间标注，未单独发布 event_id／原始阶段区间表；缺少该表不等于缺少阶段时间划分。
- 实读技能文本：other、approach target、interact at target、transport with object；ID 分别为 0／1／2／3。
- 原始快照与文件哈希：`/data1/yuankai/robot-research/data/external/guidedvla_libero/raw/477f79595e4b/`。早期下载元数据、任务索引和一个约 78 MB 的数据文件；后续 10 条可视化按清单另下载 10 个分片，未全量下载、补标或训练。

按当前 docs/data.md 重核的证据见 [check/field_evidence.json](check/field_evidence.json)：三条轨迹的数值有限，帧号连续且匹配 10 fps；首尾两视角 RGB 均可解码。按技能变化得到 20／18／12 个连续段，这些是读取原始标签整理的区间，不冒充作者原始阶段 ID 或已核验的语义 Event。上述技术检查不代表区域、边界和对象关联已人工通过。

后续原始数据可视化：seed=42，抽 10 个不同任务各 1 条完整轨迹，共 1,592 帧；视频按发布版 10 fps 输出到 `check/raw/video/`。来源和官方文件哈希见 [manifest](check/raw/manifest.json)，完整解码检查见 [verification](check/raw/verification.json)，所有人工结论保持待核验。

是否采用作者发布版本作为引擎原料，需要结合 Event 语义、执行器区域、终点对象关联及原始示范回链进一步核对。不能因 GuidedVLA 已能训练，就认为我们的监督已经全部齐备。

预处理采用 Qwen3-VL＋SAM3，共享代码位于 `scripts/preprocess/visual_grounding/` 和 `scripts/preprocess/text_annotation/`。10 条核验视频已输出到 `data/libero/guidedvla/processed/video/`，对应说明、来源与标注记录、人工表位于 `processed/`。当前仍有事件漏标、夹爪漏检和对象选错；不得将模型输出当作已验收标签。现有 `check/raw/` 保留原版视频和核验记录。

共享实现和命令见 [visual_grounding](../../../scripts/preprocess/visual_grounding/README.md)、[text_annotation](../../../scripts/preprocess/text_annotation/README.md)。[scripts/source.py](../../../scripts/preprocess/libero/guidedvla/source.py) 校验源分片哈希、episode／全局行映射和时间，保留终点 RGB；源技能连续段仍明确标为候选区间。首次模型测试结果及限制见 [processed](processed/README.md)。
