# PhysicalReasoning：10 条数据人工核验

创建日期：2026-10-30。状态：字段自查和 10 条视频已完成；技术检查通过，人工核验待完成。

用户于 2026-10-30 明确要求：各候选先对照 `docs/data.md` 检查字段与补充途径，再抽取 10 条数据渲染到对应目录的 `check/`，供人工核验。本轮执行该流程，范围为 AgiBot World 2026 的 PhysicalReasoning；不代表其他三个主题已经完成。

## 问题与范围

核对原始 RGB、状态、动作、步骤和稀疏空间标注能否正确关联，暴露缺字段、错位、归臂、持物与终点对象对应问题。10 条用于人工发现问题，不估计全库质量率。

- 固定 revision：`95b159a10305a0385f369aba5d4b16242526d17b`。
- 抽样框：前轮已读取的两个批次、29 个首分片；不是全库随机样本。
- 覆盖 10 个不同任务：取钥匙、补洗漱用品、双臂搬箱、工具箱整理、天气条件选物、烹饪、迷你吧补货、餐盒配盖、餐桌整理、抽屉搜索。
- 两批次各 5 条，每任务 1 条；在所选分片的 manifest updatedEpisodes 中用 seed=20261030 选择，记录原始 episode_id。两臂协作和长任务因此有明确覆盖，不伪装为全库均匀随机抽样。
- 预算：仅下载本轮 10 条的 Parquet 和头部／左右腕部 RGB，新增下载上限 5 GB；CPU 解码与 H.264 编码，不运行补标模型、不使用 GPU、不运行策略。
- 失败处理：保留已选样本的下载、解码或数据异常记录，不悄悄换成正常案例。确需替补时单独列出原样本与替补原因。

## 渲染约定

- 每条完整原始轨迹一个 MP4，以原始帧率播放，保留帧号和时间。
- 画面显示当前头部 RGB、当前左右腕部 RGB，以及当前步骤的头部终点预览；显示原始步骤、技能、夹爪和末端数值、共享事件时间线。
- 头部画面只叠加当前原始帧上存在的框／点／二维轨迹；不传播稀疏框，不做检测或插值。无标注时明确显示缺标。
- 终点预览暂按源区间 `[start,end)` 取 end-1，注明这是最后区间内帧、边界待核验；不声称已确定项目所需的语义完成状态。终点对象身份未自动验证。
- 臂身份从原始指令明确文本中提示，标为 text hint；它不是人工核验过的分臂事件。没有标签的另一只臂不自动填 idle／hold。
- 末端数值按源 schema 显示，单位／坐标仍未确定，不绘制假定世界坐标下的投影。
- 不使用源 succeed 默认值分类视频，不将技术检查成功写成人工验收通过。

## 交付

```text
check/
├── README.md             # 范围、协议与完成状态
├── field_check.md        # 对照 docs/data.md 的字段与补充途径
├── manifest.json         # 抽样和源文件身份、路径、哈希
├── technical_check.json  # 读取、对齐和视频检查结果
├── review.csv            # 每条视频的人工核验项，初始待核验
└── videos/               # 10 条完整轨迹的带标注视频
```

专用脚本保留于 `../scripts/`。视频用于人工核验，保留在本地；本轮不自动提交／推送大文件。原始数据留在 `/data1/yuankai/robot-research/data/external/agibot_world_2026/raw/95b159a10305/`，沿用上游路径与 SHA256。

## 人工看什么

在 `review.csv` 中逐项填写“正常／有问题／无法判断／不适用”，问题附视频时间或源帧号，最后记录核验人。

1. 头部与腕部画面是否正常，是否为同一条动作且时间对应；完整视野有无异常冻结或损坏。
2. 步骤文本、技能和边界是否对应实际动作；双臂同时操作、持物或等待是否被源标签漏掉。
3. 绿色框是否圈住本事件的被操作对象；框的 target 是否转成了容器，遮挡后是否换物。
4. 右上是该步骤 end-1 的原始画面，确认它与当前步骤、同一对象和完成状态是否对应；没有框就是当前源标签缺失。
5. 夹爪闭合是否真的持物；视频与数值不能解释的坐标／动作语义保持“无法判断”，继续查原始定义。

原始空间标注是稀疏帧，因此叠加有时只出现一帧，可以暂停、逐帧查看。源 succeed 默认值不用于判断轨迹成功或失败；技术检查结果也不代填上述人工项目。

## 执行命令

在 ESkill-VLA 根目录运行，Python 环境为现有 OpenPI venv（PyAV、PyArrow、NumPy、Pillow）：

```bash
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python data/agibot_world/scripts/prepare_check.py --fetch
OPENBLAS_NUM_THREADS=1 /home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python data/agibot_world/scripts/render_check.py --workers 2
/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python data/agibot_world/scripts/verify_check.py
```

渲染布局为 1280×944、30 fps、原速，文字放在图像外侧，原视野完整保留；仅已有空间标签叠加在图像上。`sample_fields.json` 记录本轮各轨迹的步骤和原始框覆盖，`technical_check.json` 记录自动检查；人工记录在 `review.csv`。

## 视频索引

所有视频的人工核验均待填写，见 `review.csv`。

| 编号 | 任务 | 原始 episode_id | 时长 | 视频 |
| --- | --- | --- | --- | --- |
| 1 | Retrieve key from behind box_With occlusion | 32644953 | 0:10.90 | [查看视频](videos/physical_reasoning_20261001_task_100347_shard02_ep000139.mp4) |
| 2 | Organize and return toolbox to its place | 25800368 | 0:48.17 | [查看视频](videos/physical_reasoning_20261001_task_69701_shard01_ep000056.mp4) |
| 3 | Item selection based on weather conditions | 27429515 | 0:56.67 | [查看视频](videos/physical_reasoning_20261001_task_75089_shard01_ep000072.mp4) |
| 4 | Meal box and lid shape match | 32487791 | 0:29.87 | [查看视频](videos/physical_reasoning_20261001_task_87261_shard01_ep000085.mp4) |
| 5 | Dining table turnover arrangement | 33165852 | 4:50.67 | [查看视频](videos/physical_reasoning_20261001_task_87264_shard01_ep000013.mp4) |
| 6 | Replenish guest room toiletries | 32572677 | 0:33.23 | [查看视频](videos/physical_reasoning_20261003_task_100448_shard01_ep000004.mp4) |
| 7 | Return storage box to shelf | 32982479 | 1:57.30 | [查看视频](videos/physical_reasoning_20261003_task_111349_shard01_ep000004.mp4) |
| 8 | Tomato Scrambled Eggs-ky | 36813353 | 2:27.07 | [查看视频](videos/physical_reasoning_20261003_task_75540_shard01_ep000037.mp4) |
| 9 | Guest room mini bar empty bottle recycling and replenishment | 27452131 | 1:23.00 | [查看视频](videos/physical_reasoning_20261003_task_86632_shard01_ep000000.mp4) |
| 10 | Drawer object search and retrieval | 32515858 | 0:48.20 | [查看视频](videos/physical_reasoning_20261003_task_87265_shard01_ep000001.mp4) |

## 执行结果

- 已输出 10 条独立轨迹视频，共 25,952 帧，合计约 14 分 25 秒，H.264 文件合计约 402 MB。
- 30 路原始 RGB 的逐帧解码数量均与各自 Parquet 行数一致；状态／动作数值均有限，帧号连续。时间网格一致不等于真实传感器零偏移同步，仍需人工看动作对应。
- 10 条输出均已完整解码复查，帧数、1280×944 画面、30 fps 时间网格和文件哈希检查通过。
- 实际新增源文件下载约 663 MB，低于 5 GB 上限。一次 SSL 断连在原样本上重试成功，已补录到 manifest 的 fetch_attempts；没有换样本。
- 199 个源步骤中，50 个步骤的暂定 end-1 帧有原始 bbox；这不是目标对象身份或框质量的通过率。
- review.csv 的所有语义与适用性判断仍为待人工核验；其他三个 AgiBot 主题以及其他数据集尚未完成本轮流程。
