# ESkill 数据接口核验

2026-11-10，正式入口为 `/data1/yuankai/robot-research/data/atomicdata/libero40/index.json`。

| 检查 | 结果 |
| --- | --- |
| 轨迹／任务／动作行 | 1,693／40／273,465 |
| Event | 3,873 个；保存帧长中位数 67，范围 22–221 |
| 夹爪命令 | 全量补齐 `actions.gripper_convention`；143,520 Open／129,945 Close，无零值或无效值 |
| 原动作／当前本体 | 与逐轨迹 RLDS 源缓存逐值相等，未修改数值 |
| Motion 采样 | 每臂八点，+5、10…40 个保存主相机帧；不强制时间戳 |
| 实际监督 | 最短10帧窗口：至少一个有效轨迹点 96.26%；至少一个有效夹爪标签 96.90% |
| H10／执行前缀5 跨 Event | 19,620（7.17%）／8,720（3.19%） |
| 十条真实轨迹的当前 EEF 投影 | 与已保存 effector_pixels 一致，最大误差 0 源像素 |

[Event 统计表](event_statistics.md) · [完整统计 JSON](event_statistics.json) · [逐 Event CSV](event_statistics.csv) · [夹爪核对记录](gripper_check.json) · [十条投影](projection_audit.json)。新生成的完整 JSON／CSV 报告实际存于数据盘 `workspace/reports/atomicvla_training/`，本目录保留链接和小型摘要。旧 `data_audit.json` 是补声明前的检查，不作为当前结果。

## 补齐与回读统计

在项目根目录、已有 Python 3.11 环境中运行：

```bash
export PYTHONPATH=src:.
python -m scripts.preprocess.libero.atomic_vla.gripper \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --write \
  --report /data1/yuankai/robot-research/workspace/reports/atomicvla_training/gripper_check.json

python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --split data/libero/atomic-vla/split.json \
  --output /data1/yuankai/robot-research/workspace/reports/atomicvla_training/event_statistics.json
```

省略 `--write` 只核对，不更新 Episode；重复执行不会再修改已经一致的声明。脚本先核对全量来源动作、当前八维本体及图像配对，再仅更新元数据。原始数组、Event 与未确认时间轴保持原值；未来导出器也写入相同声明。无需再保存一份二值夹爪数组。

当前双指状态本来保存在 `arrays.npz` 的 `state[:, 13:15]`，开合命令本来保存在 `actions[:, 6]`。本轮补的是各条 `episode.json` 中的 `actions.gripper_convention`；未来八点的 `gripper / gripper_valid` 在训练读取时由源命令生成，因此 `index.json` 和 NPZ 无需改写。检查更新时间应查看 `episodes/<id>/episode.json`，顶层目录时间不反映嵌套文件内容变更。补齐脚本也会同步已有 `SHA256SUMS` 中对应元数据的哈希，保留图像和数组的原校验值。 2026-11-10 已补正此前漏同步的清单：全量 5,080 个文件核对通过，其中 1,693 份元数据仅增加夹爪声明，其余 3,387 个文件与原哈希一致；273,465 帧的双指状态均有效且有限。[完整性核对记录](gripper_integrity.json)保留原清单位置。

统计可加 `--subset train` 或 `val`；同时输出 JSON、逐 Event CSV 和 Markdown。统计包含各技能／任务／划分的 Event 数量、保存帧长分位数、动作行数、开合命令及事件内命令切换次数。Event 长度 `stop-start` 是保存帧数，不能直接换算为秒。

## 跨机器补齐夹爪声明

源服务器已补齐不代表 gpu10 的旧数据副本已更新。代码仓库不包含外部数据盘的 Episode；缺少 `actions.gripper_convention` 时，先补元数据，再开始训练。

[便携核验清单](../../../gripper_manifest.json)覆盖全部 1,693 条／273,465 动作行，来自本次重新对原始 RLDS 缓存的逐值核验。每条记录对完整已保存动作与状态的 dtype、shape、数值字节计算 SHA256。迁移模式仍检查来源 revision、单臂布局、数组形状、观察／动作／相机配对与命令值，再比对清单；因此无需访问元数据里保留的旧 `/data1/yuankai/.../source/` 路径。

在新机器项目根目录执行；两处 `/your/data/...` 换成本机的数据盘路径：

```bash
export PYTHONPATH=src:.
python -m scripts.preprocess.libero.atomic_vla.gripper \
  --index /your/data/atomicdata/libero40/index.json \
  --verified-manifest data/libero/atomic-vla/gripper_manifest.json \
  --report /your/data/reports/eskill/gripper-dry.json

# dry-run 全部通过后补齐：
python -m scripts.preprocess.libero.atomic_vla.gripper \
  --index /your/data/atomicdata/libero40/index.json \
  --verified-manifest data/libero/atomic-vla/gripper_manifest.json \
  --write --report /your/data/reports/eskill/gripper-write.json
```

清单比对失败会在写入任何 Episode 前退出，不从 +/-1 数值猜测符号约定；未收录的其他版本不能套用此清单。`--write` 只增加已核实的声明并同步已有元数据校验和，数组、图像、Event 与归一化数值保持原值；再次执行是幂等的。报告中的 `verification=verified_manifest` 表示通过源核验结果及数值内容一致性继承证据，并非在 gpu10 上重新打开了原始缓存。

原始缓存可访问时，默认路径仍直接逐值核验。可用 `--export-manifest /path/to/gripper_manifest.json` 从完整源核验结果生成清单；不能从另一个清单转导出。迁移后重跑 `scripts.preprocess.libero.audit_training` 确认 `command_ready_episodes=1693`；训练配置中的数据、模型、归一化与输出目录仍需改为新机器实际路径。真实八卡训练和闭环效果需在 gpu10 单独验证。

本次核验：全量源缓存模式与便携清单模式均通过 1,693 条／273,465 动作行，只读运行、未改写正式数据。52 项针对性测试通过，包含移走原始缓存后的迁移、修改动作／状态时拒绝写入、配置缺省解析、缺少系统字体时渲染，以及标准评测相机握手。完整报告位于源服务器数据盘 `workspace/reports/atomicvla_training/portability/`；未声称已在 gpu10 执行迁移或训练。

渲染优先使用 Noto CJK／DejaVu，缺失时回退至 Pillow 内置字体；英文 LIBERO 核验可继续运行。回退字体不保证中文字形，中文核验仍需安装 Noto CJK；字体与模型训练的数据语义无关。

## 夹爪依据

源动作第 7 维沿用 Panda 命令：-1=Open，+1=Close。依据为 [Panda 控制实现](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py#L39-L54)、[观察后执行的重放顺序](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py#L121-L157) 与 [RLDS 同行转换](https://github.com/moojink/rlds_dataset_builder/blob/6174b0b6bb69df6361f1117944952bf14afb0cc3/LIBERO_10/LIBERO_10_dataset_builder.py#L38-L46)。本次 1,693 条还核对了已保存 actions 与对应源缓存全部相等，未做符号翻转或平移。

训练转换为 1=Close／0=Open；源零值缺少二值开合意图，排除该监督，本批没有零值。当前双指实际位置仍作为本体输入，不能代替未来命令；不从 Open／Close 技能名称生成夹爪标签。声明只核实命令及其配对，不宣称已确定整条轨迹的物理时钟。

## 固定帧采样与核验视频

第一版以保存主相机序列为单位，`motion_stride=5`，八个点对应 +5、10、15、20、25、30、35、40 帧。DataLoader 的 `frame_stride` 仅控制当前样本抽样，两者独立。即使原始公共帧索引有跳号，未来查询也先在保存相机帧序列中定位，再映射回对应索引。

LIBERO 环境默认控制频率为 20 Hz，重放 helper 沿用该默认值；按连续控制步估算，5 帧约 0.25 秒，40 帧约 2 秒。由于发布数据包含 no-op 过滤，这只是量级参考，未写入时间戳；视频以 10 fps 播放。精确物理时间可在后续研究补充，不阻塞当前训练标签。

当前使用 Event＋最短10帧窗口，前两个查询可跨 Event，不把终点重复作为有效标签；Episode 尾部仍屏蔽。Future Object 仍取完整当前 Event 终点。全量最短 Event 为 22 个保存帧，没有短到起始帧也无法提供 +5 帧查询的 Event；每个 Event 最后五帧没有未来夹爪标签是边界规则的预期结果。轨迹另受当前起点及未来投影有效性影响。

已有[十条核验视频索引](motion_videos.md)覆盖六类技能和四套任务，仍是此前每 4 帧的标签，本次未重渲染；不能作为当前每 5 帧标签的验收结果。视频上方保留主相机区域、当前 Event 与完整终点对象；下方显示真实训练路径、未来开合命令、来源帧及标签排除原因。白点是当前起点，青色轨迹点为 Open 命令，粉色为 Close；这与上方蓝色夹爪框／绿色物体框／黄色目标框分开说明。

每条先保存数值标签，再回读渲染；复用 Dataset 的 `event_motion_targets`。执行入口：

```bash
python -m scripts.preprocess.libero.review_motion \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --output /data1/yuankai/robot-research/workspace/reports/atomicvla_training/motion_stride5 \
  --video-dir /data1/yuankai/robot-research/workspace/reports/atomicvla_training/motion_stride5/video \
  --count 10
```

完整统计与此前视频、标签实际在数据盘；此前每 4 帧视频入口为 `check/test/video/motion_stride4`，上述命令将另存每 5 帧结果。数值核验不替代用户的视频人工验收。真实 π0.5／FLUX 权重训练与闭环效果另行验证。

此前每 4 帧实现的 270 项相关回归测试通过（不含独立环境的 RoboTwin 管线测试）。此前另用真实 Episode、训练归一化和 tokenizer 接随机初始化的小模型完成 CPU 前向／反向，六项损失与梯度均为有限值，见 [接口检查记录](frame4_interface.json)。这项检查确认数据链路可运行，尚未验证真实预训练权重的训练效果。

改为每 5 帧后，36 项相关测试通过，包含两进程梯度等价与诊断汇总；已按新偏移重算全量监督统计。此前每 4 帧视频及小模型接口记录保留原值。

## Future 区域外扩

Current 保持现有画布。Future 在现有范围上按 256×256 画布每边额外增加 8 px（宽、高各增加 16 px），裁到有效图像区域，并同步扩大 RGB 与监督权重；源 bbox、实例和 Event 终点帧保持原值。配置为 `model.future_bbox_expand_px=8`，设为 0 关闭额外外扩。

[三条真实样本对照图](future_bbox_compare.png)依次展示 Place／Open／Push 的 Current、终点 RGB、新旧 Future 画布；黄框为原范围，粉框为扩大后范围。[来源及像素范围](future_bbox_compare.json)保留对应 Event／帧号。人工验收待完成。41 项相关测试通过，覆盖 Current 不变、分辨率换算、图像边界、缺失区域及配置传递。

## 稀有技能采样核对

当前示例启用 `skill_sampling=true`，权重 `min(3, sqrt(N_max/N_skill))`。仅用 1,518 条训练轨迹的 245,553 个实际训练样本计算权重，未解码 RGB；缺失或无效技能样本数为 0。验证集和归一化保持原规则。

在 CPU 上按 8 个 rank 切片抽取一个 epoch（seed=42，共 245,560 次，每 rank 30,695 次），未启动真实训练：

| 技能 | 原始占比 | 预期占比 | 本次抽样占比 |
| --- | ---: | ---: | ---: |
| Pick | 48.91% | 38.69% | 38.65% |
| Place | 40.00% | 34.99% | 34.98% |
| Turn | 3.54% | 8.41% | 8.43% |
| Open | 3.41% | 8.10% | 8.12% |
| Close | 2.27% | 5.39% | 5.41% |
| Push | 1.86% | 4.42% | 4.41% |

各类实际占比与预期的最大差异小于 0.04 个百分点。[完整抽样报告](skill_sampling.json)。每个 epoch 是有放回抽样，不代表完整遍历全部训练帧。运行时自动保存 `sampling.json`；局部诊断另按真实技能记录分类准确率、路径误差和夹爪准确率及有效计数。

61 项相关回归测试通过，包含 1／2／8 rank 的采样器选择、全局序列切片、验证分布不变、多 worker 在 epoch 内／边界恢复与连续训练的样本及参数逐值一致，以及分布式指标汇总。上述为 CPU 小模型与采样技术核验，未运行正式训练。

## 当前五帧采样的镜像视频

已生成[十条原图／镜像对照视频](mirror_videos.md)，覆盖六类技能与四套任务。此次使用当前 +5…40 帧 Motion、Future 每边额外 8 px 的规则，可同时核验这两项设置及镜像一致性。视频位于 `check/test/video/mirror/`，标签和完整清单位于 `training/mirror/`，实文件在数据盘。已完整解码检查，等待人工验收。

本轮 76 项相关 CPU 回归通过，包含两阶段分布式梯度、镜像断点恢复、Stage 1→2 交接与策略回读。未在本机启动正式训练。

## 镜像样本的状态门控

Stage 1 的镜像样本通过 `arm_state_mask` 关闭 Motion 八维状态投影（包含 bias），原图／Stage 2／推理照常使用真实臂状态；二维起点仍随图翻转。十条真实样本核对中，九条符合镜像条件并关闭状态，一条含方向词保留原图与状态；原始数值均未修改。[对照图](state_mask/preview.jpg)显示 `BODY STATE: ON / MASKED`，[数值核对](state_mask/check.json)与[回读标签](state_mask/preview_targets.npz)实际保存在数据盘。该门控的学习效果仍待正式训练验证。

本次 91 项 CPU 回归通过，覆盖镜像状态值／bias 不影响 Motion、状态分支零梯度而视觉与二维起点仍有梯度、全镜像 batch 的两进程归约、保存恢复精确一致、Stage 2 联合训练及在线／基线接口。未用这些检查推断门控的模型效果。

## 当前最短监督窗口

Motion 最短窗口更新为10个保存帧，名义约0.5秒。全量零有效二维点从21,110（7.72%）降至10,224（3.74%）；旧的严格 Event 数字仅作对照。[覆盖率对照与十条当前核验视频](minimum_motion.md)已生成，待人工查看；96项相关回归通过。历史视频与统计保留原义，不冒充新规则结果。
