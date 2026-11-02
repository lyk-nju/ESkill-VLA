# GuidedVLA 的 LIBERO 数据管线

核对日期：2026-10-30。范围：论文、主仓库、其固定子模块中的基础转换器、已下载发布版元数据和 3 条轨迹。未运行采集／标注／训练。代码版本与数据版本见 [README](README.md)。

## 已确认的链路

```text
已有 LIBERO 示范
  → 阶段区间与描述
  → 阶段对象的逐帧区域 + 阶段技能标签 + 人工核验
  → LeRobot v3 发布数据
  → 按动作窗口读取 actions 和 skill_id，读取当前图像／状态／对象区域
  → LIBERO 字段适配、技能分布、动作变换、归一化、π0／π0.5 模型变换
```

上图是依据论文和加载代码整理的逻辑链路，不代表已有公开的一键构建程序；具体在何时补标签、何时升级容器的完整执行顺序未公开。

| 环节 | 已核对内容 | 依据／限制 |
| --- | --- | --- |
| 示范来源 | 论文使用已有 LIBERO 数据；主仓库提供标注后的 ybwowen/libero | 未公开从每条发布轨迹回链原始 HDF5 demo 的完整清单 |
| 基础转换参考 | 固定 RoboTwin 子模块有 `convert_libero_data_to_lerobot.py`：合并 Spatial／Object／Goal／10 四个 `*_no_noops` RLDS 来源，逐 step 写图像、8 维 state、7 维 actions 和任务 | 是上游示例，不是已证明生成 ybwowen/libero 的原始脚本；不生成技能、区域或 Event |
| 阶段组织 | 论文先假定 episode 已分为时间阶段，各阶段有对象及描述 | 未给出完整的阶段切分算法、边界文件格式和构建入口；不能说 Qwen3-VL 已自动解决边界 |
| 对象标签 | 阶段首帧用 Qwen3-VL 产生对象点提示，SAM2 在阶段内跟踪，人工核验；像素区域再转换为 16×16 patch mask | 是阶段相关对象／目的地区域，不保证始终对应同一被搬运物体 |
| 技能标签 | 预定义技能词表，阶段赋类并展开到逐帧；实读 ID 为 0 other、1 approach target、2 interact at target、3 transport with object | 已有定位到帧的阶段技能时间标注；无独立 Event 表不等于没有时间划分。语义粒度及相邻同技能阶段仍需核验 |
| 容器 | 作者称从 LeRobot 2.0 转为 3.0；发布包为 40 任务、1,722 轨迹、277,947 帧、10 fps，224×224 两路图像 | RGB 内嵌于 Parquet；不能仅根据 info.json 的 video_path 模板断言存在对应视频文件 |
| 加载 | 动作与 skill_id 按 `[0,1/fps,...,(H-1)/fps]` 读取；skill_id 窗口直方图转为 skill_soft，以 KL 监督 | 原 skill_id 被替换为窗口中点 ID，不能把这个输出误认为当前帧 Event 标签 |
| 深度 | 冻结 DA3 从当前主 RGB 计算特征 | 数据构建不必先生成深度真值；此方法不提供我们的末端三维路径标签 |

## 字段怎样进入模型

| 发布字段 | 适配后用途 |
| --- | --- |
| image／wrist_image | base_0／left_wrist_0；缺失右腕输入用 padding 并屏蔽 |
| state[8] | 本体输入；观察构造器使用末端 XYZ、轴角及双指夹爪状态；joint_state[7] 另存，默认适配不读取 |
| actions[7] | 单臂动作块；原始为 delta，公开实验配置还启用了 extra_delta_transform，属于需显式对齐的额外变换 |
| observation.skill_id | 未来 H 帧的技能分布；skill_text 虽在发布数据中存在，但默认 repack 不读取 |
| agentview_attention_object_mask | 当前主视角的对象注意力监督 |
| wrist_attention_object_mask | 当前腕部视角的对象注意力监督 |
| task_index／任务元数据 | 加载器生成 prompt |

两路区域最终填入统一的 `[B,3,256]` object_maps 与 `[B,3]` 视角有效性结构；“3”是视角槽位，不是我们的三种视觉角色，也不是手臂维度。

## 我们需要补建的部分

先核验已有技能连续段与项目 Event 的对应，不默认从零切分；保留原始 episode、帧和时间映射。`no_noops` 等过滤若改变索引，事件和对象追踪必须使用同一映射。发布版未携带 MuJoCo 完整状态、模型 XML 或原始 demo 标识，不能仅靠它保证精确重放；原始 LIBERO 的状态重放属于另一条需核对的上游路径。

我们需要的执行器区域、同一对象的持续关联、Event 终点与剩余末端路径，需要按 [docs/data.md](../../../docs/data.md) 构建。原始像素标签应保留，patch mask 只作为派生目标。数据渲染和人工验收仍按 [check/field_check.md](check/field_check.md) 及统一流程执行，本次未新增视频。

## 后续改进方向：VLM＋SAM3

2026-10-30 用户明确后续尝试方向。保留上文对作者原管线的记录；下面是我们的构建方案，不代表 GuidedVLA 原版已经采用 SAM3，也不代表已验证优于 SAM2。

- **视觉补标**：VLM 结合任务、阶段和关键帧确定夹爪／操作物体，生成描述及必要的位置提示；SAM3 做实例分割与视频跟踪，保存逐帧 mask，并由 mask 派生 bbox。明确实例的提示与查找同类对象的概念示例分开使用。
- **子事件标注**：参考 AtomSkill，复用现有逐帧阶段标签，再结合状态与画面核验 Event 起止、技能和操作对象；不把夹爪开合直接等同于语义边界。
- **汇合**：按原 episode、帧号和对象关联，读取 Event 终点的同对象区域及 RGB，并提取剩余末端路径。保留像素区域，按项目规则构造画布，不仅保存 16×16 patch mask。
- **核验**：补标与原版分开保存；尽可能使用现有 10 条原始样本作对应检查。重点核验夹爪范围、对象选择、遮挡、跨帧身份和终点有效性；不覆盖原始标签或把人工待核验改为通过。

具体 VLM、SAM3 权重／代码版本、提示与纠错协议、样本预算尚未确定；下载渠道可使用 ModelScope。当前仅确认方向，模型补标试点按工作区实验规则另行形成具体方案；尚未安装模型或运行新标注。

## 代码证据

- [作者数据发布](https://huggingface.co/datasets/ybwowen/libero/tree/477f79595e4bc55829706fc655419523ae1da3b9)；本地已读 `meta/info.json`、episode 元数据及一个实际数据文件。
- [GuidedVLA：LIBERO 适配](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/training/config.py#L393)、[字段到视角映射](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/shared/attention_map.py#L7)、[技能分布构造](https://github.com/GuidedVLA/GuidedVLA/blob/04be059e0d6bd448be5cb45fdbafc775f7eb5e38/src/openpi/transforms.py#L80)。
- [固定子模块中的 LIBERO 基础转换示例](https://github.com/RoboTwin-Platform/RoboTwin/blob/958a6d2910a0262f5531fcdeb7fffae4184bb586/policy/pi05/examples/libero/convert_libero_data_to_lerobot.py)。示例声明图像 256×256，发布版为 224×224，不能视为完全相同的构建配方。
- [固定 LIBERO 上游的 HDF5 状态重放与图像导出](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_dataset.py)；这是上游工具，不是 GuidedVLA 的标注程序。
- [论文 III-D、附录 H／I／J](https://arxiv.org/html/2605.12369v2)。
