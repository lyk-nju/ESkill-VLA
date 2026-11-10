# AtomicVLA：LIBERO 数据

已选定为 LIBERO 主来源，采用作者的事件标注及指定的基础 RGB、状态和动作。

**当前已完成 40 任务、1,693 条／273,465 帧的主相机逐帧夹爪／物体区域保存。** 3,873 个 Event 终点框可读取，55 条完整验收视频覆盖全部 40 任务，合计 20 分 13.7 秒。

当前训练入口为 `/data1/yuankai/robot-research/data/atomicdata/libero40/index.json`。每条 Episode 自包含主／腕相机原始字节、区域、相机矩阵和八点二维标签；旧 `libero40/episodes`、`motion_2d`、`pi05_sft` 已合并后删除。

| 内容 | 入口 | 说明 |
| --- | --- | --- |
| 人工验收 | [合并视频与章节](check/test/full_frame_v1/merged55/index.html) · [验收表](check/test/full_frame_v1/manual_review.csv) | 按轨迹、Event 和帧号记录问题 |
| 全量标签 | [标签目录](check/test/full_frame_v1/episodes/) · [核验汇总](check/test/full_frame_v1/verification.json) | JSON＋NPZ，回读结果与原生提取一致 |
| 原版核对 | [字段表](check/field_check.md) · [原版抽样视频](check/raw/index.html) | 原始字段与十条样本 |

区域统一为忽略外部遮挡的完整屏内几何，屏外裁掉；夹爪使用源关节，物体使用动作匹配的官方场景状态。45 处共享边界按后一个 Event 归属修正，33 个技能按用户确认改为 Push；原标签保留。

实例由 3,839 个自动绑定和 34 个助手补充组成。人工逐项结论仍以验收表为准，740 条原状态对齐警示保留。运行所需来源、人工实例和技能确认记录在 `check/test/` 保持原路径；旧试验已集中归档。

来源与固定版本：[AtomicVLA](https://github.com/zhanglk9/AtomicVLA) 标注 `33d2f98574a52838b9db34b334bd5cc6cbe57f50`；[基础 RLDS](https://huggingface.co/datasets/openvla/modified_libero_rlds) `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`；官方场景状态来源见 [LIBERO-40](../libero-40/README.md)。

末端 XYZ 已核实为米制世界坐标，参考点是 Panda 的 `gripper0_grip_site`。当前导出写入这三个语义字段；现有 1,693 条的元数据回填、修改前备份及验证见 [Motion 语义核验](check/test/motion_semantics_v1/README.md)。XYZ、动作、区域和 Event 不因本次回填改变；夹爪命令的列、符号与执行顺序现已全量核实，其他动作语义与物理时钟继续分别保留状态；见 [训练数据检查](check/test/training/README.md)。

处理与渲染用法见 [LIBERO 示例](../../../examples/libero/README.md)，当前文件说明见 [check/README.md](check/README.md)。

## 实例绑定的通用性约定

实例绑定实现为 [binding_v2.py](../../../scripts/preprocess/libero/atomic_vla/binding_v2.py)，接触／运动证据为 [binding_motion.py](../../../scripts/preprocess/libero/atomic_vla/binding_motion.py)。当前全量结果见 [规则检查](check/test/all_rules_v1/README.md) 与 [实例补充](check/test/manual_binding12/README.md)。

2026-10-31 用户明确要求规则尽可能通用。以下约束已用于 v2 实现与机制检查；当前 v1 及其 100 条结果保持原样，未开展未见真实任务的泛化评估。

| 环节 | 通用做法 |
| --- | --- |
| 对象描述 | 解析类别、显式属性、空间限定和部件；泛称在场景中收集候选，保留颜色／位置等约束，不能遇到未知词就删掉 |
| 场景查询 | 使用实体、区域所属实体、部件所属实体和初始／当前关系；`on stove` 通过区域归属查询，避免把参照实体与区域名直接比较 |
| 来源适配 | 类别别名、具有关节的部件与实际几何名称在配置／适配函数中描述；规则核心不出现轨迹编号、任务白名单或固定实例答案 |
| 上下文 | 根据明确且一致的任务描述补全省略的部件；同臂 Pick→Hold／Place 在连续持物证据下继承身份，释放后清除，不按 Event 顺序分配同类对象 |
| 多候选 | 综合接触、持续运动及夹爪—物体关系消歧，记录证据来源与状态对齐质量；证据不足保留歧义，不能默认第一个、最近或移动最多的候选 |
| 技能审核 | 对象提取与源技能一致性分别记录；可确认 plate 的身份，同时保留 Push 描述与 Pick 标签的冲突，不静默改源技能 |

`region_relations` 的少量完整短语查表是 v1 的局限。v2 已改为“关系解析＋场景查询”；从命名区域读取语义仍属于 LIBERO 适配，不能宣称适用于任意模拟器命名。无可解释的语义时保留未解析状态。

代码保持简洁，优先复用现有绑定函数、来源读取和几何解析入口；暂不因追求通用性新增复杂框架。具体实例／部件由运行时场景确定，BDDL goal 或 `obj_of_interest` 不作为直接选取操作对象的答案。

验证分两部分：现有 100 条固定用于回归；另选未参与规则制定的任务或对象／关系组合检验泛化。针对规则机制检查候选顺序重排、实例重命名、同类干扰物、未知属性与上下文冲突等变化，不能只验证本批成功案例，也不能把已看过的轨迹重新划分后称为未见任务。

## 作者公开的事件标注管线

2026-10-31 已静态核对同版本 [InternVideo2/](https://github.com/zhanglk9/AtomicVLA/tree/33d2f98574a52838b9db34b334bd5cc6cbe57f50/InternVideo2)：`main.py`／`batch_annotator.py` 批量读取轨迹，`action_segmenter.py` 根据夹爪动作变化及末端平移／旋转规则确定候选边界；`video_processor.py` 用 InternVideo2.5 Chat 8B 查看整段视频生成 Task Chain，再按顺序给候选区间分配步骤与技能；`training_export.py` 检查对应关系并导出 `train_annotations.json`，另存排除与失败记录。

输入为支持的 LeRobot Parquet 布局：每条 episode 一个文件，含 `image/actions/state/task_index`，任务表为 `meta/tasks.jsonl`。本项目当前抽样缓存是 RLDS，复用该代码需先适配读取；本轮未安装或运行作者标注模型，也未证明公开流程能逐条重现既有发布 JSON。

端点源码线索：`action_segmenter.py` 将末段 end 设为 `total_len`，中间段按前段 end+1 衔接；导出器允许 end 等于源帧数，而视频裁剪使用 `range(start,end)`。需在项目适配中明确末帧索引，不能将所有 end 无条件当作可读取图像编号。
