# ESkill-VLA：服务当前方法的论文与证据库

从 [idea.md](../docs/idea.md) 理解研究主线，从四份方向文档理解设计理由，从 [design.md](../docs/design.md) 查看整体实现。这里按**当前方法实际需要什么**选择核心工作，不再沿用广泛调研阶段的核心清单。

## 当前核心阅读路线

| 方向 | 优先阅读 | 对当前设计的用途 |
| --- | --- | --- |
| [Event Organization](event_organization/README.md) | AtomSkill、DeCo、WALL-WM；StructVLA、StaKe、RDD | 边界与语义、事件时序、结构监督及衔接 |
| [Atomic Skill](atomic_skill/README.md) | w²VLA、AR-WAM；AtomSkill、StaKe、CRAFT | 文本技巧条件、离散操作接口、事件标签与复用验证 |
| [Target Grounding](target_grounding/README.md) | ReconVLA、AffordanceVLA | 局部去噪与角色化视觉条件，服务三锚点设计 |
| [Action Representation](action_representation/README.md) | RoboTracer；WALL-WM、LAPA、ConLA | 整段粗空间路径、事件组织与跨本体学习的出发点 |

核心不等于照搬完整模型。各索引明确区分直接方法来源、接口参照和验证依据；LAPA / ConLA 是跨本体学习的重要出发点，首版并不直接采用其完整 tokenizer 或对比模块。WALL-WM 保留为事件组织参照，仅维护一张主卡；当前运动表示主要参考 RoboTracer。

## 分层与数量

当前共 91 张主卡：**14 核心、66 补充、11 归档**。其中 89 篇论文、2 项官方技术项目。优先级按当前设计维护，历史阅读记录保留。

| 主归属 | 核心 | 补充 | 归档 | 方法文档 |
| --- | --- | --- | --- | --- |
| [Event Organization](event_organization/README.md) | 5 | 11 | 2 | [设计与来源](../docs/event_organization.md) |
| [Atomic Skill](atomic_skill/README.md) | 3 | 15 | 2 | [设计与来源](../docs/atomic_skill.md) |
| [Target Grounding](target_grounding/README.md) | 2 | 13 | 3 | [设计与来源](../docs/target_grounding.md) |
| [Action Representation](action_representation/README.md) | 4 | 27 | 4 | [设计与来源](../docs/action_representation.md) |

## 维护规则

1. 新阅读先形成论文卡与证据；影响当前方法选择时再更新方向文档和整体设计。
2. 一项工作一张主卡，跨方向使用链接。补充文献不删除，按需展开阅读；本轮不移动旧卡路径。
3. [catalog.json](catalog.json) 是当前优先级的结构化来源，记录用途、理由与旧分层；索引和卡片页首同步更新。
4. `evidence.md` 保留历史核查。旧记录中的“核心／优先实现”仅表示记录时的判断，不覆盖当前目录。
5. 优先级、阅读深度与复现状态分别记录；核心工作仍需标明尚未核验的部分。

<a id="note-format"></a>

## 论文卡内容

记录论文身份与原文链接、实读版本和章节、机制与输入输出、训练／推理依赖、证据边界，以及具体服务本项目哪项设计。区分作者机制与我们的改造，避免将相关工作直接写成已验证的实现。

更新：2026-10-28。新增 RoboTracer 主卡并作为当前运动表示的主要参考；VH-Diffuser、Adaptive Action Chunking 转为按需补充。Stage 1 先验证完整方法的平均任务成功率，不将参考文献扩展成必做消融清单。未运行论文代码或实验。
