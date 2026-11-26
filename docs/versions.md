# 分支与模型入口

本分支为 **`motion-planner`**。Motion Planner 以主相机图像和完整指令预测 Current、阶段技能与二维末端计划。模型架构见 [设计文档](design.md)，训练、恢复与诊断见 [实验说明](../experiments/motion_planner/README.md)。

## 模型与训练入口

| 用途 | `model_type` / `training_stage` | 预测与实验 |
| --- | --- | --- |
| Motion Planner | `eskill` / `motion_planner` | `planner_version=v2` 输出 Start、End 和八个未来点，`v1` 输出八个未来点。[冻结与联合配方](../experiments/motion_planner/README.md) |
| Visual Primitive | `eskill` / `visual_primitive` | 多层原生注意力定位 Current，独立 Subject 定位夹爪。E0–E3 用于视觉对照。[训练与评测](../experiments/visual_primitive/README.md) |
| 历史联合动作 | `eskill` / `joint`、`stage1` 或 `stage2` | Skill 与对象 Grounding 形成 Motion，Action Expert 读取运动条件。[直接训练](../experiments/eskill_libero40_direct/README.md) · [两阶段训练](../experiments/eskill_libero40_mirror/README.md) |
| 历史 Grounding 对照 | `eskill` / `grounding` | Current 和 Subject Query Decoder 使用 VAE 或 BBox 监督，可加 GAC。[对照配置](../examples/libero/README.md) |
| 纯 π₀.₅ 基线 | `pi05` / `joint` | 图文特征进入 π₀.₅ Action Expert。[历史配方](../experiments/pi05_libero40/README.md) |

## 运行与权重

事件边界配方从非 LoRA KLAL 10k 继承并冻结视觉语言主干，训练一组技能专家。第一版的两份技术配方共用旧版前向结构。`planner_joint=false` 的开发配方从指定的 Visual Primitive KLAL 第 20,000 步权重继承 Q-LoRA 和文本聚合参数。`planner_joint=true` 的联合配方从原生 π₀.₅ 初始化，同时训练 Current 的 Q-LoRA、文本聚合器、技能路由和规划器。两份配方均冻结基座的原始参数。[配置与恢复](../experiments/motion_planner/README.md)

新保存的非 LoRA Planner checkpoint 使用 `full`，包含继承的普通 VLM 参数及运动模块。Q-LoRA 权重保存为 `planner_adapter`，读取时需要匹配的 π₀.₅ 基座和 tokenizer。读取二维计划用 `create_motion_planner_policy`。精简视觉权重、历史动作策略和本分支规划器的读取入口由各自的训练阶段决定。[Planner 推理入口](../src/openpi/policies/motion_planner.py) · [checkpoint 读写](../src/openpi/training/checkpoints.py)

复现实验需要保存 Git commit、配方、权重身份和数据划分。同一次恢复沿用原来的配置和进程数，跨服务器可替换为内容相同的资源路径。`eskill.episode.v2-draft` 标识保存数据的 schema，包版本另行记录。[数据协议](data.md)

## 本机工作目录

以下目录是同一 Git 仓库的 worktree。其他服务器可在独立 checkout 中使用对应分支。

| 工作目录 | 分支 | 对应工作 |
| --- | --- | --- |
| `motion-planner/` | `motion-planner` | 本分支的事件级二维规划 |
| `visual-primitive/` | `visual-primitive` | 原生注意力定位与视觉对照 |
| `ESkill-VLA/` | `main` | 联合动作策略与闭环诊断 |
| `ESkill-VLA-episode/` | `simplify-episode` | 早期 Episode 数据整理 |
