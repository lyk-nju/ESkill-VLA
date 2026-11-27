# Motion Planner 实验

两版均使用单主相机和整句指令，预测 Current、六类技能以及未来 +5 至 +40 帧的八个末端点。事件边界版还预测当前 Start 和本 Event 的 End。架构与监督见 [Motion Planner 模型设计](../../docs/design.md)。

## 事件边界 Planner 正式配方

gpu10 上使用第一版正式实验的相同基座、非 LoRA KLAL 10k、数据划分、图像增强、采样和优化器。训练 20k，每卡 16，八卡全局 128。Router 预热 8 步，学习率预热 1,000 步，随后保持 5e-5。每 1,000 步验证，每 5,000 步保存。新输出目录为 `experiments/motion_planner_boundary/run_001/train`。[完整配置](../../examples/libero/train_motion_planner_boundary.json)

```bash
export PYTHONPATH=src:.
torchrun --standalone --nproc-per-node=8 scripts/train.py --config examples/libero/train_motion_planner_boundary.json
```

恢复时在原命令末尾添加 `--resume /path/to/run/checkpoint.pt`。非 LoRA Planner checkpoint 包含继承的 VLM 权重、文本聚合器、运动模块、优化器和随机状态。损失权重为 Skill CE 0.02、十点 Motion Smooth L1 0.25。训练日志单列 Start、End、各偏移真实点与保持点误差。每 500 步还记录三个角色的视觉读出／残差 RMS 比以及每个时间槽读取 Start、End、时间槽的注意力质量。

## 第一版技术配方

| 配方 | 初始化及更新范围 | Current KL / Skill CE / Trace 权重 |
| --- | --- | --- |
| [冻结 Current 开发](../../examples/libero/train_motion_planner_frozen.json) | 从匹配的 KLAL 20k 载入 Q-LoRA 和文本聚合器，冻结 Current 与 π₀.₅ 基座，训练 Router 和 Planner | 0 / 0.02 / 0.25 |
| [联合训练](../../examples/libero/train_motion_planner_joint.json) | 从原生 π₀.₅ 初始化，同时更新 Current 的 Q-LoRA、文本聚合器、Router 和 Planner，基座保持冻结 | 0.1 / 0.02 / 0.25 |

两份模板都是 **128 步技术短训**，每卡 batch 2，前 8 步只训练路由任务。冻结配方使用有真实轨迹标签的观测；联合配方还可以在反事实图文上训练 Current。对成功移位或目标替换的样本，数据管线屏蔽 Skill 和 Trace 标签。

在训练服务器上复制选定配方，填写本机的 `data_index`、`split_path`、`tokenizer_path`、`pretrained_path`、`norm_stats` 和独立的 `output_dir`。数据使用含逐帧主相机、Current 实体标注及 EEF 投影的 LIBERO40 Grounding。冻结配方还需将 `init_checkpoint` 填为真实 KLAL 20k checkpoint，并使用相同的 π₀.₅ 基座、tokenizer、父轨迹划分、Q-LoRA rank/alpha 和注意力层。两个模板默认 `smoke=false`。

在仓库根目录激活训练环境，然后运行。以下 `recipe` 指向已经填写好的配置。

```bash
export PYTHONPATH=src:.
recipe=/path/to/planner.json
python scripts/train.py --config "$recipe"
```

多卡新训练将最后一行换成 `torchrun --standalone --nproc-per-node=2 scripts/train.py --config "$recipe"`。恢复时使用原来的进程数，在原启动命令末尾加 `--resume /path/to/run/checkpoint.pt`。恢复会加载校准参数、专家权重、AdamW 和各进程的随机状态。搬到另一台机器时，更新资源路径并保留相同文件内容。

首次训练时 rank 0 从训练划分固定抽取八个干净样本校准 Motion Query。`planner_calibration.json` 记录源帧、原生输入 RMS、时间编码比例、首层注意力、空间编码和 Value 差异。`metrics.jsonl` 的 `aux.planner_route_*_count` 是全局获得有效 Trace 监督且被路由到对应技能的样本数；`expert_gradients` 按层给出裁剪后的梯度范数。全局未使用的专家保持 `grad=None`，AdamW 跳过其权重和动量更新。

比较 Current 先验的作用时，在**同一配方**上独立训练两次，将 `model.planner_prior_ratio` 分别设为 `0.3` 和 `0.0`。第二层的比例固定为零。两组使用不同输出目录，并对齐初始化、数据划分、种子、增强、batch、步数及评测帧。

## 连续轨迹与配对干预

下面的任务、实体和帧号来自一条真实轨迹。`run_dir` 指向 Planner 输出目录，`dataset` 指向已准备的 Grounding 数据根目录。

```bash
dataset=/path/to/libero40_grounding
run_dir=/path/to/planner_run
episode="$dataset/episodes/libero_object_no_noops_s016_r001/episode.json"
diag=(python scripts/analysis/motion_planner.py --checkpoint "$run_dir/checkpoint.pt" --episode "$episode" --device cuda)
"${diag[@]}" --output "$run_dir/episode.json" episode
"${diag[@]}" --output "$run_dir/pair.json" pair --frame 16 \
  --task-a 'pick up the alphabet soup and place it in the basket' \
  --task-b 'pick up the milk and place it in the basket' \
  --entity-a alphabet_soup_1 --entity-b milk_1 --other-skill 1
```

`episode` 逐源帧输出各技能概率、Current 和接收容器区域的注意力质量，以及真实点和终点保持点的误差。若有离线仿真抓持真值，可在子命令后加 `--grasp-state grasp.json`。文件格式如下，`source` 填写实际仿真记录来源，`frames` 的键为源帧号，未提供的帧输出 `null`。

```json
{"episode_id":"libero_object_no_noops_s016_r001","source":"LIBERO replay grasp log","frames":{"15":false,"16":true}}
```

使用包含 LIBERO、robosuite 和 MuJoCo 的仿真环境，可以从同一条轨迹生成这份记录。回放工具读取匹配原示范的 XML、BDDL 和逐帧状态，按每段实际搬运的物体实例检查两侧指垫的接触。Place 段仍然检查手里的物体。跨服务器迁移源文件后，可用 `--model-xml`、`--bddl`、`--excerpt` 指向副本；文件内容按原始记录核对。在训练环境运行第二行。

```bash
python -m scripts.preprocess.libero.grasp_state --episode "$episode" --output "$run_dir/grasp.json"
"${diag[@]}" --output "$run_dir/episode.json" episode --grasp-state "$run_dir/grasp.json"
```

`pair` 在同帧 RGB 上比较两条指令，并固定 A 的路由分别替换 Current 先验、融合视觉特征或 Motion Query，亦可移除专家、强制指定专家、替换 A/B 标注先验。两实体区域都可用且指定 `--other-skill` 时，共九种条件。`trace` 为等比例缩放并补边后模型画布上的归一化 `(x, y)`，原点在左上。训练日志的 `motion_l2_px`、逐帧脚本误差与 `toward_b_px_on_observed_slots` 位移投影均以模型画布像素计，后者仅统计真实未来点。训练中的 Current 中心误差使用源图像像素。

事件边界版的 `pair` 另输出 `end_swap`。它从 B 取出归一化之前的原始 End latent，在 A 的上下文中重新规划。输出包括 Start 漂移、End 位移、八个预测点的绝对位移以及沿 End 方向的归一化响应。两预测 End 相距不足 1 模型像素时，方向响应记为 `null`，仍报告绝对位移。

对照 5k、10k、15k、20k 时，两个版本使用同一份已锁定的原生帧清单和图文配对清单。原生清单的格式是 `{"rows":[{"episode_id":"…","frame":16}]}`，frame 为源相机帧号。沿用第一版 175 条验证示范、401 个事件的 1,203 个帧条目，结果中保存清单哈希和每项指标的有效点数。A–E 的 Current 得分沿用原协议。`--motion` 增加自然换目标的轨迹指标，`--interventions` 还会在同一 RGB 的不同目标上固定 A 路由运行交换。后四点“双边更近”按每个条件的后四点到自身目标框的平均最近距离是否严格小于到另一目标框的平均最近距离（框内记 0，单位为模型画布像素）计算。

```bash
python -m scripts.analysis.motion_planner --checkpoint "$run_dir/checkpoint.pt" --device cuda \
  --output "$run_dir/native.json" native --index "$dataset/index.json" --manifest /path/to/native_frames.json
python -m scripts.analysis.visual_primitive_eval --planner --checkpoint "$run_dir/checkpoint.pt" \
  --manifest /path/to/manifest.json --output "$run_dir/ae" --device cuda --interventions
```

## 本机实现核验

用同一条 LIBERO 示范的 6 个真实帧在 RTX 3090 上各做 6 次 AdamW 更新。两组均从原生 π₀.₅ 初始化，batch 为 1，参数 FP32，前向 BF16。短测用两个帧校准查询，从第一步就以固定学习率更新 Planner。耗时取后四步中位数，包含前向、反向和更新。显存为 PyTorch 峰值已分配显存，测试时 GPU 上另有进程。[逐步数据](/data1/yuankai/robot-research/workspace/experiment/motion_planner_boundary_technical_NLyb4K9D/gpu_check.json) · [复现配置与环境](/data1/yuankai/robot-research/workspace/experiment/motion_planner_boundary_technical_NLyb4K9D/run.json)

| 实现 | 可训练参数 | 其中技能专家 | 峰值显存 | 单步耗时 |
| --- | ---: | ---: | ---: | ---: |
| 第一版 | 11,579,912 | 9,437,184 | 11.31 GiB | 234 ms |
| 事件边界版 | 6,663,432 | 4,718,592 | 11.29 GiB | 232 ms |

真实示范中，训练集 3,472 个事件有 3,428 个有效 End，验证集 401 个事件有 396 个有效 End。所有事件起始帧的 Start 投影均有效。逐事件核对的有效 Start 和 End 与已有审查记录相差 0 px。gpu10 的 KLAL 10k 权重尚未传至本机，5k–20k 的正式效果待实测。

Planner 与配对评测专项共 29 项通过，包含 CPU 双进程稀疏专家、checkpoint 恢复和 End Swap。在本机 `atomicvla` 环境，全仓回归另有三项失败，在干净基线 `9b83845` 上均可复现。两项 RoboTwin 导出依赖该环境缺少的 `lerobot.utils`，一项历史动作数据测试的 Future 画布期望与实际边距不一致。

第一版此前用真实 LIBERO 数据和随机初始化 CPU 小模型跑通两份技术配方各 4 步，并用相同初始化、样本和预算另训了 `ρ=0` 对照。在第 16 帧固定视觉特征、Motion Query 和专家，仅替换标注先验，`ρ=0` 时首层读出变化为零，`ρ=0.3` 时可测得变化。整条 136 帧轨迹的实际双指接触出现在第 56–131 帧，标注从 Pick 切到 Place 在第 76 帧。
