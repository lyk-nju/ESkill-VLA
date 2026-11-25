# Motion Planner 实验

给定当前主相机 RGB 和完整任务指令，模型预测六类阶段技能、Current 注意力图和八个未来二维末端点。预测技能通过 Hard Top‑1 选择专家，Current 作为第一层视觉注意力先验。八个点对应未来第 5、10、…、40 个保存的相机帧。事件末尾至少覆盖 10 帧的预测窗口，达到有效终点后重复该点。日志分别统计真实点与派生的终点保持点。本实验止于二维规划。

## 训练与恢复

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

与已有 KLAL 比较定位时，沿用同一份锁定的 A–E manifest。运行 `python -m scripts.analysis.visual_primitive_eval --planner --checkpoint "$run_dir/checkpoint.pt" --manifest /path/to/manifest.json --output "$run_dir/ae" --device cuda`。这个入口按原协议计分，`pair` 与 `episode` 用来检查视觉先验对轨迹的影响。

本地用真实 LIBERO 数据和随机初始化 CPU 小模型跑通两个模式各 4 步，并用相同初始化、样本和预算另训了 `ρ=0` 对照。在第 16 帧固定视觉特征、Motion Query 和专家，仅替换标注先验，`ρ=0` 时首层读出变化为零；`ρ=0.3` 时可测得变化。整条 136 帧轨迹的实际双指接触出现在第 56–131 帧，标注从 Pick 切到 Place 在第 76 帧。这里验证的是干预和逐帧对齐。正式 KLAL 20k 的定位及轨迹质量仍待训练服务器验证。
