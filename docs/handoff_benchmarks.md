# 2026-11-27 锁定测评迁移

正式协议由 `unified/eskill_bench` 维护。各研究 checkout 的 `python -m scripts.benchmark` 均可进入 `handoff`。历史模型在独立 Python 进程中使用对应快照。训练入口、LSF 队列、模型定义均由原项目维护。固定交接版本的摘要为 `a7641006c8753a580b783c97751239955c4657923ef0fb79066cb10147fee9cd`。

## 入口对照

| 历史入口（交接包内） | 统一命令 `handoff` 下的子命令 | 输出与口径 |
| --- | --- | --- |
| `source/stage1/experiments/stage1_visual_localization/{infer,report}.py` | `infer legacy_vae/bbox/bbox_gac/vae_gac CKPT RUN`，任选一种 | 原始 `records` 格式。逐查询 shard、每任务及每套件 CSV、可读／缺预测状态。源图 256×256 上中心误差 ≤10 px。BBox 使用两路归一化 xyxy，旧 VAE 使用保存的双路 head、配套 FLUX 和阈值 32 的重建支撑区质心。 |
| `source/klal_full_vlm_cf/scripts/analysis/visual_primitive_eval.py` | `infer klal CKPT RUN` | 双路原生注意力、最高质量 3×3 窗口质心，使用历史 `rows` 与 source-pixel 口径。保留 `raw/metrics.json` 和逐行预测。 |
| `run_code/native_grounding/run_001/source/{capture,evaluate,routing}.py` | `native prepare/smoke/capture-dev/select/use-lock/capture-test/score/route RUN` | E0 可见注意力总质量与条件空间分布分开。历史 `local3` 是以**单个 peak 为中心**的邻域读出。E1 三种路由只在开发 Check 比较。 |
| `source/native_grounding/experiments/vlm_capability_diagnosis/representation/{prepare,features,probe,report_tables}.py` | `representation prepare/features/fit/test/tables RUN` | Fit 学参数，Check 选择，test 核验锁定选择与模型哈希。自由中心不读 GT 框。候选检索每张图独立按可评分候选排序。独立 Pick 使用 `libero90 score/summarize/verify RUN`，输入为已封存的 LIBERO-90 run。额外的中心确认实验有独立 roster、类别和可见性筛选，记录见包内同级 `independent_pick`、`independent_center_confirmation`。 |
| `source/motion_v1_frozen/scripts/analysis/motion_planner_eval.py`、`reference/motion/{02,03}/evaluate.py` | `infer frozen/joint/boundary CKPT RUN`；`replay BUNDLE` | 锁定 202 对的三份 20k 权重各有独立源码及 SHA。输出原格式 `pairs.jsonl`、逐对及分组表。Boundary 另外输出 `boundary_pair_cases.csv`。 |
| `source/benchmarks/ESkill-VLA/scripts/evaluation/sample_{protocol,client,server}.py` | `closed libero200/plus280/sample plan/serve/run/status/report RUN ...` | 稳定键为 benchmark、suite、task_id、trial。原版每任务初态 0–4，Plus 抽样每任务初态 0。每回合重置仿真及服务端策略 RNG 为 7。同一个分组失败后重试，并按键合并。 |
| `source/benchmarks/ESkill-VLA/examples/libero_plus/main.py` | `closed plus-full plan/serve/run/status/report RUN ...` | 四套共 10,030 个变体，各一次官方初态。沿用原客户端顺序与连续策略 RNG。全套中断后按原顺序从头运行，需重启同一策略服务以恢复初始 RNG。 |
| `source/benchmarks/libero-composition/.../nine_combinations.py` | `closed nine plan/prepare/run/status/report/audit/render ...` | 调用当前 `libero_composition`；按 scene、任务谓词、初态列表、seed 对照旧九组合清单。9×50 主任务和 4×10 同场景对照分报，原生谓词首次命中即停止。 |
| `source/benchmarks/libero-composition/.../craft_appendix_a.py` | `closed craft report`；`closed craft prepare/serve/run/render/audit RUN` | 无参数 report 从本地发布表逐回合复算四模型。历史公开 π₀.₅ 的在线适配器单独接入，50 初态、seed 7。3×50 原组合和 5×50 新组合分别汇总；额外模型需其原生策略部署。 |

`visual`、`planner native` 和 `late-place` 等当前研究入口见 [测评说明](evaluation.md)。新采集的 Late Place 可单独评测。历史 A–E 的 E 保留原始零覆盖。

## 在新机运行

使用 Python 3.11+。从研究 checkout 切入时，可设置 `ESKILL_BENCHMARK_ROOT` 指向 `unified`。复制 [配置示例](../examples/benchmark/handoff.json)，修改资产、输出、模型 Python、两种 LIBERO 仿真 Python 和设备。配置里的相对路径从**配置文件所在目录**解析；虚拟环境的 Python 路径保留软链接。路径映射按最长目录前缀匹配。

```bash
python -m scripts.benchmark handoff unpack /path/eval_handoff_2026-11-27.tar.gz /data/handoff
python -m scripts.benchmark handoff replay /data/handoff --output /data/reports/motion202-regression.json
python -m scripts.benchmark handoff replay /data/handoff --suite visual --output /data/reports/visual-regression.json
cfg=/path/to/handoff.json
bench() { python -m scripts.benchmark handoff --config "$cfg" "$@"; }
bench audit
bench relocate reference/motion/01/manifest.json motion202-inputs
bench restore-dev native-dev-001
bench infer boundary /data/checkpoints/checkpoint_020000.pt boundary-technical --limit 2
bench infer frozen /data/checkpoints/checkpoint_020000.pt frozen-native --factual
bench infer klal /data/checkpoints/checkpoint_010000.pt klal-formal
bench video /data/reports/cases.mp4 /data/panels/case01.png /data/panels/case02.png
```

`unpack` 检查压缩包和 FILES.tsv，然后逐文件核验。原件始终从封存目录读取。重定位产生副本和 SHA／路径映射，保留 ID、顺序、父轨迹、源帧、指令、框、RGB 内容身份和八个名义槽。`restore-dev` 依据 204 个训练父轨迹恢复 Native 开发 RGB，逐张核对解码后 uint8 RGB 的 SHA-256。将旧 `inputs/dev_rgb` 前缀映射到恢复出的目录即可复用。默认审阅视频每例 18/24 秒，H.264、yuv420p。闭环视频保留各协议的控制帧时间。

`infer` 的输出目录按运行隔离，保存封存清单身份、重定位清单 SHA、checkpoint SHA、选用的模型快照及覆盖范围。`--limit` 的运行记作技术检查；Stage 1 若选中配对查询，也包含其被引用的原查询。训练步数或初始化不同的技术权重需显式给 `--technical-checkpoint` 和样本限制。开发图单例可用 `--native-dev-frame`。历史 VAE+GAC 若提示源文件指纹不符，可在 paths 填写 `vae_gac_source` 指向原训练源码，加载前会逐项比较摘要。

历史冻结 Planner 的 `--factual` 按 175 个父轨迹读取锁定锚点与时序窗口，输出 `factual.jsonl`、真实点／终点保持／跨事件分项、技能混淆和时间对齐表。`--limit` 在这一路按轨迹计。三版 Planner 的默认推理按反事实配对计，覆盖文件分别记录实际观测数与配对数。正式运行会核对 20k 权重、训练划分及数据索引。

### Native Grounding 与 Probe

```bash
bench native prepare native-001 --recipe /old/native_grounding/run_001/config.json
bench native smoke native-001
bench native capture-dev native-001
bench native select native-001
bench native route native-001
bench native capture-test native-001
bench native score native-001
```

复用原开发选择时，使用 `use-lock` 代替 `select`；新副本仅更新已验证的 dev 清单文件摘要。正式 E0 检查层／头、策略、阈值及开发集身份。若原基线逐样本 CSV 齐全，`score` 可传多组 `--baseline NAME=CSV`。模型源码默认取本地 Git 对象 `ce9568ae`；可用 `paths.native_source` 指定匹配 checkout。

Probe 将历史 recipe 放入独立输出目录的 `config.json`，将 `base` 指向已准备的 Native run，`historical` 指向含 `regions.json` 的候选几何目录。依次运行 `representation features`、`prepare`、`fit`、`test` 和 `tables`。开发 Check 与正式 test 的候选覆盖分母不同。

### 闭环

```bash
bench closed sample plan sample-001 --checkpoint-tag /data/model.pt --jobs /old/job-plan.json --variants /old/selected-variants.json
bench closed sample serve sample-001 --checkpoint-tag /data/model.pt --port 8000
# 在另一个终端，用相同配置运行客户端
bench closed sample run sample-001 --port 8000
bench closed sample report sample-001
bench closed nine plan
bench closed craft report
```

抽样清单校验旧运行记录里的两个 SHA；换机器后保留原始 JSON 字节。并行客户端可传 `--worker I --workers N`，每个 worker 使用自己的策略服务和端口，同一 worker 同时只运行一个实例。协议报告区分成功、任务失败、异常和未完成。200／280 的正式分母要求完整且无异常；Plus 全量报告沿用原客户端的逐回合计数。九组合的 `prepare` 接受当前部署的 `--host`、`--run-id`；已有 run 的 `report` 接受 `--run-dir`。CRAFT 在线公开模型需要 `craft_checkpoint`、固定的 `craft_upstream` 和匹配的 `craft_python`。这些命令不会自动启动全模型队列。

实测基准和缺失资产见 [回归记录](handoff_regression.md)。
