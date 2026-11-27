# LIBERO-Plus 闭环测评

工作区的官方 Plus 与独立仿真环境已接入，客户端复用 [LIBERO 公共循环](../libero/evaluate.py)，连接同一个 π0.5 策略服务。源码、环境及资产版本见 [环境说明](../../../workspace/envs/libero-evaluation/README.md)。

先按 [LIBERO 服务命令](../../docs/history/libero_joint.md#闭环测评) 启动你微调后的 checkpoint，再从项目根目录执行：

```bash
PLUS_PY=/data1/yuankai/robot-research/workspace/envs/libero-plus-eval/bin/python

# 列出某类扰动的实际任务 ID；ID 从 0 起算。
"$PLUS_PY" examples/libero_plus/main.py \
  --task-suite-name libero_spatial --category 'Camera Viewpoints' --list-tasks

# 单任务验证；任务 608 为当前固定版本 Spatial 的相机视角扰动。
"$PLUS_PY" examples/libero_plus/main.py \
  --task-suite-name libero_spatial --category 'Camera Viewpoints' --task-ids 608 \
  --num-trials-per-task 1 --port 8000 \
  --checkpoint-tag /path/to/your/checkpoint \
  --output /home/yuankai/robot-research/workspace/experiment/libero_baseline/runs/run_001/plus_camera

# 全套：10,030 个变体，每个变体默认 1 回合。
"$PLUS_PY" examples/libero_plus/main.py --task-suite-name all --port 8000 \
  --checkpoint-tag /path/to/your/checkpoint \
  --output /home/yuankai/robot-research/workspace/experiment/libero_baseline/runs/run_001/libero_plus
```

七类名称为 `Objects Layout`、`Camera Viewpoints`、`Robot Initial States`、`Language Instructions`、`Light Conditions`、`Background Textures`、`Sensor Noise`。可用 `--category`、`--difficulty`、`--task-ids` 筛选；分类按官方任务名称关联，结果同时保存零起算任务 ID 和官方变体 ID，查不到分类会报错。

客户端直接使用基准给出的语言指令，保留语言扰动；使用官方 `initial_states[trial]`，不随机替换初态。默认 seed=7、等待 10 步、每次执行 5 步动作，套件控制上限为 220／280／300／520，与本次固定 OpenPi 参考一致；全部参数写入运行配置，属于本项目固定协议，不据此宣称复现其他论文的排行榜数字。

输出布局和统计口径与 LIBERO 相同，另按七类扰动汇总并记录各自分母。`--video failures` 只保留失败视频，`--video none` 关闭录像。首次正式测评按 [实验规范](../../../workspace/experiment/README.md) 登记 checkpoint、协议、运行范围和预算。

在线输入仅为主／腕 RGB、8 维状态和当前指令；动作来自模型预测。当前已经通过七类各一个任务的零动作技术闭环检查，尚无微调 baseline 成功率。
