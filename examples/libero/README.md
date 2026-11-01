# LIBERO 闭环测评

工作区已接入独立 LIBERO 仿真环境和官方 OpenPi 模型端。仿真客户端使用 Python 3.8，以源码运行；模型服务使用现有 π0.5 Python 3.11 环境。[环境与版本](https://github.com/lyk-nju/robot-research/blob/bench-libero-20261101-v1/workspace/envs/libero-evaluation/README.md) · [Plus 测评](../libero_plus/README.md)。

在项目根目录开两个终端。模型端传入**微调时同名的训练 config** 和完整 checkpoint 目录；该目录应包含对应归一化资产：

```bash
POLICY_PY=/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
"$POLICY_PY" scripts/serve_policy.py \
  --config pi05_libero --checkpoint /path/to/your/checkpoint --port 8000
```

默认 runtime 是工作区 `openpi/`；可通过 `--openpi-root` 指定另一个包含该训练 config 的 OpenPi checkout。`--print-command` 可检查转发命令。

服务会加载该 checkout 同级 `lerobot-openpi/` 中的固定依赖，解决现有 RoboTwin 环境的 LeRobot 接口差异；可用 `--lerobot-root` 指定其他位置。具体 commit 及恢复方式见环境说明，原环境安装的包保持不变。

另一个终端运行仿真客户端：

```bash
SIM_PY=/data1/yuankai/robot-research/workspace/envs/libero-eval/bin/python

# 单任务验证；输出目录应为新的运行位置。
"$SIM_PY" examples/libero/main.py --task-suite-name libero_spatial --task-ids 0 \
  --num-trials-per-task 1 --port 8000 --checkpoint-tag /path/to/your/checkpoint \
  --output /home/yuankai/robot-research/workspace/experiment/libero_baseline/runs/run_001/libero_single

# 四套 40 个任务，每任务默认 50 个官方初态，共 2,000 回合。
"$SIM_PY" examples/libero/main.py --task-suite-name all --port 8000 \
  --checkpoint-tag /path/to/your/checkpoint \
  --output /home/yuankai/robot-research/workspace/experiment/libero_baseline/runs/run_001/libero
```

`--task-ids` 支持 `0,2,5-8`，使用零起算 ID；`--list-tasks` 可先查看清单。默认 seed=7、稳定等待 10 步、每个动作块执行前 5 步，四套控制步上限为 220／280／300／520，采用固定 OpenPi 参考。`--replan-steps`、`--max-steps`、`--num-steps-wait` 等更改均写入配置。

输入为旋转 180° 后 resize／pad 的主／腕 RGB、EEF xyz＋axis-angle＋双指位置，共 8 维状态；输出严格为 `[horizon,7]` 的 LIBERO 动作块。执行器按固定步数重新规划，不按预测动作是否重复额外重试；每个回合清空动作队列。

每个运行目录包含：

- `config.json`：checkpoint 标识、环境源码／版本、服务元信息及完整评测参数。
- `episodes.jsonl`、`summary.json`：逐回合结果、套件成功率、错误计数和分母；总计采用逐回合微平均。
- `records/*.json`、`*.npz`：原任务、初态、执行动作、8 维状态、动作块编号、原预测块和推理时刻。
- `videos/success|failure|error/`：20 fps 主视角录像；仅在录像中标记动作块，保持模型输入原图。

原始成功判定来自环境；异常会在记录中标出，不以默认动作伪造成功。首次正式测评按工作区 [实验规范](https://github.com/lyk-nju/robot-research/blob/bench-libero-20261101-v1/workspace/experiment/README.md) 登记协议与预算。技术测试服务或 `--technical` 模式会将成功率设为 null，避免混入真实模型结果。

当前已验证四套各一条的通信及控制闭环。使用微调 checkpoint 的真实成功率需在权重准备好后另行评测；数据转换与 baseline 微调属于后续步骤。
