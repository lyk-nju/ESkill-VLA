# LIBERO 数据准备与回读

在项目根目录使用已安装本项目的 Python 环境运行。数据来源与人工核验入口见 [AtomicVLA](../../data/libero/atomic-vla/README.md)。

当前处理后的全量数据在 `/data1/yuankai/robot-research/data/atomicdata/libero40/episodes/`，批次索引为 `/data1/yuankai/robot-research/data/atomicdata/libero40/index.json`。项目内对应的 `check/test/full_frame_v1/episodes/` 和 `index.json` 为链接。

导出器复用已有完整帧区域及最新实例标签。下面以作者 ID 51 为例，输出目录必须尚不存在：

```bash
python scripts/prepare_data.py atomic-vla \
  --annotation-id 51 \
  --regions data/libero/atomic-vla/check/test/manual_binding12/regions/libero_10_no_noops_s004_r004.json \
  --output /path/to/new/episode_0051

python scripts/libero_render.py \
  --episode /path/to/new/episode_0051/episode.json \
  --video data/libero/atomic-vla/check/test/video/episode_0051_review.mp4
```

保存格式见 [data/README.md](../../data/README.md)。原始数据或其他来源的既有核验入口仍可用 `python -m scripts.preprocess.libero.render_review --help` 查看；详细研究批次的模块迁移见 [scripts/preprocess/README.md](../../scripts/preprocess/README.md)。

## 全量逐帧区域与验收

全量入口为 `scripts.preprocess.libero.atomic_vla.batch`，复用已有事件与实例标签，依次完成原生区域生成、JSON＋NPZ 保存回读和验收视频。当前批次范围及固定样本见 [full_frame_v1](../../data/libero/atomic-vla/check/test/full_frame_v1/README.md)。

```bash
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export PYTHONPATH=src:.
export LIBERO_CONFIG_PATH=/data1/yuankai/robot-research/workspace/envs/libero-native/config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$LIBERO_CONFIG_PATH/nvidia_egl.json"
export NUMBA_CACHE_DIR=/tmp/eskill-full-frame-numba
export MPLCONFIGDIR=/tmp/eskill-full-frame-mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.batch prepare
"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.batch build --workers 4
"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.batch review --workers 4
"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.batch report
```

`build` 可续跑，并重新核对已保存标签；`review` 默认渲染清单中预选的验收轨迹，可用 `--annotation-id 51` 指定一条，或用 `--all` 渲染全部轨迹。单条保存结果仍可通过上面的 `scripts/libero_render.py` 渲染。

大体积标签及视频在数据盘，项目内的 `check/test/full_frame_v1/episodes/`、`check/test/video/full_frame_v1/` 提供对应链接；验收入口为 `check/test/full_frame_v1/index.html`，人工结论填写同目录 `manual_review.csv`。结构检查与完整视频生成后，用户验收仍为 pending。

## 闭环测评

工作区已接入独立 LIBERO 仿真环境和官方 OpenPi 模型端。仿真客户端使用 Python 3.8，以源码运行；模型服务使用现有 π0.5 Python 3.11 环境。[环境与版本](../../../workspace/envs/libero-evaluation/README.md) · [Plus 测评](../libero_plus/README.md)。

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

原始成功判定来自环境；异常会在记录中标出，不以默认动作伪造成功。首次正式测评按工作区 [实验规范](../../../workspace/experiment/README.md) 登记协议与预算。技术测试服务或 `--technical` 模式会将成功率设为 null，避免混入真实模型结果。

当前已验证四套各一条的通信及控制闭环；双相机 SFT LoRA checkpoint 也已通过真实策略服务的四套技术检查。下方本项目 LoRA 使用独立的 PyTorch checkpoint 接口；完整训练与真实任务成功率见对应实验进度。

## 原版 π0.5 四卡 LoRA

本轮纯 SFT 使用 [train_pi05_sft.json](train_pi05_sft.json)：全部 1,693 条原始示范，主／腕双相机，horizon=10，状态不拼进文本，q01/q99 归一化与完整32维 Flow Matching MSE。四卡全局 batch=32，LoRA rank=16，10,000 步；每1,000步另存里程碑 checkpoint。新增事件、区域及路径标签不进入模型。[实验协议与当前进度](../../../workspace/experiment/pi05_libero_sft/README.md)。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py \
  --config examples/libero/train_pi05_sft.json

# 训练后：读取本项目的基座＋LoRA与保存的归一化配置。
python scripts/serve_policy.py --runtime local \
  --checkpoint /path/to/checkpoint_010000.pt --port 8000
```

`local` 服务目前接收 `libero_sft` 双相机适配，使用与上方 LIBERO 客户端一致的原始8维状态及7维动作；官方 OpenPI checkpoint 仍用默认 `--runtime official`。不再需要为此纯 SFT 模型补成15维关节状态。数据缓存由 `scripts/preprocess/libero/atomic_vla/prepare_sft.py` 生成，保留源图像字节、轨迹配对与动作；本轮配置与输出位置以实验记录为准。

下面保留此前主相机接口检查配置，便于复现旧测试：

配置为 [train_pi05_lora.json](train_pi05_lora.json)：`model_type=pi05`，仅动作损失；不创建 ESkill 技能、视觉、运动分支或 VAE。冻结原始基座，两个 Gemma 的 attention／MLP 加 LoRA，rank=16；BF16，每卡 batch=1，四卡全局 batch=4。数据使用主相机、源 8 维状态和 7 维动作，horizon=50，padding 不计入损失。这是本项目的 π0.5 数据接入配置，不等同于复现上游全部 LIBERO 超参数。

```bash
python scripts/compute_norm_stats.py --config examples/libero/train_pi05_lora.json
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py \
  --config examples/libero/train_pi05_lora.json
```

该示例面向后续完整训练，30,000 步尚未执行；先检查路径、数据划分与预算。LoRA checkpoint 保存 adapter、优化器、配置、统计及各 rank 随机数状态；恢复和推理仍需原始 `pretrained_path` 权重及 tokenizer。恢复命令在同一四卡启动命令后添加 `--resume /path/to/checkpoint.pt`，并保持数据、batch 和 world size 一致。

已执行的两组短程测试使用另存配置、每任务一条共同轨迹和 120 个动作窗口，详见 [四卡检查记录](../../data/libero/atomic-vla/check/test/pi05_lora_v1/README.md)。其中 `data_format=source_npz` 读取 OpenVLA modified LIBERO RLDS 的原字段缓存；`episodes` 读取本项目处理后的 JSON＋NPZ。两组共享同一份统计，不使用新增事件或 bbox 标签。

## ESkill 训练与策略接口

使用独立环境安装 `python -m pip install -e '.[train,test]'`。配置示例为 [train.json](train.json)，以完整 π0.5 尺寸、主相机当前帧和新 ESkill 结构运行。路径、batch size 和训练步数是起始配置；本轮未启动此正式配置。

先设置两个本地权重路径：`pretrained_path` 指向 π0.5 的 `model.safetensors`，`vae_path` 指向预训练 `diffusers.AutoencoderKL` 目录。π0.5 已按本地上游映射将 JAX／Orbax 权重转换为 BF16 safetensors，位置为 `/data1/yuankai/robot-research/ckpt/pi05/base/pytorch_bf16/model.safetensors`，并完成完整基座的严格加载与 CUDA 前后向检查；转换信息见同目录 `conversion.json`。ESkill 的预训练 VAE 和完整分支 CUDA 训练仍待验证；正式配置缺少权重会明确报错。

```bash
python scripts/compute_norm_stats.py --config examples/libero/train.json
python scripts/train.py --config examples/libero/train.json
python scripts/train.py --config examples/libero/train.json --resume checkpoints/libero40_eskill/checkpoint.pt
```

统计只读取训练轨迹；已有统计文件不覆盖。输出目录包含 `config.json`、`norm_stats.json`、`metrics.jsonl` 和最近的 `checkpoint.pt`。恢复时保持数据、模型与 batch 配置一致，可以调大 `num_steps`。默认保留全部来源质量警示，不把统计或训练运行解释为人工验收通过。

在线 Python 接口只接收当前观察：

```python
from openpi.policies.policy_config import create_trained_policy

policy = create_trained_policy("checkpoints/libero40_eskill/checkpoint.pt", device="cuda")
result = policy.infer({
    "image": main_rgb_uint8,
    "state": source_state15,  # joint_state[7] + 源 state[8]，与保存数据同序
    "prompt": task_instruction,
})
actions = result["actions"]  # [H, 7]，反归一化后的源动作
```

该接口复用训练输入变换，加载时不创建视觉 VAE／去噪器，Flow Matching 的多步采样复用一次预测的事件条件。`scripts/serve_policy.py --runtime local` 当前只接入双相机纯 SFT 配置，尚未连接本 ESkill 策略；ESkill 闭环前仍需接通其服务观察映射并核实源动作语义。

CPU 小模型、真实数据抽样和断点恢复结果见 [training_interface_v1](../../data/libero/atomic-vla/check/test/training_interface_v1/README.md)。该结果使用 `smoke=true` 和随机小模型／冻结随机 VAE，只供工程接口核验，不供仿真控制或效果评估。
