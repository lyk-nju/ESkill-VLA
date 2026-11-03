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

## 按任务划分与验证（新训练）

正式拆分使用 [split.json](../../data/libero/atomic-vla/split.json)：**每个任务内部**按完整父轨迹 90%／10%，不是全体数据总体划分。40 任务共 1,518 train／175 val；同一轨迹的主／腕相机、事件标签和相机标定版共用归属，不改写或复制原 Episode。逐任务数量与检查见 [task_split_v1](../../data/libero/atomic-vla/check/test/task_split_v1/README.md)。

新 SFT 使用 [train_pi05_sft_val.json](train_pi05_sft_val.json)，辅助阶段使用 [train_eskill_val.json](train_eskill_val.json)，两者引用同一 `split_path`。SFT train-only 分位数已生成；若更换清单或统计路径，先执行 `scripts/compute_norm_stats.py --config ...`。辅助阶段在新 SFT checkpoint 完成后计算其新增状态统计，再启动：

```bash
POLICY_PY=/home/yuankai/robot-research/RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
export PYTHONPATH=src:.

"$POLICY_PY" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py \
  --config examples/libero/train_pi05_sft_val.json

# 新 SFT 完成后，使用其 checkpoint_010000.pt 初始化辅助分支。
"$POLICY_PY" scripts/compute_norm_stats.py --config examples/libero/train_eskill_val.json
"$POLICY_PY" -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py \
  --config examples/libero/train_eskill_val.json
```

辅助阶段使用全量带标定的 `libero40/motion_2d/index.json`，1,518／175 归属与 SFT 缓存完全一致。每 1,000 步及末步在固定验证子集运行 val，每卡最多 32 batch；只上传对应的五项 `val_*loss`，纯 SFT 为总／动作两项。统计只读 train，验证不反传，随机状态在验证后恢复。`val_interval=0` 只关闭验证，不把 val 并回 train。

旧 `train_pi05_sft.json` 和既有 10k／20k／30k 全量 SFT 记录保持原义；新独立 val 方案从原始 π0.5 基座重新开始 SFT。初始化入口拒绝将不同拆分的 SFT checkpoint 用于新辅助训练。

## 八点二维 Motion

每臂 8 个 Motion tokens，逐 token 共享 MLP 预测主相机归一化末端坐标，默认未来偏移 `[4,8,12,16,20,24,28,32]` 个来源帧；观察与动作不降采样。新 [十条核验入口](../../data/libero/atomic-vla/check/test/motion_2d_8x4_v1/README.md) 包含训练索引、保存标签、合并视频和 Dataset 回读结果。

在新的输出目录执行（沿用前文原生环境设置）：

```bash
"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.motion_2d \
  --reuse-projection data/libero/atomic-vla/check/test/motion_2d_unit4_hold_v1 \
  --output /path/to/new/motion8x4 \
  --video-dir /path/to/new/motion8x4/video
```

默认使用已有十条轨迹。`--offsets` 接受递增正整数，`--time-unit-frames` 默认 1；新八点不要再额外乘四。来源文件和旧试验保持原内容。全量补齐使用下面的批处理入口，按场景复用仿真环境，逐条检查原始模型与每帧末端，并逐帧核对 Dataset 回读：

```bash
"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.calibrate \
  --output /path/to/new/calibrated_episodes \
  --report-dir /path/to/new/check
```

已有全量结果位于 `/data1/yuankai/robot-research/data/atomicdata/libero40/motion_2d/index.json`，覆盖 40 任务、1,693 条／273,465 帧；[检查记录](../../data/libero/atomic-vla/check/test/calibration_full_v1/README.md)。`--resume` 校验来源哈希后复用完整结果；`--annotation-ids` 可指定小批量。

输出 `index.json` 指向派生 `episodes/<id>/episode.json`：复用原 RGB／数值数组，补 `cameras.image.calibration`，训练不运行仿真。Dataset 从保存的末端与相机信息构造标签，与本批 `motion.npz` 的八点核对一致。[train.json](train.json) 已指向全量带标定索引；基础 `libero40/index.json` 保留原内容，二维 Motion 训练使用 `motion_2d/index.json`。

`hold_last` 在实际 Episode 末帧有效且在画面内时补齐；内部缺帧、末帧无效或屏外点仍屏蔽。补齐参与轨迹损失，以 `future_padded` 与实际取值帧记录来源；未来技能及动作各自保留原有尾部规则。

[train_motion_2d_check.json](train_motion_2d_check.json) 保留此前八点 Motion 的接线配置及临时 SD VAE 路径。当前代码已采用 FLUX 潜变量回归，新联合检查使用 [train_grounding_check.json](train_grounding_check.json)，包含 `motion_queries=8`、`motion_stride=4`、`motion_tail_policy=hold_last`、`role_queries=16` 和 `canvas_size=256`。设置新的输出目录与统计路径后：

```bash
"$POLICY_PY" scripts/compute_norm_stats.py --config examples/libero/train_grounding_check.json
torchrun --standalone --nproc-per-node=4 scripts/train.py --config examples/libero/train_grounding_check.json
```

八点 Motion 的首次检查采用 CPU 小模型；后续 [FLUX 联合检查](../../data/libero/atomic-vla/check/test/grounding_mse_v1/README.md)已使用真实 π0.5 LoRA 完成四卡 8 步训练和恢复至 10 步。新轨迹头与旧三维辅助 checkpoint 不兼容，使用 SFT 初始化或新二维／新视觉 checkpoint 恢复。历史 [六点试验](../../data/libero/atomic-vla/check/test/motion_2d_unit4_hold_v1/README.md) 参数和结果保持原义；复跑六点须显式指定 `--offsets 2 4 5 7 9 10 --time-unit-frames 4` 及新的输出目录。

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

`local` 服务按 checkpoint 选择适配：纯 SFT 使用 `libero_sft` 双相机与原始 8 维状态；ESkill 使用 `episodes`，客户端根据握手要求额外发送真实 `robot0_joint_pos[7]`，服务拼接为 15 维状态。两者输出均为 7 维动作；官方 checkpoint 使用 `--runtime official`。数据缓存由 `scripts/preprocess/libero/atomic_vla/prepare_sft.py` 生成，保留源图像字节、轨迹配对与动作；本轮配置与输出位置以实验记录为准。

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

先设置两个本地权重路径：`pretrained_path` 指向 π0.5 的 `model.safetensors`，`vae_path` 指向预训练 `diffusers.AutoencoderKL` 目录。π0.5 已按本地上游映射将 JAX／Orbax 权重转换为 BF16 safetensors，位置为 `/data1/yuankai/robot-research/ckpt/pi05/base/pytorch_bf16/model.safetensors`；转换信息见同目录 `conversion.json`。真实 VAE 与完整 ESkill 分支已在下方 SFT 初始化配置中完成四卡短程检查；正式配置缺少权重会明确报错。

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

该接口复用训练输入变换，加载时不创建视觉 VAE／训练监督模块，Flow Matching 的多步采样复用一次预测的事件条件。`scripts/serve_policy.py --runtime local --checkpoint /path/to/eskill/checkpoint.pt` 已接入 ESkill，沿用上述 LIBERO 客户端命令。带 `sft_data_index` 时传主／腕双相机，否则按训练配置的 `camera` 传单相机；客户端发送真实七维关节位置，禁止补零。技术闭环结果见 [eskill_service_v1](../../data/libero/atomic-vla/check/test/eskill_service_v1/README.md)，正式成功率需单独测评。

CPU 小模型、真实数据抽样和断点恢复结果见 [training_interface_v1](../../data/libero/atomic-vla/check/test/training_interface_v1/README.md)。该结果使用 `smoke=true` 和随机小模型／冻结随机 VAE，只供工程接口核验，不供仿真控制或效果评估。

## 从 SFT 开始辅助联合训练

[train_grounding_check.json](train_grounding_check.json) 是当前新视觉方案的四卡检查配置：从已完成的 10k 步 π0.5 LoRA SFT 接入 ESkill，保留动作损失并加入当前技能、未来技能、FLUX 潜变量 MSE 和八点二维 Motion 监督。LoRA rank/alpha=16/16，每卡 batch=1、全局 4，学习率 1e-5，先执行 8 步；基座与 VAE 冻结，更新 LoRA 和新分支。`init_checkpoint` 只用于初始化，开始新实验时不传 `--resume`。

每角色 16 个 256 维 Token 经共享 LayerNorm → Linear(256,512) → GELU → Linear(512,1024)，按 4×4 顺序拼成 16×32×32 潜变量。VAE 使用 FP32／mode 编码，权重版本见 [Target Grounding](../../docs/target_grounding.md)。已执行的真实四卡训练、恢复与推理检查见 [grounding_mse_v1](../../data/libero/atomic-vla/check/test/grounding_mse_v1/README.md)。新预测头独立初始化，旧去噪辅助 checkpoint 不能作为新头的同配置断点恢复。

```bash
python scripts/compute_norm_stats.py --config examples/libero/train_grounding_check.json
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py \
  --config examples/libero/train_grounding_check.json
```

数据保持双相机、10 步动作窗及 SFT 的动作／状态分位数统计；新增本体状态使用训练集均值方差。只在主相机生成区域监督。此检查关闭随机图像增强，以保持输入与区域目标位置一致；动作使用完整 32 维 MSE，辅助标签各自屏蔽缺失和 padding。

执行新实验前设置新的 `output_dir` 和 `norm_stats`；已有同来源统计可直接复用，无需再次计算。VAE 加载器严格检查全部参数，自动映射所用原始 FLUX 权重名。W&B 默认读取根目录 `wandb.json`，run 名称按实际全局 batch 与配方标明 `pi05_eskill_lora_bs4_…_flux_mse…`。W&B 的 `train`／`val` 记录总／动作／视觉／技能／轨迹损失，技能主曲线合并当前／未来技能；`chart` 记录学习率、梯度范数、步耗时和步数。`aux_train`／`aux_val` 各只上传 6 项摘要：三角色 Grounding、当前／未来技能准确率、Motion 总像素误差。类别频率、前景／背景、各偏移及真实／补齐点计数等详细诊断仅保存在本地 JSONL。明细见 [训练说明](../../src/openpi/training/README.md)，本地 JSONL 同步保留。

恢复时调大 `num_steps`，在相同四卡命令后加 `--resume checkpoints/libero40_grounding_mse_check/checkpoint.pt`；之后不再需要 SFT checkpoint。Python 策略回读时同时传 `image`、`wrist_image` 和源 15 维状态，加载时不创建 VAE 或潜变量投影头。

本配置使用十条带相机标定的数据。扩展至全量 1,693 条前需补齐二维 Motion 标定；短程检查仅验证训练接口，收敛与闭环效果仍待正式训练评估。[train_sft_auxiliary.json](train_sft_auxiliary.json) 和 [train_motion_2d_check.json](train_motion_2d_check.json) 保留旧检查时的配置；历史结果按原记录解释。

## 关闭数值本体状态的试验

[train_eskill_no_state.json](train_eskill_no_state.json) 沿用按任务划分的 ESkill 配方，设置 `model.use_arm_state=false`、`model.discrete_state_input=false`。模型不创建本体状态投影层，也不在 Motion／未来技能注意力中放置状态 Token；输入只保留图像、任务文本和臂存在信息。动作／技能／视觉／运动监督继续使用原有标签。

它从相同 SFT checkpoint 开始新实验，输出目录与 W&B 名称增加 `_no_state`；四卡全局 batch=4、辅助训练 30k 步等参数沿用原配方。正式训练尚未启动。准备好 SFT 初始化权重和 train-only 统计后运行：

```bash
torchrun --standalone --nproc-per-node=4 scripts/train.py --config examples/libero/train_eskill_no_state.json
```

此配置的 LIBERO 服务不要求 `observation/joint_state`，直接传源八维末端／夹爪状态以复用现有接口；该状态不进入文本或模型条件。原始十五维训练记录仍可读取，前七维不会进入模型。来源状态和末端记录继续用于生成监督。开关、训练与回读核验见 [no_state_v1](../../data/libero/atomic-vla/check/test/no_state_v1/README.md)。
