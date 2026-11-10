# LIBERO 数据、训练与测评

从项目根目录运行，环境中安装本项目或设置 `PYTHONPATH=src:.`。正式数据为 `/data1/yuankai/robot-research/data/atomicdata/libero40/index.json`：40 任务、1,693 条／273,465 帧，每条 Episode 自包含元数据、状态／动作／区域／相机数组及双相机图像。来源与人工核验入口见 [AtomicVLA](../../data/libero/atomic-vla/README.md)，保存格式见 [data/README.md](../../data/README.md)。

## 数据准备与可视化

导出器复用已有事件、实例绑定和完整屏内区域；输出目录应尚不存在：

~~~bash
python scripts/prepare_data.py atomic-vla \
  --annotation-id 51 \
  --regions data/libero/atomic-vla/check/test/manual_binding12/regions/libero_10_no_noops_s004_r004.json \
  --output /path/to/new/episode_0051

python scripts/libero_render.py \
  --episode /path/to/new/episode_0051/episode.json \
  --video data/libero/atomic-vla/check/test/video/episode_0051_review.mp4
~~~

全量区域生成入口为 `scripts.preprocess.libero.atomic_vla.batch`，依次执行 `prepare / build / review / report`。固定批次与验收清单见 [full_frame_v1](../../data/libero/atomic-vla/check/test/full_frame_v1/README.md)，已有 55 条视频覆盖全部 40 任务。大体积标签和视频保存在数据盘；新检查视频放对应来源 `check/test/video/`。

补相机标定使用已有原生 LIBERO 环境：

~~~bash
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export PYTHONPATH=src:.
export LIBERO_CONFIG_PATH=/data1/yuankai/robot-research/workspace/envs/libero-native/config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$LIBERO_CONFIG_PATH/nvidia_egl.json"

"$NATIVE_PY" -m scripts.preprocess.libero.atomic_vla.calibrate \
  --index /path/to/staged_episodes/index.json \
  --sources data/libero/atomic-vla/check/test/all_rules_v1 \
  --output /path/to/new/calibrated_episodes \
  --report-dir /path/to/new/check
~~~

当前正式 Episode 已有逐帧区域、EEF 源像素坐标与全量标定。重新生成时使用新目录，保存来源及绑定依据，再从已保存标签回读渲染。

## 时间与标签核验

ESkill 每臂八个 Motion tokens 对应 +0.25s…+2s，分别监督当前 Event 内的二维 EEF 与夹爪控制命令。终点重复不进入主辅助损失；Future Object 仍对应完整 Event 终点。主相机 Grounding 使用 Current／Future Object 两组画布。

秒级监督要求真实时间或有证据的控制时钟＋逐帧映射，并核实夹爪命令列、符号与执行时序。现有数据尚需补齐这些声明，不能将渲染 fps 或保存行号当作控制时钟。清单 schema 和当前检查见 [训练数据核验](../../data/libero/atomic-vla/check/test/training/README.md)。

~~~bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --timing /path/to/verified_libero40_timing.json \
  --split data/libero/atomic-vla/split.json --subset train
~~~

Dataset 在解码 RGB 前检查时间依据。原始数据无需重复导出，`timing_path` 可引用独立来源清单；标签缺失只影响相应监督。

## 训练

正式 split 为 [split.json](../../data/libero/atomic-vla/split.json)：每任务内按完整父轨迹约 90%／10%，共 1,518 train／175 val。相同父轨迹的帧、视角与派生记录共用归属，归一化只读取 train。

| 配方 | 用途 |
| --- | --- |
| [train_pi05_sft_val.json](train_pi05_sft_val.json) | 使用固定 train／val 的纯 π0.5 SFT |
| [train_eskill.json](train_eskill.json) | 从同划分 SFT 初始化当前 ESkill，六项损失联合训练 |
| [train_pi05_sft.json](train_pi05_sft.json) | 全量数据 π0.5 示例；与独立验证配方分开 |
| [train_pi05_lora.json](train_pi05_lora.json) | π0.5 LoRA 数据接入示例 |

ESkill 模板为 32 步接口示例，采用双相机、horizon=10、512 维小专家、冻结 VLM 与受控 Action 视觉遮蔽。初始化 checkpoint、时间清单、预算和损失权重需按实际实验填写，示例参数未经真实训练调优。

~~~bash
PYTHONPATH=src:. python scripts/compute_norm_stats.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json \
  --resume /path/to/eskill/checkpoint.pt
~~~

`init_checkpoint` 只转移纯 SFT 策略权重，ESkill 优化器和步数重新开始；`--resume` 恢复同结构训练状态。split、动作归一化、骨干与 LoRA 设置须与初始化对应。多卡用 `torchrun --standalone --nproc-per-node=<卡数> scripts/train.py --config ...`，batch_size 为每卡大小。结构、指标与保存规则见 [训练接口](../../src/openpi/training/README.md)。

## 在线推理

推理读取当前 RGB、任务、源八维 EEF／双指状态与主相机标定。二维起点通过实际相机和图像变换投影；也可直接提供按模型主相机画布归一化的 `effector_start[2,2]` 与有效性。未来 Event 标签及终点图像不进入输入。

~~~python
from openpi.policies.policy_config import create_trained_policy

policy = create_trained_policy("/path/to/eskill/checkpoint.pt", device="cuda")
result = policy.infer({
    "images": [main_rgb_uint8, wrist_rgb_uint8],
    "state": source_state8,
    "main_camera": camera_parameters,
    "prompt": task_instruction,
})
actions = result["actions"]  # [H, 7]，反归一化的源控制命令
~~~

`main_camera` 包含 `intrinsic`、`world_to_camera`、`image_transform`、`image_shape`。训练与推理共享投影／resize 规则。纯动作推理跳过 VAE、二维路径头和夹爪读出头，返回动作与当前技能。

## 闭环测评

LIBERO／Plus 仿真客户端使用独立 Python 3.8 环境，策略服务使用 Python 3.11。[环境与版本](../../../workspace/envs/libero-evaluation/README.md) · [Plus 用法](../libero_plus/README.md) · [测评协议](../../docs/evaluation.md)。

两个终端分别启动本项目 checkpoint 服务和客户端：

~~~bash
python scripts/serve_policy.py --runtime local \
  --checkpoint /path/to/eskill/checkpoint.pt --port 8000

SIM_PY=/data1/yuankai/robot-research/workspace/envs/libero-eval/bin/python
"$SIM_PY" examples/libero/main.py --task-suite-name libero_spatial --task-ids 0 \
  --num-trials-per-task 1 --port 8000 --checkpoint-tag /path/to/eskill/checkpoint.pt \
  --output /data1/yuankai/robot-research/workspace/experiment/libero_eval/run_001
~~~

ESkill 服务握手要求真实主相机参数，客户端发送源分辨率旋转图像，服务端统一图像与投影的 resize。Plus 相机扰动使用每次请求的实际参数。官方 OpenPI checkpoint 可通过 `--runtime official` 与对应 `--config` 启动。

固定协议为 seed=7、官方初态、等待 10 步、每个动作块执行前 5 步再观察；Spatial／Object／Goal／LIBERO-10 的控制上限分别为 220／280／300／520。任务回合数、FM 步数和任何协议变化随配置记录。

结果保存逐回合成功／异常、动作块与实际执行动作、状态和视频。技术检查标为 `technical`，成功率为 `null`；真实模型效果需独立训练和闭环测评，不由接口检查推断。
