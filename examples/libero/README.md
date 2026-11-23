# LIBERO 数据、训练与测评

当前视觉模型使用 [Visual Primitive 配方](../../experiments/visual_primitive/README.md)。

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

## 帧采样与标签核验

ESkill 每臂八个 Motion tokens 对应未来 +5、10…40 个保存主相机帧，分别监督当前 Event 内的二维 EEF 与夹爪控制命令。终点重复不进入主辅助损失；Future Object 仍对应完整 Event 终点。主相机 Grounding 使用 Current／Future Object 两组画布。

Current 画布保持现有范围；`model.future_bbox_expand_px=8` 使 Future 在现有范围上每边额外扩大 8 个 256×256 画布像素，保留终点附近上下文。无需重新导出源 bbox；设为 0 可关闭额外外扩。

第一版不强制固定时间戳；夹爪命令已全量核验，按保存帧配对读取。按 LIBERO 默认 20 Hz 估算约覆盖到 2 秒，不把估算写入源时间轴。检查结果见 [训练数据核验](../../data/libero/atomic-vla/check/test/training/README.md)。

~~~bash
PYTHONPATH=src:. python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --split data/libero/atomic-vla/split.json --subset train
~~~

Dataset 在解码 RGB 前检查命令语义。原始数据无需重复导出；标签缺失只影响对应损失，不隐藏在线 Motion token。

## 训练

当前与历史实验入口见 [experiments/README.md](../../experiments/README.md)。历史目录中的 JSON 仅用于追溯，不交给当前训练器。

正式 split 为 [split.json](../../data/libero/atomic-vla/split.json)：每任务内按完整父轨迹约 90%／10%，共 1,518 train／175 val。相同父轨迹的帧、视角与派生记录共用归属，归一化只读取 train。

当前 ESkill 示例开启 `skill_sampling=true`，按当前观察的有效技能频次使用 `min(3, sqrt(N_max/N_skill))` 权重，有放回提高稀有技能的采样频率。设为 `false` 恢复普通随机采样。验证子集和归一化统计不重加权；运行目录的 `sampling.json` 记录实际类别计数与预期占比。具体多卡与恢复语义见 [训练接口](../../src/openpi/training/README.md)。

| 配方 | 用途 |
| --- | --- |
| [train_pi05_sft_val.json](train_pi05_sft_val.json) | 使用固定 train／val 的纯 π0.5 SFT |
| [train_eskill.json](train_eskill.json) | 从同划分 SFT 初始化当前 ESkill，六项损失联合训练 |
| [直接联合训练对照](../../experiments/eskill_libero40_direct/train.json) | 官方基座直接40k，全量解冻，无增强／视觉遮蔽／技能重采样；Action=1，五项辅助各0.1 |
| [颜色增强](../../experiments/eskill_libero40_direct/color.json)／[同步裁切](../../experiments/eskill_libero40_direct/color_crop.json) | 在直接联合训练对照上分别增加颜色、颜色＋主图裁切；参数和核验见[实验说明](../../experiments/eskill_libero40_direct/README.md#视觉增强对照) |
| [同步旋转](../../experiments/eskill_libero40_direct/color_crop_rotation.json) | A2上增加±5°中心旋转，合成一次采样，同步对象画布／权重与EEF轨迹；腕图几何不变 |
| [train_pi05_sft.json](train_pi05_sft.json) | 全量数据 π0.5 示例；与独立验证配方分开 |
| [train_pi05_lora.json](train_pi05_lora.json) | π0.5 LoRA 数据接入示例 |

ESkill 模板为 32 步接口示例，采用双相机、horizon=10、512 维小专家、冻结 VLM 与受控 Action 视觉遮蔽。初始化 checkpoint、预算和损失权重需按实际实验填写，示例参数未经真实训练调优。

~~~bash
PYTHONPATH=src:. python scripts/compute_norm_stats.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json
PYTHONPATH=src:. python scripts/train.py --config examples/libero/train_eskill.json \
  --resume /path/to/eskill/checkpoint.pt
~~~

在上述联合训练示例中，`init_checkpoint` 只转移纯 SFT 策略权重，ESkill 优化器和步数重新开始；`--resume` 恢复同结构训练状态。split、动作归一化、骨干与 LoRA 设置须与初始化对应。多卡用 `torchrun --standalone --nproc-per-node=<卡数> scripts/train.py --config ...`，batch_size 为每卡大小。结构、指标与保存规则见 [训练接口](../../src/openpi/training/README.md)。

## 在线推理

推理只读取当前 RGB 与任务指令，不需要相机标定或二维起点。未来 Event 标签、轨迹投影及终点图像只用于训练监督。

~~~python
from openpi.policies.policy_config import create_trained_policy

policy = create_trained_policy("/path/to/eskill/checkpoint.pt", device="cuda")
result = policy.infer({
    "images": [main_rgb_uint8, wrist_rgb_uint8],
    "prompt": task_instruction,
})
actions = result["actions"]  # [H, 7]，反归一化的源控制命令
~~~

训练与推理共享图像 resize 规则；相机标定仅用于离线生成未来轨迹标签。纯动作推理跳过 VAE、二维路径头和夹爪读出头，返回动作与当前技能。

## 闭环测评

查看当前模型的对象预测、轨迹和 Target／Object／Instruction 注意力，服务启用 `--record-predictions`，再用 `scripts/prediction_viewer.py --rollout-record ...` 回读。正式视频跟随实际策略请求更新，下方输入与注意力在默认 5 步执行块内保持，上方环境画面逐控制帧播放。`--episode` 保留为离线示范诊断。命令、图例与原始数组说明见[模型预测与 Attention 面板](../../docs/evaluation.md#模型预测与-attention-面板)。

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

当前 ESkill 服务握手不要求状态或相机标定；客户端发送源分辨率 RGB 与任务指令，服务端统一 resize，不进行在线 EEF 投影。官方 OpenPI checkpoint 可通过 `--runtime official` 与对应 `--config` 启动。

固定协议为 seed=7、官方初态、等待 10 步、每个动作块执行前 5 步再观察；Spatial／Object／Goal／LIBERO-10 的控制上限分别为 220／280／300／520。任务回合数、FM 步数和任何协议变化随配置记录。

结果保存逐回合成功／异常、动作块与实际执行动作、状态和视频。技术检查标为 `technical`，成功率为 `null`；真实模型效果需独立训练和闭环测评，不由接口检查推断。

## 两阶段镜像训练（不带复用）

当前正式预算为 Stage 1 原生 π0.5 初始化 20k，Stage 2 接续 40k；两份配置为 [train_eskill_stage1.json](train_eskill_stage1.json)、[train_eskill_stage2.json](train_eskill_stage2.json)。在远端调整数据盘路径、输出目录、每卡 batch 与卡数后执行；本机不启动训练。

```bash
export PYTHONPATH=src:.
python scripts/compute_norm_stats.py --config examples/libero/train_eskill_stage1.json
# 以下以远端八卡为例，全局 batch=8×8=64；按实际卡数调整。
torchrun --standalone --nproc-per-node=8 scripts/train.py --config examples/libero/train_eskill_stage1.json
torchrun --standalone --nproc-per-node=8 scripts/train.py --config examples/libero/train_eskill_stage2.json
```

两个阶段共用第一条命令生成的 train-only 统计；Stage 2 的 `init_checkpoint` 指向 Stage 1 的 `checkpoint_020000.pt`，不是 `--resume`。分别恢复时，对各自配置使用 `--resume /path/to/that_stage/checkpoint.pt`。

Stage 1 冻结 VLM／Action／VAE，只训练先验及辅助头，随机翻转符合条件的样本；Stage 2 使用原图，设置 `model.freeze_vlm=false`，解冻视觉编码器、视觉投影与语言主干，联合训练 Skill／Grounding／Motion 与 Action／Motion K/V，保留五项辅助损失并加入 FM，仅 VAE 编码器冻结，不做 Functional Reuse。每 5k 保存、每 1k 验证，未安排自动闭环测评。Stage 1 不能交给 `run_libero.py` 或策略服务做动作评测；Stage 2 checkpoint 可用现有 LIBERO／Plus 接口评测。

Stage 1→2 权重交接允许改变 VLM 冻结设置；同一阶段的 `--resume` 不允许改变冻结范围。解冻后需在远端重新核对每卡 batch 与显存，旧冻结短测的资源数值不适用。

完整冻结范围与镜像规则见[训练接口](../../src/openpi/training/README.md)，状态记录见[实验说明](../../experiments/eskill_libero40_mirror/README.md)。

Motion 继续使用 +5…40 帧八点，`model.motion_min_horizon_frames=10` 为最短窗口（名义约 0.5 秒），允许前两个点跨 Event；不跨 Episode、不重复终点。Skill 和 Future Object 仍使用当前 Event 标签与目标。
## Current Grounding 数据与训练接口

跨服务器操作见 [数据准备与训练使用说明](../../docs/remote_grounding_training.md)：涵盖现有 ESkill-VLA 仓库更新、完整数据同步、候选标记刷新、真实权重短测及训练恢复。

[增强配置](train_current_grounding.json) 与 [Event 对照](train_current_grounding_baseline.json) 均为 32 步接口模板，主相机、融合图文特征的单 Current 与读取未融合特征的夹爪 Subject，VLM 冻结，`model.grounding_subject=true`。初始化为原 π0.5＋FLUX VAE，尚未启动正式训练。增强配置分两阶段串联：第一阶段为单目标反事实、Object Copy-Paste 和保护区外 Patch Masking（`grounding_augmentation`），第二阶段按官方 π0.5 PyTorch 参数做固定 95% 边长裁切、±5°旋转和颜色扰动（幅度 0.3／0.4／0.5，均每次采样），水平翻转关闭（`horizontal_flip_prob`、`image_augmentation`）；详见 [训练说明](../../src/openpi/training/README.md#current-grounding-专项训练)。

首次场景标注沿用 `review_interaction_targets.json` 的 inventory／extract（`protect_robot=true`）。已有完整框后，用 [prepare_grounding.json](prepare_grounding.json) 在 LIBERO 原生环境补可见 mask、编辑资格与局部交互区：

```bash
export PYTHONPATH=src:.
python -m scripts.preprocess.libero.interaction_review \
  --config examples/libero/prepare_grounding.json --stage augment
```

上面的十条试验数据引用原资产；全量数据改用下面的自包含目录。后续训练读取不依赖仿真，也不在线调用视觉模型。普通 PyTorch 环境执行下列核验，输出目录需尚不存在：

```bash
python -m scripts.preprocess.libero.grounding_review \
  --config examples/libero/train_current_grounding.json \
  --output /data1/yuankai/robot-research/workspace/reports/grounding_review_new --vae-device cuda:1
python -m scripts.preprocess.libero.grounding_review \
  --config examples/libero/train_current_grounding.json \
  --output /data1/yuankai/robot-research/workspace/reports/grounding_benchmark_new --benchmark --device cuda:1
```

组合增强预览使用训练同一 batch 变换入口，按实际概率展示每任务一条轨迹的 Event 中点，包含原图、Dataset 编辑后图、最终输入和两个几何变换后的标签及 VAE 重建；视频每张面板停留两秒。吞吐检查使用真实数据、tokenizer、FLUX VAE，加随机小骨干和 AdamW；明确为 smoke，关闭 W&B，不代表完整 π0.5 性能。结果与视频见 [验收记录](../../data/libero/atomic-vla/check/test/online_grounding/README.md)。

历史 `--export-samples N` 只支持 Dataset 阶段的核验副本，使用时须在独立配置中设 `horizontal_flip_prob=0`、`image_augmentation={}`；完整组合效果使用上面的默认预览。它不启动训练：均匀选取源帧，每帧两次在线增强，保存图片、精确输入／监督 NPZ、来源与文本索引。CPU 开关增强交替各测三轮，写盘及可选冻结 VAE 编码单独计时，视频回读保存图片生成；各来源保留父轨迹 split。现有 [200 条样本与耗时](../../data/libero/atomic-vla/check/test/online_grounding/export200/README.md) 只覆盖已补齐标注的十条轨迹。

全量补齐使用 [prepare_grounding_full.json](prepare_grounding_full.json)，在 `libero-native` 环境执行，EGL 配置同前面的原生渲染入口：

```bash
python -m scripts.preprocess.libero.grounding_batch \
  --config examples/libero/prepare_grounding_full.json
```

每个任务一个隔离进程，同任务复用环境；完全相同的实体和相机姿态复用完整 mask。逐条检查 mask／bbox 一致性、原五帧结果及十条逐帧试验的重叠部分。完成记录用于中断后续跑，全部通过后才生成 `index.json`。新目录 `atomicdata/libero40_grounding/episodes/<episode_id>/` 内包含：

| 文件 | 内容 |
| --- | --- |
| `episode.json` | Event 目标、场景实体与本地数组引用 |
| `arrays.npz`、`images.npz` | 原数组和图像，本机使用硬链接复用，须视为只读 |
| `scene_regions.npz` | 全场景逐帧完整屏内框／mask，部件和目标区域 |
| `online_grounding.npz` | 可见 mask、可见比例、编辑资格及局部保护区 |
| `grounding_check.json` | 该条保存回读核验记录 |

同步到其他机器时复制整个数据目录，训练不需要 provenance 中的原始缓存。用训练环境检查输入链路及覆盖全部任务的标签预览：

```bash
python -m scripts.preprocess.libero.grounding_check \
  --config examples/libero/train_current_grounding.json \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40_grounding/index.json \
  --output /data1/yuankai/robot-research/workspace/reports/grounding_input_new \
  --workers 2 --smoke-device cuda:3
```

检查覆盖每条轨迹及每个 Event 的首／中／末帧、原 train／val 划分和多 worker 一致性；可选反传仅用随机小骨干＋真实冻结 FLUX VAE，不是完整 π0.5 训练。`all_tasks.mp4` 从保存结果回读，每任务选一条轨迹、每 Event 中点停留两秒；红框是目标、绿框是其他物体、蓝框是夹爪。

完成标签人工验收并确定正式预算后，训练仍用 `python scripts/train.py --config …`，无需另起训练器。在线仅传主相机：

```python
from openpi.policies.current_grounding import create_current_grounding_policy
policy = create_current_grounding_policy('/path/to/grounding/checkpoint.pt', device='cuda')
result = policy.infer([main_rgb], 'put the milk into the basket')
# grounding 为 Current tokens；subject_grounding 为夹爪 tokens。
# 直接使用完整任务；推理不读取真实 bbox 或 Event。
```

新接口没有校准的目标存在概率，不接管动作 rollout。需要新开训练，旧生成短句的专项权重不兼容。
