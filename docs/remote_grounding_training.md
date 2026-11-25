# 在其他服务器准备数据并训练 Current／Subject Grounding

下文记录冻结 VLM 的旧 Grounding 对照，包含容器 Copy、锁定标记启动检查与附属 Region 保护。远端可以在现有 `ESkill-VLA` 或 `eskillvla` 目录使用 `visual-primitive` 分支。当前多层原生注意力训练使用 [Visual Primitive 配方](../experiments/visual_primitive/README.md)。

旧对照使用 `training_stage=grounding`。模型接收主相机与完整任务指令，冻结 π₀.₅ VLM 与 FLUX VAE，训练 Current／Subject 读取模块，损失权重各 0.1。

## 1. 更新代码

在远端现有仓库检查工作区，保存已有修改后切到对应分支：

```bash
cd /你的代码目录/ESkill-VLA
git status --short
git fetch origin
git switch visual-primitive
git pull --ff-only origin visual-primitive
git rev-parse HEAD
```

远端应同步包含本说明的代码快照。旧提交 `8a41bfd` 仍使用此前的增强策略。同步后确认 `scripts/preprocess/libero/pick_precontact.py` 已存在，并用最新的 `examples/libero/train_current_grounding.json` 生成本次配置。

后续命令都从远端仓库根目录执行。先填写远端数据盘根路径：

```bash
export ESKILL_ASSETS=/你的数据盘/robot-research
export ESKILL_RUN="$ESKILL_ASSETS/workspace/experiment/current_grounding_001"
export ESKILL_DATA_INDEX="$ESKILL_ASSETS/data/atomicdata/libero40_grounding/index.json"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
mkdir -p "$ESKILL_RUN"
```

配置、启动日志放在 `ESKILL_RUN`，训练器输出放在其 `train/` 子目录；首次训练前不要在 `train/` 内放文件。

## 2. 同步完整数据与权重

已有完整 Grounding 副本时，直接执行第 4 节的增量更新。首次部署可以同步本机完整数据和权重。远端有完整 Grounding 和匹配的补标报告时，训练环境无需安装 LIBERO 或 SAPIEN。

| 资源 | 本机来源 | 远端相对 `ESKILL_ASSETS` 的位置 |
| --- | --- | --- |
| 全量 Grounding 数据 | `/data1/yuankai/robot-research/data/atomicdata/libero40_grounding/` | `data/atomicdata/libero40_grounding/` |
| 官方 π0.5 PyTorch 基座和 tokenizer | `/data1/yuankai/robot-research/ckpt/pi05/base/` | `ckpt/pi05/base/` |
| FLUX VAE | `/data1/yuankai/robot-research/ckpt/vae/flux1-dev/` | `ckpt/vae/flux1-dev/` |
| 固定父轨迹划分 | 仓库 `data/libero/atomic-vla/split.json` | 随代码同步 |
| 归一化统计 | 仓库 `experiments/eskill_libero40/norm_stats.json` | 随代码同步 |

从远端拉取示例；将 `SOURCE` 换成能访问原机器的 SSH 地址：

```bash
export SOURCE=用户名@原机器地址
mkdir -p "$ESKILL_ASSETS/data/atomicdata" "$ESKILL_ASSETS/ckpt/pi05" "$ESKILL_ASSETS/ckpt/vae"
rsync -aL --partial --info=progress2 \
  "$SOURCE:/data1/yuankai/robot-research/data/atomicdata/libero40_grounding/" \
  "$ESKILL_ASSETS/data/atomicdata/libero40_grounding/"
rsync -aL --partial --info=progress2 \
  "$SOURCE:/data1/yuankai/robot-research/ckpt/pi05/base/" "$ESKILL_ASSETS/ckpt/pi05/base/"
rsync -aL --partial --info=progress2 \
  "$SOURCE:/data1/yuankai/robot-research/ckpt/vae/flux1-dev/" "$ESKILL_ASSETS/ckpt/vae/flux1-dev/"
```

`-L` 复制软链接指向的实际内容，避免远端留下失效链接。需要时重复执行，并加 `--checksum` 做传输复核。源数据应在迁移脚本结束后再同步。

**必须复制整个 `libero40_grounding/`，不能只复制 `index.json` 或旧 `libero40/episodes/`。** 当前数据为 40 任务、1,693 条轨迹、273,465 帧、3,873 Events；固定划分为 1,518 train／175 val。

每条 Episode 包含 `episode.json`、`arrays.npz`、`images.npz`、`scene_regions.npz`、`online_grounding.npz` 等实际引用文件。索引与运行时数组引用使用相对路径，整体搬迁后无需批量替换 Episode JSON；provenance 中的原机器路径仅记录来源。当前训练不读取那些原始缓存。

若远端只有旧 `libero40`，直接补同步上述完整目录。`interaction_locks` 只能更新已有框／mask上的布尔标记，不能凭空补齐缺失的原生 mask；从头渲染的入口见 [LIBERO 数据准备](../examples/libero/README.md#current-grounding-数据与训练接口)，需要独立仿真环境及原始源资产。

## 3. 环境

优先复用远端已验证的训练环境。若新建，环境和缓存放数据盘：

```bash
export UV_CACHE_DIR="$ESKILL_ASSETS/workspace/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ESKILL_ASSETS/workspace/python"
export PIP_CACHE_DIR="$ESKILL_ASSETS/workspace/cache/pip"
export HF_HOME="$ESKILL_ASSETS/workspace/cache/huggingface"
export TMPDIR="$ESKILL_ASSETS/workspace/tmp"
mkdir -p "$TMPDIR"
uv venv --python 3.11 "$ESKILL_ASSETS/workspace/envs/eskill-vla"
source "$ESKILL_ASSETS/workspace/envs/eskill-vla/bin/activate"
uv pip install -e '.[train,preprocess,test]'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
```

PyTorch 需要与服务器驱动兼容的 CUDA 构建；以上依赖声明不是锁定环境。现有 CUDA 环境无需重建，安装前确认依赖解析不会意外替换其 PyTorch。数据核验／视频需要 `preprocess` 中的依赖，训练 Dataset 本身不依赖仿真。可视化渲染使用 Noto CJK／DejaVu 字体；缺少字体时报错的机器需先安装对应字体包。训练本身不依赖这些渲染字体。保存实际环境：

```bash
uv pip freeze > "$ESKILL_RUN/environment.txt"
git rev-parse HEAD > "$ESKILL_RUN/commit.txt"
git status --short > "$ESKILL_RUN/worktree.txt"
nvidia-smi > "$ESKILL_RUN/gpu.txt"
```

## 4. 将 11 月 18 日的数据更新到 11 月 19 日

### 代码变化

- Current Grounding 主配方的位置增强触发率从 50% 提到 **70%**。优先移动的对象是文本替换前的 **原 Current**，即 Pick 优先待抓物体，Place 优先容器。合格时优先抽中的概率从 50% 提到 **75%**。每帧最多移动三个对象，之后只在 Place 独立按 30% 概率尝试文本替换。
- Copy 的原生可见率阈值从 98% 降到 **90%**。夹爪和关联物体按各自的框保护，明确空间指代的任务按当前 Event 保留相关参照，当前监督为附属 Region 时保护父物体。矩形擦除有冲突时尝试 **可见 mask 外扩 1 px**，仍检查机器人像素、邻物像素和保护区。含明确左右方向的指令启用水平翻转时会跳过该增强。
- Pick 接触前的原 Current 可按 `copy_precontact_stop_frame` 进入例外，真实接触前两帧起使用常规保护。擦除仍保护机器人像素和其他交互区域，粘贴仍避开夹爪外扩框。正式 Copy 配方用 **每边 4 px**。2 px 对照新增的 Object Place 搬动有 48/64 帧已接触容器，因此作为实验参数保留。

### 刷新已有框和 mask 的派生标记

如果远端已有昨天的完整 `libero40_grounding`，在用于新实验的副本上执行。所有汇报目录使用新路径。

```bash
python -m scripts.preprocess.libero.interaction_locks \
  --index "$ESKILL_DATA_INDEX" \
  --output "$ESKILL_RUN/eligibility_refresh" \
  --refresh-copy-eligibility
```

它补齐基于 4 px 框相交的 `interaction_locked`，从已有完整和可见 mask 刷新 **90%** 的 `copy_eligible`。两者都复用原始标注，已有 RGB、完整 bbox 和 mask 可直接用于新版本。检查 `eligibility_refresh/summary.json` 覆盖 1,693 条轨迹，旧 Episode JSON 备份在同目录下。若 `robot::single` mask、可见 mask 或源框本身缺失，按第 2 节同步完整资产。

### 生成 Pick 接触区间

截至 11 月 19 日本轮核验，本机共享目录仍有 **1,693 条轨迹、0 个 Pick 接触区间字段**。当天的接触标注只用于 Goal／Object 的试跑索引，报告覆盖 401 条、实际核对 317 条含匹配 Pick 的轨迹。同步共享目录后仍须执行本节。

有 LIBERO 原生环境和完整匹配状态的机器可以直接补标。将 `ESKILL_LIBERO_ROOT` 设为该机 LIBERO 仓库路径，按当地 LIBERO 安装配置使用其 BDDL 和资源目录。每条轨迹必须能读取 `interaction_annotation.bddl` 和 `scene_match.excerpt_path`，源状态摘要要与 `excerpt_sha256` 一致。路径变更时使用指向同一资产的本机路径。

```bash
PYTHONPATH="src:.:$ESKILL_LIBERO_ROOT" python -m scripts.preprocess.libero.pick_precontact \
  --index "$ESKILL_DATA_INDEX" --output "$ESKILL_RUN/pick_windows.json" --write
```

脚本只给有有效窗口的 Pick Event 写入 `copy_precontact_stop_frame`。字段区间右开，值是首次接触帧减 2。找不到可靠接触证据的 Event 用常规规则。正式补标省略 `--episode-csv`，覆盖整个索引。脚本从 `scene_match` 读取按动作对齐的原生状态，无需重新渲染。

远端只运行训练时，可在能读取原生状态的源机器运行同一命令，**省略 `--write`**，生成全量 `pick_windows.json`。将报告同步到远端的 `ESKILL_RUN`，确认两端使用同一份 Episode 和 Event 绑定，再把字段写入远端副本：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from scripts.preprocess.libero.pick_precontact import CONTACT_MARGIN, STOP_FIELD, write_annotations

index = Path(os.environ['ESKILL_DATA_INDEX'])
report = json.loads((Path(os.environ['ESKILL_RUN']) / 'pick_windows.json').read_text())
paths = {row['episode_id']: row['path'] for row in json.loads(index.read_text())['episodes']}
assert report['episodes'].keys() == paths.keys(), '报告必须覆盖当前索引的全部 Episode'
assert report['contact_margin_frames'] == CONTACT_MARGIN
for episode, row in report['episodes'].items():
    row['path'] = paths[episode]
    meta = json.loads((index.parent / row['path']).read_text())
    assert meta['episode_id'] == episode
    events = {event['event_id']: event for event in meta['events']}
    for event_id, annotation in row['events'].items():
        event = events[event_id]
        assert event['skill'] == 'Pick'
        assert event['interaction_target']['entity_id'] == annotation['target']
        start, stop = event['start_frame'], event['stop_frame']
        first = annotation['first_contact_frame']
        assert first is None or max(0, start - CONTACT_MARGIN) <= first < stop
        assert annotation[STOP_FIELD] == (start if first is None else max(start, first - CONTACT_MARGIN))
write_annotations(index, report)
PY
```

检查报告 Episode 总数为 1,693，按 `status` 记录完成和未匹配的轨迹。回读有字段的 Event，核对全部为 Pick，且 `start_frame < copy_precontact_stop_frame < stop_frame`。同时确认至少有实际样本命中该例外，避免旧数据走回退路径。当天 401 条试跑报告可用于核验该子集，覆盖全量前使用新生成的报告。

## 5. 生成远端配置，先做真实权重短测

仓库 `train_current_grounding.json` 是 **32 步接口模板**，不是正式训练预算。下列脚本替换远端路径并生成 8 步检查配置，保持真实 π0.5／FLUX 权重，关闭本次检查的 W&B：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['ESKILL_ASSETS']).resolve()
run = Path(os.environ['ESKILL_RUN']).resolve()
repo = Path.cwd()
c = json.loads(Path('examples/libero/train_current_grounding.json').read_text())
c.update(
    data_index=str(Path(os.environ['ESKILL_DATA_INDEX']).resolve()),
    tokenizer_path=str(root/'ckpt/pi05/base/big_vision/paligemma_tokenizer.model'),
    pretrained_path=str(root/'ckpt/pi05/base/pytorch_bf16/model.safetensors'),
    vae_path=str(root/'ckpt/vae/flux1-dev'),
    split_path=str(repo/'data/libero/atomic-vla/split.json'),
    norm_stats=str(repo/'experiments/eskill_libero40/norm_stats.json'),
    output_dir=str(run/'smoke_train'),
    batch_size=2, num_workers=2, num_steps=8, save_every=8,
    val_interval=8, val_max_batches=2, log_every=1,
    smoke=False, wandb_info={'mode':'disabled'},
    run_name='pi05_current_bs4_libero40_check',
)
(run/'smoke.json').write_text(json.dumps(c, indent=2)+'\n')
PY
```

这里 `smoke=False` 使用真实模型。双卡全局 batch 为 4。冻结 VLM／VAE，使用 BF16 autocast。配方使用 Copy 0.7、原 Current 优先 0.75、每帧最多移动 3 个物体、Place 反事实 0.3、Patch 0.5、夹爪 Copy 外扩 4 px，随后执行水平翻转 0.5、同步裁切／旋转和输入颜色扰动。Pick 帧使用原指令和原目标。

训练前先跑代码回归、输入和增强覆盖，输出目录每次用新路径：

```bash
python -m pytest -q tests/test_augmentation.py tests/test_training_data.py \
  tests/test_current_grounding_training.py -k 'copy or counterfactual or priority or mirror'
python -m scripts.preprocess.libero.grounding_check \
  --config "$ESKILL_RUN/smoke.json" --output "$ESKILL_RUN/input_check" --workers 2
python -m scripts.preprocess.libero.grounding_coverage \
  --config "$ESKILL_RUN/smoke.json" --samples-per-task 200 \
  --output "$ESKILL_RUN/coverage"
```

验收时检查：

- `input_check/input_check.json` 的划分无交叉，覆盖 1,518 train／175 val，验证集无 Copy、文本替换和 Patch，`all_tasks.mp4` 里的 Current／Subject 指向对应对象。
- `coverage/by_task.csv` 分开看 `copy_applied`、`copy_priority_target_moved`、`target_moved` 和 `target_swapped`。`coverage/samples.jsonl` 至少有 `copy_precontact_candidate=true` 的 Pick 样本，且帧号落在补标区间内。若抽样没命中，用一个已标注 Object Pick 的区间逐帧核验。
- 用预览抽查至少一帧 Pick 接触前搬动和一帧 Place。核对机器人像素与受保护物体保留，Current bbox 跟随最终文本选中的对象，VAE 标签与编辑后的干净图一致。

预览完整增强及真实 VAE 重建：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.preprocess.libero.grounding_review \
  --config "$ESKILL_RUN/smoke.json" --output "$ESKILL_RUN/augmentation_review" --vae-device cuda:0
```

GPU 空闲后做双卡启动检查，按服务器实际空卡调整 `CUDA_VISIBLE_DEVICES`：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m scripts.train \
  --config "$ESKILL_RUN/smoke.json" 2>&1 | tee "$ESKILL_RUN/smoke.log"
```

确认 Current／Subject loss 和梯度有限、验证完成、`smoke_train/checkpoint.pt` 保存成功。今天的对照是离线 Dataset 回放。迁移后的远端副本仍须完成这次真实权重短测，再启动正式训练。

## 6. 正式训练

当前视觉专项尚未确定正式预算；先填写总步数、每卡 batch、GPU 数。下例的保存／验证间隔为运行示例，可在启动前按实验协议调整。`TRAIN_STEPS` 是总更新步数，不是 epoch。

```bash
export TRAIN_STEPS=填写本次总步数
export PER_GPU_BATCH=2
export TRAIN_GPUS=8
python - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ['ESKILL_RUN'])
c = json.loads((run/'smoke.json').read_text())
steps, batch, cards = (int(os.environ[k]) for k in ('TRAIN_STEPS','PER_GPU_BATCH','TRAIN_GPUS'))
assert min(steps, batch, cards) > 0
c.update(output_dir=str(run/'train'), num_steps=steps, batch_size=batch,
         save_every=1000, val_interval=1000, val_max_batches=32, log_every=20,
         run_name=f'pi05_current_bs{batch*cards}_libero40', wandb_info=None)
(run/'train.json').write_text(json.dumps(c, indent=2)+'\n')
PY
```

`batch_size` 是每卡值；当前无梯度累积，全局 batch＝每卡 batch×卡数。每卡 2 workers，八卡共 16 workers。学习率沿用 5e-5、warmup 1,000 步，此后恒定；增大 batch 前重新检查显存和吞吐。正式训练从原生 π0.5 新开，不继承刚才 8 步检查的权重或优化器。

正式训练默认读取仓库根目录本机 `wandb.json`，例如：

```json
{"entity":"你的账户或团队","project":"EskillVLA","mode":"online"}
```

使用 `wandb login` 登录，或通过服务器的密钥管理设置 `WANDB_API_KEY`；不要把密钥写进 Git 或报告。已有 `wandb.json` 可沿用，检查 entity／project 是否正确。

八卡示例，建议在 `tmux` 中执行并保留日志：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m scripts.train \
  --config "$ESKILL_RUN/train.json" 2>&1 | tee "$ESKILL_RUN/train.log"
```

用双卡就改成两张可用卡、`--nproc_per_node=2`，同时将配置生成时的 `TRAIN_GPUS` 改为 2，保证 run 名称正确。单卡用 `CUDA_VISIBLE_DEVICES=0 python -m scripts.train --config ...`。本机技术检查曾因通信环境禁用 P2P／IB；这些设置不应原样搬到 A800 等服务器，应先使用默认 NCCL 通信。

启动后确认 W&B 实际收到日志。主要观察 Current／Subject 各自 loss、总 loss、梯度范数、单步耗时和有效监督数量；验证关闭增强。验证 loss 不是闭环任务成功率。

## 7. 保存、恢复与验收

输出 `train/` 中包含实际配置、`metrics.jsonl`、`validation.jsonl`、最新 `checkpoint.pt` 和定期 `checkpoint_XXXXXX.pt`。检查磁盘空间：当前保存完整模型，即使 VLM 冻结也不是只保存几百万可训练参数；本机完整检查 checkpoint 约 13GB，多个里程碑会累积占用空间。

相同运行中断后，用相同卡数、相同数据／增强／batch／权重配置恢复：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m scripts.train \
  --config "$ESKILL_RUN/train.json" --resume "$ESKILL_RUN/train/checkpoint.pt" \
  2>&1 | tee -a "$ESKILL_RUN/train.log"
```

`num_steps` 为恢复后的总目标步数，必须大于 checkpoint 步数。同一运行恢复沿用 W&B run；新实验使用独立输出目录和新 run。更换当前架构、增强规则或刷新数据标记后新开运行，不能继续旧的目标文本生成／共享投影 checkpoint。更换服务器可改资源路径，但实际资源内容需一致。

完成训练后先检查独立验证集上的 Current／Subject 区域读出、反事实切换和移动目标定位，再讨论接回 Motion／Action。当前专项目录不自动启动 LIBERO／Plus 闭环。

相关入口：[Current／Subject 专项设计](https://github.com/lyk-nju/ESkill-VLA/blob/0be594de81a3b9187ea7fdbb8483b0d7bab986da/docs/design.md#current-grounding-专项入口待效果验证) · [训练接口](../src/openpi/training/README.md#current-grounding-专项训练) · [真实双卡启动记录](../experiments/current_grounding_fused/README.md) · [取消容器黑名单的核验](../data/libero/atomic-vla/check/test/copy_no_blacklist/README.md)。
