# 在其他服务器准备数据并训练 Current／Subject Grounding

适用版本：`feat/visual-primitive-v3`，2026-11-18 的融合 Current／未融合 Subject，以及取消容器 Copy 黑名单后的在线增强。远端代码目录可以继续叫 `ESkill-VLA` 或 `eskillvla`，不需要创建三个工作目录。

当前入口是 `training_stage=grounding`：主相机＋完整任务指令，冻结 π0.5 VLM 与 FLUX VAE，训练 Current／Subject 读取模块，损失权重各 0.1。这份说明用于视觉专项训练；产出的 checkpoint 不是可直接做机器人闭环的完整动作策略。旧 `train_eskill_stage1.json`／`train_eskill_stage2.json` 属于另一条训练路线。

## 1. 更新代码

在远端现有仓库检查工作区，保存已有修改后切到对应分支：

```bash
cd /你的代码目录/ESkill-VLA
git status --short
git fetch origin
git switch feat/visual-primitive-v3
git pull --ff-only origin feat/visual-primitive-v3
git rev-parse HEAD
```

请更新到包含本文的提交或更新版本；旧提交 `0e2017b` 不包含后续全部修改。更新后应有 `scripts/preprocess/libero/interaction_locks.py`，其 CLI 支持 `--refresh-copy-eligibility`；增强配置包含 `copy_target_prob=0.8`、`target_swap_prob=0.2`。

后续命令都从远端仓库根目录执行。先填写远端数据盘根路径：

```bash
export ESKILL_ASSETS=/你的数据盘/robot-research
export ESKILL_RUN="$ESKILL_ASSETS/workspace/experiment/current_grounding_001"
export PYTHONPATH=src:.
export OMP_NUM_THREADS=2
mkdir -p "$ESKILL_RUN"
```

配置、启动日志放在 `ESKILL_RUN`，训练器输出放在其 `train/` 子目录；首次训练前不要在 `train/` 内放文件。

## 2. 同步完整数据与权重

推荐直接同步本机已补齐的数据，远端无需安装 LIBERO、SAPIEN，也无需重新下载 AtomicVLA 原始 HDF5。

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

PyTorch 需要与服务器驱动兼容的 CUDA 构建；以上依赖声明不是锁定环境。现有 CUDA 环境无需重建，安装前确认依赖解析不会意外替换其 PyTorch。数据核验／视频需要 `preprocess` 中的依赖，训练 Dataset 本身不依赖仿真。保存实际环境：

```bash
uv pip freeze > "$ESKILL_RUN/environment.txt"
git rev-parse HEAD > "$ESKILL_RUN/commit.txt"
git status --short > "$ESKILL_RUN/worktree.txt"
nvidia-smi > "$ESKILL_RUN/gpu.txt"
```

## 4. 已有 Grounding 数据的标记更新

若同步的是本机最新完整目录，本节已完成，可跳过。若是较早的 `libero40_grounding` 副本，训练前执行一次：

```bash
python -m scripts.preprocess.libero.interaction_locks \
  --index "$ESKILL_ASSETS/data/atomicdata/libero40_grounding/index.json" \
  --output "$ESKILL_RUN/eligibility_refresh" \
  --refresh-copy-eligibility
```

它从已有框补 `interaction_locked`，从可见／完整 mask 刷新 `copy_eligible`，取消篮子、碗、杯、锅等类别禁用；保留非交互锁定、至少 98% 可见及源／落点保护。原标记数组保留，Episode 元数据先备份。确认汇总覆盖全部 1,693 条；不要在训练运行期间修改数据。缺少机器人 `robot::single` mask 时应重新同步完整标注，不能靠关闭检查绕过。

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
    data_index=str(root/'data/atomicdata/libero40_grounding/index.json'),
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

这里 `smoke=False` 保证使用真实模型；`smoke_train` 只是目录名。双卡全局 batch 为 4。`backbone_dtype=float32`、BF16 autocast，冻结 VLM／VAE；当前增强为 Copy 0.5、可移动原目标优先 0.8、反事实 0.2、Patch 0.5，随后同步裁切／旋转和输入颜色扰动，水平翻转 0。

先检查输入和增强覆盖，输出目录首次应不存在：

```bash
python -m scripts.preprocess.libero.grounding_check \
  --config "$ESKILL_RUN/smoke.json" --output "$ESKILL_RUN/input_check" --workers 2
python -m scripts.preprocess.libero.grounding_coverage \
  --config "$ESKILL_RUN/smoke.json" --samples-per-task 200 \
  --output "$ESKILL_RUN/coverage"
```

查看 `input_check/` 的检查结果与 `all_tasks.mp4`；`coverage/by_task.csv` 分别记录文本替换、Copy 成功和最终监督目标移动，不能把 Copy 成功率当作目标移动率。覆盖统计只测 Dataset，未包含模型计算。

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

本机同架构曾在两张 24GB RTX 3090、每卡 batch 2 下完成真实权重 8 步检查，峰值已分配显存约 15.16 GiB／卡，仅作容量参考。确认 loss／梯度有限、验证完成、`smoke_train/checkpoint.pt` 保存成功。不要把随机小骨干检查当成这一步。

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

相关入口：[模型设计](design.md) · [训练接口](../src/openpi/training/README.md#current-grounding-专项训练) · [真实双卡启动记录](../experiments/current_grounding_fused/README.md) · [取消容器黑名单的核验](../data/libero/atomic-vla/check/test/copy_no_blacklist/README.md)。
