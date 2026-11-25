# 实验入口

当前模型使用 `model_type=pi05` 或 `eskill`。新运行显式指定配置；入口不自动选择历史 recipe，也不恢复旧实验。

| 工作 | 入口 |
| --- | --- |
| Visual Primitive 多层注意力 | [E0–E3 实验与评测](visual_primitive/README.md) |
| Motion Planner | [冻结 KLAL、联合训练及同帧干预](motion_planner/README.md) |
| Current／Subject 融合特征双卡启动检查 | [current_grounding_fused](current_grounding_fused/README.md)：真实 π₀.₅／FLUX VAE，GPU 0、3，BF16 DDP 8 步及 A／C 计算开销 |
| Event／夹爪统计 | `python -m scripts.preprocess.libero.audit_training`，见 [数据检查](../data/libero/atomic-vla/check/test/training/README.md) |
| 真实 rollout 预测与注意力可视化 | [model_prediction_visualization](model_prediction_visualization/README.md)，实际请求采集与控制帧对齐，默认 5 步更新 |
| 单次训练／恢复 | [scripts/train.py](../scripts/train.py)，`--config` 必填，恢复另传 `--resume` |
| LIBERO 分段训练＋闭环测评 | [scripts/run_libero.py](../scripts/run_libero.py)，`--config` 必填，默认每 10k 测评，最后不足 10k 的预算也会执行 |
| 两阶段镜像配方 | [eskill_libero40_mirror](eskill_libero40_mirror/README.md)，官方 π0.5 → 先验 20k → 动作 40k，不带复用，远端训练 |
| 新版直接联合训练对照 | [eskill_libero40_direct](eskill_libero40_direct/README.md)，官方 π0.5 → 联合40k；关闭增强／视觉遮蔽／技能重采样，尽量对齐旧版训练配方 |
| 单阶段增强对照 | 同目录的 [color.json](eskill_libero40_direct/color.json)／[color_crop.json](eskill_libero40_direct/color_crop.json)：仅颜色、颜色＋主图同步轻裁切，保留A0无增强对照 |
| 同步旋转对照 | [color_crop_rotation.json](eskill_libero40_direct/color_crop_rotation.json)：在A2上只增加±5°主图中心旋转，同步对象与轨迹标签 |
| ESkill 联合训练接口示例 | [train_eskill.json](../examples/libero/train_eskill.json)，32 步接口模板，需填写同划分 SFT 与归一化 |
| π0.5 配置起点 | [train_pi05_sft_val.json](../examples/libero/train_pi05_sft_val.json)，固定 train／val 的 LoRA 示例 |

两份示例的冻结范围、LoRA 与精度设置并不相同，不能直接组成 SFT→ESkill 两阶段配方；转接时必须按 [训练接口](../src/openpi/training/README.md) 匹配。新的两阶段预算与初始化已确定，见上方镜像配方；未在本机启动正式训练。

从项目根目录运行，设置 `PYTHONPATH=src:.`。纯训练使用：

```bash
python scripts/train.py --config /path/to/current_recipe.json
```

只有需要周期闭环测评时才使用编排入口；沿用四套件各 50 回合、seed=7、FM10、执行前缀5的协议，至少四张 GPU：

```bash
python scripts/run_libero.py --config /path/to/current_recipe.json \
  --gpus 0 1 2 3 \
  --sim-python /data1/yuankai/robot-research/workspace/envs/libero-eval/bin/python
```

`--eval-every` 可明确指定测评间隔；`--resume-step` 只用于已经完成的测评节点，先补测再续训。脚本在启动子进程前校验当前配置 schema。示例和历史配置中的预算不自动成为新实验预算。

## 历史记录

[eskill_libero40](eskill_libero40/README.md)、[pi05_libero40](pi05_libero40/README.md) 保留原始配置和结果，旧字段不做兼容转换。重构前的编排和性能对照代码保留在 [Git 历史](https://github.com/lyk-nju/ESkill-VLA/tree/b2318c2/experiments/eskill_libero40)。旧 checkpoint 的实验含义不变；复现需使用对应代码版本。
