# FLUX VAE 三角色重建核验

2026-11-02：已完成 **10 条轨迹、72 个关键帧、216 组三角色目标**的冻结 VAE 编解码检查，人工验收 pending。[视频索引](index.html) · [视频目录](../video/flux_vae_v1/) · [完整记录](report.json) · [样本清单](manifest.json)。

样本有意覆盖 LIBERO spatial／object／goal／10 四套任务，以及 Pick、Place、Open、Close、Push、Turn 六种技能；每个任务选 episode_id 排序第一条。每个 Event 取起点、中点、末帧，按源帧顺序播放，显示速度为 1 fps。它们是关键帧对照短片，不是全轨迹逐帧重建或全数据集质量统计。

每个视频三行分别为当前夹爪、当前对象、本 Event 终点的同一对象；四列为 **原始 RGB＋bbox／目标画布／VAE 重建／四倍绝对 RGB 误差**。画布保留完整屏内 bbox 及源图 4 像素留边，区域外为黑色，尺寸为 256×256。蓝／绿／黄框分别对应三行的角色。

| 目标 | 有效数 | 区域 RGB 平均绝对误差（0–255） |
| --- | --- | --- |
| 当前夹爪 | 72 / 72 | 4.75 |
| 当前对象 | 72 / 72 | 5.41 |
| Event 终点对象 | 72 / 72 | 6.21 |

误差在 bbox＋留边区域内按像素／颜色通道平均，再对抽样目标等权平均；同一 Event 的 goal 在不同当前帧重复计入。共编码 168 个不同的角色目标，重复 goal 复用结果。三角色目标均得到有限的 `[16,32,32]` 潜变量和 RGB 重建；10 个视频已完整回读，帧数、尺寸、时间顺序及 SHA256 均核对通过。CPU 四线程总耗时约 5 分 17 秒，没有使用训练显卡。

初步查看，位置和主要轮廓可以保留，细纹理存在损失。建议重点看 **03（小块 butter 入篮）和 07（灶台旋钮）**。03 的终点区域主要显示遮挡它的篮筐纹理，该目标区域误差为 18.85／255；完整屏内 bbox 保留位置，但 RGB 不会补出被遮挡物体的外观。本轮没有预测 Token、回归头或模型训练，不能据此判断 16 个 Query 的学习效果或控制收益。

## 权重与加载

- 来源为 `John6666/flux1-dev-fp8-flux/vae`，revision `c30c11ac2ce22eb42df52f8e8659f4bba3324165`；本地权重在 `/data1/yuankai/robot-research/ckpt/vae/flux1-dev`，来源与文件哈希保存在该目录 `source.json`，并写入本次报告。
- 权重 SHA256：`0c0c8ac49850d6bd174107d4001685f4f6c876768a1c84dde751f2aed69f46ce`。该文件采用原始 FLUX 参数名，不能直接按普通 Diffusers `from_pretrained` 权重加载；使用 Diffusers 0.33.1 的 `convert_ldm_vae_checkpoint` 映射后，244 个参数严格匹配。
- 编码与解码在 FP32、冻结、eval、no_grad 下进行；后验取 `mode()`，应用 `(z−0.1159)×0.3611`，解码前做逆变换。RGB 输入为 `[-1,1]`，输出裁到该范围后转换至 `[0,1]` 计算上述误差。视频和 PNG 另外量化到 uint8。
- 现有训练代码仍使用旧视觉去噪路径，本检查只验证新 VAE 目标编解码；共享 MLP 回归头与训练配置接入尚待完成。

## 复现

实现集中在 [check_vae.py](../../../../../../scripts/preprocess/visual_grounding/check_vae.py)，复用 `EpisodeReader` 与训练侧 `role_canvas`。在项目根目录、现有 π0.5 Python 环境中运行；新的核验批次使用新的输出目录，保留本次记录：

```bash
PYTHONPATH=src python -m scripts.preprocess.visual_grounding.check_vae \
  --manifest data/libero/atomic-vla/check/test/flux_vae_v1/manifest.json \
  --vae-path /data1/yuankai/robot-research/ckpt/vae/flux1-dev \
  --output data/libero/atomic-vla/check/test/flux_vae_v2 \
  --video-dir data/libero/atomic-vla/check/test/video/flux_vae_v2 \
  --threads 4
```

参数基准见 [Target Grounding](../../../../../../docs/target_grounding.md)。
