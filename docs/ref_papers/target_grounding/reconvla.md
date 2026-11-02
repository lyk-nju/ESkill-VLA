# ReconVLA：局部视觉 latent 去噪

**用途：局部视觉监督的动机与旧去噪方案来源。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../target_grounding.md)，实现以 [design.md](../../design.md) 为准。

## 原工作机制

当前被操作 gaze 区域经 VAE 编码成视觉目标，reconstructive tokens 条件化 latent 扩散去噪，并与动作目标联合训练；区域标签由检测模型构造。

## 对本项目的参考价值

提供用操作区域重建目标训练视觉表征的动机，也是 ESkill-VLA 此前去噪实现的来源。2026-11-02 的当前设计已改为参考 AffordanceVLA 的潜变量直接 MSE；三角色、原位画布及 Event 终点目标是本项目的扩展，不能归为 ReconVLA 原方法。

## 采用边界

本地 ReconVLA 使用冻结 FLUX VAE，视觉隐藏状态条件化潜变量去噪器；其提供的 CALVIN 管线将当前目标裁剪图和腕部图拼接。ESkill-VLA 当前选择直接回归，并在推理时移除监督用 VAE／投影头，保留预测锚点。该实现边界以 design.md 为准，不从原文推导部署必须生成图像。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2508.10333v1)

既有阅读记录：v1 §3.2–3.3；ReconVLA §1 明确受 [ROSS v2 §4](https://arxiv.org/html/2410.09575v2) 的重建式视觉指令训练启发。2026-11-02 另核对本地 commit `047ccd9` 的 [重建监督](../../../../ReconVLA/reconvla/recon/model/recon_arch.py) 和 [CALVIN 目标图](../../../../ReconVLA/reconvla/scripts/helper/calvin_json.py)；本次未重新联网核验论文或复现实验。

来源关系：[MAE v3 §3](https://arxiv.org/html/2111.06377v3) 提供遮挡重建动机，[ROSS v2 §4](https://arxiv.org/html/2410.09575v2) 转为视觉 latent 去噪，ReconVLA 将监督聚焦到操作区域；概念继承不等于复制相同网络。

返回 [本方向索引](README.md)。
