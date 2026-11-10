# AffordanceVLA：角色 Query 与潜变量直接回归

**用途：当前方法参考。** 本卡解释来源，不增加实施要求；当前方法见 [方向文档](../../target_grounding.md)，实现以 [design.md](../../design.md) 为准。

## 原工作机制

不同 queries 分别学习对象 ROI 的 VAE latent、二维交互热图、三维 shape 与 layout，动作专家读取相关表征。原文使用多专家、多阶段训练，部署保留 affordance generation。

2026-11-02 本地代码核对：随仓库提供的训练配置启用 Which2Act 的 FLUX VAE 路径。目标 bbox 扩大 1.5 倍并取正方形，裁剪后缩放至 256×256；冻结 AutoencoderKL 编码出 16×32×32 潜变量。预测 Token 经 LayerNorm＋MLP 逐块投影、空间拼接后，以 MSE 回归该潜变量。标准配置为 64 个 Token，fast 布局为 16 个；后验使用 sample，并应用 shift／scaling。仓库另保留 VQ-VAE 离散码分类路径。Which2Act 的 RGB 目标不使用扩散去噪。

## 对本项目的参考价值

当前视觉监督的主要实现参考：冻结 FLUX VAE、角色 Query、共享潜变量投影头与直接 MSE。ESkill-VLA 预测 Current／Future Object 两组 tokens，保留原位画布，按区域加权并屏蔽无效标签。冻结 VAE、训练共享投影头；预测 tokens 仅通过 Motion 影响 Action。ESkill 接口与状态见 [当前设计](../../target_grounding.md)。

## 采用边界

原版 Which2Act 监督当前对象裁剪图，位置另有 Where2Act 热图监督；两组对象状态、原位画布与未来 Event 终态不是该代码已经验证的组合。本项目不据此增加热图、OBB、接触方向或完整多专家架构，也不直接照搬 64 Token、裁剪比例和损失权重。How2Act 主要监督对象几何，不等同于本项目的 Latent Motion。单向 attention mask 约束前向可见性，不自动阻断动作梯度。

## 原文与阅读范围

[论文来源](https://arxiv.org/html/2606.06155v2)

既有阅读记录：v2 §2、§3.1–3.3、§3.1.4 与附录 C/D/F/G/H；引用 ReconVLA 不等于已确认代码继承。本次新增结论来自本地仓库 commit `3bb77f1` 的 [Which2Act 实现](../../../../AffordanceVLA/src/models/which2act_decoder.py) 和 [训练配置](<../../../../AffordanceVLA/configs/stage3 Libero.yaml>)，未重新联网核验论文或复现实验。

返回 [本方向索引](README.md)。
