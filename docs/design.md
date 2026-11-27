# 视觉比较

VAE、BBox、各自的 GAC 对照，以及原生 Current KLAL。Subject 和分配损失的对照也在此分支。

模型输入为 RGB 和完整任务指令。主相机的 Current 取原生文本到主图 patch 的逐层注意力，通过学习的文本权重聚合。配置可加入同帧腕相机。八个未来点默认每五帧取一点。

VAE 与 BBox 共用相同的任务目标和交互主体标注。两类方法各自提供同图实体对比损失 GAC。视觉对照的配方包括 `train_current_grounding.json`、`train_vae_gac.json`、`train_bbox.json`、`train_bbox_gac.json` 和 `train_visual_primitive_e*.json`。多视角 Current KLAL 使用 `train_visual_primitive_multiview.json`。

运行 `python -m scripts.analysis.visual_primitive_eval --help` 查看固定清单、腕图来源和定位指标。
