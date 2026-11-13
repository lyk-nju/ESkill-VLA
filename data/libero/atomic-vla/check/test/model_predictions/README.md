# 当前模型预测面板核验

- [预测面板视频](../video/model_predictions/eskill30k_milk_predictions.mp4) · [首帧预览](../video/model_predictions/eskill30k_milk_predictions.jpg)
- [Action Attention 视频](../video/model_predictions/eskill30k_milk_attention.mp4)
- [原始数组](artifacts/video/eskill30k_milk_predictions.npz) · [运行记录](artifacts/video/eskill30k_milk_predictions.json)
- [代码／命令与核验记录](artifacts/review.json)

使用用户指定的 `eskill_libero40_direct/checkpoint_030000.pt`，选择固定验证集中的 `libero_object_no_noops_s009_r002`，指令为 “pick up the milk and place it in the basket”。抽取源帧 0、40、80，FM 10 步、seed 0，以 1 FPS 展示。实际在 CPU 使用 FP32，不是训练默认 BF16 的闭环效果测评。

主图黄色为预测八点轨迹，青色为有效 GT。右上两行分别为 Current Object、Event-end Future Object，三列为 GT 画布、保存读出头的预测、GT 经匹配 VAE 的回环。右下三张图来自真实 VLM 前向：Target=basket、Object=milk、Instruction=整句指令；均为词 Query 看向当前主相机图像 Key 的注意力。每图颜色在相机内归一化，底部 `MAIN / ALL` 保留该相机的绝对权重。

动作推理仅接收当前 RGB 与语言，未来 GT 在推理后读取。46 项回归测试通过；真实权重首个观察与无采集器的同噪声原始推理逐元素一致，动作最大误差为 0。新增词注意力后，三帧的动作、轨迹、夹爪概率、技能概率和对象读出也与此前核验完全相同。预测动作没有在环境中执行，本检查不报告成功率。读出含义和重现命令见[测评文档](../../../../../../docs/evaluation.md#模型预测与-attention-面板)。大体积产物保存在数据盘，当前目录仅保留入口链接。
