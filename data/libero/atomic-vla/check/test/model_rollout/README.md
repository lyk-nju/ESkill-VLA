# 真实 rollout 同步面板核验

- [完整预测面板](../video/model_rollout/milk_rollout.mp4) · [首帧预览](../video/model_rollout/milk_rollout.jpg)
- [同步原视频](../video/model_rollout/milk_rollout_original.mp4) · [同步 Action Attention](../video/model_rollout/milk_rollout_attention.mp4)
- [复核记录与命令](artifacts/review.json) · [原始读出数组](artifacts/video/milk_rollout.npz)

使用用户指定的新版 30k checkpoint，真实执行 LIBERO Object 的 Milk 任务，官方初态 0、seed 7、FM 10。134 个控制步对应 27 次策略推理，生成 135 帧、20 FPS 的完整视频。上方原画面逐帧变化；下方输入与 Attention 一起保持，在控制帧 0、5、10…130 更新。页首 `CTRL / INFER / AGE` 区分当前帧与注意力来源，绿框表示新动作块。黄色轨迹是来源帧生成的预测计划。

这次动作实际在仿真中执行，不是示范观察的离线 Probe。没有未来 GT，右上相关栏位明确显示不可用。27 个请求输入、134 个执行动作与每个 chunk 内热图保持均已核对，首请求原始动作对照误差为 0；71 项回归测试通过。模型服务已关闭。本轮为单回合技术核验，成功率为 null，不据此比较模型效果。

协议、实现入口和结果说明见[可视化实验](../../../../../../experiments/model_prediction_visualization/README.md)。
