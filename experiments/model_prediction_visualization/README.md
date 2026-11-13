# 模型预测与真实 rollout 同步可视化

问题：原面板逐个读取示范观察进行独立推理，不能显示真实执行时一个动作块内的注意力保持行为。当前服务可旁路记录实际请求，客户端保存每个控制帧，渲染时按实际请求 ID 对齐。模型、采样噪声与动作保持原生路径；不从视频补做观察 Probe。

本轮使用[固定核验设置](rollout.json)：用户指定的新版直接训练 30k checkpoint，LIBERO Object 的 Milk 任务，官方初态 0、seed 7、等待 10 步、执行前缀 5、FM 10、最多 280 控制步。仅执行 1 回合用于可视化验收，标记 technical，成功率为 null，不进行模型效果对比。推理 CPU FP32／8 线程、batch 1；GPU 0 仅供 EGL 渲染，不停止本机已有推理进程。

真实 rollout 已完成：134 个控制步、27 次实际推理，耗时 279.05 秒，环境原生完成判据为 true、无运行错误。该单回合仅验收接口和视频，technical 成功率为 null。模型服务 PID 1474911 已关闭；本轮不训练、不上传 W&B。

三份视频均为 135 帧、20 FPS，包含相同的最后观察。下方四类注意力在每个实际 5 步 chunk 内逐像素一致；27 次主相机推理输入与源控制观察一致，134 个实际执行动作与对应预测前缀一致。首请求同噪声原始动作对照最大误差为 0；71 项回归测试通过。录制保留原模型动作与采样 RNG，渲染阶段不重跑模型。

[人工验收入口](../../data/libero/atomic-vla/check/test/model_rollout/README.md)包含完整同步面板、原视频、Attention 视频和复核记录。大体积产物保存在 `/data1/yuankai/robot-research/workspace/reports/eskill_model_viewer/20261113_rollout_001/`，其中 `run.json`／`review.json` 保存命令、PID、源码快照、代码版本、环境与逐项核验结果；服务、客户端、渲染和测试日志均已保存。

三个输入接口职责分别为 `scripts/serve_policy.py --record-predictions` 记录实际前向、`examples/libero/main.py` 执行并记录控制帧、`scripts/prediction_viewer.py --rollout-record` 回读渲染。完整命令和图例见[测评文档](../../docs/evaluation.md#模型预测与-attention-面板)。
