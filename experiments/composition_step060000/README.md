# π0.5 SFT 60k：新组合评测

指定权重：`/data1/yuankai/robot-research/ckpt/eskill-vla/pi05_bs128_libero40/checkpoint_060000.pt`。
文件元数据为`model_type=pi05`、step60000、全参数SFT；不是带辅助模块的ESkill模型。

复用libero-composition冻结的13个Recombined任务，每任务10个共享初态，共130回合。
沿用原任务指令、仿真初态及环境seed；等待10步，最多320控制步，10步Flow Matching，预测10步执行5步。
按checkpoint配置使用双相机、BF16 autocast及保存的归一化；没有离散状态提示或arm-state条件。
Recombined相对于原任务定义，不代表已审计为训练阶段严格未见。

每回合保存完整原视频、VLM representation、Action readout及逐层逐Head原始数据。
Attention从真实推理的Q/K旁路重算FP32 softmax；原生SDPA输出不变。每回合首个请求与相同RNG的无记录推理检查动作逐元素相等。
统一五栏渲染器保留Object、Target、Instruction、Action；固定层9–12、全部8头、最终去噪Q0:5。
Action视图保持实际ACT输入图像和热图至下一动作块，新块第一帧闪红框，无离线PROBE。

模型解释器复用`RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python`，通过独立PYTHONPATH加载ESkill-VLA本地PyTorch实现；仿真使用数据盘`libero-eval`环境。
不修改权重，不启动训练。本轮测评产物保存在数据盘，未创建训练W&B run。

入口：`libero_composition.pi05_composition`（冻结协议、仿真），`libero_composition.pi05_composition_server`（真实策略），`attention_video.local_pi05`（只读观察器）。

[运行协议与结果目录](../../../libero-composition/result/pi05_bs128_libero40_step060000/report/)

状态：130/130完成，异常0。原生目标成功54/130（41.5%），正确抓取代理130/130；76条失败中67条终态满足旧目标（88.2%）。同初态AtomicVLA历史原生目标成功37/130（28.5%）。

停止时仍抓持的成功为36条，AtomicVLA为17条；目标成立且无抓持接触分别18和20条。当前协议在目标成立即停止，没有后续释放稳定性测试，不能将原生目标计数提高等同于完整放置改善。

390个视频全部生成并校验；每回合首个真实请求的无记录／有记录动作与GPU RNG完全一致。两个只读捕获测试及五个统一渲染测试通过。全部推理服务已关闭，GPU已释放。

[完整结果、13组统计与全部视频](../../../libero-composition/result/pi05_bs128_libero40_step060000/report/README.md)
