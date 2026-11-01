# 训练接口核验

本轮检查文件架构与数据到模型的计算链路。使用 **CPU 缩小模型和随机冻结 VAE**，不是正式预训练 π0.5 训练，也不代表数据人工验收通过。源文件与历史核验结果未改写。

| 检查 | 结果 |
| --- | --- |
| 来源 | 已保存 AtomicVLA：1,693 条，273,465 条动作记录 |
| 父轨迹拆分 | seed=42；训练 1,536 条，验证 157 条 |
| 统计范围 | 训练集 247,842 条动作，每条原始记录计一次 |
| 抽样 | seed=42，无放回抽 10 条；各取开始、首个事件边界前、末尾，共 30 个窗口 |
| 时间／事件 | 当前输入按动作配对的 RGB 行读取；未来技能按各动作时刻查 Event；Goal 属于当前 Event |
| 有效性 | 缺标签、末尾 padding、缺失硬件臂屏蔽对应损失；未补成 Idle |
| 小模型运行 | 两步优化，再从 checkpoint 恢复到第三步；五项损失与梯度有限 |
| 自动回归 | 沙箱内 132 通过、1 项因本地 IPC 权限跳过；另在沙箱外通过双 worker 测试，共 133 项不同测试通过 |
| 策略回读 | 独立加载，不创建 VAE／视觉去噪器；仅当前图像、状态和任务文本，输出有限的 [8,7] 动作块 |
| 人工验收 | pending，保留源状态与来源质量警示 |

详细样本、张量形状和每步损失在 [report.json](report.json)；[norm_stats.json](norm_stats.json) 保留训练轨迹清单与统计依据；原始运行参数在 [config.json](config.json)、[config_resume.json](config_resume.json)，实际运行资产和日志在 run/。inference.npz 为随机小模型的接口输出，不能用于控制或模型效果评价。

回归覆盖当前／未来技能梯度、硬类别选择的梯度边界、未来分类到 Grounding／Motion 的梯度、冻结模块、推理条件缓存、缺失臂、跨事件终点、动作配对、训练集统计、权重回读以及保存恢复。测试结果另存 tests.xml 和 tests_multiprocess.xml。0／2 worker 的测试均验证了跨 epoch 恢复后所有模型权重与连续训练逐项一致。

当前沙箱禁止 PyTorch 多进程传递张量所需的本地 socket；双 worker 测试在已批准的沙箱外 CPU 环境通过，禁用 GPU 与下载。常规测试在缺此权限时明确跳过这一项；真实数据 smoke 使用 num_workers=0。生产 loader 使用 spawn，采样与 worker 各用独立随机数生成器。

运行环境复用 RoboTwin 的 π0.5 Python，临时设置本项目 PYTHONPATH；没有修改该环境安装。完整 π0.5 尺寸使用 [正式配置示例](../../../../../../examples/libero/train.json)。开始正式训练前仍需提供并验证转换后的 PyTorch π0.5 权重和预训练 VAE，准备 CUDA 环境。RoboTwin 源数据统一接入与 ESkill 闭环服务也尚待完成。
