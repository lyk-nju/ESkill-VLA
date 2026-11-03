# 关闭数值本体状态

新增 `model.use_arm_state`，默认 `true` 以回读已有 checkpoint。设为 `false` 时不创建 `state_projection`，Motion 和未来技能 Decoder 的本体状态 Token／mask 直接移除，臂身份编码与存在标志保留。训练变换不再提供 `arm_state`。LIBERO 在线服务不请求七维关节值，直接接收现有源八维状态；它不进入辅助状态分支。

[新试验配置](../../../../../../examples/libero/train_eskill_no_state.json) 同时将 `use_arm_state` 和 `discrete_state_input` 设为 `false`，关闭辅助状态与状态文本两条通路。原始状态、末端位置、动作和未来图像仍可用于训练监督；辅助标签不作为在线条件。输出目录和 W&B 名称带 `_no_state`，正式训练未启动。

相关测试 60 项通过、1 项跳过：

- 固定图像、文本、采样噪声和 flow time，缺失数值状态与将数值状态改为 NaN 时，动作、当前／未来技能、Grounding、Motion 预测及五项损失完全一致。
- 关闭状态后，Grounding、Motion、未来技能及动作输出层仍可反传。
- SFT 初始化、短程辅助训练、恢复和独立策略回读均覆盖开启／关闭，以及 full／LoRA 两类权重。
- 在线单／双相机配置均无需关节字段；原有开启状态与 SFT 协议检查继续通过。

另用实际 LIBERO 数据完成 tiny CPU 模型的 2 步训练，每步含固定子集验证，再保存／加载 checkpoint 并通过 `LiberoWirePolicy` 推理。训练输入不含 `arm_state`，八维源状态请求可生成有限的 `[10,7]` 动作块，无需七维关节值。此检查使用已有测试规模 SFT 初始化及随机小型 VAE，不能作为正式 π0.5 收敛或任务成功率证据；没有向线上 W&B 项目上传。

详细摘要见 [report.json](report.json)，本地训练记录在 [train/metrics.jsonl](train/metrics.jsonl) 和 [train/validation.jsonl](train/validation.jsonl)。本次仅验证服务观察适配与模型回读，未额外运行无状态 checkpoint 的真实环境 WebSocket 闭环。
