# π0.5 ESkill 30k：新组合评测

权重：`/data1/yuankai/robot-research/ckpt/eskill-vla/pi05_eskill_bs128_libero40/checkpoint_030000.pt`。
SHA256：`941458d4bbd0a527e0972a2ab95aa4838311375b07742296b488debbb1bb2600`。

沿用上一轮π0.5 SFT60k的13个新组合、每组10个共享初态，共130回合；同任务指令、初态指纹、环境及策略seed，等待10步、最多320控制步、10步Flow、预测10步执行5步。
成功判定沿用LIBERO原生目标，单列停止时仍抓持／无抓持接触；没有额外延长回合验证释放稳定性。

checkpoint为完整ESkill，辅助训练30000步，配置记录从SFT30000步初始化；保留模型预测的技能、Grounding、Motion上下文，不能解读为单个模块的受控消融。
双相机，BF16，checkpoint归一化，无数值arm state或discrete state提示。数据及划分资源指纹与上一轮相同；新组合相对任务定义，不宣称严格训练未见。
这是离线闭环评测，不启动训练或W&B；不修改模型权重、核心推理或attention。

每回合原视频、VLM representation、Action readout。共享五栏渲染器，语言Object/Target/Instruction与动作Query；Target不是ESkill辅助Goal Query。
主相机Attention来自原生前向Q/K的只读FP32重算，保留18层8头；动作最终去噪。动作对辅助Key的Attention mass另存，避免将视觉热图误认成全部上下文。
每回合第一推理比较无观察器／有观察器在相同RNG下动作和技能输出完全一致。

入口复用`libero_composition.pi05_composition{,_server,_report}`，追加`--variant eskill`。
GPU0–3各一个独立策略服务；使用现有PyTorch及libero-eval环境，运行命令、PID、源码和配置在数据盘report保存。

[协议、运行日志与结果](../../../libero-composition/result/pi05_eskill_bs128_libero40_step030000/report/)

状态：130/130完成，异常0、未完成0。目标谓词成功48/130（36.9%），正确抓取代理130/130；82条失败中62条满足旧目标（75.6%）。

同初态π0.5 SFT60k为54/130（41.5%）；本模型新增成功11条、丢失17条，净少6条。AtomicVLA同130回合为37/130。
ESkill成功中28条停止时仍抓持、20条无抓持接触；SFT60k对应36和18条。本协议没有后续稳定释放验证，不能将这些计数直接称为完整放置成功率。

全部390个视频通过解码及动作时间对齐校验，130次首推理动作、技能预测、GPU RNG一致性检查通过；捕获测试3项、视频相关测试9项通过。四个策略服务已关闭，GPU已释放。

[完整报告、各组结果与视频入口](../../../libero-composition/result/pi05_eskill_bs128_libero40_step030000/report/README.md)
