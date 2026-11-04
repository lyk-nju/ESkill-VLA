# LIBERO-40 ESkill：s1 LoRA 准备方案核对

正式目标已改为八卡 A100 全参数训练，见 [gpu10 配方](gpu10.md)。本页保留本地性能检查所用设置，未启动为正式实验。

2026-11-04，依据当前配置、Dataset、transforms、模型 forward、优化器与测评客户端逐项核对。机器人状态方案已确认：仅直接帮助 Motion 和未来技能安排，配置 `use_arm_state=true`、`discrete_state_input=false`；Action 经预测 Motion／未来技能间接接收状态信息。其余两项待确认的设置见下表。唯一可执行配置为 [train.json](train.json)，训练／自动测评入口为 [run.py](run.py)。确认后应同步修改配置及本页，不依赖默认值猜测实验设置。

当前状态：此前 SFT20k 初始化的 run_001 已停止；从 π0.5 基座开始的 run_002 仅完成性能分析与归一化准备，尚未启动正式训练。性能分析产生的模型、优化器和随机状态不继承。

| 待确认项 | 当前配置及实际行为 | 可选变更的影响 |
| --- | --- | --- |
| 动作损失维度 | `mask_action_loss=false`；真实7维动作补到32维后，32维共同计算 MSE | `true` 仅按真实有效维平均，会改变 action 与辅助损失的相对尺度 |
| 数据划分 | 1,693条全量训练，`validation_fraction=0`，无独立 val | 固定1,518 train／175 val时，归一化必须重新按 train 计算；不能用全量统计 |

## 初始化、可训练参数与数值精度

| 项目 | 当前设置 |
| --- | --- |
| 训练起点 | π0.5 预训练基座；本项目微调 step 从0开始，不代表随机初始化整个模型 |
| 基座权重 | `/data1/yuankai/robot-research/ckpt/pi05/base/pytorch_bf16/model.safetensors` |
| tokenizer | 同一基座下 `big_vision/paligemma_tokenizer.model` |
| 原微调权重 | `init_checkpoint=null`，首次启动不传 `--resume` |
| 网络 | `model_type=eskill`，当前技能→三角色视觉→Motion→未来技能→动作的分支从第0步共同训练 |
| LoRA | rank=16、alpha=16，比例alpha/rank=1；无LoRA dropout；A随机、B为0 |
| LoRA位置 | VLM语言主干和Action Expert各层的q/k/v/o、gate/up/down七类Linear；原参数名及基座权重保持 |
| 原基座 | `apply_lora` 冻结所有原基座参数，包括视觉编码器、原语言／动作专家、动作输入／输出投影和时间MLP；两个Gemma流的LoRA可训练 |
| ESkill新增模块 | 技能／视觉／运动／未来技能Decoder、状态／上下文投影、技能表、臂／类型编码、auxiliary KV、step_projection、视觉回归头和trace_head均可训练 |
| `freeze_vlm=true` 的含义 | 不能解读为“整个VLM没有可训练参数”；其语言LoRA仍更新，视觉编码器不更新 |
| 参数量 | 相同结构已有四卡检查：可训练39,927,051、冻结3,437,253,555；正式启动需记录新模型实际计数 |
| 混合精度 | 基座BF16，训练BF16 autocast；新增参数及LoRA存储为FP32；VAE、视觉MSE及主要损失运算为FP32 |
| 梯度重算 | 开启；同batch关闭时24GB单卡OOM，保持现设置 |
| 新分支初始化 | step_projection权重／偏置为0；Motion FiLM初始scale=1、shift=0；auxiliary KV保留随机初始化 |
| 技能文本原型 | 初始化后使用基座文本编码生成一次并冻结，9类描述见pi0_config.py；不会每步随语言LoRA重新编码 |

## 在线输入和机器人状态的准确路径

| 内容 | 当前实现 |
| --- | --- |
| 图像 | 主相机`image`＋腕相机`wrist_image`，顺序固定；第一相机承载Grounding／Motion标注 |
| 时间窗口 | 仅当前配对观察；两相机是同一时刻的两个视角，不是历史两帧；当前没有历史帧堆叠 |
| 分辨率 | 原图256×256，VLM输入224×224；resize/pad，RGB归一化至[-1,1] |
| 增强 | `augment_images=false`；ESkill当前不支持几何标签同步增强，不偷偷启用基线SFT的增强 |
| 文本 | Episode任务指令；清理空白／下划线，BOS与换行；长度上限200 tokens，超长截断 |
| padding优化 | collate仅裁掉batch共同无效的右侧文本padding，不改变真实token顺序、mask和有效位置；已做五项损失与梯度等价检查 |
| 原始状态 | 15维：关节角7＋末端xyz3＋末端axis-angle3＋夹爪双指位置2；源物理数值不重排 |
| `state` | 适配器提取末8维，量化统计归一化后补到32；只有`discrete_state_input=true`时才离散化并拼入任务文本 |
| `arm_state` | 完整15维按字段统计归一化，放入双臂槽位的第0槽，线性投影为256维body Token；缺失第1槽为0并被mask |
| 状态→当前技能／视觉 | 当前配置下没有直接路径；两者只从图像／任务上下文预测 |
| 状态→Motion／未来技能 | body Token直接加入这两个分支的cross-attention memory |
| 状态→Action | 当前通过预测的Motion／未来技能间接影响Action；没有另加连续state输入投影给Action Expert，也没有直接把body Token加入auxiliary KV |
| 小模型因果核验 | 只变arm_state时Motion／future logits改变，current skill／grounding不变；只变未进文本的state时全部预测不变 |
| 推理状态 | ESkill＋use_arm_state开启时，客户端发送七维joint_pos以及源八维状态，服务拼成同样15维；不能只给八维而声称状态协议相同 |
| 臂槽位 | LIBERO为single→第0槽，presence=[true,false]；保留双臂共享网络与第1槽mask；本轮不是双臂数据训练 |

当前设计文档中“完整状态仍进入动作专家”和“当前／历史输入”等表述描述了更一般的设计范围，不应据此声称当前recipe已有独立Action状态通路或历史输入。需要新增通路时应作为明确的方法变更实现。

## 数据协议、标签与采样

| 内容 | 当前实现 |
| --- | --- |
| 正式数据 | `/data1/yuankai/robot-research/data/atomicdata/libero40/index.json`，40任务、1,693条、273,465帧；π0.5／ESkill共用自包含Episode |
| 文件 | episode.json＋arrays.npz＋原编码图像images.npz；运行时不解析source证据路径；图像按需映射读取 |
| 导出字段 | 每臂event_index和effector_pixels[arm][camera]离线保存；训练只查表、切窗口、缩放和抽点 |
| 来源步 | 时间轴为source_steps，物理fps／秒数未知；渲染10fps不是控制频率，不把4来源步解释成固定秒数 |
| 采样 | frame_stride=1，每条真实动作记录为一个样本，随机采样、四卡DistributedSampler；按帧采样会让长轨迹权重更大，不做任务均衡或类别重采样 |
| 当前标签 | 依据配对主相机的观察时刻查Event；未来技能依据每条动作的真实来源索引查Event，不能混用两者 |
| Event | 半开区间[start,stop)，同臂不重叠；当前AtomicVLA goal=stop−1；读取当前Event自身的对象在goal帧的区域 |
| 技能 | 词表9类：Pick/Place/Open/Close/Turn/Push/Hold/Idle/Return；共3,873 Event，边界／技能valid均为true |
| 当前真实标签分布 | Pick1801、Place1800、Open79、Close69、Turn91、Push33；Hold/Idle/Return没有Event正标签；不把缺失段自动补Idle |
| Grounding角色 | 当前夹爪、当前操作对象、同一对象在当前Event终点的区域；仅主相机标注 |
| 框约定 | full_in_frame：忽略外部遮挡，裁掉屏外区域；xyxy右／下边界不含；完全出画的框无效，仅mask掉相应视觉监督 |
| 目标画布 | bbox向四周扩4源像素并裁屏；保留原图位置的矩形RGB区域，外部置0，再resize/pad至256；不是按实例mask精确抠图；原RGB遮挡内容保留 |
| Motion | 每臂8点，来源偏移[4,8,12,16,20,24,28,32]；主相机像素中心缩放归一化到模型画布，不是当前Event剩余路径或弧长采样 |
| Motion边界 | 允许跨Event；内部缺帧不插值；末尾hold_last只有真实末帧记录有效且在屏内时才有效，区分observed／padded |
| 动作窗口 | H=10条来源动作，7个真实控制量；Episode末尾repeat_last且参与动作损失；每步未来技能只监督真实动作部分 |
| 标签泄漏边界 | 真值技能／Event／bbox／Goal RGB／未来坐标都只进入targets或metadata；在线条件由模型预测，训练不teacher-force真值技能 |
| 元数据缺项 | 当前1,693条动作的representation／coordinate_frame／rotation_convention／gripper_convention仍为null，state各通道unit也未填齐；实际训练保留来源数值、测评执行LIBERO OSC控制命令，不把未填字段视为已确认的跨来源动作定义 |

新协议收尾需将已核实的来源语义写入导出定义和Episode元数据，并重建数据摘要／资源标识。具体含义以源数据构建器和仿真控制器为依据，不凭OSC名称推断动作数值就是米／弧度。该工作不能靠增加逐样本运行时校验代替。

## 监督、动作生成和优化

| 内容 | 当前实现 |
| --- | --- |
| 当前技能 | 余弦分类温度0.07，与冻结文本原型对齐；argmax查可训练技能表 |
| 未来技能 | 每臂H=10个查询；手臂编码＋动作步序号正弦位置编码；没有已知物理时间的秒编码；硬argmax查同一技能表 |
| 梯度边界 | hard argmax不反传；action可以更新被选中的技能向量表，分类头由分类监督及其他可微路径训练 |
| Grounding网络 | d=256，两层Query Decoder、8头；每角色16 tokens（三角色／臂），共享当前／Goal角色Decoder；单臂48个有效视觉tokens |
| Grounding目标 | 冻结FLUX VAE，FP32/eval/no_grad，latent_dist.mode；scaling=0.3611、shift=0.1159；256画布→16×32×32 latent |
| 视觉损失 | latent空间前景1／背景0.1加权MSE；每图按权重归一化，每角色再按全局有效数量平均，ee/obj/goal三项相加 |
| Motion网络及损失 | 两层QueryLayer、技能FiLM；共享二维头；逐点SmoothL1，beta=0.05，先均值二维坐标，再按全局有效点平均 |
| 当前／未来技能损失 | 两项cross-entropy分别按全局有效标签平均；不存在的臂、无效Event、真实动作外的padding不参与 |
| 总损失 | L=action＋0.1×current_skill＋0.1×future_skill＋0.1×grounding＋0.1×trace；W&B skill_loss合并两项原始分类损失 |
| Flow Matching | x_t=t×noise+(1−t)×action，目标velocity=noise−action；t=Beta(1.5,1)×0.999+0.001；训练每样本采一次noise/t |
| 动作维度padding | 补齐到32维时前7维为归一化动作，其余维度目标动作为0；当前full MSE下它们也预测noise−0并参与loss，不能称为“25维完全忽略” |
| AdamW | lr=5e−5，betas=(0.9,0.95)，eps=1e−8（PyTorch默认），weight_decay=1e−10；单参数组，无分支单独学习率 |
| 调度 | 前1000步线性warmup，首步约4.995e−8；之后恒定5e−5，没有cosine decay |
| batch／更新 | 4GPU×每卡8×累积1=全局32，每batch一次optimizer.step；40k约处理128万次样本，约4.68个全量数据遍历，仅为近似 |
| 稳定性 | clip_grad_norm=1，检查总loss和梯度有限；BF16无GradScaler；没有EMA、梯度累积或额外课程学习 |
| 分布式 | DDP/NCCL、broadcast_buffers=false、find_unused_parameters=true；缺失标签损失按全局有效数归一化 |
| worker／seed | 每rank 2 workers、spawn、persistent_workers、pin_memory；seed42，初始化跨卡一致，训练noise按42+rank，采样每epoch固定规则 |

## 保存、日志与闭环测评

| 内容 | 当前实现 |
| --- | --- |
| 总步数 | 0→40,000，本轮step与W&B横轴一致；不会加旧SFT步数 |
| 保存 | 首步保存，之后每1k及阶段末保存；LoRA＋新增分支参数／buffer、optimizer、各rank RNG、配置、norm_stats、资源标识一起保存 |
| 自动测评 | 10k／20k／30k／40k暂停训练，四套件并发测评后恢复同一checkpoint／optimizer／RNG和W&B run |
| W&B | 读取根目录wandb.json，rank0上传，名称pi05_eskill_lora_bs32_libero40_base；每20步＋阶段首末记录，详细诊断本地逐步保存 |
| 云端分区 | train、val、aux_train、aux_val、chart；无val数据时不生成val曲线，不上传类别频率等详细诊断 |
| 闭环次数 | Spatial/Object/Goal/LIBERO-10各10任务×5官方初态=50回合，总计200；这是本轮约定的小规模协议，不是每任务50回合的大规模协议 |
| 推理 | seed7，10次Flow Matching迭代；每次预测H10，执行前5步后重新观察；等待10步；不运行VAE／trace监督头，不输入未来真值 |
| 控制上限 | Spatial220、Object280、Goal300、LIBERO-10 520；保存全部视频，按官方成功信号统计，报告错误和分母 |
| 图像／状态对齐 | 评测相机RGB旋转180°到来源方向，再按相同224变换；输入相机顺序取checkpoint，状态拼接取同一适配器 |
| 用时 | 真实四卡短测padding裁剪及复制优化后约1.80秒/步，40k纯训练约20小时；不含四次测评、重载及共享机器波动，见performance.md |
| 输出 | 数据盘experiments/eskill_libero40/run_002，正式启动前记录代码／环境／资源摘要、配置、命令、PID；当前只有准备资产，无正式checkpoint |

证据入口：[状态适配](../../src/openpi/policies/libero_policy.py)、[训练变换](../../src/openpi/transforms.py)、[ESkill forward](../../src/openpi/models/pi0_eskill.py)、[LoRA冻结范围](../../src/openpi/models/lora.py)、[损失和优化器](../../scripts/train.py)、[性能检查](performance.md)。本文待确认项确定后才把当前配置当作冻结的正式recipe。
