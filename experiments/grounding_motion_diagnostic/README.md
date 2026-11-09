# ESkill30k Grounding / Motion 诊断

问题：在已稳定抓持并开始运输的相同真实观测下，将 Basket 指令换成 Plate，Event-end object 视觉读出、8点 EEF 运动读出及原生控制输出是否变化。

checkpoint：`/data1/yuankai/robot-research/ckpt/eskill-vla/pi05_eskill_bs128_libero40/checkpoint_030000.pt`。沿用之前 LIBERO 新组合评测的原始策略、双相机输入、BF16、10步Flow Matching、10步预测/5步执行协议。本轮仅重放观测及预测，不训练、不执行反事实完整rollout、不合并改写历史成功率，无 W&B run。

范围：按初态编号选布丁→Plate失败2条、成功2条，Milk→Plate失败2条；全程解码实际推理观测。10个配对状态要求正确对象稳定抓持、相对首次稳定抓持已水平运输至少2cm、原策略当前技能为Place；布丁另取至少20个控制步后的首个合格状态。不强制技能，不基于解码图选择样本；重复状态不是独立回合。

实现位于 `libero-composition/libero_composition/eskill_readout.py`（旁路捕获与训练头解码）、`eskill_latent_diagnostic.py`（准备/推理）、`eskill_latent_render.py`（离线统一渲染）。不修改模型架构、正常策略或权重。

所有运行资产放在数据盘，入口：[结果与完整视频](../../../libero-composition/result/eskillvla/diagnostics/grounding_motion/README.md)。冻结协议、原始latent/动作、逐次动作和RNG一致性核验、VAE基线、10组配对图及指标均在此目录。训练VAE与checkpoint内冻结VAE张量另外逐个核验。

解释限制：Goal是当前事件终点的物体视觉状态；8点是归一化(u,v)，偏移为来源步+4..+32。RGB和2D点是辅助解码读出，不直接输入Action Expert。图像距离不能代表三维可达性，OSC控制值不能直接当世界位移；单凭这些结果不能确定A/B/C的因果归属。

## 成功终态图像 Goal 干预

用户随后指定使用已有成功案例最终图像作编码目标。已选择同一模型 Milk→Plate i000 的最终原始RGB（原生目标成立、仍与夹爪接触），按原训练 full-in-frame 物体bbox及4像素留边制作画布，使用冻结VAE编码。由于VAE编码与内部Goal token空间不同，通过冻结视觉头反求16×256临时Goal tokens：Adam600步、lr0.03、相对原latent正则0.001，保持各token均值与标准差。仅优化临时latent，无模型训练或权重更新。

在既有失败案例i002/i004的第80帧稳定抓持状态，以用户指定的两条短Place指令作对照；检查原样替换，执行前10步短程分支，并运行Basket原生、Plate原生、Plate持续注入三组各2条闭环。所有分支恢复完整历史动作前缀，闭环最多总控制帧320、每次执行5步。

结果：Basket目标2/2；原生Plate目标0/2；注入Plate Goal后Plate目标仍0/2，两条均最终满足Basket目标。Goal重建误差降低约95%，Motion变化仅0.370/0.187px，前5步真实EEF位置差仅0.024/0.040mm。两条干预分支共96次真实前向的Goal tokens都已核验确实替换；所有6条分支经过独立物理重放，推理主相机图像、EEF和终点谓词完全一致。仅是2状态/1供体的近似Oracle诊断，不是总体成功率或语义因果证明。

[完整结果、6条原视频及6条Goal/Motion视频](../../../libero-composition/result/eskillvla/diagnostics/goal_oracle/README.md)。实现为 `eskill_goal_intervention.py`、`eskill_oracle_prepare.py`、`eskill_oracle_model.py`、`eskill_oracle_run.py`、`eskill_oracle_report.py`。入口依次为 donor、model --serve、run short/closed/verify、report；使用之前的ESkill推理环境与LIBERO仿真环境分别执行。所有资产位于数据盘，正常推理逻辑和旧评测结果未修改。

## Plate 向右移动的空间对照

将Plate世界y坐标减0.35m，主相机256px图像中向右约114px；Basket不做位置干预，两个抓持状态的Basket在准备期间位移小于1mm。目标上方逆运动学检查通过，不保证路径无碰撞。首80个控制动作来自旧场景的固定前缀，在新场景开环重放，仅第80帧后是新观测下的策略推理；自由Plate在这段准备中最大位移小于1cm。

两条初态各测试原生Basket、原生Plate、Plate+Goal注入：Basket成功2/2，Plate两组均0/2且最终满足Basket谓词。在第80帧Plate原生预测的第8个Motion点，离Plate分别更远29.7/17.6px（224px尺度），离Basket分别更近53.6/56.9px。仅描述这一时刻，后续Motion按5步重算。原生Plate与Goal注入在前5步的真实EEF位置差为0.062/0.018mm。注入使用旧场景成功终帧按位移平移的合成目标画布，非新场景中的真实成功演示。

[新布局的全部视频与结果](../../../libero-composition/result/eskillvla/diagnostics/goal_oracle_plate_right/README.md)。独立保留原场景，不更改原评测。诊断是2个状态的小规模比较，控制步与训练来源帧的时间对应、合成目标的latent分布、场景位置变化等限制均须考虑。

## 抓持运输阶段联合注入 Goal 与 Motion

沿用右移Plate的i002/i004两个状态及上一轮相同Goal tokens。每次真实推理前，在独立进程中重放至完全相同的物理状态和主相机输入，用数值IK检查放置路点，再以原生OSC实际执行32步参考运动。参考控制器从两个初始抓持状态均完成了释放后稳定放置。它只提供监督点，不控制被评估的模型分支。

没有轨迹到Motion的编码器，因此冻结训练好的二维轨迹头，反求8×256临时tokens，拟合参考轨迹+4、8、…、32控制步的主相机投影；正则到当前自然tokens并保持逐token均值/标准差。Goal与Motion进入原有未来技能及Action条件路径，不强制技能、不更新模型权重。连续3步无抓持接触后停止Motion注入，Goal按原对照保持至结束；两个状态均在第140步停止Motion干预。

完整闭环结果：Goal-only对照0/2，Goal+Motion仍0/2，最终都满足Basket谓词。首个状态的Motion读出改变62.55/65.61px，拟合误差0.306/0.491px，但前5步EEF相对Goal-only只变化0.041/0.0034mm。24次有效Motion注入的平均点位拟合误差中位数为0.318px，最大单点误差小于0.91px；Motion变化中位数56.29px，动作RMS变化中位数0.000648。两个闭环共96次真实推理，原样hook与旧对照首个动作、RNG和实际注入均核验；5项相关测试通过。原视频与解码视频各2条，每条321帧，另保留2条参考控制器视频。

这说明本次反求能显著改变Motion二维读出，却没有有效改变控制或完成新目标；不能据此断言Action Expert完全不使用自然Motion。二维反求不唯一，Goal仍是近似Oracle，未来技能允许随Motion变化，来源帧与控制步的时间等价也未独立核实。开发过程的渲染上下文问题、松手后的无效参考及IK初值重试均单独保留，不混入最终两回合结果。

[联合干预报告与全部视频](../../../libero-composition/result/eskillvla/diagnostics/goal_motion_oracle_plate_right/README.md)。实现为 `eskill_motion_intervention.py`、`eskill_motion_reference.py`、`eskill_goal_motion_model.py`、`eskill_goal_motion_run.py`、`eskill_goal_motion_report.py`。无W&B run，服务已在完成后关闭，所有运行资产位于数据盘。

## 真实示范 Motion 条件移植：必要性与局部 Steering

用户最终选定黑碗→Plate／Stove 两个同场景训练任务，各取原生初态0/1，不增加Milk组。训练划分中两任务分别有44/43条父示范；按预设顺序各取3条父示范、每条3个闭合夹爪Place阶段观测，共18个donor。将这些真实观测输入同一ESkill30k checkpoint，获取自然预测的8×256 Motion tokens；不是把示范未来轨迹编码成真值latent，也不再反求二维轨迹。全部18个候选均被模型识别为Place。

新hook直接替换 `action_condition` 的左臂Motion槽位，保留受体Goal、EE/Object、视觉语言K/V和未来技能条件。实际替换后的18层Motion K/V被传入全部10个去噪步骤，其他K/V逐元素不变。微型真实模型测试验证原样hook动作/RNG一致、供体改变进入动作、上游预测保持不变；真实模型各分支首次原样调用和B原生重放也核验动作一致。模型权重冻结，不修改核心模型代码。

A：原生／同目标donor／异目标donor／阶段均值，每种条件在4个初态任务对上闭环，共16条。原生技能为Place且正确对象抓持、抬升2cm连续3步后介入；连续3步无抓持接触后永久停止。所有条件均Plate 2/2、Stove 2/2，合计4/4达到原生目标谓词。每组仅1/4在停止时无抓持接触，不能称为4/4释放后稳定放置。三个干预条件首次同噪声Action RMS变化中位数分别为0.000884、0.000859、0.000705。

B：4个原生运输状态，各执行原生／Plate donor／Stove donor三个固定观测、状态、噪声的5步分支，共12条，不计完整成功率。两种donor的二维Motion读出平均点差为57.46–65.34px（224px尺度），但5步真实EEF终点差仅0.00727–0.03048mm；原生5步EEF移动61.16–68.94mm。Plate_i000／i001的终点差为0.00783／0.02030mm，Stove_i000／i001为0.00727／0.03048mm。没有观察到有实际幅度的目标转向。

共272次预测、62次有效替换；有效Motion K/V相对L2变化3.52%–32.67%，中位数15.81%，均改变全部18层。原视频和实际条件解码视频各28条，视频逐帧解码验证通过；所有输入、模拟器状态、RNG、选中donor和条件入口核验均保存。首次初始化因LIBERO谓词大小写报错，未执行模型动作，修正后重跑；该技术记录单独保留，不作为行为失败。模型服务已关闭，无W&B run。

结论限于这4个原任务状态：直接Motion→Action支路对自然donor替换不敏感，比只改二维读出的反求实验提供了更直接的控制接口证据。未来技能支路被固定，原视觉/语言/Goal等仍可提供冗余目标信息；跨示范联合上下文也未保证分布内。因此不能据此断言整个Motion模块无用或已定位唯一训练原因。

[报告、56个视频和逐项审计](../../../libero-composition/result/eskillvla/diagnostics/motion_patch_ab/README.md)。代码为 `eskill_condition_patch.py`、`eskill_motion_ab.py`、`eskill_motion_ab_model.py`、`eskill_motion_ab_run.py`、`eskill_motion_ab_report.py`。协议与环境/source快照保存在数据盘输出目录；依次运行prepare、model服务、run、report，`report --summary-only`可复用视频刷新统计。
