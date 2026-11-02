# Target Grounding：每臂三锚点视觉表征

本页维护空间接地的设计、理由与方法来源，服务于 [idea.md](idea.md)。整体架构与数据接口见 [design.md](design.md)。当前主方案是 **每臂三角色视觉 Token＋冻结 FLUX VAE＋共享投影头的潜变量加权 MSE**。双臂可以具有不同对象和事件终点，仍共用同一网络。

2026-11-02：已采用 AffordanceVLA 的 Which2Act 连续潜变量回归机制，替换此前的视觉去噪实现。固定 FLUX 权重、共享回归头和训练日志均已接入，完成真实 π0.5 LoRA 四卡 8 步训练及恢复到 10 步，见 [grounding_mse_v1](../data/libero/atomic-vla/check/test/grounding_mse_v1/README.md)。此前 [10 条轨迹的 VAE 编解码检查](../data/libero/atomic-vla/check/test/flux_vae_v1/README.md)保留为目标质量依据；短程训练不代表预测重建已经收敛，人工验收和闭环效果仍待验证。

## 1. 学习什么，为什么是三个锚点？

操作需要知道执行器当前在哪里、针对哪个对象，以及该对象在本事件结束时应形成什么视觉空间状态。只表示对象类别不够；只回归框又难以保留形状、接触部位和相似对象之间的细节。因此将三个角色都表示为视觉 latent，而非将主输出限定为边界框。

| 锚点 | 表达内容 | 监督来源 |
| --- | --- | --- |
| 当前执行器 \(\mathrm{TargetGrounding}_t^{r,ee}\) | 本臂手／夹爪的位置、形态与局部交互状态 | 当前执行器区域 |
| 当前对象 \(\mathrm{TargetGrounding}_t^{r,obj}\) | 本臂当前被操作对象的位置、外观与局部几何 | 当前对象区域 |
| 事件目标 \(\mathrm{TargetGrounding}_t^{r,goal}\) | 同一被操作对象在本臂 Event 结束时应达到的视觉状态 | 该对象在事件终点的视觉区域 |

**事件目标定义为 Event-boundary Object State Grounding：使用被操作对象在当前 Event 结束时的视觉区域，监督模型学习该对象应达到的目标状态。** 当前对象与事件目标绑定同一对象身份，分别使用当前时刻和本臂事件终点的观测。事件目标不等于完整未来视频，也不唯一决定执行器终点或接触姿态。

## 2. 为什么采用潜变量直接回归？

借鉴 AffordanceVLA 的 Which2Act：目标 RGB 经冻结 VAE 编码，模型预测的视觉 Token 经共享 MLP 投影、空间拼接后直接回归干净潜变量。视觉监督不再采样扩散噪声或运行去噪器。VAE 潜变量是监督目标，在线 Token 是供下游使用的表征，两者通过投影头联系，不直接逐元素比较。

该选择简化训练期监督模块，是否改善表征和控制仍需验证。推理保留锚点预测及其到 Motion、未来技能和 Action 的连接，不运行 VAE 或潜变量投影头；VAE 图像解码只用于离线核验。

局部监督保留被操作对象的外观、形态和原图位置。对象移动后外观可能几乎不变，紧致裁剪再居中会丢失位置变化；保留原图布局的对象画布可同时提供外观与位置监督。当前与终点图像使用一致的视角约定，相机运动时需要对齐。

## 3. 当前网络与监督

在线分支采用三组基础角色 Query，加左右臂嵌入，通过同一轻量 Query Decoder 预测。第一版固定每臂每角色 \(R_g=16\) 个 Query、宽度 \(d=256\)，采用两层 Decoder，每层包含组内查询自注意力、对上下文的交叉注意力及 FFN。角色 Query 分开学习、跨臂共享，不复制两套预测器。单臂有 48 个有效视觉 Token，双臂有 96 个；保持现有双臂槽位接口，单臂数据以硬件 presence mask 屏蔽不存在的臂，不能将有效数量理解为已裁减计算槽位。

令 \(C_t\) 为保留图像位置的当前／历史视觉语言特征，\(r\in\{L,R\}\) 为臂，\(a\in\{ee,obj,goal\}\) 为角色。每组 \(\mathrm{TargetGrounding}_t^{r,a}\) 含 \(R_g\) 个 \(d\) 维 Token，不限定为一个向量：

\[
\mathrm{TargetGrounding}_t^{r,ee}=P_\theta(Q^{ee}+e_r^{arm},C_t),\qquad
\mathrm{TargetGrounding}_t^{r,obj}=P_\theta(Q^{obj}+e_r^{arm},C_t),
\]

\[
\mathrm{TargetGrounding}_t^{r,goal}=P_\theta(Q^{goal}+e_r^{arm},[C_t,\mathrm{TargetGrounding}_t^{r,ee},\mathrm{TargetGrounding}_t^{r,obj},\mathrm{SkillEmbedding}_t^r]).
\]

每臂当前两锚点与技巧选择并行预测，事件目标随后读取本臂所选技能向量 \(\mathrm{SkillEmbedding}_t^r\)、当前锚点与全局上下文。双臂锚点共同输入运动网络、未来技能 Decoder 和 Action Expert，每次重新观察时更新。Token 不要求与 VAE 目标 latent 具有相同形状。Grounding 每角色 16 个 tokens 表示 4×4 个空间块，Motion 每臂 8 个 tokens 表示固定未来时刻；数量独立，通过交叉注意力交互，共用辅助宽度 256。

未来技能序列可以跨 Event，但这里的 obj／goal 始终绑定当前 Event 的同一对象。预测的终点状态参与后续技能安排，不扩展为未来各事件的完整世界状态或目标图像；后续对象与位置仍由任务、观察和动作生成共同处理。

**SpatialEnc 仅属于训练目标分支。** 所有臂／角色共用冻结 FLUX VAE 编码器，采用保留原图布局的局部 RGB 目标画布：

- 当前执行器：保留夹爪及必要邻域。
- 当前对象：保留操作对象及必要邻域。
- 事件目标：保留同一被操作对象在本臂 Event 终点的区域，按统一规则留边。

区域由对应执行器或被操作对象的 bbox／mask 确定。第一版使用 bbox，在源图坐标中四周各留 4 像素并裁到屏内，区域外填黑色，整幅画布统一缩放至 256×256；不将局部区域单独放大或居中。区域采用完整屏内范围（full_in_frame），RGB 保留真实遮挡内容，不生成被遮挡部分的虚构外观。相机变化时需建立当前与目标的空间对应。监督画布与 π0.5 的观察输入分开配置，后者保持 224×224。

令 \(o_i^r\) 为本臂当前 Event 的被操作对象，\(R_\tau^{obj*}(o_i^r)\) 为它在时刻 \(\tau\) 的标注区域，则当前对象与事件目标的区域分别为：

\[
R_t^{r,obj*}=R_t^{obj*}(o_i^r),\qquad
R_{t_e^r}^{r,goal*}=R_{t_e^r}^{obj*}(o_i^r).
\]

\[
y_t^{r,a*}=E_{VAE}(\operatorname{Canvas}(I_{t_{r,a}},R_{t_{r,a}}^{r,a*})),
\quad t_{r,ee}=t_{r,obj}=t,\quad t_{r,goal}=t_e^r,
\]

\(E_{VAE}\) 包含 RGB 归一化至 [-1,1]、后验取值和 \((z-\mathrm{shift})\times\mathrm{scale}\) 变换。第一版固定使用 `latent_dist.mode()`，使同一目标画布获得确定的监督；这是与 AffordanceVLA 当前代码使用 `sample()` 的有意区别。VAE 保持冻结、eval、no_grad，编码使用 FP32 并退出混合精度上下文；潜变量差值与 MSE 同样使用 FP32。scaling／shift 从固定版本的 VAE 配置读取，分别为 0.3611／0.1159；离线解码前执行逆变换 \(z=y/\mathrm{scale}+\mathrm{shift}\)。

预测侧采用共享的 LayerNorm(256) → Linear(256,512) → GELU → Linear(512,1024) 投影头 \(P_\psi\)，只读取对应角色的预测 Token。令 \(R_g=g^2\)，第一版 \(g=4\)，目标网格为 \([C_z,h_z,w_z]=[16,32,32]\)。每个 Token 输出 1024 个值，按固定行优先顺序对应一个 \([16,8,8]\) 潜变量空间块，再拼接成完整网格：

\[
\widehat y_t^{r,a}=\operatorname{Assemble}_{g\times g}
\left(P_\psi(\mathrm{TargetGrounding}_t^{r,a})\right),
\qquad \widehat y_t^{r,a},y_t^{r,a*}\in\mathbb R^{C_z\times h_z\times w_z}.
\]

臂与角色信息已进入在线 Query；所有臂／角色共用同一个投影头。目标 RGB、bbox、mask、VAE 潜变量及其加噪版本均不作为投影头输入。FLUX VAE 空间下采样 8 倍，256×256 画布得到 16×32×32 潜变量。16 个 Token 按 4×4 分块组织，但每块预测 8×8 个潜变量位置，因此最终空间网格仍为 32×32。

固定背景可能占据大部分网格，因此按目标区域及留边加权潜变量 MSE。将画布区域 mask 按相同空间变换、覆盖率下采样到 VAE 网格，得到 \(m_{b,r,a,x}\in[0,1]\)，\(x\) 为网格位置：

\[
W_{b,r,a,x}=w_{bg}+(w_{fg}-w_{bg})m_{b,r,a,x},\qquad w_{fg}>w_{bg}>0,
\]

\[
\ell_{b,r,a}=
\frac{\sum_x W_{b,r,a,x}\,\operatorname{Mean}_{c}
[(\widehat y_{x,c}-\operatorname{stopgrad}(y^*_{x,c}))^2]}
{\sum_x W_{b,r,a,x}+\varepsilon},
\]

\[
\mathcal L_{ground}=\sum_{a\in\{ee,obj,goal\}}w_a
\frac{\sum_{b,r}v_{b,r,a}\ell_{b,r,a}}{\max(1,\sum_{b,r}v_{b,r,a})}.
\]

\(c\) 为 latent 通道，\(v_{b,r,a}\) 为逐臂、逐角色标签有效标记，并排除实际不存在的臂。先按每个目标的空间权重总和归一化，再按角色平均全局有效批次／臂样本；DDP 各卡有效数量不同或某卡全无标签时仍使用同一全局分母。区域范围包含必要邻域；VAE 感受野跨越像素边界，网格 mask 是监督权重而非精确语义分割。背景保留非零权重，分别记录三角色 MSE 及前景／背景误差。权重调整监督重点，不保证锚点一定被用于控制；新 MSE 的损失尺度不能直接沿用旧噪声损失解释。

idle 或无对象 return 的当前 ee 仍可监督，obj／对象 goal 无定义时仅屏蔽对应损失；hold 保留对象及有效终态。协作终态的对齐规则见 [Event Organization](event_organization.md)；共享对象身份本身不意味着 goal 标签相同。无法核验终点对象身份或区域时，仅将对应 goal 标签标为无效。

区域 mask 与角色有效性仅用于监督，真实 ROI、对象 ID、未来图像和终点位置不进入在线预测分支。不根据真值有效性删除在线 Token；实际存在的 idle 臂也保留预测槽位。冻结 VAE 仍接收 RGB，不直接增加 mask 输入通道。潜变量 MSE、下游轨迹、未来技能分类和动作损失共同训练锚点预测器；共享潜变量投影头只接受视觉 MSE。推理关闭 VAE 目标分支与投影头，不在线生成图像。

Stage 1 就采用上述每臂三锚点共享网络，不添加 OBB 多候选头、接触方向分类头或协作头。固定背景画布、共享角色 Decoder 和三锚点监督属于本项目实现选择，尚待训练验证。

## 4. 第一版参数与核验

| 项目 | 第一版约定 |
| --- | --- |
| 监督路线 | 冻结 FLUX VAE＋共享投影头＋潜变量加权 MSE，已接入训练 |
| VAE 权重 | `John6666/flux1-dev-fp8-flux` 的 `vae` 子目录，仅加载 AutoencoderKL；固定 revision 见下文 |
| VAE 编码 | 冻结、FP32、`latent_dist.mode()`；scaling／shift 从权重配置读取 |
| Token | 每臂每角色 16 个、宽度 256；共享两层 Query Decoder |
| 画布 | 256×256，保留原图位置，bbox 在源图中四周各留 4 像素，屏内裁剪，区域外黑色 |
| 观察输入 | π0.5 仍使用 224×224；256×256 仅为视觉监督画布 |
| 潜变量 | 16 通道、32×32 网格；16 个 Token 对应 4×4 个空间块，每块 16×8×8 |
| 共享投影头 | LayerNorm(256) → Linear(256,512) → GELU → Linear(512,1024)，跨角色、跨臂共享 |
| 损失尺度 | FP32 加权 MSE，按角色平均全局有效标签；短程配置取前景 1、背景 0.1、三角色等权、总 grounding 权重 0.1，作为训练起点 |

权重固定为 [revision c30c11ac2ce22eb42df52f8e8659f4bba3324165](https://huggingface.co/John6666/flux1-dev-fp8-flux/commit/c30c11ac2ce22eb42df52f8e8659f4bba3324165) 中的 `vae/config.json` 与 `vae/diffusion_pytorch_model.safetensors`。权重文件 SHA256 为 `0c0c8ac49850d6bd174107d4001685f4f6c876768a1c84dde751f2aed69f46ce`，本地下载已核对一致；不跟随 main 浮动，也不加载 FLUX 图像生成主模型。本地目录为 `/data1/yuankai/robot-research/ckpt/vae/flux1-dev`，附 `source.json`。文件内部是原始 FLUX 参数名，需先按 Diffusers 转换规则映射，再严格加载；核验脚本已检查 244 个参数全部匹配，不能将原文件直接交给普通 `from_pretrained`。

256×256 的依据是当前 AtomicVLA 数据的原生主相机尺寸。按 40 条任务指令各取 3 条轨迹，共 120 条，未加留边的对象 bbox 短边中位数为 29 像素，第 10 百分位为 16 像素；后者在 8 倍下采样后约占 2 个网格位置，若先缩到 128 则约占 1 个。这支持先保留 256，但不能代替小对象重建核验。

核验按“目标画布 → FLUX VAE 编解码”和“预测 Token → 潜变量 → RGB”两步组织，分别检查目标信息保留和预测链路。按来源在 check/test/ 保存记录，视频放 check/test/video/。两步均已完成首轮接口检查，梯度、有效性、DDP 平均、保存恢复与无 VAE 策略回读均有记录；预测重建的学习质量仍需足够训练后评估。上述参数作为第一版实现基准，训练起步权重的后续调整依据实际损失和控制表现。

首批 VAE 编解码检查已完成：10 条轨迹、72 个事件起／中／末关键帧，216 组三角色目标均有效，视频回读通过，记录见 [flux_vae_v1](../data/libero/atomic-vla/check/test/flux_vae_v1/README.md)。它只验证冻结目标编解码；人工细节验收、预测头学习效果和全量逐帧覆盖不由本检查替代。

训练实现位于 [grounding.py](../src/openpi/models/grounding.py)：`load_vae` 统一转换并严格加载权重，`VisualSupervision.predict_latents` 共享投影并按行拼接，`encode_targets` 在冻结 FP32 下生成确定目标。`Pi0ESkill` 保持五项主损失接口；训练日志额外记录 `grounding_ee/obj/goal` 及各自 `_fg`／`_bg` 指标。每个角色按全局有效数量平均，包括空标签 rank，诊断指标不重复计入总损失。新回归头与旧去噪 checkpoint 结构不同，使用纯 SFT 权重初始化新实验。

## 5. 参考来源与验证重点

| 来源 | 借鉴的机制 | 本项目的扩展 |
| --- | --- | --- |
| [AffordanceVLA](ref_papers/target_grounding/affordancevla.md) | Which2Act 的冻结 FLUX VAE、视觉 Query 与潜变量直接 MSE | 扩展为当前执行器、当前对象与 Event 终点对象；共享预测器／投影头，保留原位画布并按区域加权 |
| [ReconVLA](ref_papers/target_grounding/reconvla.md) | 以操作区域重建监督视觉表征，是此前去噪方案的来源 | 保留局部视觉监督的动机，当前视觉损失已改为潜变量直接回归 |

更多机制保留在 [空间接地论文库](ref_papers/target_grounding/README.md)。三锚点、原位画布与未来局部目标的组合是本项目设计；AffordanceVLA 的 Which2Act 使用当前对象裁剪图，并另有位置热图监督，不能将其效果直接视为本方案的验证。

示范终点只提供一个已实现结果，同一观察对应多个合理终态时，确定性 MSE 可能产生折中的潜变量预测，不保证覆盖所有合法目标状态。Stage 1 的实施与评测标准统一见 [design.md](design.md) 第 6 节。
