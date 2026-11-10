# ESkill 数据接口核验

2026-11-10。入口为现有 `atomicdata/libero40/index.json`，原 Episode 数据未改写；本目录只保存数值检查与人工核验入口。

| 检查 | 结果 |
| --- | --- |
| 轨迹／动作行 | 1,693／273,465 |
| H=10 动作窗口跨 Event | 19,620（7.17%） |
| 实际执行前缀 5 步跨 Event | 8,720（3.19%） |
| 已保存物理时间轴／夹爪命令语义 | 尚未核验补齐，不能生成已核验的秒级监督 |
| 十条真实轨迹的当前 EEF 投影 | 原始 state 的 EEF xyz 经已保存相机投影，与 effector_pixels 逐有效帧完全一致；最大误差 0 源像素 |

全量结果见 [data_audit.json](data_audit.json)，十条投影结果见 [projection_audit.json](projection_audit.json)。时间覆盖率为 `null` 表示未知，不是零覆盖率。技术检查不替代视频人工验收。

## 运行与时间接口

```bash
python -m scripts.preprocess.libero.audit_training \
  --index /data1/yuankai/robot-research/data/atomicdata/libero40/index.json \
  --output data/libero/atomic-vla/check/test/training/full_audit.json
```

可加 `--split data/libero/atomic-vla/split.json --subset train` 或 `val` 分开统计。工具不解码 RGB，不启动模型或仿真。

ESkill Dataset 接受 Episode 已保存的 `timestamps_s` 数组引用；也可通过训练配置 `timing_path` 提供独立清单：

```json
{
  "defaults": {
    "gripper": {
      "channels": {"single": 6},
      "close_sign": 1,
      "alignment": "observation_then_action",
      "evidence": "填写已核验的夹爪符号与先观察后执行的来源依据"
    }
  },
  "episodes": {
    "示例episode_id": {
      "control_frequency_hz": 20,
      "source_step_indices": [0, 1, 2, 4],
      "evidence": "填写此轨迹真实控制时钟及保留帧映射依据"
    }
  }
}
```

上例只是四帧接口示例，频率和映射不代表本数据已核验。每条记录也可直接提供 `timestamps_s` 秒数组。`source_step_indices` 必须逐帧对应，不能默认等于过滤后的行号。清单供 Dataset 使用，不修改原数组。

第一版精确匹配 +0.25s…+2s（容差 1e-6s），不把最近一帧当作缺失时刻；路径与开合使用相同查询，各有独立有效性。终点重复不参与主监督。未知／未支持技能保留编号并排除 Skill CE，其他有效监督保留。

## 时钟证据边界

现有 `timeline=source_steps`、空时间戳和空 `step_duration_s` 无法单独建立秒级监督。场景缓存的 `source_to_official_indices` 是动作身份对应，不足以定义再生成视频的物理时间。

本地 OpenVLA_OFT 的 `experiments/robot/libero/regenerate_libero_dataset.py` 在执行 `env.step` 之前跳过 no-op，并在保留动作执行前记录观察；其环境 helper 使用 LIBERO 默认控制频率。这表明原始示范的跳号可能已被重放压缩。尚需核实发布 RLDS 的固定生成／转换版本和是否再次过滤，不能仅据这份邻近实现将全部数据标成连续 20Hz。

夹爪输入为当前真实双指位置，输出标签为未来命令；当前二值编码约定 1=Close、0=Open。每来源必须提供命令列、符号和执行对齐依据，不从真实开度或技能名称补造命令。

## 代码技术核验

旧模型分支与版本兼容接口移除后，CPU 回归共 277 项通过：训练环境 263 项，RoboTwin 专用环境 14 项；最后配置清理后另复核 32 项通过。ESkill 测试位于 `tests/test_eskill_model.py`、`test_eskill_data.py`、`test_eskill_training.py`、`test_eskill_policy.py`，覆盖硬路由与连续梯度、两进程全局梯度、终点／缺失标签、坐标、SFT 转接、训练保存恢复及在线回读。

模型训练核验使用随机初始化的 tiny CPU 模型与小型测试 Episode，没有运行真实 π0.5／VAE 权重训练、仿真成功率测评或本数据的 ESkill 视频人工验收。待物理时间与命令声明核实后，再从真实数据采样渲染。
