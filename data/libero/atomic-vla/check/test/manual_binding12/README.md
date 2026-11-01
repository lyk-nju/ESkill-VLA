# 12 条轨迹的实例人工补充

**助手已逐条完成实例对应，补充 34 个 `manual_bound`，原有 14 个自动绑定保留。** 本批 12 条的四个 Event 均对应：`moka_pot_2 → moka_pot_2 → moka_pot_1 → moka_pot_1`，分别为第一组 Pick／Place、第二组 Pick／Place。这是本批逐条视觉检查的结论，不作为通用规则中的编号映射。

入口：**[12 条补齐视频、章节与证据图](index.html)** · [合并视频：8 分 18.2 秒](../../archive/history_20261101/test/video/atomicvla_manual_binding12.mp4) · [34 个补充标注](annotations.json) · [全部 1,693 条的合并实例标签](merged_bindings.json) · [用户核验表](user_review.csv)。助手标注和用户验收分开：`reviewer=assistant_visual_review`，用户仍为 pending。

| 作者标注 ID | 本次补充 Event 数 | 第一组 Pick／Place | 第二组 Pick／Place |
| --- | ---: | --- | --- |
| 23 | 2 | moka_pot_2 | moka_pot_1 |
| 46（原 100 条的 015） | 1 | moka_pot_2 | moka_pot_1 |
| 51 | 4 | moka_pot_2 | moka_pot_1 |
| 67 | 4 | moka_pot_2 | moka_pot_1 |
| 70 | 2 | moka_pot_2 | moka_pot_1 |
| 73 | 4 | moka_pot_2 | moka_pot_1 |
| 143 | 4 | moka_pot_2 | moka_pot_1 |
| 204 | 2 | moka_pot_2 | moka_pot_1 |
| 283 | 1 | moka_pot_2 | moka_pot_1 |
| 288 | 4 | moka_pot_2 | moka_pot_1 |
| 306 | 2 | moka_pot_2 | moka_pot_1 |
| 341 | 4 | moka_pot_2 | moka_pot_1 |

实体 ID 先由初始帧的原生 mask 确认：本批画面左侧是 `moka_pot_1`，较靠炉子的右前侧是 `moka_pot_2`。随后沿原始 RGB 中的夹持、抬起、搬运和释放过程确认身份，并用官方状态的连续位姿／原始接触轨迹佐证。每条保存 20 个进度关键帧、初始实体对照及完整逐帧接触记录；助手实际查看了全部 12 张事件检查图，不将此说成用户逐帧验收。

部分值得保留的证据：67 的 Pick 末尾原图尚未明显抬起物体，后续 Place 明确搬运对应壶；288 第二次 Pick 的原始双指接触计数为 0，但原图和接下来的 Place 连续搬运明确对应左侧 1；306 末段两壶接触或被碰动，当前对象仍是连续持有的 1；341 使用了 `first/second coffee pot` 的表述，实际实体仍是两只 moka pot。没有为了补标签放松自动运动阈值。

合并后共 **3,839 个自动绑定＋34 个助手视觉标注＝3,873 个 Event**，1,693 条轨迹的实例身份字段无缺口。后续需要补齐实例的导出使用本目录 `merged_bindings.json`，或在自动输出后显式调用来源专用 `apply_manual`；纯自动规则评估继续使用 `all_rules_v1` 原结果。

本次没有消除官方状态与原图的位置偏差：原 740 条对齐警示仍保留，其中 11 条属于本批。身份补充不代表框的位置已通过验收。源事件边界、技能、原始描述、动作和帧号保持原值。

标注依据见 [visual_decisions.json](visual_decisions.json)、`evidence/<标注ID>.jpg/json` 和 `anchors/`。补充条目锁定源轨迹哈希、事件起止及技能；自动结果留在 `automatic_result` 中，不把 `manual_bound` 伪装成规则成功。

脚本为 [manual_binding12.py](../../../scripts/manual_binding12.py)，分为 `inspect`、`record`、`render`、`merge` 四步。通用规则文件未改；共用原生渲染助手只增加已选定人工实例的几何渲染支持。66 项测试及 6 个子测试通过，见 [tests.log](tests.log)，覆盖源轨迹一致性、边界变更拒绝和自动绑定保留。

```bash
NATIVE_PY=/data1/yuankai/robot-research/workspace/envs/libero-native/bin/python
export LIBERO_CONFIG_PATH=/data1/yuankai/robot-research/workspace/envs/libero-native/config
export NUMBA_CACHE_DIR=/tmp/eskill-review100-numba
export MPLCONFIGDIR=/tmp/eskill-review100-mpl
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$LIBERO_CONFIG_PATH/nvidia_egl.json"
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.manual_binding12 inspect
# 检查证据图，填写 visual_decisions.json；本批已由助手完成。
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.manual_binding12 record
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.manual_binding12 render --workers 4
"$NATIVE_PY" -m data.libero.atomic-vla.scripts.manual_binding12 merge
```

复跑视频需使用新的空输出目录。大缓存位于 `/data1/yuankai/robot-research/data/external/atomic_vla/manual_binding12/`。

---

以下为执行前记录的范围和预算。

2026-11-01 用户授权手工确定全量规则检查遗留的 12 条实例对应。本轮固定标注 ID：23、46、51、67、70、73、143、204、283、288、306、341；沿用 `all_rules_v1` 的来源、状态和规则快照，只补 34 个未绑定事件。自动规则的判断和已有 14 个绑定保留。

方法：先在每条轨迹的初始场景中用原生实例 ID 标出两只候选壶，再查看原始主相机在各 Event 中的连续动作与接触关键帧，结合逐帧接触、物体位姿记录确认身份。助手视觉检查作为单独的人工补充，不改通用规则、不按“第一只”直接指定实例 1。若证据仍有矛盾，继续查看原图，不以自动阈值失败作为手工结论。

预算：已缓存数据，新增下载 0、模型调用 0；最多 4 个工作进程，CPU／GPU 各不超过一小时，新增缓存上限 512 MiB。检查全部 12 条，不另抽十条替代。使用已确认的共用渲染器，主相机及完整屏内区域，视频在 `check/test/video/manual_binding12/`，结果由用户继续核验。源事件、技能、帧号、动作、相机及数据集版本保持原值，已有对齐风险另行记录。
