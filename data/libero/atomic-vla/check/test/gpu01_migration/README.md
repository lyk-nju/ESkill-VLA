> 新版训练改用 `/data1/yuankai/robot-research/data/atomicdata/libero40/`。直接搬运该完整目录即可，运行时无需旧 `external/` 或 `pi05_sft/`，也无需重写路径。目录内 `SHA256SUMS` 列出全部 5,080 个文件（共 10,335,558,214 bytes）；在目标目录执行 `sha256sum -c SHA256SUMS` 校验。旧源目录已于 2026-11-03 删除，下方 tar 包是旧链路的唯一副本。以下清单和归档属于迁移前的旧训练链路，保持原样，不作为新版交付。模型权重仍按目标服务器配置另行下载。

# LIBERO 训练数据迁移清单

范围：现有 LIBERO-40 双相机 SFT 与 ESkill Episode／二维 Motion 训练；不包含模型权重、仿真测评资产、历史实验产物或训练配置。模型权重由目标服务器另行准备。

- [训练文件清单](training_files.txt)：按当前 Dataset 读取路径从 1,693 条带标定 Episode 的图像、状态、动作、末端、相机及区域字段收集 URI，并加入 Episode JSON、索引和完整 `pi05_sft` 目录。
- [统计](manifest.json)：6,779 个训练文件，约 15.07 GB（十进制），源端全部存在。含 1,693 个 `arrays.npz`、1,693 条带标定 Episode JSON、1,693 个 `motion.npz`、1,693 个引用源数组、两个索引及 SFT 缓存文件。
- [源文件 SHA256](sha256sums.txt)：用于目标端逐文件内容校验。
- [其他来源引用](provenance_references.txt)：Episode 元数据中额外引用的来源和核验文件，不计入当前 Dataset 训练必需清单；若要完整保留来源追溯，应进一步核对并复制。

源端对 40 条轨迹各取首／中／末帧，共 120 个 Dataset 样本回读，实际访问的 120 个文件均包含在训练清单中。该检查不替代目标服务器上的全量文件完整性及训练回读。

源 Episode 内含绝对路径。目标端若保持原路径，按清单保留目录结构；若更换路径，须改写实际读取引用并检查。迁移后核对每个目标文件的存在性和内容哈希，再抽样回读训练数据及双相机 SFT 配对。当前仅完成源端清单与抽样，尚未传输或核验目标端。

中转包已生成：`/data1/yuankai/robot-research/data/migration/gpu01_libero40/libero40_training.tar`。包内结构为 `libero40/`、`external/`、`target_sha256sums.txt` 和 `prepare_target.py`。目标根目录为 `/data/home/shengqiuProf_user_yuankai/data/robot-research/atomicdata/`。在目标根目录解包后，运行 `python3 prepare_target.py .`：先逐文件检查 6,779 项 SHA256，再将 Episode JSON 的源数据 `uri` 替换为目标路径。该过程会改变 Episode JSON，因此源文件哈希只用于改写之前的校验；报告保存为 `libero40_migration_report.json`。路径转换脚本已使用合成样本测试，尚未在目标服务器执行或训练回读。

中转包 SHA256：`6f4bff03ef9f6f6d5aad4a12f68b14224d0c257d12a00c810736547dfbae3f64`。tar 文件包含 6,781 个条目（6,779 个训练文件及两个辅助文件）。
