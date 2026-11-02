# 修复前结果

保留 run_002 原始 manifest、区域、mask、视频和人工表。SAM3 CPU 状态卸载导致 mask 异常，本批次不可用作训练或模型质量评价。text/ 和 prompts/ 仍位于 processed 根目录，内容及哈希未修改；原 report 中相对视频路径以本目录为根。原汇总表保存在实验 run_002/metadata/pre_fix_summary.csv，其中旧的绝对 video 路径需映射到本目录下的同名 video 文件，身份以原 report SHA256 为准。
