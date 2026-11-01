# 迁移前代码

`data_code_20261101.tar.gz` 保存目录调整前的数据处理代码、规则、模型配置和回归测试，保留原相对路径及文件内容。既有 `data/**/check/` 报告中的代码 SHA256 对应当时的文件，不能用迁移后修改了 import 和路径的新文件替代核对。

这个快照不参与包安装或运行时 import。当前代码与命令见 [scripts/preprocess/README.md](../../scripts/preprocess/README.md)。如需查看某个原文件，可直接读取归档而无需恢复旧目录：

```bash
tar -xOf examples/archive/data_code_20261101.tar.gz data/io.py
```
