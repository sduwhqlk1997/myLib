### 自主开发的一些工具（自用）
- `include/mat2eigen.hpp`：将 Matlab 导出的 `.mat` 文件写入 Eigen 变量，或者把 Eigen 变量写入 `.mat` 文件的工具函数。
- `include/myfem.hpp`：进行简单的有限元网格剖分、刚度矩阵组装和网格合并操作。
- `include/myEigen.hpp`：基于Eigen库实现的一些常用矩阵操作。
- `include/myOMP.hpp`：用于openMP并行任务分配。
- `include/SAW.hpp`：用于SAW谐振器准三维有限元模型的仿真求解。  

更多文档：
- [SAW.hpp API 文档](docs/SAW.md)
- [myfem.hpp API 文档](docs/myfem.md)
