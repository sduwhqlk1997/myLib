#pragma once
#include <omp.h>
#include <vector>

namespace myOMP
{
    std::vector<std::pair<int, int>> distributeTasks(int numThreads, int totalTasks);
    void initOmpMklNestedParallel();                                            // 初始化OMP、MKL嵌套并行设置
    std::vector<int> assignThreadCount(int numThreads, std::vector<int> nJobs); // TODO
} // namespace myOMP
