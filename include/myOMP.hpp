#pragma once
#include <omp.h>
#include <vector>
namespace myOMP
{
    std::vector<std::pair<int, int>> distributeTasks(int numThreads, int totalTasks);
    void initOmpMklNestedParallel(); // 初始化OMP、MKL嵌套并行设置
    template <typename T>
    std::vector<int> assignThreadCount(int numThreads, std::vector<T> nJobs); // TODO
} // namespace myOMP
