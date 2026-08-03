#pragma once
#include <omp.h>
#include <vector>
namespace myOMP
{
    std::vector<std::pair<int, int>> distributeTasks(int numThreads, int totalTasks);
} // namespace myOMP
