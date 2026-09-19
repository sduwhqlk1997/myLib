#include <myOMP.hpp>
#include <mkl.h>
namespace myOMP
{
    std::vector<std::pair<int, int>> distributeTasks(int numThreads, int totalTasks)
    {
        // 返回值：first为对应线程的起始任务编号，second为任务数量
        int quo = totalTasks / numThreads;
        int rem = totalTasks % numThreads;
        std::vector<std::pair<int, int>> task_omp(numThreads);
        for (int i = 0; i < numThreads; ++i)
        {
            int count = rem < 1 ? quo : quo + 1;
            int start = i == 0 ? 0 : task_omp[i - 1].first + task_omp[i - 1].second;
            rem--;
            task_omp[i] = {start, count};
        }
        return task_omp;
    }
    void initOmpMklNestedParallel()
    {
        omp_set_dynamic(0);

        // 允许 nested OpenMP
        omp_set_max_active_levels(2);

        // 禁止 MKL 动态缩小线程数
        mkl_set_dynamic(0);
    }

}