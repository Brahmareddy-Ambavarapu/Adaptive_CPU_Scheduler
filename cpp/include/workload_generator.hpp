#pragma once
#include <vector>
#include <random>
#include "process.hpp"

enum class WorkloadType {
    ShortJobHeavy,
    LongJobHeavy,
    Mixed,
    CpuHeavy,
    HighArrivalRate,
    LowArrivalRate,
    PriorityHeavy,
    Bimodal
};

struct WorkloadConfig {
    int num_processes = 10;
    int max_arrival_time = 100;
    int min_burst_time = 1;
    int max_burst_time = 20;
    int min_priority = 1;
    int max_priority = 5;
    int round_robin_quantum = 4;
    unsigned int seed = 42;
};

class WorkloadGenerator {
private:
    std::mt19937 gen;

public:
    explicit WorkloadGenerator(unsigned int seed);

    std::vector<Process> generate(WorkloadType type, const WorkloadConfig& config);
};
