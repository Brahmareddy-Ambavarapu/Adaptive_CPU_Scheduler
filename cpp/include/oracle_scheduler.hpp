#ifndef ORACLE_SCHEDULER_HPP
#define ORACLE_SCHEDULER_HPP

#include "process.hpp"
#include "scheduler_type.hpp"
#include "metrics.hpp"
#include <vector>
#include <array>
#include <string>

struct OracleDecision {
    SchedulerType optimal_scheduler;
    std::string optimal_scheduler_name;
    double optimal_score;
    
    // Ordered: [0]=FCFS, [1]=SJF, [2]=RR, [3]=Priority
    std::array<SchedulerMetrics, 4> base_metrics;
    Metrics::MaxMetrics maximums; // The max values used to normalize
};

class OracleScheduler {
public:
    static OracleDecision evaluate(const std::vector<Process>& processes, int rr_quantum = 2);
};

#endif // ORACLE_SCHEDULER_HPP
