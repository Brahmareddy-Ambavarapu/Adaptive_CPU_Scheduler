#ifndef RULE_BASED_ADAPTIVE_SCHEDULER_HPP
#define RULE_BASED_ADAPTIVE_SCHEDULER_HPP

#include "process.hpp"
#include "scheduler_type.hpp"
#include "workload_features.hpp"
#include "simulator.hpp"
#include <vector>
#include <string>

struct RuleBasedConfig {
    double priority_std_threshold = 1.0;
    double high_priority_ratio_threshold = 0.3; // Proportion of jobs with priority >= average
    double short_job_ratio_threshold = 0.6;
    double high_arrival_rate_threshold = 0.8;
    double long_job_ratio_threshold = 0.6;
    int round_robin_quantum = 2;
};

struct AdaptiveDecision {
    SchedulerType selected_scheduler;
    WorkloadFeatures features;
    std::string rule_name;
};

class RuleBasedAdaptiveScheduler {
public:
    explicit RuleBasedAdaptiveScheduler(const RuleBasedConfig& config = {});

    AdaptiveDecision selectScheduler(const std::vector<Process>& processes) const;
    void run(Simulator& simulator, const std::vector<Process>& processes) const;

private:
    RuleBasedConfig config;
};

#endif // RULE_BASED_ADAPTIVE_SCHEDULER_HPP
