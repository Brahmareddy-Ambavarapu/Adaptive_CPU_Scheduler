#include "rule_based_adaptive_scheduler.hpp"
#include "workload_feature_extractor.hpp"
#include "simulator.hpp"
#include "metrics.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

bool approx_equal(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

std::vector<Process> createUniformProcesses(int count, int burst, int priority, int arr_time_step = 1) {
    std::vector<Process> processes;
    for (int i = 0; i < count; ++i) {
        processes.push_back(Process(i+1, i * arr_time_step, burst, priority));
    }
    return processes;
}

void test_empty_workload() {
    RuleBasedConfig config;
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes;
    AdaptiveDecision decision = scheduler.selectScheduler(processes);
    
    assert(decision.selected_scheduler == SchedulerType::RR);
    assert(decision.rule_name == "DEFAULT_EMPTY_WORKLOAD");
    assert(decision.features.num_processes == 0);
}

void test_invalid_quantum() {
    RuleBasedConfig bad_config;
    bad_config.round_robin_quantum = 0;
    try {
        RuleBasedAdaptiveScheduler scheduler(bad_config);
        assert(false); // Should not reach here
    } catch (const std::invalid_argument&) {}
}

void test_feature_extraction() {
    std::vector<Process> processes = {
        Process(1, 0, 10, 2),
        Process(2, 2, 20, 1),
        Process(3, 4, 30, 3)
    };
    
    WorkloadFeatures features = WorkloadFeatureExtractor::extract(processes);
    
    assert(features.num_processes == 3);
    assert(approx_equal(features.average_burst, 20.0));
    assert(features.min_burst == 10);
    assert(features.max_burst == 30);
    assert(approx_equal(features.average_priority, 2.0));
    assert(approx_equal(features.burst_std, 8.1649658, 1e-5));
    assert(approx_equal(features.priority_std, 0.81649658, 1e-5));
    assert(approx_equal(features.arrival_rate, 0.6));
    assert(approx_equal(features.short_job_ratio, 2.0/3.0));
    assert(approx_equal(features.long_job_ratio, 1.0/3.0));
}

void test_priority_heavy() {
    RuleBasedConfig config;
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes = {
        Process(1, 0, 10, 1),
        Process(2, 0, 10, 1),
        Process(3, 0, 10, 10),
        Process(4, 0, 10, 10)
    };
    AdaptiveDecision decision = scheduler.selectScheduler(processes);
    assert(decision.selected_scheduler == SchedulerType::Priority);
    assert(decision.rule_name == "PRIORITY_HEAVY");
}

void test_short_job_heavy() {
    RuleBasedConfig config;
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes = {
        Process(1, 0, 1, 2),
        Process(2, 0, 2, 2),
        Process(3, 0, 3, 2),
        Process(4, 0, 100, 2)
    };
    AdaptiveDecision decision = scheduler.selectScheduler(processes);
    assert(decision.selected_scheduler == SchedulerType::SJF);
    assert(decision.rule_name == "SHORT_JOB_HEAVY");
}

void test_high_arrival_rate() {
    RuleBasedConfig config;
    config.short_job_ratio_threshold = 1.1; // Disable SJF to test RR
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes = {
        Process(1, 0, 10, 2),
        Process(2, 0, 10, 2),
        Process(3, 0, 10, 2),
        Process(4, 0, 10, 2)
    };
    AdaptiveDecision decision = scheduler.selectScheduler(processes);
    assert(decision.selected_scheduler == SchedulerType::RR);
    assert(decision.rule_name == "HIGH_ARRIVAL_RATE");
}

void test_long_job_heavy() {
    RuleBasedConfig config;
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes = {
        Process(1, 0, 1, 2),
        Process(2, 100, 10, 2),
        Process(3, 101, 10, 2),
        Process(4, 102, 10, 2)
    };
    AdaptiveDecision decision = scheduler.selectScheduler(processes);
    assert(decision.selected_scheduler == SchedulerType::FCFS);
    assert(decision.rule_name == "LONG_JOB_HEAVY");
}

void test_default() {
    RuleBasedConfig config;
    config.short_job_ratio_threshold = 1.1;
    config.long_job_ratio_threshold = 1.1;
    config.high_arrival_rate_threshold = 1.1;
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes = {
        Process(1, 0, 10, 2),
        Process(2, 10, 10, 2)
    };
    AdaptiveDecision decision = scheduler.selectScheduler(processes);
    assert(decision.selected_scheduler == SchedulerType::RR);
    assert(decision.rule_name == "DEFAULT");
}

void test_execution() {
    RuleBasedConfig config;
    RuleBasedAdaptiveScheduler scheduler(config);
    std::vector<Process> processes = {
        Process(1, 0, 10, 2)
    };
    Simulator sim;
    scheduler.run(sim, processes);
    
    SchedulerMetrics m = Metrics::calculateMetrics("Test", sim);
    assert(m.average_turnaround_time > 0.0);
}

int main() {
    test_empty_workload();
    test_invalid_quantum();
    test_feature_extraction();
    test_priority_heavy();
    test_short_job_heavy();
    test_high_arrival_rate();
    test_long_job_heavy();
    test_default();
    test_execution();
    
    std::cout << "All Rule-Based Adaptive Scheduler tests passed!\n";
    return 0;
}
