#include "rule_based_adaptive_scheduler.hpp"
#include "workload_feature_extractor.hpp"
#include "schedulers/fcfs.hpp"
#include "schedulers/sjf.hpp"
#include "schedulers/rr.hpp"
#include "schedulers/priority.hpp"
#include <stdexcept>
#include <memory>

RuleBasedAdaptiveScheduler::RuleBasedAdaptiveScheduler(const RuleBasedConfig& config)
    : config(config) {
    if (config.round_robin_quantum <= 0) {
        throw std::invalid_argument("Round Robin quantum must be positive.");
    }
}

AdaptiveDecision RuleBasedAdaptiveScheduler::selectScheduler(const std::vector<Process>& processes) const {
    AdaptiveDecision decision;
    
    if (processes.empty()) {
        decision.selected_scheduler = SchedulerType::RR;
        decision.rule_name = "DEFAULT_EMPTY_WORKLOAD";
        return decision;
    }
    
    // Extract purely pre-execution workload characteristics
    decision.features = WorkloadFeatureExtractor::extract(processes);
    
    // Evaluate Rule 1: Priority-Heavy Workload
    // '1' is highest priority, so smaller numeric values are "higher" priority.
    // Let's define a "high priority" job as one with priority <= average_priority.
    int high_priority_count = 0;
    for (const auto& p : processes) {
        if (p.priority <= decision.features.average_priority) {
            high_priority_count++;
        }
    }
    double high_priority_ratio = static_cast<double>(high_priority_count) / decision.features.num_processes;
    
    if (decision.features.priority_std >= config.priority_std_threshold && 
        high_priority_ratio >= config.high_priority_ratio_threshold) {
        decision.selected_scheduler = SchedulerType::Priority;
        decision.rule_name = "PRIORITY_HEAVY";
        return decision;
    }
    
    // Evaluate Rule 2: Short-Job-Heavy Workload
    if (decision.features.short_job_ratio >= config.short_job_ratio_threshold) {
        decision.selected_scheduler = SchedulerType::SJF;
        decision.rule_name = "SHORT_JOB_HEAVY";
        return decision;
    }
    
    // Evaluate Rule 3: High-Arrival / Interactive Workload
    if (decision.features.arrival_rate >= config.high_arrival_rate_threshold) {
        decision.selected_scheduler = SchedulerType::RR;
        decision.rule_name = "HIGH_ARRIVAL_RATE";
        return decision;
    }
    
    // Evaluate Rule 4: Long-Job / CPU-Heavy Workload
    if (decision.features.long_job_ratio >= config.long_job_ratio_threshold) {
        decision.selected_scheduler = SchedulerType::FCFS;
        decision.rule_name = "LONG_JOB_HEAVY";
        return decision;
    }
    
    // Evaluate Rule 5: Default
    decision.selected_scheduler = SchedulerType::RR;
    decision.rule_name = "DEFAULT";
    return decision;
}

void RuleBasedAdaptiveScheduler::run(Simulator& simulator, const std::vector<Process>& processes) const {
    AdaptiveDecision decision = selectScheduler(processes);
    
    // We instantiate the correct scheduler and pass it to the simulator
    // using clean processes vector (no state mutation before scheduling).
    std::unique_ptr<Scheduler> scheduler;
    
    switch (decision.selected_scheduler) {
        case SchedulerType::FCFS:
            scheduler = std::make_unique<FCFSScheduler>();
            break;
        case SchedulerType::SJF:
            scheduler = std::make_unique<SJFScheduler>();
            break;
        case SchedulerType::RR:
            scheduler = std::make_unique<RRScheduler>(config.round_robin_quantum);
            break;
        case SchedulerType::Priority:
            scheduler = std::make_unique<PriorityScheduler>();
            break;
        default:
            scheduler = std::make_unique<RRScheduler>(config.round_robin_quantum);
            break;
    }
    simulator.reset();
    for (const auto& p : processes) {
        simulator.addProcess(p);
    }
    
    scheduler->run(simulator);
}
