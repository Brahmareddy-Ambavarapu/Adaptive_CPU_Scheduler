#include "oracle_scheduler.hpp"
#include "simulator.hpp"
#include "schedulers/fcfs.hpp"
#include "schedulers/sjf.hpp"
#include "schedulers/rr.hpp"
#include "schedulers/priority.hpp"
#include <cmath>

OracleDecision OracleScheduler::evaluate(const std::vector<Process>& processes, int rr_quantum) {
    OracleDecision decision;
    
    // Create Simulators and schedulers
    Simulator sim_fcfs, sim_sjf, sim_rr, sim_pri;
    FCFSScheduler fcfs;
    SJFScheduler sjf;
    RRScheduler rr(rr_quantum);
    PriorityScheduler pri;
    
    // Execute independently on clean copies
    std::vector<Process> p_fcfs = processes;
    std::vector<Process> p_sjf = processes;
    std::vector<Process> p_rr = processes;
    std::vector<Process> p_pri = processes;
    
    for (const auto& p : p_fcfs) sim_fcfs.addProcess(p);
    fcfs.run(sim_fcfs);
    
    for (const auto& p : p_sjf) sim_sjf.addProcess(p);
    sjf.run(sim_sjf);
    
    for (const auto& p : p_rr) sim_rr.addProcess(p);
    rr.run(sim_rr);
    
    for (const auto& p : p_pri) sim_pri.addProcess(p);
    pri.run(sim_pri);
    
    // Calculate metrics
    std::vector<SchedulerMetrics> metrics_list;
    metrics_list.push_back(Metrics::calculateMetrics("FCFS", sim_fcfs));
    metrics_list.push_back(Metrics::calculateMetrics("SJF", sim_sjf));
    metrics_list.push_back(Metrics::calculateMetrics("RR", sim_rr));
    metrics_list.push_back(Metrics::calculateMetrics("Priority", sim_pri));
    
    // Store maximums and normalize
    decision.maximums = Metrics::getMaximums(metrics_list);
    Metrics::normalizeAndScoreWithReference(metrics_list, decision.maximums);
    
    // Store base metrics
    for (size_t i = 0; i < 4; ++i) {
        decision.base_metrics[i] = metrics_list[i];
    }
    
    // Find optimum. Tie-break order: FCFS, SJF, RR, Priority.
    int best_index = 0;
    double min_score = metrics_list[0].objective_score;
    
    for (int i = 1; i < 4; ++i) {
        // Tie-breaker epsilon: 1e-9
        if (metrics_list[i].objective_score < min_score - 1e-9) {
            min_score = metrics_list[i].objective_score;
            best_index = i;
        }
    }
    
    decision.optimal_score = min_score;
    decision.optimal_scheduler_name = metrics_list[best_index].scheduler_name;
    decision.optimal_scheduler = static_cast<SchedulerType>(best_index);
    
    return decision;
}
