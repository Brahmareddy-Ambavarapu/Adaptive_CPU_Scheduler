#include "metrics.hpp"
#include <algorithm>

SchedulerMetrics Metrics::calculateMetrics(const std::string& name, const Simulator& sim) {
    SchedulerMetrics result;
    result.scheduler_name = name;

    const auto& processes = sim.getProcesses();
    int num_processes = processes.size();

    if (num_processes == 0) {
        return result; // Empty workload gracefully handled, all 0s
    }

    long long total_waiting = 0;
    long long total_turnaround = 0;
    long long total_response = 0;

    for (const auto& p : processes) {
        ProcessMetrics pm;
        pm.pid = p.pid;
        
        // If the process didn't run, handle gracefully. In our simulation, it should run.
        if (p.completion_time == -1 || p.start_time == -1) {
            pm.turnaround_time = 0;
            pm.waiting_time = 0;
            pm.response_time = 0;
        } else {
            pm.turnaround_time = p.completion_time - p.arrival_time;
            pm.waiting_time = pm.turnaround_time - p.burst_time;
            pm.response_time = p.start_time - p.arrival_time;
        }

        total_waiting += pm.waiting_time;
        total_turnaround += pm.turnaround_time;
        total_response += pm.response_time;

        result.process_metrics.push_back(pm);
    }

    result.average_waiting_time = static_cast<double>(total_waiting) / num_processes;
    result.average_turnaround_time = static_cast<double>(total_turnaround) / num_processes;
    result.average_response_time = static_cast<double>(total_response) / num_processes;

    // Calculate context switches
    int context_switches = 0;
    const auto& history = sim.getExecutionHistory();
    if (!history.empty()) {
        int last_pid = history[0];
        for (size_t i = 1; i < history.size(); ++i) {
            if (history[i] != last_pid) {
                context_switches++;
                last_pid = history[i];
            }
        }
    }
    result.context_switches = context_switches;

    return result;
}

Metrics::MaxMetrics Metrics::getMaximums(const std::vector<SchedulerMetrics>& metrics_list) {
    MaxMetrics maxes;
    for (const auto& m : metrics_list) {
        if (m.average_waiting_time > maxes.max_wait) maxes.max_wait = m.average_waiting_time;
        if (m.average_turnaround_time > maxes.max_turn) maxes.max_turn = m.average_turnaround_time;
        if (m.average_response_time > maxes.max_resp) maxes.max_resp = m.average_response_time;
        if (m.context_switches > maxes.max_cs) maxes.max_cs = m.context_switches;
    }
    return maxes;
}

void Metrics::normalizeAndScoreWithReference(std::vector<SchedulerMetrics>& metrics_list, const MaxMetrics& maxes) {
    if (metrics_list.empty()) return;

    for (auto& m : metrics_list) {
        m.normalized_waiting = (maxes.max_wait > 0.0) ? (m.average_waiting_time / maxes.max_wait) : 0.0;
        m.normalized_turnaround = (maxes.max_turn > 0.0) ? (m.average_turnaround_time / maxes.max_turn) : 0.0;
        m.normalized_response = (maxes.max_resp > 0.0) ? (m.average_response_time / maxes.max_resp) : 0.0;
        m.normalized_context_switches = (maxes.max_cs > 0) ? (static_cast<double>(m.context_switches) / maxes.max_cs) : 0.0;

        // Score = 0.4*Wait + 0.3*Resp + 0.2*Turn + 0.1*CS
        m.objective_score = (0.40 * m.normalized_waiting) +
                            (0.30 * m.normalized_response) +
                            (0.20 * m.normalized_turnaround) +
                            (0.10 * m.normalized_context_switches);
    }
}

void Metrics::normalizeAndScore(std::vector<SchedulerMetrics>& metrics_list) {
    if (metrics_list.empty()) return;
    MaxMetrics maxes = getMaximums(metrics_list);
    normalizeAndScoreWithReference(metrics_list, maxes);
}
