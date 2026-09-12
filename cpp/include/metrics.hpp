#pragma once
#include "simulator.hpp"
#include <string>
#include <vector>

struct ProcessMetrics {
    int pid;
    int waiting_time;
    int turnaround_time;
    int response_time;
};

struct SchedulerMetrics {
    std::string scheduler_name;

    double average_waiting_time = 0.0;
    double average_turnaround_time = 0.0;
    double average_response_time = 0.0;

    int context_switches = 0;

    double normalized_waiting = 0.0;
    double normalized_turnaround = 0.0;
    double normalized_response = 0.0;
    double normalized_context_switches = 0.0;

    double objective_score = 0.0;
    
    std::vector<ProcessMetrics> process_metrics;
};

class Metrics {
public:
    // Calculates unnormalized metrics for a single simulator run
    static SchedulerMetrics calculateMetrics(const std::string& name, const Simulator& sim);

    // Normalizes a list of metrics and calculates objective scores for comparison
    static void normalizeAndScore(std::vector<SchedulerMetrics>& metrics_list);

    // Explicitly provided max values to ensure fair regret evaluation
    struct MaxMetrics {
        double max_wait = 0.0;
        double max_turn = 0.0;
        double max_resp = 0.0;
        int max_cs = 0;
    };

    // Computes max values from a given list (e.g. from the 4 base schedulers)
    static MaxMetrics getMaximums(const std::vector<SchedulerMetrics>& metrics_list);

    // Normalizes using explicitly provided maximums
    static void normalizeAndScoreWithReference(std::vector<SchedulerMetrics>& metrics_list, const MaxMetrics& maxes);
};
