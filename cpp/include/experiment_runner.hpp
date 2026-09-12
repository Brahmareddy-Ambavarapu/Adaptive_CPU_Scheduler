#ifndef EXPERIMENT_RUNNER_HPP
#define EXPERIMENT_RUNNER_HPP

#include "process.hpp"
#include "oracle_scheduler.hpp"
#include <string>
#include <vector>

struct ExperimentConfig {
    int num_workloads = 1000;
    int rr_quantum = 2;
    unsigned int random_seed = 42;
    std::string output_dir = "results/";
};

struct AdaptiveStats {
    double average_score = 0.0;
    double median_score = 0.0;
    double average_waiting = 0.0;
    double average_response = 0.0;
    double average_turnaround = 0.0;
    double average_context_switches = 0.0;

    double average_regret = 0.0;
    double median_regret = 0.0;
    double worst_regret = 0.0;
    double min_regret = 1e9; // Initially very large
    
    int zero_regret_count = 0;
    int within_5_percent_count = 0;
    int within_10_percent_count = 0;
    
    // For ML Only
    int correct_predictions = 0;
};

class ExperimentRunner {
public:
    static void run(const ExperimentConfig& config);
};

#endif // EXPERIMENT_RUNNER_HPP
