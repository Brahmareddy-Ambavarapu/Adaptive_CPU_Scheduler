#include "experiment_runner.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

void test_experiment_output() {
    ExperimentConfig config;
    config.num_workloads = 2; // small test
    config.output_dir = "test_results/";
    
    ExperimentRunner::run(config);
    
    // Check if files were created
    std::ifstream per_wl("test_results/experiments/per_workload.csv");
    assert(per_wl.is_open());
    
    std::string line;
    int lines = 0;
    while(std::getline(per_wl, line)) lines++;
    assert(lines == 3); // 1 header + 2 workloads
    
    std::ifstream perf_sum("test_results/summaries/performance_summary.csv");
    assert(perf_sum.is_open());
    lines = 0;
    while(std::getline(perf_sum, line)) lines++;
    assert(lines == 3); // 1 header + ML + Rule-Based
    
    std::ifstream ml_sum("test_results/summaries/ml_selection_summary.csv");
    assert(ml_sum.is_open());
}

int main() {
    try {
        test_experiment_output();
        std::cout << "All Experiment Runner tests passed!\n";
    } catch(const std::exception& e) {
        std::cerr << "Tests skipped or failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
