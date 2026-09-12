#include <iostream>
#include "experiment_runner.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    cout << "==================================================\n";
    cout << "CPU SCHEDULER EXPERIMENTAL RUNNER\n";
    cout << "==================================================\n\n";

    ExperimentConfig config;
    
    // Defaulting
    config.num_workloads = 20; 
    config.random_seed = 42;
    config.rr_quantum = 2;
    config.output_dir = "results/";

    // Simple CLI parser
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            config.random_seed = std::stoi(argv[++i]);
        } else if (arg == "--workloads" && i + 1 < argc) {
            config.num_workloads = std::stoi(argv[++i]);
        } else if (arg == "--out-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
            // Ensure trailing slash
            if (!config.output_dir.empty() && config.output_dir.back() != '/' && config.output_dir.back() != '\\') {
                config.output_dir += "/";
            }
        }
    }
    
    cout << "Config: Workloads=" << config.num_workloads 
         << ", Seed=" << config.random_seed 
         << ", OutDir=" << config.output_dir << "\n";

    try {
        ExperimentRunner::run(config);
    } catch (const exception& e) {
        cerr << "Error during Experiment execution: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
