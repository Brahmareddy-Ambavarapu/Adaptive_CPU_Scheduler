#ifndef ML_ADAPTIVE_SCHEDULER_HPP
#define ML_ADAPTIVE_SCHEDULER_HPP

#include "process.hpp"
#include "scheduler_type.hpp"
#include "workload_features.hpp"
#include "simulator.hpp"
#include <vector>
#include <string>
#include <array>

struct MLDecisionResult {
    SchedulerType selected_scheduler;
    std::array<double, 10> features;
    std::string selected_scheduler_name;
    int predicted_label;
    double confidence;
};

class MLAdaptiveScheduler {
public:
    // Configurable default quantum for RR
    explicit MLAdaptiveScheduler(int round_robin_quantum = 2);

    MLDecisionResult selectScheduler(const std::vector<Process>& processes) const;
    void run(Simulator& simulator, const std::vector<Process>& processes) const;

private:
    int round_robin_quantum;
    
    // Helper to invoke the Python CLI
    std::string runPythonInference(const std::array<double, 10>& features) const;
};

#endif // ML_ADAPTIVE_SCHEDULER_HPP
