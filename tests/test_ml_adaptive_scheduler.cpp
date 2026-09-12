#include "ml_adaptive_scheduler.hpp"
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

std::vector<Process> createDeterministicWorkload() {
    std::vector<Process> processes;
    // Let's create a perfectly valid deterministic workload
    processes.push_back(Process(1, 0, 10, 2));
    processes.push_back(Process(2, 2, 20, 1));
    processes.push_back(Process(3, 4, 30, 3));
    processes.push_back(Process(4, 6, 5, 2));
    return processes;
}

void test_empty_workload() {
    MLAdaptiveScheduler scheduler;
    std::vector<Process> processes;
    try {
        scheduler.selectScheduler(processes);
        assert(false);
    } catch (const std::invalid_argument&) {}
}

void test_invalid_quantum() {
    try {
        MLAdaptiveScheduler scheduler(0);
        assert(false);
    } catch (const std::invalid_argument&) {}
}

void test_execution_and_inference() {
    // This requires python and the artifact files to exist.
    // If they don't, it will throw an exception, which we can catch or expect.
    // Since we are running on a machine where python is setup, it should pass.
    MLAdaptiveScheduler scheduler(2);
    auto processes = createDeterministicWorkload();
    
    // We isolate selection from execution
    MLDecisionResult decision = scheduler.selectScheduler(processes);
    
    assert(decision.predicted_label >= 0 && decision.predicted_label <= 3);
    assert(decision.selected_scheduler_name != "Unknown");
    assert(decision.confidence >= -1.0 && decision.confidence <= 1.0);
    
    // Now run simulation
    Simulator sim;
    scheduler.run(sim, processes);
    
    SchedulerMetrics m = Metrics::calculateMetrics("ML_Test", sim);
    assert(m.average_turnaround_time > 0.0);
}

void test_feature_passing() {
    MLAdaptiveScheduler scheduler(2);
    auto processes = createDeterministicWorkload();
    MLDecisionResult decision = scheduler.selectScheduler(processes);
    
    WorkloadFeatures feats = WorkloadFeatureExtractor::extract(processes);
    
    assert(decision.features[0] == static_cast<double>(feats.num_processes));
    assert(approx_equal(decision.features[1], feats.average_burst));
    assert(approx_equal(decision.features[2], feats.burst_std));
}

int main() {
    test_empty_workload();
    test_invalid_quantum();
    
    try {
        test_feature_passing();
        test_execution_and_inference();
        std::cout << "All ML Adaptive Scheduler tests passed!\n";
    } catch (const std::exception& e) {
        std::cerr << "ML Tests skipped or failed due to environment: " << e.what() << "\n";
        // We'll return 0 if Python is missing, but log it.
        // If Python is present and our code fails, we might want to fail the build.
        // For strict testing, we'll assert it should pass if models are present.
        return 1; 
    }
    
    return 0;
}
