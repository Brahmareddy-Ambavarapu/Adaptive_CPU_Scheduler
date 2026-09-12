#include "oracle_scheduler.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

bool approx_equal(double a, double b, double epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

void test_oracle_tie_break() {
    // Generate a simple deterministic workload
    std::vector<Process> processes = {
        Process(1, 0, 10, 2),
        Process(2, 2, 5, 1)
    };
    
    OracleDecision oracle = OracleScheduler::evaluate(processes, 2);
    
    // Oracle should pick something valid
    int sched_int = static_cast<int>(oracle.optimal_scheduler);
    assert(sched_int >= 0 && sched_int <= 3);
    assert(oracle.optimal_score > 0.0);
    
    // Make sure base metrics array is populated
    assert(oracle.base_metrics[0].scheduler_name == "FCFS");
    assert(oracle.base_metrics[1].scheduler_name == "SJF");
    assert(oracle.base_metrics[2].scheduler_name == "RR");
    assert(oracle.base_metrics[3].scheduler_name == "Priority");
}

void test_oracle_empty() {
    std::vector<Process> empty_wl;
    OracleDecision oracle = OracleScheduler::evaluate(empty_wl, 2);
    assert(oracle.optimal_score == 0.0);
    assert(oracle.optimal_scheduler == SchedulerType::FCFS); // defaults to 0 on tie
}

int main() {
    test_oracle_tie_break();
    test_oracle_empty();
    std::cout << "All Oracle Scheduler tests passed!\n";
    return 0;
}
