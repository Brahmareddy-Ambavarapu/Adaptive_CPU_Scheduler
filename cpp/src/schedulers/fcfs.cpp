#include "schedulers/fcfs.hpp"
#include <algorithm>
#include <iostream>
#include <limits>

void FCFSScheduler::run(Simulator& simulator) {
    std::cout << "Execution Order:\n";
    bool first = true;

    while (!simulator.isFinished()) {
        const auto& processes = simulator.getProcesses();
        int current_time = simulator.getCurrentTime();
        
        int best_pid = -1;
        int earliest_arrival = std::numeric_limits<int>::max();

        // 1. Look at all available processes
        for (const auto& p : processes) {
            if (p.remaining_time > 0 && p.arrival_time <= current_time) {
                if (p.arrival_time < earliest_arrival) {
                    earliest_arrival = p.arrival_time;
                    best_pid = p.pid;
                } else if (p.arrival_time == earliest_arrival) {
                    // 3. Deterministic Tie-breaker (PID)
                    if (best_pid == -1 || p.pid < best_pid) {
                        best_pid = p.pid;
                    }
                }
            }
        }

        // 5. Execute the selected process
        if (best_pid != -1) {
            if (!first) std::cout << " -> ";
            std::cout << "P" << best_pid;
            first = false;

            // FCFS is non-preemptive: Run for its entire remaining time
            int burst_to_run = 0;
            for (const auto& p : processes) {
                if (p.pid == best_pid) {
                    burst_to_run = p.remaining_time;
                    break;
                }
            }
            simulator.executeProcess(best_pid, burst_to_run);
        } else {
            // 4. CPU is idle (no arrived processes). Jump time to next arrival.
            int next_arrival = std::numeric_limits<int>::max();
            for (const auto& p : processes) {
                if (p.remaining_time > 0 && p.arrival_time > current_time) {
                    if (p.arrival_time < next_arrival) {
                        next_arrival = p.arrival_time;
                    }
                }
            }
            
            if (next_arrival != std::numeric_limits<int>::max()) {
                simulator.advanceTime(next_arrival - current_time);
            }
        }
    }
    std::cout << "\n\n";
}
