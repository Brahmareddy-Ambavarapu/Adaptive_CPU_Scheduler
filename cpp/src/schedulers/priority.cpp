#include "schedulers/priority.hpp"
#include <algorithm>
#include <iostream>
#include <limits>

void PriorityScheduler::run(Simulator& simulator) {
    std::cout << "Execution Order:\n";
    bool first = true;

    while (!simulator.isFinished()) {
        const auto& processes = simulator.getProcesses();
        int current_time = simulator.getCurrentTime();
        
        int best_pid = -1;
        int highest_priority = std::numeric_limits<int>::max(); // Smaller number = higher priority
        int earliest_arrival = std::numeric_limits<int>::max();

        for (const auto& p : processes) {
            // Only consider processes that have arrived and are not finished
            if (p.remaining_time > 0 && p.arrival_time <= current_time) {
                // Primary check: Lowest priority number (highest priority)
                if (p.priority < highest_priority) {
                    highest_priority = p.priority;
                    earliest_arrival = p.arrival_time;
                    best_pid = p.pid;
                } else if (p.priority == highest_priority) {
                    // Tie-breaker 1: Earliest arrival time
                    if (p.arrival_time < earliest_arrival) {
                        earliest_arrival = p.arrival_time;
                        best_pid = p.pid;
                    } else if (p.arrival_time == earliest_arrival) {
                        // Tie-breaker 2: Lowest PID
                        if (best_pid == -1 || p.pid < best_pid) {
                            best_pid = p.pid;
                        }
                    }
                }
            }
        }

        if (best_pid != -1) {
            if (!first) std::cout << " -> ";
            std::cout << "P" << best_pid;
            first = false;

            // Non-preemptive: Run the selected process for its entire remaining time
            int burst_to_run = 0;
            for (const auto& p : processes) {
                if (p.pid == best_pid) {
                    burst_to_run = p.remaining_time;
                    break;
                }
            }
            simulator.executeProcess(best_pid, burst_to_run);
        } else {
            // CPU is idle. Find the next arriving process and advance time.
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
