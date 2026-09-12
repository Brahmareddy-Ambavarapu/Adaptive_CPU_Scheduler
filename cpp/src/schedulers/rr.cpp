#include "schedulers/rr.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <queue>
#include <vector>

RRScheduler::RRScheduler(int time_quantum) : quantum(time_quantum) {
    if (quantum <= 0) {
        throw std::invalid_argument("Time quantum must be > 0");
    }
}

void RRScheduler::run(Simulator& simulator) {
    std::cout << "Execution Order:\n";
    bool first = true;
    
    std::queue<int> ready_queue;
    std::vector<int> added_to_queue; // Tracks PIDs to prevent duplicate arrivals

    // Helper to add arrived processes to the ready queue
    auto add_ready_processes = [&]() {
        std::vector<int> newly_arrived;
        int cur_time = simulator.getCurrentTime();
        const auto& procs = simulator.getProcesses();
        
        for (const auto& p : procs) {
            if (p.arrival_time <= cur_time && p.remaining_time > 0) {
                if (std::find(added_to_queue.begin(), added_to_queue.end(), p.pid) == added_to_queue.end()) {
                    newly_arrived.push_back(p.pid);
                    added_to_queue.push_back(p.pid);
                }
            }
        }
        
        // Ensure deterministic ordering for simultaneous arrivals
        std::sort(newly_arrived.begin(), newly_arrived.end(), [&procs](int pid1, int pid2) {
            auto it1 = std::find_if(procs.begin(), procs.end(), [pid1](const Process& p) { return p.pid == pid1; });
            auto it2 = std::find_if(procs.begin(), procs.end(), [pid2](const Process& p) { return p.pid == pid2; });
            if (it1->arrival_time != it2->arrival_time) return it1->arrival_time < it2->arrival_time;
            return it1->pid < it2->pid;
        });
        
        for (int pid : newly_arrived) {
            ready_queue.push(pid);
        }
    };

    while (!simulator.isFinished()) {
        const auto& processes = simulator.getProcesses();
        
        // 1. Add processes that arrived while the CPU was idle or right at the start
        add_ready_processes();

        if (!ready_queue.empty()) {
            int current_pid = ready_queue.front();
            ready_queue.pop();

            if (!first) std::cout << " -> ";
            std::cout << "P" << current_pid;
            first = false;

            int remaining = 0;
            for (const auto& p : processes) {
                if (p.pid == current_pid) {
                    remaining = p.remaining_time;
                    break;
                }
            }

            // 2. Execute process up to the time quantum
            int time_to_run = std::min(quantum, remaining);
            simulator.executeProcess(current_pid, time_to_run);

            // 3. Check for new arrivals DURING the execution interval
            // Must be added to queue BEFORE the current process is re-added
            add_ready_processes();

            // 4. If current process isn't finished, put it at the back of the queue
            int new_remaining = 0;
            for (const auto& p : processes) {
                if (p.pid == current_pid) {
                    new_remaining = p.remaining_time;
                    break;
                }
            }

            if (new_remaining > 0) {
                ready_queue.push(current_pid);
            }

        } else {
            // CPU Idle. Find the next arriving process and jump time.
            int next_arrival = std::numeric_limits<int>::max();
            for (const auto& p : processes) {
                if (p.remaining_time > 0 && p.arrival_time > simulator.getCurrentTime()) {
                    if (p.arrival_time < next_arrival) {
                        next_arrival = p.arrival_time;
                    }
                }
            }
            if (next_arrival != std::numeric_limits<int>::max()) {
                simulator.advanceTime(next_arrival - simulator.getCurrentTime());
            }
        }
    }
    std::cout << "\n\n";
}
