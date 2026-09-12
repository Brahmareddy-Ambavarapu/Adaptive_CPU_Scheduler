#include "simulator.hpp"
#include <stdexcept>
#include <algorithm>

Simulator::Simulator() : current_time(0) {}

void Simulator::addProcess(const Process& p) {
    processes.push_back(p);
}

int Simulator::getCurrentTime() const {
    return current_time;
}

void Simulator::advanceTime(int time_slice) {
    if (time_slice > 0) {
        current_time += time_slice;
    }
}

void Simulator::executeProcess(int pid, int time_slice) {
    if (time_slice <= 0) return; // Handle execution for 0 or negative time

    auto it = std::find_if(processes.begin(), processes.end(), [pid](const Process& p) {
        return p.pid == pid;
    });

    if (it == processes.end()) {
        return; // Process not found
    }

    if (it->remaining_time <= 0) {
        return; // Process already completed
    }

    if (it->arrival_time > current_time) {
        return; // Process has not arrived yet
    }

    // Record start time on first execution
    if (it->start_time == -1) {
        it->start_time = current_time;
    }

    // Record execution history
    execution_history.push_back(pid);

    // Ensure we don't execute longer than the remaining time
    int actual_time_run = std::min(time_slice, it->remaining_time);

    it->remaining_time -= actual_time_run;
    current_time += actual_time_run;

    // Check if the process completed in this slice
    if (it->remaining_time == 0) {
        it->completion_time = current_time;
    }
}

bool Simulator::isFinished() const {
    if (processes.empty()) return true; // Empty process list is considered finished
    
    for (const auto& p : processes) {
        if (p.remaining_time > 0) {
            return false;
        }
    }
    return true;
}

void Simulator::reset() {
    current_time = 0;
    processes.clear();
    execution_history.clear();
}

const std::vector<Process>& Simulator::getProcesses() const {
    return processes;
}

const std::vector<int>& Simulator::getExecutionHistory() const {
    return execution_history;
}
