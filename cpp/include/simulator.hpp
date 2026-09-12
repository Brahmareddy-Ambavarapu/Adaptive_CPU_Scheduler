#pragma once
#include "process.hpp"
#include <vector>

class Simulator {
private:
    int current_time;
    std::vector<Process> processes;
    std::vector<int> execution_history;

public:
    Simulator();

    void addProcess(const Process& p);
    
    // Returns the current simulated time
    int getCurrentTime() const;

    // Advances the simulator time by the specified amount (for CPU idle periods)
    void advanceTime(int time_slice);

    // Executes a process with the given PID for a specified time_slice
    // Updates remaining_time, start_time, completion_time, and advances current_time.
    void executeProcess(int pid, int time_slice);

    // Checks if all processes added to the simulator are completed (remaining_time == 0)
    bool isFinished() const;

    // Resets the simulator state
    void reset();

    // Returns a reference to the list of processes for inspection
    const std::vector<Process>& getProcesses() const;

    // Returns the execution history
    const std::vector<int>& getExecutionHistory() const;
};
