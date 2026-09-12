#pragma once

class Process {
public:
    int pid;
    int arrival_time;
    int burst_time;
    int priority;
    int remaining_time;
    int start_time;
    int completion_time;
    int waiting_time;
    int turnaround_time;
    int response_time;

    Process(int pid, int arrival_time, int burst_time, int priority);
};
