#include "process.hpp"
using namespace std;

Process::Process(int pid, int arrival_time, int burst_time, int priority)
    : pid(pid),
      arrival_time(arrival_time),
      burst_time(burst_time),
      priority(priority),
      remaining_time(burst_time),
      start_time(-1),
      completion_time(-1),
      waiting_time(0),
      turnaround_time(0),
      response_time(0) {
}
